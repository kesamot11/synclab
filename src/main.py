import librosa
import numpy as np
import os
import pandas as pd
import pickle
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Key profiles for template matching
# ---------------------------------------------------------------------------
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
                             2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0,
                             2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

PITCH_CLASSES = ['c', 'c#', 'd', 'd#', 'e', 'f',
                 'f#', 'g', 'g#', 'a', 'a#', 'b']

KEY_NAMES = [f"{pc} major" for pc in PITCH_CLASSES] + \
            [f"{pc} minor" for pc in PITCH_CLASSES]


def key_correlations(chroma_vector, major_prof, minor_prof):
    """Correlate normalized chroma with all 24 key profiles."""
    # L1 normalize so we're comparing distributions, not magnitudes
    chroma_norm = chroma_vector / (np.sum(chroma_vector) + 1e-10)
    correlations = np.zeros(24)
    for i in range(12):
        correlations[i] = np.corrcoef(chroma_norm, np.roll(major_prof, i))[0, 1]
        correlations[12 + i] = np.corrcoef(chroma_norm, np.roll(minor_prof, i))[0, 1]
    return correlations


def ks_predict(chroma_vector):
    """Pure KS prediction (no ML) — returns key name."""
    corrs = key_correlations(chroma_vector, MAJOR_PROFILE, MINOR_PROFILE)
    return KEY_NAMES[np.argmax(corrs)]


def extract_features(filepath):
    # Skip the intro (first 30s) — in EDM the intro is often drums/FX
    # with no harmonic content. Load the middle section instead.
    y_full, sr = librosa.load(filepath, sr=22050)
    total_duration = len(y_full) / sr

    if total_duration > 90:
        # Skip first 30s, take 60s from the middle
        offset_samples = int(30 * sr)
        duration_samples = int(60 * sr)
        y = y_full[offset_samples:offset_samples + duration_samples]
    elif total_duration > 45:
        # Skip first 15s
        offset_samples = int(15 * sr)
        y = y_full[offset_samples:]
    else:
        y = y_full

    y_harmonic, _ = librosa.effects.hpss(y)

    # --- Multiple chroma representations ---
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)

    chroma_cqt_mean = np.mean(chroma_cqt, axis=1)
    chroma_cens_mean = np.mean(chroma_cens, axis=1)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    chroma_std = np.std(chroma_cqt, axis=1)

    # Energy-weighted chroma
    energy = np.sum(chroma_cqt, axis=0, keepdims=True)
    energy = np.maximum(energy, 1e-10)
    chroma_weighted = np.sum(chroma_cqt * (energy / np.sum(energy)), axis=1)

    # --- Key profile correlations (4 sets: 2 profiles x 2 chroma sources) ---
    ks_cqt = key_correlations(chroma_cqt_mean, MAJOR_PROFILE, MINOR_PROFILE)       # 24
    ks_weighted = key_correlations(chroma_weighted, MAJOR_PROFILE, MINOR_PROFILE)   # 24
    temp_cqt = key_correlations(chroma_cqt_mean, TEMPERLEY_MAJOR, TEMPERLEY_MINOR) # 24
    temp_weighted = key_correlations(chroma_weighted, TEMPERLEY_MAJOR, TEMPERLEY_MINOR)  # 24

    # Clarity features from best correlations
    ks_sorted = np.sort(ks_weighted)[::-1]
    temp_sorted = np.sort(temp_weighted)[::-1]
    clarity = np.array([
        ks_sorted[0] - ks_sorted[1],
        ks_sorted[0] - ks_sorted[2],
        ks_sorted[0] - ks_sorted[3],
        temp_sorted[0] - temp_sorted[1],
        temp_sorted[0] - temp_sorted[2],
        float(np.argmax(ks_weighted) >= 12),  # 1 if minor predicted
    ])  # 6

    # --- Tonnetz ---
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)   # 6

    # Also store raw chroma for pure KS baseline
    return np.concatenate([
        chroma_cqt_mean,     # 12
        chroma_cens_mean,    # 12
        chroma_stft_mean,    # 12
        chroma_std,          # 12
        chroma_weighted,     # 12
        ks_cqt,              # 24
        ks_weighted,         # 24
        temp_cqt,            # 24
        temp_weighted,       # 24
        clarity,             # 6
        tonnetz_mean,        # 6
    ])
    # Total: 168


def clean_label(label):
    key = label.split('\t')[0].strip()
    key = key.split('/')[0].strip()
    return key


VALID_KEYS = {
    'c major', 'c# major', 'd major', 'd# major', 'e major', 'f major',
    'f# major', 'g major', 'g# major', 'a major', 'a# major', 'b major',
    'c minor', 'c# minor', 'd minor', 'd# minor', 'e minor', 'f minor',
    'f# minor', 'g minor', 'g# minor', 'a minor', 'a# minor', 'b minor'
}


def _process_one(audio_path, label):
    try:
        features = extract_features(audio_path)
        return {'features': features, 'label': label}
    except Exception as e:
        print(f"  Skipping {os.path.basename(audio_path)}: {e}")
        return None


def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_v6.pkl"):
    if os.path.exists(cache_path):
        print("Loading from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    tasks = []
    for key_file in os.listdir(annotation_dir):
        if not key_file.endswith('.key'):
            continue
        track_id = key_file.replace('.key', '')
        audio_path = os.path.join(audio_dir, track_id + '.mp3')
        if not os.path.exists(audio_path):
            continue
        with open(os.path.join(annotation_dir, key_file)) as f:
            label = clean_label(f.read().strip())
            if label not in VALID_KEYS:
                continue
        tasks.append((audio_path, label))

    print(f"Extracting features from {len(tasks)} tracks (parallel)...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_process_one)(path, label) for path, label in tasks
    )
    data = [r for r in results if r is not None]

    df = pd.DataFrame(data)
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    print("Saved to cache.")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(len([f for f in os.listdir("../data/audio") if f.endswith(".mp3")]))
df = load_dataset("../data/audio", "../data/annotations/key")
print(f"Dataset: {df.shape[0]} samples")

le = LabelEncoder()
X = np.stack(df['features'].values)
labels = df['label'].values
y = le.fit_transform(labels)
print(f"Feature dimensions: {X.shape[1]}")

# --- Baseline: Pure KS algorithm (no ML) ---
print("\n=== Pure KS Baseline (no ML) ===")
# The chroma_cqt_mean is the first 12 features; use chroma_weighted (features 48-59)
ks_preds = []
for i in range(len(X)):
    chroma_weighted = X[i, 48:60]  # chroma_weighted slice
    ks_preds.append(ks_predict(chroma_weighted))
ks_encoded = le.transform(ks_preds)
ks_acc = accuracy_score(y, ks_encoded)
print(f"Pure KS Accuracy (full dataset): {ks_acc:.3f}")

# --- ML models ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n=== Model Comparison (5-fold stratified CV) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42),
    "SVM (RBF)": SVC(
        kernel='rbf', C=10, gamma='scale',
        class_weight='balanced', random_state=42),
}

best_cv = 0
best_name = None

for name, clf in models.items():
    scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_score = scores.mean()
    print(f"{name}: CV={mean_score:.3f} (+/- {scores.std():.3f})")
    if mean_score > best_cv:
        best_cv = mean_score
        best_name = name

print(f"\nBest CV model: {best_name} (CV={best_cv:.3f})")

# Final evaluation on test set
best_clf = models[best_name]
best_clf.fit(X_train_s, y_train)
y_pred = best_clf.predict(X_test_s)

test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
