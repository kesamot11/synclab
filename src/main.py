import librosa
import numpy as np
import os
import pandas as pd
import pickle
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles (the standard for key detection)
# ---------------------------------------------------------------------------
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Temperley profiles (alternative, often better for pop/electronic)
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
                             2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0,
                             2.0, 4.5, 3.5, 2.0, 1.5, 4.0])


def key_correlations(chroma_vector, major_prof, minor_prof):
    """Correlate chroma with all 24 key profiles (12 major + 12 minor)."""
    correlations = np.zeros(24)
    for i in range(12):
        correlations[i] = np.corrcoef(chroma_vector, np.roll(major_prof, i))[0, 1]
        correlations[12 + i] = np.corrcoef(chroma_vector, np.roll(minor_prof, i))[0, 1]
    return correlations


def extract_features(filepath):
    y, sr = librosa.load(filepath, duration=30)
    y_harmonic, _ = librosa.effects.hpss(y)

    # --- Two chroma representations ---
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    chroma_cqt_mean = np.mean(chroma_cqt, axis=1)       # 12
    chroma_cens_mean = np.mean(chroma_cens, axis=1)      # 12
    chroma_std = np.std(chroma_cqt, axis=1)              # 12

    # Energy-weighted chroma (louder frames count more)
    energy = np.sum(chroma_cqt, axis=0, keepdims=True)
    energy = np.maximum(energy, 1e-10)
    chroma_weighted = np.sum(chroma_cqt * (energy / np.sum(energy)), axis=1)  # 12

    # --- Key profile correlations (both Krumhansl + Temperley) ---
    ks_corr = key_correlations(chroma_weighted, MAJOR_PROFILE, MINOR_PROFILE)    # 24
    temp_corr = key_correlations(chroma_weighted, TEMPERLEY_MAJOR, TEMPERLEY_MINOR)  # 24

    # Clarity: how confident is the key detection?
    ks_sorted = np.sort(ks_corr)[::-1]
    temp_sorted = np.sort(temp_corr)[::-1]
    clarity = np.array([
        ks_sorted[0] - ks_sorted[1],    # KS margin
        ks_sorted[0] - ks_sorted[2],
        temp_sorted[0] - temp_sorted[1], # Temperley margin
        temp_sorted[0] - temp_sorted[2],
    ])  # 4

    # --- Tonnetz (tonal centroid, captures harmonic relationships) ---
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)   # 6

    # Only key-relevant features — NO MFCCs, NO spectral features (they capture
    # timbre/texture, not pitch, and add noise for this task)
    return np.concatenate([
        chroma_cqt_mean,     # 12
        chroma_cens_mean,    # 12
        chroma_std,          # 12
        chroma_weighted,     # 12
        ks_corr,             # 24
        temp_corr,           # 24
        clarity,             # 4
        tonnetz_mean,        # 6
    ])
    # Total: 106 (focused, key-relevant features only)


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


def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_v5.pkl"):
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
print(f"Raw feature dimensions: {X.shape[1]}")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale then PCA — remove redundancy among correlated chroma/KS features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=0.98, random_state=42)  # keep 98% variance
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)
print(f"PCA dimensions: {X_train_pca.shape[1]} (from {X.shape[1]})")

# --- Train models with cross-validation to find the best ---
print("\n=== Model Comparison (5-fold stratified CV) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF (raw)": (RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        random_state=42, n_jobs=-1), X_train_s),
    "RF (PCA)": (RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        random_state=42, n_jobs=-1), X_train_pca),
    "GB (raw)": (GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42), X_train_s),
    "GB (PCA)": (GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42), X_train_pca),
}

best_cv = 0
best_name = None
best_data_key = None

for name, (clf, X_tr) in models.items():
    scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_score = scores.mean()
    print(f"{name}: CV={mean_score:.3f} (+/- {scores.std():.3f})")
    if mean_score > best_cv:
        best_cv = mean_score
        best_name = name
        best_data_key = name

print(f"\nBest CV model: {best_name} (CV={best_cv:.3f})")

# Retrain best model on full training set and evaluate on test
best_clf, best_X_train = models[best_name]
best_clf.fit(best_X_train, y_train)

# Select matching test data
best_X_test = X_test_pca if "PCA" in best_name else X_test_s
y_pred = best_clf.predict(best_X_test)

test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Show feature importances for RF
if "RF" in best_name and hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
    top_k = min(15, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_k]
    print(f"\nTop {top_k} feature importances:")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1}. Feature {idx}: {importances[idx]:.4f}")
