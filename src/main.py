import librosa
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles — music-theory correlation templates
# These represent the expected pitch-class distribution for each key/mode.
# ---------------------------------------------------------------------------
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Pitch class names in chroma order (C, C#, D, ..., B)
PITCH_CLASSES = ['c', 'c#', 'd', 'd#', 'e', 'f',
                 'f#', 'g', 'g#', 'a', 'a#', 'b']

KEY_NAMES = []
for pc in PITCH_CLASSES:
    KEY_NAMES.append(f"{pc} major")
for pc in PITCH_CLASSES:
    KEY_NAMES.append(f"{pc} minor")


def ks_key_correlations(chroma_vector):
    """Compute Krumhansl-Schmuckler correlation for all 24 keys.

    Returns a 24-dim vector: correlation with each of the 12 major
    and 12 minor key profiles (rotated to match each root note).
    """
    correlations = np.zeros(24)
    for i in range(12):
        # Rotate profile to match root note i
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)
        correlations[i] = np.corrcoef(chroma_vector, major_rotated)[0, 1]
        correlations[12 + i] = np.corrcoef(chroma_vector, minor_rotated)[0, 1]
    return correlations


def ks_predict_key(chroma_vector):
    """Return the key name predicted by the KS algorithm (for reference)."""
    corrs = ks_key_correlations(chroma_vector)
    return KEY_NAMES[np.argmax(corrs)]


def extract_features(filepath):
    y, sr = librosa.load(filepath, duration=30)

    # Separate harmonics
    y_harmonic, _ = librosa.effects.hpss(y)

    # --- Chroma features from harmonic component ---
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)   # 12-dim
    chroma_std = np.std(chroma, axis=1)     # 12-dim

    # Weighted chroma: weight frames by energy so louder parts count more
    energy = np.sum(chroma, axis=0, keepdims=True)
    energy = np.maximum(energy, 1e-10)  # avoid division by zero
    chroma_weighted = np.sum(chroma * (energy / np.sum(energy)), axis=1)  # 12-dim

    # --- Krumhansl-Schmuckler correlations (the key ingredient) ---
    ks_corr_mean = ks_key_correlations(chroma_mean)          # 24-dim
    ks_corr_weighted = ks_key_correlations(chroma_weighted)   # 24-dim

    # Top-K KS features: difference between best and 2nd-best correlation,
    # and the index of the best key (as a normalized feature)
    ks_sorted = np.sort(ks_corr_weighted)[::-1]
    ks_clarity = np.array([
        ks_sorted[0] - ks_sorted[1],   # confidence margin
        ks_sorted[0] - ks_sorted[2],   # margin to 3rd
        ks_sorted[0],                   # best correlation value
        float(np.argmax(ks_corr_weighted)) / 23.0,  # best key index normalized
    ])

    # --- Segment-level chroma stats (split audio into 3 segments) ---
    n_frames = chroma.shape[1]
    seg_len = n_frames // 3
    seg_chromas = []
    for s in range(3):
        start = s * seg_len
        end = start + seg_len if s < 2 else n_frames
        seg_chromas.append(np.mean(chroma[:, start:end], axis=1))
    # Variance of chroma across segments captures key stability
    chroma_seg_var = np.var(np.array(seg_chromas), axis=0)  # 12-dim

    # --- Tonnetz (tonal centroid features) ---
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)   # 6-dim
    tonnetz_std = np.std(tonnetz, axis=1)     # 6-dim

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)   # 13-dim
    mfcc_std = np.std(mfcc, axis=1)     # 13-dim

    # --- Spectral features ---
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)  # 7-dim

    # Combine all features
    return np.concatenate([
        chroma_mean,         # 12
        chroma_std,          # 12
        chroma_weighted,     # 12
        chroma_seg_var,      # 12
        ks_corr_mean,        # 24
        ks_corr_weighted,    # 24
        ks_clarity,          # 4
        tonnetz_mean,        # 6
        tonnetz_std,         # 6
        mfcc_mean,           # 13
        mfcc_std,            # 13
        spec_contrast,       # 7
    ])
    # Total: 145 features


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


def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_v3.pkl"):
    if os.path.exists(cache_path):
        print("Loading from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    data = []
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

        features = extract_features(audio_path)
        data.append({'features': features, 'label': label})

    df = pd.DataFrame(data)
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    print("Saved to cache.")
    return df


print(len([f for f in os.listdir("../data/audio") if f.endswith(".mp3")]))
df = load_dataset("../data/audio", "../data/annotations/key")
print(f"Dataset: {df.shape[0]} samples")
print(f"Keys: {sorted(df['label'].unique())}")

# encode labels
le = LabelEncoder()
X = np.stack(df['features'].values)
y = le.fit_transform(df['label'].values)
print(f"Feature dimensions: {X.shape[1]}")

# train/test split with stratification to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights to handle imbalance
from collections import Counter
class_counts = Counter(y_train)
n_samples = len(y_train)
n_classes = len(class_counts)
class_weight_dict = {c: n_samples / (n_classes * count) for c, count in class_counts.items()}

# --- Train models ---
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=5000,
        random_state=42,
        learning_rate_init=0.0005,
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.15,
        alpha=0.001,
    ),
}

best_acc = 0
best_name = None

for name, clf in models.items():
    if name == "MLP":
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        best_name = name

print(f"\nBest model: {best_name} with accuracy {best_acc:.3f}")

# Detailed report for best model
best_model = models[best_name]
if best_name == "MLP":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))
