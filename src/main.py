import librosa
import numpy as np
import os
import pandas as pd
import pickle
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_CLASSES = ['c', 'c#', 'd', 'd#', 'e', 'f',
                 'f#', 'g', 'g#', 'a', 'a#', 'b']


def ks_key_correlations(chroma_vector):
    """Correlate chroma with all 24 Krumhansl-Schmuckler key profiles."""
    correlations = np.zeros(24)
    for i in range(12):
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)
        correlations[i] = np.corrcoef(chroma_vector, major_rotated)[0, 1]
        correlations[12 + i] = np.corrcoef(chroma_vector, minor_rotated)[0, 1]
    return correlations


def extract_features(filepath):
    y, sr = librosa.load(filepath, duration=30)
    y_harmonic, _ = librosa.effects.hpss(y)

    # --- Multiple chroma representations for robustness ---
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    # Percentile-based chroma (captures distribution, not just mean)
    chroma_cqt_mean = np.mean(chroma_cqt, axis=1)            # 12
    chroma_cqt_median = np.median(chroma_cqt, axis=1)        # 12
    chroma_cqt_std = np.std(chroma_cqt, axis=1)              # 12
    chroma_cqt_p90 = np.percentile(chroma_cqt, 90, axis=1)   # 12

    chroma_cens_mean = np.mean(chroma_cens, axis=1)           # 12
    chroma_cens_median = np.median(chroma_cens, axis=1)       # 12

    # Energy-weighted chroma
    energy = np.sum(chroma_cqt, axis=0, keepdims=True)
    energy = np.maximum(energy, 1e-10)
    chroma_weighted = np.sum(chroma_cqt * (energy / np.sum(energy)), axis=1)  # 12

    # --- KS correlations from multiple chroma sources ---
    ks_cqt = ks_key_correlations(chroma_cqt_mean)          # 24
    ks_cens = ks_key_correlations(chroma_cens_mean)         # 24
    ks_weighted = ks_key_correlations(chroma_weighted)       # 24

    # KS clarity features
    for ks in [ks_cqt, ks_cens, ks_weighted]:
        ks_sorted = np.sort(ks)[::-1]
    # Use the best (weighted) for clarity
    ks_best_sorted = np.sort(ks_weighted)[::-1]
    ks_clarity = np.array([
        ks_best_sorted[0] - ks_best_sorted[1],
        ks_best_sorted[0] - ks_best_sorted[2],
        ks_best_sorted[0],
        float(np.argmax(ks_weighted)) / 23.0,
    ])  # 4

    # --- Segment-level chroma (3 segments) ---
    n_frames = chroma_cqt.shape[1]
    seg_len = max(n_frames // 3, 1)
    seg_chromas = []
    for s in range(3):
        start = s * seg_len
        end = start + seg_len if s < 2 else n_frames
        seg_chromas.append(np.mean(chroma_cqt[:, start:end], axis=1))
    chroma_seg_var = np.var(np.array(seg_chromas), axis=0)  # 12

    # --- Tonnetz ---
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)   # 6
    tonnetz_std = np.std(tonnetz, axis=1)     # 6

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)   # 13
    mfcc_std = np.std(mfcc, axis=1)     # 13

    # --- Spectral ---
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)  # 7

    return np.concatenate([
        chroma_cqt_mean, chroma_cqt_median, chroma_cqt_std, chroma_cqt_p90,  # 48
        chroma_cens_mean, chroma_cens_median,    # 24
        chroma_weighted,                          # 12
        chroma_seg_var,                           # 12
        ks_cqt, ks_cens, ks_weighted,            # 72
        ks_clarity,                               # 4
        tonnetz_mean, tonnetz_std,                # 12
        mfcc_mean, mfcc_std,                      # 26
        spec_contrast,                            # 7
    ])
    # Total: 217


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
    """Extract features for a single track. Returns (features, label) or None."""
    try:
        features = extract_features(audio_path)
        return {'features': features, 'label': label}
    except Exception as e:
        print(f"  Skipping {os.path.basename(audio_path)}: {e}")
        return None


def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_v4.pkl"):
    if os.path.exists(cache_path):
        print("Loading from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Collect valid (audio_path, label) pairs
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
# Two-stage classifier: root note (12 classes) + mode (major/minor)
# ---------------------------------------------------------------------------
class TwoStageKeyClassifier:
    """Splits the 24-key problem into two easier problems:
    Stage 1: Predict root note (12 classes, ~2x samples per class)
    Stage 2: Predict mode - major or minor (2 classes, ~12x samples per class)
    Final prediction = root + mode
    """

    def __init__(self, root_clf, mode_clf):
        self.root_clf = root_clf
        self.mode_clf = mode_clf
        self.root_le = LabelEncoder()
        self.mode_le = LabelEncoder()

    def _split_labels(self, labels):
        """Split 'c# minor' -> root='c#', mode='minor'."""
        roots = [l.rsplit(' ', 1)[0] for l in labels]
        modes = [l.rsplit(' ', 1)[1] for l in labels]
        return roots, modes

    def fit(self, X, labels):
        roots, modes = self._split_labels(labels)
        y_root = self.root_le.fit_transform(roots)
        y_mode = self.mode_le.fit_transform(modes)
        self.root_clf.fit(X, y_root)
        self.mode_clf.fit(X, y_mode)
        return self

    def predict(self, X):
        root_pred = self.root_le.inverse_transform(self.root_clf.predict(X))
        mode_pred = self.mode_le.inverse_transform(self.mode_clf.predict(X))
        return np.array([f"{r} {m}" for r, m in zip(root_pred, mode_pred)])


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

X_train, X_test, y_train, y_test, lab_train, lab_test = train_test_split(
    X, y, labels, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Approach 1: Flat 24-class voting ensemble ---
print("\n=== Flat 24-class Voting Ensemble ===")
flat_ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=1000, min_samples_split=2, class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), max_iter=5000,
            learning_rate_init=0.0005, learning_rate='adaptive',
            early_stopping=True, validation_fraction=0.15,
            alpha=0.001, random_state=42)),
    ],
    voting='hard',
    n_jobs=-1,
)
flat_ensemble.fit(X_train_s, y_train)
y_pred_flat = flat_ensemble.predict(X_test_s)
flat_acc = accuracy_score(y_test, y_pred_flat)
print(f"Flat Ensemble Accuracy: {flat_acc:.3f}")

# --- Approach 2: Two-stage (root + mode) ---
print("\n=== Two-Stage (Root Note + Mode) ===")

# Strong ensemble for root note (12 classes)
root_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=1000, class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)),
    ],
    voting='soft',
    n_jobs=-1,
)

# Strong ensemble for mode (2 classes)
mode_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=500, class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=42)),
    ],
    voting='soft',
    n_jobs=-1,
)

two_stage = TwoStageKeyClassifier(root_clf, mode_clf)
two_stage.fit(X_train_s, lab_train)
pred_labels = two_stage.predict(X_test_s)

# Map predictions back to encoded labels for accuracy
pred_encoded = le.transform(pred_labels)
two_stage_acc = accuracy_score(y_test, pred_encoded)
print(f"Two-Stage Accuracy: {two_stage_acc:.3f}")

# --- Pick best approach and print full report ---
if two_stage_acc >= flat_acc:
    print(f"\nBest: Two-Stage with accuracy {two_stage_acc:.3f}")
    best_pred = pred_encoded
else:
    print(f"\nBest: Flat Ensemble with accuracy {flat_acc:.3f}")
    best_pred = y_pred_flat

print("\nClassification Report:")
print(classification_report(
    y_test, best_pred, target_names=le.classes_, zero_division=0
))
