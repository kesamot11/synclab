import librosa
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


def extract_features(filepath):
    y, sr = librosa.load(filepath, duration=30)

    # Separate harmonics and percussives
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Chroma features (mean + std) from harmonic component
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Tonnetz (tonal centroid features) — captures harmonic relations
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    # MFCCs — capture timbral/spectral shape
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral features
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Combine all features into a single vector
    return np.concatenate([
        chroma_mean,       # 12
        chroma_std,        # 12
        tonnetz_mean,      # 6
        mfcc_mean,         # 13
        mfcc_std,          # 13
        spec_contrast,     # 7
        [spec_cent, spec_bw, spec_rolloff],  # 3
    ])

def clean_label(label):
    # take only the part before the first tab
    key = label.split('\t')[0].strip()
    # take first key if multiple (e.g. "c# major / f minor" -> "c# major")
    key = key.split('/')[0].strip()
    return key

VALID_KEYS = {
                'c major', 'c# major', 'd major', 'd# major', 'e major', 'f major',
                'f# major', 'g major', 'g# major', 'a major', 'a# major', 'b major',
                'c minor', 'c# minor', 'd minor', 'd# minor', 'e minor', 'f minor',
                'f# minor', 'g minor', 'g# minor', 'a minor', 'a# minor', 'b minor'
            }

def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_v2.pkl"):
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
print(df.shape)
print(df['label'].unique())
print(df.head())

# encode labels
le = LabelEncoder()
X = np.stack(df['features'].values)
y = le.fit_transform(df['label'].values)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features — critical for MLP performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train multiple models and pick the best ---
models = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=3000,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    ),
}

best_acc = 0
best_name = None

for name, clf in models.items():
    # MLP uses scaled data; tree-based models use raw data
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

# Print detailed report for best model
best_model = models[best_name]
if best_name == "MLP":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))