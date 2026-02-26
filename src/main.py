import librosa
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def extract_chroma(filepath):
    # Re-add the load function
    y, sr = librosa.load(filepath, duration=30)

    # Separate harmonics and percussives
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Extract chroma from the harmonic component only
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    return np.mean(chroma, axis=1)

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

def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache.pkl"):
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

            # after clean_label:
            if label not in VALID_KEYS:
                continue

        features = extract_chroma(audio_path)
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

# train baseline model
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42, learning_rate_init=0.001)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")