import librosa
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# Mel spectrogram extraction — let the CNN learn its own features
# ---------------------------------------------------------------------------
N_MELS = 128
N_FRAMES = 130  # ~3 seconds per chunk at hop=512, sr=22050

def extract_mel_spectrogram(filepath):
    """Load audio (middle section), compute mel spectrogram."""
    y_full, sr = librosa.load(filepath, sr=22050)
    total_duration = len(y_full) / sr

    # Skip intro — use the harmonically rich middle section
    if total_duration > 90:
        offset = int(30 * sr)
        duration = int(60 * sr)
        y = y_full[offset:offset + duration]
    elif total_duration > 45:
        offset = int(15 * sr)
        y = y_full[offset:]
    else:
        y = y_full

    # Compute mel spectrogram (log-scaled)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)

    return mel_db.astype(np.float32)


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
        mel = extract_mel_spectrogram(audio_path)
        return {'mel': mel, 'label': label}
    except Exception as e:
        print(f"  Skipping {os.path.basename(audio_path)}: {e}")
        return None


def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_cnn.pkl"):
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

    print(f"Extracting mel spectrograms from {len(tasks)} tracks (parallel)...")
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
# Dataset: slice each spectrogram into overlapping chunks for data augmentation
# ---------------------------------------------------------------------------
class KeyDataset(Dataset):
    def __init__(self, mels, labels, n_frames=N_FRAMES, augment=False):
        self.samples = []
        self.labels = []
        self.augment = augment
        self.n_frames = n_frames

        for mel, label in zip(mels, labels):
            n_total = mel.shape[1]
            if n_total < n_frames:
                # Pad short spectrograms
                padded = np.zeros((N_MELS, n_frames), dtype=np.float32)
                padded[:, :n_total] = mel
                self.samples.append(padded)
                self.labels.append(label)
            else:
                # Extract multiple overlapping chunks for more training data
                stride = n_frames // 2 if augment else n_frames
                for start in range(0, n_total - n_frames + 1, stride):
                    chunk = mel[:, start:start + n_frames]
                    self.samples.append(chunk)
                    self.labels.append(label)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Shape: (1, N_MELS, N_FRAMES) — 1 channel like grayscale image
        x = torch.FloatTensor(self.samples[idx]).unsqueeze(0)
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------
class KeyCNN(nn.Module):
    def __init__(self, n_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def evaluate_by_track(model, mels, labels, le, device):
    """Evaluate per-track by averaging predictions across all chunks of a track."""
    model.eval()
    n_classes = len(le.classes_)
    preds = []

    with torch.no_grad():
        for mel, label in zip(mels, labels):
            n_total = mel.shape[1]
            chunk_logits = []

            if n_total < N_FRAMES:
                padded = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
                padded[:, :n_total] = mel
                x = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0).to(device)
                chunk_logits.append(model(x).cpu().numpy()[0])
            else:
                for start in range(0, n_total - N_FRAMES + 1, N_FRAMES // 2):
                    chunk = mel[:, start:start + N_FRAMES]
                    x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    chunk_logits.append(model(x).cpu().numpy()[0])

            # Average logits across chunks, then take argmax
            avg_logits = np.mean(chunk_logits, axis=0)
            preds.append(np.argmax(avg_logits))

    return np.array(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(len([f for f in os.listdir("../data/audio") if f.endswith(".mp3")]))
df = load_dataset("../data/audio", "../data/annotations/key")
print(f"Dataset: {df.shape[0]} tracks")

le = LabelEncoder()
le.fit(df['label'].values)
y_all = le.transform(df['label'].values)
n_classes = len(le.classes_)
print(f"Classes: {n_classes}")

# Stratified split
mels = df['mel'].values
labels = y_all

indices = np.arange(len(mels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)

train_mels = [mels[i] for i in train_idx]
train_labels = labels[train_idx]
test_mels = [mels[i] for i in test_idx]
test_labels = labels[test_idx]

# Further split train into train/val
train_idx2, val_idx2 = train_test_split(
    np.arange(len(train_mels)), test_size=0.15, random_state=42,
    stratify=train_labels
)

val_mels = [train_mels[i] for i in val_idx2]
val_labels = train_labels[val_idx2]
final_train_mels = [train_mels[i] for i in train_idx2]
final_train_labels = train_labels[train_idx2]

# Create datasets with chunk augmentation for training
train_ds = KeyDataset(final_train_mels, final_train_labels, augment=True)
val_ds = KeyDataset(val_mels, val_labels, augment=False)

print(f"Training chunks: {len(train_ds)} (from {len(final_train_mels)} tracks)")
print(f"Validation chunks: {len(val_ds)} (from {len(val_mels)} tracks)")
print(f"Test tracks: {len(test_mels)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

# --- Train CNN ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = KeyCNN(n_classes=n_classes).to(device)

# Class weights for imbalanced dataset
class_counts = np.bincount(final_train_labels, minlength=n_classes).astype(float)
class_weights = len(final_train_labels) / (n_classes * class_counts + 1e-10)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop with early stopping
best_val_acc = 0
patience = 15
patience_counter = 0
n_epochs = 100

print(f"\nTraining for up to {n_epochs} epochs...")
for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_acc, _, _ = evaluate(model, val_loader, device)
    scheduler.step(1 - val_acc)

    if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), '../data/best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model and evaluate on test set (per-track averaging)
print(f"\nBest validation accuracy: {best_val_acc:.3f}")
model.load_state_dict(torch.load('../data/best_model.pth', weights_only=True))

# Per-track evaluation (average predictions across chunks)
print("\n=== Test Set Evaluation (per-track averaging) ===")
test_preds = evaluate_by_track(model, test_mels, test_labels, le, device)
test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.3f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=le.classes_, zero_division=0))
