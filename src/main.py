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
# Chroma feature extraction — pitch-class features for key detection
# ---------------------------------------------------------------------------
N_CHROMA = 12
N_FRAMES = 640  # ~15 seconds per chunk at hop=512, sr=22050

def extract_chroma(filepath):
    """Load audio (middle section), apply HPSS, compute chroma CQT."""
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

    # Separate harmonic content from percussion
    y_harmonic, _ = librosa.effects.hpss(y)

    # Chroma CQT — maps directly to 12 pitch classes
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512)

    # Normalize each frame to sum to 1 (pitch class distribution)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)

    return chroma.astype(np.float32)


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
        chroma = extract_chroma(audio_path)
        return {'chroma': chroma, 'label': label}
    except Exception as e:
        print(f"  Skipping {os.path.basename(audio_path)}: {e}")
        return None


def load_dataset(audio_dir, annotation_dir, cache_path="../data/cache_chroma.pkl"):
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

    print(f"Extracting chroma features from {len(tasks)} tracks (parallel)...")
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
# Dataset: slice each chroma into overlapping chunks for data augmentation
# ---------------------------------------------------------------------------
class KeyDataset(Dataset):
    def __init__(self, chromas, labels, n_frames=N_FRAMES, augment=False):
        self.samples = []
        self.labels = []
        self.n_frames = n_frames

        for chroma, label in zip(chromas, labels):
            n_total = chroma.shape[1]
            if n_total < n_frames:
                padded = np.zeros((N_CHROMA, n_frames), dtype=np.float32)
                padded[:, :n_total] = chroma
                self.samples.append(padded)
                self.labels.append(label)
            else:
                # 50% overlap when augmenting, no overlap otherwise
                stride = n_frames // 2 if augment else n_frames
                for start in range(0, n_total - n_frames + 1, stride):
                    chunk = chroma[:, start:start + n_frames]
                    self.samples.append(chunk)
                    self.labels.append(label)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Shape: (N_CHROMA, N_FRAMES) — 12 chroma bins as channels for 1D conv
        x = torch.FloatTensor(self.samples[idx])
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y


# ---------------------------------------------------------------------------
# 1D CNN on chroma features — chroma bins are input channels
# ---------------------------------------------------------------------------
class KeyCNN(nn.Module):
    def __init__(self, n_classes=24, n_chroma=N_CHROMA):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: learn local pitch-time patterns
            nn.Conv1d(n_chroma, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            # Block 2: learn longer-range harmonic patterns
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256),
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
    n_batches = len(loader)
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  batch {i+1}/{n_batches} — loss={loss.item():.4f}", flush=True)
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


def evaluate_by_track(model, chromas, labels, le, device):
    """Evaluate per-track by averaging predictions across all chunks of a track."""
    model.eval()
    preds = []

    with torch.no_grad():
        for chroma, label in zip(chromas, labels):
            n_total = chroma.shape[1]
            chunk_logits = []

            if n_total < N_FRAMES:
                padded = np.zeros((N_CHROMA, N_FRAMES), dtype=np.float32)
                padded[:, :n_total] = chroma
                x = torch.FloatTensor(padded).unsqueeze(0).to(device)
                chunk_logits.append(model(x).cpu().numpy()[0])
            else:
                for start in range(0, n_total - N_FRAMES + 1, N_FRAMES // 2):
                    chunk = chroma[:, start:start + N_FRAMES]
                    x = torch.FloatTensor(chunk).unsqueeze(0).to(device)
                    chunk_logits.append(model(x).cpu().numpy()[0])

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
chromas = df['chroma'].values
labels = y_all

indices = np.arange(len(chromas))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)

train_chromas = [chromas[i] for i in train_idx]
train_labels = labels[train_idx]
test_chromas = [chromas[i] for i in test_idx]
test_labels = labels[test_idx]

# Further split train into train/val
train_idx2, val_idx2 = train_test_split(
    np.arange(len(train_chromas)), test_size=0.15, random_state=42,
    stratify=train_labels
)

val_chromas = [train_chromas[i] for i in val_idx2]
val_labels = train_labels[val_idx2]
final_train_chromas = [train_chromas[i] for i in train_idx2]
final_train_labels = train_labels[train_idx2]

# Create datasets with chunk augmentation for training
train_ds = KeyDataset(final_train_chromas, final_train_labels, augment=True)
val_ds = KeyDataset(val_chromas, val_labels, augment=False)

print(f"Training chunks: {len(train_ds)} (from {len(final_train_chromas)} tracks)")
print(f"Validation chunks: {len(val_ds)} (from {len(val_chromas)} tracks)")
print(f"Test tracks: {len(test_chromas)}")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
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
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop with early stopping
best_val_acc = 0
patience = 15
patience_counter = 0
n_epochs = 100

print(f"\nTraining for up to {n_epochs} epochs...", flush=True)
for epoch in range(n_epochs):
    print(f"\n--- Starting epoch {epoch+1}/{n_epochs} ---", flush=True)
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_acc, _, _ = evaluate(model, val_loader, device)
    scheduler.step(1 - val_acc)

    print(f"Epoch {epoch+1:3d}/{n_epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

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
test_preds = evaluate_by_track(model, test_chromas, test_labels, le, device)
test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.3f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=le.classes_, zero_division=0))
