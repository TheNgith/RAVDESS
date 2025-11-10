from src.configs import *
from pathlib import Path
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def parse_ravdess_filename(fname: str):
    # Example: 03-01-06-01-02-01-12.wav
    stem = Path(fname).stem
    parts = stem.split('-')
    assert len(parts) == 7, f"Unexpected filename format: {fname}"
    modality = int(parts[0])
    vocal_channel = int(parts[1])
    emotion = int(parts[2])
    intensity = int(parts[3])
    statement = int(parts[4])
    repetition = int(parts[5])
    actor = int(parts[6])
    gender = "female" if actor % 2 == 0 else "male"
    return dict(
        modality=modality,
        vocal_channel=vocal_channel,
        emotion_id=emotion,
        emotion=EMOTION_MAP[emotion],
        intensity=intensity,
        statement=statement,
        repetition=repetition,
        actor=actor,
        gender=gender
    )

def collect_metadata(root: str):
    entries = []
    root = Path(root)
    actor_dirs = sorted([p for p in root.glob("Actor_*") if p.is_dir()])
    if not actor_dirs:
        raise RuntimeError(f"No Actor_* folders found in {root.resolve()}")
    for adir in actor_dirs:
        for wav in sorted(adir.glob("*.wav")):
            meta = parse_ravdess_filename(wav.name)
            meta["path"] = str(wav)
            entries.append(meta)
    df = pd.DataFrame(entries)
    return df


def mask_from_actors(df, actor_set):
    return df["actor"].isin(actor_set)


def load_mono(path, sr=TARGET_SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=30)
    # normalize
    maxv = np.max(np.abs(y)) if y.size else 1.0
    if maxv > 0:
        y = y / maxv
    return y

def pad_or_crop(y, sr=TARGET_SR, duration=DURATION_S):
    target_len = int(sr * duration)
    if len(y) < target_len:
        pad = target_len - len(y)
        # Pad both sides to keep words centered-ish
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode='constant')
    elif len(y) > target_len:
        # center crop
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    return y

def wav_to_logmel(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=FFT, hop_length=HOP):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)
    return S_db

def spec_augment(mel, freq_mask_param=12, time_mask_param=12, p=0.5):
    if np.random.rand() > p:
        return mel
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    # frequency mask
    f = np.random.randint(0, freq_mask_param + 1)
    f0 = np.random.randint(0, max(1, n_mels - f))
    mel[f0:f0+f, :] = mel.min()
    # time mask
    t = np.random.randint(0, time_mask_param + 1)
    t0 = np.random.randint(0, max(1, n_frames - t))
    mel[:, t0:t0+t] = mel.min()
    return mel

def plot_train_curve(train_loss_epochs, val_loss_epochs, plot_type):
    epochs = range(1, len(train_loss_epochs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_epochs, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss_epochs, label='Validation Loss', color='orange', linewidth=2)
    plt.title(f'Model Training and Validation {plot_type} Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(epochs[::max(1, len(epochs) // 10)])
    plt.tight_layout()
    plt.savefig(f'./figs/training_curve_{plot_type}.png')

def draw_confusion_matrix_heatmap(test_labels, test_preds, label_encoder):
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=list(label_encoder.classes_), yticklabels=list(label_encoder.classes_),
        ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(f'./figs/confusion_matrix.png')