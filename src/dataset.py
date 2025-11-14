from torch.utils.data import Dataset, DataLoader
from src.configs import *
from src.utils import *

class RavdessMelDataset(Dataset):
    def __init__(self, df, train_mode=False):
        self.df = df.reset_index(drop=True)
        self.train_mode = train_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]

        y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        maxv = np.max(np.abs(y)) if y.size else 1.0
        if maxv > 0:
            y = y / maxv

        # waveform augmentations (train only)
        if self.train_mode:
            y = augment_waveform(y, sr=TARGET_SR)

        # pad / crop with random crop in train mode
        y = pad_or_crop(y, sr=TARGET_SR, duration=DURATION_S, train_mode=self.train_mode)

        # mel spectrogram ---
        mel = wav_to_logmel(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=FFT, hop_length=HOP)

        # SpecAugment on spectrogram (train only)
        if self.train_mode:
            mel = spec_augment(mel, p=0.7)

        # per-sample standardization
        mean = mel.mean()
        std = mel.std() + 1e-6
        mel = (mel - mean) / std

        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = int(row["y"])
        return x, label