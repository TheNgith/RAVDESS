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
        y = load_mono(path, sr=TARGET_SR)
        y = pad_or_crop(y, sr=TARGET_SR, duration=DURATION_S)
        mel = wav_to_logmel(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=FFT, hop_length=HOP)
        if self.train_mode:
            mel = spec_augment(mel, p=0.5)
        # standardize per-sample
        mean = mel.mean()
        std = mel.std() + 1e-6
        mel = (mel - mean) / std
        # to torch tensor [1, n_mels, T]
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = int(row["y"])
        return x, label