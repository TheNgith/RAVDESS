import torch

EARLY_STOPPING = False

DATASET_ROOT = "./data"
TARGET_SR = 16000
DURATION_S = 4.0
N_MELS = 128
FFT = 1024
HOP = 256
BATCH_SIZE = 32
EPOCHS = 40
LR = 2e-3
SEED = 42
NUM_WORKERS = 1

EMOTION_MAP = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Device:", device)

best_val_acc = 0.0
best_state = None
patience, bad = 5, 0