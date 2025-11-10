from src.utils import *
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.dataset import RavdessMelDataset
from torch.utils.data import Dataset, DataLoader
from src.models import EmotionCNN
from torch import nn
import os, random, math, time, itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

df = collect_metadata(DATASET_ROOT)
print("Total files:", len(df))
print(df.head())

actors = sorted(df["actor"].unique())
rng = np.random.default_rng(SEED)
rng.shuffle(actors)

n = len(actors)
n_train = int(0.67 * n)       # ~ 16
n_val = int(0.165 * n)        # ~ 4
train_actors = set(actors[:n_train])
val_actors   = set(actors[n_train:n_train+n_val])
test_actors  = set(actors[n_train+n_val:])

df_train = df[mask_from_actors(df, train_actors)].reset_index(drop=True)
df_val   = df[mask_from_actors(df, val_actors)].reset_index(drop=True)
df_test  = df[mask_from_actors(df, test_actors)].reset_index(drop=True)

print(f"Actors -> train:{sorted(train_actors)}, val:{sorted(val_actors)}, test:{sorted(test_actors)}")
print("Splits:", len(df_train), len(df_val), len(df_test))

le = LabelEncoder()
le.fit(df["emotion"])
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

def label_encode(series):
    return le.transform(series.values)

df_train["y"] = label_encode(df_train["emotion"])
df_val["y"]   = label_encode(df_val["emotion"])
df_test["y"]  = label_encode(df_test["emotion"])

train_ds = RavdessMelDataset(df_train, train_mode=True)
val_ds   = RavdessMelDataset(df_val,   train_mode=False)
test_ds  = RavdessMelDataset(df_test,  train_mode=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

model = EmotionCNN(num_classes).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    torch.set_grad_enabled(train)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, acc, all_preds, all_labels

train_loss_epochs = []
val_loss_epochs = []
train_acc_epochs = []
val_acc_epochs = []
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    train_loss, train_acc, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_acc, _, _ = run_epoch(val_loader, train=False)
    train_loss_epochs.append(train_loss)
    train_acc_epochs.append(train_acc)
    val_loss_epochs.append(val_loss)
    val_acc_epochs.append(val_acc)
    scheduler.step()
    dt = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS} | {dt:.1f}s | "
          f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.3f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            if EARLY_STOPPING:
                print("Early stopping.")
                break
    
plot_train_curve(train_loss_epochs, val_loss_epochs, "loss")
plot_train_curve(train_acc_epochs, val_acc_epochs, "accuracy")

# Load best val model
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

test_loss, test_acc, test_preds, test_labels = run_epoch(test_loader, train=False)
print(f"\nTEST | loss {test_loss:.4f} acc {test_acc:.3f}")

print("\nClassification report:")
print(classification_report(test_labels, test_preds, target_names=list(le.classes_), digits=3))

draw_confusion_matrix_heatmap(test_labels, test_preds, le)