import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ======================= FIXED SECTION (do not edit) =======================
SEED = 42
SMOKE = bool(os.environ.get("SMOKE_TEST"))   # quick crash test used by the agent
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_FILE = BASE_DIR / "training_log.csv"
TIME_BUDGET_SECONDS = 300  # 5 minutes maximum runtime


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ===========================================================================

# -------------------- MUTABLE ZONE 1: HYPERPARAMETERS --------------------
BATCH_SIZE = 32
IMAGE_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 2e-3
DROPOUT_RATE = 0.5
# -------------------------------------------------------------------------


# -------------------- MUTABLE ZONE 2: MODEL ARCHITECTURE -----------------
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
# -------------------------------------------------------------------------


def run_training() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- MUTABLE ZONE 3: PREPROCESSING & AUGMENTATION ---------
    # Training transforms only. Add augmentation between Resize and ToTensor
    # (PIL-image transforms), e.g. transforms.RandomHorizontalFlip(p=0.5).
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    # ----------------------------------------------------------------------

    # Validation transforms: fixed, never augmented.
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_path = DATA_DIR / "Training"
    test_path = DATA_DIR / "Testing"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset folders were not found at {DATA_DIR}. Expected 'Training' and 'Testing'."
        )

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    test_ds = datasets.ImageFolder(test_path, transform=eval_tf)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = BrainTumorCNN(num_classes=len(train_ds.classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if not SMOKE:
        with open(LOG_FILE, mode="w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * x.size(0)
            if SMOKE and i >= 1:
                break

        epoch_train_loss = running_train_loss / len(train_dl.dataset)

        model.eval()
        running_val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for j, (x, y) in enumerate(test_dl):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                running_val_loss += criterion(logits, y).item() * y.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                if SMOKE and j >= 1:
                    break

        if SMOKE:
            print("--- SMOKE OK ---")
            return

        epoch_val_loss = running_val_loss / len(test_dl.dataset)
        epoch_val_acc = 100.0 * correct / len(test_dl.dataset)
        best_val_loss = min(best_val_loss, epoch_val_loss)

        elapsed_seconds = time.time() - start_time
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.2f}% | "
            f"Elapsed: {elapsed_seconds:.1f}s"
        )

        with open(LOG_FILE, mode="a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_train_loss, epoch_val_loss, epoch_val_acc])

        if elapsed_seconds >= TIME_BUDGET_SECONDS:
            print(f"\n[TIME LIMIT REACHED] Stopped training after {elapsed_seconds:.1f}s.")
            break

    # Standardized output line for the AutoResearch runner to parse
    print(f"--- METRIC: val_loss={best_val_loss:.5f} ---")


if __name__ == "__main__":
    run_training()