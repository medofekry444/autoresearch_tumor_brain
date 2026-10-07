import csv
from pathlib import Path
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_FILE = BASE_DIR / "training_log.csv"

# 2. Hyperparameters & Constraints
# -------------------- MUTABLE ZONE 1: HYPERPARAMETERS --------------------
BATCH_SIZE = 32
IMAGE_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
# -------------------------------------------------------------------------
TIME_BUDGET_SECONDS = 300  # 5 minutes maximum runtime 


# 3. Model Architecture
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

# 4. Training and Evaluation Pipeline
def run_training() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# ---------------- MUTABLE ZONE 3: PREPROCESSING & AUGMENTATION ---------
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # ----------------------------------------------------------------------

    train_path = DATA_DIR / "Training"
    test_path = DATA_DIR / "Testing"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset folders were not found at {DATA_DIR}. Expected 'Training' and 'Testing'."
        )

    train_ds = datasets.ImageFolder(train_path, transform=tf)
    test_ds = datasets.ImageFolder(test_path, transform=tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
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

    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * x.size(0)

        epoch_train_loss = running_train_loss / len(train_dl.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                running_val_loss += criterion(logits, y).item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()

        epoch_val_loss = running_val_loss / len(test_dl.dataset)
        epoch_val_acc = 100.0 * correct / len(test_dl.dataset)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

        elapsed_seconds = time.time() - start_time
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.2f}% | "
            f"Elapsed: {elapsed_seconds:.1f}s"
        )

        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, epoch_val_acc])

        # Enforce strict 5-minute time budget
        if elapsed_seconds >= TIME_BUDGET_SECONDS:
            print(f"\n[TIME LIMIT REACHED] Stopped training after {elapsed_seconds:.1f}s.")
            break

    # Standardized output line for the AutoResearch runner to parse
    print(f"--- METRIC: val_loss={best_val_loss:.5f} ---")


if __name__ == "__main__":
    run_training()