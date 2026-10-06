import csv
from pathlib import Path
import matplotlib.pyplot as plt

LOG_FILE = Path("autoresearch_tumor_brain/Model/training_log.csv")
OUTPUT_IMAGE = Path("autoresearch_tumor_brain/Model/learning_curves.png")


def main() -> None:
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file {LOG_FILE} does not exist. Run train.py first.")

    epochs, train_losses, val_losses, val_accs = [], [], [], []

    with open(LOG_FILE, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_accs.append(float(row["val_accuracy"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Curves
    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="darkorange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy Curve
    ax2.plot(epochs, val_accs, label="Val Accuracy (%)", color="forestgreen", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy Over Time")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Plot saved successfully to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    main()