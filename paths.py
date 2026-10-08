from pathlib import Path

ROOT = Path(__file__).resolve().parent   


def get_paths():
    return {
        "root": ROOT,
        "train_path": ROOT / "Model" / "train.py",
        "progress_path": ROOT / "progress.csv",
        "log_path": ROOT / "Model" / "training_log.csv",
    }