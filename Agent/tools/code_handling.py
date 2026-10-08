import subprocess
import sys

allowed_zones = [
    [1, '''# -------------------- MUTABLE ZONE 1: HYPERPARAMETERS --------------------
BATCH_SIZE = 32
IMAGE_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
# -------------------------------------------------------------------------'''],
    [2, '''# -------------------- MUTABLE ZONE 2: MODEL ARCHITECTURE -----------------
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(...)
        self.classifier = nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
# -------------------------------------------------------------------------'''],
    [3, '''# ---------------- MUTABLE ZONE 3: PREPROCESSING & AUGMENTATION ---------
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
# ----------------------------------------------------------------------'''],
]


class code_handling:
    def __init__(self, file_path):
        self.file_path = file_path
        self.code = self.read_code()

    def read_code(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.code = f.read()
            return self.code
        except FileNotFoundError:
            return f"File not found: {self.file_path}"
        except Exception as e:
            return f"Error reading file: {e}"

    def run_model(self):
        try:
            result = subprocess.run(
                [sys.executable, str(self.file_path)], 
                capture_output=True, text=True, check=True, timeout=600,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return "Error during code execution: timeout exceeded 600s"
        except subprocess.CalledProcessError as e:
            return f"Error during code execution: {(e.stderr or e.stdout or '')[-3000:]}"

    def update_code(self, updates: dict):
        try:
            for old_code, new_code in updates.items():
                self.code = self.code.replace(old_code, new_code)
            return self.code
        except Exception as e:
            return f"Error during code update: {e}"

    def write_updated_code(self, updated_code=None):
        code_to_write = updated_code if updated_code else self.code
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(code_to_write)
            return "Code updated successfully."
        except Exception as e:
            return f"Error writing updated code: {e}"