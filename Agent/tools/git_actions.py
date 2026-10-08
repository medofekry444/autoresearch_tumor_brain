import subprocess
import sys
from pathlib import Path

Base_url = Path(__file__).resolve().parents[2]
sys.path.append(str(Base_url))
from paths import get_paths

file_path = get_paths().get("train_path")


class GitActions:
    def __init__(self, target_file=None):

        self.target_file = str(target_file) if target_file else None

    def _run(self, *args):
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True)

    def commit(self, message="Auto commit by agent"):
        try:
            stage_target = self.target_file or "."
            self._run("add", stage_target)
            if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
                return "Nothing to commit."
            self._run("commit", "-m", message)
            return "Changes committed successfully."
        except subprocess.CalledProcessError as e:
            return f"Error during git commit: {(e.stderr or str(e)).strip()}"

    def push(self):
        try:
            res=self._run("push")
            return f"Changes pushed successfully.{res.stdout.strip()}"
        except subprocess.CalledProcessError as e:
            return f"Error during git push: {(e.stderr or str(e)).strip()}"

    def reset(self):
        if not self.target_file:
            return "Error: reset requires a target_file to avoid wiping the whole repo."
        try:
            self._run("checkout", "--", self.target_file)
            return "File restored to last commit."
        except subprocess.CalledProcessError as e:
            return f"Error during git reset: {(e.stderr or str(e)).strip()}"