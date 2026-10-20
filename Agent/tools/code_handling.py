import os
import subprocess
import sys
import tempfile
from pathlib import Path


class code_handling:
    """Read, modify, validate, and execute the training script."""

    def __init__(self, file_path, timeout=600):
        self.file_path = Path(file_path).resolve()
        self.timeout = timeout

        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"Training file not found: {self.file_path}"
            )

        self.code = self.read_code()

    def read_code(self) -> str:
        """Read the current training script from disk."""
        code = self.file_path.read_text(encoding="utf-8")
        self.code = code
        return code

    def run_model(self) -> str:
        """Execute the training script and capture its output."""
        try:
            result = subprocess.run(
                [sys.executable, str(self.file_path)],
                cwd=self.file_path.parent,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )

            output = result.stdout or ""

            if result.returncode != 0:
                return (
                    f"Training failed with exit code "
                    f"{result.returncode}.\n"
                    f"STDOUT:\n{output[-3000:]}\n"
                    f"STDERR:\n{result.stderr[-3000:]}"
                )

            return output

        except subprocess.TimeoutExpired:
            return (
                "Error during code execution: "
                f"timeout exceeded {self.timeout}s"
            )

        except OSError as error:
            return f"Error starting training: {error}"

    def update_code(self, updates: dict) -> str:
        """
        Generate updated code without modifying the file.

        The caller must validate the result before writing it.
        """
        if not isinstance(updates, dict) or not updates:
            raise ValueError("updates must be a non-empty dictionary.")

        original_code = self.read_code()

        # Validate all search strings against the original version.
        for old_code, new_code in updates.items():
            if not isinstance(old_code, str) or not old_code:
                raise ValueError(
                    "Update search strings must be non-empty strings."
                )

            if not isinstance(new_code, str):
                raise ValueError(
                    "Replacement values must be strings."
                )

            count = original_code.count(old_code)

            if count != 1:
                raise ValueError(
                    f"Expected exactly one occurrence of "
                    f"{old_code[:80]!r}; found {count}."
                )

        # Reject overlapping search strings.
        keys = list(updates)

        for index, first in enumerate(keys):
            for second in keys[index + 1:]:
                if first in second or second in first:
                    raise ValueError(
                        "Overlapping update patterns are not allowed."
                    )

        # Apply replacements simultaneously, avoiding cascading edits.
        import re

        pattern = re.compile(
            "|".join(re.escape(key) for key in sorted(
                keys,
                key=len,
                reverse=True,
            ))
        )

        updated_code = pattern.sub(
            lambda match: updates[match.group(0)],
            original_code,
        )

        return updated_code

    def write_updated_code(self, updated_code: str) -> str:
        """Validate and atomically replace the training script."""
        if not isinstance(updated_code, str):
            raise TypeError("updated_code must be a string.")

        compile(
            updated_code,
            str(self.file_path),
            "exec",
        )

        temporary_path = None

        try:
            original_mode = self.file_path.stat().st_mode

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.file_path.parent,
                prefix=f".{self.file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(updated_code)
                temporary_path = Path(temporary_file.name)

            os.chmod(temporary_path, original_mode)
            os.replace(temporary_path, self.file_path)

            self.code = updated_code

            return "Code updated successfully."

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)