import csv
import json
import math
import re
from pathlib import Path


class output_handeling:
    """Parse agent responses, training output, and experiment logs."""

    @staticmethod
    def clean_output(text: str) -> str:
        """Remove model thinking blocks."""
        return re.sub(
            r"<think>.*?</think>",
            "",
            text or "",
            flags=re.DOTALL,
        ).strip()

    @staticmethod
    def extract_tool_call(text: str):
        """Extract the first valid tool-call JSON object."""
        decoder = json.JSONDecoder()

        for match in re.finditer(r"\{", text or ""):
            try:
                data, _ = decoder.raw_decode(text[match.start():])

                if isinstance(data, dict) and isinstance(data.get("tool"), str):
                    return data

            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def normalize_call(tool_data: dict):
        """Normalize a model-generated tool call."""
        raw_arguments = tool_data.get("arguments") or {}

        if not isinstance(raw_arguments, dict):
            return tool_data.get("tool"), None, {}

        arguments = raw_arguments.copy()
        action = arguments.get("action") or tool_data.get("action")

        if not arguments.get("updates") and tool_data.get("updates"):
            arguments["updates"] = tool_data["updates"]

        updates = arguments.get("updates")

        if (
            isinstance(updates, dict)
            and "exact_text" in updates
            and "replacement_text" in updates
        ):
            arguments["updates"] = {
                updates["exact_text"]: updates["replacement_text"]
            }

        return tool_data.get("tool"), action, arguments

    @staticmethod
    def parse_val_loss(output: str):
        """Extract a finite validation loss from training output."""
        matches = re.findall(
            r"METRIC:\s*val_loss\s*=\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            output or "",
        )

        if not matches:
            return None

        value = float(matches[-1])

        return value if math.isfinite(value) else None

    @staticmethod
    def error_summary(text: str) -> str:
        """Return the last exception or error message."""
        lines = [
            line.strip()
            for line in (text or "").splitlines()
            if line.strip()
        ]

        for line in reversed(lines):
            if re.search(r"\b\w*(?:Error|Exception):", line):
                return line[:300]

        return lines[-1][:300] if lines else "no output"

    @staticmethod
    def describe_updates(updates: dict) -> str:
        """Generate a short description of code modifications."""
        return "; ".join(
            f"{str(old).strip()[:40]} -> {str(new).strip()[:40]}"
            for old, new in updates.items()
        )

    @staticmethod
    def truncate(text, limit: int) -> str:
        """Limit the length of a tool response."""
        return str(text)[:max(0, limit)]

    @staticmethod
    def log_progress(progress_file, exp, value) -> None:
        """Append an experiment result to progress.csv."""
        progress_file = Path(progress_file)
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        new_file = (
            not progress_file.exists()
            or progress_file.stat().st_size == 0
        )

        with progress_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if new_file:
                writer.writerow(["experiment", "val_loss"])

            writer.writerow([exp, value])

    @staticmethod
    def build_state(
        code: str,
        best_val_loss: float,
        experiment_log: list,
    ) -> str:
        """Build the experiment state supplied to the model."""
        zones = re.findall(
            r"# -+ MUTABLE ZONE[^\n]*\n(.*?)# -{10,}",
            code,
            re.DOTALL,
        )

        zone_text = (
            "\n---\n".join(zone.rstrip() for zone in zones)
            or code[:3000]
        )

        best = (
            f"{best_val_loss:.5f}"
            if math.isfinite(best_val_loss)
            else "none yet"
        )

        past = "\n".join(experiment_log[-15:]) or "(none)"

        return (
            f"CURRENT train.py mutable zones:\n{zone_text}\n\n"
            f"BEST val_loss so far: {best}\n"
            f"Past experiments (do NOT repeat):\n{past}"
        )

    @staticmethod
    def format_result(
        tool_name,
        action,
        tool_output,
        state: str,
    ) -> str:
        """Format a tool result for the next agent message."""
        return (
            f"Result of {tool_name}/{action}:\n"
            f"{tool_output}\n\n"
            f"{state}\n\n"
            "Continue. /no_think"
        )