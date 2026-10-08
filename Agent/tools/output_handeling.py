
import csv
import json
import re
from pathlib import Path

class output_handeling:

    def clean_output(text: str) -> str:
        """Remove qwen3 <think> blocks."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


    def extract_tool_call(text: str):
        """Find the tool-call JSON in the model reply. Returns a dict or None."""
        candidates = []
        m = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if m:
            candidates.append(m.group(1))
        raw = re.search(r"(\{.*\})", text, re.DOTALL)
        if raw:
            candidates.append(raw.group(1))
        for c in candidates:
            try:
                data = json.loads(c)
                if isinstance(data, dict) and "tool" in data:
                    return data
            except json.JSONDecodeError:
                continue
        return None


    def normalize_call(tool_data: dict):
        """Tolerate common formatting mistakes. Returns (tool_name, action, arguments)."""
        arguments = dict(tool_data.get("arguments") or {})
        action = arguments.get("action") or tool_data.get("action")
        if not arguments.get("updates") and tool_data.get("updates"):
            arguments["updates"] = tool_data["updates"]
        u = arguments.get("updates")
        if isinstance(u, dict) and "exact_text" in u and "replacement_text" in u:
            arguments["updates"] = {u["exact_text"]: u["replacement_text"]}
        return tool_data.get("tool"), action, arguments


    def parse_val_loss(output: str):
        """Read the METRIC line printed by train.py. Returns float or None."""
        m = re.search(r"METRIC:\s*val_loss=([0-9.]+)", output or "")
        return float(m.group(1)) if m else None


    def error_summary(text: str) -> str:
        """Last 'XxxError: ...' line, or the last non-empty line."""
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        for line in reversed(lines):
            if re.match(r"^\w*(Error|Exception)\b", line):
                return line[:300]
        return lines[-1][:300] if lines else "no output"


    def describe_updates(updates: dict) -> str:
        """Short one-line description of an edit, for the experiment log."""
        return "; ".join(f"{o.strip()[:40]} -> {str(n).strip()[:40]}" for o, n in updates.items())


    def truncate(text, limit: int) -> str:
        return str(text)[:limit]


    def log_progress(progress_file, exp, value) -> None:
        """Append one row to progress.csv (creates the file and header if needed)."""
        progress_file = Path(progress_file)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        new_file = not progress_file.exists()
        with open(progress_file, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["expermint", "val_loss"])
            w.writerow([exp, value])


    def build_state(code: str, best_val_loss: float, experiment_log: list) -> str:
        """Current mutable zones + best score + past experiments, sent to the model."""
        zones = re.findall(r"# -+ MUTABLE ZONE[^\n]*\n(.*?)# -{10,}", code, re.DOTALL)
        zone_text = "\n---\n".join(z.rstrip() for z in zones) or code[:3000]
        best = f"{best_val_loss:.5f}" if best_val_loss != float("inf") else "none yet"
        past = "\n".join(experiment_log[-15:]) or "(none)"
        return (f"CURRENT train.py mutable zones:\n{zone_text}\n\n"
                f"BEST val_loss so far: {best}\n"
                f"Past experiments (do NOT repeat):\n{past}")


    def format_result(tool_name, action, tool_output, state: str) -> str:
        """The message sent back to the model after each tool call."""
        return (f"Result of {tool_name}/{action}:\n{tool_output}\n\n"
                f"{state}\n\nContinue. /no_think")