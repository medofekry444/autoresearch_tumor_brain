import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from Agent.agent_tools import get_tools
from paths import get_paths


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

PATHS = get_paths()

TRAIN_PATH = PATHS["train_path"]
PROGRESS_FILE = PATHS["progress_path"]
PROGRAM_FILE = PROJECT_ROOT / "program.md"

if not PROGRAM_FILE.is_file():
    sys.exit(f"program.md not found at: {PROGRAM_FILE}")


# ============================================================
# 2. CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API")

if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API environment variable is missing.")

MODEL = "gemini-3.5-flash-lite"

MAX_HISTORY = 12
MAX_OUTPUT = 6000
MAX_EXPERIMENTS = 20

PROTECTED = [
    "TIME_BUDGET_SECONDS = 300",
    "--- METRIC: val_loss=",
    "SEED = 42",
    "set_seed()",
]


# ============================================================
# 3. INITIALIZE LLM CLIENT
# ============================================================

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=900.0,
)


# ============================================================
# 4. INITIALIZE TOOL REGISTRY
# ============================================================

tools = get_tools(
    train_path=TRAIN_PATH,
    progress_path=PROGRESS_FILE,
    repo_path=PROJECT_ROOT,
)

ch = tools["code_handling"]
git = tools["git_actions"]
oh = tools["output_handeling"]
plot = tools["plot_progress"]


# ============================================================
# 5. SYSTEM PROMPT
# ============================================================

FORMAT = """
---
HOW TO ACT:

The system runs training, saves progress.csv,
commits and resets for you.

You only reply with ONE ```json block and no other text:

{"tool": "code_handling", "arguments": {"action": "run_model"}}

{"tool": "code_handling", "arguments": {
    "action": "update_code",
    "updates": {
        "LEARNING_RATE = 2e-3": "LEARNING_RATE = 1e-3"
    }
}}

Experiment 0:
Call run_model with no changes.

After that:
1. Update code with ONE small idea.
2. Run model.
3. Repeat.

Keys in "updates" must be copied exactly
from the current code shown to you.

Each key must appear exactly once.

Do not repeat past experiments.
Never stop.

/no_think
"""

system_msg = {
    "role": "system",
    "content": PROGRAM_FILE.read_text(encoding="utf-8") + FORMAT,
}

history = [
    {
        "role": "user",
        "content": "Start experiment 0: call run_model.",
    }
]


# ============================================================
# 6. EXPERIMENT STATE
# ============================================================

end_loop = False
experiment = 0

best = float("inf")
changed = False

idea = "baseline"
log = []


# ============================================================
# 7. UPDATE CODE
# ============================================================

def update(updates):
    global changed, idea

    if experiment == 0:
        return (
            "ERROR: run the baseline first "
            "(run_model with no changes)."
        )

    if not isinstance(updates, dict) or not updates:
        return (
            "ERROR: 'updates' must be a dict "
            "{old_text: new_text}."
        )

    code = ch.read_code()

    for old in updates:
        count = code.count(old)

        if count != 1:
            return (
                f"ERROR: text must appear exactly once, "
                f"found {count}x: {old[:60]!r}"
            )

    new_code = ch.update_code({
        old: str(new)
        for old, new in updates.items()
    })

    if new_code == code:
        return "ERROR: edit changes nothing."

    for protected in PROTECTED:
        if protected not in new_code:
            return (
                f"ERROR: protected line removed: "
                f"{protected!r}"
            )

    try:
        compile(new_code, str(TRAIN_PATH), "exec")

    except SyntaxError as error:
        return f"ERROR: syntax error: {error}"

    ch.write_updated_code(new_code)

    changed = True
    idea = oh.describe_updates(updates)

    return "Edit applied. Now call run_model."


# ============================================================
# 8. RUN EXPERIMENT
# ============================================================

def run():
    global experiment, best, changed, end_loop

    if experiment > 0 and not changed:
        return (
            "ERROR: no change since the last run. "
            "Call update_code first."
        )

    print("Training...")

    start_time = time.time()

    output = ch.run_model()

    elapsed = (time.time() - start_time) / 60

    print(f"Finished in {elapsed:.2f} minutes.")

    val = oh.parse_val_loss(output)

    n = experiment

    experiment += 1
    changed = False

    if MAX_EXPERIMENTS and experiment >= MAX_EXPERIMENTS:
        end_loop = True

    # --------------------------------------------------------
    # CRASH
    # --------------------------------------------------------

    if val is None:

        git.reset()

        oh.log_progress(
            PROGRESS_FILE,
            n,
            "crash",
        )

        log.append(
            f"exp {n}: {idea} => crash"
        )

        return (
            f"Experiment {n} CRASHED: "
            f"{oh.error_summary(output)}. "
            "Code restored."
        )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    if val < best:

        best = val

        oh.log_progress(
            PROGRESS_FILE,
            n,
            val,
        )

        log.append(
            f"exp {n}: {idea} => "
            f"{val:.5f} (kept)"
        )

        message = git.commit(
            f"exp {n}: val_loss={val:.5f}"
        )

        return (
            f"Experiment {n}: "
            f"val_loss={val:.5f} "
            f"IMPROVED. {message}"
        )

    # --------------------------------------------------------
    # NO IMPROVEMENT
    # --------------------------------------------------------

    git.reset()

    oh.log_progress(
        PROGRESS_FILE,
        n,
        val,
    )

    log.append(
        f"exp {n}: {idea} => "
        f"{val:.5f} (discarded)"
    )

    return (
        f"Experiment {n}: "
        f"val_loss={val:.5f} "
        f"not better than {best:.5f}. "
        "Code restored."
    )


# ============================================================
# 9. FINISH
# ============================================================

def finish():
    print("\nFinishing: plot and push...")

    try:
        plot()
        print("Plot completed.")

    except Exception as e:
        print(f"Plot failed: {type(e).__name__}: {e}")

    print("Push:", git.push())


# ============================================================
# 10. AGENT LOOP
# ============================================================

def main():
    global end_loop

    print(
        "Baseline commit:",
        git.commit("baseline before agent run"),
    )

    print(f"Agent running with {MODEL}")

    while not end_loop:

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[system_msg] + history[-MAX_HISTORY:],
            )

            text = oh.clean_output(
                response.choices[0].message.content or ""
            )

            print(f"\n[Agent]\n{text}\n")

            history.append({
                "role": "assistant",
                "content": text,
            })

            call = oh.extract_tool_call(text)

            if not call:

                history.append({
                    "role": "user",
                    "content": (
                        "Reply with one valid "
                        "```json tool call. /no_think"
                    ),
                })

                continue

            tool, action, args = oh.normalize_call(call)

            print(f"Executing: {tool} -> {action}")

            # ------------------------------------------------
            # TOOL DISPATCH
            # ------------------------------------------------

            if tool == "code_handling":

                if action == "run_model":
                    result = run()

                elif action == "update_code":
                    result = update(
                        args.get("updates")
                    )

                elif action == "read_code":
                    result = (
                        "The current code is shown below."
                    )

                else:
                    result = (
                        f"ERROR: unknown action: {action}"
                    )

            else:
                result = (
                    f"ERROR: unknown tool: {tool}"
                )

            # ------------------------------------------------
            # HANDLE RESULT
            # ------------------------------------------------

            result = oh.truncate(
                result,
                MAX_OUTPUT,
            )

            print(f"[Result] {result}\n")

            state = oh.build_state(
                ch.read_code(),
                best,
                log,
            )

            history.append({
                "role": "user",
                "content": oh.format_result(
                    tool,
                    action,
                    result,
                    state,
                ),
            })

        except KeyboardInterrupt:

            if changed:
                git.reset()

            print("\nStopped by user.")
            break

        except Exception as error:

            print(
                f"Error: "
                f"{type(error).__name__}: "
                f"{error}. Retrying in 5s..."
            )

            time.sleep(5)

    finish()

    print("Loop finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()