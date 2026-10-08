import sys
import time
from pathlib import Path
from openai import OpenAI

dir_name = Path(__file__).resolve().parent
sys.path.append(str(dir_name))
sys.path.append(str(dir_name.parent))

from agent_tools import get_tools
from paths import get_paths



PATHS = get_paths()
PROGRESS_FILE = PATHS["progress_path"]
PROGRAM_FILE = dir_name / "program.md"

if not PROGRAM_FILE.is_file():
    sys.exit(f"program.md not found at: {PROGRAM_FILE}")

MODEL = "qwen3:4b"
MAX_HISTORY = 12
MAX_OUTPUT = 6000
MAX_EXPERIMENTS = 20         
PROTECTED = ["TIME_BUDGET_SECONDS = 300", "--- METRIC: val_loss=", "SEED = 42", "set_seed()"]

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=900.0)
tools = get_tools()
ch, git, oh = tools["code_handling"], tools["git_actions"], tools["output_handeling"]


FORMAT = """

---
HOW TO ACT: the system runs training, saves progress.csv, commits and resets for you.
You only reply with ONE ```json block and no other text:
{"tool": "code_handling", "arguments": {"action": "run_model"}}
{"tool": "code_handling", "arguments": {"action": "update_code",
  "updates": {"LEARNING_RATE = 2e-3": "LEARNING_RATE = 1e-3"}}}
Experiment 0: call run_model with no changes. After that: update_code with ONE small idea,
then run_model, and repeat. Keys in "updates" must be copied exactly from the current code
shown to you and must appear once. Do not repeat past experiments. Never stop. /no_think
"""
system_msg = {"role": "system", "content": PROGRAM_FILE.read_text(encoding="utf-8") + FORMAT}
history = [{"role": "user", "content": "Start experiment 0: call run_model."}]


end_loop = False
experiment = 0
best = float("inf")
changed = False         
idea = "baseline"
log = []


def update(updates):
    global changed, idea
    if experiment == 0:
        return "ERROR: run the baseline first (run_model with no changes)."
    if not isinstance(updates, dict) or not updates:
        return "ERROR: 'updates' must be a dict {old_text: new_text}."

    code = ch.read_code()
    for old in updates:
        count = code.count(old)
        if count != 1:
            return f"ERROR: text must appear exactly once, found {count}x: {old[:60]!r}"

    new_code = ch.update_code({o: str(n) for o, n in updates.items()})
    if new_code == code:
        return "ERROR: edit changes nothing."
    for p in PROTECTED:
        if p not in new_code:
            return f"ERROR: protected line removed: {p!r}"
    try:
        compile(new_code, "train.py", "exec")
    except SyntaxError as e:
        return f"ERROR: syntax error: {e}"

    ch.write_updated_code(new_code)
    changed, idea = True, oh.describe_updates(updates)
    return "Edit applied. Now call run_model."


def run():
    global experiment, best, changed, end_loop

    if experiment > 0 and not changed:
        return "ERROR: no change since the last run. Call update_code first."

    print("Training...")
    t0 = time.time()
    output = ch.run_model()
    print(f"Finished in {(time.time() - t0) / 60:.2f} minutes.")

    val = oh.parse_val_loss(output)
    n = experiment
    experiment += 1
    changed = False
    if MAX_EXPERIMENTS and experiment >= MAX_EXPERIMENTS:
        end_loop = True

    if val is None:                                 
        git.reset()
        oh.log_progress(PROGRESS_FILE, n, "crash")
        log.append(f"exp {n}: {idea} => crash")
        return f"Experiment {n} CRASHED: {oh.error_summary(output)}. Code restored."

    if val < best:                                  
        best = val
        oh.log_progress(PROGRESS_FILE, n, val)
        log.append(f"exp {n}: {idea} => {val:.5f} (kept)")
        msg = git.commit(f"exp {n}: val_loss={val:.5f}")
        return f"Experiment {n}: val_loss={val:.5f} IMPROVED. {msg}"

    git.reset()                                     
    oh.log_progress(PROGRESS_FILE, n, val)
    log.append(f"exp {n}: {idea} => {val:.5f} (discarded)")
    return f"Experiment {n}: val_loss={val:.5f} not better than {best:.5f}. Code restored."


def finish():
    print("\nFinishing: plot and push...")
    try:
        tools["plot_progress"]()
        print("Plot saved.")
    except Exception as e:
        print(f"Plot failed: {type(e).__name__}: {e}")
    print("Push:", git.push())



print("Baseline commit:", git.commit("baseline before agent run"))
print(f"Agent running with {MODEL}")


while not end_loop:
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[system_msg] + history[-MAX_HISTORY:])
        text = oh.clean_output(response.choices[0].message.content or "")
        print(f"\n[Agent]\n{text}\n")
        history.append({"role": "assistant", "content": text})

        call = oh.extract_tool_call(text)
        if not call:
            history.append({"role": "user",
                            "content": "Reply with one valid ```json tool call. /no_think"})
            continue

        tool, action, args = oh.normalize_call(call)
        print(f"Executing: {tool} -> {action}")

        if tool == "code_handling" and action == "run_model":
            result = run()
        elif tool == "code_handling" and action == "update_code":
            result = update(args.get("updates"))
        elif tool == "code_handling" and action == "read_code":
            result = "The current code is shown below."
        else:
            result = f"ERROR: unknown tool/action: {tool}/{action}"

        result = oh.truncate(result, MAX_OUTPUT)
        print(f"[Result] {result}\n")
        state = oh.build_state(ch.read_code(), best, log)
        history.append({"role": "user", "content": oh.format_result(tool, action, result, state)})

    except KeyboardInterrupt:
        if changed:              # stopped in the middle of an experiment
            git.reset()
        print("\nStopped by user.")
        break
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}. Retrying in 5s...")
        time.sleep(5)

finish()
print("Loop finished.")