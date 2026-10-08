# autoresearch

This is an experiment to make autoresearch by andrej karpathy on brain tumor detection


## overview 
 
Your are Ai agent implemints concept of autoresearch by Andrej Karpathy to recursively and iteratively improve brain tumor detection to get least val_loss and best accurecy , 
you have some tools to use it (code_handling ,git_actions,plot_progress):
 



## Brain Tumor Detection Model

### Overview
* **Primary Goal**: Analyze brain MRI scans to determine whether a tumor is present and classify its specific type.
* **The 4 Target Categories**:
  1. Glioma tumor (`glioma`)
  2. Meningioma tumor (`meningioma`)
  3. Healthy tissue / No tumor (`notumor`)
  4. Pituitary tumor (`pituitary`)

### How an Image Travels Through the System
1. **Preparation**: Every image is resized to a standardized dimension and its values are balanced so the system can process it efficiently[cite: 2].
2. **Feature Detection**: The image passes through multiple visual stages, finding details from simple to complex:
   * *Stage 1*: Identifies basic lines and edges.
   * *Stage 2*: Recognizes brain patterns, curves, and textures.
   * *Stage 3*: Maps out the full tumor shape and its location within the scan.
3. **Classification**: Gathers all detected patterns and assigns a confidence score to each category, choosing the one with the highest score.

### How the Model Measures Success
* The system relies on a single score called **Validation Loss (`val_loss`)**, which measures prediction error on unseen scans.
* **The Golden Rule**: If a proposed change lowers this error score, the experiment is kept and saved. If the error increases or breaks, the change is immediately discarded to return to the best known version.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch_tumor_brain/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch_tumor_brain/<tag>` from current master.
3. **Read Model files**:Read these files for full context:
   - `data` This is folder for Brain Tumor MRI Dataset of Model which is categorized to Testing,Training each has (glioma,meningioma,notumor,pituitary) 
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `training_log.csv` -this file contains epochs(25 one) ,train_loss, val_loss, val_accuracy per each expierment
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Tools 
   Have alook on tools
- `code_handling` --> this for editing Model to gain optimal metrics that fits least val_loss
               contains of : - `allowed_zones` --> zones need to update 
                             - `read_code`
                             - `run_model`
                             - `update_code`
                             - `write_updated_code`

- `git_actions`   --> you have reset and commit 
               contains of : - `commit` --> commit if only val_loss is lower than prev
                             - `push` --> use it in the end of experminets
                             - `reset` --> reset if val_loss = > lower than prev
                             - `_run` --> to encapsulate actionts above
- `plot_progress` --> after gaining least val_loss you should plot all of experiments output (final val_loss) (depend on progress.csv)

- `output_handeling` --> to handel output its contain (clean_output, extract_tool_call, normalize_call, parse_val_loss,
    error_summary, describe_updates, truncate, log_progress,
    build_state, format_result)



## Experimentation

 Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply with code_handeling.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. only allowed zones (in code_handling)

**What you CANNOT do:**
- Modify any file unless train.py .
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- protected = ["TIME_BUDGET_SECONDS = 300", "--- METRIC: val_loss="]

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. 

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is

## After expermint/s

1- After each expermint:
 
add in progress.csv --> expermint,val_loss 
like :
```
expermint,val_loss
1,0.4
2,0.3
3,0.1
.
.

```
 
2- After you fininsh loop and all of expermints
use plot_progress on progress.csv
 
Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:


## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch_tumor_brain/mar5` or `autoresearch_tumor_brain/mar5-gpu0`).

LOOP FOREVER:

 1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment
5. Read val_loss
6. Record the results in the progress.csv 
8. If val_loss improved (lower), you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, you git reset back to where you started 

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
