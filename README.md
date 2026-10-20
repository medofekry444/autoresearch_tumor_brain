This is an experiment to make autoresearch by andrej karpathy on brain tumor detection
data here :
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset


# AutoResearch: Brain Tumor Detection

An autonomous research agent, inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), A local LLM (Ollama) proposes small edits to `train.py`. The code trains the model, keeps an edit if the validation loss improves, and reverts it if not.

## How it works

```
        +--------------------------------------------+
        |  LLM (qwen3:4b via Ollama)                 |
        |  reads program.md + current code           |
        |  proposes ONE small edit as JSON           |
        +----------------------+---------------------+
                               |
                               v
   +-----------------------------------------------------------+
   |  agent.py (the code makes every decision, not the model)  |
   |  1. validate the edit (see "Safety checks")               |
   |  2. write it to train.py                                  |
   |  3. train (5-minute budget)                               |
   |  4. read val_loss from the METRIC line                    |
   |  5. better  -> git commit                                 |
   |     worse   -> git checkout (restore best train.py)       |
   |  6. append the result to progress.csv                     |
   +-----------------------------------------------------------+
```

The loop runs until it reaches `MAX_EXPERIMENTS` or you press Ctrl+C. It then plots the progress and pushes to git.

## The task

Classify brain MRI scans into 4 classes using a small CNN:

| Class | Meaning |
|---|---|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | Healthy tissue |

The score is **`val_loss`** on the `Testing` split. Lower is better.

## Project structure

```
autoresearch_tumor_brain/
├── paths.py                 # central paths (train.py, progress.csv)
├── pyproject.toml
├── progress.csv             # experiment results (created by the agent)
├── Agent/
│   ├── agent.py             # the main loop
│   ├── agent_tools.py       # builds the tool set used by the agent
│   ├── program.md           # instructions sent to the LLM
│   └── tools/
│       ├── code_handling.py     # read / edit / run train.py
│       ├── git_actions.py       # commit / reset / push
│       ├── output_handeling.py  # parse and format output
│       └── plot_progress.py     # plot progress.csv
└── Model/
    ├── train.py             # the file the agent improves
    ├── data/
    │   ├── Training/        # glioma, meningioma, notumor, pituitary
    │   └── Testing/
    ├── training_log.csv     # per-epoch log of the last run
    └── learning_curves.png
```

## What the agent can and cannot change

`train.py` has three **mutable zones**. The agent may edit only these:

1. **Hyperparameters:** batch size, image size, epochs, learning rate, dropout
2. **Model architecture:** the `BrainTumorCNN` class
3. **Training augmentation:** the `train_tf` transform pipeline

Everything else in `train.py` is fixed: the seed, the 5-minute time budget, the validation transforms, and the `METRIC` print line. The agent also cannot edit any other file or install packages.

### Safety checks (enforced in code, not only in the prompt)

- The text to replace must appear **exactly once** in `train.py`.
- Protected lines (`TIME_BUDGET_SECONDS`, `SEED`, `set_seed()`, the `METRIC` line) must still be present.
- The edited file must compile.
- A failed or crashed experiment is reverted with `git checkout -- train.py`, so only that file is touched.
- The agent cannot rerun the same code twice. It must make a change first.

## Reproducibility

`set_seed()` fixes the seed (42) for Python, NumPy and PyTorch, and seeds the DataLoader. Without it, two runs of the same code can differ by chance, and the agent could keep a change that only got lucky. GPU operations are not perfectly deterministic, so tiny differences can remain.

## Setup

### Requirements
- Python 3.10
- An NVIDIA GPU (CUDA) is recommended
- [Ollama](https://ollama.com) with a model pulled, for example `qwen3:4b`
- [uv](https://docs.astral.sh/uv/) or pip
- The Brain Tumor MRI dataset, placed in `Model/data/` with `Training/` and `Testing/` folders

### Install

```bash
git clone <your-repo-url>
cd autoresearch_tumor_brain
uv sync            # or: pip install -r requirements.txt
ollama pull qwen3:4b
```

### Set the Ollama context length

Ollama's default context depends on your VRAM (4k below 24 GiB) and truncates silently. Set it when you start the server, and check it with `ollama ps`:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

## Usage

1. Check that the baseline trains by hand:
   ```bash
   python Model/train.py
   ```
   The last line must be `--- METRIC: val_loss=... ---`. The agent depends on it.
2. Commit the current `train.py`, so there is a version to return to:
   ```bash
   git add Model/train.py && git commit -m "baseline"
   ```
3. Remove old results to start a fresh log:
   ```bash
   rm -f progress.csv Model/progress.csv
   ```
4. Run the agent:
   ```bash
   python Agent/agent.py
   ```

Press **Ctrl+C** to stop. The agent restores the best `train.py`, plots the results and pushes.

## Settings

Edit these at the top of `Agent/agent.py`:

| Setting | Meaning |
|---|---|
| `MODEL` | Ollama model name |
| `MAX_HISTORY` | Number of recent messages sent to the model |
| `MAX_OUTPUT` | Maximum characters of tool output kept in history |
| `MAX_EXPERIMENTS` | Stop after this many experiments (`None` = run until Ctrl+C) |
| `PROTECTED` | Lines that must always stay in `train.py` |

`NUM_CTX` is not passed in the request. Set the context length on the Ollama server as shown above.

## Output

- **`progress.csv`**: one row per experiment (`expermint,val_loss`). A crashed run is recorded as `crash`.
- **`Model/training_log.csv`**: per-epoch train loss, validation loss and accuracy for the latest run.
- **Git history**: one commit per improvement, with the experiment number and `val_loss` in the message.
- **Progress plot**: made by `plot_progress` when the loop ends.

## Results

| Experiment | Change | val_loss | Kept |
|---|---|---|---|
| 0 | Baseline | | |

*(Fill this in from `progress.csv` and `git log`.)*

## Known limitations

- A 4B model sometimes writes malformed tool calls or edits text that is not in the file. The checks reject these edits, and the agent continues.
- There is no quick crash test before training. A bad edit costs one failed run before it is reverted.
- `val_loss` is the best value over all epochs on the test split, which is also used for selection. Differences between experiments can be small, and the score is optimistic. For a final result, evaluate on data the agent never saw.
- The model sees only the mutable zones and the last 15 experiments.

## Credits

- Concept: [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- Dataset: Brain Tumor MRI Dataset (Kaggle)