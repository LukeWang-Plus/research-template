# Research

**A lightweight, workflow-complete PyTorch template for empirical machine learning research.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.7.1](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/Managed_with-uv-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Optuna 5.0+](https://img.shields.io/badge/Optuna-5.0%2B-0068B7)](https://optuna.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code_Style-Ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

**Research** handles the boilerplate of empirical experimentation—local run tracking, checkpoint management, hyperparameter tuning, multi-seed aggregation, statistical testing, and publication-quality plotting—while leaving datasets, models, and training algorithms entirely under your control.

Adopt the built-in supervised training loop for standard experiments, or write a completely custom training loop and call the recording utilities directly. No monolithic `Trainer` subclasses, forced inheritance, or opaque lifecycle hooks required.

---

## Features

- **Structured Local Tracking:** Automatic recording of resolved TOML configurations, per-epoch/step metric histories, execution logs, checkpoints, tags, and notes under `experiments/`.
- **Training & Checkpointing:** Epoch-level recovery (`--resume`), validation-ranked Top-$K$ checkpoint retention, configurable early stopping, and isolated test evaluations.
- **Hyperparameter Search:** Built-in Optuna integration supporting TPE and Grid sampling, Median pruning, failed-trial retries, and one-command best-trial config export.
- **Multi-Seed Experiment Groups:** Organize and run multi-seed or multi-config sweeps, selectively evaluate checkpoints, and resume interrupted group members.
- **Statistical Analysis & Paper-Ready Figures:** Side-by-side comparison tables, multi-seed summary statistics ($\text{mean} \pm \text{SD}$), two-sided paired $t$-tests and Wilcoxon signed-rank tests, plus SciencePlots/Colorcet figures exported to PDF or PNG.
- **Unified Entry Points:** Identical behavior across the `research` CLI, interactive Jupyter notebooks, and Bash/Slurm cluster jobs, with optional TensorBoard, Weights & Biases, and SwanLab integrations.

---

## Installation

### Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **Windows or Linux**

Clone the repository and synchronize the environment:

```bash
git clone https://github.com/LukeWang-Plus/research-template.git
cd research-template
uv sync
uv run research --help
```

> **Note on PyTorch & GPU Support:** PyTorch is pinned to **2.7.1** (CUDA 11.8 wheel on Windows and Linux, which also runs on CPU out of the box). To use a GPU, ensure a compatible NVIDIA driver is installed and set `device = "cuda:0"` in your configuration or via `--set 'device="cuda:0"'`.

---

## Quick Start

The repository includes two self-contained synthetic examples that run on CPU in seconds without external downloads:

1. **`configs/linear.toml`** — Standard supervised classification using the built-in training loop.
2. **`configs/ragged.toml`** — Variable-length regression demonstrating a fully custom training loop, conditional hyperparameter sampling, and custom checkpoint serialization.

Run the complete supervised workflow in four commands:

```bash
# 1. Generate the synthetic dataset once under data/toy-classification
uv run research prepare configs/linear.toml

# 2. Train the model and retain the best validation checkpoint
uv run research train configs/linear.toml --id first-run

# 3. Evaluate the best checkpoint on the held-out test split in an isolated run
uv run research test experiments/runs/first-run/config.toml \
    --source first-run \
    --id first-test

# 4. Inspect the detailed run record and evaluation metrics
uv run research show first-test
```

- `prepare` never overwrites existing prepared data; skip it on subsequent runs.
- Omit `--id` on any `train`, `test`, `tune`, or `group create` command to auto-generate a unique timestamped ID.

---

## Project Structure

```text
├── core/          # Experiment management, CLI, analysis, plotting, and optional supervised loop
├── methods/       # Research methods, model definitions, and custom training loops
├── dataio/        # Dataset preparation, loading, and splitting logic
├── configs/       # TOML experiment configurations
├── notebooks/     # Interactive exploration and publication figure examples
├── scripts/       # Bash and Slurm launch scripts
├── data/          # Local datasets (excluded from Git)
├── experiments/   # Generated runs, Optuna studies, groups, and exported reports (excluded from Git)
└── tests/         # Template test suite and smoke checks
```

When adapting this template for your own project, almost all of your work takes place in **`dataio/`**, **`methods/`**, and **`configs/`**.

### Directory Conventions

- **Data (`data/`):** Keep immutable raw files under `data/raw/<dataset>/` and write prepared splits to `data/<dataset>/`. Data preparation is always triggered explicitly via `research prepare` so training never silently rebuilds data or alters splits.
- **Outputs (`experiments/`):** Organized automatically into four subdirectories:
  - `experiments/runs/` — Individual training and test run directories (`config.toml`, `record.json`, `metrics.jsonl`, `checkpoint.json`, checkpoints, and logs).
  - `experiments/studies/` — Optuna SQLite databases and study records.
  - `experiments/groups/` — Multi-seed / multi-config group definitions and member links.
  - `experiments/reports/` — Exported JSON/CSV tables, selected TOML configs, and PDF/PNG figures.

---

## Experiment Workflow

### Configuration & CLI Overview

Experiments are defined in declarative TOML files under `configs/`:

```toml
method = "linear"
seed = 42
device = "cpu"            # "cpu" or "cuda:0", "cuda:1", ...
output = "experiments"
name = "tiny classification"
tags = ["example"]

[data]
name = "toy-classification"
path = "data/toy-classification"

[protocol]                # Researcher-declared experimental protocol identifiers
task = "toy-binary-classification"
split = "fixed-generator-17-v1"
metrics = "accuracy-v1"
evaluation = "held-out-v1"

[params]                  # Method-specific hyperparameters (free-form dict)
lr = 0.1
batch_size = 8
hidden = 0
scheduler_gamma = 0.95

[train]
epochs = 4
metric = "accuracy"       # Validation metric used for checkpoint ranking & pruning
direction = "max"         # "max" or "min"
keep_top_k = 1            # Number of best validation checkpoints to keep
# patience = 5            # Optional early stopping patience (in epochs)
# min_delta = 0.0         # Minimum improvement required to reset patience
```

Use `--set KEY=VALUE` on the CLI to override any field using TOML literal syntax (quote string values in Bash):

```bash
uv run research train configs/linear.toml \
    --id lr-sweep \
    --set params.lr=0.02 \
    --set seed=7 \
    --set 'device="cuda:0"'
```

#### CLI Command Reference

| Category | Command | Description |
| :--- | :--- | :--- |
| **Data** | `research prepare <config>` | Run the method's `prepare()` hook to build local dataset splits |
| **Execution** | `research train <config>` | Run training, save checkpoints, or `--resume` / `--overwrite` a run |
| | `research test <config>` | Evaluate a checkpoint (`--source <run_id>` or `--external <path>`) |
| **Tuning** | `research tune <config> --trials N` | Start or continue (`--continuation`) an Optuna hyperparameter study |
| | `research export <config> <study> <dest>` | Export the best (or `--trial N`) parameters to a new TOML file |
| **Groups** | `research group create <configs...> --seed S` | Create a multi-seed / multi-config experiment group |
| | `research group run <config> <group_id>` | Execute group members serially (supports `--action resume\|rerun`) |
| | `research group test <config> <group_id>` | Evaluate selected group members (`--member <run_id>`) |
| **Inspection** | `research list` | Filter runs by `--kind`, `--method`, `--dataset`, `--seed`, `--status`, or `--tag` |
| | `research show <id>` | Display full run, study (`--folder studies`), or group (`--folder groups`) details |
| | `research annotate <id>` | Attach `--name`, `--tag` (`--clear-tags`), or `--note` to a record |
| **Analysis** | `research compare <ids...> --metric M` | Compare runs side by side and optionally export `--csv <path>` |
| | `research summarize <ids...> --metric M` | Compute multi-seed summary statistics ($\text{mean} \pm \text{SD}$, min/max) |
| | `research paired <metric> --left ... --right ...` | Run a two-sided paired `ttest` or `wilcoxon` significance test |
| **Plotting** | `research plot <ids...> --metric M --split S` | Plot training or validation learning curves to `.pdf` or `.png` |
| | `research bars --runs "..." --metric M` | Plot final-metric bar charts with multi-seed standard deviation error bars |

---

### Training, Resuming, and Testing

#### Resuming vs. Overwriting

To continue an interrupted run or extend training to more epochs, pass the run's saved `config.toml` with `--resume`:

```bash
uv run research train experiments/runs/first-run/config.toml \
    --id first-run \
    --resume \
    --set train.epochs=8
```

- **`--resume`** restores the latest completed epoch state (`last.pt`) and allows updating `train.epochs`, `device`, and `platforms`. Data, model architecture, and hyperparameters remain locked to preserve experimental validity.
- **`--overwrite`** permanently deletes the target run directory and starts fresh from scratch:
  ```bash
  uv run research train configs/linear.toml --id first-run --overwrite
  ```

#### Evaluating Checkpoints

Testing always creates an independent run record without mutating the training run. By default, `--source` loads the `best` validation checkpoint:

```bash
# Evaluate the best checkpoint from a training run
uv run research test experiments/runs/first-run/config.toml \
    --source first-run \
    --id first-test

# Evaluate a specific retained top-k checkpoint filename listed in checkpoint.json
uv run research test experiments/runs/first-run/config.toml \
    --source first-run \
    --checkpoint epoch_0003.pt

# Evaluate an external standalone checkpoint file
uv run research test configs/linear.toml --external /path/to/weights.pt
```

> **Security Note:** Checkpoints are loaded with `torch.load(..., weights_only=True)` by default. If a custom method implements `load_state(file)` with `weights_only=False` (such as `configs/ragged.toml`), only load checkpoints from trusted sources.

---

### Hyperparameter Search

Configure search spaces, samplers (`tpe` or `grid`), and pruning (`none` or `median`) under `[search]` in your TOML file:

```toml
[search.sampler]
strategy = "tpe"          # "tpe" (default) or "grid"

[search.pruning]
strategy = "median"       # "none" (default) or "median"

[search.space."params.lr"]
type = "float"            # "float", "int", or "categorical"
low = 0.01
high = 0.2
log = true
```

Run a study, inspect trial results, and export the winning configuration:

```bash
# Run 3 trials in a new study
uv run research tune configs/linear.toml --id search-demo --trials 3

# Add 2 more trials to the existing study
uv run research tune configs/linear.toml --id search-demo --continuation --trials 2

# Retry a failed trial (replace N with the trial number)
uv run research tune configs/linear.toml --id search-demo --continuation --retry N --trials 1

# Inspect study trials and export the best configuration to a new TOML file
uv run research show search-demo --folder studies
uv run research export configs/linear.toml search-demo configs/selected.toml
```

- **Grid Search:** Set `strategy = "grid"` under `[search.sampler]` and define every search parameter with `type = "categorical"` and `choices = [...]`.
- **Tolerated Trial Exceptions:** Set `search.catch = ["torch.cuda.OutOfMemoryError"]` to mark OOM trials as failed without aborting the study.

---

### Multi-Seed Experiment Groups

Once you have selected a configuration, use `research group` to run and evaluate across multiple random seeds:

```bash
# 1. Create a group across seeds 1, 2, and 3
uv run research group create configs/selected.toml \
    --seed 1 --seed 2 --seed 3 \
    --id seeds-demo

# 2. Execute all member runs serially
uv run research group run configs/selected.toml seeds-demo

# 3. Inspect group status and member run IDs
uv run research show seeds-demo --folder groups

# 4. Evaluate specific completed members (repeat --member for multiple runs)
uv run research group test configs/selected.toml seeds-demo \
    --member MEMBER_RUN_1 \
    --member MEMBER_RUN_2
```

If a group execution is interrupted, re-running `research group run` skips already completed members. Pass `--action resume` to resume interrupted members from their latest checkpoint, or `--action rerun` to overwrite and restart failed members.

---

## Analysis & Plotting

### Inspecting & Annotating Records

Filter, tag, and export experiment metadata from the terminal:

```bash
# Filter runs by kind, method, status, or tag
uv run research list --kind train --method linear --status completed

# Add tags, a display name, or research notes
uv run research annotate first-run \
    --name "Linear Baseline" \
    --tag baseline \
    --note "Initial synthetic benchmark"

# Export full query or analysis output to JSON while keeping the terminal table view
uv run research show first-test --save experiments/reports/first-test.json
```

`list`, `show`, `compare`, `summarize`, and `paired` all support `--save <path.json>` (add `--overwrite` to replace an existing file).

### Statistical Comparison

Compare individual runs, aggregate multi-seed metrics, or run paired significance tests across aligned seeds:

```bash
# Side-by-side table comparison (with optional CSV export)
uv run research compare first-test OTHER_TEST_ID \
    --metric accuracy \
    --csv experiments/reports/comparison.csv

# Multi-seed aggregation (mean, sample SD, min, max) for one method/config across seeds
uv run research summarize SEED_1_ID SEED_2_ID SEED_3_ID --metric accuracy

# Two-sided paired significance test (--test ttest or --test wilcoxon)
uv run research paired accuracy \
    --left BASELINE_SEED_1 --left BASELINE_SEED_2 --left BASELINE_SEED_3 \
    --right PROPOSED_SEED_1 --right PROPOSED_SEED_2 --right PROPOSED_SEED_3 \
    --test ttest
```

### Paper-Ready Plotting

Plotting is powered by **Matplotlib**, **SciencePlots**, and **Colorcet** (Glasbey categorical palette and colorblind-safe `CET_CBL2_r` sequential colormap), producing publication-ready `.pdf` or `.png` figures without requiring TensorBoard or cloud services.

#### CLI Plotting

```bash
# Plot training or validation learning curves
uv run research plot first-run \
    --metric accuracy --split validation \
    --label "Baseline" --ylabel "Validation Accuracy" \
    --width 3.4 --height 2.5 \
    --destination experiments/reports/learning_curve.pdf

# Plot multi-seed final metric comparison bars (mean ± 1 sample SD)
uv run research bars \
    --runs "BASE_S1 BASE_S2 BASE_S3" \
    --runs "PROP_S1 PROP_S2 PROP_S3" \
    --label "Baseline" --label "Proposed" \
    --metric accuracy --ylabel "Test Accuracy" \
    --destination experiments/reports/method_comparison.pdf
```

#### Python & Notebook Plotting API

All plotting functions (`plotting.curves`, `plotting.bars`, and `plotting.classification`) return standard `matplotlib.figure.Figure` objects and accept an optional `style` dictionary or list for local rcParams customization:

```python
from core import plotting

# Multi-seed bar comparison
fig = plotting.bars(
    [["BASE_S1", "BASE_S2", "BASE_S3"], ["PROP_S1", "PROP_S2", "PROP_S3"]],
    "accuracy",
    labels=[r"Baseline $\alpha$", r"Proposed $\beta$"],
    figsize=(3.4, 2.5),
    ylabel="Accuracy",
)
plotting.save_figure(fig, "experiments/reports/comparison.pdf", overwrite=True)

# Confusion matrix from 1D ground-truth and predicted class labels
cm_fig = plotting.classification(
    targets=["cat", "cat", "dog", "dog"],
    predictions=["cat", "dog", "dog", "dog"],
    labels=["cat", "dog"],
)
plotting.save_figure(cm_fig, "experiments/reports/confusion.pdf", overwrite=True)
```

For conference-specific figure sizing and typography (e.g., NeurIPS/ICML/CVPR bundles via **`tueplots`**, custom palettes, and LaTeX rendering), see **[`notebooks/paper_figures.ipynb`](notebooks/paper_figures.ipynb)**.

---

## Adding Your Own Method

Registering a new research method requires three steps:

1. Add your data loading/preparation functions in `dataio/<your_data>.py`.
2. Create a method module in `methods/<your_method>.py` and register it in `METHODS` inside `methods/__init__.py`.
3. Create a configuration file in `configs/<your_method>.toml`.

Choose between two integration styles depending on your algorithm:

| Integration Mode | Best For | What the Template Handles Automatically | Reference Implementation |
| :--- | :--- | :--- | :--- |
| **Supervised Loop** | Standard single-optimizer, epoch-based supervised learning | Epoch loop, progress bar, metric aggregation, Top-$K$ checkpoints, early stopping, `--resume`, Optuna pruning, and platform logging | [`methods/linear.py`](methods/linear.py) |
| **Custom Loop** | Ragged/non-tensor data, multiple optimizers (GANs/RL), step-based schedules, or custom state | Run directory creation, config snapshotting, and on-demand `run.log()` / `run.save_checkpoint()` helpers | [`methods/ragged.py`](methods/ragged.py) |

### Option 1: Built-in Supervised Loop

Implement `build(config, root) -> Training` for training, and `build_test(config, root) -> Evaluation` plus `load_checkpoint(model, file, state=None)` for testing:

```python
import torch
from torch.utils.data import DataLoader
from core.training import Evaluation, Training

supports_resume = True
supports_pruning = True


def batch(model, data, device):
    inputs, targets = (tensor.to(device) for tensor in data)
    logits = model(inputs)
    return {
        "loss": torch.nn.functional.cross_entropy(logits, targets),
        "weight": len(targets),        # Sample (or token) count for exact epoch loss averaging
        "predictions": logits,         # Concatenated across batches on CPU for metrics()
        "targets": targets,
    }


def metrics(predictions, targets):
    return {"accuracy": (predictions.argmax(1) == targets).float().mean().item()}


def build(config, root) -> Training:
    net = MyModel(config.params).to(config.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.params["lr"])
    train_loader, val_loader = ...
    return Training(net, train_loader, val_loader, optimizer, batch, metrics)


def build_test(config, root) -> Evaluation:
    net = MyModel(config.params)
    test_loader = ...
    return Evaluation(net, test_loader, batch, metrics)


def load_checkpoint(model, file, state=None):
    if state is None:
        state = torch.load(file, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
```

### Option 2: Custom Training Loop

Implement `run(config, run, root, trial=None) -> dict` to take full control of training while using `run.log()` and `run.save_checkpoint()` as needed:

```python
def run(config, run, root, trial=None):
    # Restore from run.last["state"] if supports_resume = True and resuming
    for epoch in range(1, config.train.epochs + 1):
        train_loss = ...
        val_metrics = {"mse": ...}
        score = val_metrics[config.train.metric]

        # Record metrics (pass step=... instead of epoch for step-based algorithms)
        run.log({"loss": train_loss}, epoch, split="train")
        run.log(val_metrics, epoch, split="validation")

        # Save latest recovery state and update validation-ranked top-k checkpoints
        run.save_checkpoint({"model": model.state_dict()}, epoch, score=score)

    return val_metrics
```

### Optional Method Hooks

| Function / Attribute | Purpose |
| :--- | :--- |
| `prepare(config, root)` | Invoked by `research prepare` to preprocess and save dataset splits |
| `test(config, run, root, file, state=None)` | Custom test evaluation entry point (used when `build_test` is not defined) |
| `suggest(trial, config_dict)` | Sample conditional or dynamic hyperparameters via Optuna's `trial` object |
| `objective(config, run_directory, trial)` | Compute a custom scalar objective for Optuna (defaults to best validation metric) |
| `load_state(file)` | Custom checkpoint deserialization hook for trusted non-standard formats |
| `supports_resume = True` | Declares that the method supports `--resume` recovery |
| `supports_pruning = True` | Declares that the method supports Optuna trial pruning |

---

## Execution Environments & Tracking Integrations

### Jupyter Notebooks

`ipykernel` is installed by default. Select the project `.venv` as your notebook kernel and explore **[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb)** or run experiments programmatically:

```python
from core.config import ROOT, load
from core.runner import execute

config = load(ROOT / "configs/linear.toml", ["train.epochs=2"])
run_id = execute(config)
```

### Bash & Slurm Cluster Jobs

Use the included launch scripts in `scripts/` for local batch execution or Slurm clusters:

```bash
# Local wrapper (forwards extra CLI flags to research train)
bash scripts/run.sh configs/linear.toml --id batch-run --set 'device="cuda:0"'

# Submit a Slurm job (edit GPU, partition, and time limits in scripts/train.slurm)
sbatch scripts/train.slurm configs/linear.toml --id slurm-run
```

When output is redirected to a Slurm log file, interactive progress bars automatically switch to clean, single-line plain-text epoch logs.

### Optional Experiment Trackers (TensorBoard, W&B, SwanLab)

Local recording in `experiments/` is always active. To enable external experiment trackers, install the corresponding optional extras:

```bash
uv sync --extra tensorboard --extra wandb --extra swanlab
```

Enable any combination in your TOML configuration or via `--set`:

```toml
[platforms]
tensorboard = true

[platforms.wandb]
project = "my-research"

[platforms.swanlab]
project = "my-research"
```

```bash
# Or enable TensorBoard on the fly for a single run:
uv run research train configs/linear.toml --id tb-run --set platforms.tensorboard=true
uv run tensorboard --logdir experiments/runs/tb-run
```
