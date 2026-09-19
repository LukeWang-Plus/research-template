# Research

A lightweight PyTorch template for individual research projects. Research provides a shared workflow for training, hyperparameter search, experiment tracking, evaluation, and result analysis while leaving data structures and algorithms under the researcher's control.

Use the built-in supervised training loop for conventional experiments, or write a custom loop and adopt the experiment-management utilities as needed. The template is intended to be adapted directly to a project, rather than used as a general-purpose training framework.

## Features

- **Local experiment records:** effective configurations, metrics, checkpoints, execution history, tags, and notes.
- **Training and evaluation:** epoch-level recovery, validation-based Top-K checkpoints, optional early stopping, and independent test runs.
- **Hyperparameter search:** single-objective Optuna studies with TPE or grid sampling, optional median pruning, and explicit failed-trial retries.
- **Multi-seed experiments:** lightweight experiment groups, selective testing, and aggregate statistics.
- **Analysis:** result tables, paired significance tests, learning curves, and PNG/PDF export from saved results.
- **Flexible execution:** shared Python and CLI entry points for notebooks, local machines, and cluster jobs.
- **Optional integrations:** TensorBoard, Weights & Biases, and SwanLab. Local recording is always enabled.

## Installation

Requirements:

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Windows or Linux

Clone the repository and install the project environment:

```bash
git clone https://github.com/LukeWang-Plus/research-template.git
cd research-template
uv sync
uv run research --help
```

PyTorch is pinned to **2.7.1**. Windows and Linux installations use the **CUDA 11.8** build, which also supports CPU execution. GPU use requires a compatible NVIDIA driver; changing the PyTorch version or installation source should be an explicit environment decision.

Commands below use Bash syntax and run from the project root. Git Bash can be used on Windows. Other shells may require different quoting for configuration overrides.

## Quick Start

The included linear classification example uses a small synthetic dataset and requires no downloads or GPU.

```bash
# Prepare the example data once.
uv run research prepare configs/linear.toml

# Train and select checkpoints using the validation metric.
uv run research train configs/linear.toml --id first-run

# Evaluate the best checkpoint in a separate test run.
uv run research test experiments/runs/first-run/config.toml \
    --source first-run \
    --id first-test

# Inspect the saved results.
uv run research show first-test
```

The terminal shows a completion summary and a metric table. The saved test record reports `"status": "completed"` and contains a `result` dictionary with the evaluation metrics. Training and test outputs are stored separately under `experiments/runs/`.

Data preparation refuses to overwrite existing example data. Skip that command on subsequent runs. Use a new run ID for a new experiment; omit `--id` to generate one automatically.

A second example, `configs/ragged.toml`, demonstrates regression over variable-length dictionary data using a fully custom training loop.

## Project Layout

```text
core/          Experiment management and the optional supervised loop
methods/       Research methods, models, and custom loops
dataio/        Data preparation, loading, and splitting code
configs/       TOML experiment configurations
notebooks/     Interactive exploration examples
scripts/       Bash and Slurm launch scripts
data/          Local datasets and prepared data
experiments/   Run records, studies, groups, and exported reports
tests/         Template checks and CPU/GPU smoke tests
```

Most research-specific changes belong in `methods/`, `dataio/`, and `configs/`. Adding a method does not require changes to the CLI or experiment-management code.

## Configuration and Data

### Experiment configuration

Start from one of the configurations in `configs/`.

| Field       | Purpose                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------ |
| `method`    | Method name from the mapping in `methods/__init__.py`                                            |
| `data.name` | Dataset identity used in records and comparisons                                                 |
| `data.path` | Local data location, when required by the method                                                 |
| `params`    | Method-specific parameters, such as learning rate and batch size                                 |
| `train`     | Epoch limit, validation metric, optimization direction, checkpoint retention, and early stopping |
| `protocol`  | Identifiers for the task, split, metric definitions, and evaluation procedure                    |
| `seed`      | Explicit experiment seed                                                                         |
| `device`    | `cpu` by default, or an explicit device such as `cuda:0`                                         |
| `output`    | Output root; defaults to `experiments`                                                           |

Recorded experiments require an explicit seed and dataset identity. The supervised loop also requires `train.epochs`, `train.metric`, and `train.direction` (`min` or `max`). Custom methods only need training fields used by their selected capabilities. In-memory and Hugging Face datasets do not require a placeholder local path; both included examples require `data.path`.

Protocol identifiers are declarations supplied by the researcher, not conditions inferred by the template. They should accurately describe the experimental setup.

Use `--set` for small overrides:

```bash
uv run research train configs/linear.toml \
    --id lr-test \
    --set params.lr=0.02 \
    --set seed=7 \
    --set 'device="cuda:0"'
```

Override values use TOML syntax, including quoted strings. Each execution saves its effective configuration. Relative data and output paths resolve against the project root; absolute paths are also supported. Use `--root /path/to/project` when an explicit root is needed.

### Data organization

The recommended layout separates original files from prepared datasets without adding an extra `processed/` level:

```text
data/
├── raw/
│   ├── dataset_a/    # Original files
│   └── dataset_b/
├── dataset_a/       # Prepared data
└── dataset_b/
```

Preprocessing must not overwrite original files. Keep preparation and loading code in `dataio/`, not in the data directory. Hugging Face downloads can remain in their default cache; derived data may be saved under `data/`.

This layout is a convention, not a requirement. Methods control their own formats and splitting procedures. Training does not automatically rebuild data, change splits, or invalidate cached artifacts. The synthetic examples generate prepared data directly and do not require files in `raw/`.

Local data and experiment outputs are excluded from Git. Maintain any required backups separately.

## Training and Evaluation

CLI training and testing display a start summary (run ID, method, device, and output location) and a final status with results or an error. The supervised loop shows an epoch progress bar in an interactive terminal, including elapsed time, training loss, and the validation metric. Slurm logs and redirected output receive one plain-text line per completed epoch instead. Testing has no progress bar; custom loops control their own intermediate display. Optuna retains its native trial output.

Presentation does not replace or duplicate experiment records. Final supervised results remain validation-selected, not simply the last epoch's metrics. Python entry points remain quiet by default.

### Resume or restart

Resume an existing run with a higher epoch limit:

```bash
uv run research train experiments/runs/first-run/config.toml \
    --id first-run \
    --resume \
    --set train.epochs=6
```

The supervised loop resumes from the last saved complete epoch. An interrupted, unsaved epoch is repeated from its beginning; batch-level recovery is not supported. If the checkpoint already reaches the requested epoch limit, the command returns without creating another launch record or opening platform integrations.

Resuming permits changes to the epoch limit, device, and platform options. Changes to data, model architecture, or training hyperparameters require a new experiment.

To discard a run and start again:

```bash
uv run research train configs/linear.toml --id first-run --overwrite
```

**Overwriting deletes the target run's generated contents and cannot be undone.** It is not a recovery operation. Active runs are protected by a writer lock; use a new ID when previous results must remain available.

### Checkpoints and testing

`train.keep_top_k` controls validation-ranked checkpoint retention and defaults to 1. The latest training state is retained separately for recovery. Optional early stopping is disabled by default; enable it with `train.patience` and configure `train.min_delta` as needed.

Testing selects `best` by default. To evaluate another retained checkpoint, use its filename from the run's `checkpoint.json`:

```bash
uv run research test experiments/runs/first-run/config.toml \
    --source first-run \
    --checkpoint CHECKPOINT_FILE

# Evaluate an external checkpoint with an appropriate method configuration.
uv run research test configs/linear.toml --external /path/to/model.pt
```

`CHECKPOINT_FILE` is a placeholder. Source-based testing uses the selected checkpoint's training configuration while allowing separate test data and evaluation settings. A failed or interrupted run can still be tested if it has a valid checkpoint. Test results are independent records and do not change the source run's status or weights.

Checkpoint loading uses `weights_only=True` by default. Methods can supply `load_state(file)` for trusted custom formats. Enable unsafe deserialization or remote model code only after verifying its source; loading errors do not trigger an automatic fallback.

## Hyperparameter Search

### Run and continue a study

The example configurations include a learning-rate search space:

```bash
uv run research tune configs/linear.toml --id search-demo --trials 3
uv run research show search-demo --folder studies
uv run research export configs/linear.toml search-demo configs/selected.toml
```

`--trials` is required and specifies the additional budget for that invocation. The default objective is the best validation value of the configured primary metric. Test results are not used automatically.

Export selects the best completed trial unless `--trial N` is provided. It writes a configuration with study/trial provenance, not model weights, and does not start training.

```bash
# Add two trials to the same study.
uv run research tune configs/linear.toml \
    --id search-demo --continuation --trials 2

# Retry a failed trial as a new trial; replace N with its number.
uv run research tune configs/linear.toml \
    --id search-demo --continuation --retry N --trials 1
```

A continued study must retain its method, data, objective, search space, and sampling/pruning configuration. Retrying fixes the parameters already sampled by the failed trial and starts training from scratch. There is no automatic retry or recovery of trials left in `RUNNING` after a process is killed; inspect them before using Optuna's native tools to change their state.

### Sampling

`search.space` supports Optuna `float`, `int`, and `categorical` distributions under `params.*`. Methods may implement `suggest(trial, config_dict)` for conditional or derived parameters, changing only `params`, and `objective(config, run_directory, trial)` for a custom scalar objective.

The sampler is selected through `search.sampler.strategy`: `tpe` by default, or `grid`. Both use the experiment seed, and all trial training runs use that same seed. Exact search trajectories are not guaranteed across process restarts or parallel execution.

TPE exposes five optional parameters in `search.sampler.params`:

| Parameter          | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `n_startup_trials` | Number of initial trials before model-based sampling |
| `n_ei_candidates`  | Number of expected-improvement candidates            |
| `multivariate`     | Joint parameter modeling                             |
| `group`            | Search-space decomposition for multivariate sampling |
| `constant_liar`    | Account for running trials during sampling           |

Omitted values follow the installed Optuna version's native defaults; explicit values take precedence. Resolved non-null values are saved in the effective configuration. Native `None` defaults, such as automatic multivariate selection, remain omitted in TOML. Preserve the dependency lockfile for reproducibility. A short study may finish within TPE's startup period; the template does not reduce that period to fit the trial budget.

Grid sampling requires finite categorical candidates for every search parameter. For example, replace the existing sampler and learning-rate search definitions with:

```toml
[search.sampler]
strategy = "grid"

[search.space."params.lr"]
type = "categorical"
choices = [0.01, 0.05, 0.1]
```

Remove any TPE parameter table and the replaced distribution's `low`, `high`, or `log` fields. Grid search displays its combination count and stops when exhausted. It does not discretize continuous ranges or switch samplers automatically. Use TPE when the method dynamically introduces search parameters.

### Pruning and tolerated errors

Pruning is disabled by default. To enable Optuna's MedianPruner:

```toml
[search.pruning]
strategy = "median"

# Optional overrides; omitted values use native Optuna defaults.
[search.pruning.params]
n_startup_trials = 5
n_warmup_steps = 0
interval_steps = 1
n_min_trials = 1
```

The values above are the defaults in Optuna 5.0.0, not universal research recommendations. The supervised loop reports once per completed epoch, starting at 1; interpret warmup and interval settings in that unit. Custom loops define their own reporting scale and must explicitly implement pruning support. To disable pruning, use `strategy = "none"` and remove the parameter table.

Configure specific tolerated exception types with `search.catch`, for example `search.catch = ["torch.cuda.OutOfMemoryError"]`. A matching exception marks that trial as failed and allows the study to continue. Other exceptions, including SDK errors exposed by enabled tracking integrations, terminate the invocation. Broad `Exception` and `RuntimeError` catches are not accepted.

## Experiment Groups

Groups organize independent runs across configurations and seeds:

```bash
uv run research group create configs/selected.toml \
    --seed 1 --seed 2 --seed 3 --id seeds-demo
uv run research group run configs/selected.toml seeds-demo
uv run research show seeds-demo --folder groups

# Replace MEMBER_ID with a member shown in the group record.
uv run research group test configs/selected.toml seeds-demo --member MEMBER_ID
```

Repeat `--member` to select multiple runs. Group testing evaluates each selected member's best checkpoint; use an individual test command for other checkpoints.

Group execution is serial and stops on an error without discarding completed members. Subsequent execution skips completed members. Choose `--action resume` or `--action rerun` explicitly for failed or interrupted members: the former requires a usable checkpoint, while the latter overwrites that member's run.

Group names, tags, and notes are initially empty. Set them through `annotate`:

```bash
uv run research annotate seeds-demo --folder groups \
    --name "Multi-seed evaluation" --note "Selected hyperparameters"
```

For parallel work, launch independent runs from separate terminals or use a scheduler. Small-scale local Optuna concurrency is available by continuing the same study from separate processes. The template does not allocate GPUs or guarantee SQLite concurrency on shared network filesystems.

## Results and Analysis

### Find and annotate runs

```bash
uv run research list --kind train --method linear --status completed
uv run research annotate first-run --tag baseline --note "Initial parameter evaluation"
uv run research list --tag baseline
uv run research show first-run
```

Use `--folder studies` or `--folder groups` to inspect or annotate those records. Names, tags, and notes do not affect experiment calculations.

| Output directory       | Contents                                     |
| ---------------------- | -------------------------------------------- |
| `experiments/runs/`    | Training and independent test records        |
| `experiments/studies/` | Optuna studies and trial associations        |
| `experiments/groups/`  | Group definitions and member references      |
| `experiments/reports/` | Exported tables, figures, and configurations |

Run directories contain TOML configurations, `record.json`, metric history, logs, and any checkpoints or method-specific artifacts. An `interrupted` status may be inferred from an inactive writer lock. A study with unfinished launch records is shown as `unconfirmed`, rather than assumed to be active.

### Display and save results

`list` displays a compact experiment table; `show` presents details, results, sources, and execution history. Studies include trial tables and errors, while groups include member states. Missing values appear as `—`; long IDs and text wrap rather than being truncated.

Query and analysis commands (`list`, `show`, `compare`, `summarize`, and `paired`) accept `--save PATH.json` to export their complete return value while retaining the readable terminal view:

```bash
uv run research show first-test --save experiments/reports/first-test.json
uv run research summarize TEST_SEED_1 TEST_SEED_2 \\
    --metric accuracy --save experiments/reports/summary.json
```

Replace the uppercase IDs with actual runs. Existing destinations require `--overwrite`. Relative export paths resolve against the project root; saving is explicit, with no automatic report copies. There is no `--json` terminal mode. CSV and figure exports remain available separately.

Terminal numbers use six significant digits, including scientific notation for small values. JSON/CSV exports and experiment records retain their original numeric precision. Exported query results may include derived information and are not necessarily copies of `record.json`. Python return structures and statistical calculations are unchanged.

### Compare results

Replace the uppercase IDs below with actual run IDs:

```bash
uv run research compare first-test OTHER_TEST_ID \
    --metric accuracy --csv experiments/reports/comparison.csv

uv run research summarize TEST_SEED_1 TEST_SEED_2 TEST_SEED_3 \
    --metric accuracy

uv run research paired accuracy \
    --left A_SEED_1 --left A_SEED_2 \
    --right B_SEED_1 --right B_SEED_2 \
    --test ttest
```

Supervised training records expose validation-selected results; custom runs expose their returned final metrics; test records expose their own evaluation results. Different datasets or protocols may be displayed side by side with a warning, but cannot be pooled for aggregation or significance testing.

Multi-seed summaries require one method and training configuration with distinct seeds. Paired tests require two disjoint collections with matching, unique seeds and at least two pairs. Matching seeds provide alignment, not proof that a paired design is statistically appropriate.

Select `ttest` or `wilcoxon` explicitly. Both tests are two-sided and report the pair count, each side's mean and sample standard deviation, the mean difference (left minus right), and the p-value. The template does not choose a test or declare a result significant. For constant paired differences in a t-test or all-zero Wilcoxon differences, descriptive statistics remain available and the test returns `status="not_computed"`, a null p-value, and a reason. Other invalid inputs raise errors.

### Plot saved metrics

```bash
uv run research plot first-run \
    --metric accuracy \
    --split validation \
    --destination experiments/reports/curve.png
```

Both the metric and split are required. Use `--metric loss --split train` for training loss. Figures default to 300 DPI; change this with `--dpi`, or use a `.pdf` destination for PDF output. Tables, figures, and configuration exports refuse existing destinations unless `--overwrite` is provided.

Python plotting functions return Matplotlib figures for further customization:

```python
from core.analysis import curves, save_figure

fig = curves(["first-run"], "accuracy", split="validation", title="Validation accuracy")
fig.axes[0].set_ylabel("Accuracy")
save_figure(fig, "experiments/reports/curve-paper.pdf")
```

Python exports accept `overwrite=True` when replacement is intended. Plotting reads saved results and does not rerun a model. `core.analysis.classification(targets, predictions)` provides a confusion matrix for class labels, not raw logits.

## Adding a Method

### Supervised methods

Use `methods/linear.py` as a minimal reference:

1. Add a method module under `methods/` and data preparation/loading functions under `dataio/`.
2. Implement `build(config, root) -> Training`, providing the model, loaders, optimizer, batch function, metric function, and optional scheduler.
3. Implement `build_test(config, root) -> Evaluation` and `load_checkpoint(model, file, state=None)` for independent testing.
4. Add the module to `METHODS` in `methods/__init__.py` and create a TOML configuration.

A classification batch function can be written as:

```python
import torch


def batch(model, data, device):
    inputs, targets = (tensor.to(device) for tensor in data)
    logits = model(inputs)
    return {
        "loss": torch.nn.functional.cross_entropy(logits, targets),
        "weight": len(targets),
        "predictions": logits,
        "targets": targets,
    }
```

Training requires a scalar mean `loss` and its averaging `weight`; use the number of valid tokens when averaging over tokens rather than samples. Evaluation requires predictions and targets that can be concatenated along dimension 0; loss and weight may be omitted together. The metric function receives the collected CPU tensors and returns a numeric dictionary. If the batch function supplies loss, the metric function must not return another `loss` entry.

The default loop supports one optimizer and an optional epoch-level scheduler. `Training.eval_batch` can replace the evaluation batch function. Additional stateful components can be supplied through `Training.stateful`; their state and any independent random generators are the method's responsibility.

### Custom loops

Use `methods/ragged.py` as a reference. Implement `run(config, run, root, trial=None)` and return a dictionary of final numeric metrics. Process logs and checkpoints are optional unless the requested capability requires them.

Within the loop, use the recording utilities as needed:

```python
# Values and state are computed by the method.
run.log({"loss": loss_value}, epoch, split="train")
run.save_checkpoint(state, epoch, score=validation_score)
```

For a non-epoch algorithm, pass `step=updates` instead of an epoch. Ranked checkpoints require an explicit `train.direction`.

- **Recovery:** implement state restoration from `run.last["state"]` and declare `supports_resume = True`.
- **Pruning:** declare `supports_pruning = True` and use Optuna's reporting and pruning APIs in the loop.
- **Testing:** implement `test(config, run, root, file, state=None)` and return test metrics.
- **Custom serialization:** provide `load_state(file)` when trusted artifacts require a different loader.

No Trainer subclass or empty lifecycle hooks are required. Install research-specific dependencies with `uv add` as the project evolves.

## Notebook and Cluster Use

### Notebooks

Open `notebooks/quickstart.ipynb` with the project's virtual environment as the kernel. `ipykernel` is included in the project dependencies.

Individual model, data, and evaluation functions can be used directly without registering a method or creating an experiment directory. For a recorded experiment, use the same entry point as the CLI:

```python
from core.config import ROOT, load
from core.runner import execute

config = load(ROOT / "configs/linear.toml", ["train.epochs=1"])
run_id = execute(config)  # Requires prepared data; generates a new run ID.
```

Notebook execution is synchronous within the current kernel. Closing the kernel does not leave a managed background job.

### Bash and Slurm

Both launch scripts require a configuration path:

```bash
bash scripts/run.sh configs/my_method.toml --id my-run
sbatch scripts/train.slurm configs/my_method.toml
```

`run.sh` forwards additional arguments to the training command. Submit the Slurm script from the project root after preparing the environment and data. Adjust its CPU/GPU requests, partition, account, and time limit for the cluster. The provided script requests one GPU and a one-day time limit; requesting more GPUs does not enable distributed training.

## Optional Tracking Integrations

Install the desired extras, repeating `--extra` to retain multiple integrations:

```bash
uv sync --extra tensorboard --extra wandb --extra swanlab
```

Enable TensorBoard through the configuration or an override:

```bash
uv run research train configs/linear.toml --id tb-demo \
    --set platforms.tensorboard=true
uv run tensorboard --logdir experiments/runs/tb-demo
```

For W&B or SwanLab, authenticate as required by the service and add the corresponding table:

```toml
[platforms.wandb]
project = "my-research"

# Enable separately or alongside W&B.
# [platforms.swanlab]
# project = "my-research"
```

The supervised loop records training and validation metrics against epoch coordinates, with stage-prefixed names. Local records remain available independently of these services.

SDK exceptions propagate as experiment failures. Errors handled internally by an SDK are not intercepted, and successful local recording does not guarantee remote synchronization. Training failures are reported to enabled platforms with failure status and error information; normal completion and pruning are treated as non-failure endings.

## Supported Scope

Windows and Linux CPU workflows have passed CI, and both examples have been exercised locally on an NVIDIA RTX 4090. Linux GPU execution and live W&B/SwanLab service integration have not been verified.

The template targets trusted, individual research workflows. It does not provide automatic GPU allocation, distributed training, built-in AMP control, multi-objective optimization, or automatic repair of failed experiments. Custom methods may implement additional training behavior. Dataset validity, evaluation protocols, statistical assumptions, and research conclusions remain the researcher's responsibility.

To check a local installation without writing into existing experiment directories:

```bash
uv run python -m tests.smoke --device cpu
uv run python -m tests.smoke --device cuda:0
```

The test suite requires the TensorBoard extra. For a non-interactive environment:

```bash
uv sync --extra tensorboard
MPLBACKEND=Agg uv run pytest -q
```
