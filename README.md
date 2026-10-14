# Research

A lightweight PyTorch template for individual research projects. Research provides a shared workflow for training, hyperparameter search, experiment tracking, evaluation, and result analysis while leaving data structures and algorithms under the researcher's control.

Use the built-in supervised training loop for conventional experiments, or write a custom loop and adopt the experiment-management utilities as needed. The template is intended to be adapted directly to a project, rather than used as a general-purpose training framework.

## Features

- **Local experiment records:** effective configurations, metrics, checkpoints, execution history, tags, and notes.
- **Training and evaluation:** epoch-level recovery, validation-based Top-K checkpoints, optional early stopping, and independent test runs.
- **Hyperparameter search:** single-objective Optuna studies with TPE or grid sampling, optional median pruning, and explicit failed-trial retries.
- **Multi-seed experiments:** lightweight experiment groups, selective testing, and aggregate statistics.
- **Analysis:** result tables, paired significance tests, learning curves, final-metric bars with sample-standard-deviation error bars, confusion matrices, and PNG/PDF export.
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

A second example, `configs/ragged.toml`, demonstrates regression over variable-length dictionary data using a fully custom training loop and custom state loading; only load its checkpoints from trusted sources.

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

CLI training and testing display the run ID, method, and device at the start, followed by the final status, output location, and results or an error at the end. The supervised loop shows an epoch progress bar in an interactive terminal, including elapsed time, training loss, and the validation metric. Slurm logs and redirected output receive one plain-text line per completed epoch instead. Testing has no progress bar; custom loops control their own intermediate display. Optuna retains its native trial output.

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

`list` displays a compact experiment table; `show` presents details, results, sources, and execution history. Studies include trial tables and errors, while groups include member states. Missing values appear as `—`; long IDs and text wrap rather than being truncated. Terminal timestamps are displayed in UTC with second-level precision.

Query and analysis commands (`list`, `show`, `compare`, `summarize`, and `paired`) accept `--save PATH.json` to export their complete return value while retaining the readable terminal view:

```bash
uv run research show first-test --save experiments/reports/first-test.json
uv run research summarize TEST_SEED_1 TEST_SEED_2 \
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

Plotting works without TensorBoard, W&B, or SwanLab. SciencePlots supplies the base appearance; Colorcet supplies Glasbey categorical colors and the sequential colormap for confusion matrices. CLI and Python/Notebook plotting use the same base style automatically, without an initialization call or process-wide style changes. All three plotting functions return native Matplotlib figures for further customization.

```bash
uv run research plot first-run \
    --metric accuracy --split validation \
    --label Baseline --ylabel Accuracy \
    --width 3.4 --height 2.5 \
    --destination experiments/reports/curve.pdf
```

Both metric and split are required. Use `--metric loss --split train` for training loss. Each run produces one curve, without smoothing or multi-seed aggregation. Repeat `--label` in the same order as run IDs, or omit labels to show IDs. Custom histories recorded with `step` use those coordinates; specify `--xlabel` to describe their meaning.

Use `research bars` to compare final metrics. Each quoted `--runs` argument supplies the run IDs for one bar; it does not refer to an experiment-group ID:

```bash
uv run research bars \
    --runs "A_SEED_1 A_SEED_2 A_SEED_3" \
    --runs "B_SEED_1 B_SEED_2 B_SEED_3" \
    --label Baseline --label Proposed \
    --metric accuracy --ylabel Accuracy \
    --destination experiments/reports/comparison.pdf
```

Replace the uppercase IDs with completed runs. Each multi-run bar shows the mean ± one **sample standard deviation** (`ddof=1`), not a confidence interval. A single-run bar shows its value without an error bar. Runs within each bar must have one method/training setup and distinct seeds; all bars must share compatible dataset/protocol identifiers. Repeated run IDs are rejected. Different bars can have different methods, configurations, seed sets, and sample counts. No ranking or significance test is performed. Sample counts and error-bar explanations are not added to the figure; describe them in your paper or caption.

Both commands accept `--title`, `--xlabel`, `--ylabel`, and `--width`/`--height` in inches (default 6×4). This is a general inspection size, not a prescribed single-column or full-width publication size. Label counts must match curves/bars. Colors follow input order; keep that order consistent across figures or customize the native color cycle. Glasbey provides 256 categorical colors; shorter custom cycles repeat using Matplotlib's native behavior, without an automatic palette replacement. Glasbey does not guarantee that every color is distinguishable for all forms of color-vision deficiency.

Python uses the same functions:

```python
from core import plotting

fig = plotting.bars(
    [["A_SEED_1", "A_SEED_2"], ["B_SEED_1", "B_SEED_2"]],
    "accuracy",
    labels=["Baseline", "Proposed"],
    figsize=(3.4, 2.5),
    ylabel="Accuracy",
)
plotting.save_figure(fig, "experiments/reports/comparison.pdf")
```

Confusion matrices accept one-dimensional class labels, not probabilities or logits. Convert model outputs explicitly in research code. This helper has a Python/Notebook entry point; the template does not assume a common prediction-file format:

```python
fig = plotting.classification(
    ["cat", "cat", "dog", "dog"],
    ["cat", "dog", "dog", "dog"],
    labels=["cat", "dog"],
)
plotting.save_figure(fig, "experiments/reports/confusion.pdf")
```

The matrix shows raw counts with black/white annotations chosen for contrast. Colorcet `CET_CBL2_r` maps low counts to light colors and high counts to dark colors, with a blue progression designed for common red-green color-vision deficiencies. It is independent of the categorical palette. `labels` specifies the full class set/order and must include all observed classes without duplicates. The default canvas is 6×4 inches; an explicit `figsize` overrides it. `cmap` accepts native Matplotlib colormap names or objects.

Base settings mostly follow SciencePlots. Advanced customization is available in Python/Notebooks: base style < explicit `style` < explicit drawing arguments < manual Figure/Axes changes. A style overrides only the fields it supplies. Pass external settings through `style`; an enclosing `plt.rc_context(...)` is not guaranteed to override base settings. Drawing and saving use local style contexts rather than changing global defaults.

For example, customize the categorical colors at creation, then adjust an individual axis:

```python
import matplotlib.pyplot as plt

style = {"axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00"])}
fig = plotting.bars(series, "accuracy", style=style)
fig.axes[0].set_ylabel("Accuracy")
plotting.save_figure(fig, "experiments/reports/custom-bars.pdf", style=style)
```

Mathematical labels use Matplotlib's built-in MathText without requiring an external LaTeX installation, for example `labels=[r"Baseline $\alpha$", r"Proposed $\beta$"]`. Full LaTeX rendering can be enabled explicitly with `style={"text.usetex": True}` after preparing the required tools; rendering errors do not trigger an automatic fallback.

The [paper_figures.ipynb](notebooks/paper_figures.ipynb) example demonstrates tueplots conference/journal sizing and typography with synthetic data. Pass the native bundle through `style` when drawing and again when saving. Bundles override only their own fields and do not necessarily define colors. Check the actual submission instructions for the target venue/year. Venue adaptation is not part of the default plotting flow or CLI.

Export to `.png` or `.pdf`. Saving locally applies base export defaults: 300 DPI, PDF TrueType fonts, and no tight cropping. An explicit saving `style` overrides those defaults; explicit `dpi` takes precedence (`--dpi` on the CLI). Saving does not retain the drawing style implicitly or restyle an existing figure. Tight cropping may change saved dimensions; set `savefig.bbox=None` in the saving style when exact canvas size matters. PDF vector elements are resolution-independent. Existing files require `--overwrite` or Python `overwrite=True`. Plotting does not change source records or rerun models.

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
