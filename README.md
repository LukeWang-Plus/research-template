# Research

Research 是面向个人研究的 PyTorch 实验模板：接入自己的数据和方法后，用同一套命令训练、调参、保存结果，并比较不同方法或多个随机种子的表现。结果始终保存在本地，不需要外部平台账号。

你可以直接修改模板来开展文本、时序或图网络等研究。普通监督学习可复用默认训练流程；特殊方法可以自己写循环，不必继承 Trainer。

## 1. 跑通第一次实验

### 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在终端进入项目根目录，然后执行：

```bash
uv sync --extra tensorboard
uv run research --help
```

项目使用 Python 3.12、固定的 `torch==2.7.1`。Windows/Linux 使用 CUDA 11.8 构建，也可以在 CPU 上运行；不需要 GPU 即可完成下面的示例。不要随意升级 PyTorch，使用 GPU 前需确认驱动兼容。

本文命令按 Bash 编写，Windows 可使用 Git Bash。尤其带引号的 `--set` 参数，在其他终端中可能需要调整转义。除非特别说明，命令均从项目根执行。

### 准备 → 训练 → 测试 → 查看

先只运行一个小型分类示例，不需要下载数据或预训练模型：

```bash
# 生成合成数据，仅需准备一次
uv run research prepare configs/linear.toml

# 训练并根据验证指标保存最佳模型
uv run research train configs/linear.toml --id first-run

# 独立测试，默认使用 first-run 的最佳模型
uv run research test experiments/runs/first-run/config.toml --source first-run --id first-test

# 查看测试状态、accuracy 和 loss
uv run research show first-test
```

成功后，最后一条命令的输出中应有 `status: completed`，`result` 中有测试指标。具体数值可能随环境变化，不应把示例分数作为验收目标。

结果保存在：

```text
experiments/
└─ runs/
   ├─ first-run/       # 训练配置、指标、日志和 checkpoint
   └─ first-test/      # 本次测试配置、结果和预测
```

`prepare` 不覆盖已有数据。若数据已准备好，跳过该步即可；重复实验请换一个 `--id`，恢复与覆盖的区别见下文。

想看看完全自定义的训练循环，可另行运行 `configs/ragged.toml`：它是变长字典数据的轻量回归示例，对应代码为 `methods/ragged.py`。

## 2. 设置实验参数与数据

### 修改配置

从 `configs/linear.toml` 或 `configs/ragged.toml` 复制一份 TOML，再修改自己的设置：

| 配置 | 你需要填写什么 |
|---|---|
| `method` | 方法名称，对应 `methods/__init__.py` 中的映射 |
| `data.name` / `data.path` | 数据集身份与数据位置 |
| `params` | 方法参数，例如学习率、batch size、模型维度 |
| `train` | epoch 上限、验证主指标、优化方向、Top-K 与可选早停 |
| `protocol` | 任务、数据划分、指标定义、评估方式的标识 |
| `seed` / `device` | 随机种子和 `cpu` 或 `cuda:0` 等设备 |
| `output` | 实验输出根目录，默认 `experiments` |

seed、数据身份没有默认值，必须填写；设备默认 CPU。默认监督训练还必须填写 `train.epochs/metric/direction`。自定义流程仅按所用能力填写训练字段，HF/内存数据不必提供本地路径；两个 toy 示例需要 `data.path`。

`protocol` 是你对实验条件的声明，不是模板自动检测出来的。同名指标或数据集不代表实验条件一定一致，请准确填写。

临时改几个参数无需复制文件：

```bash
uv run research train configs/linear.toml --id lr-test --set params.lr=0.02 --set seed=7
uv run research train configs/linear.toml --id gpu-test --set 'device="cuda:0"'
```

字符串覆盖需要 TOML 引号。每次实验都会保存实际生效配置，不必靠回忆终端命令复现实验。

相对路径以项目根为基准，也可使用绝对路径。从其他目录启动时，例如：

```bash
uv run --project /path/to/project research train configs/linear.toml --root /path/to/project
```

### 原始数据放哪里

推荐这样组织，不强制采用：

```text
data/
├─ raw/
│  ├─ dataset_a/       # 原始文件，预处理不要覆盖
│  └─ dataset_b/
├─ dataset_a/          # 处理后的数据
└─ dataset_b/
```

- `dataio/` 放数据准备、读取和划分的 **Python 代码**，不放实际数据。
- 处理产物直接按数据集放在 `data/` 下，不必增加 `processed/` 层。
- Hugging Face 下载的数据留在默认缓存，无需复制进项目；自行加工的结果可放到上述数据目录。
- toy 示例直接生成合成数据，不需要先放原始文件到 `raw/`。
- 本地数据和实验产物默认不进入 Git，请自行保管或备份。

接入真实数据时，由研究代码定义转换与划分。多个数据集可以复用同一任务内的读取函数；训练不会替你重新划分、重建数据或判断缓存是否过期。

## 3. 换成自己的方法

大多数改动发生在以下位置，不需要修改 `core/` 或 CLI：

| 位置 | 用途 |
|---|---|
| `dataio/` | 数据准备、读取与划分 |
| `methods/` | 模型、损失、方法构建或自定义循环 |
| `configs/` | 可重复运行的实验设置 |
| `notebooks/` | 探索和小规模验证 |
| `scripts/` | Bash/Slurm 启动示例 |

### 普通监督学习：从 linear 示例改起

1. 将 `methods/linear.py` 复制为自己的方法文件，例如 `methods/my_method.py`。
2. 替换模型和数据读取逻辑；需要处理原始数据时，在 `dataio/` 编写函数，由方法的 `prepare(config, root)` 调用。
3. 修改 `batch(model, batch, device)`：搬运数据、执行前向、计算 loss；修改 `metrics(predictions, targets)` 返回自己的指标。
4. 在 `methods/__init__.py` 的 `METHODS` 中增加 `"my_method": "methods.my_method"`。
5. 复制一份配置，设置 `method = "my_method"`，检查参数及协议标识，再依次执行 prepare、train 和 test。

默认循环替你完成优化器更新、验证、可选早停和 checkpoint 保存。方法提供：

- `build(config, root) -> Training`：模型、训练/验证加载器、优化器、batch 与指标函数，以及可选 scheduler。
- `build_test(config, root) -> Evaluation`：模型、测试加载器、batch 与指标函数；不需要构建训练集。
- `load_checkpoint(model, file, state=None)`：将已加载的状态应用于模型。

以分类 batch 为例（可放入方法文件）：

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

`loss` 是标量平均损失，`weight` 是其平均所用分母；按有效 token 平均时，就填有效 token 数。训练只必须返回这两项。评估必须有可沿第 0 维拼接的预测与目标 Tensor，loss/weight 可以一起省略。指标函数在 CPU 上接收汇总后的预测和目标，返回例如 `{"accuracy": 0.85}` 的数值字典；有 batch loss 时不要再返回同名 `loss` 指标。

默认流程支持单优化器、每 epoch 更新的可选 scheduler。前向计算不同可以用 `Training.eval_batch`；流程本身明显不同时，直接使用自定义循环。

### 特殊方法：自己控制循环

从 `methods/ragged.py` 参考接入。提供 `run(config, run, root, trial=None)`，自己控制算法，最后返回指标字典即可参与比较；不强制验证历史或 checkpoint。

在你的循环中按需使用：

```python
# 以下是已有循环内的调用片段，变量由方法自行计算
run.log({"loss": loss_value}, epoch, split="train")
run.save_checkpoint(state, epoch, score=validation_score)
# 方法结束时，例如：return {"mse": final_mse}
```

- 不按 epoch 运行时，可用 `run.log(values, step=updates, split="train")` 和 `run.save_checkpoint(state, step=updates)`，不必维护第二个计数；排名保存需要明确 `train.direction`。
- 要恢复训练：实际实现状态保存/加载后声明 `supports_resume = True`。恢复状态由 `run.last["state"]` 提供，保存范围由方法负责。
- 要接入剪枝：声明 `supports_pruning = True`，并在循环中调用 Optuna trial 的报告与剪枝接口；声明本身不会替你实现这些能力。
- 要独立测试特殊方法：提供 `test(config, run, root, file, state=None)`，返回测试指标，自行保存特殊预测产物。
- 普通流程额外启用的有状态组件，可通过 `Training.stateful` 提供名称到 `state_dict/load_state_dict` 对象的映射；独立随机源也需自己保存。

研究专属依赖自行 `uv add`。模板不限定你的数据格式，也不要求每个 baseline 都使用调参或恢复功能。

## 4. 在 Notebook 中探索

打开 `notebooks/quickstart.ipynb`，使用本项目环境的 kernel；需要时安装 `uv add ipykernel`。

你可以直接调用自己的数据/模型函数和 `core.training.evaluate`，不必注册方法或创建实验目录。整理成配置后，也可先跑一个小实验：

```python
from core.config import ROOT, load
from core.runner import execute

config = load(ROOT / "configs/linear.toml", ["train.epochs=1"])
# 假定数据已按快速开始准备好；省略 id 会自动分配新 ID
run_id = execute(config)
```

确认无误后用相同配置通过 CLI 正式运行即可。Notebook 在当前 kernel 中同步执行，不提供关闭 kernel 后继续运行或后台重连。

## 5. 恢复、重新运行与选择模型

### 继续原实验

```bash
uv run research train experiments/runs/first-run/config.toml --id first-run --resume --set train.epochs=6
```

默认从最近完整 epoch 继续，未保存的中间进度重跑，不记录累计 batch step。若已达到 epoch 上限，提示无需继续，不新增启动或连接平台。恢复只允许调整 epoch 上限、设备和平台选项；更换数据、模型或训练超参数应新建实验。

### 丢弃旧结果重新开始

```bash
uv run research train configs/linear.toml --id first-run --overwrite
```

**覆盖会删除该实验旧的生成内容，不是恢复，也不能撤销。** 正在写入的实验不能覆盖。已有测试、报告可能引用旧实验，因此希望保留对比记录时请使用新 ID。

### 测试其他 checkpoint

`train.keep_top_k` 控制保留多少个验证排名靠前的模型，默认 1；续训状态另行保留。测试默认用 best，也可选择保留的 Top-K 文件：

```bash
# 文件名从该实验 checkpoint.json 的 top 列表获取
uv run research test experiments/runs/first-run/config.toml --source first-run --checkpoint CHECKPOINT_FILE

# 无来源实验的外部 checkpoint；配置需准确描述模型和评估设置
uv run research test configs/linear.toml --external /path/to/model.pt
```

`CHECKPOINT_FILE` 是占位符。来源测试自动采用该 checkpoint 对应的训练设置，你仍可在测试配置中选择测试数据与评估协议。来源失败/中断但 checkpoint 有效时也可测试，不改变训练状态。测试每次独立保存；显式覆盖也只覆盖指定测试记录，不改来源模型。

Python 中用 `core.records.checkpoint(directory, "last")` 或 `"best"` 获取状态路径，不要硬编码文件名。示例 checkpoint 是含 `model` 的状态字典；特殊格式可在方法中提供 `load_state(file)`。默认安全加载，可信特殊对象确有需要时才显式使用 `weights_only=False`，不要对不明来源文件开启该选项。

## 6. 调参和多 seed 实验

### 搜索并导出选定配置

示例配置已包含学习率搜索空间：

```bash
uv run research tune configs/linear.toml --id search-demo --trials 3
uv run research show search-demo --folder studies
uv run research export configs/linear.toml search-demo configs/selected.toml
```

`--trials` 必须显式指定，是本次追加预算。默认依据验证主指标选 trial，不自动访问测试集。导出只生成配置，不继续训练或继承权重；可以用 `export ... --trial N` 指定已完成 trial。

采样通过 `search.sampler.strategy` 选择 `tpe`（默认）或 `grid`。两者复用配置 seed，所有 trial 训练也用同一 seed；不保证跨进程续跑或并行的采样轨迹完全复现。

TPE 开放以下五项。未填写项读取当前 Optuna 的原生默认值，随安装版本变化；显式填写则固定该值。以下为 Optuna 5.0.0 的示例：

```toml
[search.sampler]
strategy = "tpe"

[search.sampler.params]
n_startup_trials = 10
n_ei_candidates = 24
# multivariate = true  # 按需显式设置；省略时保留 Optuna 自动选择
group = false
constant_liar = true
```

原生默认值会保存到生效配置；`multivariate` 当前默认 None（自动模式），因 TOML 不支持 null 而省略，不强制写成 true。复现时保留依赖锁文件；已有快照中明确保存的值不会自动改写。

`multivariate` 联合建模参数；`group` 在此基础上按条件搜索空间分组，不能在 multivariate=false 时启用。`constant_liar` 避开尚在 RUNNING 的候选附近，包括硬杀残留的 trial。预热数量不随预算缩小：新 study 只跑 3 次通常仍是预热随机采样。

网格搜索需将所有候选预先枚举，例如用以下配置替换 sampler 和学习率搜索表（不保留 TPE 参数表或 float 的 low/high/log）：

```toml
[search.sampler]
strategy = "grid"

[search.space."params.lr"]
type = "categorical"
choices = [0.01, 0.05, 0.1]
```

启动时显示组合总数，耗尽后提前结束。可由已选值派生参数；需要动态新增搜索参数时主动选 TPE，不自动离散化或切换。

常见设置：

- `search.space` 支持 `float`、`int`、`categorical`，搜索范围是 `params.*`；具体写法参考 `configs/linear.toml`。
- 条件参数可在方法中提供 `suggest(trial, config_dict)`，返回只改变 `params` 的完整字典。特殊标量目标可用 `objective(config, run_directory, trial)`，不得用测试结果选参数。
- 剪枝通过 `search.pruning.strategy` 选择 `none`（默认关闭）或 `median`。启用 `median` 时可在 `search.pruning.params` 填写 Optuna 原生参数；未填写的参数按当前 Optuna 默认值补齐，并保存到生效配置及 trial 导出配置。未知参数、错误类型或取值会报错，`none` 不接受参数。自定义循环必须实际支持并主动接入剪枝。
- 可配置 `search.catch = ["torch.cuda.OutOfMemoryError"]`：当前 trial 失败后继续搜索。未声明异常或平台失败停止当前搜索，不自动改变参数重试。不要笼统容忍所有 `Exception` 或 `RuntimeError`。

例如，在 TOML 中启用 median，以下列出当前 Optuna 的原生默认值（不是推荐所有实验照搬的设置）：

```toml
[search.pruning]
strategy = "median"

[search.pruning.params]
n_startup_trials = 5
n_warmup_steps = 0
interval_steps = 1
n_min_trials = 1
```

也可只写 `strategy = "median"`，省略整个参数表。默认监督流程每个完整 epoch 报告一次，Optuna 报告的 step 使用 epoch 编号，从 1 开始；自定义循环需明确自己的报告尺度，再按该尺度设置预热与检查间隔。关闭时使用 `strategy = "none"` 并移除参数表；旧布尔写法不再接受，不自动迁移。继续同一 study 时保持策略和参数不变。

继续搜索或手动重试失败 trial：

```bash
# 本次再追加两个 trial
uv run research tune configs/linear.toml --id search-demo --continuation --trials 2

# N 替换为实际失败 trial 编号；新 trial 从头训练，不恢复权重
uv run research tune configs/linear.toml --id search-demo --continuation --retry N --trials 1
```

同一 study 继续时不要改变方法、数据、目标、采样/剪枝配置或搜索空间。重试仅固定已采样参数；硬杀进程留下的 RUNNING trial 不会自动修复，需要确认进程停止后通过 Optuna 原生 API 处理。

### 运行多 seed 并选择测试

```bash
uv run research group create configs/selected.toml --seed 1 --seed 2 --seed 3 --id seeds-demo
uv run research group run configs/selected.toml seeds-demo
uv run research show seeds-demo --folder groups

# MEMBER_ID 换成 show 输出的成员 ID，可重复 --member 选择多个成员
uv run research group test configs/selected.toml seeds-demo --member MEMBER_ID
```

组内每个成员都有独立运行记录，保留导出配置的 study/trial 来源。组名称、标签和备注默认空，可用 `research annotate seeds-demo --folder groups --name "多 seed 验证" --note "用途"` 填写。批量测试仅选各成员 best，其他候选用单次 test。批量遇错即停，已完成成员保留；再次执行跳过已完成项，失败成员要显式选择 `group run ... --action resume` 或 `--action rerun`。前者需要可用 checkpoint，后者覆盖失败成员从头运行。

普通实验可多开终端或使用 `scripts/` 下的 Slurm 示例，自己分配设备。Optuna 小规模本机并行可先创建 study，再从不同终端使用相同 ID 继续；不承诺共享网络盘上的并发存储行为，也不提供自动 GPU 调度。

## 7. 找结果、比较与论文作图

### 查找和备注

```bash
uv run research list --kind train --method linear --status completed
uv run research annotate first-run --tag baseline --note "首轮参数验证"
uv run research list --tag baseline
uv run research show first-run
```

`annotate --name` 可修改名称（空字符串可清空）。标签用于筛选，备注用于备忘，不影响实验计算。修改 study/group 的标签备注时使用 `--folder studies` 或 `--folder groups`。

结果默认按以下目录组织：

| 目录 | 内容 |
|---|---|
| `experiments/runs/` | 训练、自定义运行和独立测试 |
| `experiments/studies/` | 调参记录 |
| `experiments/groups/` | 多 seed / 多配置实验组 |
| `experiments/reports/` | 你导出的表格和图片 |

查询普通运行会结合写锁显示 interrupted，而 JSON 保留最后写入状态；study 有未结束启动时显示 unconfirmed 和说明，不断言进程仍活动，并列出完成/失败/剪枝数量及未结束 trial。

每个 run 内可以直接查看 `record.json` 的最终指标、TOML 配置、`log.txt` 和指标历史。常用操作优先通过 CLI，不需要手工修改记录文件。

### 比较与汇总

以下大写 ID 均需替换为实际运行 ID：

```bash
uv run research compare first-test OTHER_TEST_ID --metric accuracy --csv experiments/reports/comparison.csv
uv run research summarize TEST_SEED_1 TEST_SEED_2 TEST_SEED_3 --metric accuracy
uv run research paired accuracy --left A_SEED_1 --left A_SEED_2 --right B_SEED_1 --right B_SEED_2 --test ttest
```

- 普通训练比较验证主指标选中的结果；自定义流程比较方法返回的结果；测试比较本次测试结果。
- 不同数据集或评估协议可以并列表格查看，但不能混合汇总或检验；训练和测试结果也不能混合汇总。
- 多 seed 汇总要求同一方法和训练设置、不同 seed，输出均值和样本标准差。恢复延长训练的设置会正确参与检查。
- 配对检验输入是两方法逐次运行的指标，不是两个均值。两侧须有完全匹配且唯一的 seed，至少两对；相同 seed 不自动证明配对合理，实验设计由研究者负责。
- `paired` 的指标名是位置参数（上例为 `accuracy`）。检验类型必须显式指定；`ttest` 是双侧配对 t 检验，也可手动选择 `wilcoxon`，不会自动换检验。差值固定为 left − right，保留指标原单位；是否更好取决于指标方向。
- 返回 JSON 包含 `pairs`、`left` / `right` 各自的 `mean` 与样本 `std`（`ddof=1`）、`mean_difference`、`test`、`alternative="two-sided"`、`p_value` 和 `status`。正常结果为 `status="ok"`；不输出 `significant` 判断，请自行事先确定显著性阈值。
- t 检验差值无方差或 Wilcoxon 全零差值时，仍返回上述描述性结果，但 `p_value=null`、`status="not_computed"`，并用 `reason` 说明原因。数据集/协议不兼容、配对缺失或重复、记录重叠、数量不足、非有限指标等输入错误仍报错。少量 seed 的证据有限，不要根据哪种检验显著来选择检验。

### 不重跑实验就绘图

快速开始完成后，可以直接导出曲线：

```bash
uv run research plot first-run --metric accuracy --split validation --destination experiments/reports/curve.png
```

阶段 `--split` 和指标 `--metric` 均必填；画训练损失用 `--split train --metric loss`。图片默认 300 DPI，可用 `--dpi` 调整；`.pdf` 后缀导出 PDF（矢量部分不受 DPI 影响）。CSV、图片和配置导出默认拒绝覆盖，确需替换目标文件时加 `--overwrite`，Python 对应 `overwrite=True`，不会改来源实验。

需要调整论文图的尺寸、字体或标签时，在 Notebook 调用绘图函数：

```python
from core.analysis import curves, save_figure

fig = curves(["first-run"], "accuracy", split="validation", title="Validation accuracy")
fig.axes[0].set_ylabel("Accuracy")
save_figure(fig, "experiments/reports/curve-paper.pdf")
```

绘图读取已保存记录，不重新训练。分类预测还可传给 `core.analysis.classification(targets, predictions)` 绘制混淆矩阵；这里需要真实类别与预测类别，不是原始 logits。

## 8. 可选平台与常见问题

### 外部记录平台

本地记录始终启用。安装时用 `--extra tensorboard`、`--extra wandb`、`--extra swanlab` 选择平台；需要多个时重复指定 `--extra`。

```bash
uv run research train configs/linear.toml --id tb-demo --set platforms.tensorboard=true
uv run tensorboard --logdir experiments/runs/tb-demo
```

W&B 或 SwanLab 安装后，按平台要求完成账号认证，再在 TOML 添加对应设置，例如：

```toml
[platforms.wandb]
project = "my-research"
```

默认监督流程本地和平台均按 epoch 记录，同轮训练/验证各自保留；指标按阶段命名。

SwanLab 对应 `[platforms.swanlab]`。未启用的平台不会被导入；SDK 抛出的异常会让当前运行报错，请检查连接或移除配置后再继续；SDK 内部自行处理的错误不由模板接管，不保证云端实时同步。训练失败会向平台传递失败状态和错误说明，正常完成及正常剪枝按非失败结束。

### 遇到问题先检查什么

| 情况 | 处理方式 |
|---|---|
| 数据目录已存在 | 已准备好就跳过 prepare；不要为重跑训练删除原始数据 |
| 运行 ID 已存在 | 换新 ID；确需续训用 resume，确需丢弃旧结果才用 overwrite |
| 写锁冲突 | 确认是否有另一进程仍在使用该输出，不强制绕过锁 |
| CUDA 不可用 | 先用 CPU 检查流程，再检查驱动与固定 PyTorch 构建的兼容性 |
| 比较被拒绝 | 检查数据集、协议、训练设置及 seed，不要为通过检查随意改标识 |
| 旧 checkpoint 或记录无法读取 | 不自动迁移旧 `last.pt` 布局或缺少结果来源的记录；新建实验或显式处理旧产物，勿直接改名冒充新格式 |

可在临时目录做环境自检，不会覆盖正式实验：

```bash
uv run python -m tests.smoke --device cpu
uv run python -m tests.smoke --device cuda:0
```

安装了 TensorBoard 后，也可用 `uv run pytest -q` 运行模板检查。两个示例已在 Windows CPU 和 RTX 4090 上运行；Linux 环境、W&B/SwanLab 真实服务尚未实测。集群脚本的资源参数需要按所在环境调整。

模板不内置 AMP、DDP、多机训练或自动 GPU 分配；自定义流程可以按研究需要自行实现。
