# DeepBench 重构需求与决策记录

> 状态：历史参考，旧方案不构成新设计约束，尚未授权实施。
> 本文保留旧现状分析与旧重构方案；下文的“已确认”“必须”、首版范围及变更控制均属于历史记录，不再具有现行规范效力。旧现状分析未在本轮重新核实。
> 当前已确认内容以 [产品需求](../PRD.md) 和 [技术架构](../Architecture.md) 为准；阅读入口见 [文档索引](../README.md)。
> 原核对方式：静态只读核对；未执行训练或数据预处理

## 1. 文档目的

本文记录当前代码库的结构与行为、静态核对发现的问题，以及待后续修订的重构目标和约束。

本文仅为重新规划提供历史依据，不包含最终目录结构、类接口或详细迁移步骤。新方案无需继承本文要求，偏离旧方案也不要求遵循本文原有的变更控制。

## 2. 项目目标

将当前论文研究代码重构为仅在本仓库中使用的通用深度学习实验模板 **DeepBench**，使其能够：

- 以清晰的模块边界接入不同数据集、模型、指标和训练流程；
- 允许 baseline 完全自定义复杂训练过程，而不被固定 Trainer 限制；
- 统一管理配置、数据划分、运行记录、制品、checkpoint 和实验比较；
- 以本地记录为权威源，同时支持 TensorBoard、Weights & Biases 和 SwanLab；
- 使用 Optuna 进行可恢复、可审计的超参数优化；
- 支持统一、可信、可复现的多 baseline 与多 seed 对比；
- 同时支持 Notebook 自由探索、正式 Python API 提交和跨平台 CLI 工作流，并明确区分非正式探索与审计实验。

## 3. 审计范围与仓库状态

当前代码库共有 19 个 Python 文件，共 7,676 行。现状分析覆盖入口、配置、数据处理、模型训练与测试、指标和日志等模块。

项目根目录还包含：

- `README.md`
- `pyproject.toml`
- `.python-version`
- `.gitignore`

当前目录没有 Git 元数据或 `uv.lock`。

当前目录结构为：

```text
.
├── main.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
├── models/
│   ├── __init__.py
│   ├── common.py
│   ├── EANN.py
│   ├── HFGD.py
│   ├── M3FEND.py
│   ├── MDFEND.py
│   ├── MetaAdapt.py
│   ├── Model.py
│   └── T3RD.py
├── utils/
│   ├── __init__.py
│   ├── clean.py
│   ├── config.py
│   ├── domain.py
│   ├── log.py
│   └── process.py
├── data/
├── experiments/
├── docs/
│   └── refactor-requirements.md
├── README.md
└── pyproject.toml
```

仓库现状：

- 当前目录没有 `.git`，Git 命令报告不是 Git 仓库，无法确认分支、提交历史或文件跟踪状态；
- `README.md` 为空；
- `data/` 与 `experiments/` 当前仅保留占位文件；
- 仓库中没有代码所依赖的 `pretraind/` 模型目录；
- 没有测试、CI、lint、格式化或静态类型检查配置；
- `pyproject.toml` 当前仅直接声明 `torch==2.7.1`；没有 `uv.lock`，无法核实锁定的传递依赖。

当前 uv 依赖缺失只作为现状记录，不作为独立的首要整改事项；后续实现需要什么依赖，再随实际模块加入。

## 4. 当前系统理解

### 4.1 总入口与配置

`main.py` 是一个 578 行的顶层单体脚本，负责：

- argparse 参数定义；
- `train/test/clean/process` 模式选择；
- 七个模型的静态分发；
- 实验目录创建；
- W&B 登录、初始化与 Sweep；
- 训练和独立测试入口。

当前主要问题：

- 七个模型通过大段重复的 `if/elif` 分发；
- 模型名、调参空间与部分模型参数硬编码；
- 每个模型分支重复目录创建、W&B 初始化和调用逻辑；
- `filter_args()` 会静默丢弃目标函数不接收的参数，配置错误不易发现；
- `--type test` 实际只实现 `Model`，其余模型不会执行独立测试（`main.py:562-578`）；
- 独立测试没有初始化 W&B run，却会在模型测试函数中写入 `wandb.run.summary`。

### 4.2 数据处理与划分

当前预处理流程位于 `utils/process.py`：

```text
<data_path>/<dataset>.jsonl
  → 解析新闻、评论、领域、环境和标签
  → E5 提取新闻、评论和环境特征
  → 构造评论传播图
  → 按固定 source 划分目标测试集
  → 源数据分层拆分 train/valid
  → 构造固定 44 领域树
  → 保存 train.pt / valid.pt / test.pt / domain_tree.pt
```

原始 JSONL 被假定包含：

- 新闻字段：`id`、`content`、`domain`、`label`、`source`、`environment`、`comment`；
- 评论字段：`id`、`comment`、`parent`。

当前数据层高度绑定论文任务：

- 固定 44 领域映射与领域树；
- 固定评论传播图结构；
- 固定目标域 source 名单；
- 固定本地预训练模型路径；
- 缺少显式 schema、字段校验和数据版本。

划分语义存在偏差：

- `train_ratio` 未实际参与划分；
- 测试集由三个硬编码 source 决定；
- 验证集规模由目标测试集规模反推；
- 领域树使用包含验证样本的整个源数据池（`utils/process.py:291-320`）。

### 4.3 模型与训练流程

`models/` 中每个文件不仅定义模型，还重复实现：

- `define_metrics()`
- `Tune()`
- `prepare()`
- `Train()`
- `Test()`

七个模型的接口和生命周期并不统一：

- **EANN**：返回分类 logits 和领域 logits，需要梯度反转损失；
- **MDFEND**：接近普通 logits 分类模型；
- **M3FEND**：训练前需要初始化 memory，训练中持续写入 memory；
- **Model**：通过 `flag=train/adapt/test` 返回不同类型；
- **T3RD**：在一个复杂 `forward()` 中混合训练、对比学习和适应逻辑；
- **HFGD**：依赖全局异构图和 `NeighborLoader`；
- **MetaAdapt**：没有独立模型类，直接实现 functional XLM-R 与元学习循环。

因此当前不存在稳定的模型基类、批数据契约、Step 输出契约、训练生命周期协议或组件注册表。接入一个普通新 baseline，除模型本身外，通常还需要复制约 400 行实验脚手架并修改 `main.py`。

### 4.4 实验记录与制品

当前本地通常保存：

- `log/log.txt`；
- `state.pt` 或模型导出文件；
- `saver.json`；
- HFGD 的特征与图缓存。

训练与验证指标、测试 summary 和调参配置主要写入 W&B。当前没有统一保存：

- 解析后的完整配置；
- 运行命令与代码版本；
- 数据和 split 指纹；
- 完整指标历史；
- 最终 summary；
- 每样本预测；
- 标准学习曲线、混淆矩阵和跨运行比较图；
- 可独立重绘的本地数据文件。

W&B 当前不是单纯的展示后端，而是侵入了入口、模型模块、指标注册、调参和测试结果保存。

### 4.5 超参数调优

当前流程使用 W&B Sweep：

- `main.py` 定义 Sweep；
- 每个模型文件重复实现 `Tune()`；
- 每个 trial 都执行完整训练并自动测试；
- `Model` 的 `alpha × beta × gamma` 网格为 512 组；
- `alpha + beta + gamma` 没有约束，可能令分类损失系数为负；
- 其余模型大多只提供一个固定学习率，不构成有效搜索。

### 4.6 复现性与环境

当前已有：

- Python、NumPy、PyTorch 和 CUDA seed；
- CuDNN benchmark 关闭、deterministic 开启；
- 部分 DataLoader 使用带 seed 的 generator。

但仍存在：

- `PYTHONHASHSEED` 在解释器启动后设置；
- MetaAdapt 主动使用系统熵重置 NumPy seed；
- KMeans 未设置 `random_state`；
- W&B Sweep trial 没有统一重新初始化随机状态；
- 缓存缺少数据、代码和模型指纹；
- checkpoint 没有完整保存实验上下文；
- 测试时在线更新使结果依赖测试顺序和 batch size。

## 5. 静态审计发现的高风险问题

以下问题属于当前代码事实，但不自动等同于新框架的实施优先级。

### 5.1 运行与正确性风险

1. `process_one()` 对根特征额外执行一次 `unsqueeze`，领域树节点特征很可能成为错误维度（`utils/process.py:99-103,174-186`）。
2. `print_redirect_tqdm()` 会捕获并吞掉其上下文中的任意异常，可能隐藏真实训练错误（`utils/log.py:36-45`）。
3. 独立测试没有 W&B run，却写入 `wandb.run.summary`。
4. AUC 使用 `argmax` 后的类别而不是正类概率或连续 score；领域子集只有一个类别时会报错（`metrics/metrics.py:8-36`）。
5. `Model` 的 BU 领域编码器实际使用 TD 边（`models/Model.py:327-338,514-525`）。
6. `Model` 的损失允许 `1-alpha-beta-gamma < 0`。
7. M3FEND 的一个 `nn.Parameter` 经 `unsqueeze().cuda()` 后可能不再被注册。
8. M3FEND/MDFEND 硬编码 `.cuda()`，破坏 CPU 和指定 GPU 支持。
9. M3FEND 保存完整模型对象，其他模型保存 state dict，checkpoint 契约不统一。
10. 大量 `torch.load(..., weights_only=False)` 与完整对象 pickle 依赖隐含的可信制品边界。

### 5.2 实验协议风险

1. **HFGD** 将 train/valid/test 合并构建异构图，并使用全体新闻特征建立新闻间边，属于传导式协议。
2. **MetaAdapt** 从测试集中各选一个带标签样本参与 query 更新，且仍将它们计入最终测试，属于直接测试标签泄漏。
3. **Model/T3RD** 在每个测试 batch 上更新参数，并将更新累积到后续 batch，结果依赖测试顺序和 batch size。
4. 所有普通训练和 Sweep trial 都会自动访问测试集，破坏 blind-test 纪律。

这些行为可能来自原论文实验设计，也可能是实现错误。因为旧模型只作为历史参考，新框架不以保持这些行为或复现旧数值为验收目标。

### 5.3 工程风险

- 工具层反向依赖模型层；
- `utils/__init__.py` eager import 完整数据与模型依赖；
- root logger 和全局 W&B run 造成隐式状态；
- 相对路径依赖当前工作目录；
- HFGD 缓存无版本与数据指纹；
- 无自动化测试和质量门禁；
- 当前目录没有 Git 元数据，无法通过本地 Git 历史恢复代码。

## 6. 已确认的重构决策

### 6.1 项目边界

采用 **通用核心＋领域包**：

- 通用核心不认识 44 领域、评论树或具体模型；
- 当前谣言检测任务作为独立领域包和示例存在；
- 项目只在本模板仓库中使用，不发布 PyPI；
- 项目名、Python 包名和 CLI 命令统一为 `deepbench`；
- 采用 `src` layout 的可安装包和统一 CLI；
- 提供精简、稳定的 `deepbench.api` Python Facade；
- 目标仓库保留 `notebooks/` 和 `scripts/`。

### 6.2 旧代码策略

- 当前七个模型及旧流程只作历史参考；
- 首先建立不可变的 Git 初始快照或 tag；
- 随后从新主干移除旧实现；
- 不维护旧 CLI、旧模块路径或旧 checkpoint 兼容；
- 不以旧模型数值复现作为新框架验收条件。

### 6.3 首版示例范围

首版包含：

1. 小型合成数据，用于文档、CI 和 CPU 端到端 smoke test；
2. `rumor_detection` 领域包；
3. 不依赖外部预训练资产的轻量 TextCNN；
4. train、validate、test、记录、checkpoint、Optuna、multi-seed benchmark 和 compare 全流程。

首版 TextCNN 仅消费正文与标签。旧数据中的领域、source、评论和环境仍可保留，以支持未来 baseline。

### 6.4 可替换 Recipe

框架不强迫所有方法共享一个训练循环：

- 核心提供稳定基础设施与运行上下文；
- 常规监督学习使用默认 Recipe；
- 复杂 baseline 可以替换整个 Recipe；
- 自定义 Recipe 可以拥有自己的优化器数量、阶段顺序、内外循环、memory 初始化或其他训练过程；
- 自定义 Recipe 仍需通过公共服务记录配置、指标、制品、checkpoint 和结果。

统一的是实验输入、公共服务和标准输出，而不是模型内部算法或训练循环。

### 6.5 类型化组件注册

以下组件采用类型化显式注册表：

- model；
- dataset/data component；
- importer；
- splitter；
- view builder；
- recipe；
- metric；
- 其他需要稳定身份和 schema 的扩展能力。

YAML 使用稳定名称组合组件。新增 baseline 不应修改 CLI 分支，也不应依赖任意 Python 导入路径。注册表由无副作用的显式 Catalog 组合，不使用 import decorator、文件扫描、PyPI entry point 或配置中的任意类路径。

### 6.6 配置系统

- 使用 Pydantic v2 定义严格配置 schema；
- 使用 YAML 保存实验配置；
- 配置复用采用显式、类型化的一层片段，不支持递归继承；
- CLI 使用可重复的 `--set path=value` 点式覆盖；
- 未知字段、禁止路径和类型错误必须在创建运行或初始化外部后端前失败；
- 保存完整展开的 resolved config 和配置来源报告；
- 不再使用静默参数过滤。

### 6.7 CLI

CLI 使用 Typer。已确认需要覆盖的顶层动作包括：

- `prepare`；
- `train`；
- `evaluate`；
- `tune`；
- `benchmark`；
- `compare`。

高频动作保持顶层命令；run、study、benchmark 和 report 的查看、恢复、验证等管理操作使用复数资源分组。CLI 默认输出人类可读信息，并提供版本化 `--json` 模式；结构化 stdout 与日志/进度 stderr 分离。

### 6.8 数据 schema 与 Importer

采用 **规范化内部 schema＋按原始 schema 家族复用的 importer**：

```text
原始数据
  → Importer
  → 领域规范化 Example
  → prepare 数据制品
```

原则：

- Dataset 配置描述具体数据来源；
- Importer 按“原始 schema 家族”复用，而不是每个数据集都新写一个；
- 相同原始结构的数据集复用同一 importer；
- 只有结构或解析逻辑实质不同才增加 importer；
- 完全不同任务应增加新的领域 schema 或领域包，而不是强行转换为 `RumorExample`；
- 首个数据垂直切片使用新的规范化 Rumor JSONL Importer；
- 旧 JSONL importer 的架构位置保留，但等真实旧数据重新进入范围并提供脱敏样本后再实现，不按旧代码猜测 raw schema；
- 训练侧 baseline 不理解原始文件格式。

### 6.9 独立数据准备与 Split Manifest

- 数据验证、清洗、规范化、缓存和划分通过独立 `prepare` 动作完成；
- `train` 不会隐式重建数据；
- prepared data 必须包含 schema 版本、原始数据指纹和 split 指纹；
- 官方划分可直接导入；
- 比例或分层划分只生成一次显式样本 ID manifest；
- 后续所有 baseline 与 seed 复用同一 manifest；
- split manifest 是 train/valid/test 样本归属的唯一权威源。

### 6.10 阶段数据权限

- v0.1 的 train/tune/Benchmark-training Context 不提供 test handle，Optuna trial 不能访问测试集；
- v0.1 不交付传导式、测试时适应、alternate evaluator 或其他特殊 test-access protocol，只保留 action-specific capability/registered protocol 的扩展边界；
- 未来引入此类协议时，必须先增加显式 typed permission descriptor、专用 Context/capability、manifest 数据访问记录、protocol fingerprint 和拒绝/泄漏测试；
- 在该扩展正式实现前，配置中请求任何训练阶段 test access 必须 precheck 失败，不能以字符串 split、Recipe 私自读取或未注册 override 绕过。

### 6.11 本地记录为权威源

每个事件先写本地，再分发给零个或多个外部 sink：

- 本地记录始终启用；
- TensorBoard、W&B、SwanLab 是可选 sink；
- 首版实际实现并验证三种 sink；
- 外部后端失败不得破坏训练或本地记录；
- 外部平台不拥有框架的配置、指标或制品语义。

### 6.12 完整本地审计包

所有 Recipe Run 默认保存共同审计核心：

- 解析后的完整配置；
- 运行 metadata、状态与 provenance；
- 控制台/文件日志；
- 完整指标事件历史；
- 结构化 Result 与 canonical summary。

Action-specific 内容按权限和职责保存：

- training run：`last`/`best` checkpoint、训练/验证曲线和 Recipe 附加制品；
- evaluate run：逐样本预测、混淆矩阵/ROC/指标图，不取得训练 checkpoint 写权限；
- Study/Benchmark/Report orchestration Bundle：各自的 canonical export、aggregate、Coverage Snapshot 或 Report，而不是模型 checkpoint；
- derived revision 可以从 sealed authoritative core 重建，不能反向定义运行结论。

大型或领域专用制品可以由配置关闭，但各 action 的必需本地核心记录不可关闭。

### 6.13 代码与环境快照

每次运行自动记录：

- Git commit；
- dirty 状态；
- 未提交代码 patch；
- `uv.lock` 指纹；
- 完整启动命令或 canonical formal request；
- Python、PyTorch、CUDA 与主要依赖版本；
- 操作系统与设备信息。

Notebook 只是可选提交客户端；正式 run 不保存 Notebook 路径、哈希、cell 或快照。

### 6.14 运行目录与索引

- 每个 run 目录是可携带、可审计的权威记录；
- 项目级 SQLite 只作为可重建索引；
- 删除索引后可从运行目录重新构建；
- 查询、筛选、排序、聚合和跨运行绘图由本地 compare 能力完成。

### 6.15 Checkpoint

默认监督 Recipe 同时保存：

- `last`：用于中断恢复；
- `best`：用于最终正式评估。

`last` 必须覆盖当前 Recipe 实际参与续训语义的全部状态。首版默认监督 Recipe 至少包括模型、AdamW、epoch/step、RNG、early-stopping 和 loader epoch-boundary 状态；首版不使用 scheduler 或 AMP。未来 Recipe 启用 scheduler、AMP/GradScaler 或其他有状态组件时，必须通过统一 checkpoint 服务登记相应 StateProvider 后才能声明 resumable。

禁止序列化任意 Python 对象；配置、manifest、指标、权重和运行状态应使用可校验的结构化表示。具体安全格式在架构方案中确定。

### 6.16 Optuna

- W&B Sweep 被 Optuna 取代；
- study 使用本地持久化存储，支持中断恢复；
- 常见搜索空间在 YAML 中定义；
- 条件参数和派生参数通过 Recipe 的编程扩展定义；
- trial 仅使用 train/valid；
- study 选出最佳配置后，才进行正式测试；
- trial、study 与最终最佳运行都进入本地审计与索引体系。

### 6.17 多 seed

- 多 seed 是一等工作流；
- 每个 seed 生成独立 run 和完整审计包；
- 父级 benchmark 记录子运行关系；
- 聚合至少包含均值和标准差，并支持可选置信区间；
- 不把多个 seed 混入同一运行状态。

### 6.18 指标协议

- 核心指标协议任务无关；
- 领域包定义分类、回归或其他任务指标；
- 配置明确模型选择主指标及优化方向；
- metric state 应可聚合并生成标准结果；
- AUC 等指标必须使用正确输入，并明确边界样本行为；
- 指标实现与版本进入兼容性指纹。

### 6.19 结果可比性

compare 根据以下信息判断结果是否同质：

- dataset 与 prepared-data 指纹；
- split manifest 指纹；
- metric 定义与版本；
- evaluation protocol；
- 其他影响实验语义的声明。

默认拒绝将不兼容运行合并为同一正式排行榜。用户可以显式请求非同质 **Diagnostic Report**，但 hard mismatch 不能通过 override 形成 Formal ranking、paired delta 或显著性结论；诊断输出必须醒目标记兼容性差异与非正式性质。

### 6.20 复现策略

采用 **可审计默认＋严格模式**：

- 默认固定 split、所有可控 RNG、配置、环境和代码状态；
- strict 模式启用确定性算法及必要环境设置；
- strict 模式遇到不支持的算子应明确失败，不得静默降级；
- 性能和确定性之间的选择必须写入运行配置。

### 6.21 平台与硬件

首版正式支持：

- Windows；
- Linux；
- CPU；
- 指定单张 GPU。

首版不实现 AMP/GradScaler 或 DDP，但公共接口不得绑定 `cuda:0` 或阻塞未来 precision/distributed 扩展。macOS 暂作 best effort。Required CI 以 CPU 为权威；`v0.1.0` 最终验收仍需单独完成一次真实单 CUDA smoke。

### 6.22 工程质量门禁

首版建立：

- Ruff 格式化与 lint；
- 静态类型检查；
- pytest 单元测试；
- CPU 端到端 smoke test；
- CI 自动执行上述检查。

### 6.23 公共 Python API

- 模板提供精简的 `deepbench.api` Facade；
- Explore Facade 用于非正式、当前 kernel 的数据/组件/结果探索；
- Formal Client 只接受 YAML 路径或不可变强类型 Request；
- 正式提交禁止接收模型对象、callable、Notebook 局部类或未校验 dict；
- DataBuild、Run、Study、Benchmark 和 Report 使用各自的 persistent Handle；
- 纯预检失败不创建正式资源或 Handle。

### 6.24 Notebook

- Notebook 默认是自由探索环境，不自动创建 run；
- 非正式输出进入 Git 忽略的 `notebooks/_scratch/`；
- Notebook 源码和 Markdown 可以进入 Git，但 outputs 与易变 metadata 应在提交前清除；
- Explore 默认隐藏 test，显式确认后才允许非正式读取；
- 只有显式 Client submit 才创建正式审计资源；
- 正式 submit 使用独立本地 worker，worker 不依赖 kernel 生命周期；
- 原型方法进入正式实验前必须迁入领域包、注册、配置并测试。

### 6.25 CLI 与 Worker

- CLI 自身是干净进程，直接调用 shared application service；
- Notebook 正式提交通过隔离 worker 调用同一 service；
- Tune/Benchmark 可按资源计划启动 trial/seed workers；
- worker 使用 PID、heartbeat、锁和协作取消实现重连；
- 首版不要求常驻远程服务或任务队列。

### 6.26 Scripts 与项目布局

- `scripts/` 按 prepare/train/evaluate/tune/benchmark/compare 功能组织；
- 首版 scripts 只正式维护 Bash/Slurm；
- scripts 只做 `uv run deepbench`、环境和调度器薄封装，不承载 split、seed、Study、Benchmark 或指标聚合语义；
- `data/` 保存 DataBuild 与不可变数据制品；
- `experiments/` 保存 runs/studies/benchmarks/reports；
- `configs/` 区分可复用片段和动作根配置；
- 可选后端与开发工具使用 uv dependency groups，并按实际实现加入。

## 7. 首版明确非目标

首版不包含：

- 迁移或修复全部七个旧模型；
- 兼容旧 CLI、旧 W&B Sweep 或旧 checkpoint；
- 复现原论文历史数值；
- AMP/GradScaler；
- DDP 或多机训练；
- PyPI 发布与第三方插件 entry-point 发现；
- 自动下载或打包真实数据与大型预训练模型；
- 训练时隐式重建 prepared data；
- 将传导式、测试时适应等特殊阶段强制纳入所有 baseline 的公共生命周期；
- 递归 Hydra-style 配置继承；
- Notebook 内存对象直接成为正式运行组件；
- Notebook 自动审计所有 cell 或将其来源写入 run；
- PowerShell scripts 镜像；
- 远程 DeepBench server、Celery 或 Kubernetes worker backend。

## 8. 后续架构方案必须回答的问题

下一阶段的具体方案至少需要明确：

1. `src/deepbench` 的模块边界与依赖方向；
2. Run Context、Recipe、Registry、Tracker、Artifact、Checkpoint 和 Metric 协议；
3. 标准监督 Recipe 与完全自定义 Recipe 的最小契约；
4. Pydantic 配置组合、CLI override 和组件配置校验流程；
5. prepared-data manifest、split manifest 和兼容性指纹格式；
6. 本地事件 schema、运行目录布局和可重建 SQLite 索引；
7. TensorBoard、W&B、SwanLab sink 的能力映射与故障隔离；
8. 安全 checkpoint 与数据序列化格式；
9. Optuna study、trial、最佳配置和多 seed benchmark 的层级关系；
10. `prepare/train/evaluate/tune/benchmark/compare` 的 CLI 行为；
11. 公共 Python API、Explore/Formal 边界、persistent Handles 和本地 worker 生命周期；
12. configs/data/experiments/notebooks/scripts 的目录职责与依赖方向；
13. 合成规范化数据、`RumorExample` 和 TextCNN 的垂直切片，以及 Legacy Importer 的后续接入边界；
14. 为当前无 Git 元数据的项目建立 Git 仓库、保存旧代码快照并切换到新主干的实施顺序；
15. 测试矩阵、CI、验收命令和首版完成标准。

## 9. 变更控制

- 本文记录的是已确认需求，不代表已经授权修改代码。
- 进入实施前，应先评审具体架构与分阶段计划。
- 任何会改变数据权限、评估协议、制品权威源或 Recipe 自由度的方案变更，都必须重新确认。
