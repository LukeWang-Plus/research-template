# 实施计划

> 状态：首版、四项审查修复及剪枝/统计调整已有实现和下述本地验证。D1–D13 调整已获后续编码授权并完成，本次验证见第 6 节。双系统 CI 与整体验收仍未完成。

## 1. 范围与执行方式

- 依赖文档：[PRD](PRD.md) 定义首版功能与行为，[Architecture](Architecture.md) 定义架构约束；旧参考方案不增加实施要求。
- 一个实施阶段、一次完整交付，不拆多个版本，也不为每个内部步骤设置审批。
- 内部按依赖顺序推进，每完成一块就进行必要验证，最后整体验收；不等全部编码结束才验证。
- 获得实施授权后，已确认设计直接落实；出现设计冲突、需要改变边界或外部阻碍时再确认。
- 下表记录实际实施状态；代码或工作流存在不等于所有验收项通过。未执行项保留明确状态。

## 2. 实施前置

1. 获得代码实施授权，重新核实当前文件、Git 状态和用户改动；不以历史参考中的仓库状态代替现场核实。
2. 实施按 Architecture 的命名约定使用项目名 Research、公共包 `core/` 和 CLI `research`，同步实际项目元数据、入口及使用示例；保留该文档规定的 PyTorch 固定版本与源配置，依赖解析发生冲突时不自行升级。
3. 保留现有文档和用户资产，核实移动范围，不自动迁移旧模型或让新模板依赖旧实现。

## 3. 单阶段内部任务顺序

以下是同一阶段的任务清单，不是多阶段评审门禁。详细行为以依赖文档为准，实施时更新状态和证据。

| 顺序 | 目标与范围 | 交付物 | 最小必要验证 | 状态 |
|---|---|---|---|---|
| 1 | 前置核实、浅层布局 | 基础项目结构与依赖配置 | 新结构不依赖旧代码 | 已实施；未暂存/提交 |
| 2 | 配置、记录、写入保护与方法接入 | TOML/Pydantic 配置、Typer 薄入口、本地记录、写锁、按需加载方法映射 | 生效配置与路径规则、必要报错、覆盖/互斥及最小自定义调用 | 已实施，本地回归通过 |
| 3 | 默认训练、自定义示例、checkpoint、恢复和独立测试 | 默认监督流程、状态字典保存、Top-K、启动记录、测试结果、两个轻量示例 | CPU 训练/测试闭环、恢复边界与有效记录、自定义流程接入 | 已实施，本次含 checkpoint 局部布局修复及验证 |
| 4 | 调参、实验组及辅助平台 | Optuna 搜索/继续/重试/导出、实验组继续、三种可选平台适配 | 可容忍与不可容忍异常、失败 trial 显式重试、组成员恢复/重跑、TensorBoard 写入 | 已实施，本地回归通过；真实服务边界见下 |
| 5 | 查询、比较、显著性检验与重绘 | 列表筛选、标签备注、汇总/配对检验、CSV/图片、独立绘图入口 | 可比性条件、有效结果选择、基础统计输入与从保存结果重绘 | 已实施，本次三项结果比较修复及回归通过 |
| 6 | 使用说明与整体验收 | 精简 Notebook/CLI/Bash/Slurm 示例、CI、实际验证记录 | 按第 4 节的验收矩阵执行并报告结果 | 说明/示例/工作流已提供；整体验收待 CI 与用户确认 |

## 4. 验证与验收

### 4.1 精简质量门禁

- Ruff 格式化和 lint 必须通过。
- 使用少量 pytest 测试和两个 CPU 轻量示例验证关键行为，不为每个函数建立独立测试，不追求覆盖率指标。
- 不增加全量静态类型检查门禁，不认证研究方法。
- 重点覆盖配置、覆盖与写锁、checkpoint 恢复、调参异常及显式重试、实验组继续、比较和独立重绘；优先复用小型端到端场景，不扩大到理论故障矩阵。

此前四项审查修复阶段实际执行的验收命令：

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m tests.smoke --device cpu
uv run python -m tests.smoke --device cuda:0
```

Ruff 排除生成产物，pytest 仅收集 tests；smoke 位于 `tests/smoke.py`，由模块入口手动/CI 调用，不由 pytest 自动收集。上述命令本地通过：Ruff 格式化/lint、11 项 pytest、两个示例的 CPU 与 Windows GPU 训练/恢复/测试闭环。日志见本地 `experiments/reports/review-fixes-verification.txt`（生成产物，受 Git 忽略）。

### 4.2 环境与平台矩阵

| 验证项 | 方式 | 本次要求 | 当前状态 |
|---|---|---|---|
| Windows CPU | GitHub Actions | 精简测试与 CPU 示例通过 | CI 未执行；本机 pytest 与 CPU smoke 已通过，不替代 CI |
| Linux CPU | GitHub Actions | 精简测试与 CPU 示例通过 | 未执行；当前无 Git 仓库/远端，尚不能触发 |
| Ruff 格式化/lint | GitHub Actions，可先本地执行 | 必须通过，无需在所有环境重复运行 | 本地通过；CI 未执行 |
| Windows 单 GPU | 当前电脑本地 smoke test | 实施时核实 CUDA 可用，记录实际设备和结果 | 已通过：RTX 4090，torch 2.7.1+cu118，CUDA 构建 11.8 |
| Linux GPU | 不单独配置 | 不作为首版验收前置条件，明确未单独实测 | 本次不测 |
| TensorBoard | 本地事件文件写入验证，可在 CI 执行 | 无需外部服务，检查实际写入 | pytest 实际写入并读取事件文件通过 |
| W&B / SwanLab | 实现接入，暂不做真实服务测试 | 明确标注未实测，不要求账号或凭据，不用模拟结果宣称接入已验证 | 本次不实测 |

GitHub Actions 工作流文件存在不等于 CI 已通过，必须记录实际执行结果。Windows GPU 与 Linux CPU 通过也不等于 Linux GPU 已验证；首版接受此覆盖边界，不配置专门 GPU CI。

若当前环境无法触发 GitHub Actions，或本地 PyTorch 无法使用 GPU，保留未完成状态并报告具体阻碍，不静默改成 CPU 验证或宣称通过。

### 4.3 完成标准

- PRD 首版能力与已确认架构全部交付，无未经确认的范围扩展。
- 两个轻量示例展示默认流程和自定义流程，文档命令与实现一致；研究者无需修改核心或 CLI 分支即可增加方法。
- 本节要求的 Ruff、CPU CI、Windows 本地 GPU 和 TensorBoard 检查有实际通过证据。
- W&B/SwanLab 暂不实测、Linux GPU 不单独实测的边界明确记录，不作为隐含失败或伪造成功。
- 提交整体验收说明，列出结果、限制和剩余问题，等待用户确认整体验收。

## 5. 执行状态与剩余工作

- 设计与计划讨论、首版实施及本次局部修复授权：已明确确认。
- `dataio/` 与 `tests/smoke.py` 的既有目录调整已核实，未重复移动；本次实际重跑训练验证。PyTorch 版本和安装来源未改。
- 四项审查修复已完成：数据集身份参与协议可比性；结果使用对应完成/来源 checkpoint 启动配置；显式区分默认验证选择、自定义最终指标与独立测试；checkpoint 管理 JSON 与研究状态分离并支持方法级显式加载。
- 新增最小回归覆盖：仅返回指标的自定义方法；同协议不同数据集及同名不同协议；4→10 epoch 恢复与不同来源启动的测试配置；可信特殊对象恢复/测试及不反序列化的查询；管理 JSON 发布前/后的中断、同 epoch 重存和恢复有效历史。连同原有检查共 11 项 pytest 通过。
- PRD、Architecture、本文及使用说明已同步；旧产物不自动迁移、不改写。没有新增注册框架、通用事务、清理服务或管理系统。
- 外部阻碍：本次 `git status` 仍报告当前目录不是 Git 仓库，未创建仓库或推送，GitHub Actions 无实际执行证据。W&B/SwanLab 真实服务和 Linux GPU 按原约定不测；本次不重复执行 Notebook kernel/UI 验证。
- 用户已确认另有备份并授权删除旧代码临时副本，相关排除和忽略规则已清理；这不替代尚未完成的外部验收。无暂存或提交。

### 后续已确认调整（本次已实施）

- 搜索范围仍为 `params.*`，未增加数据集/数据处理参数搜索，也未改动既有目录、数据或 checkpoint 方案。
- `search.pruning` 已改为 `strategy`（`none / median`）和 `params`；默认 `none`。解析时读取当前 Optuna 原生默认参数，严格检查类型和未知参数，并复用原生取值校验。配置快照、trial 记录及导出均保存生效参数；保留方法能力检查，无布尔兼容分支或自动迁移。两个示例配置已同步。
- `paired()` 保留原接口和 CLI，补齐两侧均值/样本标准差、双侧标识与状态。仅约定的差值无方差/全零 Wilcoxon 返回 null p 值和明确原因；输入错误、未约定异常仍报错，不输出显著性判断。
- 在原测试内补充默认 NopPruner、实际 MedianPruner 按 epoch 预热后剪枝、参数校验与持久化/导出、未支持方法拒绝启用、搜索边界，以及正常/退化统计与 CLI JSON、原有输入错误和异常传播检查。不增加测试矩阵或研究方法认证。
- 本次最终验证：`uv run ruff format --check .`、`uv run ruff check .`、`MPLBACKEND=Agg uv run pytest -q` 全部通过，pytest 为 11 项（原用例补充断言，8.47 秒）。普通 pytest 复跑曾因本机 Tk 的 `combobox.tcl` 缺失而出现 1 项绘图失败；未修改绘图代码或修复环境，显式选用非交互 Agg 后端完成验证。失败及复跑输出均保存在 `experiments/reports/pruning-statistics-verification.txt`。本次未重跑 smoke、GPU 或 Notebook，前述 CPU/GPU smoke 证据属于此前版本，不冒充本次执行。
- Architecture §7.2/§8.2、PRD 及根 README 的受影响用法已同步；README 保持用户向、自包含且无开发文档依赖，未加入内部实施细节。CLI 不改；配对命令示例按实际位置参数写为 `research paired accuracy ...`，已通过 CLI 回归。
- 当前目录仍无 Git 仓库，CI 未触发，不宣称通过；W&B/SwanLab 真实服务和 Linux GPU 不在本次范围。PyTorch 固定版本/来源、`data/raw/.gitkeep` 及忽略规则不变，无暂存或提交。


## 6. 实现隐含决策复核后的实施结果

需求以 PRD 为准，参数与接口约束以 Architecture 为准；[讨论记录](research/implementation-choices-pending.md) 仅供溯源。用户随后明确授权本轮代码实施，以下已落实；前述日志仍只代表此前实现，本轮另存验证证据。

| 对应项 | 最小实施范围 | 状态 |
|---|---|---|
| D1、D9、D12 | TPE/Grid、共享 seed、五项 TPE 参数与生效值保存、Grid 候选和耗尽边界；CLI/Python 显式追加预算 | 已实施 |
| D2、D12 | 去除研究语义字段的隐含默认，按能力检查必填；保留 CPU 等便利默认，同步示例 | 已实施 |
| D3、D7、D10 | 保留现有比较边界、早停/平局规则、实验组仅批量测 best | 保留；回归通过 |
| D4 | study 未结束记录的不确定性展示及 trial 数量；保留普通运行只读派生状态 | 已实施 |
| D5、D6、D8 | 清理组级旧错误；保留 study/trial/group 来源；组元数据默认空且可填写 | 已实施 |
| D11 | 默认监督改为 epoch 记录并省略累计 step；同 epoch 外部平台指标完整；残行与有效历史规则、统一显式导出覆盖、整表警告、无操作恢复不建启动 | 已实施；既有残行/有效历史规则保留 |
| D11 | 保留文件职责和命名；允许测试失败/中断来源的有效 checkpoint，不改来源状态 | 保留；回归通过 |
| D12 | 检验类型显式必填；曲线 split/metric 显式必填；DPI 300 可覆盖及混淆矩阵标题；保留 UUID/尺寸等约定 | 已实施 |
| D13 | group create/run/test 分组命令，无旧别名；同步 CLI、脚本、Notebook、用户 README 和测试 | 已实施 |

实施顺序：配置和 sampler → 记录/恢复/实验组 → 分析导出与 CLI → 示例、用户说明及精简验证，仍属同一实施阶段，不另建逐项审批流程。具体配置字段组织等低层细节由编码按现有结构最小实现，不新增管理框架、兼容迁移或无关能力。

验证优先在现有小型测试中补充关键断言：必填与按需缺席、采样器参数/有限 Grid、epoch 历史与平台同横轴记录、来源合并/组错误清理、显式覆盖、无操作恢复不写记录、分组 CLI 与显式绘图/检验参数。执行 Ruff 和受影响的精简测试；默认训练和恢复变动后复跑 CPU smoke，本地 GPU 与外部 CI 按第 4 节记录实际结果，不用旧日志替代。无需为每个默认值单建测试或扩展 W&B/SwanLab 真实服务测试范围。

已排除且用户确认无需修复的两项：
- 测试允许原始 epoch 上限与 checkpoint 来源启动不同，使用来源训练配置是既定语义，保留 Architecture 第 4.3 节实现。
- catch 吞 Ctrl+C 或将 TrialPruned 转为失败的指控不成立，保留 Architecture 第 7.3 节的原生边界，不增加防御代码。

本轮实际结果：
- 已完成上述代码、配置示例、Notebook 和根 README 同步。采样配置为 `search.sampler.strategy/params`；组名称通过既有 annotate 增加 name 参数填写；自定义尺度支持单独 step，不强迫维护 epoch。没有新增依赖、框架或旧产物迁移。训练脚本没有旧实验组命令，未作无意义修改。
- 同步用户用法时发现原 README 的 compare/summarize/plot `--metric`、plot `--destination` 与位置参数实现不一致；已显式声明为必填选项，并验证 compare/plot 的实际 CLI 调用。paired 的 metric 仍为位置参数。
- Ruff 格式化/lint、13 项 pytest（现有用例补断言和两个聚焦检查）、两个示例的 CPU 与 Windows RTX 4090 GPU 训练/恢复/测试 smoke 全部通过。测试显式使用 `MPLBACKEND=Agg`，不改公共绘图后端；TensorBoard 实测确认训练/验证的 epoch 横轴。
- 初次全局格式检查发现 Notebook 长行，格式化后最终检查通过。完整最终命令与输出在 `experiments/reports/implementation-decisions-verification.txt`（本地生成产物，Git 忽略）。同坐标合并发送有本地适配逻辑检查，但不冒充 W&B/SwanLab 真实服务验证。
- CI、Linux GPU、W&B/SwanLab 真实服务及 Notebook kernel/UI 本次未执行；Notebook 仅同步用法并通过格式/lint。外部验收边界不变。
- PyTorch 固定版本/来源和其他依赖未改；README 保持用户向、自包含且不依赖 docs。未暂存或提交；整体验收仍需用户确认。


### 平台 SDK 本地源码复核与结束状态修复

- 本地安装并只读核对锁定的 W&B 0.30.0、SwanLab 0.10.0：初始化/指标接口匹配，SwanLab 兼容 experiment_name；同时确认其部分日志/结束错误只记录、不抛出。用户确认仅处理 SDK 抛出的异常，边界已同步 PRD、Architecture 与用户 README。
- 已将 Run.finish 的状态和错误传至平台关闭入口：W&B 使用退出码及 summary，SwanLab 使用 state/error；最后一批发送抛错及初始化部分失败按失败清理，不改 SDK、不建监控机制。
- Ruff 检查及 14 项 pytest 通过（MPLBACKEND=Agg）；新增单个本地替身回归检查成功/失败/剪枝映射及 SDK 抛错传播，不冒充真实服务验证。本次未重跑 GPU/smoke/CI，未登录或连接外部平台。
- 核对 optional-dependencies 的三个下限已经分别等于 lock 中的 2.21.0/0.30.0/0.10.0，故未改版本约束或锁文件。未暂存或提交。


### TPE 默认值语义纠正

用户明确纠正此前理解：TPE 五项开放参数的缺省值与 MedianPruner 一样，读取当前 Optuna 原生默认，不固定在模板中。已改为签名补全并保留显式覆盖；multivariate 原生 None 不改写成 true，以 TOML 省略字段保留自动行为，seed 仍复用实验 seed。已保存的显式配置不自动改写，依赖与锁文件未变。

已同步 Architecture、用户 README 和讨论记录的替代说明。在现有测试中验证原生默认读取、模拟默认变化、显式覆盖及配置快照回读；14 项 pytest（Agg）、Ruff 格式/lint 通过。本次不重跑 GPU、smoke 或外部平台，未暂存或提交。
