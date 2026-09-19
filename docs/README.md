# Research 文档索引

## 当前阶段与阅读入口

个人研究模板的定位、安全与工程边界、首版功能、核心架构及单阶段实施计划已确认。首版实现及四项授权审查修复已落地，本地验证已通过；D1–D13 实现决策复核结论已纳入正式文档，已按授权落实代码和用户说明，本次验证单独记录，不沿用此前日志冒充。Windows/Linux CPU CI 已于 2026-09-18 对提交 `84ed8be` 验证通过；用户整体验收仍待确认。CI 证据及覆盖边界见 [实施计划第 4.2 节](ImplementationPlan.md#42-环境与平台矩阵)。此前本地验证日志已按要求清理，保留历史执行摘要；Linux GPU 与 W&B/SwanLab 真实服务仍未验证。

新增终端展示与显式 JSON 保存方案已讨论确认，尚未实施：需求见 PRD 第 5.10 节，展示约束见 Architecture 第 2.4 节，待办见 ImplementationPlan 第 7 节。本轮仅同步开发文档，用户 README 待代码落实后更新。

建议依次阅读 [产品需求](PRD.md)、[技术架构](Architecture.md) 和 [实施计划](ImplementationPlan.md)。当前公共包为 `core/`、CLI 为 `research`，命名与固定 PyTorch 约束见技术架构。旧方案只供参考，不构成新设计约束；其中 DeepBench 的历史称呼保留，不代表当前命名。

## 文档清单与职责

| 文档 | 内容与边界 |
|---|---|
| [DocumentationRules.md](DocumentationRules.md) | 通用文档职责与内容归属规则 |
| [PRD.md](PRD.md) | 项目目标、安全边界、首版功能范围、验收主线、非目标及待讨论事项 |
| [Architecture.md](Architecture.md) | 已确认的浅层目录、配置与 CLI、方法接入、默认循环契约、记录与恢复、checkpoint、调参与平台实现边界；已实现的入口、结果来源与 checkpoint 管理/状态分离契约 |
| [ImplementationPlan.md](ImplementationPlan.md) | 单阶段任务顺序、精简验证、平台实测边界与执行状态 |
| [references/refactor-requirements.md](references/refactor-requirements.md) | 旧现状分析与旧重构方案的历史参考，不具有规范效力 |
| [research/implementation-choices-pending.md](research/implementation-choices-pending.md) | 已完成的实现决策复核过程；正式结论已同步，保留用于溯源，不是第二份规范 |
| [decisions/_Template.md](decisions/_Template.md) | 后续高变更成本决策的 ADR 模板；目前没有正式 ADR |

尚未创建变更记录或设计系统文档；按实际需要创建，不用空文档预设功能或实施阶段。

## 维护约定

- 文档内容按 [通用项目文档规则](DocumentationRules.md) 归属，文件名沿用其中约定。
- 同一规则只在一个权威文档中完整定义，其他文档链接引用。
- 正式结论仅写入已确认内容，待讨论事项明确标注，不把历史方案自动转为新要求。
- 参考文档保留历史背景；其中的旧状态、旧路径和现状分析不代表当前已重新核实的事实。
- 文档确认或修改不等于授权代码实施。

## ADR 维护规则

目前尚未建立正式 ADR，编号、状态流转和 CHANGELOG 同步细则暂未制定；首次需要记录高变更成本决策时再补充。使用 [模板](decisions/_Template.md) 时，不将其中的维护提示视为已经存在的流程。
