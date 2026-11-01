# 17：Agent 框架取舍——先看任务结构，再选工具

> 对应课程：Phase 11 / 17-agent-framework-tradeoffs。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/17-agent-framework-tradeoffs/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/17-agent-framework-tradeoffs/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、框架解决的是组织问题

一次任务需要先规划、再查资料、再写报告、再检查引用。每一步可用模型或普通代码，**orchestration（编排）**负责安排顺序、分支、共享数据、失败处理与结束。

框架不是另一个大模型，也不会凭空提高事实准确率。选择依据是任务状态和控制需求，不是看演示中有几个“专家角色”。本课是 Learn 类型，没有原文 Build It 逐步大项目；仓库附带一个规则决策脚本，我们仍会解释其代码和使用。

## 二、四种抽象先用日常语言解释

**抽象**是框架让你用来描述问题的基本对象。

| 框架 | 主要理解入口 | 适合先考察的需求 |
|---|---|---|
| LangGraph | 状态、节点、边、检查点 | 明确分支、长期状态、暂停恢复与回放 |
| CrewAI | Agent角色、Task任务、Crew协作及Flow流程 | 职责清晰的协作任务，配合流程管理 |
| AutoGen | Agent、消息、Team及运行时 | 提案—评论等多轮交流与团队协作 |
| Agno | Agent、Team、Workflow和运行存储 | 工具型助手、团队及工作流应用 |

原文用“图、组织图、聊天、带工具的盒子”帮助记忆，这是入口，不是能力边界。实际框架已经相互覆盖很多功能。

## 三、五个必须问清楚的问题

### 3.1 谁决定下一步

**显式路由**用 Python 条件/图边决定，例如参数不完整就补充，验证通过才继续。便宜、可测、容易审计。

**LLM 路由**由模型根据上下文选择角色或工具，灵活但增加调用、耗时和不确定性。许多系统将两者组合：模型提出计划，代码限定可执行范围和结束条件。

### 3.2 要保存什么状态

聊天记录、结构化计划、进度、预算、已完成操作、待审批动作是不同类型状态。**Durable state** 指进程退出后仍能恢复的状态；只是把消息打印出来不算持久恢复。

数据库能保存聊天，不一定就保存了下一步节点、并行分支结果和审批状态。反过来，也不能说除了某个框架其他都不能存状态。应自己做“运行到一半→退出→重启→继续”的实验。

### 3.3 怎样暂停、重试、回放

明确动作前暂停，明确审批人和恢复值。回放不能重复发消息或重复扣款；所有框架都需要业务层幂等与授权。检查失败重试范围：重试一个工具还是整个流程？两者费用和副作用完全不同。

### 3.4 能看见哪些执行证据

日志、trace、节点/角色输入输出、token、费用、错误和检查点。框架支持观测接口不等于你的项目已经启用观测。需要实际验证能定位一次错误。

### 3.5 成本和集成是不是合理

把总成本拆成模型调用、路由调用、工具、框架调度和存储；多数任务中额外模型调用比框架的纯 Python 开销更显著。固定模型和任务才可公平比较。工具适配/MCP 支持只是连接方式，不保证同样的 schema、错误和授权语义。

## 四、原文中需要更新的绝对判断

原文表格声称 CrewAI 不具备持久状态、AutoGen 不能恢复、只有 LangGraph 能正式并行等，这些不适合作为当前能力定论：

- CrewAI 官方有 Flows、状态与持久化相关能力，不能只看 Crew 的早期聊天抽象。[官方 Flows](https://docs.crewai.com/en/concepts/flows)
- AutoGen AgentChat 提供 save_state/load_state 等状态接口；旧 ConversableAgent、较新 AgentChat 与 AG2 不应混当同一版本。[官方状态管理](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html)
- Agno 官方包含团队、工作流、持久执行/审批等能力，不能把它限定为只能做单 Agent 的存储盒子。[官方介绍](https://docs.agno.com/)

因此，本课保留原文“按问题形状思考”的方法，同时把独占功能、绝对最快/最便宜的判断改成需要测试的假设。框架选型文档应注明版本与实验日期。

## 五、仓库附带规则推荐器：它不是训练好的模型

`Problem` 用布尔字段描述 typed_state、roles、dialogue、fanout、resume、human_interrupt、session_memory 及调用次数。`Recommendation` 保存推荐名和理由。`recommend` 是一串按顺序执行的 if/return，前面命中的分支会遮蔽后面分支。

原规则大致是：简单少调用选 plain python；恢复/审批/并行优先 LangGraph；对话型选 AutoGen；角色型选 CrewAI；会话型单助手选 Agno；剩余结构化状态倾向 LangGraph。这是作者写入的偏好，不是比较四框架后学出的分类器。

### 核心代码：规则推荐器的核心

下面直接保留仓库的Problem/Recommendation与recommend核心，不包含交互界面和测试样本。它是有优先顺序的手写规则，输出只代表这套规则的偏好；能力过时及小任务分支问题见正文。

````python
# 本块中文说明：规则推荐器的核心
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Problem:
    """Shape descriptor for an agentic task."""

    has_typed_state: bool = False
    has_roles: bool = False
    has_dialogue: bool = False
    has_parallel_fanout: bool = False
    needs_resume: bool = False
    needs_human_interrupt: bool = False
    total_llm_calls: int = 1
    needs_session_memory: bool = False


@dataclass(frozen=True)
class Recommendation:
    framework: str
    reason: str


# 依次匹配手写规则，命中后立即返回；这是规则优先级，不是训练结果。
def recommend(p: Problem) -> Recommendation:
    # Smallest-first: if it's 2 or fewer calls, skip the framework entirely.
    if p.total_llm_calls <= 2 and not any(
        (p.has_roles, p.has_dialogue, p.needs_resume, p.has_parallel_fanout, p.needs_human_interrupt)
    ):
        return Recommendation(
            "plain python",
            "Two or fewer LLM calls with no state, roles, dialogue, fanout, "
            "or resume needs; a framework is pure overhead.",
        )

    # Durable state or human interrupts or time-travel -> LangGraph.
    if p.needs_resume or p.needs_human_interrupt or p.has_parallel_fanout:
        return Recommendation(
            "langgraph",
            "Typed state, checkpointer, interrupts, and Send fanout are only "
            "first-class in LangGraph.",
        )

    # Dialogue-shaped problem -> AutoGen.
    if p.has_dialogue and not p.has_typed_state:
        return Recommendation(
            "autogen",
            "Proposer-critic or teacher-student dialogue is AutoGen's native "
            "shape; GroupChat selects speakers without hand-wiring.",
        )

    # Role-driven pipeline -> CrewAI.
    if p.has_roles and not p.has_typed_state:
        return Recommendation(
            "crewai",
            "Specialist roles with a short sequential or hierarchical plan "
            "are cheapest to express in CrewAI.",
        )

    # Single agent + sessions -> Agno.
    if p.needs_session_memory and not p.has_roles and not p.has_dialogue:
        return Recommendation(
            "agno",
            "Single agent with tools and persistent session memory; Agno's "
            "storage drivers are built in.",
        )

    # Typed state but no other signals still points at LangGraph.
    if p.has_typed_state:
        return Recommendation(
            "langgraph",
            "Typed state is LangGraph's core abstraction; map your TypedDict "
            "onto a StateGraph.",
        )

    # Fallback.
    return Recommendation(
        "langgraph",
        "Default for multi-step agents with any uncertainty about future state "
        "or branching needs.",
    )


# Tests -----------------------------------------------------------------------

````


在本课目录运行 `python code/main.py` 执行附带测试；`python code/main.py --ask` 进入交互问答。只需标准库，不会安装或调用四种框架，也不会产生模型请求费用。

**核对：**原第一个“小任务无需框架”分支没有完整考虑 has_typed_state/needs_session_memory，少调用但有状态要求的输入可能被过早归到 plain python。它仍可能是合理实现选择，但理由“无状态需求”不准确。测试通过只能说明规则与预期标签一致，不能证明推荐客观最优。

## 六、用同一个任务公平比较

原课任务是查一个机构总部、写200词简介、带来源。将其抽象为共同契约：

```json
{
  "question":"查询指定机构总部并写简介",
  "evidence":[{"source_id":"S1","text":"经核实的资料片段"}],
  "draft":"",
  "citations":[],
  "status":"pending"
}
```

固定模型、可用工具、资料和长度规则，各实现输出相同结构。记录成功率、来源支持率、总模型调用、token、耗时、失败恢复、审批是否生效。代码行数能表示部分实现复杂度，但不能单独衡量可维护性。

**DAG** 是有向无环图，适合只向前的固定流程；ReAct 往往有循环，不是 DAG。**Fanout** 是分发多个并行子任务，**fan-in** 是聚合结果。**Supervisor/worker** 是管理者分派到工作者，不一定每个管理判断都要调用模型。

## 七、原文三项练习

1. 用 LangGraph 的 plan/search/write/cite 节点与 CrewAI 的研究/写作/编辑角色实现同一任务，比较实际token和代码量，标注真实搜索或模拟搜索。
2. 再用 AutoGen 与 Agno 实现同一契约，比较成本、重启恢复、写入前审批。不要只比较“都打印了一篇文章”。
3. 写 JSON 输入的推荐脚本，用至少六个案例验证，加入“调用少但需要恢复”“有角色又有强类型状态”“多个需求冲突”等案例，并解释优先级。

对应原文的 The Skill 是学习交付要求，而不是此聊天的新指令：能用一句话说明“为什么这个结构适合当前任务”，并提供测试证据。无需额外创建 skill 文件。

自测：一个任务只有两次模型调用是否永远不需框架？不是，还要看持久恢复、审批、并发和团队维护需求。反过来，多个角色也不意味着必须上多 Agent 框架；普通 Python 流程可能已经足够。


## 本次验证记录（2026-11-01）

原仓库标准库推荐规则7/7附带案例通过；没有实际运行四框架对比。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<16：Agent状态机——节点、边、检查点与暂停恢复.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
