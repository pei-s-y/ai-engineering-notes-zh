# 22：异步Hogwild推理——共享KV协作、模拟器与加速边界

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/22-async-hogwild-inference) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/22-async-hogwild-inference/docs/en.md)。原题：Async and Hogwild! Inference。

## 零、初学者先读：多个模型实例怎样边做边看彼此进展

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

多个实例可以独立解题再投票，也可以显式分工；Hogwild研究更底层的共享注意力缓存，让实例生成过程中能利用彼此进展。先把协作目标与真实并发机制分清。

### 0.2 本课术语先翻成白话

- worker=执行生成的实例

- shared cache=可共享的中间表示

- async=不同工作不必总同时抵达同一步

- coordination=判断别人做了什么、自己接着做什么

- wall time=从开始到结束真实经历的时间。

### 0.3 用小数字走一遍

总任务100单位，其中20必须串行、80可均分，4worker理想耗时20+80/4=40，加速100/40=2.5倍。若协调还多花20，总耗时60，只剩1.67倍。这解释了为何人多不必然快，不能把总算力直接当加速比。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

SharedCache.tokens只装类别标记，不是各层KV。decide_next_category直接规定去较空的类别，run_hogwild顺序调用这些规则；因此它演示共享信息如何帮助分工，不是运行真实LLM自主协作。

前置阅读：[第12课](<12：推理优化——KV缓存、连续批处理、分页与前缀复用.md>)；[第15课](<15：推测解码与EAGLE-3——保分布采样、残差与KV回滚.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 两worker每轮发两token，是否代表答案完成快两倍？

**参考答案：** 还要扣重复、噪声、协调并看正确性。

**问题2：** 普通Python列表是否已经实现真实共享GPU KV？

**参考答案：** 没有，后者还涉及布局、位置和同步。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

推测解码加速一条序列的生成；Hogwild探索多个LLM实例通过共享注意力状态动态协作。不要与只在应用层传消息的multi-agent混淆。本课只运行标准库顺序调度toy，不能证明真实模型涌现协作或GPU加速。

## 二、核心概念与推导

### 2.1 四种并行思路
voting让各模型独立解题再投票；sub-task由协调器先拆任务再合并；tree-of-thought显式展开/筛选推理分支；Hogwild让实例在生成中看到彼此部分进展并自适应协作。目标可以是质量，也可以是延迟，二者要分别评估。

### 2.2 共享KV不等于所有线程乱写一个线性序列
[原论文第3节](https://arxiv.org/html/2504.06261v1)组织公共prompt与各worker缓存块，针对每个worker安排块顺序；还有按完整步骤提交的interleaved布局、即时可见的contiguous与combined布局。因此课程“一份按到达顺序增长的token数组”只是toy类比，不能替代真实布局、可见性和位置处理。

### 2.3 RoPE的可利用性质
旋转保持点积，共同位移对应的相对位置不变。对连续缓存块统一平移时，可调整该块对应query/位置关系来复用缓存；不意味着任意重排token都正确，也不表示token内容的所有上下文依赖会重新计算。需要指定块布局和每个worker视角。

### 2.4 模型如何协作
用规则提示、示例、周期性冗余检查鼓励实例读他人进展、切换工作。论文是初步实验，不保证每次成功；toy则把“选较少类别”直接写进函数，完全没有学习出协作。没有微调不等于没有提示工程。

### 2.5 加速预算先统一c的单位
Amdahl基础：T_N=T_1[(1-p)+p/N]+C_total，speedup=T_1/T_N。c若是每worker一次总开销，并假设这些开销串行累加，可设C_total=Nc；若c是每步时间，则必须乘协调步数，不能混用。课程函数用Nc，却收了一个未使用的steps_per_worker参数。并行重叠的协调成本也不能无条件按N倍墙钟相加。

### 2.9 补充算例与理解检查

沿课程Nc toy约定：T1=10000,p=.7,c=200,N4，T4=4750+800=5550，约1.8018倍。固定p=.7，即使任务无限长，上限也只有1/(.3+.7/4)=2.1053倍，原文“长任务自然2.5—3倍”与该设定不一致。练习长任务50000,p.8,c500,N4→22000，约2.2727倍；短任务1000,p.3,c200,N4→1575，约0.6349倍。

### 原课概念部分的公式与结构示例



## 三、操作前准备

模拟器只用标准库，已有random/dataclasses/typing。不需要加载QwQ或DeepSeek。原课main没有真实多进程和锁；顺序for循环把每轮当抽象wall-step，先写worker的结果会立即被后写worker看到，存在固定顺序偏置。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 本课动手范围

原课先概述本课要实现的环节；以下逐段区分任务目标、可执行代码与未实现的训练功能。



### 4.2 共享缓存

SharedCache.tokens存(worker_id,category)，counts每次扫描全表，随长度增加会变慢。它不是各层各头KV张量。



### 4.3 worker循环

每轮遍历所有worker：读cache、选择、append；这是单进程顺序模拟，不可拿实际运行秒数当并行解码延迟。



### 4.4 协调启发式

概率coordination_weight触发选择A/B中计数较小者，否则偏好A，另有noise/coord概率。函数参数target_per_category实际上没用，类别不会达到目标后停止。



### 4.5 指标

work_tokens只数A/B，unique_progress在每轮对同类别只计一次；它衡量作者定义的toy进度，不验证答案。2worker每轮可发2token，raw work增多部分来自资源翻倍，应比较同等预算。



### 4.6 削弱协调

扫描权重0到1观察unique_progress；数值随seed和规则变化。原文要求必定低于1倍不成立，必须实测而非预写结果。



## 五、把代码运行起来：补充入口与检查

### 5.1 原仓库必要核心代码（内嵌）
Build It原文只列任务，实际函数来自code/main.py。去掉长打印入口，以下定义加后面的补充调用可直接在普通Python运行。

~~~python
"""Hogwild! Inference toy simulator — stdlib Python.

Two workers run concurrently against a shared token cache. Each worker reads
the cache and decides whether to add a work-token to category A or B, using
a simple coordination heuristic: if the other worker already produced enough
tokens in a category, switch.

Outputs:
  - total work-tokens produced in fixed step budget
  - wall-time speedup vs a single-worker baseline
  - a trace of which worker wrote which token and what category
  - a coordination-weight sweep showing the effect of poor coordination

Not a faithful LLM simulation. The point is to demonstrate emergent work
division driven by shared-cache reads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Literal


Category = Literal["A", "B", "noise", "coord"]


@dataclass
class SharedCache:
    tokens: List[tuple[int, Category]] = field(default_factory=list)

    def counts(self) -> dict:
        c = {"A": 0, "B": 0, "noise": 0, "coord": 0}
        for _, cat in self.tokens:
            c[cat] += 1
        return c


@dataclass
class Worker:
    id: int
    intended: Category
    coordination_weight: float
    rng: random.Random


def decide_next_category(worker: Worker, cache: SharedCache,
                         target_per_category: int) -> Category:
    """Read the shared cache. With probability coordination_weight, switch
    to the least-filled work category (noticing redundancy). Otherwise stay
    on the worker's intended category. coordination_weight = 0 models
    workers that cannot coordinate (full redundancy). weight = 1 models
    ideal reasoning-model coordination.
    """
    if worker.rng.random() < 0.05:
        return "noise"

    counts = cache.counts()
    base = worker.intended

    if worker.rng.random() < worker.coordination_weight:
        candidates = sorted(("A", "B"), key=lambda c: counts[c])
        return candidates[0]

    if worker.rng.random() < 0.1:
        return "coord"

    return base


def run_hogwild(n_workers: int, step_budget: int, target_per_category: int,
                coordination_weight: float, seed: int = 42) -> dict:
    """All workers default to category A. Coordination makes them diverge.
    Without coordination, redundant tokens (same category from multiple
    workers) are counted once. With coordination, workers pick different
    categories so each token is unique and contributes to total progress."""
    cache = SharedCache()
    workers = []
    for i in range(n_workers):
        workers.append(Worker(
            id=i, intended="A",
            coordination_weight=coordination_weight,
            rng=random.Random(seed + i),
        ))

    trace: List[tuple[int, Category, str]] = []
    step = 0
    progress = 0
    while step < step_budget:
        this_step_categories: List[tuple[int, Category]] = []
        for w in workers:
            cat = decide_next_category(w, cache, target_per_category)
            cache.tokens.append((w.id, cat))
            this_step_categories.append((w.id, cat))

        seen_work_categories = set()
        for w_id, cat in this_step_categories:
            tag = "redundant"
            if cat in ("A", "B") and cat not in seen_work_categories:
                seen_work_categories.add(cat)
                progress += 1
                tag = "unique"
            trace.append((w_id, cat, tag))
        step += 1

    counts = cache.counts()
    work_tokens = counts["A"] + counts["B"]
    return {
        "workers": n_workers,
        "step_budget": step_budget,
        "tokens_emitted": len(cache.tokens),
        "work_tokens": work_tokens,
        "unique_progress": progress,
        "category_counts": counts,
        "coord_tokens": counts["coord"],
        "noise_tokens": counts["noise"],
        "tokens_per_step": len(cache.tokens) / step_budget,
        "work_per_step": work_tokens / step_budget,
        "progress_per_step": progress / step_budget,
        "sample_trace": trace[:12],
    }


def expected_speedup(T_serial: int, p: float, c: int, N: int,
                     steps_per_worker: int) -> float:
    parallel = T_serial * ((1 - p) + p / N) + c * N
    return T_serial / parallel
~~~

~~~python
# 补充：比较同样200轮，但明确2worker使用了更多计算资源。
base = run_hogwild(1, 200, 100, 0.8)
for weight in (0.0, 0.1, 0.8, 1.0):
    row = run_hogwild(2, 200, 100, weight)
    print(weight, row['work_tokens'], row['unique_progress'],
          row['unique_progress'] / base['unique_progress'])
print('baseline:', base['unique_progress'])
print('long:', expected_speedup(50000, 0.8, 500, 4, 12500))
print('short:', expected_speedup(1000, 0.3, 200, 4, 250))
~~~

输出是类别计数、toy进度比、两条理论公式结果；不会生成真实自然语言，不会验证“模型自主分工”。本次实际数值见导航。若修改循环让所有worker先读同一snapshot再写入，结果也可能变化，应作为同步语义实验记录。

## 六、Use It：使用现成工具

先在真实业务测独立解题、投票、显式分工的基线，再决定是否值得研究共享KV。论文[参考实现](https://github.com/eqimp/hogwild_llm)是进一步实验入口，本次未安装或运行。原文1.3/1.5倍阈值只是选型建议，Hogwild与推测解码的收益不保证相乘，因为共享硬件、上下文变更和验证成本会相互影响。



## 七、失败模式、原文边界与工程注意

1. 课程“所有worker按到达次序写同一序列”忽略真实多块缓存布局。
2. 模拟器硬编码分工，不能称其为LLM涌现行为；unique_progress也不是正确性。
3. target_per_category和steps_per_worker两个参数未被使用，改变前者不会按目标截止。
4. 原文短任务/非推理模型/跨节点的判断适合作为待测假设，不是证明永远不适用；不要虚构当前生产普及程度。
5. 把两个工作器所有开销相加称“墙钟”会误导；报告资源数、总token、正确率和实测延迟。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
你是并行推理方案选择助手。输入任务长度、独立子问题比例、模型、GPU与延迟预算，比较独立投票、显式分工、树搜索、共享KV与推测解码。先统一协调成本单位，再给Amdahl上限和资源成本。列出缓存布局、位置处理、读写可见性、质量基线与停止条件。toy与真实LLM结果分别标识，不假设加速乘法，也不把工作token数当正确答案数。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **默认模拟。** 比较N1与N2同200轮的work_tokens和unique_progress，记录seed，解释每轮资源量差异；不把toy比值写成GPU速度。
2. **coordination_weight=.1。** 实测并记录比值，分析重复类别。可以出现仍高于1的结果，不能为迎合原题改输出。
3. **理论长短任务。** 按本课Nc约定得到2.2727与0.6349倍；同时说明若c改成每步开销应重写公式。
4. **论文观察与缓解。** 第4节描述长预算contiguous布局不易持续关注别人、短预算interleaved协作形成慢；冗余检查、简短状态总结/适当同步是可测试的改进，不保证解决。不要冒称论文列了固定“两类全部失败模式”。
5. **组合2-token推测。** 先把验证/回滚时间加进toy时间轴；同一前缀被他人扩展后，草稿基于哪个版本要明确。缓存版本、提交边界、接受分支与旧KV清理都是必要账目，不能只把两倍率相乘。

## 十、术语与完成标准

|术语|含义|
|---|---|
|Hogwild inference|共享注意力缓存支持的多实例并行生成研究|
|shared KV block|可跨实例复用的token表征块|
|emergent coordination|模型在提示下形成协作行为，区别硬编码规则|
|p|可并行工作比例|
|c|协调成本，必须说明每步/每worker及时间单位|
|Amdahl law|串行部分限制总体加速的公式|
|cache layout|每个worker看到的共享/私有块排列方式|

完成检查：

- [ ] 能区分同一序列推测与多实例协作。
- [ ] 能解释源码模拟器为何不证明自主协作。
- [ ] 能正确计算长短任务公式并检查c单位。
- [ ] 能列出真实KV并发比共享list多出的工程问题。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行seed42、200轮toy：N1进度189；N2协调权重0/.1/.8/1的进度197/219/341/385，比值约1.042328/1.158730/1.804233/2.037037。弱协调没有如原文断言跌破1；长短任务Nc公式分别2.272727/0.634921倍。没有真实并行LLM。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 6 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

原项目按 [MIT License](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/LICENSE) 发布；作者 Rohit Ghumare。新增的中文解释、注释及纠错为学习整理。保留许可声明：

~~~text
MIT License

Copyright (c) 2026 Rohit Ghumare

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[21：Jamba混合架构——SSM状态、注意力比例与长上下文预算]] · 下一课：[[25：推测解码与EAGLE——拒绝采样、分布验证与树形Mask]]
