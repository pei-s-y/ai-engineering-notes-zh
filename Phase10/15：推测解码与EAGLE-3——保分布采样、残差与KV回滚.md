# 15：推测解码与EAGLE-3——保分布采样、残差与KV回滚

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/15-speculative-decoding-eagle3) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/15-speculative-decoding-eagle3/docs/en.md)。原题：Speculative Decoding and EAGLE-3。

## 零、初学者先读：让小模型先猜，为什么还能保持大模型的分布

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

推测解码让便宜的草稿先猜几个token，昂贵的目标模型再验证。它不只是猜对就用、猜错随便改；随机采样要用接受概率和残差补偿，才能仍像直接从目标模型采样。

### 0.2 本课术语先翻成白话

- 本课q=目标概率、p=草稿概率

- 接受率α=草稿通过验证的比例

- residual=被拒绝时用的补偿分布

- bonus=整段都通过后额外采出的一个token

- rollback=撤回未被接受的缓存。

### 0.3 用小数字走一遍

本课记号取q=[.5,.5]、p=[.8,.2]。两个类别的接受概率贡献分别min(.5,.8)=.5与min(.5,.2)=.2；剩余.3全补第二类，结果为[.5,.5]。关键不是逐次与目标给同一个答案，而是长期抽样比例相同。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先读residual(q,p)，确认减法是目标减草稿，再读spec_step接受前缀。遇拒绝应停在第一处，后续基于错误前缀的草稿和KV都不能继续当有效。此处KVBuffer是逻辑长度演示，不是GPU缓存。

前置阅读：[第04课](<04：从零搭建Mini GPT——因果注意力、前向计算与真实训练.md>)；[第12课](<12：推理优化——KV缓存、连续批处理、分页与前缀复用.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么q/p大于1也不能超过100%接受？

**参考答案：** 接受概率取min(1,q/p)。

**问题2：** 第25课公式为什么像反了？

**参考答案：** 它交换了p/q的命名，目标除草稿这一含义没变。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

draft便宜提议，target/verifier一次验证多位置，以增加每次大模型调用产出。关键是保持目标分布，而不是选一条最高概率路径。此课代码是无上下文固定分布的标准库模拟，不是训练EAGLE3，也未测真实GPU加速。

## 二、核心概念与推导

### 2.1 本课记号
这里p=draft、q=target，与另些课p/q命名相反。d从p采样，按a(d)=min(1,q(d)/p(d))接受；拒绝按normalize(max(q−p,0))补采。对token t，接受质量min(p(t),q(t))，拒绝补充(q(t)−p(t))+，合计q(t)，这是单步精确性的直接证明。

### 2.2 多位置与产出数
draft链条件分布pi、target一次并行给qi；从左往右至首拒绝，接受前缀+一纠正，若全接受再bonus。每次总产出含纠正/bonus，均值在独立恒α理想模型为Σi=0..N α^i；不是纯“接受draft数”，后者少1。真实各步相关，公式是简化。

speedup≈E[emit]/(1+Nc)，假设verify成本不随N、无额外KV/调度。α=1时E=N+1，α=0时E=1；增N收益饱和成本上升，需扫最佳N，不照抄源最优5～7或8～10。

### 2.3 EAGLE路线
vanilla用独立小模型；EAGLE早期在target特征层自回归；EAGLE2动态draft tree；EAGLE3转直接token预测、多层特征融合与training-time test，缓解训练/自回归分布差。依据[EAGLE3论文](https://arxiv.org/abs/2503.01840)，up to6.5×和batch64的1.38×是论文配置结果，不是默认所有服务或当前机器保证。

### 2.4 KV与统计
丢弃拒绝后所有未接受draft路径，修正token的真实KV通常需在后续forward补算，不能只改length便声称已经生成内容。tree mask只允许祖先路径；兄弟互看破坏条件分布。finite样本χ²只是诊断，理论需算法证明；95%门槛也可能偶然失败，不能硬assert每次pass。

### 2.9 补充算例与理解检查

p=[.8,.2]，q=[.5,.5]，α=Σmin=.7；接受概率mass=[.5,.2]，总拒绝.3，残差全落第二token，合计[.5,.5]。若拒绝直接从q采样，则结果[.65,.35]，不再是q。α=.9、N5总产出均值4.68559，而不是原文约4.1。

### 原课概念部分的公式与结构示例

$$
\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{N+1}}{1 - \alpha}
$$

## 三、操作前准备

accept/residual及下方核心全部标准库。概率必须同长度、非负、有限、归一，N为非负整数。原accept在p≤0直接True只是防除0，不是合法地从零概率抽到token。下方先运行已定义函数/类再做测试，无需下载70B或EAGLE权重。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 本课动手范围

原课先概述本课要实现的环节；以下逐段区分任务目标、可执行代码与未实现的训练功能。



### 4.2 接受规则

u均匀[0,1)，使用min(1,q/p)。不要按argmax匹配替代采样，temperature/top-k/p等处理后两模型分布也要与实际proposal/target一致。

~~~python
# 本课p是draft、q是target，别与别课记号倒置。
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
~~~

### 4.3 拒绝残差

逐项取正q−p再归一。q=p时拒绝概率为0，残差零可兜底q；若非法概率导致此情况应先校验。zip会静默截短，需长度检查。

~~~python
# 按正(q-p)归一，p=q时理论无拒绝。
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
~~~

### 4.4 完整一步

原正文仅描述spec_step，下一节嵌入仓库完整核心。q/p在各位置不随上下文变化，故可以传列表；真实LM必须对每个prefix返回各自pi/qi并复用target一次forward。



### 4.5 逻辑回滚

KVBuffer只保存length，无任何真实KV字节或写回rollback。len增加到发出token数是计数模型，不能证明修正token的物理KV已正确；真实engine需commit有效前缀并补算。



### 4.6 分布检查

原distribution_check仅取每轮第一个输出做公平统计。原chi_square把另一随机直采hist当固定expected，却用单样本临界值，尺度不对；应对n*q作单样本检查或正确两样本同质性检验。



### 4.7 产出与成本扫描

measure_alpha从p按真实rule测接受，expected函数算理想总产出。名字p_eagle3是对q小扰动，并非真实训练EAGLE；成本单位归一于一次verify，不是秒。



## 五、把代码运行起来：补充入口与检查

以下为原仓库code/main.py核心，包含完整无上下文spec_step、统计与成本函数，省去打印主程序：
~~~python
"""Speculative decoding (Leviathan 2023) with N-token drafts and KV rollback.

Implements the full production speculative-decoding loop:
  - draft N tokens from p (cheap)
  - verify N positions in one parallel q forward
  - rejection rule: accept with min(1, q(d)/p(d))
  - residual sampling on rejection: (q - p)_+ renormalized
  - bonus token on full acceptance
  - KV cache rollback bookkeeping

Stdlib only. Numbers match what Phase 7 · 16 proved mathematically and what
Phase 10 · 12 described operationally. Here we stitch both together.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


def sample(probs: List[float], rng: random.Random) -> int:
    u = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def residual(q: List[float], p: List[float]) -> List[float]:
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0.0:
        return list(q)
    return [r / s for r in raw]


def kl(q: List[float], p: List[float]) -> float:
    total = 0.0
    for qi, pi in zip(q, p):
        if qi > 0 and pi > 0:
            total += qi * math.log(qi / pi)
    return total


@dataclass
class KVBuffer:
    """Tracks logical cache length for verifier. Physical bytes are notional."""
    length: int = 0

    def extend(self, n: int) -> None:
        self.length += n

    def truncate_to(self, n: int) -> None:
        self.length = n


def spec_step(q: List[float], p: List[float], N: int, kv: KVBuffer,
              rng: random.Random) -> tuple[List[int], int]:
    """One speculative step: draft N tokens from p, verify with q.

    Returns (tokens_emitted, verifier_forwards_used). verifier_forwards_used
    is always 1 here — that is the point. tokens_emitted is between 1 and N+1.

    For pedagogical simplicity q and p are context-free distributions shared
    across positions. The math extends to position-dependent q_i, p_i without
    changing the loop.
    """
    prefix_len = kv.length
    drafts: List[int] = []
    p_probs: List[float] = []
    for _ in range(N):
        d = sample(p, rng)
        drafts.append(d)
        p_probs.append(p[d])

    emitted: List[int] = []
    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q[d]
        p_prob = p_probs[i]
        ratio = q_prob / p_prob if p_prob > 0 else float("inf")
        if u < min(1.0, ratio):
            emitted.append(d)
            kv.extend(1)
        else:
            correction = sample(residual(q, p), rng)
            emitted.append(correction)
            kv.truncate_to(prefix_len + len(emitted))
            return emitted, 1

    bonus = sample(q, rng)
    emitted.append(bonus)
    kv.extend(1)
    return emitted, 1


def direct_sample(q: List[float], n: int, rng: random.Random) -> List[int]:
    return [sample(q, rng) for _ in range(n)]


def distribution_check(q: List[float], p: List[float], n_steps: int,
                       rng: random.Random) -> tuple[List[int], List[int]]:
    """Check that the FIRST emitted token (the Leviathan-sampled one) is
    distributed as q. On accept that is the draft; on reject it is the
    residual correction. The bonus token that follows on full acceptance is
    also distributed as q but is a second draw and should not be mixed in
    here."""
    spec_counts = [0] * len(q)
    direct_counts = [0] * len(q)
    for _ in range(n_steps):
        kv = KVBuffer()
        tokens, _ = spec_step(q, p, N=1, kv=kv, rng=rng)
        spec_counts[tokens[0]] += 1
        direct_counts[sample(q, rng)] += 1
    return spec_counts, direct_counts


def chi_square(observed: List[int], expected: List[int]) -> float:
    total_obs = sum(observed)
    total_exp = sum(expected)
    if total_obs == 0 or total_exp == 0:
        return 0.0
    result = 0.0
    for o, e in zip(observed, expected):
        e_norm = e * total_obs / total_exp
        if e_norm > 0:
            result += (o - e_norm) ** 2 / e_norm
    return result


def measure_alpha(q: List[float], p: List[float], n_samples: int,
                  rng: random.Random) -> float:
    hits = 0
    for _ in range(n_samples):
        d = sample(p, rng)
        u = rng.random()
        q_prob = q[d]
        p_prob = p[d]
        if p_prob > 0 and u < min(1.0, q_prob / p_prob):
            hits += 1
    return hits / n_samples


def expected_tokens_per_verify(alpha: float, N: int) -> float:
    if alpha >= 1.0:
        return N + 1
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (N + 1)) / (1.0 - alpha)


def wall_time_per_token(alpha: float, N: int, c: float) -> float:
    """Draft cost is c per token relative to the verifier (cost 1.0).

    Each verifier call costs 1.0 plus N * c for the draft. Expected tokens
    emitted is (1 - alpha^(N+1)) / (1 - alpha).
    """
    return (1.0 + N * c) / expected_tokens_per_verify(alpha, N)


def perturb(q: List[float], amount: float, rng: random.Random) -> List[float]:
    p = [max(1e-6, qi + amount * rng.gauss(0, 1)) for qi in q]
    s = sum(p)
    return [pi / s for pi in p]



~~~
补充解析检查（无需大量随机样本）：
~~~python
p,q = [.8,.2], [.5,.5]
accepted_mass = [min(a,b) for a,b in zip(p,q)]
reject_prob = 1 - sum(accepted_mass)
res = residual(q,p)
recovered = [a + reject_prob*r for a,r in zip(accepted_mass,res)]
assert all(abs(a-b)<1e-12 for a,b in zip(recovered,q))
assert abs(expected_tokens_per_verify(.9,5)-4.68559) < 1e-9
kv = KVBuffer()
rng = random.Random(42)
for _ in range(100):
    tokens,_ = spec_step(q,p,3,kv,rng)
    assert 1 <= len(tokens) <= 4
print(recovered, kv.length)
~~~
若做χ²，用理论expected=[n*x for x in q]，或两组数据合并估期望；不要将“样本数大看起来接近”当证明。原kl在q>0且p=0时跳过项，真实KL应∞，使用前补校验。

## 六、Use It：使用现成工具

vLLM speculative-config命令需匹配具体版本、draft checkpoint及target，受权重授权/显存/设备数量限制。不是所有负载都默认启用EAGLE3；长prompt、高并发或draft贵可能无收益。不要在单4060直接尝试源70B命令；本次只算概率与模拟。

~~~bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
~~~

## 七、失败模式、原文边界与工程注意

1. p/q符号先统一，接收与残差方向不能写反。
2. 树里总选最高概率已接受路径会改变采样分布，要使用对应正确算法。
3. fullaccept有bonus，拒绝有correction，均值不是仅draft接受数。
4. 有限采样检查不保证每次χ²<临界，也不是完整多步LM证据。
5. KV logical length不证明physical缓存含正确token。
6. 固定分布小扰动不是真EAGLE训练与其alpha。
7. 理论不改分布≠任意实现/量化/截断也严格等价，验证数值与协议。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
指定draft/target版本、相同tokenizer和采样分布、N与成本，验证接受与残差/bonus/KV位置。区分数学证明、固定分布toy、真实engine质量和性能；按负载测接受长度、TTFT/TPOT、总TPS、显存与尾延迟，不预设3～6.5×。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 5万样本，用理论n*q单样本或正确两样本χ²检查，报告seed/自由度/预期计数，不把95%门槛当确定性单测。
2. α=.9,c=.04,N1～10，画理想产出/成本曲线；若测wall clock说明CPU toy不是GPU真实verify成本。
3. [2,2,2]树8叶，检查祖先mask与正确采样路径，不直接取最大概率；比较相同node预算。
4. 两序列分别全接受/第2位置拒绝，验证每序列物理有效前缀、逻辑长度及修正KV补算，不能承诺完全无浪费。
5. 用两句解释TTT缓解teacher-forced与自预测输入差，联系scheduled sampling；“缓解”不是理论消灭所有误差。

## 十、术语与完成标准

proposal/draft：提议分布；verifier：目标；residual：正差归一；α：接受率；bonus：全接受额外token；TTT：训练时模拟推理输入；KV rollback：撤销未接受路径。

完成检查：

- 能用两token证明目标分布恢复。
- 能区分emit与accepted drafts。
- 能跑无上下文完整循环。
- 能指出χ²与物理KV的边界。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行解析残差复原[0.5,0.5]、100轮长度与逻辑KV检查；另固定seed42、20000轮首token采样，经验TV到目标为0.003。该检验是固定合成分布，不证明任意真实模型实现正确。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 7 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[14：开源模型架构解析——RMSNorm、RoPE、GQA、MoE与配置预算]] · 下一课：[[16：差分注意力V1与V2——有符号权重、降噪与版本纠错]]
