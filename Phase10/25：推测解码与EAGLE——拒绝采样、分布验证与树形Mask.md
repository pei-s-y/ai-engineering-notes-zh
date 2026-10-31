> 路径约定：本文的 `~/projects/ai-engineering-from-scratch` 是通用示例，不表示仓库已位于该处。执行命令前，请将它替换为实际克隆目录；以 `phases/` 开头的路径均相对于课程仓库根目录。

# 25：推测解码与EAGLE——拒绝采样、分布验证与树形Mask

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/25-speculative-decoding) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/25-speculative-decoding/docs/en.md)。原题：Speculative Decoding and EAGLE。

## 零、初学者先读：把第15课的数学变成可检查的实验

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

本课用合成小分布检验拒绝采样，再构建候选树的可见关系。先读本课符号：这里p是目标、q是草稿，与第15课反过来。名字可变，目标除草稿的接受含义不变。

### 0.2 本课术语先翻成白话

- TV=两分布差异的一种度量，等于各类别概率绝对差之和的一半

- 经验分布=按采样次数统计的频率

- tree mask=每个候选允许看哪些祖先

- 蒸馏=让小模型学习大模型的概率或输出。

### 0.3 用小数字走一遍

p=[.6,.4]、q=[.8,.2]，TV=(.2+.2)/2=.2，理论单步接受率1-TV=.8。抽100次不必恰好80次通过，就像公平硬币100次不一定50次正面；需要区分理论概率和一次实验频数。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

make_target造目标分布，make_draft造草稿，verify_distribution检查样本频率，build_tree只生成节点关系。树中兄弟节点是竞争分支，不允许互看；构好mask还没完成真实模型验证或EAGLE训练。expected_output连同草稿成本才能用于选K。

前置阅读：[第15课](<15：推测解码与EAGLE-3——保分布采样、残差与KV回滚.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** TV两个值相减是否就是两分布之间TV？

**参考答案：** 不是，距离之差不是点之间距离。

**问题2：** K越长是否越快？

**参考答案：** 更长草稿花时间，接受前缀又可能很短，须比较总耗时。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

本课和第15课有交叠，仍完整保留：用目标p、草稿q的记号重新实现拒绝采样，检验分布和每轮输出长度，并理解树形attention mask。与第15课p/q定义相反，读公式前先看这里的符号。原文概述承诺的真实GPT2蒸馏并未在main.py实现，仓库实际是合成概率分布实验。

## 二、核心概念与推导

### 2.1 一轮发生什么
给定相同历史prefix，草稿顺序提出K个token并记录每个条件q；目标对草稿前缀做带正确因果mask的teacher-forced验证，得到每位置p和最后bonus分布。从左到右接受，首个拒绝处从残差采样并停止；全通过再出bonus。需要正确对齐“预测token”的logits位置和缓存，不是随意把K个token并行各调用一次。

### 2.2 为什么接受率是min(1,p/q)
先x~q，再以min(1,p(x)/q(x))接受，贡献概率min(p(x),q(x))。拒绝总概率为1−Σmin(p,q)=TV(p,q)，拒绝后采样r=(p−q)_+/Σ(p−q)_+，补上max(p−q,0)，所以总概率等于p。草稿抽到的token应有q(x)>0；p=q时不会发生数学上的拒绝，无须除零。数值实现要处理极小概率和舍入。

### 2.3 同分布不等于同一随机种子的文字一模一样
随机数调用顺序不同，逐条输出可不同，长期目标分布相同。温度、top-p、top-k若使用，p/q都必须是各自实际采样时的归一化分布；不能用过滤前logits比值。greedy温度0是另一条argmax接受分支，不能在概率代码里除以0。

### 2.4 接受长度与耗时
若各深度接受事件用同一α近似，E[输出数]=1+α+...+α^K，α=1时为K+1；含拒绝纠正或bonus那1个。若每草稿步相对目标耗时c，理想speedup≈E/(1+Kc)，还应补验证变长和同步成本。仅最大化E会偏向更长K并逐渐饱和，不能推出有限最优K；要包含耗时。

### 2.5 EAGLE、树和Medusa
EAGLE利用目标特征帮助草稿预测；EAGLE2动态树，EAGLE3融合多层特征并用Training-Time Test降低草稿训练/推理落差。具体方法应按[原论文](https://arxiv.org/abs/2503.01840)理解，固定4.5token和4—5倍不是通用指标。Medusa在目标上加多预测头，是不同设计。树节点只能看公共prefix、祖先及自身；同胞不能互看。一次模型forward里仍有很多层和矩阵计算，树更大也会增加成本，原文“one matmul/没有额外verify成本”不成立。

### 2.6 高温不必然接受率崩溃
α=Σmin(p,q)=1−TV(p,q)，取决于两个实际分布的距离。若高温后两者都更接近均匀，接受率反而可能提高；若p=q，无论温度多高都全接受。高随机性本身不能推出α≈1/vocab，更不能预定加速一定消失。

### 2.9 补充算例与理解检查

p=[.6,.4]、q=[.8,.2]：A接受贡献.6，B接受贡献.2，拒绝总概率.2，残差只在B，所以B再得.2，最终仍[.6,.4]，α=.8。K4时E=1+.8+.64+.512+.4096=3.3616；若c=.1，理想speedup=3.3616/1.4≈2.4011，而不是3.36倍实际墙钟。

### 原课概念部分的公式与结构示例

~~~text
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
~~~

$$
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} \qquad K = \text{draft length}, \ \alpha \in [0, 1]
$$

~~~text
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
~~~

## 三、操作前准备

原docs的speculative_step是依赖draft_state_init/draft_step/target_forward_batched等未定义函数的片段，不可直接运行。第五节给实际main.py必要函数，需NumPy；本次不安装、不下载GPT2。K取正整数，分布一维、非负、和1，目标与草稿词表ID语义一致。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 原课拒绝采样片段

先存每步q分布，再对齐目标p；r<p/q利用r∈[0,1)隐含了min1。片段无prefix传参、无真实KV管理、温度0分支也没实现；用来读算法，不宣称完整推理引擎。注意返回列表包括纠正/bonus，docstring只写accepted并不准确。

~~~python
# 目标p、草稿q；返回含纠正或bonus的1..K+1个token。
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_dists = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_dists.append(probs)
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_dists[k][tok]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_dists[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
~~~

## 五、把代码运行起来：补充入口与检查

### 5.1 仓库实际核心代码
以下来自本课code/main.py，包含合成target/draft、拒绝采样、分布对比、长度统计和树mask。为不让你寻找辅助定义，必要函数一起内嵌；没有模型蒸馏。

~~~python
"""Speculative decoding harness: exact rejection rule, alpha sweep, tree mask.

Three things this file proves, on synthetic toy distributions so the math
stays visible:

1. The Leviathan-Kalai-Matias rejection rule preserves the target's
   sampling distribution. Empirical total-variation distance between
   plain target sampling and speculative-with-draft sampling is < 0.01
   over 50_000 draws.
2. The expected-tokens-per-verify formula holds. For acceptance rate
   alpha and draft length K, E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)
   matches the measured throughput within sampling noise.
3. Tree drafting verifies multiple candidate paths in a single target
   forward via a topological causal mask. We build a depth-K tree, emit
   the verification mask, and confirm every node attends only to its
   ancestors.

Stdlib + numpy only.

Run:
    python main.py
    python main.py --vocab 64 --alpha 0.75 --k 4 --samples 50000
"""

from __future__ import annotations

import argparse
import numpy as np


def make_target(vocab: int, rng: np.random.Generator) -> np.ndarray:
    logits = rng.standard_normal(vocab) * 1.4
    e = np.exp(logits - logits.max())
    return e / e.sum()


def make_draft(target: np.ndarray, alpha_hint: float,
               rng: np.random.Generator) -> np.ndarray:
    """A draft distribution whose expected token-level acceptance is near
    alpha_hint. We linearly blend target with a uniform distribution; the
    blend ratio controls how close the draft is to the target."""
    vocab = target.size
    uniform = np.full(vocab, 1.0 / vocab)
    draft = alpha_hint * target + (1.0 - alpha_hint) * uniform
    noise = rng.uniform(0.95, 1.05, size=vocab)
    draft = draft * noise
    return draft / draft.sum()


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(probs.size, p=probs))


def speculative_step(target: np.ndarray, draft: np.ndarray, K: int,
                     rng: np.random.Generator) -> list[int]:
    """One round. Returns 1..K+1 tokens whose distribution equals target."""
    proposed: list[int] = []
    q_at: list[float] = []
    for _ in range(K):
        t = sample(draft, rng)
        proposed.append(t)
        q_at.append(float(draft[t]))

    accepted: list[int] = []
    for k, tok in enumerate(proposed):
        ratio = float(target[tok]) / max(q_at[k], 1e-12)
        if rng.random() < min(1.0, ratio):
            accepted.append(tok)
        else:
            residual = np.maximum(target - draft, 0.0)
            s = residual.sum()
            if s == 0.0:
                accepted.append(sample(target, rng))
            else:
                accepted.append(sample(residual / s, rng))
            return accepted
    accepted.append(sample(target, rng))
    return accepted


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def empirical_dist(samples: list[int], vocab: int) -> np.ndarray:
    counts = np.bincount(samples, minlength=vocab).astype(np.float64)
    return counts / counts.sum()


def verify_distribution(target: np.ndarray, draft: np.ndarray, K: int,
                        n_samples: int, rng: np.random.Generator
                        ) -> tuple[float, float]:
    """Compare next-token distributions under plain target sampling and
    speculative sampling. They must be statistically indistinguishable."""
    vocab = target.size
    plain = [sample(target, rng) for _ in range(n_samples)]
    spec_first: list[int] = []
    while len(spec_first) < n_samples:
        toks = speculative_step(target, draft, K, rng)
        spec_first.append(toks[0])
    p_plain = empirical_dist(plain, vocab)
    p_spec = empirical_dist(spec_first, vocab)
    return total_variation(p_plain, target), total_variation(p_spec, target)


def measure_alpha(target: np.ndarray, draft: np.ndarray,
                  n_samples: int, rng: np.random.Generator) -> float:
    accepted = 0
    for _ in range(n_samples):
        t = sample(draft, rng)
        ratio = float(target[t]) / max(float(draft[t]), 1e-12)
        if rng.random() < min(1.0, ratio):
            accepted += 1
    return accepted / n_samples


def expected_tokens(alpha: float, K: int) -> float:
    if alpha >= 1.0:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


def measure_throughput(target: np.ndarray, draft: np.ndarray, K: int,
                       n_rounds: int, rng: np.random.Generator) -> float:
    total = 0
    for _ in range(n_rounds):
        total += len(speculative_step(target, draft, K, rng))
    return total / n_rounds


def build_tree(branch_factor: tuple[int, ...]) -> list[tuple[int, list[int]]]:
    """Return nodes as (parent_index, depth-path). Index 0 is root."""
    tree: list[tuple[int, list[int]]] = [(-1, [])]
    frontier = [0]
    for depth, b in enumerate(branch_factor):
        next_frontier: list[int] = []
        for parent in frontier:
            for _ in range(b):
                tree.append((parent, tree[parent][1] + [len(tree)]))
                next_frontier.append(len(tree) - 1)
        frontier = next_frontier
    return tree


def tree_attention_mask(tree: list[tuple[int, list[int]]]) -> np.ndarray:
    """N x N causal mask where each row attends to its ancestors only."""
    n = len(tree)
    mask = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        cur = i
        while cur != -1:
            mask[i, cur] = 1
            cur = tree[cur][0]
    return mask


def validate_tree_mask(mask: np.ndarray,
                       tree: list[tuple[int, list[int]]]) -> bool:
    n = len(tree)
    for i in range(n):
        cur = i
        ancestors = set()
        while cur != -1:
            ancestors.add(cur)
            cur = tree[cur][0]
        attends = {j for j in range(n) if mask[i, j] == 1}
        if attends != ancestors:
            return False
    return True
~~~

~~~python
# 补充入口：先小样本检查通路，再按需扩大样本；不下载任何模型。
rng = np.random.default_rng(0)
p = make_target(8, rng)
q = make_draft(p, 0.75, rng)
print('两条到真目标的TV:', verify_distribution(p, q, 4, 10000, rng))
alpha_exact = float(np.minimum(p, q).sum())
print('精确alpha:', alpha_exact)
print('理论每轮输出:', expected_tokens(alpha_exact, 4))
print('实测每轮输出:', measure_throughput(p, q, 4, 3000, rng))
tree = build_tree((3, 2, 2))
mask = tree_attention_mask(tree)
print('树节点/形状/祖先校验:', len(tree), mask.shape, validate_tree_mask(mask, tree))
# 这里1代表允许访问；给需要加性mask的后端应转成允许0、禁止-inf。
additive_mask = np.where(mask == 1, 0.0, -np.inf)
~~~

预期树有1+3+6+12=22节点，mask为22×22，校验True。TV和平均长度有抽样波动，固定阈值.01不是每次必过；本次NumPy运行情况见导航。若在你仓库运行原入口，可用：

~~~bash
cd ~/projects/ai-engineering-from-scratch
python phases/10-llms-from-scratch/25-speculative-decoding/code/main.py --vocab 32 --alpha 0.75 --k 4 --samples 20000 --seed 0
~~~

原函数verify_distribution实际分别算TV(plain,target)、TV(spec,target)，不是直接TV(plain,spec)，main的delta TV也不等于后者。只检查每轮第一个token，尚不足以验证依赖历史的整条自回归分布；本合成p/q固定不随历史变化。

### 5.2 补充：标准库算有限成本下的K

~~~python
def expected_output(alpha, k):
    # 直接求和避免alpha接近1时两小量相除。
    return sum(alpha**i for i in range(k + 1))

for alpha in (0.5, 0.7, 0.9):
    scores = [(expected_output(alpha, k) / (1 + 0.1 * k), k) for k in range(1, 33)]
    speed, k = max(scores)
    print(alpha, 'toy最佳K=', k, '理想倍率=', round(speed, 4))
~~~

c=.1是人为预算，不是你4060实测。真实调K需记录目标verify长度相关延迟、草稿耗时、接受前缀分布、TTFT和TPOT。

## 六、Use It：使用现成工具

原文列vLLM/SGLang/TensorRT-LLM支持与示例草稿模型，都是需要版本与目标配对核对的工程入口。本课不新增未经当前配置校验的服务命令；核对模型类型、词表/chat模板、草稿权重训练对应关系和该版本method配置，不能因为“都是Llama/Qwen”就认定兼容。先用短请求验证输出分布与缓存，再测同等负载基线。



## 七、失败模式、原文边界与工程注意

1. make_draft的alpha_hint是与均匀分布混合的系数，还叠了noise，绝不等于精确接受率；用sum(min(p,q))计算。
2. 原source源码分母max(q,1e-12)只是防护，对低于阈值的正概率会改变算法，应注明精确性需真实比值和稳定数值。
3. 原docs列“树drafter+真实GPT2蒸馏”，实际main只有随机分布和树索引/mask，没有模型或训练。
4. 只画出正确mask不等于实现EAGLE，也没实现多候选树的保分布分支选择；不能简单选任意最长分支就宣称同分布。
5. 10k采样的TV<.01受词表、概率、随机性影响；应和普通采样噪声一起比较，必要时扩大样本或作正式检验。
6. 树mask通常要含自身及公共prefix；输入给模型的mask极性需按接口转换，不能把1/0直接当加性logit偏置。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
你是推测推理调优助手。给定目标/草稿、词表模板、采样参数、负载和硬件，先验证p/q记号、真实过滤后的概率、残差采样与缓存回滚。测接受前缀长度、目标verify与草稿耗时、同任务质量和TTFT/TPOT。对K、树宽、温度做预算内扫描，保留plain decode基线。分别报告数学分布保证的前提、合成实验、真实模型实验，不承诺固定倍数或固定alpha。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **拒绝规则与TV。** 先用两类手算例，再采样10000次，与真p及plain样本都比较；扩大样本观察噪声下降。原题<.01作为观察目标，不当必定断言；扩展到依赖prefix的模型才验证自回归情况。
2. **扫描α=.5/.7/.9与K。** 先画E的饱和曲线，再在给定c下求E/(1+Kc)有限最优值；原题只最大化E没有有限最优K。
3. **124M→30M蒸馏。** 真实可选扩展：定义draft架构、统一词表，教师无梯度，最小化KL(p_target||q_draft)，正确mask，留出验证。100M tokens不是课堂toy，先估显存/时长；α=.6—.7是原课预期，不能保证。
4. **top3树。** 明确每层节点数3^d、父索引与position，校验每行只看祖先/自身/公共prefix。补真实target验证及正确分支采样后才谈EAGLE效果。
5. **温度1.5。** 重算两分布并测α、耗时，允许上升或下降；p=q构造反例说明“高温必崩”不成立。失败时记录草稿开销是否超过省下的目标调用。

## 十、术语与完成标准

|术语|含义|
|---|---|
|p/q|本课目标/草稿分布，区别第15课符号|
|α|单步接受概率=1−TV(p,q)|
|residual|拒绝后的归一化正差(p−q)+|
|bonus|整段草稿接受后从目标多采的一个token|
|tree attention mask|屏蔽同胞和未来的树因果访问规则|
|distillation|用教师概率监督小模型|
|TTFT/TPOT|首token延迟/输出每token时间|

完成检查：

- [ ] 能手推两类例最终分布仍为p。
- [ ] 能区分α_hint、实测α、精确Σmin值。
- [ ] 能构建22节点树并检查mask极性。
- [ ] 能解释为什么E输出数不等于墙钟倍率。
- [ ] 能看出原docs未定义函数的片段不能直接运行。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

仅执行补充标准库成本扫描：c=.1、K1…32，α=.5/.7/.9时最佳K分别2/4/10，对应理想倍率约1.458333/1.980786/3.430947。NumPy概率分布、22节点树mask及真实蒸馏均未执行；第15课另有标准库拒绝采样实测。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 1 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[22：异步Hogwild推理——共享KV协作、模拟器与加速边界]] · 下一课：[[34：梯度检查点——激活重计算、分段反传与显存测量]]
