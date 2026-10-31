# 18：多Token预测MTP——顺序模块、共享参数与辅助损失

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/18-multi-token-prediction) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/18-multi-token-prediction/docs/en.md)。原题：Multi-Token Prediction (MTP)。

## 零、初学者先读：为什么训练时还要预测更后面的token

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

普通下一token训练只对紧接着的一个目标给反馈。MTP增加更远位置的预测任务，让训练信号更丰富。它不意味着推理时可以直接跳过所有前面的条件关系。

### 0.2 本课术语先翻成白话

- depth=额外预测模块的第几层

- 辅助loss=主任务之外的训练目标

- shared head=不同模块共用输出映射

- concat=首尾拼接向量

- λ=辅助任务总权重。

### 0.3 用小数字走一遍

序列[A,B,C,D,E]，从位置A的表示出发，主头预测B；第1个顺序MTP模块再接收B的信息预测C；第2个接收C的信息预测D。训练时这些后续token是已知数据；真实生成时要用已有前缀或草稿，不能偷用未知答案。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先理解mtp_loss里的i、k与目标i+k+1，再看mtp_forward。两个长度h的向量concat后是2h长，投影矩阵要能接2h。原toy用加法代替拼接并用单tokenattention，正文已标明局限，不代表论文全部计算。

前置阅读：[第04课](<04：从零搭建Mini GPT——因果注意力、前向计算与真实训练.md>)；[第06课](<06：指令微调SFT——聊天格式、响应Mask与真实梯度.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 两层额外loss为2和4，λ=.3时平均加权多少？

**参考答案：** .3/2×(2+4)=.9，再按设计和主loss组合。

**问题2：** 共享输出头是否证明输入embedding也与输出头绑定？

**参考答案：** 不证明，它们是不同的共享关系。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

MTP在主下一token任务外增加未来多个位置的监督。DeepSeek顺序设计额外接入中间token，训练后可作draft；额外模块不是无成本。本课toy是随机前向/损失与参数统计，不会训练出MTP，也不保证降低噪声时loss单调下降。

## 二、核心概念与推导

### 2.1 位置与顺序
主hi0预测t(i+1)；第k模块接hi(k−1)和E(t(i+k))，预测t(i+k+1)。训练用真实中间token，推理用draft，需防把目标本身直接输入。两个向量分别RMSNorm后concat2h→Mk投影h→Tk→共享输出head。

Ltotal=Lmain+(λ/D)ΣLk；每depth仅有效未越界/非PAD/不跨文档目标参与，分母按定义统一。源码用所有depth共同有效位置，较浅depth会少用部分尾token，应说明非唯一做法。

### 2.2 共享怎么计
MTP和main可共享输入embedding与输出head；这不代表main embedding一定与main LM head绑权。原toy将二者也tied，是额外假设，真实DeepSeek config tie_word_embeddings=false。每模块concat投影2h²，attention/FFN/Norm按真实dense或MoE计，不能用dense14h²解释精确14B。

### 2.3 并行与顺序
并行多head从同backbone预测不同偏移，不是非法看未来。原文“并行MTP不能用于spec decoding”错误；[Gloeckle原论文](https://arxiv.org/abs/2404.19737)也报告多token预测带来的推理加速。顺序条件结构更便于某些draft设计，不证明所有任务都必更低loss。

### 2.4 训练收益与推理
辅助损失可影响backbone梯度，并不等于显式蒸馏模型。MTP作proposal仍要target验证、残差和KV回滚；argmax一致率不等于随机接受率。原文80%/1.8倍与λ调度是来源配置，不当toy复现，参数与额外算量都不是免费。

### 2.9 补充算例与理解检查

tokens=[a,b,c,d,e]，i0主预测b，depth1先看到b来预测c，depth2再看到c来预测d。若把c embedding直接交depth1再预测c就泄漏。假设Lmain=2、L1=3、L2=4、λ=.3，则辅助=.3/2×7=1.05，总3.05。

### 原课概念部分的公式与结构示例

$$
\begin{aligned}
h_i^{(0)} &= \text{main model backbone at position } i \\
h_i^{(k)} &= T_k\Big( M_k \cdot \mathop{\mathrm{concat}}\big(\mathop{\mathrm{RMSNorm}}(h_i^{(k-1)}), \mathop{\mathrm{RMSNorm}}(E(t_{i+k}))\big) \Big) \quad \text{for } k \ge 1
\end{aligned}
$$

$$
\text{logits}_{i+k+1} = \mathop{\mathrm{Out}}(h_i^{(k)}) \quad \text{for } k = 1, \ldots, D
$$

$$
L_k = \mathop{\mathrm{CE}}(\text{logits}_{i+k+1}, t_{i+k+1})
$$

$$
L_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_k
$$

## 三、操作前准备

标准库toy核心如下，序列需T>D+1，D≥1，token IDs合法。不同课都有matvec/softmax等同名函数，建议独立练习会话。原正文combine用+：Python list是拼接、NumPy数组是相加，注释不可替代类型检查。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 本课动手范围

原课先概述本课要实现的环节；以下逐段区分任务目标、可执行代码与未实现的训练功能。



### 4.2 共享embedding

只建立一份E，并让模块读取；检查对象与词表一致，不能另建同shape随机表冒充共享。main输入输出是否tied另算。



### 4.3 组合前深度状态与下一token

正文说“加法替身”但list +实际concat；仓库mtp_forward明确调用add逐元素相加，Mk也h×h，不是正确2h→h concat模块。下面保留toy，并给形状正确的补充combine。

~~~python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
~~~

### 4.4 模块block

仓库attention_single只有一个token，weight恒1，Q/K分数算后没参与输出，不能表示序列因果注意力/训练。SwiGLU可观察门控形状，但也无梯度。



### 4.5 共享输出头

toy用E行与hidden点积，共享main/MTP head同时将input/output tied；真实系统不必把这两种共享混同。



### 4.6 多depth loss

mtp_loss按i/k取未来token，累加CE再λ/D。T不足或D0会除零，需先断言；背后hidden是随机，不是已训练backbone，loss输出不代表模型能力。



### 4.7 参数计数

源码count_parameters用h²投影、普通MHA+dense，Norm也未全计；只适用其toy假设。真实concat应2h²，MoE全部专家影响总参数、active另算，不可把toy表叫DeepSeek精确值。



## 五、把代码运行起来：补充入口与检查

仓库核心实现保留如下，不含print main；其中加法投影与单token注意力均为已说明简化：
~~~python
"""DeepSeek-V3 Multi-Token Prediction (MTP) module — stdlib Python.

Implements:
  - shared embedding table (used by main model and every MTP module)
  - per-depth MTP module: projection + 1-block transformer + shared head
  - joint MTP loss across depths
  - parameter-count accounting (per module, shared, total)
  - a toy sequential evaluation that matches DeepSeek-V3's Section 2.2 equations

Pedagogical: single-head linear-projection attention, element-wise SwiGLU.
The goal is to show the structure of the MTP module, not to train a real LLM.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


def rand_matrix(rows: int, cols: int, rng: random.Random,
                scale: float = 0.1) -> List[List[float]]:
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def matvec(M: List[List[float]], v: List[float]) -> List[float]:
    out = [0.0] * len(M)
    for i, row in enumerate(M):
        out[i] = sum(row[j] * v[j] for j in range(len(v)))
    return out


def add(a: List[float], b: List[float]) -> List[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def rms_norm(v: List[float], eps: float = 1e-6) -> List[float]:
    ms = sum(x * x for x in v) / len(v)
    r = 1.0 / math.sqrt(ms + eps)
    return [x * r for x in v]


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def swiglu(v: List[float], W_gate: List[List[float]], W_up: List[List[float]],
           W_down: List[List[float]]) -> List[float]:
    gate = [silu(x) for x in matvec(W_gate, v)]
    up = matvec(W_up, v)
    inner = [g * u for g, u in zip(gate, up)]
    return matvec(W_down, inner)


def softmax(row: List[float]) -> List[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


@dataclass
class MTPModule:
    """A single depth-k MTP module."""
    hidden: int
    ff: int
    # Projection M_k: input is concat of 2 RMSNorm'd vectors of size h. We
    # approximate the concat as addition to keep the toy manageable while
    # preserving the projection structure.
    M_k: List[List[float]]
    # Transformer block: attention q/k/v/out + SwiGLU MLP
    Wq: List[List[float]]
    Wk: List[List[float]]
    Wv: List[List[float]]
    Wo: List[List[float]]
    W_gate: List[List[float]]
    W_up: List[List[float]]
    W_down: List[List[float]]


def make_mtp_module(hidden: int, ff: int, rng: random.Random) -> MTPModule:
    return MTPModule(
        hidden=hidden, ff=ff,
        M_k=rand_matrix(hidden, hidden, rng),
        Wq=rand_matrix(hidden, hidden, rng),
        Wk=rand_matrix(hidden, hidden, rng),
        Wv=rand_matrix(hidden, hidden, rng),
        Wo=rand_matrix(hidden, hidden, rng),
        W_gate=rand_matrix(ff, hidden, rng),
        W_up=rand_matrix(ff, hidden, rng),
        W_down=rand_matrix(hidden, ff, rng),
    )


def attention_single(v_in: List[float], Wq: List[List[float]], Wk: List[List[float]],
                     Wv: List[List[float]], Wo: List[List[float]]) -> List[float]:
    """One-token self-attention stand-in. For a full sequence you would
    attend over K_cache; here the toy uses a degenerate q=k=self to keep
    the structure visible. A full implementation is a drop-in replacement."""
    q = matvec(Wq, v_in)
    k = matvec(Wk, v_in)
    v = matvec(Wv, v_in)
    score = sum(q[i] * k[i] for i in range(len(q))) / math.sqrt(len(q))
    weight = 1.0
    attended = [weight * vi for vi in v]
    return matvec(Wo, attended)


def mtp_forward(prev_hidden: List[float], next_embed: List[float],
                module: MTPModule) -> List[float]:
    """Equation from DeepSeek-V3 Section 2.2:
        h^(k) = T_k( M_k * [RMSNorm(h^(k-1)); RMSNorm(E(t_{i+k}))] )
    We use addition as a toy stand-in for concat + linear."""
    a = rms_norm(prev_hidden)
    b = rms_norm(next_embed)
    folded = add(a, b)
    projected = matvec(module.M_k, folded)
    post_attn = add(projected, attention_single(projected, module.Wq, module.Wk,
                                                  module.Wv, module.Wo))
    post_mlp = add(post_attn, swiglu(rms_norm(post_attn), module.W_gate,
                                       module.W_up, module.W_down))
    return post_mlp


def shared_head_logits(hidden: List[float], E: List[List[float]]) -> List[float]:
    """Tied LM head: reuse the embedding table transposed. logits[v] = E_v . hidden."""
    return [sum(E[v][i] * hidden[i] for i in range(len(hidden)))
            for v in range(len(E))]


def cross_entropy(logits: List[float], target: int) -> float:
    probs = softmax(logits)
    return -math.log(max(probs[target], 1e-12))


def mtp_loss(backbone_hidden: List[List[float]], tokens: List[int],
             modules: List[MTPModule], E: List[List[float]],
             lam: float) -> tuple[float, List[float]]:
    """Compute joint MTP loss over D depths.

    backbone_hidden[i] is h_i^(0), the main-model output at position i.
    modules[k-1] is the depth-k MTP module.
    tokens[i] is t_i. The main model predicts t_{i+1}; module k predicts
    t_{i+k+1}, for each i such that i + D + 1 is in range.
    """
    D = len(modules)
    per_depth = [0.0] * D
    n_valid = 0
    for i in range(len(backbone_hidden) - D - 1):
        h_prev = backbone_hidden[i]
        for k in range(1, D + 1):
            next_embed = E[tokens[i + k]]
            h_prev = mtp_forward(h_prev, next_embed, modules[k - 1])
            logits = shared_head_logits(h_prev, E)
            per_depth[k - 1] += cross_entropy(logits, tokens[i + k + 1])
        n_valid += 1
    per_depth = [loss / n_valid for loss in per_depth]
    total = (lam / D) * sum(per_depth)
    return total, per_depth


@dataclass
class ParamReport:
    embedding: int
    head_shared: bool
    per_mtp: int
    main_attention_per_layer: int
    main_mlp_per_layer: int
    main_total: int
    mtp_total: int
    total: int


def count_parameters(vocab: int, hidden: int, ff: int, n_layers: int,
                     D: int) -> ParamReport:
    emb = vocab * hidden
    attn = 4 * hidden * hidden
    mlp = 3 * hidden * ff
    main = emb + n_layers * (attn + mlp) + hidden
    per_mtp = hidden * hidden + attn + mlp
    mtp_total = D * per_mtp
    return ParamReport(
        embedding=emb, head_shared=True,
        per_mtp=per_mtp,
        main_attention_per_layer=attn, main_mlp_per_layer=mlp,
        main_total=main, mtp_total=mtp_total, total=main + mtp_total,
    )



~~~
补充形状正确的concat投影：
~~~python
def combine_concat(prev_hidden, next_embed, matrix):
    if len(prev_hidden) != len(next_embed):
        raise ValueError("两分支hidden维数必须相同")
    joined = rms_norm(prev_hidden) + rms_norm(next_embed)  # list拼接，长度2h
    if not matrix or any(len(row) != len(joined) for row in matrix):
        raise ValueError("矩阵需为h×2h")
    return matvec(matrix, joined)

assert len(combine_concat([1.,0.],[0.,1.],[[1.,0.,0.,0.],[0.,0.,0.,1.]])) == 2
rng = random.Random(0)
E = rand_matrix(8,4,rng)
tokens = [0,1,2,3,4,5]
bh = [row[:] for row in E[:6]]
modules = [make_mtp_module(4,8,rng) for _ in range(2)]
loss,depth_losses = mtp_loss(bh,tokens,modules,E,lam=.3)
assert math.isfinite(loss) and len(depth_losses) == 2
assert abs(loss - .15*sum(depth_losses)) < 1e-12
print(loss,depth_losses)
~~~
前半只验证concat形状，后半仍运行原加法toy，不声称已替换全部模块或训练收敛。

## 六、Use It：使用现成工具

本课main打印随机toy的各depth loss、参数表与噪声扫描；没有optimizer，不能验证“越训越好”。实际MTP需要预训练/继续训练后的模块与服务支持，并非拿未训练头直接推理。是否改造既有模型、何种规模值得使用由实验决定，hidden4096不是硬性门槛。



## 七、失败模式、原文边界与工程注意

1. list +与向量加法不同，Mk形状必须匹配。
2. 原toy单tokenattention不能验证跨位置因果。
3. 主输入输出共享与MTP复用主head是两种共享。
4. 无反传不可能用噪声扫描证明训练单调收敛。
5. 并行MTP也可加速，不能断言不能spec。
6. acceptance与argmax-match分开，必须用目标验证算法。
7. T/D/越界/PAD/文档边界和训练用真token的偏移要单测。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
列D、λ、主与辅助目标偏移、有效mask、concat投影、模块attention/FFN与共享关系；按dense/MoE算total/active/激活开销。要求真实训练与独立任务对照，作draft时测接受规则和KV，不把随机toy或50%argmax命中叫论文复现。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 跑toy噪声扫描仅记录实际loss，不要求单调；若验证收敛，增加真实优化与固定任务/验证。
2. 70B dense h8192 D1用真实attention/FFN/2h²投影+Norm算开销，对比MoE14B不同口径。
3. D2输入前depth状态、预测i+3，检查全部有效索引，验证总Lmain+λ/DΣLk。
4. 并行/顺序同预算真训练，控制参数与teacher forcing，不能预设顺序必更低深层loss。
5. 训练后作draft，区分top1一致与随机接受/产出长度，50%toy命中不代表复现DeepSeek。

## 十、术语与完成标准

depth：未来预测层数；auxiliary loss：辅助监督；concat：特征拼接；shared head：复用主输出表；teacher forcing：真实中间token条件；active/total：计算/存储口径。

完成检查：

- 能手写i/k/目标索引。
- 能分清两种共享。
- 能验证concat矩阵h×2h。
- 能指出toy无训练/非序列attention。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行concat投影形状检查及原加法toy两层MTP：总loss0.6191600273642114，各depth约[2.03503813,2.09269538]，λ/D关系通过。没有autograd训练或推测服务。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[17：原生稀疏注意力NSA——压缩、选择、局部窗口与门控]] · 下一课：[[19：DualPipe流水并行——双向调度、通信重叠与气泡模型]]
