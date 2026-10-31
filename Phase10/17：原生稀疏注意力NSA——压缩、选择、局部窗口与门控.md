# 17：原生稀疏注意力NSA——压缩、选择、局部窗口与门控

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/17-native-sparse-attention) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/17-native-sparse-attention/docs/en.md)。原题：Native Sparse Attention (DeepSeek NSA)。

## 零、初学者先读：长文章是否每次都必须细看所有位置

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

稀疏注意力试图把有限计算用在重要部分。可以类比看书时先看章节摘要、再翻相关段落、同时留意最近读的几页。真实算法通过计算决定这些选择，而不是人工知道答案。

### 0.2 本课术语先翻成白话

- compressed=压缩后的粗视图

- selected=选择的细粒度块

- window=最近窗口

- block=一组连续token

- gate=各分支贡献的控制系数

- top-k=按分数选前k项。

### 0.3 用小数字走一遍

8个历史token、每块4个，会得到2个粗摘要；选1块要读4个细token；窗口再读最近2个。按三个独立分支计，共2+4+2=8次key访问。重叠处不一定能省掉，因为分支可能分别计算，短序列也未必划算。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

沿nsa_step读：compress_mean→粗attention→top_k_blocks→细attention→sliding_window→gate混合。toy用均值作摘要，真实可学习压缩更复杂。训练时任何摘要也不能含当前query不该看的未来信息。

前置阅读：[第04课](<04：从零搭建Mini GPT——因果注意力、前向计算与真实训练.md>)；[第12课](<12：推理优化——KV缓存、连续批处理、分页与前缀复用.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 这次没选某块，能永久删掉它吗？

**参考答案：** 不一定，后面的query可能需要它。

**问题2：** 门权重一定加起来为1吗？

**参考答案：** 本toy三路独立sigmoid不要求，因此别强行按概率归一化。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

长上下文注意力可通过压缩全局视图、保留少数细粒度块和局部窗口减少每次读KV的工作。本课是三分支标准库模拟，不是训练NSA或实现Triton内核；论文加速/精度不由key计数自动得到，也不能把NSA与DSA/任一DeepSeek checkpoint混称。

## 二、核心概念与推导

### 2.1 三条视图
compressed把块压成summary，真实方法可学习并涉及窗口/步长，toy用mean；selected按粗分数选块，读其原KV；window看最近W。每支做自己的attention，再按gate组合，独立sigmoid gate不必和为1。不是一个softmax上简单把未选项清零。

### 2.2 梯度流向
top-k索引选择离散，不能说索引本身可导。compressed分支输出给它的参数训练信号，selected分支对选中KV做普通可微attention，联合训练不必对索引求导。stdlib列表并无autograd，所以本toy没验证训练可行性。

### 2.3 复杂度
对一个decode query读约ceil(N/l)+kb+W个key；处理全部N位置约N乘这个数量。固定l时仍有N²/l项，不是严格线性。还需计压缩、选块、gate和数据移动；理论key减少不等于GPU墙钟同比加速，GQA组复用/块对齐才使稀疏更易高效。

### 2.4 因果与内存
给query的K/V必须只含合法历史；若训练时压缩块含未来token，粗summary也会泄漏。不同query会选不同原块，通常不能因此删光未选历史KV；细粒度分支可能仍需保存完整KV，加压缩状态。稀疏读取/计算不是自动减少持久KV。

### 2.5 来源边界
原文70%～80%延迟、9倍kernel、95%召回与各后端状态是特定来源描述，不是本机/所有模型保证；“两年后都默认NSA”是预测，不当事实。NSA/MoBA/StreamingLLM各有设计，不能只按三视图硬归并。

### 2.9 补充算例与理解检查

N1024,l32,k4,W128，按整块计为32+128+128=288，相对1024约3.56倍key访问减少。selected与window即使重叠也属于不同分支计算，不能简单去重后当同一算子。N65,l32时粗块是ceil65/32=3，不是源码count_nsa的65//32=2。

### 原课概念部分的公式与结构示例

$$
\text{out} = g_{\text{cmp}} \cdot \text{out}_{\text{cmp}} + g_{\text{sel}} \cdot \text{out}_{\text{sel}} + g_{\text{win}} \cdot \text{out}_{\text{win}}
$$

## 三、操作前准备

全部核心标准库。输入非空、K/V等长且维数一致，l>0、k≥0、W>0；toy fallback selected空向量以len(q)创建，若value维不同会错，要用value_dim。query视作最后位置，K/V只给历史前缀，暂不做整序列训练。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 本课动手范围

原课先概述本课要实现的环节；以下逐段区分任务目标、可执行代码与未实现的训练功能。



### 4.2 块压缩

源码compress逐连续l块求均值，尾块不足l也平均。均值不是学习MLP，无位置信息/重叠stride，不能复制论文全部能力。

~~~python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
~~~

### 4.3 粗视图attention

query对summary keys求softmax，加权summary values；分数作为选择线索。不同query必须重新选，不能固定全局top块当NSA。



### 4.4 选择细块

按最高粗分数挑k块，再取原KV，多头GQA group、block对齐和论文具体选择规则在toy省略。k超过块数需截断，负k原切片有反直觉行为，先校验。



### 4.5 局部窗口

取最后min(W,N)，这里只适用于末位置query；训练各位置都用同一个末窗口会未来泄漏，需按各自prefix。



### 4.6 三支门控

仓库gate是线性+独立sigmoid，未训练时不会天然学出远块权重大；也不用强制sum1。分支输出value_dim应一致。



### 4.7 计数

count_nsa用N//l和固定kl/W，N不可整除或窗口/选择超过N时与真实分支计数不符，应用info的实数。省key不是省了全部KV内存/端到端时延。



## 五、把代码运行起来：补充入口与检查

原仓库核心实现（与正文compress命名不同的是compress_mean，作用相同），省去打印main：
~~~python
"""Native Sparse Attention (DeepSeek NSA) in stdlib Python.

Implements the three parallel branches from Yuan et al. 2025:
  - compressed branch: coarse-grained attention over block-averaged keys
  - selected branch: fine-grained attention over top-k uncompressed blocks
  - sliding-window branch: attention over the last W tokens

Combines them with a gate and prints the per-query key count for each branch
vs. full attention. Scales the key-count report to 64k and 128k contexts to
show the long-sequence savings NSA targets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


def dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def softmax(row: List[float]) -> List[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def attention(q: List[float], K: List[List[float]],
              V: List[List[float]]) -> tuple[List[float], List[float]]:
    """Returns (weights, output)."""
    d = len(q)
    scale = math.sqrt(d)
    scores = [dot(q, k) / scale for k in K]
    w = softmax(scores)
    d_v = len(V[0])
    out = [sum(w[j] * V[j][c] for j in range(len(V))) for c in range(d_v)]
    return w, out


def compress_mean(K: List[List[float]], l: int) -> List[List[float]]:
    """Collapse every l consecutive keys into their mean. Real NSA uses a
    learned MLP here — mean-pool is the pedagogical baseline."""
    n = len(K)
    d = len(K[0])
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[c] for row in block) / len(block) for c in range(d)]
        out.append(summary)
    return out


def top_k_blocks(scores: List[float], k: int) -> List[int]:
    indexed = sorted(range(len(scores)), key=lambda i: -scores[i])
    return sorted(indexed[:k])


def fine_grained_keys(K: List[List[float]], V: List[List[float]], l: int,
                      block_indices: List[int]) -> tuple[List[List[float]], List[List[float]]]:
    """Load the raw (uncompressed) tokens from the selected blocks."""
    k_out, v_out = [], []
    for b in block_indices:
        start, end = b * l, min((b + 1) * l, len(K))
        k_out.extend(K[start:end])
        v_out.extend(V[start:end])
    return k_out, v_out


def sliding_window(K: List[List[float]], V: List[List[float]],
                   W: int) -> tuple[List[List[float]], List[List[float]]]:
    n = len(K)
    start = max(0, n - W)
    return K[start:], V[start:]


def gate(q: List[float], Wg: List[List[float]]) -> List[float]:
    """Gate MLP: 1-layer linear + sigmoid, produces 3 branch weights."""
    logits = [dot(q, Wg[i]) for i in range(3)]
    return [1.0 / (1.0 + math.exp(-x)) for x in logits]


@dataclass
class NSAConfig:
    l: int
    k: int
    W: int


def nsa_step(q: List[float], K: List[List[float]], V: List[List[float]],
             Wg: List[List[float]], cfg: NSAConfig) -> tuple[List[float], dict]:
    K_cmp = compress_mean(K, cfg.l)
    V_cmp = compress_mean(V, cfg.l)
    cmp_w, cmp_out = attention(q, K_cmp, V_cmp)

    picks = top_k_blocks(cmp_w, cfg.k)
    K_sel, V_sel = fine_grained_keys(K, V, cfg.l, picks)
    if K_sel:
        _, sel_out = attention(q, K_sel, V_sel)
    else:
        sel_out = [0.0] * len(q)

    K_win, V_win = sliding_window(K, V, cfg.W)
    _, win_out = attention(q, K_win, V_win)

    g = gate(q, Wg)
    combined = [g[0] * cmp_out[i] + g[1] * sel_out[i] + g[2] * win_out[i]
                for i in range(len(cmp_out))]

    info = {
        "cmp_keys": len(K_cmp),
        "sel_keys": len(K_sel),
        "win_keys": len(K_win),
        "total_keys": len(K_cmp) + len(K_sel) + len(K_win),
        "full_keys": len(K),
        "selected_blocks": picks,
        "gates": g,
    }
    return combined, info


def synthesize_sequence(n: int, d: int, signal_blocks: List[int], l: int,
                        rng: random.Random) -> tuple[List[List[float]], List[List[float]], List[float]]:
    """Build K, V where `signal_blocks` carry a shared pattern and the query
    is aligned to that pattern. The rest is Gaussian noise."""
    pattern = [rng.gauss(0, 1) for _ in range(d)]
    norm = math.sqrt(sum(x * x for x in pattern))
    pattern = [x / norm for x in pattern]
    K = [[rng.gauss(0, 0.3) for _ in range(d)] for _ in range(n)]
    V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    for b in signal_blocks:
        start, end = b * l, min((b + 1) * l, n)
        for i in range(start, end):
            K[i] = list(pattern)
    q = list(pattern)
    return K, V, q


def count_full_attention(N: int) -> int:
    return N


def count_nsa(N: int, l: int, k: int, W: int) -> int:
    return (N // l) + (k * l) + W



~~~
补充小例：
~~~python
q = [1.0,0.0]
K = [[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0]]
V = [row[:] for row in K]
Wg = [[0.0,0.0]] * 3
out,info = nsa_step(q,K,V,Wg,NSAConfig(l=2,k=1,W=2))
assert info["cmp_keys"] == 2
assert info["sel_keys"] == 2 and info["win_keys"] == 2
assert info["gates"] == [.5,.5,.5]
assert len(out) == 2
print(info, out)
~~~
gate和为1.5是独立门的正常行为，不是错误。测试只验证结构/维度，不验证训练或长文召回。

## 六、Use It：使用现成工具

原main有1024toy、不同N/key计数和block/k扫描；可运行不需GPU。真实NSA需模型原生支持、正确因果压缩/选择和内核，并按具体版本核对；不要把dense checkpoint在推理时换toy稀疏后称等价。本次不部署任何服务。



## 七、失败模式、原文边界与工程注意

1. top-k索引没有梯度，native trainable不是每个离散动作可导。
2. 细块KV往往仍完整保留，不把每query读得少当缓存自动变小。
3. 训练压缩未来会泄漏；末窗口代码不是任意位置mask。
4. gate未训练不保证语义路由，value维不能默认等query维。
5. floor计数/固定kl/W对尾块不准确，需按实际索引统计。
6. N²/l项仍在，Python循环也不是GPU sparse kernel。
7. NSA与DSA不同，来源推测和后端支持要单独核实。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
明确压缩窗口/stride、细块大小、top-k、局部窗口、GQA组、gate与因果mask。分别估持久KV、每query访存/算量和内核；测领域长文召回、答案质量、TTFT/TPOT，不将288keys推成3.56倍整体加速或省内存。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 1024toy三组l/k/W，统计真实key与独立needle召回；95%是试验目标非保证。
2. 两层hidden32 compressor做真实监督回归，若目标均值可用MSE；没定义语言预测概率时不能直接报PPL差。
3. 训练gate需明确目标与优化器，检查远/近证据权重；随机初始化不必均匀且不保证能自动偏selected。
4. 70B/128k算KV，先给层数/存储政策；细分支可需全KV，没有普遍“某长度才与full相等”交叉点。
5. 解释粗分支同时提供路由与可微信号、selected索引不可导，勿声称反传穿过top-k。

## 十、术语与完成标准

compressed：粗压缩；selected：选块细粒度；window：局部；gate：分支系数；native trainable：训练中使用该模式；sparse read：稀疏读取；KV retention：历史持久保存。

完成检查：

- 能算1024例288keys。
- 能说明top-k梯度边界。
- 能区分稀疏读与缓存保存。
- 能验证独立gate不必和为1。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行4key小例：压缩2、选中2、窗口2，三gate均0.5；total_keys6大于full_keys4，说明小序列也可能不划算。未训练NSA或测GPU稀疏内核。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[16：差分注意力V1与V2——有符号权重、降噪与版本纠错]] · 下一课：[[18：多Token预测MTP——顺序模块、共享参数与辅助损失]]
