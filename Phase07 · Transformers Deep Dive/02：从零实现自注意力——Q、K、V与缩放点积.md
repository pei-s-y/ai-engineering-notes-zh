# 02：从零实现自注意力——Q、K、V与缩放点积

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F02-self-attention-from-scratch) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md)。原题：Self-Attention from Scratch。

## 零、初学者先读：注意力就是一次“软查询”

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第01课](<01：为什么是Transformer——RNN的三个瓶颈.md>)讲了为什么要把 RNN 换掉；本课把换来的是什么从零写出来——这是整个第七阶段最核心的一课，后面的多头、位置编码、总装全都踩在这块地基上。

### 0.1 先理解这一课在做什么

想象你在图书馆查资料：你拿着一个**问题**（query）去比对每本书封面上的**标签**（key），标签越对口，你就越多地引用那本书的**内容**（value）。传统数据库是硬匹配——只有一个精确命中；注意力是软匹配——所有书都按“对口程度”被引用一部分。本课要用 NumPy 把这套机制从零写出来：每个词生成三个向量（问什么、像什么、给什么），两两点积打分，softmax 变权重，加权求和出结果。写完你会发现整个“自注意力”不到 30 行代码，但它就是 GPT 和 BERT 的心脏。

### 0.2 本课术语先翻成白话

- 自注意力（self-attention）=序列自己看自己：每个词同时向所有词“提问”并取回混合信息
- 查询 Q（query）=每个词发出的“我在找什么”的向量
- 键 K（key）=每个词挂在身上的“我这里有什么”的向量，用来被别人的 Q 匹配
- 值 V（value）=一旦被选中，实际交出去的信息向量
- 点积（dot product）=两个向量对应位置相乘再求和，衡量“方向有多一致”
- 投影矩阵（projection matrix）=把词嵌入变成 Q/K/V 的可学习矩阵 $W_q, W_k, W_v$
- 注意力分数（attention score）=一个 Q 和一个 K 的点积，还没归一化的原始打分
- 缩放（scaling）=分数除以 $\sqrt{d_k}$，防止维度大时点积数值过大
- softmax=把一行任意大小的数变成一行的概率（全正、和为 1）
- 注意力权重（attention weights）=softmax 之后的分数，一行是一个词对全序列的“注意力分配”
- 上下文向量（context vector）=该词的最终输出：所有词 value 的加权和
- 因果掩码（causal mask）=把“未来位置”的分数设为负无穷，让词看不见后面的词（GPT 式）
- 维度记号：n=词数，d=嵌入维数，$d_k$=Q/K 维数，$d_v$=V 维数

### 0.3 用小数字走一遍

2 个词、每个词 2 维的迷你注意力。词 A=“猫”、词 B=“追”，嵌入取 $X = \begin{bmatrix}1 & 0\\1 & 1\end{bmatrix}$（第 1 行是 A，第 2 行是 B）。为了让手算可行，取三个投影矩阵都是单位矩阵 $W_q = W_k = W_v = I$（真实模型里它们是学出来的，这里只是让数字干净）。于是 Q=K=V=X。

**第 1 步：打分 $QK^\top$。** 每个词的 Q 去点积每个词的 K：

- A 问 A：$[1,0]\cdot[1,0] = 1$；A 问 B：$[1,0]\cdot[1,1] = 1$
- B 问 A：$[1,1]\cdot[1,0] = 1$；B 问 B：$[1,1]\cdot[1,1] = 2$

原始分数矩阵 $=\begin{bmatrix}1 & 1\\ 1 & 2\end{bmatrix}$。

**第 2 步：缩放 $\div\sqrt{d_k}$。** $d_k = 2$，$\sqrt{2} \approx 1.414$：

$$\begin{bmatrix}0.71 & 0.71\\ 0.71 & 1.41\end{bmatrix}$$

**第 3 步：逐行 softmax。** 需要 $e^{0.71} \approx 2.03$、$e^{1.41} \approx 4.11$：

- A 行：$[2.03, 2.03]$，归一化 → $[0.50, 0.50]$
- B 行：$[2.03, 4.11]$，和为 6.14，归一化 → $[0.33, 0.67]$

**第 4 步：加权求和（乘 V）。** 注意力输出 = 权重 × 各词的 value：

- A 的输出：$0.50 \times [1,0] + 0.50 \times [1,1] = [1.00,\ 0.50]$
- B 的输出：$0.33 \times [1,0] + 0.67 \times [1,1] = [1.00,\ 0.67]$

读一遍结果：A（猫）对两个词平分注意力，输出是两个词的“平均味”；B（追）更关注自己（权重 0.67），输出里自己的成分更重。这就是自注意力的全部：**打分 → 缩放 → 归一 → 混合**。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

原课代码按五步递进，正好对应 0.3 的四步外加可视化：先 `softmax`（数值稳定版），再 `scaled_dot_product_attention`（0.3 的第 1—4 步打包），然后 `SelfAttention` 类（把三个投影矩阵装进去），第 4 步在一个 6 词句子上跑，第 5 步把权重画成 ASCII 热图。读代码时把每个函数对回 0.3 的对应步骤，就不会迷路。核心数据结构全是 NumPy 数组：`X`（n×d）、`Q/K/V`（n×dk 或 n×dv）、`weights`（n×n）。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么要除以 $\sqrt{d_k}$，不除会怎样？

**参考答案：** 点积是 $d_k$ 项乘积之和，维度越大数值越容易大。$d_k=64$ 时点积可能到几十，softmax 里 $e^{几十}$ 会把差距指数放大，一行里几乎只剩一个 1、其余全 0——softmax 饱和，梯度趋近 0，训练动不了。除以 $\sqrt{d_k}$ 把数值拉回“softmax 还有梯度”的区间。

**问题2：** “自”注意力与一般注意力差在哪？

**参考答案：** Q、K、V 三个向量是否来自**同一个序列**。自注意力三者都从自己这串词投影出来（本课）；交叉注意力（cross-attention）里 Q 来自一个序列（如译文的词），K/V 来自另一个序列（如原文的词），用在翻译解码器里。

**问题3：** 注意力权重矩阵每一行的和是多少？为什么是行不是列？

**参考答案：** 每行和为 1，因为 softmax 按行归一。行代表“提问者”（谁的 Q），列代表“被看者”（谁的 K）：第 i 行是第 i 个词对全序列的注意力分配。“每行一个分布”保证了每个词拿到的上下文是 value 的一种加权平均。

**问题4：** 0.3 的例子里，把“猫”的嵌入从 $[1,0]$ 换成 $[2,0]$，B 对自己的权重会怎么变？

**参考答案：** 会变小。B 问 B 的分数仍是 2、B 问 A 的分数变为 2，缩放后同为 1.41，softmax 拉平成 $[0.5, 0.5]$。A 的“存在感”变强，B 分给它的注意力就变多——注意力权重是相对的、随内容变化的。

能讲清这四个问题后，进入正文。

---

## 一、这一课解决什么问题？

RNN 一个 token 一个 token 地处理序列：走到第 50 个词时，第 1 个词的信息已经被 50 次压缩挤过，长程依赖被压碎在一个固定尺寸的隐状态里——这是 LSTM 门控也解决不了的瓶颈（[第01课](<01：为什么是Transformer——RNN的三个瓶颈.md>)的三个瓶颈之二）。

2014 年 Bahdanau 注意力论文给出修法：让解码器回头看编码器的每个位置，自己决定当前步该关注谁。但它仍是装在 RNN 上的补丁。2017 年《Attention Is All You Need》问得更狠：如果注意力是**唯一**的机制呢？不要循环、不要卷积，只有注意力。自注意力让序列中每个位置在**一次并行步骤**里关注所有其他位置——这正是 Transformer 快、可缩放、统治一切的原因。本课就用 NumPy 把它写出来。

## 二、核心概念与推导

### 2.1 数据库查找类比：Q、K、V 的分工

~~~text
传统数据库：
  查询: “法国的首都”  →  精确匹配   →  “巴黎”

注意力：
  查询: “法国的首都”  →  与所有键比相似度  →  所有值的加权混合
~~~

每个 token 生成三个向量：**Query（Q）**：我在找什么？**Key（K）**：我包含什么？**Value（V）**：被选中时我提供什么信息？Q 与所有 K 的点积产生注意力分数，分数高表示“这个键和我的查询对口”，分数加权混合 V 得到输出。

### 2.2 Q、K、V 的计算：三次矩阵乘法

设输入嵌入 $X$ 形状为 $(n, d)$（n 个 token、每个 d 维），三个可学习矩阵：

~~~text
Wq 形状 (d, dk)      Wk 形状 (d, dk)      Wv 形状 (d, dv)

Q = X @ Wq  → (n, dk)   每个词的查询
K = X @ Wk  → (n, dk)   每个词的键
V = X @ Wv  → (n, dv)   每个词的值
~~~

一个词的视角（原课示意图的文字版）：同一个嵌入 $x_i$ 分别过三个矩阵，得到 $q_i$（我要找什么）、$k_i$（我像什么）、$v_i$（我能提供什么）。注意 $d_k$ 和 $d_v$ 可以不同——Q/K 负责打分，V 负责内容，两者维度各自独立。

### 2.3 注意力矩阵：n×n 的“谁看谁”

分数矩阵 $\text{Scores} = QK^\top$，形状 $(n, n)$：第 $(i,j)$ 格是“词 i 的查询 × 词 j 的键”。原课给了一个 5 词句子的示意分数矩阵（原课示例数字）：

~~~text
              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   ← q1 对每个键的打分
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+
~~~

每一行是一个词对全序列的打分。原课接着演示 q1 行 softmax 后约为 `[0.52, 0.09, 0.07, 0.14, 0.08]`（原课参考）。（核对：按 softmax 定义实算这一行约为 `[0.58, 0.10, 0.08, 0.16, 0.09]`——$e^{2.1}\approx 8.17$ 占总和 $14.07$ 的 0.58；原课页面标 0.52 与实算有出入，排序与量级一致，以实算为准。）

### 2.4 为什么要缩放：$\sqrt{d_k}$ 拯救梯度

点积随维度增长：$d_k=64$ 时点积可达几十，把 softmax 推进梯度消失区。修法：

$$\text{Scaled scores} = \frac{QK^\top}{\sqrt{d_k}}$$

直觉版解释（补充）：维度每翻一倍，点积的典型幅度大约也翻一倍（每维贡献独立小量，方差相加）；除以 $\sqrt{d_k}$ 恰好把方差拉回不随维度变的水平——0.3 的例子里 $d_k=2$，除以 $\sqrt{2}\approx 1.414$ 把 2 变回 1.41。

### 2.5 softmax：分数变权重

softmax 把一行原始分数变成一行概率：先取指数（放大差距、保证非负），再除以该行指数和（归一）。数值稳定技巧：先减去该行最大值再取指数（$e^{x-\max}$ 与 $e^x$ 归一化结果相同，但避免 $e^{1000}$ 溢出）。

### 2.6 加权求和：输出即混合

每个词的最终输出是所有 value 的加权和：$output_i = \sum_j a_{ij} v_j$。原课示例：token 1 的输出 $= 0.52\,v_1 + 0.09\,v_2 + 0.07\,v_3 + 0.14\,v_4 + 0.08\,v_5$（原课示例权重）。

### 2.7 完整公式与流水线

$$\text{Attention}(Q, K, V) = \mathop{\mathrm{softmax}}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

流水线（原课流程图文字版）：$X$ 分三路投影成 Q、K、V → Q、K 打分并除以 $\sqrt{d_k}$ → softmax → 与 V 加权求和 → 输出。0.3 节的迷你例就是这条流水线在 2×2 数字上的完整运行。

### 2.8 因果掩码：让词看不见未来（补充，原课练习伏笔）

自注意力默认“全场互看”（双向，BERT 式）。GPT 式生成模型要求第 i 个词只看前 1～i 个词，做法是在 softmax **之前**把所有 $j > i$ 的分数改成 $-\infty$（实现上常写成一个加性掩码矩阵，取值 0 或 $-\infty$）。softmax 之后 $e^{-\infty}=0$，未来位置的权重恰好为 0。原课把这一机制放在练习 1，见第九节。

## 三、操作前准备

Build It 与 Use It 需要 NumPy（Use It 另需 PyTorch）：

~~~bash
python -m pip install numpy torch
~~~~

（核对：本课原课页面未展示安装命令，上行按原课语言要求 Python + NumPy/PyTorch 整理，供按需执行。）先激活你已有的 Python 环境；本次没有替你安装或下载任何模型。代码按定义顺序运行，无外部数据与服务调用。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：从零写 softmax

减最大值是数值稳定的关键——本课几乎所有数值坑都从这里防起。

~~~python
# 本块中文说明：数值稳定版 softmax，逐行归一。
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)  # 减最大值防溢出
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
~~~

（核对：`softmax` 沿最后一个轴（axis=-1）归一，二维输入即按行，与 2.5 节一致。）

### 4.2 第 2 步：缩放点积注意力

全课核心函数，四行对回 0.3 的四步。

~~~python
# 本块中文说明：缩放点积注意力，返回输出和权重矩阵。
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)   # 打分并除以 sqrt(dk)
    weights = softmax(scores)        # 逐行归一成注意力权重
    output = weights @ V             # 加权求和 value
    return output, weights
~~~

（核对/补充：`K.T` 只对二维输入正确——`.T` 会反转**所有**维度，若 Q/K 是三维张量（多头场景，见[第03课](<03：多头注意力——并行的视角子空间.md>)）应改用 `K.transpose(0, 2, 1)`。本课全部按二维使用，代码无误。）

### 4.3 第 3 步：带可学习投影的 SelfAttention 类

三个权重矩阵用 Xavier 式缩放初始化（标准差随输入输出维度缩小的初始化，让信号方差在前向后向都大致稳定）。

~~~python
# 本块中文说明：完整的自注意力模块，内含 Wq、Wk、Wv 三个投影矩阵。
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
~~~

（核对：本类没有输出投影矩阵 $W_o$——原版 Transformer 在注意力后还有一次线性变换；原课此处省略，第 03 课的多头实现会把它补上。这是简化不是错误。）

### 4.4 第 4 步：在句子上运行

6 词句子，随机嵌入，观察权重矩阵形状与数值。

~~~python
# 本块中文说明：给 6 个词造随机嵌入，跑一遍自注意力。
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
~~~

（核对/边界：嵌入是随机数、投影是随机初始化，所以打印出的权重**没有语言意义**——这只是形状与流程演示。真实语义要训练之后才出现。）

### 4.5 第 5 步：ASCII 热图可视化

把权重映射成字符深浅，快速目测“谁在看谁”。

~~~python
# 本块中文说明：把注意力权重视为热图，用字符深浅打印。
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
~~~

（核对：`weights.max()` 是全矩阵最大值，所以亮度按全局最大值归一；由于每行和为 1，长句里每行会整体变暗——n 越大，单格权重上限越接近 1/n 量级。）

## 五、把代码运行起来：补充入口与检查

（补充）以下是最小检查入口：用 0.3 节的迷你例给原课函数做断言，数字全部纸笔可验：

~~~python
# 本块中文说明：补充检查入口，用 0.3 的手算例验证 scaled_dot_product_attention。
import math

X_mini = np.array([[1.0, 0.0],   # 词 A = “猫”
                   [1.0, 1.0]])  # 词 B = “追”
I2 = np.eye(2)                   # 投影取单位矩阵，让数字可手算

out_mini, w_mini = scaled_dot_product_attention(X_mini @ I2, X_mini @ I2, X_mini @ I2)

# 0.3 节手算：权重 [[0.50, 0.50], [0.33, 0.67]]，输出 [[1.00, 0.50], [1.00, 0.67]]
assert np.allclose(w_mini, [[0.5, 0.5], [0.3302, 0.6698]], atol=1e-3)
assert np.allclose(out_mini, [[1.0, 0.5], [1.0, 0.6698]], atol=1e-3)
assert np.allclose(w_mini.sum(axis=-1), 1.0, atol=1e-9)  # 每行和为 1
print("迷你注意力检查通过")
~~~

（补充）再手算一个因果掩码的最小例（对应 2.8 节与练习 1）：

~~~python
# 本块中文说明：补充的因果掩码演示——词 A 不许看词 B。
scores_mini = X_mini @ X_mini.T / math.sqrt(2)      # 缩放后分数 [[0.71, 0.71], [0.71, 1.41]]
mask = np.array([[0.0, -np.inf],                   # A 在位置 1：未来（B）= -inf
                 [0.0, 0.0]])                      # B 在位置 2：过去全可见
masked_weights = softmax(scores_mini + mask)
# 手算：A 行 softmax([0.71, -inf]) = [1.0, 0.0]；B 行不变 = [0.33, 0.67]
assert np.allclose(masked_weights, [[1.0, 0.0], [0.3302, 0.6698]], atol=1e-3)
print("因果掩码检查通过")
~~~

运行 4.4 后还可自查：`weights.shape == (6, 6)`、`output.shape == (6, 4)`（dv=4）、`weights.sum(axis=1)` 全为 1。

## 六、Use It：使用现成工具

PyTorch 的 `nn.MultiheadAttention` 做的就是本课手写的东西，外加多头拆分与输出投影：

~~~python
# 本块中文说明：PyTorch 版多头注意力，Q=K=V=同一输入即自注意力。
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
~~~

关键区别（原课）：多头注意力并行跑多套注意力函数，每套有自己的 Q/K/V 投影、维度 $d_k = d_{\text{model}}/n_{\text{heads}}$，最后拼接结果——让模型同时关注不同类型的关系。这是[第03课](<03：多头注意力——并行的视角子空间.md>)的主题。另注意 PyTorch 返回的权重默认是**各头平均**后的，与手写版“每头一个矩阵”不同。

## 七、失败模式、原文边界与工程注意

1. 忘记除以 $\sqrt{d_k}$：小 $d_k$ 时侥幸能跑，$d_k$ 到 64 就会 softmax 饱和、梯度消失，训练卡死。这是新手复现 Transformer 最常见的坑之一。
2. softmax 忘减最大值：`np.exp` 遇到大分数（如 1000）直接溢出为 inf/nan。原课第 1 步的 `shifted` 就是为这个。
3. 三维张量上用 `.T`：会把所有维度倒序（头、序列、维度全乱）。多头场景必须 `transpose(0, 2, 1)`（见第 03 课核对）。
4. 掩码加在 softmax 之后：等于直接抹掉概率，行和不再是 1 之后的再分配，语义错误。必须在 softmax 之前以 $-\infty$（或极大负数）加进去。
5. 随机权重的注意力矩阵没有语义：4.4 的演示只验证机制与形状，“猫注意 mat”这类解读要等训练后才成立（原文边界）。
6. $d_k$ 与 $d_v$ 是两回事：Q/K 用于打分（必须同维才能点积），V 决定输出宽度。原课代码取了相同值（4）只是本例的便利。
7. 自注意力是 $O(n^2)$ 的：6 词无所谓，长上下文时 n×n 矩阵就是[第01课](<01：为什么是Transformer——RNN的三个瓶颈.md>)说的内存墙，工程解法见[第12课](<12：KV缓存与Flash Attention——推理优化.md>)。
8. 本课的 `SelfAttention` 没有输出投影 $W_o$，也没有残差连接和层归一化——那些属于完整 Transformer 块（[第05课](<05：完整Transformer——编码器与解码器.md>)）；单层裸注意力直接堆叠训练会不稳（补充）。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/prompt-attention-explainer.md`：一个“用数据库查找类比解释注意力”的提示词文档。以下是本课的中文学习版自查模板（直接保存在本篇，不另建输出文件）：

~~~markdown
向别人讲注意力时逐项自查：能用“查询/键/值”三个白话定义开场吗？能写出完整公式并指出 √dk、softmax、加权求和各防什么坑吗？能用一个 2×2 的手算例子（如 0.3 节）从头走到尾吗？能解释“每行和为 1”与“掩码在 softmax 前加”吗？若用于代码评审：检查 softmax 是否减最大值、点积是否缩放、三维转置是否用 transpose(0,2,1)、掩码时机是否正确。所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课练习 1：给 `scaled_dot_product_attention` 加可选 mask 参数，把部分位置在 softmax 前设为负无穷（因果/解码器掩码的原理）。验收标准：用第五节的补充掩码例做断言（遮住未来位置后，该行权重集中在可见位置且行和为 1）；再对 4 词序列验证下三角掩码的形状正确。
2. 原课练习 2：从零实现多头注意力——把 Q、K、V 拆成 `n_heads` 份，各跑注意力，拼接后过输出矩阵 $W_o$。验收标准：与单头版在 $n_{\text{heads}}{=}1$ 时数值一致；输出形状恢复 (n, d_model)。完整讲解见[第03课](<03：多头注意力——并行的视角子空间.md>)。
3. 原课练习 3：取两个等长句子，过同一个 `SelfAttention` 实例，比较注意力模式：什么变了、什么没变？验收标准：能指出“变的”是内容驱动的分数与权重，“不变的”是投影矩阵与机制本身；能解释为什么权重矩阵的行和恒为 1。

## 十、术语与完成标准

查询/键/值：同一个嵌入经三个可学习矩阵投影出的“我要找什么/我像什么/我提供什么”；缩放点积注意力：$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$，缩放防止 softmax 饱和；自注意力：Q、K、V 同源的注意力，每个位置可看全序列；注意力权重：softmax 后的行概率分布；上下文向量：value 的加权和；因果掩码：softmax 前把未来位置设为 $-\infty$；softmax 稳定技巧：先减行最大值。

完成检查：

- 能不看书画出 X → Q/K/V → 打分 → 缩放 → softmax → 加权求和的流程。
- 能手算 0.3 的 2×2 迷你例（含 $e^{0.71}\approx 2.03$ 的用法）。
- 能解释 $\sqrt{d_k}$、减最大值、掩码在 softmax 前，各自防什么错。
- 能说出注意力权重的两个不变量（非负、行和为 1）及其来源。
- 能区分自注意力与交叉注意力、$d_k$ 与 $d_v$。
- 知道本课实现缺什么（$W_o$、多头、残差、归一化），以及它们分别在哪一课补齐。

## 十一、验证范围与来源许可

原课 Build It 共 5 步、Use It 1 段，代码已逐字转录嵌入本篇（仅按本篇格式添加中文注释，未改逻辑）。原课页面未展示运行输出，本篇未虚构任何“原课参考”数值；2.3 节指出了原课 softmax 示例数值（0.52）与实算（约 0.58）的出入，以实算为准。

本次验证记录（2026-11-06）：

- 抓取课程页成功（一次成功，无重试），内容以原文为准整理。
- 0.3 节迷你注意力（权重 [[0.5, 0.5], [0.3302, 0.6698]]、输出 [[1.0, 0.5], [1.0, 0.6698]]）与 2.3 节 softmax 实算（0.58/0.10/0.08/0.16/0.09）均用本机 Python 3.12 + NumPy 2.5 复核通过。
- 第五节两段补充断言代码已在本机实际执行通过（属 2×2 玩具断言，不构成训练或性能实验）；第四、六节原课演示代码未在本机运行，其输出未预设。
- 正文 8 个 python 代码块全部通过 `ast.parse` 静态语法检查。
- 未做：训练任何模型、与 PyTorch 实现的数值比对。

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
