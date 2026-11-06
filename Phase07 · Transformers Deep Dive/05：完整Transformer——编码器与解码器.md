# 05：完整Transformer——编码器与解码器

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F05-full-transformer) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/05-full-transformer/docs/en.md)。原题：The Full Transformer — Encoder + Decoder。

## 零、初学者先读：把零件组装成整栋楼

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。前几课（[第02课](<02：从零实现自注意力——Q、K、V与缩放点积.md>)、[第03课](<03：多头注意力——并行的视角子空间.md>)、[第04课](<04：位置编码——正弦、RoPE与ALiBi.md>)）已经造好了零件，本课讲怎么把零件组装成一栋能盖很高的楼；后面三课分别是这座楼的两种“减配版”和“完整版”。

### 0.1 先理解这一课在做什么

单层的自注意力再精巧，也只是一个“特征提取器”，不是一个模型——一次矩阵乘法撑不起语言所需的容量，你需要**深度**（很多层堆起来）。但层数一深，训练就会崩：梯度消失、数值爆炸、学不动。2017 年的 Vaswani 论文给出的答案是六个设计决策，把一层注意力变成一个可以无限堆叠的“块”（block）。此后所有 Transformer——编码器型的 BERT、解码器型的 GPT、编码器-解码器型的 T5——都继承这同一副骨架。到 2026 年，块内的零件被翻新过（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架一模一样。本课就是讲这副骨架。

### 0.2 本课术语先翻成白话

- 块（block）=一层 Transformer：归一化 + 注意力 + 归一化 + 前馈网络，外面包着残差连接
- 编码器（encoder）=双向看输入的堆叠块：任何位置能看到所有位置，用来“理解”
- 解码器（decoder）=单向看输入的堆叠块：每个位置只能看到自己和更早的位置，用来“生成”
- 残差连接（residual connection）=`x + 子层(x)`：给信息留一条“高速公路”，没有它约 6 层以上梯度就消失
- 层归一化（LayerNorm）=每个向量自己减均值、除标准差，把数值稳定住
- RMSNorm=LayerNorm 的简化版：不减均值，只除均方根，少一次减法
- 前馈网络（feed-forward network, FFN）=作用在每个位置各自身上的两层 MLP：`W2 · 激活(W1 · x)`
- 扩展率（expansion ratio）=FFN 中间层比 d_model 宽几倍，经典 4 倍，SwiGLU 时代约 2.6 倍
- 交叉注意力（cross-attention）=Q 来自解码器、K/V 来自编码器输出的注意力——编码器信息进入解码器的唯一通道
- pre-norm / post-norm=归一化放在子层之前（`x + 子层(LN(x))`）还是之后（`LN(x + 子层(x))`）；2026 年默认 pre-norm
- SwiGLU=现代 FFN 激活：`Swish(W1x) ⊙ W3x` 再乘 W2，用三个矩阵
- GQA（grouped-query attention）=多组查询共享一组 K/V 的注意力省钱变体（本课只在选型表里出现名字）

### 0.3 用小数字走一遍

设一个玩具配置：3 个 token 的句子、d_model=4、2 个头（每头 d_k=2）、FFN 中间维度 8（扩展率 r=2）、词表大小 10。看一个向量如何流过**一层编码器**：

| 步骤 | 变换 | 形状变化 | 参数（本例大小） |
|---|---|---|---|
| 1 词嵌入查表 | token ID → 向量 | (3,) → (3,4) | emb: (10,4)，40 个 |
| 2 加位置编码 | x + pos | (3,4) | 0（正弦无参数） |
| 3 pre-norm | LN(x) | (3,4) | 本课简化版 0 |
| 4 Q/K/V 投影 | x·W_Q 等 | (3,4) → 三个 (3,4) | W_Q、W_K、W_V 各 (4,4)，共 48 |
| 5 拆 2 头 | (3,4) → 2 × (3,2) | 每头 d_k=2 | 0 |
| 6 注意力打分 | Q·Kᵀ | 每头 (3,3) | 0 |
| 7 softmax+加权 | A·V | 每头 (3,2) | 0 |
| 8 拼回+输出投影 | concat → ·W_O | (3,4) | W_O: (4,4)，16 |
| 9 残差 | x + attn(x) | (3,4) | 0 |
| 10 FFN（SwiGLU） | Swish(xW1)⊙xW3 → ·W2 | (3,4)→(3,8)→(3,4) | W1、W3 各 (4,8)，W2 (8,4)，共 96 |
| 11 残差 | x + ffn(x) | (3,4) | 0 |

参数量核对（这是本课公式在玩具尺度上的自洽检查）：注意力部分 4·d² = 4×16 = 64（即 48+16）；FFN（SwiGLU）3·d·(r·d) = 3×4×8 = 96；归一化忽略不计。一个块共 160 个参数，与公式 4d²+3rd² 一致。2 层编码器 + 2 层解码器 = 4 块 ≈ 640 个参数（不算嵌入表）——所以别被“175B 参数”吓到，它只是同一块重复堆了几百次。

再手算两个块内零件。**LayerNorm**：取向量 [1,2,3,4]，均值 2.5，方差 (2.25+0.25+0.25+2.25)/4 = 1.25，标准差 ≈1.118，输出 ≈ [-1.342, -0.447, 0.447, 1.342]。**RMSNorm**：均方根 = √((1+4+9+16)/4) = √7.5 ≈ 2.739，输出 ≈ [0.365, 0.730, 1.096, 1.461]——注意它不做减均值，所以中心偏了但形状（相对大小关系）没变。**残差**：若 x=[1,0,0,0]、注意力输出 [0,1,0,0]，则相加得 [1,1,0,0]——原来那个 1 永远在，子层只往里“添东西”。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：先看第四节转录的 `encode`/`decode` 两个函数——它们就是上表第 1—2 步加一个“逐块循环”，编码器循环调 `encoder_block`，解码器循环调 `decoder_block` 且多传一个 `enc_out`（编码器输出，交叉注意力要用）。课程页把块的内部接线放在 `code/main.py`（页面未内嵌全文，见第七节的原文边界说明）。核心记忆点只有两个数据流：**残差流**（x 一路加下去）和**块间流**（前一层的输出是后一层的输入，编码器输出喂给每个解码器块）。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么残差连接是不可选的？

**参考答案：** 没有 `x + f(x)` 而只有 `f(x)` 时，梯度要逐层连乘 ∂f/∂x，乘几十次小数就消失——原课说约 6 层以上就训不动。有了残差，∂(x+f(x))/∂x = 1 + ∂f/∂x，那个常数 1 给梯度留了一条不衰减的直通高速公路。

**问题2：** pre-norm 和 post-norm 哪个是 2026 年的默认？为什么？

**参考答案：** pre-norm（LN 在子层之前）。post-norm 是 2017 原论文写法，深堆时需要精心设计的 warmup 才训得稳，2019 年前后失宠；Llama、Qwen、GPT-3+、Mistral 都是 pre-norm。

**问题3：** 解码器块比编码器块多什么？这个“多出来的东西”里张量从哪来？

**参考答案：** 多一个交叉注意力子层，所以解码器每块有三个子层（掩码自注意力、交叉注意力、FFN），编码器只有两个。交叉注意力里 Q 来自解码器，K 和 V 来自编码器的输出——这是编码器信息进入解码器的唯一通道。

**问题4：** 用 0.3 的配置算：把 d_model 换成 512、SwiGLU 扩展率 2.6，一个块多少参数？

**参考答案：** 4·d² + 3·r·d² = 4×512² + 3×2.6×512² ≈ 1.05M + 2.04M ≈ 3.1M。公式记不住没关系，记得“注意力占 4d²、FFN 占大头（SwiGLU 约 7.8d²）”即可。

**问题5：** 位置信息为什么必须在这一课注入，而不是留给注意力自己学？

**参考答案：** 第02课讲过：裸的自注意力对所有位置的置换是等变的——打乱 token 顺序，输出只是跟着打乱，内容不变。也就是说注意力本身“看不见顺序”，不注入位置信号的话“猫咬狗”和“狗咬猫”在模型眼里一模一样（第04课专题讲了正弦/RoPE/ALiBi 三种注法）。

能看懂这张维度表和三个小算例后，进入正文。

---

## 一、这一课解决什么问题？

一层注意力不够，语言需要深度；但深度没有正确的“管道”就会崩。原课的话：注意力是明星，其他一切——残差、归一化、前馈网络、交叉注意力——都是让你能堆深的脚手架。2017 年 Vaswani 论文把六个设计决策打包成一个可堆叠的块，此后每个 Transformer（BERT、GPT、T5）都继承同一副骨架。2026 年块内零件翻新（RMSNorm、SwiGLU、pre-norm、RoPE），骨架未变。本课讲骨架本身，后面三课做特化：[第06课](<06：BERT——掩码语言建模.md>)只留编码器、[第07课](<07：GPT——因果语言建模.md>)只留解码器、[第08课](<08：T5与BART——编码器-解码器模型.md>)两个都留。

## 二、核心概念与推导

### 2.1 六个部件

1. **嵌入 + 位置信号**：token → 向量，位置经 RoPE（现代）或正弦（经典）注入。
2. **自注意力**：每个位置看到其他所有位置；解码器里要加掩码。
3. **前馈网络（FFN）**：逐位置的两层 MLP：`W2 · 激活(W1 · x)`，默认扩展率 4 倍。“逐位置”指每个 token 的向量独立过同一个 MLP，互相不串门——混合信息靠注意力，加工信息靠 FFN，分工明确。
4. **残差连接**：`x + 子层(x)`，没有它约 6 层以上梯度消失。
5. **层归一化**：LayerNorm 或 RMSNorm（现代），稳住残差流。
6. **交叉注意力（仅解码器）**：Q 来自解码器，K/V 来自编码器输出。

盯住一个向量流过一块的样子：注意力跨位置混合它，残差把它带下去，FFN 再加工它，归一化让整条流保持稳定。如此往复堆叠。

### 2.2 编码器块与解码器块（原课接线图）

编码器块（BERT、T5 编码器在用）：

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的，无掩码，所有位置互相可见。

解码器块（GPT、T5 解码器在用）：

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每块有三个子层。中间那个交叉注意力是编码器信息流入解码器的唯一地方。纯解码器架构（GPT）没有交叉注意力子层，只剩“掩码自注意力 + FFN”两个子层。

### 2.3 pre-norm 对 post-norm

原论文写法：`x + 子层(LN(x))`（pre）对 `LN(x + 子层(x))`（post）。原课表述：post-norm 约 2019 年失宠——不做精心 warmup 就很难训深；pre-norm（LN 在子层之前）是 2026 年默认，Llama、Qwen、GPT-3+、Mistral 都在用。直觉（补充）：pre-norm 把归一化放进残差分支里，主干 `x` 不被反复缩放，深堆时更稳。

### 2.4 2026 年的现代块：改了哪些零件

Vaswani 2017 出厂配置是 LayerNorm + ReLU，现代生产块换掉了两者（原课对照表）：

| 部件 | 2017 | 2026 |
|------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活 | ReLU | SwiGLU |
| FFN 扩展率 | 4× | 2.6×（SwiGLU 用三个矩阵，总参数量相当） |
| 位置 | 正弦绝对位置 | RoPE |
| 注意力 | 全量 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 砍掉 LayerNorm 的减均值（少一次减法），省计算且经验上至少同样稳。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM、Qwen 论文里比 ReLU/GELU FFN 稳定好约 0.5 点困惑度（原课参考，引用自论文，非本机复测）。注意扩展率的账：ReLU 版 2 个矩阵 × 4 倍 = 8d² 参数，SwiGLU 版 3 个矩阵 × 2.6 倍 = 7.8d²，所以“降到 2.6 倍”不是省参数，是换三个矩阵后总账差不多。

### 2.5 参数量怎么数

对 d_model=d、FFN 扩展率 r 的一个块：

- MHA：`4·d²`（Q、K、V、O 四个投影）
- FFN（SwiGLU）：`3·d·(r·d) ≈ 3rd²`
- 归一化：可忽略

用 d=4096、r=2.6、层数=32（大致 Llama 3 8B）验算：32 × (4×4096² + 3×2.6×4096²) ≈ 32 × (67M + 131M) ≈ 32 × 198M ≈ 6.3B 在块里，加上嵌入和输出头约 1B（128K 词表 × 4096，不共享），落到接近公开的 8B。真实模型用 GQA（K/V 投影更小）和更宽的 FFN（r=3.5），把参数从注意力挪到 FFN，总数仍在 8B 附近。（以上全部为原课参考的估算，非本机实测；0.3 节的玩具版 d=4、r=2 用同一公式核对过。）

### 2.6 把六个部件串成一段话（补充）

把 2.5 的参数账和 2.1—2.3 的接线合起来读一遍整栋楼：token 先变成向量（部件1），每进一块，注意力让它们互相“对齐意见”（部件2），残差把过去的自己原封不动带下去（部件4），归一化防止数值越滚越大（部件5），FFN 在每个位置独立做深加工（部件3）；如果是解码器，还会中途隔着栅栏看一眼编码器的笔记（部件6）。出楼后再过一个输出投影，得到对词表每个词的打分。参数几乎全花在注意力投影（4d²）和 FFN（约 8d²）上，层数乘上去就是十亿级——所以“堆多少层”本质上是在问“你有多少钱买 8d² 的倍数”。

## 三、操作前准备

原课 Build It 声称只用纯 Python——它复用第03课那个极简 `Matrix` 类（原话："copied to this file for independence"，即复制进本课文件以保持独立），无第三方依赖。Use It 部分涉及 PyTorch/HF `transformers`（按需安装）：

~~~bash
python -m pip install torch transformers
~~~

（核对/说明：课程页 Build It 只内嵌了 `encode`/`decode` 两个函数的代码，其余步骤以文字描述指向 `code/main.py`，页面未展示其全文；本篇按课程页实际内容整理，不虚构未见的代码，缺失部分用（补充）自写示意补足并显式标注。）

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：块内零件

原课列出的零件清单（见 `code/main.py`，页面未内嵌代码，此处如实转录清单）：`layer_norm(x, eps=1e-5)`（减均值除标准差）、`rms_norm(x, eps=1e-6)`（除 RMS、不减均值）、`gelu(x)` 与 `silu(x) * W3 x`（SwiGLU）、`ffn_swiglu(x, W1, W2, W3)`、`encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

（补充）为让上述清单可动手验证，下面给出等价功能的最小纯 Python 实现，**非原课代码**：

~~~python
# 本块中文说明：（补充）自写的最小归一化实现，用于核对 0.3 节手算，不是原课代码。
def layer_norm(x, eps=1e-5):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    return [(v - mean) / (var + eps) ** 0.5 for v in x]

def rms_norm(x, eps=1e-6):
    rms = (sum(v * v for v in x) / len(x) + eps) ** 0.5
    return [v / rms for v in x]

print([round(v, 3) for v in layer_norm([1, 2, 3, 4])])   # 预期约 [-1.342, -0.447, 0.447, 1.342]
print([round(v, 3) for v in rms_norm([1, 2, 3, 4])])     # 预期约 [0.365, 0.730, 1.096, 1.461]
~~~

### 4.2 第 2 步：接一个 2 层编码器 + 2 层解码器

原课代码（逐字转录，仅加中文注释）：堆叠两个编码器块、两个解码器块，把编码器输出传给**每一个**解码器块的交叉注意力，输出投影前再加一个最终 LN。

~~~python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    # 中文注释：查嵌入表 + 正弦位置编码，得到 (N, d)。
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
        # 中文注释：逐块过编码器；每块 = LN→自注意力→+残差→LN→FFN→+残差。
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
        # 中文注释：解码器块多一个入参 encoder_out——交叉注意力的 K/V 从这里来。
    return x
~~~

（核对/说明：`embed`、`sinusoidal`、`encoder_block`、`decoder_block` 与 `params` 的定义在课程页未展示；`encoder_blocks`/`decoder_blocks` 按上文是块参数的列表。）

### 4.3 第 3 步：玩具样例上前向

喂一个 6 token 的源序列和一个 5 token 的目标序列，验证输出形状是 `(5, vocab)`。原课明确：不做训练——本课讲架构不讲损失。

### 4.4 第 4 步：换成 RMSNorm + SwiGLU

把 LayerNorm 和 ReLU-FFN 换成 RMSNorm 和 SwiGLU，确认形状仍然对上。原课的话：这就是“一次函数替换”完成的 2026 现代化。

## 五、把代码运行起来：补充入口与检查

（补充）课程页未提供完整可运行脚本，以下自写最小形状检查，用 0.3 节同款配置（d_model=4、2 头、FFN=8、3 token）把维度流转表在 numpy 里走一遍，只验形状与归一化手算，不含任何训练：

~~~python
# 本块中文说明：（补充）自写的形状检查，非原课代码；只验形状与 0.3 节手算。
import numpy as np

np.random.seed(0)
N, d, h, ffn = 3, 4, 2, 8
x = np.random.randn(N, d)                      # 词嵌入+位置后的 (3,4)
Wq, Wk, Wv, Wo = (np.random.randn(d, d) for _ in range(4))
Q, K, V = x @ Wq, x @ Wk, x @ Wv               # 各 (3,4)
Qh = Q.reshape(N, h, d // h).transpose(1, 0, 2)  # (2,3,2)：拆 2 头
Kh = K.reshape(N, h, d // h).transpose(1, 0, 2)
Vh = V.reshape(N, h, d // h).transpose(1, 0, 2)
scores = Qh @ Kh.transpose(0, 2, 1) / (d // h) ** 0.5   # (2,3,3)：每头打分
A = np.exp(scores - scores.max(-1, keepdims=True))
A = A / A.sum(-1, keepdims=True)               # 每行和为 1
out_h = A @ Vh                                  # (2,3,2)
out = out_h.transpose(1, 0, 2).reshape(N, d) @ Wo  # 拼回 (3,4)
attn = x + out                                   # 残差
W1, W3 = np.random.randn(d, ffn), np.random.randn(d, ffn)
W2 = np.random.randn(ffn, d)
def silu(t): return t / (1 + np.exp(-t))
ffn_out = (silu(attn @ W1) * (attn @ W3)) @ W2  # SwiGLU：(3,4)→(3,8)→(3,4)
block_out = attn + ffn_out
assert block_out.shape == (N, d)                # 一层块进出形状不变
assert np.allclose(A.sum(-1), 1.0)              # 注意力行随机

# 归一化手算复核（0.3 节）：
xs = np.array([1.0, 2, 3, 4])
ln = (xs - xs.mean()) / (xs.var() + 1e-5) ** 0.5
rmsn = xs / ((xs ** 2).mean() + 1e-6) ** 0.5
assert np.allclose(ln, [-1.342, -0.447, 0.447, 1.342], atol=1e-3)
assert np.allclose(rmsn, [0.365, 0.730, 1.096, 1.461], atol=1e-3)
print("shape & norm checks passed:", block_out.shape)
~~~

检查要点：块进块出形状都是 (3,4)——这正是“块”可堆叠的前提；拆头/拼回是纯 reshape，参数量不变。

## 六、Use It：使用现成工具

PyTorch/TF 的参考实现是 `nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但原课指出，2026 年多数生产代码自己写块，因为：

- Flash Attention 要在注意力内部调用，不是走 `nn.MultiheadAttention`；
- GQA / MLA 不在标准库参考里；
- RoPE、RMSNorm、SwiGLU 都不是 PyTorch 默认。

HF `transformers` 里有干净的参考块值得读：`modeling_llama.py` 是 2026 解码器块的范本，约 500 行，值得通读一遍（原课建议）。

**编码器 / 解码器 / 编码器-解码器怎么选**（原课对照表）：

| 需求 | 选择 | 例子 |
|------|------|------|
| 分类、嵌入、文本问答 | 编码器-only | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 解码器-only | GPT、Llama、Claude、Qwen |
| 结构化输入→结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

原课的判断：解码器赢得语言建模是因为它扩展性最干净、且理解与生成通吃；编码器-解码器在输入有明确“源序列”身份时（翻译、语音识别、结构化任务）仍是最佳。

## 七、失败模式、原文边界与工程注意

1. 拿掉残差连接再堆深：约 6 层梯度消失，训练直接不动——这是本课第一失败模式（原课强调）。
2. post-norm 深堆不配 warmup：原课明说它“更难训深”，新手复现 2017 配置时容易在这里翻车。
3. SwiGLU 记错参数账：它有三个矩阵（W1、W2、W3），若顺手沿用 4× 扩展率，参数量会比预期多一半（4×3=12d² vs 传统 8d²）；正确做法是降到约 2.6×，总账才与传统相当（补充：由 2.4 的算式可直接推出）。
4. 交叉注意力方向接反：Q 必须来自解码器、K/V 来自编码器；接反后解码器会“自己看自己”，翻译任务立刻崩（补充提醒，原课在术语表里强调方向）。
5. 本课没有训练、没有损失函数——它只是架构前向。别把这里的输出当任何质量证据。
6. 原文边界：SwiGLU “好约 0.5 点困惑度”是原课引用 Llama/PaLM/Qwen 论文的参考数值；Llama 3 8B 的参数估算是原课的粗略对账（真实模型 GQA + r=3.5 的细节未在估算里完整展开）。均非本机实测。
7. 课程页 Build It 的块内代码（`layer_norm`、`encoder_block` 等）只给函数签名清单，全文在 `code/main.py` 且页面未展示；本篇以（补充）标注的自写代码示意，不等同原课实现（原文边界，如实说明）。
8. 偏置项：2026 块默认无偏置（对照表），从 2017 配置迁移时记得删掉 `+b`，参数和推理都略省（原课：“slight ppl improvement, smaller model”，原课参考）。
9. 本课的维度表只跟踪了形状、没跟踪数值范围：真实实现中若忘记 pre-norm 或残差，形状照样全对、代码照样能跑——形状检查抓不出训练稳定性问题，所以练习2才要你量激活范数（补充提醒）。
10. “一个块”不等于“一层 LSTM 意义上的层”（补充）：Transformer 的“层”内部有两个（编码器）或三个（解码器）子层各带自己的参数。看模型卡里的“32 层”时，心里要乘上 4d²+3rd²，而不是只想到一次矩阵乘法——参数预算错觉是读架构表的常见误判。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/skill-transformer-block-reviewer.md`（审查一个新 Transformer 块实现是否符合 2026 默认配置并标记缺失项）。中文学习版自查模板（直接保存在本篇，不另建文件）：

~~~markdown
审查一个 Transformer 块实现：是否 pre-norm（LN 在子层前）？归一化用 LayerNorm 还是 RMSNorm、是否漏掉残差分支？位置方案是 RoPE 还是可学习绝对位置？注意力是否支持 GQA/MLA、K/V 共享数是多少？FFN 激活是 SwiGLU（三个矩阵）还是 ReLU/GELU（两个矩阵）、扩展率多少、参数总量是否与 4d²+3rd² 对得上？偏置项是否按现代默认去掉？以上每项给出：现状、与 2026 默认的差距、是否需要改。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：数出你的 `encoder_block` 在 d_model=512、n_heads=8、ffn_expansion=4、swiglu=True 时的参数量。验收标准：动手实现该块并用 `sum(p.numel() for p in block.parameters())` 核对，与手算（4d² + 3·4·d² = 16d² = 4.19M）一致。
2. 原课 Medium：从 post-norm 切到 pre-norm，初始化两个版本，在随机输入上量 12 层堆叠后的激活范数。验收标准：post-norm 的激活应爆炸、pre-norm 应保持有界；能解释差异来自主干是否被反复归一化缩放。
3. 原课 Hard：在玩具复制任务（把 x 逆序复制出来）上实现 4 层编码器-解码器，训练 100 步并报告损失；再换成 RMSNorm + SwiGLU + RoPE，看损失是否下降。验收标准：两个配置的训练损失曲线各一条，能说明换零件带来的差异（或无差异）及可能原因。

## 十、术语与完成标准

块（block）：归一化+注意力+归一化+FFN 外包残差的堆叠单元；残差（residual）：`x + f(x)`，让深堆可训练；pre-norm：`x + 子层(LN(x))`，2026 默认；RMSNorm：不减均值、只除均方根的归一化；SwiGLU：`Swish(W1 x) ⊙ W3 x → W2` 的 FFN；交叉注意力（cross-attention）：Q 来自解码器、K/V 来自编码器的注意力；FFN 扩展率：中间层与 d_model 之比（经典 4，SwiGLU 约 2.6）；无偏置（bias-free）：现代线性层省掉 +b。

完成检查：

- 能画出编码器块和解码器块的接线图，说清后者多出的子层及 K/V 来源。
- 能背出六个部件并各说一句“没有它会怎样”。
- 能用 d_model=4、2 头、ffn=8 走完 0.3 的维度表，并用 4d²+3rd² 算出块参数量。
- 能手算 [1,2,3,4] 的 LayerNorm 与 RMSNorm 输出。
- 能说出 2017→2026 的六项零件替换及 SwiGLU 扩展率降到 2.6 的参数账原因。
- 能说出三种架构（编码器/解码器/编码器-解码器）各自的适用场景与代表模型。
- 能解释位置信号为什么必须在块外注入（注意力对位置置换等变，见问题5）。

## 十一、验证范围与来源许可

课程页抓取成功（2026-11-06）；原课文档的 raw.githubusercontent 与 github.com 页面均返回 502/504 不可达，与此前阶段笔记遇到的情况相同，故以课程页内容为准整理，未参考或虚构页面之外的代码。原课页内嵌代码仅 `encode`/`decode` 两函数，已逐字转录（仅加中文注释，未改逻辑）；其余 Build It 步骤在页面上只有文字描述，本篇自写的补充代码均显式标注“（补充）”“非原课代码”。0.3 节的 LayerNorm/RMSNorm 手算、维度流转表与参数量公式，已于本机 Python 3.12 + numpy 2.5 实际执行复核通过（玩具尺度，不含任何训练）。正文 python 代码块已抽出为临时 .py 文件通过 `ast.parse` 静态语法检查。未执行任何训练、未下载模型。标“原课参考”的数值（SwiGLU 困惑度改进、Llama 3 8B 估算等）均来自课程页原文。

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
