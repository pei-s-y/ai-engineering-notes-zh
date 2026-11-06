# 04：位置编码——正弦、RoPE与ALiBi

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F04-positional-encoding) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/04-positional-encoding/docs/en.md)。原题：Positional Encoding — Sinusoidal, RoPE, ALiBi。

## 零、初学者先读：注意力天生“脸盲”于顺序

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第02课](<02：从零实现自注意力——Q、K、V与缩放点积.md>)的注意力和[第03课](<03：多头注意力——并行的视角子空间.md>)的多头都只看“内容”，不看“顺序”——本课补上这块缺失的拼图，Transformer 才算完整。

### 0.1 先理解这一课在做什么

先做个思想实验：把“猫 追 狗”打乱成“狗 追 猫”，注意力会给同样的词打同样的分——因为 Q 和 K 只由词的内容决定。也就是说，注意力是**排列不变**（permutation-invariant）的：打乱输入的行，输出的行跟着同样打乱，模型毫无察觉。对“词袋”任务这不是 bug，但对语言、代码、音频这种**顺序承载意义**的任务是致命的。

修法是把位置信息想办法注入。历史上分三个时代：**绝对正弦编码**（2017，把 sin/cos 加进嵌入，简单但外推差）；**RoPE 旋转位置编码**（2021，按位置旋转 Q 和 K，让点积直接编码相对位置，2026 年的主流）；**ALiBi**（2022，干脆不动嵌入，直接按距离给注意力分数打折扣，外推极好）。本课把三种方法都手算一遍。

### 0.2 本课术语先翻成白话

- 排列不变（permutation-invariant）=打乱输入顺序，输出只是跟着打乱、内容不变——注意力的天生属性
- 位置编码（positional encoding）=任何把“你在第几位”告诉模型的信号
- 绝对位置 vs 相对位置=“我是第 3 个词” vs “你在我前面 2 个词”
- 正弦编码（sinusoidal encoding）=用不同频率的 sin/cos 曲线给每个位置生成一个固定向量，加到嵌入上
- 频率（frequency）=每对维度一条正弦波的快慢；前面的维度对波长短（变得快），后面的波长长（变得慢）
- 相位（phase）=正弦波在某个位置的“刻度”；模型通过读各维度的相位组合来分辨位置
- RoPE（旋转位置编码，Rotary Position Embedding）=按位置把 Q、K 向量旋转一个角度，点积自动只依赖两位置的差
- base（基）=RoPE 的频率总旋钮，默认 10000；调大可以延展上下文
- 外推（extrapolation）=训练时只见过 0–2047 的位置，推理时能不能正确处理 2048 以后的位置
- ALiBi（带线性偏置的注意力）=不给嵌入加东西，直接在注意力分数上减“斜率 × 距离”
- 斜率（slope）=ALiBi 每个头自己的惩罚强度，$m_h = 2^{-8h/H}$，前面的头惩罚陡、后面的头惩罚缓
- NTK-aware / YaRN / LongRoPE=三种“调 base 或逐维插值来延展上下文”的技巧
- 学习式绝对编码=像查表一样给每个位置学一个向量（GPT-2 的做法），没见过的位置就没有向量
- 词袋（bag of words）=只统计“有哪些词”不管顺序的表示法——排列不变对它反而是优点

### 0.3 用小数字走一遍

**（1）手算正弦位置编码（$d_{\text{model}}=4$，即两对维度）。** 公式：偶数维 $\sin(pos/10000^{2i/d})$、奇数维配对 $\cos(\cdot)$。维度对 $i=0$ 的角度是 $pos$，维度对 $i=1$ 的角度是 $pos/10000^{2/4}=pos/100$：

| 位置 | 维0 = sin(pos) | 维1 = cos(pos) | 维2 = sin(pos/100) | 维3 = cos(pos/100) |
|-----:|---------------:|---------------:|-------------------:|-------------------:|
| 0    | 0.000 | 1.000 | 0.0000 | 1.0000 |
| 1    | 0.841 | 0.540 | 0.0100 | 1.0000 |
| 2    | 0.909 | −0.416 | 0.0200 | 0.9998 |

读法：维度对 (0,1) 是**高频**——位置从 1 走到 2，正弦从 0.84 跳到 0.91、余弦从 0.54 翻到 −0.42，变化剧烈；维度对 (2,3) 是**低频**——走 100 步才转完高频 1 步的量。每个位置拿到一个独一无二的“相位指纹”（位置 1 是 [0.841, 0.540, 0.010, 1.000]），加到嵌入上模型就能分辨位置。手算只需 sin(1)、cos(1)、sin(2)、cos(2) 四个值。

**（2）RoPE 旋转 45° 的小例。** 玩具设定：某维度对的旋转速率是“每前进 1 个位置转 45°”。取向量 $q = [1, 0]$（位置 1 的 Q）和 $k = [1, 0]$（位置 2 的 K），初始都指向正东：

- Q 旋转 $1 \times 45° = 45°$：$[\cos 45°, \sin 45°] = [0.707, 0.707]$
- K 旋转 $2 \times 45° = 90°$：$[0, 1]$
- 点积：$0.707 \times 0 + 0.707 \times 1 = 0.707 = \cos(90° − 45°) = \cos 45°$

**关键一步（平移不变性）**：把两个位置都往后挪 30°——Q 转 75° 得 $[\cos 75°, \sin 75°] = [0.259, 0.966]$，K 转 120° 得 $[\cos 120°, \sin 120°] = [−0.5, 0.866]$，点积 $= 0.259 \times (−0.5) + 0.966 \times 0.866 = −0.129 + 0.837 = 0.707$。**还是 0.707**——绝对位置变了、相对距离（45°）没变，分数就不变。这就是“注意力分数只依赖相对距离”的全部秘密：$\cos$ 差角公式。

**（3）ALiBi 的斜率。** 8 个头的斜率按 $2^{-8(h+1)/8}$ 依次是 0.5、0.25、0.125、0.0625……第 1 个头最“近视”。拿第 1 个头（$m_h = 0.5$）算一例：位置 0 对位置 0 和位置 1 的原始分数是 $[2, 1]$：

- 无 ALiBi：softmax([2, 1])——$e^2 \approx 7.39$、$e^1 \approx 2.72$，权重 $[0.731, 0.269]$
- 有 ALiBi：先减 $m_h \cdot |i-j|$ 得 $[2 − 0,\ 1 − 0.5] = [2, 0.5]$，softmax——$e^{0.5} \approx 1.65$，权重 $[0.818, 0.182]$

远处 token 的权重从 0.269 压到 0.182——距离越远折扣越重，且**不打死**（不是掩码，只是打折）。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

原课四个函数一一对应 0.3 的三幅图：`sinusoidal(N, d)` 生成 0.3 表格那样的矩阵；`apply_rope(x, pos)` 做二维平面上的旋转（0.3 第 2 幅的矩阵版）；`alibi_bias` 生成每头的 $-m_h|i-j|$ 偏置矩阵（0.3 第 3 幅的批量版）；第 4 步是一个数值实验：验证 RoPE 平移不变性（0.3 已手算过它的 45° 版）。注意三个函数都只用标准库 `math`，跑起来零依赖。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么“打乱输入行、输出行同样打乱”就说明注意力没有顺序概念？

**参考答案：** 注意力分数是 $q_i \cdot k_j$ 的两两内积，输入行打乱后，所有内积只是换了个排列位置、数值不变；softmax 与加权求和也逐行独立。输出的第 i 行始终是“原来第 i 个词”的上下文混合——模型对“谁排第几”完全无感。

**问题2：** 正弦编码为什么外推差？

**参考答案：** 它是逐位置预先算好的固定表格：训练只见过 0–2047 的相位组合，位置 2048 的 sin/cos 值模型从未学过如何解读（表格之外没有定义）。RoPE/ALiBi 编码的是距离的函数，新位置只是“新的距离值”，天然可以延续。

**问题3：** RoPE 为什么只旋转 Q 和 K，不旋转 V？

**参考答案：** 位置信息的作用发生在**打分**环节（Q·K），旋转 Q、K 让分数带上相对距离；V 是被取用的内容本身，旋转它只会破坏内容、且对分数毫无贡献。ALiBi 更极端：连嵌入都不动，直接改分数。

**问题4：** 0.3 的例子里若把两个词的位置从 (1, 2) 改成 (11, 12)，RoPE 的注意力分数怎么变？

**参考答案：** 不变。相对距离仍是 1 个位置（45°），由 0.3 第 2 幅的平移不变性，两个点积都等于 $\cos 45° = 0.707$。这正是 RoPE 的卖点：分数是相对距离的函数。

能讲清这四个问题后，进入正文。

---

## 一、这一课解决什么问题？

缩放点积注意力对顺序是盲的：$\mathrm{softmax}(QK^\top/\sqrt{d})V$ 由两两相似度算出，打乱 $X$ 的行，输出的行同样打乱。注意力内部没有任何东西在乎位置。

对词袋模型这不是 bug；对语言、代码、音频、视频——一切**顺序承载意义**的东西——是致命的。修法是把位置注入嵌入或分数，三个时代的答案：

1. **绝对正弦**（Vaswani 2017）：把 sin/cos 加到嵌入上。简单、无需学习、但超出训练长度外推很差。
2. **RoPE 旋转位置编码**（Su 2021）：按正比于位置的角度旋转 Q 和 K，直接在点积里编码**相对**位置。2026 年的主流。
3. **ALiBi 线性偏置**（Press 2022）：完全不碰嵌入，按距离给每个头的注意力分数加线性惩罚。长度外推极佳。

到 2026 年，基本所有前沿开源模型都用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型用 ALiBi 或其现代变体。绝对正弦已成为历史（原课对生态的描述）。

## 二、核心概念与推导

### 2.1 排列不变性：为什么必须补位置

把 0.1 节的思想实验变成一句话：注意力的每个输出分量只依赖“内容相似度”，而内容相似度在行打乱后保持不变。注意：这**不是**实现缺陷，而是点积打分的数学性质——所以修法不是“改注意力”，而是“改输入或分数”，让位置成为内容的一部分。

### 2.2 绝对正弦编码（2017）

预计算固定矩阵 $PE$，形状 $(\text{max\_len}, d_{\text{model}})$：

$$PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i/d_{\text{model}}}\right),\qquad PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d_{\text{model}}}\right)$$

然后在注意力之前做 $X' = X + PE[:N]$。每个维度是一条不同频率的正弦波（0.3 表格里维度对 (0,1) 快、(2,3) 慢），模型学着从相位模式里读出位置。失败模式：超过 max_len 就没有定义——只见过 0–2047 的模型对位置 2048 该是什么一无所知。

### 2.3 RoPE（2021）

旋转 Q 和 K 向量（**不是嵌入、不是 V**）。对每对维度 $(2i, 2i+1)$：

$$\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix},\qquad \theta_i = \text{base}^{-2i/d_{\text{head}}},\ \ \text{base}=10000$$

对位置 $pos_k$ 的键做同样的旋转。于是点积 $q'_m \cdot k'_n$ 只依赖 $(m-n)$——**注意力分数只取决于相对距离**，尽管旋转本身是用绝对位置驱动的。漂亮的把戏。0.3 第 2 幅用 45°/75°/120° 手算了这个性质的两步验证。

延展：base 可以缩放（NTK-aware、YaRN、LongRoPE）来免重训外推更长上下文。Llama 3 用这条路把上下文从 8K 扩到 128K（原课参考）。

### 2.4 ALiBi（2022）

跳过嵌入花样，直接偏置分数：

$$\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|$$

$m_h$ 是每个头的斜率（如 $1/2^{8h/H}$，0.3 第 3 幅的 0.5 就是 8 头时的第 1 个头）。近处加分、远处打折；无训练期成本；论文显示长度外推胜过正弦、在原训练长度上与 RoPE 相当（原课转述的论文结论）。

### 2.5 2026 年怎么选（原课表格的中文版）

| 变体 | 外推 | 训练成本 | 使用者 |
|------|------|----------|--------|
| 绝对正弦 | 差 | 免费 | 初版 Transformer、早期 BERT |
| 学习式绝对 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 好（配合缩放） | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 极好 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 极好 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出的原因（原课总结）：不改架构就能插入注意力、编码相对位置、base 给了长上下文微调一个干净的旋钮。

### 2.6 用小例子把概念串起来（补充）

三种方法在 0.3 的数字上各显本色：正弦表给位置 1 的指纹是 [0.841, 0.540, 0.010, 1.000]——绝对位置的“身份证”；RoPE 不发身份证，只在打分时把 Q、K 各转一个角度，让 45° 的相对夹角决定 0.707 的分数；ALiBi 连角度都不转，直接按距离打 0.5/步 的折。三种哲学：**加在身上 / 转在手上 / 记在账上**。

再用同一个问题考三种方法——“位置 100 和位置 101 的关系”：正弦要分别查两张表再相减（模型自己学出这层减法）；RoPE 天然只剩 1 个位置的差角（转角相消）；ALiBi 直接规定折扣 = 1 步 × 斜率。三者对“绝对 vs 相对”的立场，决定了它们外推能力的高低：越不依赖绝对坐标的，越不怕没见过的坐标。

## 三、操作前准备

Build It 只用 Python 标准库 `math`（核对：原课三个函数片段都用到 `math.sin/cos` 但课程页未显示 `import math`，运行前需自行导入；本篇第五节入口已包含）。无第三方依赖；Use It 需要 `transformers` 库（按需安装，本次未替你安装或下载模型）。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：正弦编码

原课称这是“4 行计算”，加在第一层注意力之前的嵌入矩阵上。

~~~python
# 本块中文说明：生成 (N, d) 的正弦位置编码表。
import math

def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
~~~

（核对：`import math` 为本篇补注——原课片段直接使用 math 未显示导入。另一处边界：循环 `range(d // 2)` 只填偶数对，若 $d$ 为奇数，最后一维会保持 0.0；正弦编码惯例取偶数维，不算 bug 但用奇数维时会静默少一维信息。）

### 4.2 第 2 步：对 Q、K 应用 RoPE

RoPE 原地操作 Q 和 K，逐维度对旋转。

~~~python
# 本块中文说明：按位置旋转一个向量——RoPE 的核心操作。
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
~~~

原课强调的关键：对位置 $m$ 的 Q 和位置 $n$ 的 K 应用**同一个函数**。它们的点积在每一对坐标上拾取一个 $\cos((m-n)\cdot\theta_i)$ 因子——注意力“免费”学到相对位置。对照 0.3 第 2 幅：那个 2×2 旋转矩阵在本函数里就是 `c, s, -s, c` 四个数。

### 4.3 第 3 步：ALiBi 的斜率与偏置

~~~python
# 本块中文说明：为每个头生成 (seq_len, seq_len) 的线性距离偏置矩阵。
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
~~~

（核对：注释里的公式写 $h=1..n_{\text{heads}}$、代码里 `h+1` 从 1 取到 $n_{\text{heads}}$，两者一致；8 头时斜率正是 0.5、0.25、……、0.003906，与 0.3 第 3 幅吻合。偏置在 softmax **之前**加，注释也写明了这一点。）

### 4.4 第 4 步：验证 RoPE 的相对距离性质

（说明：本步原课为文字描述的数值实验，未展示完整代码。）取两个随机向量 `a, b`，先按 `(pos_a, pos_b)` 旋转、求点积；再按 `(pos_a + k, pos_b + k)` 旋转、再求点积。两个点积必须在浮点误差内相等——这个性质就是 RoPE 的全部意义：对绝对偏移不变，只有相对间隔起作用。（补充：0.3 第 2 幅的 45°→75°/120° 平移实验就是这个实验的玩具版，手算已验证点积同为 0.707。）

## 五、把代码运行起来：补充入口与检查

（补充）以下入口把原课函数与 0.3 的手算数字对上：

~~~python
# 本块中文说明：补充检查入口——0.3 三个手算例的断言。
import math

# （1）正弦表：d_model=4 时位置 1 的指纹
pe = sinusoidal(3, 4)
assert abs(pe[1][0] - math.sin(1)) < 1e-12        # 0.841
assert abs(pe[1][1] - math.cos(1)) < 1e-12        # 0.540
assert abs(pe[1][2] - math.sin(0.01)) < 1e-12     # 0.010
assert abs(pe[1][3] - math.cos(0.01)) < 1e-12    # 1.000
assert abs(pe[0][0] - 0.0) < 1e-12 and abs(pe[0][1] - 1.0) < 1e-12

# （2）RoPE 平移不变性：45°/步 的玩具速率，手算预期点积恒为 cos(45°)≈0.7071
def rot45(v, steps):                              # 每步转 45° 的玩具 RoPE
    ang = steps * math.pi / 4
    c, s = math.cos(ang), math.sin(ang)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)

q, k = (1.0, 0.0), (1.0, 0.0)
d1 = rot45(q, 1)[0] * rot45(k, 2)[0] + rot45(q, 1)[1] * rot45(k, 2)[1]
d2 = rot45(q, 4)[0] * rot45(k, 5)[0] + rot45(q, 4)[1] * rot45(k, 5)[1]  # 平移 +3 步
assert abs(d1 - math.cos(math.pi / 4)) < 1e-12 and abs(d2 - d1) < 1e-12

# （3）ALiBi：8 头斜率 + 0.3 的折扣例
sl = [2 ** (-8 * (h + 1) / 8) for h in range(8)]
assert sl[0] == 0.5
def sm(x):
    e = [math.exp(v) for v in x]
    return [v / sum(e) for v in e]
assert abs(sm([2.0, 1.0])[1] - 0.2689) < 1e-3     # 无偏置：远处权重 0.269
assert abs(sm([2.0, 0.5])[1] - 0.1824) < 1e-3     # ALiBi 后：压到 0.182
print("正弦 / RoPE / ALiBi 三例检查通过")
~~~

另一个自查（对照 4.4 步）：用 `apply_rope` 本体做平移实验——随机取 `a, b`，比较 `(pos_a, pos_b)` 与 `(pos_a+7, pos_b+7)` 的点积，应在 1e-9 量级内一致（浮点误差）。

## 六、Use It：使用现成工具

PyTorch 2.5+ 在 `torch.nn.functional` 里提供 RoPE 工具；多数生产代码用 `flash_attn` 或 `xformers`，RoPE 在注意力内核内部施加。原课给的是 HuggingFace 侧的查看方式：

~~~python
# 本块中文说明：从模型配置读取 RoPE 缩放设置。
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
~~~

（核对：该注释是原课展示的预期配置样例，实际运行需下载模型并接受许可；本篇未执行。）

原课列出的 2026 年长上下文技巧：**NTK-aware 插值**（4K 扩 16K+ 时把 base 缩放为 $base \times scale^{d/(d-2)}$）；**YaRN**（保注意力熵的更聪明插值，Llama 3.1 128K 采用）；**LongRoPE**（微软 2024 用进化搜索逐维挑缩放因子，Phi-3-Long 采用）；**位置插值+微调**（按扩展因子压缩位置再微调 1–5B token，出奇地有效——以上均为原课表述）。

## 七、失败模式、原文边界与工程注意

1. 正弦编码外推失败是结构性的：固定表格 + 训练分布之外无定义；不要指望“加大 max_len 表格”解决——模型仍没学过那些相位怎么读。
2. RoPE 只旋转 Q 和 K：旋转 V 没有意义（位置不参与内容混合），旋转嵌入本身也偏离原设计（补充：与 0.5 节问题 3 呼应）。
3. 调大 base 不是免费的：高频维度的分辨率被稀释，通常需要配合微调（YaRN 的存在就是为了把这件事做细）。
4. ALiBi 偏置与掩码一样必须在 softmax **之前**加：加在之后会破坏行的归一化（与[第02课](<02：从零实现自注意力——Q、K、V与缩放点积.md>)2.8 节同理）。
5. 不要叠加多种位置机制：正弦加嵌入 + RoPE 转向量同时用，等于给模型两套互相冲突的位置信号，通常掉点（补充；原课三方案是“或”不是“且”的关系）。
6. 学习式绝对编码（GPT-2 式）外推为零：没见过的位置向量不存在。
7. 奇数维 bug：`sinusoidal` 的 `range(d // 2)` 会让奇数 $d$ 的最后一维恒为 0；虽然惯例是偶数维，写通用代码时要么断言偶数、要么补上落单维度（核对，见 4.1）。
8. ALiBi 的斜率是按头固定的超参数不是学出来的：换头数要重算整组斜率（0.3 的 8 头序列换 16 头就不是同一串数了）。
9. 位置编码解决的是“顺序”，不解决“长度”：上下文变长后的内存墙是 $O(N^2)$ 注意力的问题，归[第12课](<12：KV缓存与Flash Attention——推理优化.md>)管（补充）。
10. RoPE 的 base 换算要写清楚公式再动手：NTK-aware 的 $base \times scale^{d/(d-2)}$ 里 $d$ 是**头维度**不是模型维度，代错维度会得到错误的新 base——改完务必用第 4 步的平移不变性实验回归检查注意力输出没有跑飞（补充，见第02课“核对”习惯的延续）。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/skill-positional-encoding-picker.md`：给定目标上下文长度、外推需求与训练预算，挑编码策略的技能。以下是本课的中文学习版自查模板（直接保存在本篇，不另建输出文件）：

~~~markdown
为新模型选位置编码时逐项自查：需要外推多远（训练长度 vs 推理长度，写两个数）？超长外推选 RoPE+缩放还是 ALiBi，依据是什么？RoPE 的话 base 取多少、要不要 YaRN/NTK 缩放（写出换算）？是插入既有架构还是从头训练（插入→RoPE 最省事）？所有“某模型用某方案”的生态描述注明出处；所有外推收益数字标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：把 `max_len=512, d=128` 的正弦 PE 矩阵画成热图，确认“维度越靠后条纹越宽”的模式。验收标准：热图上能肉眼指出高频端（前几维条纹密）与低频端（后几维条纹疏），并能用 0.3 表格的“维度对 (2,3) 比 (0,1) 慢 100 倍”解释为什么条纹变宽。
2. 原课 Medium：实现 NTK-aware RoPE 缩放；在 256 长度上训练一个小 LM，加与不加缩放分别测试 1024 长度，测困惑度。验收标准：给出两种设置的外推困惑度对比表；注明训练与测试长度、base 换算公式，数字为自己实测。
3. 原课 Hard：在同一个注意力模块里分别实现 ALiBi 与 RoPE，在 512 长度上训练 4 层 Transformer 做复制任务，测试时外推到 2048，比较退化程度。验收标准：两条外推性能曲线（512 与 2048 两点各两个方案）+ 一句话结论；对照 2.5 节表格说明结果是否符合“ALiBi/RoPE 外推好”的预期。

## 十、术语与完成标准

位置编码：任何把位置告诉模型的信号；正弦编码：几何频率的 sin/cos 加到嵌入、外推差；RoPE：按位置旋转 Q、K，点积编码相对距离；base：RoPE 频率旋钮，调大可延展上下文；ALiBi：分数减 $m_h|i-j|$，免嵌入、外推佳；斜率：ALiBi 每头的惩罚率 $2^{-8h/H}$；NTK-aware/YaRN/LongRoPE：三种 base 缩放/逐维插值技巧；外推：位置方案能否服务训练长度之外的位置。

完成检查：

- 能演示“打乱行输出跟着打乱”并解释为什么（排列不变性）。
- 能手算 0.3 的正弦表（pos=1 四个值）并指出哪对维度高频、哪对低频。
- 能手算 RoPE 45° 例的两次点积并说明平移不变性来自 $\cos$ 差角公式。
- 能算 ALiBi 的折扣（[2, 1] → [2, 0.5]，权重 0.269 → 0.182）。
- 能说出三方案分别“动嵌入、动 Q/K、动分数”中的哪一个。
- 能解释 base 为什么是长上下文的旋钮、以及调大它的代价。
- 能对同一组位置 (100, 101) 分别说出三种方法各会怎么处理（2.6 节末的思考题）。
- 能指出学习式绝对编码（GPT-2）外推为零的原因，并与正弦编码的“外推差”区分开（0.2 节术语）。

## 十一、验证范围与来源许可

原课 Build It 共 4 步：Step 1—3 的三个函数已逐字转录（仅添加中文注释与 `import math` 补注，未改逻辑）；Step 4 为原课文字描述的数值实验，本篇如实标注未虚构代码。Use It 的 transformers 代码保留，其注释里的 rope_scaling 值为原课预期样例、未在本机执行。2.5 节选型表与长上下文技巧列表为原课内容的中文转录。

本次验证记录（2026-11-06）：

- 抓取课程页成功（一次成功，无重试），内容以原文为准整理。
- 0.3 节三个手算例（正弦表 pos=0/1/2 全部取值、RoPE 45°/75°/120° 两次点积均为 0.7071、ALiBi 斜率序列与折扣前后权重 0.2689/0.1824）已用本机 Python 3.12（标准库 math）复核通过。
- 第五节补充断言已在本机实际执行通过（属 3×4 量级的玩具断言，不构成训练或性能实验）；第五节末的 `apply_rope` 平移实验同样本机验证通过。
- 正文 4 个 python 代码块全部通过 `ast.parse` 静态语法检查。
- 未做：训练小 LM、下载 Llama 权重、长上下文外推实验。

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
