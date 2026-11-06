# 04：激活函数——非线性、ReLU与饱和区

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/04-activation-functions) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/04-activation-functions/docs/en.md)。原题：Activation Functions。

## 零、初学者先读：没有它，一百层网络等于一层

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第03课](<03：从零实现反向传播——链式法则与计算图.md>)里你已经会算梯度了，本课回答的是另一个问题：网络里那些“非线性零件”为什么必须存在、选错了会怎么死。

### 0.1 先理解这一课在做什么

把两层线性变换叠起来：$y = W_2(W_1 x + b_1) + b_2$。展开后就是 $y = (W_2 W_1)x + (W_2 b_1 + b_2)$——一层线性变换。叠一百层也一样：全部塌缩成一次矩阵乘法。这意味着没有非线性零件的“深度网络”和单层网络表达力相同，连 XOR、螺旋数据这种弯曲边界都学不会——深度是幻觉。激活函数（activation function）就是插在层与层之间的那个“弯折零件”，它让叠加不再可以合并，深度才真正有意义。但零件选错了会更糟：梯度在层层传递中消失（sigmoid）、神经元永久死亡（ReLU 的负区）。本课 = 认识这些零件 + 学会诊断它们的病。

### 0.2 本课术语先翻成白话

- 激活函数（activation function）=加在每个神经元输出上的非线性函数，负责“弯折”
- 线性（linear）=能写成 $Wx+b$ 形式的关系；线性叠加可合并，所以“纯线性深度”无意义
- sigmoid=把任意实数压到 (0,1) 的 S 形函数，$1/(1+e^{-x})$
- 饱和区（saturation）=函数图像两端“躺平”的区域，那里导数趋近 0，梯度过不去
- 消失梯度（vanishing gradient）=每过一层导数乘一个小于 1 的数，多层相乘后梯度趋近 0
- 爆炸梯度（exploding gradient）=每层乘的数大于 1，梯度指数增大，训练不稳定
- tanh（双曲正切）=sigmoid 的“居中版”，输出范围 (−1,1)
- ReLU（Rectified Linear Unit）=max(0, x)：负数归零、正数原样通过
- 死亡神经元（dead neuron）=ReLU 的输入长期为负 → 输出恒 0 → 梯度恒 0 → 永不更新
- Leaky ReLU=负区给一个小斜率 α（典型 0.01）而不是 0，让死神经元有救
- GELU（Gaussian Error Linear Unit）=光滑门控激活，BERT/GPT 等 transformer 的默认
- Swish / SiLU=$x \cdot \text{sigmoid}(x)$，自动搜索发现，EfficientNet 在用
- softmax=把一串原始分数（logit）变成和为 1 的概率分布，多分类输出层专用
- logit（ logits）=最后一层未经 softmax/sigmoid 的原始输出分数

### 0.3 用小数字走一遍

**ReLU 家族。** 输入 2.5：relu(2.5)=2.5，导数 1；输入 −3：relu(−3)=0，导数 0——这就是“负区无梯度”。换成 Leaky ReLU（α=0.01）：leaky(−3)=0.01×(−3)=−0.03，导数 0.01——输出很小但不是 0，梯度还有一丝，神经元还能被拉回来。

**sigmoid 与饱和。** sigmoid(0)=1/(1+1)=0.5，导数 0.5×0.5=0.25（这是它的最大导数）。sigmoid(2)：$e^{-2}≈0.135$，所以 ≈1/1.135≈0.881，导数 0.881×(1−0.881)=0.881×0.119≈0.105。到 x=4 时 sigmoid(4)≈0.982，导数 ≈0.0176——已经接近躺平。十层 sigmoid 网络最坏情况梯度乘 $0.25^{10}≈0.00000095$，不到原来的百万分之一：第一层基本收不到信号。

**tanh。** tanh(1)：$e^2≈7.389$，(7.389−1)/(7.389+1)=6.389/8.389≈0.762，导数 $1-0.762^2≈1-0.58=0.42$。tanh 在 0 处导数为 1，是 sigmoid 最大值的 4 倍，但两端同样饱和。

**GELU 与 Swish 的“允许小负值”。** gelu(1)≈0.5×1×(1+tanh(0.798×(1+0.045)))≈0.5×(1+0.683)≈0.841；gelu(−1)≈−0.159——注意是小的负值而非硬归零。swish(−1)=−1×sigmoid(−1)≈−1×0.269=−0.269；swish(1)≈0.731。

**softmax。** 两个 logit [2, 0]：$e^2≈7.389$，$e^0=1$，和为 8.389，输出 [0.881, 0.119]——都大于 0、和为 1，大的那个分到大概率。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：先看 Step 1 的 12 个函数（7 个激活 + 5 个导数 + softmax），它们都是“一个浮点进、一个浮点出”，逐个和 0.3 节的数字对上；再看 Step 2 的 gradient_scan（数一数每个激活在 [−5,5] 上有多少点导数近 0）；Step 3 把信号过 10 层看幅度衰减；Step 4 统计 ReLU 网络里多少神经元从不激活；Step 5 用三种激活训练同一个圆环分类网络比收敛速度。核心数据结构全是普通浮点列表，没有矩阵库。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么“叠一百层线性层”没有意义？

**参考答案：** 矩阵乘法可结合：先乘 $W_1$ 再乘 $W_2$ 等于一次乘 $W_2W_1$。任意多层线性变换都能合并成 $y=Ax+c$ 一层，表达力和单层相同，参数却更多——深度没有换来任何非线性表达能力。

**问题2：** sigmoid 的最大导数是多少？为什么十层 sigmoid 会“消失梯度”？

**参考答案：** 0.25（在 x=0 处）。反向传播中各层导数连乘，每层至多乘 0.25，十层最多 $0.25^{10}≈9.5\times10^{-7}$，早期层梯度近乎为零、权重几乎不更新，网络看起来在学（后层损失在降）但前层是冻住的。

**问题3：** 什么是死亡神经元？Leaky ReLU 为什么能救？

**参考答案：** ReLU 神经元的加权输入长期为负时，输出恒 0、导数恒 0，梯度永远传不回来，参数永久停更。Leaky ReLU 给负区一个小斜率 α，输出和导数都不再是 0，还剩一条细细的梯度通道让权重能被修正。GELU 则用光滑门控从根上避免硬归零。

能对上这组小数字后，进入下面的推导与代码。

---

## 一、这一课解决什么问题？

上一课你学会了反向传播：梯度沿链式法则逐层回传。但链式法则里每一层都要乘一次激活函数的导数——如果这个导数长期小于 1，信号就指数式衰减。本课解决两个问题：（1）没有激活函数，深度是幻觉（线性塌缩证明）；（2）激活选错，深度是毒药（sigmoid 消失梯度、ReLU 死亡神经元）。原课的学习目标包括：从零实现 7 种激活及其导数、用 10+ 层实验测量梯度消失、检测死亡神经元、并学会按架构（transformer/CNN/RNN/输出层）选激活函数。

## 二、核心概念与推导

### 2.1 线性塌缩：深度为什么需要非线性

线性层计算 $f(x)=Wx+b$。叠两层并代入：$y = W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2) = Ax+c$。一层。在两层之间插入非线性 $g()$：$h=g(W_1x+b_1)$，$y=W_2h+b_2$——代入不再能合并，$W_2\,g(W_1x+b_1)+b_2$ 无法化简成一次矩阵乘法。从这层开始，每加一个带激活的层，表达能力都在真实增加。（补充：这就是 [第01课](<01：感知机——权重、偏置与更新规则.md>)里单层感知机学不了 XOR 的解药。）

### 2.2 sigmoid：元老与它的病

$$\text{sigmoid}(x)=\frac{1}{1+e^{-x}},\qquad \text{sigmoid}'(x)=s(1-s)$$

输出范围 (0,1)，光滑可导，天然像概率。两个问题：一是最大导数只有 0.25（x=0 处），十层连乘最坏 $0.25^{10}\approx 9.5\times10^{-7}$——消失梯度；二是输出恒正，导致对权重的梯度符号单一，梯度下降来回“之字形”震荡。深 sigmoid 网络实际上训不动。

### 2.3 tanh：居中版，病轻但没好

$$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}},\qquad \tanh'(x)=1-\tanh^2(x)$$

输出 (−1,1)、零中心（zero-centered），消除 sigmoid 的震荡问题；0 点最大导数 1.0，是 sigmoid 的 4 倍。但两端依旧饱和，深了照样压扁梯度，只是没那么狠。

### 2.4 ReLU：让深度学习成为可能的那个函数

$$\text{relu}(x)=\max(0,x),\qquad \text{relu}'(x)=\begin{cases}1 & x>0\\ 0 & x\le 0\end{cases}$$

正区导数恒等于 1：梯度原封不动穿过任意多层。深度网络由此变得可训练。代价是死亡神经元：偏置过负或初始化不巧，某神经元的输入长期为负 → 输出恒 0 → 梯度恒 0 → 永久死亡。原课参考：训练中一个 ReLU 网络可有 10—40% 的神经元死亡。历史注记：Nair & Hinton 2010 年把它带火，函数本身可追溯到 1969 年 Fukushima 的工作。

### 2.5 Leaky ReLU：最简单的修法

$$\text{leaky\_relu}(x)=\begin{cases}x & x>0\\ \alpha x & x\le0\end{cases},\qquad \alpha\approx 0.01$$

负区给个小斜率，死神经元保留一条细细的梯度通道，可以复活。

### 2.6 GELU：现代 transformer 的默认

$$\text{gelu}(x)=x\cdot\Phi(x)\approx 0.5x\left(1+\tanh\left(\sqrt{2/\pi}\,(x+0.044715x^3)\right)\right)$$

$\Phi(x)$ 是标准正态的累积分布函数。直觉：用“输入为正的概率”给输入加权——大概率为正就几乎全通过，大概率为负就几乎压到 0，中间平滑过渡。处处光滑、允许小的负值（不像 ReLU 硬归零）、没有死亡神经元问题、梯度流更好。Hendrycks & Gimpel 2016 年提出；BERT、GPT 和大多数现代 transformer 默认用它。

### 2.7 Swish / SiLU：自动搜索出来的激活

$$\text{swish}(x)=x\cdot\text{sigmoid}(x)$$

Ramachandran 等 2017 年用自动搜索在激活函数空间里“搜”出来的——用神经网络设计神经网络零件。和 GELU 一样光滑、非单调（non-monotonic）、允许小负值；区别在门控用 sigmoid 而非高斯 CDF。实践中与 GELU 性能几乎相同：EfficientNet 等视觉模型用 Swish，语言模型 GELU 占主导。

### 2.8 softmax：输出层的激活

不在隐藏层用。把一串 logit 变成概率分布：

$$\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}$$

每项都在 0 到 1 之间、总和为 1，是多分类输出层的标准配置。它像“可导版的 argmax”：最大的 logit 拿最大概率，但保留相对置信度信息，能被梯度下降优化。

### 2.9 选哪个：一张对照表（补充）

| 激活 | 输出范围 | 最大导数 | 主要风险 | 典型场景 |
|---|---|---|---|---|
| sigmoid | (0,1) | 0.25 | 消失梯度、输出非零中心 | 二分类输出层、门控 |
| tanh | (−1,1) | 1.0 | 两端饱和 | RNN/LSTM 隐藏状态 |
| ReLU | [0,∞) | 1（正区） | 死亡神经元 | CNN/MLP 隐藏层 |
| Leaky ReLU | (−∞,∞) | 1（正区） | 极少 | 抢救死神经元的备选 |
| GELU | 约(−0.17,∞) | ~1 | 计算稍贵 | transformer 隐藏层 |
| Swish | 约(−0.28,∞) | ~1 | 计算稍贵 | EfficientNet 等视觉模型 |
| softmax | (0,1)、和为 1 | — | 数值溢出需减最大值 | 多分类输出层 |

原课的选择口诀：transformer/NLP 隐藏层用 GELU；CNN 隐藏层用 ReLU 或 Swish；RNN 用 tanh；二分类输出用 sigmoid；多分类输出用 softmax；回归输出用线性（不加激活）。

## 三、操作前准备

Build It 全程只用 Python 标准库（math、random），无第三方依赖；Use It 需要 PyTorch：

~~~bash
python -m pip install torch
~~~

先激活你已有的 Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或运行任何训练。代码按定义顺序执行即可，各步之间只有函数定义的依赖。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：实现全部激活函数与导数

每个函数一个浮点进一个浮点出；导数函数同样吃输入、回梯度。注意 sigmoid 里把 x 裁剪到 ±500 防止 math.exp 溢出；softmax 先减去最大值再做指数（同样的数值稳定技巧）。

~~~python
# 本块中文说明：7 种激活 + 各自导数 + softmax 的纯 Python 实现。
import math

def sigmoid(x):
    x = max(-500, min(500, x))  # （补充注释）裁剪防止 exp 溢出
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # s(1-s)：只需算一次前向即可得导数

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    # tanh 近似式：与 erf 精确版差距在实用中可忽略
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    # 用精确的高斯 CDF/PDF 推导的导数：phi(x) + x*pdf(x)
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)  # 减最大值：防止大 logit 直接 exp 溢出
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
~~~

（核对：gelu 前向用 tanh 近似式、gelu_derivative 用精确 erf 表达式，两者来自原课，近似误差在小数点后几位，不影响教学结论。）

### 4.2 第 2 步：可视化梯度死在哪里

在 [−5,5] 上均匀取 100 个点，数每个激活的导数有多少点接近 0（阈值 0.01）。

~~~python
# 本块中文说明：扫描每个激活的导数，统计“死区”比例。
def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

gradient_scan("Sigmoid", sigmoid_derivative)
gradient_scan("Tanh", tanh_derivative)
gradient_scan("ReLU", relu_derivative)
gradient_scan("Leaky ReLU", leaky_relu_derivative)
gradient_scan("GELU", gelu_derivative)
gradient_scan("Swish", swish_derivative)
~~~

预期（可推理）：sigmoid 两端大面积饱和；tanh 饱和区更窄；ReLU 恰好一半（负区）导数为 0；Leaky ReLU 负区导数 0.01——按此阈值仍算 near-zero 但有梯度可救；GELU/Swish 无硬死区。

### 4.3 第 3 步：消失梯度实验

把一个信号过 N 层，观察激活幅度逐层变化。

~~~python
# 本块中文说明：10 层前向，观察激活幅度衰减。
import random

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_layers} layers:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_gradient_experiment(sigmoid, "Sigmoid")
vanishing_gradient_experiment(relu, "ReLU")
vanishing_gradient_experiment(gelu, "GELU")
~~~

（核对：本实验把单次激活的结果复制成下一层的全部输入（`[activated] * n_inputs`），是教学简化，不是真实的多神经元前向；它展示的是激活输出幅度的衰减趋势。sigmoid 输出被压进 (0,1)，重复作用后幅度趋小；ReLU 对正输入原样通过、不会越乘越小。）

### 4.4 第 4 步：死亡神经元检测器

构造 20 个 ReLU 神经元，喂 1000 个随机输入，统计每个神经元的激活次数。

~~~python
# 本块中文说明：统计多少神经元从未激活（死）、多少几乎不激活（弱）。
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()
~~~

注意区分“死”（从未激活，梯度恒 0）与“弱”（激活率 <5%）。本随机初始化下大概率没有真正死亡的神经元（核对：本篇做静态检查时顺带实际运行了本块，20 个神经元全部健康、死亡率 0.0%，与该预期一致；原课参考的 10—40% 死亡率说的是训练过程中的真实网络，不是这个静态实验）。

### 4.5 第 5 步：Sigmoid vs ReLU vs GELU 训练对比

同一个两层网络（隐藏层 8 个神经元）在圆环数据集上分别用三种激活训练 200 轮，比较收敛速度。圆环数据：点落在半径平方 <1.5 的圆内为类别 1，圆外为类别 0。

~~~python
# 本块中文说明：同一网络换三种激活，比收敛速度。
def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)  # 平方误差 × 输出 sigmoid 导数

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")
~~~

（核对：本网络实际最小化的是平方误差（`total_loss += (pred - y) ** 2`），`backward` 里 `d_out` 也是“平方误差×输出层 sigmoid 导数”，不是交叉熵——这正是 [第05课](<05：损失函数——MSE、交叉熵与选择.md>)要展开的损失选择问题。此外这是逐样本更新（batch size 1）的 SGD。）原课参考的结论方向：ReLU 与 GELU 收敛明显快于 sigmoid；具体数值未在课程页展示，本篇不虚构。

## 五、把代码运行起来：补充入口与检查

（补充）原课各步是连续脚本，按 4.1→4.5 顺序执行即可。以下补充的最小检查不依赖随机数，可手算复核：

~~~python
# 本块中文说明：补充的手算复核入口。
import math

assert sigmoid(0) == 0.5
assert abs(sigmoid_derivative(0) - 0.25) < 1e-12
assert relu(-3.0) == 0.0 and relu_derivative(-3.0) == 0.0
assert relu(2.5) == 2.5 and relu_derivative(2.5) == 1.0
assert abs(leaky_relu(-3.0) - (-0.03)) < 1e-12   # 0.01 × (-3)
assert abs(tanh_act(1) - 0.7615941) < 1e-6        # tanh(1)≈0.762
assert abs(tanh_derivative(1) - (1 - 0.7615941**2)) < 1e-6  # ≈0.42
assert abs(swish(-1.0) - (-0.2689414)) < 1e-6     # -1×sigmoid(-1)≈-0.269
probs = softmax([2.0, 0.0])                        # ≈[0.881, 0.119]
assert abs(sum(probs) - 1.0) < 1e-12 and abs(probs[0] - 7.389056/8.389056) < 1e-6
# gelu(-1)≈-0.159（小负值，非硬归零），gelu(1)≈0.841
assert -0.17 < gelu(-1.0) < -0.15 and 0.83 < gelu(1.0) < 0.85
print("all activation checks passed")
~~~

补充检查方法：运行 4.2 的 gradient_scan，对照 2.9 表格解释每个死区比例；运行 4.3 后把 sigmoid 与 ReLU 的逐层幅度抄成两列，观察 sigmoid 列是否单调趋小。训练对比（4.5）未在本机执行，其结论标为原课参考。

## 六、Use It：使用现成工具

PyTorch 把这些激活同时提供为函数式（F.xxx）与模块式（nn.Xxx）两种形态：

~~~python
# 本块中文说明：PyTorch 中的激活函数用法。
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)  # Swish 在 PyTorch 里叫 silu

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)
~~~

原课的默认口诀：transformer 隐藏层 GELU；CNN 隐藏层 ReLU；分类输出层 softmax；回归输出层不加激活；要概率输出用 sigmoid。先按默认走，有证据再改。RNN/LSTM 用 tanh 做 hidden state、sigmoid 做门；如果你的 ReLU 网络在死神经元，换成 GELU；没有特别理由不要专门用 Leaky ReLU——GELU 既解决死亡问题又给更好的梯度流。

## 七、失败模式、原文边界与工程注意

1. 消失梯度是“静默失败”：后层在学、前层冻住，损失曲线看起来正常。诊断方法就是 4.2/4.3 的扫描与逐层幅度实验。
2. 死亡神经元同样是静默的：网络不报错，只是容量缩水。原课参考：训练中 10—40% 的 ReLU 神经元可能死亡。诊断靠 4.4 那种激活计数。
3. sigmoid 输出恒正导致之字形震荡；tanh 零中心可避免。深层隐藏层不要用 sigmoid（补充：它属于输出层和门控）。
4. 4.3 的实验是教学简化（复制单值当下一层输入），不能当作严格的梯度测量；严格的版本要真的反向传播逐层记录梯度范数（原课练习 4 就是这个方向）。
5. gradient_scan 的 0.01 阈值是人为选择，换个阈值比例就变；结论看趋势（谁的死区大），不看具体百分比。
6. softmax 必须先减最大值再 exp，否则大 logit 直接溢出为 inf（原课代码已内置，自己复刻时别漏）。
7. 不要在隐藏层用 softmax：强行让每层输出和为 1 会破坏表示；它只属于输出层（补充）。
8. 原文边界：原课说“GELU 的最小值约 −0.17”，这是近似式的性质，精确 $\Phi$ 版本略有差异，不影响使用。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/prompt-activation-selector.md`（一个帮你为任意架构挑激活函数的可复用提示词）。本篇改写成自查模板，直接嵌在这里，不另建输出文件：

~~~markdown
检查激活函数选择：说明网络每个位置（各隐藏层、输出层）用的是哪种激活及理由；隐藏层是否避免了 sigmoid；输出层是否按任务选择（二分类 sigmoid / 多分类 softmax / 回归线性）；是否检查过死神经元比例或逐层梯度幅度；所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：实现 PReLU（Parametric ReLU），负斜率 α 为可学习参数，在圆环数据上与固定 Leaky ReLU 对比。验收标准：α 在训练中确实被梯度更新（打印其变化轨迹），并能解释“学习到的 α 与固定 0.01 谁更合适取决于数据”。
2. 原课：把消失梯度实验改成 50 层，画出 sigmoid、tanh、ReLU、GELU 每层的幅度曲线。验收标准：能指出每种激活的信号大约在第几层“有效归零”，且能解释为什么 ReLU 不归零（正区导数恒 1）。
3. 原课：实现 ELU（$\text{elu}(x)=x$ 若 $x>0$，$\alpha(e^x-1)$ 若 $x\le0$），在同一网络上与 ReLU 比死亡率。验收标准：能展示 ELU 负区输出饱和到 −α 而非 0、导数不为 0，死亡率应低于 ReLU。
4. 原课：做一个“梯度健康监视器”：训练中每个 epoch 计算各层平均梯度幅度，低于 0.001 或高于 100 就告警。验收标准：人为把隐藏层换成 sigmoid 后监视器能触发低梯度告警。
5. 原课：把训练对比换成 [第01课](<01：感知机——权重、偏置与更新规则.md>)的 XOR 数据。哪种激活在 XOR 上收敛最快？为什么与圆环结果可能不同？验收标准：给出排序并能从“两类激活的可分性/梯度流”角度解释差异。

## 十、术语与完成标准

激活函数：加在神经元输出上的非线性变换，打破线性叠加的可合并性；sigmoid：$1/(1+e^{-x})$，范围 (0,1)，最大导数 0.25；tanh：范围 (−1,1)、零中心，最大导数 1；ReLU：$\max(0,x)$，正区梯度恒 1，风险是死亡神经元；Leaky ReLU：负区斜率 α 的 ReLU；GELU：$x\Phi(x)$，光滑门控，transformer 默认；Swish/SiLU：$x\cdot\text{sigmoid}(x)$；softmax：logit 归一化为概率分布；饱和区：导数趋 0 的平坦区；消失/爆炸梯度：导数连乘导致的指数衰减/增长；logit：输出层归一化前的原始分数。

完成检查：

- 能手算 relu/leaky/sigmoid/tanh/softmax 在具体小数上的输出与导数（0.3 节算例）。
- 能写出“两层线性可合并成一层”的推导并解释其后果。
- 能说出 sigmoid 的两个缺陷（最大导数 0.25、输出非零中心）及各自的症状。
- 能解释死亡神经元的成因链（输入恒负→输出 0→梯度 0→永不更新）与两种解法。
- 能背出选型口诀：transformer→GELU、CNN→ReLU、RNN→tanh、二分类→sigmoid、多分类→softmax、回归→线性。

## 十一、验证范围与来源许可

原课 Build It 共 5 个分段，代码已逐字转录嵌入本篇（仅按本篇格式添加中文说明与注释，未改逻辑）；Use It 的 PyTorch 代码同样保留。原课页面未展示任何运行输出，本篇未虚构“原课参考”数值；第五节标注“可手算复核”的断言均对应 0.3 节的算例。本次未执行训练或下载任何数据/模型。写作完成后，将正文各 python 代码块抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 逐一做静态语法检查，结果见下方记录。

本次验证记录（2026-11-06）：
- 抓取课程页一次成功，以原文为准整理；未使用凭记忆的内容。
- 正文 7 个 python 代码块（4.1—4.5、第五、六节）全部通过 `ast.parse` 静态语法检查；第五节的手算断言（sigmoid/tanh/relu/leaky/swish/softmax/gelu 的具体数值）在临时脚本中实际执行复核通过（属玩具断言检查，不构成训练或性能实验）；4.2—4.4 的演示脚本在检查时顺带实际运行，梯度扫描、十层幅度实验、死亡神经元检测（20/20 神经元全部健康、死亡率 0.0%）与文中预期一致。
- 未做：4.5 的训练对比、50 层实验、任何超参搜索。

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
