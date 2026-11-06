# 08：权重初始化——零、随机、Xavier与Kaiming

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/08-weight-initialization) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/08-weight-initialization/docs/en.md)。原题：Weight Initialization and Training Stability。

## 零、初学者先读：为什么网络在训练开始前就可能已经死了

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。上一课（[第07课](<07：正则化——L2、Dropout与早停.md>)）讲怎么防止训过头，本课回到训练的第 0 秒——权重从哪里出发；下一课（[第09课](<09：学习率调度——warmup与衰减策略.md>)）讲训练过程中步子怎么变。

### 0.1 先理解这一课在做什么

想象一个 512 人的合唱团，指挥喊“开始”，结果所有人张嘴唱的是完全同一个音——你花了 512 个人的钱，听到的却是 1 个人。这就是全零初始化：所有神经元算出同样的数、收到同样的梯度、做同样的更新，训 10000 轮后隐藏层里还是 512 个克隆。另一种灾难是音响系统逐级放大：每层把信号放大一点，10 层后数值大到溢出；或者逐级静音：每层把信号压小一点，10 层后信号消失。本课做的事只有一件：给权重选一个“出生点”，让信号穿过 50 层既不爆炸也不消失，而且每个神经元从不同的地方出发。

### 0.2 本课术语先翻成白话

- 权重初始化（weight initialization）=训练开始前给权重设的初始值；不是随便给，而是有公式的策略
- 对称性（symmetry）=层内所有神经元权重相同、输出相同、梯度相同，永远学成 clones 的困境
- 对称破坏（symmetry breaking）=用随机初始化让每个神经元从不同起点出发
- fan-in（扇入）=一个神经元接收的输入个数（上一层宽度）
- fan-out（扇出）=一个神经元往下游连接的个数（下一层宽度）
- Xavier/Glorot 初始化=方差取 2/(fan_in+fan_out)，配 sigmoid/tanh
- Kaiming/He 初始化=方差取 2/fan_in，配 ReLU/GELU
- 方差传播（variance propagation）=研究每层输出方差是输入方差的多少倍
- 残差缩放（residual scaling）=GPT-2 把残差层的投影权重乘 1/√(2N)，N 为层数
- 死网络（dead network）=梯度全零或激活全饱和，怎么训都不动的网络
- 爆炸激活（exploding activations）=方差逐层指数放大，最终溢出为 inf

### 0.3 用小数字走一遍

**A. 全同初始化为什么学不动。** 2 个输入、2 个隐藏神经元，权重全 0.5，偏置 0，激活 sigmoid；输出层 1 个神经元，权重也是 [0.5, 0.5]。输入 x=[1, −1]：两个神经元的 z 都是 0.5×1+0.5×(−1)=0，sigmoid(0)=0.5，所以 h=[0.5, 0.5]。输出 z₂=0.5×0.5+0.5×0.5=0.5，ŷ=sigmoid(0.5)≈0.6225，目标 0。按平方误差反传：δ₂=2(ŷ−0)·ŷ(1−ŷ)≈1.2449×0.2350≈0.2925。输出权重梯度=[0.2925×0.5, 0.2925×0.5]，两值相等；回传到两个隐藏神经元的 δ 也相等，于是第一层权重梯度是 [[0.0366, −0.0366], [0.0366, −0.0366]]——两行完全一样。更新后两行还是一样：神经元 1 和神经元 2 永远是克隆，不管训多少轮。有效神经元数=1。

**B. Xavier 方差手算。** 取 fan_in=4、fan_out=2 的小层：Var(w)=2/(4+2)=2/6=1/3≈0.333，标准差=√(1/3)≈0.577；均匀分布版本 limit=√(6/6)=1，即从 U(−1,1) 采样——这个均匀分布的方差=2²/12=4/12=1/3，与正态版一致（两种采样方式方差相同，可以互相替换）。同一层的 Kaiming：Var(w)=2/fan_in=2/4=0.5，标准差≈0.707。为什么 Kaiming 更大？ReLU 把约一半的激活砍成 0，信号平均只剩一半，所以方差要预先乘 2 补偿。

**C. 方差守恒检查。** 取宽度 2 的层（fan_in=fan_out=2）：Xavier Var=2/(2+2)=0.5，前向每层方差乘 fan_in×Var(w)=2×0.5=1——不变。若错误地用 Var(w)=1（标准正态），每层乘 2，10 层后方差放大 2¹⁰=1024 倍；若 Var(w)=0.01，每层乘 0.02，10 层后 0.02¹⁰≈10⁻¹⁷，信号消失。选对方差，就是让这个“每层乘数”恰好等于 1。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：先看第 1 步的四个初始化函数（本课主角，每个不到 5 行）；第 2 步激活函数（配角）；第 3 步 `forward_deep` 是核心实验机——把随机数据灌进 50 层网络，逐层记录激活幅值；第 4 步 `run_experiment` 把六种“初始化×激活”组合跑成一张表；第 5 步 `symmetry_demo` 是 0.3 节 A 的代码版；第 6 步 `magnitude_report` 画逐层柱状图。核心数据结构是“列表的列表”当矩阵，没有任何第三方库。

### 0.5 理解检查（含参考答案）

**问题1：** 随机初始化已经破坏了对称性，为什么还不够？

**参考答案：** 对称只要求“各不相同”，随机确实做到了。但训练能否进行还取决于方差的尺度：随机尺度稍大，信号逐层放大直到溢出；稍小则逐层缩小直到消失。层数越深，“能用”与“崩坏”之间的边界越窄，50 层时几乎不存在侥幸。

**问题2：** fan_in=512 的层用 Xavier，标准差是多少？量级为什么变小了？

**参考答案：** Var=2/(512+fan_out)，即便 fan_out 也是 512，Var≈2/1024≈0.002，std≈0.044。输入越多，每个权重要分摊的“放大贡献”越小，否则 512 项相加会把方差放大 512 倍——这就是公式里分母是 fan_in 的直觉。

**问题3：** 全零初始化加 sigmoid 的 50 层实验里，激活幅值既不爆也不消失，这说明它没问题吗？

**参考答案：** 不。sigmoid(0)=0.5，幅值恒为 0.5，看起来“稳定”，但它的问题是 0.3 节 A 演示的对称性：所有神经元是克隆，网络宽度形同虚设。幅值稳定≠能学习。

能解释这个三点小例后，进入下面的公式推导。

---

## 一、这一课解决什么问题？

权重初始化是深度学习里最被低估的决定：架构有人写论文、优化器有人写博客、初始化只配得上脚注——但出生点选错，网络在训练开始前就死了。原课的三个场景：全零初始化，10000 轮后 512 个神经元仍是同一个神经元的 512 份拷贝；初始化太大，第 10 层激活到 1e15，第 20 层溢出为无穷；标准正态初始化，3 层能用，50 层时信号要么塌成零要么炸成无穷——取决于随机尺度“稍小”还是“稍大”，中间的可用区间像刀刃一样薄（以上描述为原课参考）。本课的目标：理解并实现零、随机、Xavier、Kaiming 四种策略，学会按激活函数选对初始化。前置：[第04课](<04：激活函数——非线性、ReLU与饱和区.md>)的激活函数与[第03课](<03：从零实现反向传播——链式法则与计算图.md>)的反向传播。

## 二、核心概念与推导

### 2.1 对称问题：全零（或全同）学不动

层内每个神经元结构相同：乘权重、加偏置、过激活。若所有权重初值相同（全零是极端情形），每个神经元输出相同；反向传播时每个神经元收到相同梯度；更新时每个神经元改变相同量。网络有几百个参数，却整体齐步走——这叫对称，随机化是打破它的蛮力办法：每个神经元从权重空间的不同点出发，才可能学出不同的特征。但“随机”只解决方向问题，不解决尺度问题——尺度决定网络能不能训。

### 2.2 方差逐层传播

单层，fan_in 个输入：

$$z = w_1 x_1 + \cdots + w_n x_n$$

若各权重独立同分布、方差 Var(w)，各输入独立、方差 Var(x)，则：

$$\mathrm{Var}(z) = \text{fan}_{\text{in}} \cdot \mathrm{Var}(w) \cdot \mathrm{Var}(x)$$

直觉：n 个独立随机项相加，方差相加 n 次，所以乘 fan_in。若 Var(w)=1 且 fan_in=512，输出方差是输入的 512 倍，10 层后 512¹⁰≈1.2×10²⁷——爆炸（原课参考）。若 Var(w)=0.001，每层乘 0.001×512=0.512，10 层后 0.512¹⁰≈0.00013——消失（原课参考）。目标：选 Var(w) 使 Var(z)=Var(x)，信号幅度逐层不变。

### 2.3 Xavier/Glorot 初始化

Glorot 与 Bengio（2010）为 sigmoid/tanh 推出的解。要让前向和反向的方差都稳定，取调和折中：

$$\mathrm{Var}(w) = \frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}$$

（补充：这是“前向要求 1/fan_in”与“反向要求 1/fan_out”的调和平均——2/(fan_in+fan_out) 恰为 1/fan_in 与 1/fan_out 的调和平均。）实践采样两种等价方式：

$$w \sim \text{Uniform}(-\text{limit}, \text{limit}),\ \text{limit}=\sqrt{\frac{6}{\text{fan}_{\text{in}}+\text{fan}_{\text{out}}}} \quad\text{或}\quad w \sim \text{Normal}\left(0, \sqrt{\frac{2}{\text{fan}_{\text{in}}+\text{fan}_{\text{out}}}}\right)$$

为什么对 sigmoid/tanh 有效：这两者在零点附近近似线性，而正确初始化的激活恰好生活在零附近，方差能稳定穿过几十层。

### 2.4 Kaiming/He 初始化

ReLU 把负半轴全部置零，约一半输出被砍掉，相当于有效 fan_in 减半——Xavier 低估了所需方差。He 等（2015）修正：

$$\mathrm{Var}(w) = \frac{2}{\text{fan}_{\text{in}}},\qquad w \sim \text{Normal}\left(0, \sqrt{\frac{2}{\text{fan}_{\text{in}}}}\right)$$

因子 2 就是补偿 ReLU 砍掉的那一半。没有它，信号每层缩到约 0.5 倍，50 层后 0.5⁵⁰≈8.8×10⁻¹⁶（原课参考）。GELU/Swish 行为接近 ReLU，配 Kaiming。

### 2.5 Transformer 的残差缩放

GPT-2 引入的模式。残差连接把子层输出加回输入：

$$x = x + \text{sublayer}(x)$$

每加一次方差就涨一点，N 个残差层后方差正比于 N 增长。GPT-2 把残差层的投影权重按 1/√(2N) 缩放（N 为层数），让累积的信号幅度保持稳定。Llama 3（405B 参数、126 层）用类似方案——没有这层缩放，残差流穿过 126 层注意力与前馈块会无界增长（原课参考）。

### 2.6 怎么选：一张对照表（依原课决策图整理）

| 激活函数 | 初始化 | 公式 |
|---|---|---|
| Sigmoid / Tanh | Xavier/Glorot | Var = 2/(fan_in+fan_out) |
| ReLU / Leaky ReLU | Kaiming/He | Var = 2/fan_in |
| GELU / Swish | Kaiming/He（同 ReLU） | Var = 2/fan_in |
| Transformer 残差投影 | GPT-2 残差缩放 | 额外乘 1/√(2N) |

验证标准：激活幅值在所有层保持在 0.5—2.0 之间（原课给的检查区间）。

### 2.7 用 0.3 的小数字把公式串起来（补充）

0.3 节 B 算过 fan_in=4、fan_out=2 时 Xavier std≈0.577、Kaiming std≈0.707；0.3 节 C 验证了宽度 2 的层 Xavier 乘数恰为 1、错误尺度的乘数是 2 与 0.02。读第 4 步代码时把每个 config 对应回这些数字：“Random N(0,1)”对应乘数>1（爆炸），“N(0,0.01)”对应乘数≪1（消失），“Xavier/Kaiming”对应乘数≈1（稳定），“Zero”对应幅值恒 0.5 但对称僵死。

## 三、操作前准备

Build It 只用 Python 标准库（math、random），无需第三方依赖；Use It 部分需要 PyTorch：

~~~bash
python -m pip install torch
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行。补充提醒：第 4 步 `run_experiment` 在纯 Python 下要把 6 种组合各跑 50 层×100 样本×宽度 64 的前向，没有 numpy 会花一阵子，属正常现象。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：四种初始化策略

四个函数都返回 fan_out 行 × fan_in 列的二维列表（用“列表的列表”当矩阵）。zero 全 0；random 按指定尺度采高斯；xavier、kaiming 分别按 2.3、2.4 节公式算标准差再采样。注意 kaiming_init 不用 fan_out——它只保前向。

~~~python
# 本块中文说明：零、随机、Xavier、Kaiming 四种初始化策略。
import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
~~~

### 4.2 第 2 步：激活函数

 sigmoid 里先夹到 ±500 是防 math.exp 溢出的常用保护（补充点明其作用）。

~~~python
# 本块中文说明：测试用的三种激活函数。
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)
~~~

### 4.3 第 3 步：50 层前向实验机

核心函数。生成 100 个宽度 64 的随机样本，逐层新建权重（每层独立调用 init_fn）、前向、记录该层平均激活幅值（绝对值平均再对样本平均）。开头 `random.seed(42)` 让每次调用用同一批输入与同一种子——六种策略的比较因此公平（补充：这是原课设计，值得注意的细节）。

~~~python
# 本块中文说明：把随机数据灌过 n_layers 层，逐层记录平均激活幅值。
def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes
~~~

### 4.4 第 4 步：跑六种组合

表格打印第 1/5/10/25/50 层的幅值，超过 1e6 标 EXPLODED、低于 1e-6 标 VANISHED。原课预期（原课参考）：N(0,1) 到第 5 层就爆、N(0,0.01) 到第 10 层消失、Xavier+Sigmoid 与 Kaiming+ReLU 到第 50 层都停在约 1.0。核对：Zero+Sigmoid 这一行幅值会恒为 0.5，既不爆也不消失——它的问题是对称而非幅值，看表时别被“数值正常”骗了（见 0.5 节问题 3）。

~~~python
# 本块中文说明：六种“初始化×激活”组合的 50 层对比实验。
def run_experiment():
    configs = [
        ("Zero init + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("Random N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("Random N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"{'Strategy':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'EXPLODED':>10}"
            elif val < 1e-6:
                row += f" {'VANISHED':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)
~~~

### 4.5 第 5 步：对称性演示

0.3 节 A 的代码版：2 输入、4 个神经元、全零权重，输入 [0.5, −0.3]，四个输出全部是 sigmoid(0)=0.5。补充：把权重换成全 0.1 或任何全同值，结论一样——对称不要求是零。

~~~python
# 本块中文说明：全零初始化下四个神经元输出完全相同。
def symmetry_demo():
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("\nSymmetry Demo (4 neurons, zero init):")
    for i, out in enumerate(outputs):
        print(f"  Neuron {i}: output = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  All identical: {all_same}")
    print(f"  Effective parameters: 1 (not {len(weights) * len(weights[0])})")
~~~

### 4.6 第 6 步：逐层幅值报告

每 5 层打一行“柱状图”：幅值×10 取整画 # 号，超界画 EXPLODED/VANISHED。核对：原课页面没有调用它的入口，`run_experiment` 也不调用它——想看图需自己调 `magnitude_report(name, forward_deep(init_fn, act_fn))`（见第五节补充入口）。

~~~python
# 本块中文说明：50 层激活幅值的文本柱状图。
def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " EXPLODED"
            elif mag < 1e-6:
                bar = "." + " VANISHED"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  Layer {i+1:3d}: {bar} ({mag:.6f})")
~~~

## 五、把代码运行起来：补充入口与检查

（补充）原课各步是函数定义，页面未给主入口；按第四节顺序执行后，需要自己调用。最小检查与入口：

~~~python
# 本块中文说明：补充的检查入口（不依赖随机数的部分可精确断言）。
import math

# 可手算预期（0.3 节 B）：fan_in=4、fan_out=2
assert abs(math.sqrt(2.0 / (4 + 2)) - 0.5773502) < 1e-6   # Xavier std ≈ 0.577
assert abs(math.sqrt(6.0 / (4 + 2)) - 1.0) < 1e-12        # 均匀版 limit = 1
assert abs(math.sqrt(2.0 / 4) - 0.7071067) < 1e-6         # Kaiming std ≈ 0.707
# 均匀 Xavier 的方差 = (2*1)^2/12 = 1/3，与 2/(4+2) 一致（两种采样等方差）

symmetry_demo()          # 预期：4 个输出全 0.5，All identical: True
run_experiment()          # 纯 Python 较慢，六种组合各 50 层
magnitude_report("Kaiming + ReLU", forward_deep(kaiming_init, relu))
~~~

补充检查方法：把 `magnitude_report` 分别用在爆炸与消失的组合上，看柱子逐层变长（爆炸）或塌掉（消失）；用 Kaiming 时柱长应基本不变。本篇没有运行 50 层实验，不虚构其具体输出；原课预期数值（第 50 层约 1.0 等）均为原课参考。

## 六、Use It：使用现成工具

PyTorch 把这些初始化器做成了 `torch.nn.init` 的内置函数：

~~~python
# 本块中文说明：PyTorch 内置初始化器的标准用法。
import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

nn.init.zeros_(layer.bias)
~~~

关键事实（原课）：`nn.Linear(512, 256)` 的默认初始化就是 Kaiming 均匀——这就是多数简单网络“开箱即好训”的原因，PyTorch 已经替你做了对的选择。但自建架构、或网络深过 20 层时，需要懂默认值在做什么并按需覆盖。Transformer 方面，HuggingFace 模型通常在 `_init_weights` 里处理初始化，GPT-2 的实现把残差投影按 1/√N 缩放；自己从零搭 transformer 要自己加这层缩放。

## 七、失败模式、原文边界与工程注意

1. 症状对号入座：损失一开始就 NaN/inf → 疑似爆炸（尺度太大）；损失完全不降且各神经元输出一样 → 疑似对称（全同初始化）；深层网络梯度逐层趋零 → 疑似消失（尺度太小或激活饱和）。
2. “随机初始化”不是一件事而是一个家族：N(0,1) 只对很浅的网络侥幸可用；深度一上来，尺度必须随 fan_in 走公式（第 1 步四个函数的差别全在尺度）。
3. 激活与初始化必须配对：Xavier 配 ReLU 会低估方差（信号逐层缩 0.5），Kaiming 配 tanh 会偏大——对照表见 2.6。
4. 原文边界：原课的方差推导假设权重独立、输入独立、激活近似线性（零点附近），是工程上够用的近似而非精确定律；BatchNorm 等Normalization 层会改变传播行为，本课模型里没有它们。
5. Transformer 的残差缩放（1/√(2N)）与 Xavier/Kaiming 是两套并存的机制：后者管“每层内部”，前者管“跨层累加”，别混为一谈（补充）。
6. 零初始化偏置、非零权重的组合是常见做法（本课实验里 biases 全 0）；真正致命的是权重全同。
7. 工程习惯（补充）：初始化只在建模时定一次，之后换学习率、换优化器都救不回一个对称僵死的网络——排错时先查出生点。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/prompt-init-strategy.md`（一个“诊断权重初始化问题并推荐策略”的提示词）。此处改写成自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
检查权重初始化：说明每层的 fan_in/fan_out 与所用初始化（零/随机/Xavier/Kaiming/残差缩放）；确认初始化与激活函数配对（sigmoid/tanh→Xavier，ReLU/GELU→Kaiming）；对 20 层以上的网络，报告逐层激活幅值是否保持在 0.5—2.0 区间；若有残差连接，说明是否做了 1/√(2N) 类缩放；确认没有整层全同初始化；所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课练习 1：加入 LeCun 初始化（Var=1/fan_in，为 SELU 设计），用 LeCun+tanh 跑 50 层实验并与 Xavier+tanh 对比。验收标准：能给出两种策略在第 50 层的幅值对比，并解释 LeCun 方差（1/fan_in）与 Xavier（2/(fan_in+fan_out)）在 fan_in=fan_out 时差一个因子 2、在 fan_out≪fan_in 时几乎重合。
2. 原课练习 2：实现 GPT-2 残差缩放——每层输出乘 1/√(2N) 后再加回残差流；50 层对比有无缩放时残差幅值的增长速度。验收标准：无缩放时幅值约随层数线性增长，有缩放时基本平稳，能用手算说明“每层加一个同量级项 → 方差随 N 累积”的机制。
3. 原课练习 3：写“初始化体检”函数——输入各层维度与激活类型，推荐正确初始化并对危险配置告警。验收标准：对 2.6 节对照表的四种情况都能给出正确推荐，对“全零/全同/尺度明显过大过小”能给出告警理由。
4. 原课练习 4：fan_in=16 与 fan_in=1024 各跑一遍实验，展示随机初始化的“能用/崩坏”区间如何随宽度变窄，而 Xavier/Kaiming 自动适配。验收标准：能解释公式里的 fan_in 如何抵消宽度带来的方差放大（乘数=fan_in×Var(w)）。
5. 原课练习 5：实现正交初始化（随机矩阵做 SVD 取正交矩阵 U），50 层 ReLU 网络上与 Kaiming 对比。验收标准：能说明正交矩阵保持向量范数（前向幅度不漂移），并与 Kaiming 的方差守恒视角对照。

## 十、术语与完成标准

权重初始化：训练前设定权重初始值的策略；对称性：全同初值导致神经元永远克隆的困境；对称破坏：随机化让神经元各奔前程；fan-in/fan-out：神经元的输入/输出连接数；Xavier/Glorot：Var=2/(fan_in+fan_out)，配 sigmoid/tanh，兼顾前向与反向；Kaiming/He：Var=2/fan_in，因子 2 补偿 ReLU 砍半；方差传播：Var(z)=fan_in·Var(w)·Var(x) 的逐层分析；残差缩放：1/√(2N) 抑制残差流方差累积；死网络：梯度为零或激活饱和导致无法训练；爆炸激活：方差逐层指数放大直至溢出。

完成检查：

- 能手算 fan_in=4、fan_out=2 的 Xavier 与 Kaiming 标准差（0.3 节 B）。
- 能写出方差传播公式并算出“每层乘数”，解释乘数>1、<1、=1 分别对应什么命运。
- 能用两神经元全 0.5 的小例说明对称性为何无法被训练打破（0.3 节 A）。
- 能按激活函数说出该配哪种初始化以及配错会怎样。
- 能说出 `nn.Linear` 的默认初始化是什么、什么时候需要手动覆盖。

## 十一、验证范围与来源许可

原课 Build It 共 6 个分段，代码已逐字转录嵌入本篇（仅按本篇格式在代码块首行添加中文说明，未改逻辑）；Use It 的 PyTorch 代码同样保留。原课页面未展示运行输出，本篇未虚构任何“原课参考”数值；标注“可手算预期”的数值（Xavier std≈0.577、Kaiming std≈0.707、均匀 limit=1、sigmoid(0)=0.5 等）均为纯算术，可直接手工复核，并已在临时脚本中用 Python 复核通过（仅公式数值断言与 0.3 节两神经元对称算例的梯度复核，属玩具检查，不构成训练或性能实验）。第 4、5 步的 50 层实验未执行，其预期表现以原课参考为准。本次未做：实际训练、与 PyTorch 输出的数值比对。

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
