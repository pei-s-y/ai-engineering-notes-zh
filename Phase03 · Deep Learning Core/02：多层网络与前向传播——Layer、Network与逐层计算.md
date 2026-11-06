# 02：多层网络与前向传播——Layer、Network与逐层计算

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F03-deep-learning-core%2F02-multi-layer-networks) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/02-multi-layer-networks/docs/en.md)。原题：Multi-Layer Networks and Forward Pass。

## 零、初学者先读：把神经元接成流水线

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第01课](<01：感知机——权重、偏置与更新规则.md>)证明了单个感知机只会画直线、画不出 XOR 需要的弯边界；本课把神经元叠成层，搭出 `Layer` 和 `Network` 两个类，让数据从输入一路“流”到输出（前向传播）。权重怎么自动学出来，留给[第03课](<03：从零实现反向传播——链式法则与计算图.md>)。

### 0.1 先理解这一课在做什么

本课只做一件事：把“数据流过网络”这条流水线搭出来，权重先不管——要么手工设、要么随机。所谓前向传播（forward pass），就是“乘权重、加偏置、过激活”三步逐层重复，纯计算、无学习。剧情线：sigmoid → `Layer`（一层怎么写）→ `Network`（多层怎么串）→ 用手工权重解 XOR（2-2-1 结构）→ 随机权重跑圆形分类（前向照跑、结果很差，逼出“需要学习”）。原课的一句话纲领：一个神经元画一条线，叠起来能画任何形状。

### 0.2 本课术语先翻成白话

- 前向传播（forward pass）=数据从输入流到输出：乘权重、加偏置、过激活，逐层重复；纯计算，无学习
- 输入层（input layer）=放原始数据的地方；不算真正的层，零计算
- 隐藏层（hidden layer）=中间做变换的层；“隐藏”指这些值从不出现在训练数据里
- 输出层（output layer）=给最终答案：二分类 1 个神经元，多分类每类一个
- 隐藏状态（hidden state）=每层输出的那个向量；文本任务升维（一个词 768 个数），图像任务降维
- 激活函数（activation function）=线性变换之后套的非线性函数，把“直线”掰弯的来源
- sigmoid（S 型曲线）=把任意实数压进 (0,1)：大正→接近 1，大负→接近 0，0→0.5
- 权重矩阵（weight matrix）=一层所有神经元的权重排成的表，形状 (本层神经元数, 上层神经元数)
- 偏置向量（bias vector）=矩阵乘法之后加上的那截，让输入全 0 时神经元也能激活
- 线性变换（linear transformation）=z=Wx+b，激活之前的那步
- 万能逼近定理（universal approximation theorem）=一个隐藏层+足够多神经元可任意精度逼近任何连续函数；“可行”不等于“划算”
- 2-2-1 / 2-3-1=结构速记：2 输入-2 隐藏-1 输出
- 可组合性（composability）=网络可以堆叠、串联、并行（encoder/decoder 各种组合）

### 0.3 用小数字走一遍

**第一段：一个神经元的前向。** 单神经元，权重 [0.5, −0.5]、偏置 0.1，输入 [1, 0]：

1. 乘：z=0.5×1+(−0.5)×0=0.5；
2. 加：z=0.5+0.1=0.6；
3. 压：sigmoid(0.6)=1/(1+e^{−0.6})，e^{−0.6}≈0.5488，所以 a≈1/1.5488≈0.6457。

“乘、加、压”三步，一个神经元的全部就是这么多。

**第二段：2-2-1 网络手工解 XOR 的前向。** 两个隐藏神经元（权重 [20,20]/[−20,−20]，偏置 −10/30）+ 输出神经元（权重 [20,20]，偏置 −30）：

- 输入 (0,0)：h1 的 z=20×0+20×0−10=−10 → sigmoid(−10)≈0.0000454（e^{10}≈22026，1/22026≈0.0000454）；h2 的 z=−20×0−20×0+30=30 → sigmoid(30)≈1.0（离 1 只差约 10^{−13}）；输出 z=20×0.0000454+20×1−30≈−9.9991 → sigmoid≈0.0000454 → 判 0 ✓
- 输入 (0,1)：h1 的 z=20−10=10 → ≈0.9999546；h2 的 z=−20+30=10 → ≈0.9999546；输出 z≈20×0.9999546×2−30=9.9982 → sigmoid≈0.9999546 → 判 1 ✓（(1,0) 对称，同样判 1）
- 输入 (1,1)：h1 的 z=40−10=30 → ≈1.0；h2 的 z=−40+30=−10 → ≈0.0000454；输出 z≈20×(1+0.0000454)−30≈−9.9991 → 判 0 ✓

门道在“大权重”：±20 把 sigmoid 拉得极陡，行为近似阶跃——h1 近似 OR、h2 近似 NAND、输出近似 AND，正是[第01课](<01：感知机——权重、偏置与更新规则.md>)的手工 XOR，只是硬开关换成了陡峭的 sigmoid。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：`sigmoid`（带防溢出钳位）→ `Layer`（一层=权重矩阵+偏置+forward）→ `Network`（把层串起来，四行逻辑）→ 4.4 手工权重 XOR（0.3 第二段）→ 4.5 圆形分类（随机权重）。核心数据结构：`weights`（嵌套列表，第 i 行是第 i 个神经元的权重）、`biases`（列表）、`layers`（Layer 对象列表）。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么说输入层“不算真正的层”？

**参考答案：** 输入层只存放原始数据：不做任何计算、没有参数。2-2-1 网络其实只有一个隐藏层加一个输出层；“N 层网络”数的是带权重和偏置的层。

**问题2：** 权重矩阵的形状怎么记？2-3-1 网络里两个矩阵各是什么形状？

**参考答案：** 第 k 层的 W 形状是 (第 k 层神经元数, 第 k−1 层神经元数)——行对当前层、列对上一层。2-3-1 里 W1 是 (3,2)、b1 是 (3,)，W2 是 (1,3)、b2 是 (1,)。形状接不上就是 bug。

**问题3：** 前向传播里有学习吗？随机权重的圆分类网络为什么也能算出准确率？

**参考答案：** 没有，前向是纯计算，权重一个都不改；学习（改权重）是第03课的反向传播。随机权重照样能完成“乘、加、压”并输出 0/1，所以准确率算得出来——只是接近瞎猜（原课参考：常常不如直接猜多数类）。

---

## 一、这一课解决什么问题？

单个神经元是“画直线的”，而 AI 里的真实问题——图像识别、语言理解、下围棋——都需要曲线。1969 年 Minsky 和 Papert 证明这不是“难学”而是“数学上不可能”：单层网络学不会 XOR。真值表里 [0,1]、[1,0] 一边，[0,0]、[1,1] 另一边，没有一条直线能分开。这个结论让神经网络经费断了十年。事后看解法很直白：别用一层。让第一层把输入空间切成新特征，第二层把这些特征组合成单条直线做不出的决定。这个叠起来的结构就是多层网络，是今天所有生产级深度学习模型的地基。而前向传播——数据从输入经隐藏层流向输出——是把地基打好之前必须先会走的第一步。

## 二、核心概念与推导

### 2.1 三种层：输入、隐藏、输出

- 输入层：不是真正的层，只放原始数据。两个特征就是两个输入节点，零计算。
- 隐藏层：干活的地方。每个神经元接收上一层的全部输出，加权、加偏置、过激活。
- 输出层：最终答案。二分类 1 个 sigmoid 神经元；多分类每类一个。

每层产出一个向量，叫隐藏状态（hidden state）。文本任务里隐藏状态升维——把一个词编码成 768 个数来承载语义；图像任务里降维——把上百万像素压成可处理的表示。原课说：学习就藏在隐藏状态里。

### 2.2 神经元与激活：乘、加、压

每个神经元做三件事：每个输入乘对应权重；求和加偏置；过激活函数。本课的激活是 sigmoid：

$$\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

sigmoid 把任意实数压进 (0,1)：大正数推向 1，大负数推向 0，0 映到 0.5。正是这条光滑曲线让学习成为可能——与感知机的硬阶跃不同，sigmoid 处处有梯度。

### 2.3 前向传播：数据怎么流

前向传播把输入逐层推向输出，期间没有学习发生，纯计算：乘、加、激活、重复。每层两步：

$$z = W \cdot x + b \quad \text{（线性变换）},\qquad a = \text{sigmoid}(z) \quad \text{（激活）}$$

一层的输出是下一层的输入。整条流水线：输入 [x1,x2] → 乘 W1 → 加 b1 → sigmoid → 隐藏输出 [h1,h2,h3] → 乘 W2 → 加 b2 → sigmoid → 输出 y。这就是前向传播的全部。

### 2.4 矩阵维度：最重要的调试技能

2-3-1 网络的维度表（原课表格）：

| 步骤 | 运算 | 维度 | 结果形状 |
|------|------|------------|-------------|
| 输入 | x | -- | (2,) |
| 隐藏层线性 | W1·x+b1 | W1: (3,2)，b1: (3,) | (3,) |
| 隐藏层激活 | sigmoid(z1) | -- | (3,) |
| 输出层线性 | W2·h+b2 | W2: (1,3)，b2: (1,) | (1,) |
| 输出层激活 | sigmoid(z2) | -- | (1,) |

规则：第 k 层的权重矩阵形状是 (本层神经元数, 上一层神经元数)。行对当前层，列对上一层。形状对不上，就有 bug。原课判断：维度追踪是深度学习最重要的调试技能，没有之一。

### 2.5 万能逼近定理

1989 年 George Cybenko 证明：单个隐藏层、足够多神经元的网络，可以以任意精度逼近任何连续函数。但这不意味着一个隐藏层总是最优——它只说明结构“理论上可行”。实践中更深的网络（层数多、每层神经元少）用远少于“浅而宽”的总参数量学到同样的函数，这就是深度学习成立的原因。直觉：隐藏层里每个神经元学一个“鼓包”或特征，足够多的鼓包放在合适位置就能逼近任何光滑曲线——神经元越多，鼓包越多，逼近越好（4 个隐藏神经元粗糙、16 个接近、64 个近乎完美）。

### 2.6 可组合性

神经网络可组合：能堆叠、串联、并行。Whisper 用 encoder 网络处理音频、另一个 decoder 网络生成文本；现代大模型多为 decoder-only；BERT 是 encoder-only；T5 是 encoder-decoder。架构的选择决定了模型能做什么。

## 三、操作前准备

Build It 部分是纯 Python（math、random），每个矩阵运算手写、不用 NumPy；Use It 部分需要 PyTorch：

~~~bash
python -m pip install torch
~~~

先激活你已有的 Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行：4.2 起依赖 4.1 的 `sigmoid`，4.4 依赖 `Layer`/`Network`，4.5 的两段代码相互衔接。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：sigmoid 激活

~~~python
# 本块中文说明：带数值钳位的 sigmoid，防止 exp 溢出。
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
~~~

钳位到 [−500,500] 防溢出：`math.exp(500)` 很大但有限，`math.exp(1000)` 是无穷。

### 4.2 第 2 步：Layer 类

原课强调：深度学习里最重要的运算是矩阵乘法（matrix multiplication）——每一层、每个注意力头、每次前向传播，底层全是矩阵乘。线性层做的事就一个公式 y=Wx+b；原话说这一个公式占了神经网络 90% 的计算量。

~~~python
# 本块中文说明：一层=权重矩阵+偏置向量+forward；权重形状 (n_neurons, n_inputs)。
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
~~~

权重矩阵形状是 (n_neurons, n_inputs)：第 i 行是第 i 个神经元跨所有输入的权重。`forward` 逐神经元算加权和加偏置、过 sigmoid、收集成列表。（补充：`last_input`/`last_output` 被保存但本课从未使用——这是给第03课反向传播预留的伏笔，反向传播需要前向时存的中间值。）

### 4.3 第 3 步：Network 类

~~~python
# 本块中文说明：网络=层列表；前向传播就是把层按顺序串起来。
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
~~~

这就是前向传播的全部：四行逻辑。数据进去，流过每一层，从另一头出来。

### 4.4 第 4 步：手工权重解 XOR

[第01课](<01：感知机——权重、偏置与更新规则.md>)用 OR、NAND、AND 感知机组合解 XOR；现在用 Layer/Network 类重做。2-2-1 结构：两个输入、两个隐藏神经元、一个输出。

~~~python
# 本块中文说明：2-2-1 网络 + 手工设定的大权重，sigmoid 版 XOR。
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")
~~~

大权重（20、−20）让 sigmoid 表现得像阶跃函数：第一个隐藏神经元近似 OR、第二个近似 NAND、输出神经元把它们组合成 AND，整体就是 XOR。0.3 第二段已逐点手算四个输出（约 0.000045、0.999955、0.999955、0.000045）。

### 4.5 第 5 步：圆形分类

更难的问题：把 2D 点分为“在半径 0.5 的圆内”还是“圆外”（圆心在原点）。这需要弯的决策边界——单个感知机不可能做到。

~~~python
# 本块中文说明：生成圆内/圆外两类点，搭一个 2-8-1 网络（随机权重）。
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])
~~~

随机权重下网络不会分类好，但前向传播照样能跑。这就是要点——前向传播只是计算；学出正确的权重是反向传播的事，第03课见。

~~~python
# 本块中文说明：用随机权重跑完 200 个点，统计准确率。
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"Accuracy with random weights: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
~~~

原课参考：随机权重的准确率很差，常常不如直接猜多数类。训练之后（第03课），同一个 8 隐藏神经元的结构将画出把圆内外分开的弯边界。（补充算例：均匀采样 [−1,1]² 时圆内点约占 π×0.5²÷4≈19.6%，所以“全猜圆外”本来就有约八成正确率——“不如猜多数类”里说的多数类就是它。）

## 五、把代码运行起来：补充入口与检查

（补充）原课各步是连续脚本：4.2 起依赖 4.1 的 `sigmoid`，4.4 依赖 `Layer`/`Network`，4.5 两段相互衔接。以下是补充的最小检查，不依赖训练：

~~~python
# 本块中文说明：补充的最小结构与手算检查入口。
tiny = Layer(2, 1, weights=[[0.5, -0.5]], biases=[0.1])
out = tiny.forward([1, 0])
# 可手算预期：z=0.5×1+(−0.5)×0+0.1=0.6，sigmoid(0.6)≈0.645656（0.3 第一段）。
assert abs(out[0] - 0.6456563062257954) < 1e-9

preds = [1 if xor_net.forward(i)[0] >= 0.5 else 0 for i, _ in xor_data]
# 可手算预期：见 0.3 第二段，四个输出约 0.000045/0.999955/0.999955/0.000045。
assert preds == [0, 1, 1, 0]

l = Layer(3, 5)
# 可手算预期：形状规则——5 个神经元、每个 3 个输入，权重矩阵 5 行 3 列。
assert len(l.weights) == 5 and all(len(r) == 3 for r in l.weights)
~~~

补充检查方法：手工算一遍练习 3 的参数量口径——(784×256+256)+(256×128+128)+(128×10+10)=235,146 个参数；再把 4.5 的随机权重准确率与“全猜圆外”约八成的基线对照，掂一掂“没训练的模型连瞎猜都不如”这句话的分量。

## 六、Use It：使用现成工具

PyTorch 四行做完上面的一切：

~~~python
# 本块中文说明：PyTorch 版的两层网络与前向传播。
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
~~~

`nn.Linear(2, 8)` 就是你的 `Layer` 类：权重矩阵形状 (8,2)、偏置向量 (8,)。`nn.Sigmoid()` 是逐元素的 sigmoid。`nn.Sequential` 就是你的 `Network` 类：按顺序串层。差别在速度与规模：PyTorch 跑 GPU、处理百万级样本批次、自动为反向传播算梯度。但前向传播的逻辑与你手写的完全相同。

## 七、失败模式、原文边界与工程注意

1. 形状不匹配是第一类 bug：牢记 (本层神经元, 上层神经元) 规则，写代码前先把每层形状列在纸上（原课：维度追踪是最重要的调试技能）。
2. 前向能跑≠模型可用：随机权重的网络照样输出数字，4.5 就是活例子（原文边界：原课明确说随机权重不会分类好）。
3. 忘了激活函数，多层塌回一层：线性套线性还是线性，W2(W1x+b1)+b2 仍是仿射变换，非线性全部来自激活函数（补充：原课术语表说激活函数“把曲线引入决策边界”，此条是其数学推论）。
4. 大权重让 sigmoid 饱和：20/−20 的手工网络输出贴着 0 和 1，那些位置 sigmoid 的导数接近 0——第03课会看到这正是梯度消失的温床（补充，衔接下一课）。
5. 随机初始化范围 uniform(−1,1) 对深网络偏大、容易饱和；怎么初始化是第08课的主题（补充预告，原课本课未展开）。
6. sigmoid 的 ±500 钳位防的是极端输入的溢出，不是普通数值（原课解释：exp(500) 大但有限，exp(1000) 无穷）。
7. 手工调权重不可扩展：XOR 能手调，图像分类的百万级权重不能——这是“为什么需要学习算法”的原文边界。
8. 原课说随机权重“常常不如猜多数类”：本机玩具复跑（seed 42）得 82.5%，恰好等于全部猜“圆外”的多数类基线（补充核对，见第十一节记录；不同种子结果会变）。

## 八、Ship It：本课可复用的中文检查模板

原课本步产出 `outputs/prompt-network-architect.md`（设计网络架构时可复用的提示词）。此处改写成自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
设计网络架构时：先写清输入/输出维度，用 (本层神经元, 上层神经元) 规则核对每个权重矩阵的形状；从 1 个隐藏层、少量神经元起步，说明加宽（每层更多神经元）与加深（更多层）在参数量和训练难度上的代价；二分类输出层用 1 个 sigmoid 神经元、多分类每类一个；激活函数的选择注明依据（本阶段第04课展开）；报告总参数量（练习 3 的口径：全部权重与偏置个数）；未实测的结论一律标“参考”。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：搭一个 2-4-2-1 网络（两个隐藏层），随机权重下对 XOR 数据跑前向，打印中间隐藏层输出，观察表示如何逐层变换。验收标准：第一隐藏层打出 4 个数、第二隐藏层 2 个数，都在 (0,1) 内；能描述相邻两层输出分布的差异。
2. 原课：把圆分类器隐藏层从 8 改成 2、再改成 32，各用随机权重跑前向。隐藏神经元数量改变输出的范围或分布吗？为什么？验收标准：能回答“改变的是隐藏表示的维度与表达能力，输出范围不变——输出永远是输出层 sigmoid 的 (0,1)”并解释原因（输出范围由输出层激活决定）。
3. 原课：给 Network 类实现 `count_parameters`，返回可训练权重与偏置总数；在 784-256-128-10（经典 MNIST 结构）上测试。验收标准：得到 235,146，且能手算 (784×256+256)+(256×128+128)+(128×10+10) 复核。
4. 原课：给 3-4-4-2 网络写前向传播，输入归一化到 0-1 的 RGB 值，观察两个输出。验收标准：形状链 (3,)→(4,)→(4,)→(2,) 核对无误，两个输出都在 (0,1) 内；能说明这是两类颜色分类器的架构。
5. 原课：把 sigmoid 换成“漏阶跃”（leaky step）：z<0 时返回 0.01·z，否则返回 1.0。用 4.4 同样的手工权重对 XOR 跑前向，还能工作吗？为什么光滑的 sigmoid 更受偏爱？验收标准（补充手算要点）：四个点判类仍然全对——判 1 的输出恰为 1.0，判 0 的输出为负数（如 (0,0)：h1=0.01×(−10)=−0.1、h2=1.0，输出 z=20×(−0.1)+20×1−30=−12 → 0.01×(−12)=−0.12，判 0）——但输出越界出现负值、不再是概率，且函数在 z=0 处不连续、z<0 段梯度只有 0.01，撑不起梯度学习。这正是偏爱光滑 sigmoid 的原因。

## 十、术语与完成标准

前向传播（forward pass）：数据逐层流过网络——乘权重、加偏置、激活——产出输出；隐藏层（hidden layer）：输入与输出之间、值不被直接观测的层；多层网络（multi-layer network）：神经元按层堆叠、上层输出喂下层输入；激活函数（activation function）：线性变换之后施加、把曲线引入决策边界的非线性函数；sigmoid：σ(z)=1/(1+e^{−z})，压到 (0,1)、处处光滑可导；权重矩阵（weight matrix）：形状为 (当前层神经元, 上一层神经元) 的可学习连接强度；偏置向量（bias vector）：矩阵乘之后加的向量，让输入全零时也能激活；万能逼近（universal approximation）：单隐藏层足够多神经元可逼近任何连续函数，但“足够多”可能意味着数十亿；线性变换（linear transformation）：z=Wx+b，激活前的计算；决策边界（decision boundary）：网络输出跨过分类阈值的输入空间曲面。

完成检查：

- 能手算 0.3 两段（单神经元前向、2-2-1 手工 XOR 前向）。
- 能背出权重矩阵形状规则并写出 2-3-1 的两个矩阵形状。
- 能解释“输入层不是层”和“前向传播没有学习”。
- 能说出万能逼近定理说了什么、没说什么（可行≠划算，深比宽省参数）。
- 能解释为什么没有激活函数的多层网络等价于一个线性模型。

## 十一、验证范围与来源许可

原课 Build It 共 5 个步骤（6 个 python 代码块），代码已逐字转录嵌入本篇（仅按本篇格式添加中文说明与注释，未改逻辑）；Use It 的 PyTorch 代码同样保留。原课页面未展示运行输出数值，本篇未虚构任何“原课参考”数值——标注处均为原课文字结论（如“常常不如猜多数类”）。标为“可手算预期”的数值均可用 0.3 节算例手工复核。本次未安装 PyTorch、未下载任何数据或模型。写作完成后，将正文各 python 代码块抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 逐一做静态语法检查，结果见下方记录。

本次验证记录（2026-11-06）：

- 抓取课程页成功获取原课全文，以原文为准整理。
- 正文 8 个 python 代码块（含 Use It 与第五节补充检查）全部通过 `ast.parse` 静态语法检查。
- 第五节三个最小检查在临时脚本中实际执行通过（玩具数据，属教学断言复核，不构成训练或性能实验）：单神经元前向 sigmoid(0.6)=0.6456563062（与 0.3 手算一致）；手工 XOR 网络四点输出约 0.000045/0.999955/0.999955/0.000045（与 0.3 第二段一致）；`Layer(3,5)` 权重矩阵 5 行 3 列。
- 另将 4.5 的随机权重圆分类在临时脚本中复跑（seed 42，纯前向、无训练）：准确率 82.5%（165/200），恰好等于全部猜“圆外”的多数类基线（该数据里圆内点 35/200）——印证原课“常常不如猜多数类”的定性结论；此数值为玩具复跑参考，不同种子会不同。
- 未做：PyTorch 版运行、训练后的圆分类（属第03课内容）。

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
