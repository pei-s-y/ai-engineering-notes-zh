# 06：优化器——从SGD到Adam

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/06-optimizers) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/06-optimizers/docs/en.md)。原题：Optimizers。

## 零、初学者先读：梯度只给方向，优化器决定怎么走

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第05课](<05：损失函数——MSE、交叉熵与选择.md>)定义了“错多少”，本课解决“拿到梯度之后怎么走”：走多远、要不要带惯性、要不要给每个参数自己的步长。

### 0.1 先理解这一课在做什么

梯度下降告诉你“下坡方向是哪边”，但一个裸的更新规则 $w = w - \text{lr}\cdot g$ 有三个实际毛病。第一，震荡：真实损失面常是细长山谷，梯度主要指向“横穿山谷”的陡方向，你在窄方向来回弹跳，在有用的长方向上挪得很慢。第二，一套学习率伺候所有参数是错的：有的权重还在欠拟合需要大步，有的已接近最优需要小步。第三，鞍点与平坦区：高维空间里大片区域梯度近零，裸 SGD 爬得像蜗牛。动量（momentum）治震荡，自适应学习率治“一刀切”，Adam 把两者合起来再加偏差修正（bias correction），成为 80% 问题的默认答案；AdamW 再把权重衰减（weight decay）从梯度通道里解耦出来。本课从零写这四个优化器，让你知道那 20% 失败时该往哪看。

### 0.2 本课术语先翻成白话

- 学习率（learning rate，lr）=每步走多远的标量，影响最大的超参数
- SGD（随机梯度下降，Stochastic Gradient Descent）=在小批量上估梯度，沿负方向走一步
- 小批量（mini-batch）=每步用来估计梯度的一小撮样本，带来的噪声既是麻烦也是逃离尖锐极小值的帮手
- 动量（momentum）=历史梯度的指数滑动平均，像速度：同向的累加、反向的抵消
- β（beta）=动量的记忆长度旋钮，0.9 约等于平均最近 10 步（$1/(1-\beta)=10$）
- RMSProp=每个参数除以“自己近期梯度平方的滑动平均的平方根”，各自调速
- 一阶矩 / 二阶矩（first/second moment）=Adam 里梯度的均值估计 m 和未中心化方差估计 v
- 偏差修正（bias correction）=把刚开始被零初始化压小的滑动平均除以 $(1-\beta^t)$ 拉回真实值
- Adam=momentum + RMSProp + 偏差修正
- AdamW=Adam + 解耦权重衰减：衰减直接作用在权重上，不经过自适应缩放
- 权重衰减（weight decay）=每步按比例把权重往 0 缩一点，正则化手段
- 鞍点（saddle point）=一方向上一方向下的平坦点，高维损失面里大量存在
- 学习率调度（learning rate schedule）=训练中调 lr 的方案：warmup 升、cosine 降
- 梯度裁剪（gradient clipping）=梯度范数超过阈值就整体缩小，防爆炸

### 0.3 用小数字走一遍

设 w=1.0，梯度恒为 0.3（假设连续两步都量到 0.3），lr=0.1。

**裸 SGD。** 第一步：w ← 1.0 − 0.1×0.3 = **0.97**。第二步：0.97 − 0.03 = **0.94**。每步固定走 0.03。

**动量（β=0.9）。** 第一步：v₁ = 0.9×0 + 0.3 = 0.3，w = 1 − 0.1×0.3 = 0.97。第二步：v₂ = 0.9×0.3 + 0.3 = **0.57**，w = 0.97 − 0.1×0.57 = **0.913**。方向一致时步子越走越大——这就是动量在“顺着山谷”方向的加速。再看震荡：若梯度交替为 +0.2、−0.2，SGD 每步 ±0.02 来回摆；动量下 v₁=0.2，v₂=0.9×0.2−0.2=**−0.02**——第二个来回几乎清零，震荡被抵消。

**Adam（β₁=0.9，β₂=0.999，lr=0.001）。** 第一步：m₁ = 0.1×0.3 = 0.03（只有真实梯度的十分之一！），v₁ = 0.001×0.09 = 0.00009。偏差修正：m̂ = 0.03/(1−0.9) = **0.3**，v̂ = 0.00009/(1−0.999) = **0.09**。更新量 = lr×m̂/(√v̂+ε) = 0.001×0.3/0.3 = **0.001**，w = 1 − 0.001 = **0.999**。第二步：m₂ = 0.9×0.03+0.1×0.3 = 0.057，修正 m̂ = 0.057/(1−0.81) = 0.057/0.19 = **0.3**；v₂ = 0.999×0.00009+0.001×0.09 = 0.00017991，修正 v̂ = 0.00017991/0.001999 ≈ **0.09**；更新量又 ≈ 0.001。两个要点：没有偏差修正，第一步只走 0.00003（0.001×0.03/0.00949… 被严重低估）；有了它，恒定梯度下 Adam 每步稳定走约 lr（0.001）——步长上限天然封在 lr 附近。

**AdamW（λ=0.01）。** 在 Adam 更新后额外减 lr×λ×w = 0.001×0.01×1.0 = 0.00001，w = 0.999 − 0.00001 = **0.99899**。每步都按当前权重值的 0.001% 缩水——权重越大缩得越多，这就是“衰减”。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：Step 1 的 `SGD`（两行核心）；Step 2 的 `SGDMomentum`（多一个 velocities 列表）；Step 3 的 `Adam`（m、v、t 三个状态 + 偏差修正两行）；Step 4 的 `AdamW`（在 Adam 末尾加一行衰减）；Step 5 的 `OptimizerTestNetwork`（把第04/05课的网络改成“扁平参数列表 + 优化器接管更新”的写法——注意 get_params/set_params 的展平与回填）。四个优化器共享同一个 `step(params, grads)` 接口，可互换。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 Adam 第一步没有偏差修正时更新会“缩水十倍”？

**参考答案：** m 初始化为 0，第一步 m₁ = (1−β₁)·g = 0.1g；同时 v₁ = (1−β₂)·g² 也很小，但两者相除后量级恢复，真正的问题是“滑动平均还没热身”。除以 (1−β₁^t) 后第一步 m̂ 恰好等于 g。约 50 步后修正因子趋近 1，自动失效。

**问题2：** 动量为什么能治震荡？

**参考答案：** 震荡方向的梯度每步换号，在指数平均里互相抵消（0.3 和 −0.3 的平均是 0）；前进方向的梯度号不变，逐步累加放大。于是横向分量被滤掉、纵向分量被加速。

**问题3：** AdamW 与“Adam + L2 正则化”差在哪？

**参考答案：** 把 L2 加进损失后，正则项的梯度会跟任务梯度一起被 Adam 的自适应缩放处理——梯度方差大的参数被正则得少、方差小的被正则得多，不均匀。AdamW 把衰减直接从权重里扣（lr·λ·w），每个参数得到同样的按比例收缩。原课参考：几乎所有任务上 AdamW 收敛到更好的解。

能对上这组小数字后，进入下面的推导与代码。

---

## 一、这一课解决什么问题？

你算出了梯度：4 号权重该减 0.003。可是按什么单位？乘什么尺度？第 1 步和第 1000 步该走一样远吗？裸梯度下降对三个现实病无能为力：山谷震荡、参数共享一个 lr、鞍点爬行。原课的口号：SGD 是指南针，Adam 是带实时路况的 GPS。本课从零实现 SGD → 动量 → Adam → AdamW，理解每一层在修什么、默认超参（lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8）为什么能覆盖 80% 的问题，以及另外 20% 失败时先查什么（先 lr，再 β₂，几乎不动 β₁ 和 ε）。

## 二、核心概念与推导

### 2.1 SGD：基线

$$w = w - \text{lr}\cdot g$$

“随机”指用小批量估计梯度而非全量数据——噪声有助于逃离尖锐极小值，但也造成震荡。现代网络上典型 lr 是 0.01—0.1。唯一的旋钮就是 lr：太大发散、太慢耗时；且同一次训练里理想 lr 也在变（早期要大、后期要小）。

### 2.2 动量：给下坡加惯性

$$m_t = \beta m_{t-1} + g,\qquad w = w - \text{lr}\cdot m_t$$

β=0.9 时 m 大约是最近 10 步梯度的平均（$1/(1-\beta)=10$）。同向累加、反向抵消。原课参考的量级：病态损失面上 SGD 要 10000 步的地方，动量 SGD 通常 3000—5000 步——提速不是零头。（补充：这是“标准动量”写法；Nesterov 变体先看前瞻点，原课练习 1。）

### 2.3 RMSProp：每个参数自己调速

$$s_t = \beta s_{t-1} + (1-\beta)\,g^2,\qquad w = w - \text{lr}\cdot\frac{g}{\sqrt{s_t}+\epsilon}$$

$s_t$ 是近期梯度平方的滑动平均。梯度一直很大的参数除以大数（实际 lr 变小）——“你可能到位了，慢点”；梯度一直很小的参数除以小数（实际 lr 变大）——“你可能没吃饱，快点”。ε（典型 1e-8）防除零。Hinton 在 Coursera 课上提出，从未正式发表论文。

### 2.4 Adam：动量 + RMSProp + 偏差修正

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2$$

$$\hat{m}=\frac{m_t}{1-\beta_1^t},\qquad \hat{v}=\frac{v_t}{1-\beta_2^t},\qquad w = w - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$$

偏差修正（0.3 节算过）：第 1 步把 m̂ 拉回真实梯度；第 100 步 $(1-0.9^{100})≈1$，修正自动消失——它只在前 ~10 步重要、50 步后无关紧要。默认值 lr=0.001、β₁=0.9、β₂=0.999、ε=1e-8 覆盖 80% 的问题；不灵时先调 lr，再 β₂。

### 2.5 AdamW：把权重衰减解耦

在 SGD 里，“损失加 $\lambda w^2$”与“每步减 $\lambda w$”等价。在 Adam 里不等价：L2 项进了梯度就会连同任务梯度一起被 $\hat{m}/\sqrt{\hat{v}}$ 缩放，正则强度随参数的梯度统计变化——不是你想要的。AdamW 的修法（Loshchilov & Hutter 2017）：

$$w = w - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} - \text{lr}\cdot\lambda w$$

衰减项不经过自适应缩放，每个参数获得同样的按比例收缩。原课参考：AdamW 在几乎所有任务上收敛到比 Adam+L2 更好的解，是 BERT、GPT、LLaMA、Stable Diffusion 的训练默认。

### 2.6 学习率：最重要的超参数

调一个超参就调它——lr 差 10 倍的影响大过多数架构改动。常用默认：SGD 0.01—0.1；Adam/AdamW 1e-4 到 3e-4；微调预训练模型 1e-5 到 5e-5。现代标配是 warmup（前 1—10% 步从 0 线性升到峰值）+ 衰减（cosine 或线性）。细节在[第09课](<09：学习率调度——warmup与衰减策略.md>)展开。

### 2.7 按架构选优化器（原课决策表）

| 在训什么 | 优化器 | 起手超参 |
|---|---|---|
| Transformer / LLM | AdamW | lr=1e-4，wd=0.01—0.1 |
| CNN / ResNet | SGD + momentum | lr=0.1，momentum=0.9 |
| GAN | Adam | lr=2e-4，β₁=0.5 |
| 微调预训练模型 | AdamW | lr=2e-5，wd=0.01 |
| 不知道选什么 | AdamW | lr=3e-4，wd=0.01 |

（补充：CNN 阵营偏爱 SGD 的理由是它常找到更平坦的极小值、泛化更好——平坦与否与泛化的关系详见[第07课](<07：正则化——L2、Dropout与早停.md>)的 BatchNorm 一节。）

## 三、操作前准备

Build It 全程只用 Python 标准库（math、random），无第三方依赖；Use It 需要 PyTorch：

~~~bash
python -m pip install torch
~~~

先激活你已有的 Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或运行任何训练。四个优化器类相互独立，Step 5 的网络依赖前面的 sigmoid 定义。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：裸 SGD

~~~python
# 本块中文说明：两行核心的最简优化器。
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
~~~

接口约定：`step(params, grads)` 原地修改参数列表。用 0.3 节的数字：params=[1.0]、grads=[0.3]、lr=0.1 → step 后 params=[0.97]。

### 4.2 第 2 步：动量 SGD

~~~python
# 本块中文说明：维护速度列表，历史梯度指数累加。
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)  # 惰性初始化
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
~~~

手算对照：两步梯度都 0.3 → v 依次 0.3、0.57，对应 0.3 节的 0.97、0.913。

### 4.3 第 3 步：Adam

~~~python
# 本块中文说明：一阶/二阶矩 + 偏差修正 + 自适应步长。
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1  # 步数计数器，偏差修正要用

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)  # 偏差修正
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
~~~

手算对照：0.3 节算过，恒定梯度 0.3 下第一步 w 从 1.0 到 0.999（没有偏差修正只有约 0.00003）。

### 4.4 第 4 步：AdamW

~~~python
# 本块中文说明：Adam 之后追加一行不经过自适应缩放的衰减。
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]  # 解耦衰减：直接作用于权重
~~~

注意最后一行的顺序：衰减用的是刚更新过的 w。与 Adam 的差异只有这一行（0.3 节：0.999 → 0.99899）。

### 4.5 第 5 步：四种优化器训练对比

同一两层网络（圆环数据、ReLU 隐藏、sigmoid 输出、BCE 损失）分别用四个优化器训练，比收敛。网络把全部参数展平成一个列表交给优化器，更新后再回填。

~~~python
# 本块中文说明：参数展平/回填 + 优化器接管更新步。
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        # 梯度顺序与 get_params 展平顺序一一对应
        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)  # 优化器原地改 params
                self.set_params(params)             # 回填网络

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
~~~

（核对/补充：原课页面到类定义为止，未展示四个优化器的运行调用。仿照前两课结构可补：）

~~~python
# 本块中文说明：（补充）原课未展示的对比运行入口。
data = make_circle_data()
for name, opt in [
    ("SGD", SGD(lr=0.1)),
    ("Momentum", SGDMomentum(lr=0.1, beta=0.9)),
    ("Adam", Adam(lr=0.01)),
    ("AdamW", AdamW(lr=0.01, weight_decay=0.01)),
]:
    print(f"\n=== Training with {name} ===")
    net = OptimizerTestNetwork(opt, hidden_size=8)
    net.train(data, epochs=300)
~~~

（补充说明：该入口的 lr 是本篇给的保守起点、未经原课确认，输出数值未经本机运行。另注意本网络是逐样本更新（batch size 1），Adam 类优化器在单样本噪声下的表现与常规小批量训练不完全相同。）

## 五、把代码运行起来：补充入口与检查

（补充）四个优化器是独立类，先于网络定义即可。以下最小检查不依赖随机数，全部可手算复核：

~~~python
# 本块中文说明：补充的手算复核入口（对应 0.3 节算例）。

# SGD：w=1.0，两步梯度 0.3
opt = SGD(lr=0.1)
p = [1.0]
opt.step(p, [0.3]); assert abs(p[0] - 0.97) < 1e-12
opt.step(p, [0.3]); assert abs(p[0] - 0.94) < 1e-12

# 动量：v1=0.3, v2=0.57
opt = SGDMomentum(lr=0.1, beta=0.9)
p = [1.0]
opt.step(p, [0.3]); assert abs(p[0] - 0.97) < 1e-12
opt.step(p, [0.3]); assert abs(p[0] - 0.913) < 1e-12  # 0.97 - 0.1*0.57

# 动量抵消震荡：+0.2 后 -0.2，第二步只回挪 0.002（v2 = 0.9*0.2-0.2 = -0.02）
opt = SGDMomentum(lr=0.1, beta=0.9)
p = [1.0]
opt.step(p, [0.2])
before = p[0]
opt.step(p, [-0.2])
assert abs(p[0] - before - 0.002) < 1e-9  # 第二步位移 = lr*|v2| = 0.002

# Adam：恒定梯度 0.3，第一步位移 ≈ lr
opt = Adam(lr=0.001)
p = [1.0]
opt.step(p, [0.3])
assert abs(p[0] - 0.999) < 1e-9  # 偏差修正后 m_hat/sqrt(v_hat)=1
opt.step(p, [0.3])
assert abs(p[0] - 0.998) < 1e-9

# AdamW：比 Adam 多缩 0.001*0.01*w
opt = AdamW(lr=0.001, weight_decay=0.01)
p = [1.0]
opt.step(p, [0.3])
assert abs(p[0] - 0.99899) < 1e-6  # 0.999 - 0.001*0.01*0.999 ≈ 0.998990
print("all optimizer checks passed")
~~~

补充检查方法：训练时打印 Adam 内部 `self.t` 与某个参数的 m/v，验证前 10 步偏差修正显著、之后趋近 1（本机未运行 4.5，结论为原课参考）。

## 六、Use It：使用现成工具

PyTorch 的优化器自带参数组、梯度裁剪接口与调度器：

~~~python
# 本块中文说明：PyTorch 标准训练循环的骨架。
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()                                  # 1. 清梯度
    output = model(torch.randn(32, 784))                   # 2. 前向
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))  # 3. 损失
    loss.backward()                                        # 4. 反向
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. 裁剪（可选）
    optimizer.step()                                       # 6. 更新
    scheduler.step()                                       # 7. 调度（可选）
~~~

顺序要背下来：zero_grad、forward、loss、backward、(clip)、step、(schedule)。调错顺序（比如 scheduler.step() 先于 optimizer.step()）是经典的隐蔽 bug。原课的实践建议：CNN 用 SGD+momentum（lr=0.1, momentum=0.9, wd=1e-4）配 step/cosine 调度；transformer/LLM 用 AdamW + warmup + cosine 衰减；没有测量过的理由，不要跟这些共识对着干。

## 七、失败模式、原文边界与工程注意

1. lr 不当仍是第一死因：Adam 也会发散，只是更宽容。损失 NaN/暴涨 → 先降 10 倍 lr；配合梯度裁剪（max_norm=1.0）可以救高 lr 训练（原课练习 4 的方向）。
2. Adam 不是总更快到“更好”：它收敛快，但泛化上 SGD+momentum 在 CNN 上常反超（平坦极小值假说）。别用“前 100 轮谁快”下结论。
3. 偏差修正写错（忘了 `self.t += 1` 或用了错误的幂）不报错，只是前几步更新异常小/大——0.3 节的手算就是抓这个的。
4. 4.5 的展平/回填是教学写法，真实框架里参数是引用共享的；自己复刻时若 get_params 返回的是拷贝，务必记得 set_params 回填（核对：原课正是这么做的）。
5. Adam 的 ε 不是可随便调的“安全余量”：ε 过大等价于给所有参数加同一个下限步长缩放，会破坏自适应（补充）。
6. 原文边界：原课说动量提速“3 倍量级（10000 步 vs 3000—5000 步）”——这是特定病态问题上的参考值，不是普适常数。
7. 逐样本（batch size 1）更新时二阶矩估计噪声很大，本课网络如此、真实训练很少如此（补充：把 4.5 改成小批量是很好的练习）。
8. 调参优先级牢记：lr → β₂ → （几乎从不动）β₁、ε。把力气花在刀刃上。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/prompt-optimizer-selector.md`（为任意架构选优化器与学习率的决策提示词）。本篇改写成自查模板，直接嵌在这里，不另建输出文件：

~~~markdown
检查优化器选择：说明模型类型对应的优化器与起手超参（Transformer→AdamW、CNN→SGD+momentum、GAN→Adam β₁=0.5）；学习率是多少、依据什么选定；是否有 warmup/衰减调度及其配置；是否启用梯度裁剪及阈值；训练循环顺序是否为 zero_grad→forward→loss→backward→(clip)→step→(schedule)；损失发散时先做什么（降 lr、加裁剪）写进预案。所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：实现 Nesterov 动量——在“前瞻点”$w - \text{lr}\cdot\beta v$ 而非当前位置求梯度，与标准动量比收敛。验收标准：能解释“先看前方的坡”与“先走再修正”的区别，并在圆环数据上给出两者的损失曲线对比。
2. 原课：实现 warmup 调度——前 10% 步线性升到 max_lr、之后 cosine 衰减到 0；对比 Adam 加与不加 warmup 到达 90% 准确率所需的轮数。验收标准：报告“到达 90% 的轮数”这一单一指标，并解释为什么初期大 lr 会伤害刚初始化的二阶矩估计。
3. 原课：跟踪 Adam 训练中每个参数的有效学习率 $\text{lr}\cdot\hat{m}/(\sqrt{\hat{v}}+\epsilon)$，画 10、50、200 步后的分布。验收标准：能回答“所有参数在以同样速度更新吗”——预期答案是否定的，并解释 m 与 v 的比值如何造成差异（原课参考方向）。
4. 原课：实现按全局范数的梯度裁剪（max_norm=1.0），用高学习率（Adam lr=0.01）在 10 个随机种子上数“多少次发散（损失变 NaN）”。验收标准：能展示裁剪显著降低发散次数（原课参考结论），并解释裁剪不改变梯度方向、只缩放幅度。
5. 原课：在权重初始化为 [−5, 5] 均匀分布的大权重网络上对比 Adam vs AdamW（weight_decay=0.1，200 轮），画权重 L2 范数随训练的变化。验收标准：能展示 AdamW 的权重范数下降更快（原课预期），并说明为什么 Adam 里的 L2 会被自适应缩放削弱。

## 十、术语与完成标准

学习率：更新步长的标量倍数，最重要的超参数；SGD：小批量梯度上 $w \leftarrow w-\text{lr}\cdot g$；动量：梯度的指数滑动平均，同向加速、反向抵消；RMSProp：按梯度平方滑动平均给每个参数调速；Adam：一阶矩+二阶矩+偏差修正；AdamW：衰减不经过自适应缩放的 Adam；偏差修正：除以 $(1-\beta^t)$ 补偿零初始化；权重衰减：每步按比例向 0 缩权重；学习率调度：warmup+cosine 是现代默认；梯度裁剪：范数超阈值时整体缩小；鞍点：高维损失面的大片平坦区。

完成检查：

- 能手算 SGD/动量/Adam/AdamW 各走两步（0.3 节算例）。
- 能写出偏差修正公式并算出“第一步 m̂ 恰好等于 g”。
- 能解释动量为什么抵消震荡方向、放大一致方向（并算出 +0.2/−0.2 的例子）。
- 能说出 Adam 与 AdamW 那一行差异及其后果。
- 能背出按架构选优化器的表格与训练循环的固定顺序。
- 知道调参优先级：lr → β₂ → 其他。

## 十一、验证范围与来源许可

原课 Build It 共 5 个分段，代码已逐字转录嵌入本篇（仅按本篇格式添加中文说明与注释，未改逻辑）；其中 4.5 的运行调用原课未展示，本篇以（补充）块显式补写并标注。Use It 的 PyTorch 代码同样保留。原课页面未展示运行输出，本篇未虚构“原课参考”数值；第五节“可手算复核”的断言均对应 0.3 节算例。本次未执行训练或下载任何数据/模型。写作完成后，将正文各 python 代码块抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 逐一做静态语法检查，结果见下方记录。

本次验证记录（2026-11-06）：
- 抓取课程页一次成功，以原文为准整理；未使用凭记忆的内容。
- 正文 8 个 python 代码块（4.1—4.5 含补充入口、第五、六节）全部通过 `ast.parse` 静态语法检查；第五节手算断言（SGD 0.97/0.94、动量 0.97/0.913、Adam 0.999/0.998、AdamW 0.99899）在临时脚本中实际执行复核通过（属玩具断言检查，不构成训练或性能实验）。
- 未做：4.5 的四优化器训练对比、warmup 实验、梯度裁剪种子实验。

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
