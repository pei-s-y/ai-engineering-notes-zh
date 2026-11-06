# 05：损失函数——MSE、交叉熵与选择

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/05-loss-functions) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/05-loss-functions/docs/en.md)。原题：Loss Functions。

## 零、初学者先读：模型唯一真正优化的那个数

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第04课](<04：激活函数——非线性、ReLU与饱和区.md>)给了网络非线性表达能力，本课决定“往哪个方向学”；[第06课](<06：优化器——从SGD到Adam.md>)的优化器只是让这个数变小的机器。

### 0.1 先理解这一课在做什么

模型真正优化的不是准确率、不是 F1，只有损失函数（loss function）这一个数：优化器拿它的梯度去调权重。损失函数没写对，模型就会找到“数学上最省钱”的方式去满足它——而那几乎从来不是你想要的。原课的招牌例子：二分类（各占一半）用 MSE 训练，模型发现“对一切都预测 0.5”就能把平均 MSE 压到 0.25 这个“不学任何东西也能达到的最小值”；换成交叉熵（cross-entropy），预测 0.5 的损失是 −log(0.5)=0.693（很贵），预测 0.99 且对了只有 0.01（很便宜）——模型被逼着把预测推向 0 或 1。选损失 = 定义“错”的含义，是分类问题里和选架构同等重要的决定。本课还要顺带走三个进阶损失：标签平滑、对比损失、焦点损失。

### 0.2 本课术语先翻成白话

- 损失函数（loss function）=把“预测和真实的差距”写成一个可导的数，越小越好
- MSE（均方误差）=误差平方的平均，回归默认
- 梯度（gradient）=损失对预测（或参数）的导数，指出“怎么改能让损失变小”
- 交叉熵（cross-entropy）=用 −log(预测概率) 惩罚错误，分类的标准损失
- BCE（二元交叉熵，Binary Cross-Entropy）=两个类版本的交叉熵
- CCE（分类交叉熵，Categorical Cross-Entropy）=多类版本，配 one-hot 标签
- one-hot（独热编码）=类别标签写成 [0,0,1,0] 这种“只有正确类是 1”
- logit=softmax/sigmoid 之前的原始分数
- 数值裁剪（clipping）=把概率夹在 [ε, 1−ε] 内，避免 log(0) 算出 −∞
- 标签平滑（label smoothing）=把硬标签 0/1 软化成 0.01/0.91 之类，防过度自信
- 对比损失（contrastive loss）=没有标签时用“相似/不相似”定义的损失，拉近正对、推开负对
- InfoNCE / NT-Xent=对比损失的“分类化”版本：在一批相似度上做 softmax 交叉熵
- 温度（temperature）=除在相似度/logit 上的标量，越小分布越尖锐
- triplet 损失（triplet loss）=锚点-正例-负例三元组，要求正例比负例至少近一个 margin
- 焦点损失（focal loss）=交叉熵乘 $(1-p_t)^\gamma$，压低简单样本的权重，处理类别不平衡

### 0.3 用小数字走一遍

两条数据：预测 [0.9, 0.3]，真实 [1.0, 0.0]（第一条是正类、第二条是负类）。

**MSE。** 误差：0.9−1=−0.1，0.3−0=0.3。MSE = ((−0.1)² + 0.3²)/2 = (0.01+0.09)/2 = **0.05**。对每条预测的梯度 = 2(p−t)/n：第一条 2×(−0.1)/2 = −0.1（预测要变大），第二条 2×0.3/2 = 0.3（预测要变小）。注意梯度与误差成**线性**比例。

**BCE。** 第一条：y=1，p=0.9，损失 = −ln(0.9) ≈ 0.105；第二条：y=0，p=0.3，损失 = −ln(1−0.3) = −ln(0.7) ≈ 0.357。平均 ≈ (0.105+0.357)/2 ≈ **0.231**。梯度 = −y/p + (1−y)/(1−p)：第一条 −1/0.9 ≈ −1.111；第二条 1/0.7 ≈ 1.429。同样大小 0.1 的“差 0.1 没到 1”与“多 0.3 越过 0”，交叉熵给的梯度比 MSE 狠得多——这就是它纠正自信错误的速度来源。

**“预测 0.5”惨案。** 真值 0 或 1、预测恒 0.5：每条 MSE = 0.25（不学习也能达到）；每条 CE = −ln(0.5) ≈ 0.693，交叉熵不奖励偷懒。若预测 0.01 而真值是 1：CE = −ln(0.01) = 4.605，是 0.693 的 6.6 倍——自信地错，重罚。

**softmax + CCE。** 三个类的 logit [2, 0, 0]，正确类是第 0 类：$e^2≈7.389$，$e^0=1$、$e^0=1$，和 9.389，概率 ≈ [0.787, 0.107, 0.107]。CCE = −ln(0.787) ≈ 0.239。梯度有个漂亮的化简：概率 − one-hot = [0.787−1, 0.107, 0.107] = [−0.213, 0.107, 0.107]。正确类的 logit 要加 0.213、其余各减 0.107（三者和为 0，softmax 的和不变性质）。

**标签平滑。** 10 类、α=0.1：正确类目标 = 0.9+0.1/10 = 0.91，其余每类 = 0.01。模型不再被迫把 logit 推向无穷大去逼近概率 1。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：Step 1 的 `mse`/`mse_gradient`（0.3 节第一段）；Step 2 的 `binary_cross_entropy`/`bce_gradient`（注意 eps 裁剪）；Step 3 的 softmax+CCE 和它的“梯度=概率−one-hot”化简；Step 4 标签平滑版 CCE；Step 5 简化 InfoNCE（先看 `cosine_similarity`，再看减最大值的 softmax 技巧）；Step 6 是把第04课的网络分别用 MSE 和 BCE 训练的对比类。所有函数都吃普通列表，先手算再读代码，每个数字都能对上。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么“用 MSE 做二分类”会收敛到“全预测 0.5”？

**参考答案：** 两类各占一半时，0.5 是让平方误差期望最小的常数预测：任何偏离都会在一半样本上产生更大误差。MSE 对“自信错误”只给线性惩罚，模型没有动力冒险；交叉熵用 −log 把自信错误（p→0 而真值 1）的损失推到无穷大，逼模型表态。

**问题2：** 为什么 `binary_cross_entropy` 里要 clip？

**参考答案：** 模型可能输出恰好 0 或 1（浮点舍入或 sigmoid 饱和），而真值是另一边时 −log(0) = −∞，损失直接爆掉。裁剪到 [1e-15, 1−1e-15] 给了个有限值，梯度也存在。

**问题3：** softmax+交叉熵的梯度为什么恰好是“概率减 one-hot”？

**参考答案：** 这是两个函数的导数相乘后大量项相消的结果（分母的指数和求导与分子相消）。正因为这个简洁性，两者总是配对使用——这也是 PyTorch 用 `F.cross_entropy` 直接吃 logit 的原因之一。

能对上这组小数字后，进入下面的推导与代码。

---

## 一、这一课解决什么问题？

网络能算出预测了，接下来要回答：这个预测有多错？这个数决定优化器把权重往哪推。损失是模型唯一真正优化的东西；报告给老板的准确率只是副产品。选错损失有三种典型后果：（1）分类用 MSE → “全预测 0.5”的躺平最优解；（2）不平衡数据用普通交叉熵 → 被海量简单样本淹没；（3）自监督里对比损失设计错 → 所有嵌入塌缩成同一个点，损失为零、模型无用。本课从零实现 MSE、BCE、CCE、InfoNCE 并做 MSE vs 交叉熵的训练对比。

## 二、核心概念与推导

### 2.1 MSE：回归默认

$$\text{MSE}=\frac{1}{n}\sum(y_{\text{pred}}-y_{\text{true}})^2,\qquad \frac{\partial\text{MSE}}{\partial y_{\text{pred}}}=\frac{2}{n}(y_{\text{pred}}-y_{\text{true}})$$

平方让大误差被平方级放大：误差 2 的代价是误差 1 的 4 倍、误差 10 是 100 倍。代价是离群点敏感——房价预测里大多数房子差 1 万、一栋豪宅差 20 万时，MSE 会过度迁就那一栋，可能伤到其余 99 栋（原课的“真实数字”示例）。梯度对误差是线性的：大误差给大梯度，回归里这是特性（大错需要大修正）。

### 2.2 二元交叉熵：分类的标准答案

$$\text{BCE}=-\left(y\log p+(1-y)\log(1-p)\right),\qquad \frac{\partial\text{BCE}}{\partial p}=-\frac{y}{p}+\frac{1-y}{1-p}$$

−log(p) 的威力：真值 1、预测 0.99 → 损失 0.01；预测 0.01 → 损失 4.6（460 倍差距）。梯度更直白：真值 1、p 趋 0 时梯度是 −1/p 趋 −∞——模型收到一个巨大的“快改”信号；已经对了则梯度趋 0，不再瞎动。

### 2.3 分类交叉熵：多类版本

$$\text{CCE}=-\sum_i y_i\log p_i$$

one-hot 标签下只有正确类贡献损失。10 类问题：正确类概率 0.1（瞎猜）→ 损失 −log(0.1)=2.3；概率 0.9 → 0.105。模型学会把概率质量集中到正确答案上。

### 2.4 为什么 MSE 在分类上失灵

两个原因叠加。第一（0.3 节的“0.5 惨案”）：MSE 允许躺平解。第二：MSE 梯度经过 sigmoid 时还要乘一次 sigmoid 导数，而 sigmoid 在输出接近 0 或 1 时饱和（导数趋 0），错误越自信、修正信号反而越弱。交叉熵的 −log 恰好抵消 sigmoid 的平坦区：p 趋 0 时 −log 的 1/p 发散盖过饱和，梯度该大的时候就是大。原课的表述：MSE 的梯度在饱和区变平 → 修正慢；CE 的梯度在错误答案附近爆发 → 修正快。

### 2.5 标签平滑：不许过度自信

$$\text{smooth}=(1-\alpha)\cdot\text{one\_hot}+\frac{\alpha}{K}$$

硬标签要求“这是 100% 的第 3 类”。要经 softmax 输出恰好 1.0，logit 必须推向无穷——导致过度自信、泛化差、对分布偏移脆。α=0.1、10 类时目标变成“正确类 0.91、其余各 0.01”，logit 被封在合理范围。GPT 等现代模型都在用等价机制（原课陈述）。

### 2.6 对比损失：没有标签时的学习信号

自监督学习没有标签，只有“这两个视图来自同一张图”（正对）与“其他都算负对”。SimCLR 式 InfoNCE：

$$L=-\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_k\exp(\text{sim}(z_i,z_k)/\tau)}$$

sim 是余弦相似度、τ 是温度（SimCLR 默认 0.07）、分母对批内所有负对求和（批 256 → 255 个负对）。形式上就是“在相似度上做 softmax 交叉熵”：让正对的相似度在 256 个选项里拿第一。温度越小，分布越尖锐，负对被推得越狠。triplet 损失是另一个路线：

$$L=\max\left(0,\ d(a,p)-d(a,n)+\text{margin}\right)$$

margin（典型 0.2—1.0）规定正例必须比负例至少近多少；负例已经够远时损失为 0、无梯度——高效，但需要仔细的三元组挖掘（triplet mining，挑“难负例”）。

### 2.7 焦点损失：类别不平衡的解

$$\text{FL}=-\alpha(1-p_t)^\gamma\log p_t$$

$p_t$ 是正确类的预测概率。γ=0 退化为普通交叉熵；γ=2（默认）时：简单样本 $p_t=0.9$ → 权重 $(0.1)^2=0.01$（近乎忽略）；困难样本 $p_t=0.1$ → 权重 $(0.9)^2=0.81$（全力学习）。Lin 等为目标检测提出：99% 候选区域是背景（简单负例），不用焦点损失模型会淹死在简单样本里。

### 2.8 选择速查（原课决策树的表格化，补充排版）

| 任务 | 首选 | 备注 |
|---|---|---|
| 回归、要罚离群点 | MSE | 平方放大大误差 |
| 回归、要抗离群点 | MAE / Huber | 线性/混合惩罚 |
| 二分类 | BCE | 配 sigmoid 输出 |
| 多分类 | CCE | 配 softmax，one-hot |
| 多分类+过度自信 | CCE + 标签平滑 | α=0.1 起步 |
| 类别不平衡 | Focal Loss | γ=2 起步 |
| 成对数据学嵌入 | 对比损失 | InfoNCE/NT-Xent |
| 有三元组 | Triplet | 注意难例挖掘 |
| 大批自监督 | InfoNCE | 温度是关键旋钮 |

## 三、操作前准备

Build It 全程只用 Python 标准库（math、random），无第三方依赖；Use It 需要 PyTorch：

~~~bash
python -m pip install torch
~~~

先激活你已有的 Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或运行任何训练。各步函数相互独立定义，Step 6 依赖 Step 内部定义的 sigmoid 与数据生成器。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：MSE 及其梯度

~~~python
# 本块中文说明：MSE 与逐预测的梯度。
def mse(predictions, targets):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / n

def mse_gradient(predictions, targets):
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        grads.append(2.0 * (p - t) / n)  # d/dp：误差的线性函数
    return grads
~~~

用 0.3 节的两条数据手算：mse([0.9, 0.3], [1.0, 0.0])=0.05，梯度 [−0.1, 0.3]。

### 4.2 第 2 步：二元交叉熵

log(0) 问题是真实的：正样本被预测成恰好 0 时 log(0)=−∞。裁剪防住它。

~~~python
# 本块中文说明：BCE 与逐预测的梯度，含 eps 裁剪。
import math

def binary_cross_entropy(predictions, targets, eps=1e-15):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))  # 夹进 [eps, 1-eps]
        total += -(t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped))
    return total / n

def bce_gradient(predictions, targets, eps=1e-15):
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        grads.append(-(t / p_clipped) + (1 - t) / (1 - p_clipped))
    return grads
~~~

手算对照：bce([0.9, 0.3], [1.0, 0.0]) ≈ 0.231，梯度 ≈ [−1.111, 1.429]。

### 4.3 第 3 步：softmax + 分类交叉熵

先 softmax 把 logit 变概率，再对 one-hot 目标算交叉熵。注意那个著名的化简。

~~~python
# 本块中文说明：softmax + CCE，以及“梯度=概率-one_hot”的化简。
def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]  # 减最大值防溢出
    total = sum(exps)
    return [e / total for e in exps]

def categorical_cross_entropy(logits, target_index, eps=1e-15):
    probs = softmax(logits)
    p = max(eps, probs[target_index])
    return -math.log(p)

def cce_gradient(logits, target_index):
    probs = softmax(logits)
    grads = list(probs)
    grads[target_index] -= 1.0  # softmax+CE 联合求导后只剩这个
    return grads
~~~

手算对照：logit [2,0,0]、目标 0 类 → 概率 ≈ [0.787, 0.107, 0.107]，损失 ≈ 0.239，梯度 ≈ [−0.213, 0.107, 0.107]。这个化简不是巧合——正是 softmax 与交叉熵总是配对的原因。

### 4.4 第 4 步：标签平滑

~~~python
# 本块中文说明：标签平滑版 CCE：正确类目标是 1-α+α/K，其余 α/K。
def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    probs = softmax(logits)
    loss = 0.0
    for i in range(num_classes):
        if i == target_index:
            smooth_target = 1.0 - alpha + alpha / num_classes
        else:
            smooth_target = alpha / num_classes
        p = max(eps, probs[i])
        loss += -smooth_target * math.log(p)
    return loss
~~~

（核对：目标分布的总和 = (1−α+α/K) + (K−1)·α/K = 1 − α + α·K/K... 展开即 (1−α)·1 + α·(1/K)·K = 1，仍是一个合法分布；10 类 α=0.1 时正确类 0.91、其余各 0.01，与 2.5 节一致。）

### 4.5 第 5 步：简化 InfoNCE 对比损失

~~~python
# 本块中文说明：余弦相似度 + 减最大值的稳定 softmax + InfoNCE。
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    max_sim = max(sim_pos, max(sim_negs)) if sim_negs else sim_pos
    exp_pos = math.exp(sim_pos - max_sim)  # 减最大值防 exp 溢出
    exp_negs = [math.exp(s - max_sim) for s in sim_negs]
    total_exp = exp_pos + sum(exp_negs)

    return -math.log(max(1e-15, exp_pos / total_exp))
~~~

结构上这就是 4.3 的 CCE：把“正对的相似度”当成正确类、批内负对当成其他类，在相似度上做 softmax 交叉熵。温度 τ=0.07 意味着相似度被放大 14 倍后进入 softmax——分布尖锐、惩罚凶狠。

### 4.6 第 6 步：MSE vs 交叉熵训练对比

同一个网络（第04课的圆环数据、ReLU 隐藏层、sigmoid 输出）分别用两种损失训练，观察交叉熵收敛更快。

~~~python
# 本块中文说明：同一网络、两种损失的对比实验类。
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


class LossComparisonNetwork:
    def __init__(self, loss_type="bce", hidden_size=8, lr=0.1):
        random.seed(0)
        self.loss_type = loss_type
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
            self.h.append(max(0.0, z))  # 隐藏层用 ReLU

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        if self.loss_type == "mse":
            d_loss = 2.0 * (self.out - target)
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, self.out))
            d_loss = -(target / p) + (1 - target) / (1 - p)

        d_sigmoid = self.out * (1 - self.out)  # MSE 路线：饱和时这一项拖后腿
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return (pred - target) ** 2
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += self.compute_loss(pred, y)
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
~~~

（核对/补充：原课页面到类定义为止，没有给出运行两种配置的调用代码。仿照第04课 Step 5 的写法，可补一个入口：）

~~~python
# 本块中文说明：（补充）原课未展示的对比运行入口，仿第04课结构补写。
data = make_circle_data()
for loss_type in ["mse", "bce"]:
    print(f"\n=== Training with {loss_type.upper()} ===")
    net = LossComparisonNetwork(loss_type=loss_type, hidden_size=8, lr=0.1)
    net.train(data, epochs=200)
~~~

（补充说明：这个补充入口与原课练习方向一致，但其输出数值未经本机运行，属待验证；原课正文声称的“交叉熵收敛更快”标为原课参考。另注意 BCE 分支里 `d_loss` 乘 `d_sigmoid` 后恰好等于 $(p-t)$——这正是 sigmoid+BCE 的著名化简，而 MSE 分支保留 $2(p-t)\cdot p(1-p)$，饱和时被压扁。）

## 五、把代码运行起来：补充入口与检查

（补充）原课各步按 4.1→4.6 顺序执行即可；4.6 的补充入口依赖前面的类定义。以下最小检查不依赖随机数，全部可手算复核：

~~~python
# 本块中文说明：补充的手算复核入口。
import math

# MSE（0.3 节算例）
assert abs(mse([0.9, 0.3], [1.0, 0.0]) - 0.05) < 1e-12
assert all(abs(a - b) < 1e-9 for a, b in zip(mse_gradient([0.9, 0.3], [1.0, 0.0]), [-0.1, 0.3]))

# BCE（0.3 节算例）
assert abs(binary_cross_entropy([0.9, 0.3], [1.0, 0.0]) - 0.2311) < 1e-3
assert all(abs(a - b) < 1e-3 for a, b in zip(bce_gradient([0.9, 0.3], [1.0, 0.0]), [-1.1111, 1.4286]))

# 裁剪生效：预测恰好 0/1 不产生 inf
assert math.isfinite(binary_cross_entropy([0.0, 1.0], [1.0, 0.0]))

# softmax + CCE（0.3 节算例）
probs = softmax([2.0, 0.0, 0.0])
assert abs(sum(probs) - 1.0) < 1e-12
assert abs(categorical_cross_entropy([2.0, 0.0, 0.0], 0) - 0.2390) < 1e-3
grads = cce_gradient([2.0, 0.0, 0.0], 0)
assert abs(sum(grads)) < 1e-12  # 梯度总和为 0（softmax 的性质）
assert abs(grads[0] - (probs[0] - 1.0)) < 1e-12

# 标签平滑 CCE：3 类、α=0.1、logit [1,0,0]（真实类 0）→ 约 0.618（可手算复核）
assert abs(label_smoothed_cce([1.0, 0.0, 0.0], 0, 3, alpha=0.1) - 0.6181) < 1e-3
print("all loss checks passed")
~~~

补充检查方法：对 4.6 分别记录两种损失的最终准确率与损失曲线（本机未运行，数值属原课参考）；画 loss 曲线时应看到 BCE 在前几十轮下降明显快于 MSE（原课声称）。

## 六、Use It：使用现成工具

PyTorch 内置全部标准损失且自带数值稳定处理：

~~~python
# 本块中文说明：PyTorch 的标准损失函数。
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.tensor([0.9, 0.1, 0.7], requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])

mse_loss = F.mse_loss(predictions, targets)
bce_loss = F.binary_cross_entropy(predictions, targets)

logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 9])
ce_loss = F.cross_entropy(logits, labels)  # 直接吃 logit，内部做 log-softmax
ce_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)
~~~

原课的关键工程建议：用 `F.cross_entropy` 而不是“先手动 softmax 再 `F.nll_loss`”。前者把 log-softmax 和负对数似然合成一次数值稳定的运算；分开做会在大指数相减中丢精度。对比学习大多用自定义实现或 lightly、pytorch-metric-learning 这类库，核心循环永远一样：算两两相似度 → 正负对上做 softmax → 反向传播。

## 七、失败模式、原文边界与工程注意

1. 分类任务用 MSE 是本课第一反面教材：躺平解 0.5 + 饱和区梯度被压扁，双重减速。诊断特征：损失早早停在 0.25 附近不动。
2. 忘记 clip：一个 log(0) 让整个训练 NaN。自写损失时 eps=1e-15 是标配。
3. 焦点损失不是免费午餐：γ 调太猛会让模型忽略一切它已经差不多会的样本，训练信号变稀（补充提醒）。
4. 对比损失的温度是敏感旋钮：τ 太小梯度尖峰、训练不稳；太大区分度不足。SimCLR 用 0.07 是在特定 batch size 下调出来的（原课参考）。
5. 嵌入塌缩是对比损失的独特失败模式：所有输出同一个向量，损失技巧性地归零、模型完全无用。InfoNCE 靠批内负对抗塌缩，batch 越小风险越大（补充：这是对比学习偏爱大 batch 的原因之一）。
6. 原文边界：原课说“MSE 梯度在分类上变平”——严格说变平的是 sigmoid 输出层的导数与 MSE 的乘积；纯 MSE 对预测 p 本身梯度仍是线性的。理解时要把“损失对 logit 的梯度”当作完整链条。
7. 标签平滑的 α 也不是越大越好：过大的 α 等于故意教模型出错，会伤最终准确率（补充）。
8. triplet 的 margin 决定“要学到的间隔”：margin=0 时只要正例比负例近一丁点就停；需要鲁棒间隔的场景要显式调大（补充）。

## 八、Ship It：本课可复用的中文检查模板

原课产出两个文件：`outputs/prompt-loss-function-selector.md`（选损失的提示词）与 `outputs/prompt-loss-debugger.md`（损失曲线诊断提示词）。本篇合并改写成自查模板，直接嵌在这里，不另建输出文件：

~~~markdown
检查损失函数选择：说明任务类型与所选损失及理由（回归/二分类/多分类/嵌入学习各对应什么）；是否讨论过 MSE 用于分类的失败模式；概率路径上有没有做 log(0) 防护（clip 或 log-softmax）；类别是否平衡、是否需要 focal loss；是否需要标签平滑及 α 取值；对比学习要说明温度与负对来源。损失曲线异常时：先查数值稳定性（NaN/inf），再查标签编码是否与损失配套（one-hot 对 CCE、0/1 对 BCE），最后才怀疑模型。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：实现 Huber 损失（小误差时 MSE、大误差时 MAE），在 $y=\sin(x)$ 回归上对 5% 加随机噪声的目标比较 MSE 与 Huber 的最终测试误差。验收标准：能展示 Huber 对离群点的损失曲线明显更平、最终测试误差更低（原课预期方向，具体数值以自己运行为准）。
2. 原课：把焦点损失加进二分类训练循环，构造 90%/10% 不平衡数据，比较 BCE 与 focal（γ=2）在 200 轮后的少数类召回率。验收标准：报告少数类 recall 而非总体准确率（总体准确率会被多数类淹没），且能解释 $(1-p_t)^\gamma$ 如何把梯度集中到难样本。
3. 原课：实现带半难负例挖掘（semi-hard negative mining）的 triplet 损失：为 5 个类生成二维嵌入，对每个锚点选“比正例远但最靠近”的负例，与随机选负例比收敛。验收标准：能解释为什么半难负例既不像随机那么松、又不像最难负例那么容易把训练带崩。
4. 原课：重跑 MSE vs 交叉熵对比并逐层记录梯度范数随轮数的变化。验收标准：验证“训练早期（模型最不确定时）交叉熵产生更大梯度”（原课参考结论），画平均梯度范数曲线。
5. 原课：实现 KL 散度损失，验证真分布为 one-hot 时其梯度与交叉熵一致；再试软目标（知识蒸馏：教师模型的 softmax 输出当真值）。验收标准：one-hot 情形两种梯度数值相同（差一个与参数无关的常数项），软目标时能写出 KL 与 CE 的关系式。

## 十、术语与完成标准

损失函数：把预测与目标的差映射成一个可导标量的函数，模型唯一真正优化的对象；MSE：误差平方平均，回归默认、离群敏感；BCE：二类交叉熵 $-(y\log p+(1-y)\log(1-p))$；CCE：多类交叉熵 $-\sum y_i\log p_i$；one-hot：只有正确类为 1 的标签编码；logit：归一化前的原始分数；裁剪：把概率限制在 [ε,1−ε] 防 log(0)；标签平滑：把硬目标软化成 0.91/0.01 类分布；对比损失：拉近正对、推开负对；InfoNCE：相似度上的温度化 softmax 交叉熵；温度：控制分布尖锐度的除数；triplet 损失：锚-正-负三元组加 margin 的间隔学习；焦点损失：$(1-p_t)^\gamma$ 加权的交叉熵。

完成检查：

- 能手算两条数据的 MSE 与 BCE 及各自梯度（0.3 节算例）。
- 能解释“为什么 MSE 分类会躺平在 0.5”并算出 0.25 与 0.693 两个数。
- 能写出 softmax+CCE 梯度的化简并验证梯度总和为 0。
- 能算出 10 类 α=0.1 标签平滑后的目标分布。
- 能说出三类任务（回归/分类/嵌入）各自的默认损失和一个进阶选项。
- 能列出至少三个损失函数的失败模式（log(0)、嵌入塌缩、简单样本淹没）。

## 十一、验证范围与来源许可

原课 Build It 共 6 个分段，代码已逐字转录嵌入本篇（仅按本篇格式添加中文说明与注释，未改逻辑）；其中 4.6 的运行入口原课未展示，本篇以（补充）块显式补写并标注。Use It 的 PyTorch 代码同样保留。原课页面未展示运行输出，本篇未虚构“原课参考”数值；第五节“可手算复核”的断言均对应 0.3 节算例。本次未执行训练或下载任何数据/模型。写作完成后，将正文各 python 代码块抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 逐一做静态语法检查，结果见下方记录。

本次验证记录（2026-11-06）：
- 抓取课程页一次成功，以原文为准整理；未使用凭记忆的内容。
- 正文 9 个 python 代码块（4.1—4.6 含补充入口、第五、六节）全部通过 `ast.parse` 静态语法检查；第五节手算断言在临时脚本中实际执行复核通过（属玩具断言检查，不构成训练或性能实验）。
- 未做：4.6 的训练对比、不平衡数据实验、KL 散度验证。

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
