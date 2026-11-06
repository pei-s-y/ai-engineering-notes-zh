# 07：正则化——L2、Dropout与早停

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/07-regularization) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/07-regularization/docs/en.md)。原题：Regularization。

## 零、初学者先读：给模型交“复杂税”

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第06课](<06：优化器——从SGD到Adam.md>)让模型能学了，本课解决“学过头”：训练 99%、测试 60%，模型把训练集背下来了。下一课[第08课](<08：权重初始化——零、随机、Xavier与Kaiming.md>)的初始化与本课的 BatchNorm 配合最紧密。

### 0.1 先理解这一课在做什么

足够大的网络能把任何数据集背下来——这不是比喻：Zhang 等（2017）给 ImageNet 换上随机标签，标准网络照样把训练损失压到近零，背下了一百万对毫无规律的输入输出。训练完美、测试归零。这就是过拟合（overfitting）：训练分和测试分之间的沟（泛化差距，generalization gap）是它的体温计。正则化（regularization）就是给“复杂”征税，逼模型走“记住规律、忘掉噪声”的路：Dropout 随机掐掉神经元逼出冗余表示；权重衰减不许任何权重长太大；批归一化让损失面更平滑；早停在过拟合之前刹车。本课全部从零实现，并给出“差距多大用多重的药”的剂量表。

### 0.2 本课术语先翻成白话

- 过拟合（overfitting）=训练表现远好于测试表现：背了题没学会；欠拟合（underfitting）=训练分也低：模型太简单，规律都没抓住
- 泛化差距（generalization gap）=训练准确率 − 测试准确率，本课的核心监控量；正则化（regularization）=一切“压复杂度换泛化”的手段的统称
- Dropout=训练时以概率 p 随机把神经元输出置零，测试时全开；反向缩放（inverted dropout）=训练时就把留下的输出除以 (1−p)，测试时代码完全不用改
- 共适应（co-adaptation）=神经元之间形成“你死我活”的依赖；Dropout 就是打破它
- L2 正则化 / 权重衰减（weight decay）=损失里加 $\frac{\lambda}{2}\sum w^2$，每步把权重按比例往 0 缩
- 批归一化（Batch Normalization，BatchNorm）=对每个特征按“当前小批量”的均值方差归一化；层归一化（Layer Normalization，LayerNorm）=对每个样本按“自己的特征维度”归一化，与 batch 大小无关
- RMSNorm=LayerNorm 去掉减均值那一步，快约 10%，LLaMA/Mistral 在用
- γ、β（gamma/beta）=归一化层的可学习缩放与平移参数，允许网络“撤销”归一化；运行统计（running statistics）=BatchNorm 训练时攒下的均值/方差滑动平均，推理时用
- 数据增强（data augmentation）=改输入不改标签（翻转/裁剪/噪声），等效扩数据集；早停（early stopping）=验证损失不再下降就停并回滚到最好权重；patience（耐心窗口）=宣布“没救”前允许连续没进步的轮数，典型 5—20

### 0.3 用小数字走一遍

**L2 惩罚。** 三个权重 [2.0, −1.0, 0.5]，λ=0.1。惩罚项 = $\frac{\lambda}{2}\sum w^2$ = 0.1/2 × (4 + 1 + 0.25) = 0.05 × 5.25 = **0.2625**。惩罚的梯度 = λw = [0.2, −0.1, 0.05]——注意权重 2.0 每步被“拽”0.2，权重 0.5 只被拽 0.05：越大的权重税越重，这就是“大权重更可疑”的实现方式。加上任务梯度 g 后，更新变成 $w \leftarrow w - \text{lr}(g + \lambda w)$。

**Dropout（p=0.5，反向缩放）。** 4 个神经元输出 [1, 2, 3, 4]，某次抽到的掩码 [1, 0, 0, 1]（每个神经元独立以 0.5 概率存活）。训练时输出 = [1/0.5, 0, 0, 4/0.5] = **[2, 0, 0, 8]**。存活神经元的值被放大一倍，是为了“期望不变”：每个位置训练时的期望输出 = 0.5×(值/0.5) + 0.5×0 = 原值。于是测试时直接原样输出、无需任何缩放。验证：另一次抽到 [0, 1, 1, 0] → [0, 4, 6, 0]，网络被迫用不同子集完成同一个任务——冗余表示由此产生。

**BatchNorm。** 两个样本、两个特征的小批量 [[1, 2], [3, 4]]。逐特征统计：特征 1 均值 = (1+3)/2 = 2、方差 = ((−1)²+1²)/2 = 1；特征 2 均值 = 3、方差 = 1。归一化后 = **[[−1, −1], [1, 1]]**。注意统计是“竖着算”的（跨样本、每列一个均值）——这就是它离不开 batch 的原因：batch=1 时方差没意义。

**LayerNorm 与 RMSNorm。** 单个样本 x=[1, 2, 3]：均值 = 2、方差 = 2/3 ≈ 0.667、标准差 ≈ 0.816，LayerNorm 归一化 ≈ **[−1.225, 0, 1.225]**；RMSNorm 不减均值，rms = $\sqrt{(1+4+9)/3}$ = $\sqrt{4.667}$ ≈ 2.160，输出 = x/rms ≈ **[0.463, 0.926, 1.389]**。统计都是“横着算”的（每个样本自己算），batch 多大都无所谓——这就是 transformer 选它们的原因；RMSNorm 还不学 β，是三档里最便宜的。

**早停的读法。** 训练分一路涨到 99%，验证分在第 60 轮到顶 91% 后回落：应停在第 60 轮附近并加载那一份权重；patience=20 是允许验证损失连续 20 轮没进步才宣布放弃，防被小波动骗停。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：Step 1 的 `Dropout` 类（forward 里看掩码和 / (1−p)，backward 里看梯度同样被掩码和缩放）；Step 2 的 `l2_regularization`/`l2_gradient`（两个各四行的函数）；Step 3 的 `BatchNorm`（训练/推理分支 + running 统计的滑动更新）；Step 4 的 `LayerNorm` 与 Step 5 的 `RMSNorm`（对比 0.3 节的横竖统计）；Step 6 的 `RegularizedNetwork`（把 Dropout、L2 装进第05课的网络，并额外维护测试集评估——第一次出现 train/test 双轨记录）。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 Dropout 要在训练时除以 (1−p)，而不是测试时乘 (1−p)？
**参考答案：** 两者数学等价（让期望输出匹配），但工程上“训练时缩放”让推理代码完全不知道 Dropout 的存在——部署简单、不会忘。忘掉测试侧缩放的经典 bug 在“训练时缩放”方案里被结构性消灭了。

**问题2：** BatchNorm 和 LayerNorm 的“统计方向”分别是什么？各自因此怕什么？
**参考答案：** BatchNorm 跨样本、逐特征算（竖着）；怕 batch 小：batch=1 统计无意义，<32 就噪。LayerNorm 跨特征、逐样本算（横着）；不怕 batch 小、不怕序列变长，训练/推理计算完全一致——所以 transformer 用它。

**问题3：** 权重衰减为什么“偏爱”小权重？它和过拟合有什么关系？
**参考答案：** 梯度 λw 与权重自身成比例：大权重受强拽、小权重几乎不动。过拟合模型往往靠少数大权重放大训练集噪声；衰减压制大权重，等于压缩模型的有效容量，逼它依赖稳健特征而非记住怪癖。（注意与[第06课](<06：优化器——从SGD到Adam.md>)的衔接：Adam 下要用解耦的 AdamW 版本衰减。）

能对上这组小数字后，进入下面的推导与代码。

---

## 一、这一课解决什么问题？

模型参数越多、记性越好，越容易把训练集整个背下来。Zhang 等（2017）的随机标签实验证明了容量上限：GPT-3 有 1750 亿参数对约 5000 亿 token，足够把训练数据大段背下来。没有正则化，模型会“复读训练样本”而不是学可泛化的模式。本课的学习目标：从零实现带反向缩放的 Dropout、L2 权重衰减、BatchNorm、LayerNorm、RMSNorm；测量 train/test 差距并诊断过拟合；解释 transformer 为什么用 LayerNorm 而不是 BatchNorm、现代 LLM 为什么偏爱 RMSNorm；并按过拟合严重程度开药方。

## 二、核心概念与推导

### 2.1 过拟合谱：从欠到过

每个模型都坐在“欠拟合—合适—过拟合”谱上某个位置：欠拟合时训练分测试分都低（模型太简单），合适时两边都高且接近（如原课参考：train 95% / test 92%），过拟合时训练接近满分、测试塌方（train 99.9% / test 65%）。所有正则化手段都是把模型从“过”的一侧往中间推。判断依据不是绝对分数，而是 train-test 差距。

### 2.2 Dropout：随机掐人，逼出冗余

训练时每个神经元以概率 p 被置零：

$$\text{output} = \text{activation}(z)\cdot\text{mask},\qquad \text{mask}[i]\sim\text{Bernoulli}(1-p)$$

p=0.5 时每次前向一半神经元被掐，网络无法预测谁会被掐，只能学冗余表示——这打破了共适应（co-adaptation，神经元之间“你死我活”的依赖）。集成（ensemble）视角：N 个神经元有 $2^N$ 种开/关子网络，Dropout 近似同时在训练它们；测试时用全部神经元等价于对 $2^N$ 个子网络取平均——一个模型当一支军队。工程实现用反向缩放（0.3 节算过）：训练时除以 (1−p)，测试时零改动。默认剂量：transformer p=0.1、MLP p=0.5、CNN p=0.2—0.3。p 越大正则越强、欠拟合风险越大。

### 2.3 权重衰减（L2）：对大权重征税

$$\text{total\_loss} = \text{task\_loss} + \frac{\lambda}{2}\sum_i w_i^2$$

正则项梯度是 λw：每步每个权重按自身比例向 0 缩（0.3 节：权重 2.0 被拽 0.2、0.5 只被拽 0.05）。过拟合模型惯用大权重放大噪声，衰减逼它用小而稳的解。λ 典型值：AdamW 训 transformer 0.01；SGD 训 CNN 1e-4；严重过拟合 0.1。衔接第06课：SGD 里 L2 与衰减等价、Adam 里不等价，用 Adam 就配 AdamW。

### 2.4 BatchNorm：按批归一化，附带平滑损失面

$$\mu=\frac{1}{B}\sum_i x_i,\quad \sigma^2=\frac{1}{B}\sum_i(x_i-\mu)^2,\quad \hat{x}=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\hat{x}+\beta$$

逐特征、跨样本统计（0.3 节的 [[1,2],[3,4]] 例子）。γ、β 可学习，让网络保留“撤销归一化”的权利。训练用当前批统计，推理用训练期攒的滑动平均（动量 0.1，即 90% 旧 + 10% 新）。为什么有效仍有争议：原论文说是减少“内部协变量偏移”，Santurkar 等（2018）证明这个解释不对，真因是让损失面更平滑、梯度更有预测性，于是能用更大学习率。命门：依赖批统计——batch=1 无意义、<32 统计噪声伤性能，目标检测（内存限制批大小）和语言建模（序列长度可变）都吃亏。

### 2.5 LayerNorm 与 RMSNorm：按样本归一化的两档

$$\mu=\frac{1}{D}\sum_j x_j,\quad \sigma^2=\frac{1}{D}\sum_j(x_j-\mu)^2,\quad y_j=\gamma\hat{x}_j+\beta$$

LayerNorm 逐样本、跨特征维 D 统计（0.3 节的 [1,2,3] 例子）：与 batch 大小解耦、训练推理计算一致，transformer 在每个自注意力块和前馈块之后（Post-LN）或之前（Pre-LN，训练更稳）使用。RMSNorm 再进一步——去掉减均值：

$$\text{rms}=\sqrt{\frac{1}{D}\sum_j x_j^2},\qquad y=\gamma\cdot\frac{x}{\text{rms}}$$

Zhang & Sennrich（2019）的观察：LayerNorm 里的“重新中心化”（减均值）对性能贡献很小却花计算。去掉它：准确率持平、开销省约 10%，还没有 β 参数。LLaMA 全系、Mistral 及多数现代 LLM 用它——在数十亿参数、数万亿 token 的规模上，10% 是真金白银。

### 2.6 数据增强、早停与剂量表

数据增强改输入不改标签（图像：随机裁剪/翻转/色彩抖动；文本：同义替换/回译；音频：变速/变调/加噪），效果等同正则：模型每张图见到 50 种变形就无法背原图，只能学不变结构。早停是最简单的正则：每轮记验证损失、存最好权重；连续 patience 轮没进步就停并回滚（0.3 节最后一段的读法）。三者怎么配，看差距大小：

| train−test 差距 | Dropout p | 权重衰减 | 其他 |
|---|---|---|---|
| > 10%（重） | 0.3—0.5 | 0.01—0.1 | 激进数据增强 + 早停 |
| 5—10%（中） | 0.1—0.2 | 0.001—0.01 | 加 BatchNorm/LayerNorm |
| < 5%（轻） | 0.05—0.1 | 1e-4 | 观察，别过度治疗 |

（补充：先看差距方向——如果训练分本身也低，那是欠拟合，加正则只会更糟，应该先加容量。）

## 三、操作前准备

Build It 全程只用 Python 标准库（random、math），无第三方依赖；Use It 需要 PyTorch：

~~~bash
python -m pip install torch
~~~

先激活你已有的 Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或运行任何训练。Step 6 依赖 Step 1 的 `Dropout` 类，Step 3—5 相互独立。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：Dropout（训练/评估双模式）

~~~python
# 本块中文说明：带反向缩放的 Dropout，含 forward/backward。
import random
import math


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return list(x)  # 推理：原样通过，无需缩放（反向缩放的红利）
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:      # 以概率 p 被掐
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                output.append(val / (1 - self.p))  # 幸存者放大 1/(1-p)
        return output

    def backward(self, grad_output):
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)              # 被掐的神经元梯度为 0
            else:
                grads.append(g / (1 - self.p)) # 幸存者梯度同样放大
        return grads
~~~

对照 0.3 节：p=0.5、输入 [1,2,3,4]、掩码 [1,0,0,1] → forward 输出 [2, 0, 0, 8]；backward 里梯度走同样的缩放。

### 4.2 第 2 步：L2 权重衰减

~~~python
# 本块中文说明：L2 惩罚值与其梯度（加进任务梯度即可）。
def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty   # (λ/2)·Σw²

def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]  # λw
~~~

手算对照：[2.0, −1.0, 0.5]、λ=0.1 → 惩罚 0.2625、梯度 [0.2, −0.1, 0.05]。

### 4.3 第 3 步：批归一化

~~~python
# 本块中文说明：BatchNorm：训练用批统计并更新 running 值，推理用 running 值。
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            # 滑动更新 running 统计：90% 旧 + 10% 新（momentum=0.1）
            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output
~~~

手算对照：输入 [[1,2],[3,4]]、初始 γ=1、β=0 → 输出 [[−1,−1],[1,1]]（0.3 节）。核对：本类只实现了 forward，γ/β 的梯度与 x_hat 反传未写——原课到此为止；真实使用要补 backward（γ 的梯度 = Σ grad·x̂、β 的梯度 = Σ grad）。（补充：这是原课的简化边界，不是完整实现。）

### 4.4 第 4 步：层归一化

~~~python
# 本块中文说明：LayerNorm：单样本、跨特征统计。
class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output
~~~

手算对照：[1,2,3] → [−1.225, 0, 1.225]。与 BatchNorm 的区别全在统计方向上（横 vs 竖）。

### 4.5 第 5 步：RMSNorm

~~~python
# 本块中文说明：RMSNorm：不减均值、只除 RMS，无 β。
class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output
~~~

手算对照：[1,2,3] → rms≈2.160 → [0.463, 0.926, 1.389]。对比 4.4：少一次均值计算、少一个参数向量。

### 4.6 第 6 步：带正则化的训练对比网络

在第05课网络上装 Dropout 与 L2，并首次维护 train/test 双轨评估。

~~~python
# 本块中文说明：RegularizedNetwork：Dropout + L2 装进圆环分类网络。
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


class RegularizedNetwork:
    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)  # 推理：原样返回副本

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            # 权重更新带衰减项；偏置不衰减（常见约定）
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)  # 评估时关掉 Dropout
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    Epoch {epoch:3d}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%, gap={gap:.1f}%")
        return history
~~~

（核对：①权重更新把衰减加在梯度里（$g + \lambda w$），对 SGD 等价于损失加 L2；②偏置 b1、b2 不衰减——常见约定，与原课一致；③`backward` 中 Dropout 的梯度缩放没有显式出现——隐藏层梯度 d_h 直接用了未被掩码缩放的路径，是简化处理（补充：严格版应把 d_h 再乘 mask/(1−p)，即 4.1 的 backward；原课未展开，本篇如实保留并在此标注）。另核对：原课页面到类定义为止，未展示“有无正则化”的对比调用，可仿照前两课补一个四配置循环（无正则 / 仅 L2=0.01 / 仅 Dropout p=0.3 / 两者都开，各训 300 轮），独立测试集可用 `make_circle_data(n=200, seed=7)` 生成——此入口及其输出数值均未经本机运行；原课“正则化缩小 gap”为原课参考，且圆环数据分布固定、过拟合未必严重，真要看 gap 得用更小的数据集或更大的网络。）

## 五、把代码运行起来：补充入口与检查

（补充）原课各步按 4.1→4.6 顺序执行；4.6 依赖 4.1 的 `Dropout`。以下最小检查不依赖（大部分）随机数，可手算复核：

~~~python
# 本块中文说明：补充的手算复核入口。

# L2（0.3 节算例）
assert abs(l2_regularization([2.0, -1.0, 0.5], 0.1) - 0.2625) < 1e-12 and l2_gradient([2.0, -1.0, 0.5], 0.1) == [0.2, -0.1, 0.05]

# Dropout：推理模式原样通过；训练模式幸存者为 val/(1-p)、被掐者为 0
d = Dropout(p=0.5)
d.training = False
assert d.forward([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
d.training = True
assert all(o in (0.0, 2.0) for o in d.forward([1.0, 1.0, 1.0, 1.0]))

# BatchNorm（0.3 节算例，γ=1、β=0）；running 统计被更新（90% 初值 + 10% 批统计）
bn = BatchNorm(num_features=2, momentum=0.1)
out = bn.forward([[1.0, 2.0], [3.0, 4.0]])
assert all(abs(out[i][j] - (-1.0 if i == 0 else 1.0)) < 1e-4 for i in range(2) for j in range(2))
assert abs(bn.running_mean[0] - 0.2) < 1e-12  # 0.9*0.0 + 0.1*2.0

# LayerNorm 与 RMSNorm（0.3 节算例）
ln = LayerNorm(num_features=3)
out = ln.forward([1.0, 2.0, 3.0])
assert abs(out[0] + 1.2247) < 1e-3 and abs(out[1]) < 1e-6 and abs(out[2] - 1.2247) < 1e-3
rn = RMSNorm(num_features=3)
out = rn.forward([1.0, 2.0, 3.0])
assert abs(out[0] - 0.4629) < 1e-3 and abs(out[1] - 0.9258) < 1e-3 and abs(out[2] - 1.3887) < 1e-3
print("all regularization checks passed")
~~~

补充检查方法：用 4.6 的 `history` 画 train/test 两条准确率曲线，gap 的走势是本课唯一的“成绩单”；早停实验（原课练习 4）就是把 `history` 里 test_acc 最好的轮号找出来。

## 六、Use It：使用现成工具

PyTorch 把全部归一化与正则化做成模块；train/eval 开关是命门：

~~~python
# 本块中文说明：PyTorch 版 MLP：Linear→BatchNorm→ReLU→Dropout 的标准叠法。
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)

model.train()                       # 训练模式：Dropout 开、BatchNorm 用批统计
out_train = model(torch.randn(32, 784))

model.eval()                        # 评估模式：Dropout 关、BatchNorm 用 running 统计
out_test = model(torch.randn(1, 784))
~~~

`model.train()` / `model.eval()` 的切换至关重要：它同时关开 Dropout 和切换 BatchNorm 的统计来源。推理前忘了 `model.eval()` 是深度学习最常见的 bug 之一——Dropout 还在随机掐人、BatchNorm 还在用小批统计，测试准确率会无规律地抖。transformer 的模式则不同：

~~~python
# 本块中文说明：transformer 块的标准正则化配置。
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))  # Pre-LN 风格的残差接法
        x = self.norm2(x + self.ff(x))
        return x
~~~

原课口诀：transformer 用 LayerNorm（不是 BatchNorm）、Dropout p=0.1（不是 0.5）——这就是现代默认。（补充：LLaMA 系把 `nn.LayerNorm` 换成 `nn.RMSNorm` 即可对齐 4.5 的实现。）

## 七、失败模式、原文边界与工程注意

1. 忘 `model.eval()`：最常见 bug，测试分数随机抖动；评估完也别忘切回 `model.train()`。
2. Dropout 过量：p 太大网络欠拟合，训练分都上不去——先看训练分再谈 gap（2.6 节的“先看方向”）。
3. BatchNorm + 小 batch：批 <32 统计噪声伤性能，批=1 直接失效。小批场景换 LayerNorm/RMSNorm（原课练习 3 从反面验证）。
4. BatchNorm 类只实现了 forward（核对：无 backward、γ/β 无梯度），是原课的教学简化；真实项目直接用 nn.BatchNorm1d。
5. 原文边界：“RMSNorm 快 10%”是原课引述的量级参考，具体取决于硬件与实现；准确率“持平”也是大模型实验上的结论，小任务可能有零点几的差异。
6. 早停的 patience 是双刃剑：太小被噪声骗停、太大浪费算力；跟踪的是验证损失不是训练损失（补充）。
7. 数据增强要用“保持标签”的变换（把数字 6 翻转成 9 就是反例）；正则化各旋钮要联动看，单独拧一个到极限不如均衡调（补充）。
8. train/test 差距要全程监控：每轮记录两套指标，别等训练结束才第一次看测试集（本课 4.6 的 history 结构就是为了这个）。
## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/prompt-regularization-advisor.md`（诊断过拟合并推荐正则化策略的提示词）。本篇改写成自查模板，直接嵌在这里，不另建输出文件：

~~~markdown
检查正则化方案：报告训练与验证/测试两侧的准确率与损失（不止训练分）；计算 train-test 差距并按剂量表定位严重程度（>10% 重 / 5—10% 中 / <5% 轻）；列出所用手段及参数（Dropout p、weight decay λ、归一化类型、数据增强种类、早停 patience）；确认评估前调用了 model.eval()；确认用 Adam 时衰减走 AdamW 解耦路径；所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：为 2D 数据实现空间 Dropout（spatial dropout）——把连续特征组当通道、整组一起掐，在 hidden_size=32 的圆环网络上与普通 Dropout 比 gap。验收标准：能解释“掐通道更强地强制特征间冗余”并展示 gap 差异。
2. 原课：把 [第05课](<05：损失函数——MSE、交叉熵与选择.md>)的标签平滑与本课 Dropout 组合，跑四种配置（都无/只 Dropout/只平滑/都有）。验收标准：四组数字表格化，指出 gap 最小的组合并解释。
3. 原课：在隐藏层与激活之间加 BatchNorm，用 lr=0.01、0.05、0.1 三档做有无对照。验收标准：能展示 BatchNorm 让高 lr 下训练依然稳定、裸网络发散（原课预期方向），并联系“平滑损失面”解释。
4. 原课：实现早停——每轮记录测试损失、存最好权重、20 轮无改善即停；与跑满 1000 轮对照。验收标准：报告最佳测试准确率的轮号与早停省下的计算量。
5. 原课：在 4 层（不是 2 层）网络上对比 LayerNorm vs RMSNorm，同权重初始化、训 200 轮，比较准确率、每轮耗时、第一层梯度幅度。验收标准：验证“RMSNorm 更快且准确率相当”（原课参考结论），耗时多次平均。

## 十、术语与完成标准

过拟合：训练性能显著超过测试性能，学的是噪声；正则化：约束复杂度以改善泛化的一切手段；Dropout：训练时以概率 p 置零神经元，等效训练 $2^N$ 个子网络的集成；反向缩放：训练时除以 (1−p)，推理零改动；权重衰减：每步减 λw 的按比例收缩；BatchNorm：跨样本逐特征归一化，训练用批统计、推理用滑动平均；LayerNorm：逐样本跨特征归一化，与 batch 无关；RMSNorm：去掉减均值的 LayerNorm，省约 10%；早停：验证损失不再改善即停并回滚；数据增强：保标签的输入变换；泛化差距：train-test 之差，本课的监控核心。

完成检查：

- 能手算 L2 惩罚值与梯度、p=0.5 的 Dropout 掩码输出，以及 [[1,2],[3,4]] 的 BatchNorm 与 [1,2,3] 的 LayerNorm/RMSNorm 输出（0.3 节算例）。
- 能解释“为什么训练时缩放而不是测试时缩放”。
- 能说出 BatchNorm 与 LayerNorm 的统计方向及各自的命门。
- 能按 train-test 差距开剂量（>10%、5—10%、<5% 各用什么），并列出 `model.eval()` 忘切的症状与原因。

## 十一、验证范围与来源许可

原课 Build It 共 6 个分段，代码已逐字转录嵌入本篇（仅按本篇格式添加中文说明与注释，未改逻辑）；其中 4.6 的对比运行调用原课未展示，本篇在 4.6 末尾的核对段中以文字显式补写并标注；4.3 BatchNorm 无 backward 属原课简化边界，已显式标注。Use It 的 PyTorch 代码同样保留。原课页面未展示运行输出，本篇未虚构“原课参考”数值；第五节“可手算复核”的断言均对应 0.3 节算例。本次未执行训练或下载任何数据/模型。写作完成后，将正文各 python 代码块抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 逐一做静态语法检查，结果见下方记录。

本次验证记录（2026-11-06）：
- 抓取课程页一次成功，以原文为准整理；未使用凭记忆的内容。
- 正文 9 个 python 代码块（4.1—4.6、第五、六节两块）全部通过 `ast.parse` 静态语法检查；第五节手算断言（L2 0.2625、Dropout 结构检查、BatchNorm ±1、LayerNorm ±1.225、RMSNorm 0.463/0.926/1.389）在临时脚本中实际执行复核通过（属玩具断言检查，不构成训练或性能实验）。
- 未做：4.6 的四配置训练对比、早停实验、BatchNorm 高学习率实验。

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
