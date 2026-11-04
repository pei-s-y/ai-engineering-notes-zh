# 03：逻辑回归——sigmoid、交叉熵与决策边界

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/03-logistic-regression) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/02-ml-fundamentals/03-logistic-regression/docs/en.md)。原题：Logistic Regression。

## 零、初学者先读：当“预测一个数”变成“回答是或否”

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要先掌握[第02课](<02：线性回归——最小二乘、梯度下降与正规方程.md>)的模型—损失—优化循环。

### 0.1 先理解这一课在做什么

线性回归输出的是随便多大都行的数（比如 1.7、-0.5），但“肿瘤是不是恶性”这种问题需要 0 到 1 之间的概率和一个明确的判断。逻辑回归的做法很省事：先照旧算一个分数 wx+b，再过一个叫 sigmoid 的“S 形压缩器”把分数压成 0～1 之间的概率，超过 0.5 就判为“是”。名字里虽有“回归”，它是不折不扣的分类算法。

### 0.2 本课术语先翻成白话

- 分类（classification）=预测类别标签（0/1 或多类），区别于预测数值的回归
- sigmoid 函数=把任意实数压进 (0,1) 的 S 形函数，σ(0)=0.5
- 概率 p=模型输出的“属于类别 1 的可能性”
- 阈值（threshold）=概率超过多少判为 1，默认 0.5
- 决策边界（decision boundary）=让概率恰好 0.5 的输入集合，是一条直线/超平面
- 二元交叉熵（binary cross-entropy / log loss）=分类版的损失函数，惩罚“自信地答错”
- softmax=多分类版的 sigmoid：把一串分数变成和为 1 的概率
- 独热编码（one-hot）=类别 k 表示成“第 k 位是 1、其余是 0”的向量
- 混淆矩阵（confusion matrix）=TP/TN/FP/FN 四个计数的表格
- 精确率（precision）=报成阳性的里面有多少真是阳性
- 召回率（recall）=真阳性里面被抓住了多少
- F1=精确率与召回率的调和平均
- 过拟合/凸性等术语见[第02课](<02：线性回归——最小二乘、梯度下降与正规方程.md>)

### 0.3 用小数字走一遍

两个训练样本 x=1（标签 0）、x=2（标签 1），初始 w=0、b=0。

前向：两个样本的分数都是 0，σ(0)=0.5，所以预测概率都是 0.5。误差（预测−真实）分别是 +0.5 和 −0.5。

梯度：dw=(1/2)×(0.5×1+(−0.5)×2)=−0.25，db=(1/2)×(0.5−0.5)=0。取学习率 1 更新：w=0.25、b=0。

再看效果：x=1 的分数变 0.25，σ(0.25)≈0.562（更接近…等等，x=1 的标签是 0，概率反而升了？注意这只是单步更新，另一边的 x=2 从 0.5 升到 σ(0.5)≈0.622，两个样本在“竞争”同一个 w；训练很多轮后边界会落在两类之间）。这个小步只演示“梯度怎么算、参数怎么动”，不演示收敛。

损失再看一眼：真实 y=1 而预测 p=0.5 时，交叉熵损失 −log(0.5)≈0.693；若模型自信地给 p=0.01，损失是 −log(0.01)≈4.6——答错越自信罚越重。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：`sigmoid`（S 形压缩器）→ `LogisticRegression` 类（predict_proba 算概率、compute_loss 算交叉熵、fit 跑梯度下降、accuracy 数对了几个）→ `ClassificationMetrics`（数 TP/FP/FN/TN 再算精确率/召回率/F1）→ 第 4 步的边界打印 → `SoftmaxRegression`（每类一套权重，多分类推广）→ 第 6 步的阈值扫描。核心数据结构：`weights` 列表 + `bias` 标量（与第02课同构），多分类时变成“每类一行”的嵌套列表。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么不直接用线性回归做分类？

**参考答案：** 线性回归输出无界（可为负、可超过 1），不是概率；一个极端离群点会把整条线拖歪。分类需要输出在 0～1、对远离边界的离群点不敏感的函数。

**问题2：** 决策边界在哪里？为什么是直的？

**参考答案：** 在 wx+b=0 的输入集合上（此时 σ 恰为 0.5）。因为 σ 单调且只在 0 处等于 0.5，“概率=0.5”等价于“wx+b=0”，这是个线性方程；想要弯的边界就得加多项式特征或换非线性模型。

**问题3：** 99% 样本是阴性、1% 是阳性的数据里，准确率 99% 说明什么？

**参考答案：** 什么都不说明——全预测阴性的模型也有 99% 准确率但毫无用处。要看精确率/召回率/混淆矩阵，这正是本课引入这些指标的原因。

能解释这三问后，进入正文。

---

## 一、这一课解决什么问题？

想根据肿瘤大小判断恶性还是良性。套线性回归会输出 0.3、1.7、-0.5 这类数：1.7 是“非常恶性”？-0.5 是“非常良性”？没有意义。分类需要 0～1 之间的概率和一个决策。逻辑回归把线性组合 wx+b 过一遍 sigmoid，输出概率，再用阈值（通常 0.5）做决定。它名字带“回归”却是分类算法，名字来自它用的 logistic（sigmoid）函数。原课强调它是实践中使用最广的算法之一；本课同时教你分类问题的整套评估语言（混淆矩阵、精确率、召回率、F1）。

## 二、核心概念与推导

### 2.1 为什么线性回归做不了分类

用学习时长预测通过（1/0），线性拟合会在 1 小时处给 -0.2、10 小时处给 1.3：低于 0、高于 1，都不是概率；一个学了 50 小时的极端样本还会把整条线拖走。分类需要：输出在 0～1 之间、边界附近过渡陡、不受远处离群点影响。

### 2.2 sigmoid 函数

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

性质：z 很大时趋近 1，很小（负得多）时趋近 0，z=0 时恰为 0.5；输出永远在 (0,1)；处处光滑可导。导数有个漂亮的形式 $\sigma'(z)=\sigma(z)(1-\sigma(z))$，让梯度计算很省事。

### 2.3 模型：线性打分 + sigmoid

流程（补充示意）：

~~~text
输入 x ──▶ 线性打分 z = wx + b ──▶ sigmoid p = 1/(1+e^-z) ──▶ p >= 0.5 ? 是→1 : 否→0
~~~

输出 p 解释为 $P(y=1\mid x)$：输入属于类别 1 的概率。决策边界是 $wx+b=0$ 的集合（那里 p 恰为 0.5）。

### 2.4 二元交叉熵损失

MSE 配 sigmoid 会得到非凸损失面，容易卡在局部极小。改用二元交叉熵（binary cross-entropy，log loss）：

$$\text{Loss} = -\frac{1}{n}\sum\left(y\log(p)+(1-y)\log(1-p)\right)$$

逐项读：y=1 且 p 接近 1 时 log(1)=0，损失近 0（对，代价小）；y=1 且 p 接近 0 时 log(0)→−∞，损失巨大（自信答错，重罚）；y=0 的两种情形对称。核对：此损失对逻辑回归是凸的，局部极小即全局极小；但若数据完全可分，损失随权重增大一直下降、没有有限最小点——原文指出此时加 L2 正则化能让最优解唯一。

### 2.5 梯度下降

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum(p-y)\,x,\qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum(p-y)$$

形式与第02课线性回归的梯度几乎一样，只是 p 换成了 σ(wx+b)。非线性全部藏在 sigmoid 里，更新规则不变。

### 2.6 决策边界

二维输入时边界是 $w_1 x_1 + w_2 x_2 + b = 0$ 这条直线，两侧各判一类。逻辑回归永远是线性边界；要曲线边界，加多项式特征或换非线性模型（与[第04课](<04：决策树与随机森林.md>)的轴对齐矩形边界是两种不同的“非线性”来源）。

### 2.7 多分类：softmax

k 个类各有一套权重，模型给每类打分 $z_i$，softmax 把分数变成和为 1 的概率：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

损失换成类别交叉熵（categorical cross-entropy）：$-\frac{1}{n}\sum\sum_k y_k\log(p_k)$，其中 $y_k$ 是独热编码的真实标签。

### 2.8 评估指标：准确率不够

95% 阴性、5% 阳性的数据里，“全预测阴性”有 95% 准确率但没用。混淆矩阵把预测分成四格：

| | 预测阳性 | 预测阴性 |
|---|---|---|
| 实际阳性 | TP | FN |
| 实际阴性 | FP | TN |

$$\text{Precision}=\frac{TP}{TP+FP},\qquad \text{Recall}=\frac{TP}{TP+FN},\qquad F_1=2\cdot\frac{P\cdot R}{P+R}$$

选哪个：误报代价高（垃圾邮件拦截）重精确率；漏报代价高（癌症筛查）重召回率；要单一均衡指标用 F1。

### 2.9 用一个小例子把概念串起来（补充）

三个测试点 [3,3]、[4,4]、[2,5]，假设训练后 w=[1,1]、b=−6（补充数值，只为串概念）。分数分别是 0、2、−3；概率 σ(0)=0.5、σ(2)≈0.881、σ(−3)≈0.047。按 0.5 阈值判为 1、1、0；点 [2,5] 落在边界直线 x1+x2=6 的“负侧”。把阈值提到 0.9：σ(2)≈0.881<0.9，[4,4] 被改判为 0——这就是第 6 步阈值扫描在干的事：精确率与召回率的互换。损失侧：若 [4,4] 真实标签是 1，模型给 0.881 的损失是 −log(0.881)≈0.127；给 0.5 则是 0.693；给 0.05 则约 3.0。概率—阈值—损失—指标在这组小数字上全部串起来了。

## 三、操作前准备

Build It 部分只用 Python 标准库（random、math）；Use It 需要 scikit-learn 和 NumPy：

~~~bash
python -m pip install numpy scikit-learn
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：sigmoid 与数据生成

sigmoid 里先把 z 截断到 ±500，避免 math.exp 溢出（工程细节，不改变数学）。数据是两个高斯团：类 0 中心 (2,2)、类 1 中心 (5,5)，各 100 个样本，打散后备用。这是合成数据，两个团在二维上近似线性可分。

~~~python
# 本块中文说明：定义截断版 sigmoid，并生成两团二维高斯分类数据。
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
~~~

### 4.2 第 2 步：从零实现逻辑回归

结构与第02课同构。`compute_loss` 里把 p 夹在 [1e-15, 1−1e-15]，防止 log(0)；`fit` 是全量批量的梯度下降，梯度公式即 2.5 节。数据按 80/20 切训练/测试。

~~~python
# 本块中文说明：从零实现二元逻辑回归（sigmoid + 交叉熵 + 梯度下降）。
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
~~~

训练/测试准确率应在 0.95 上下（原课参考，两团数据分得开；非本机实测）。

### 4.3 第 3 步：混淆矩阵与指标

数四个格子再算比例。所有分母都有零保护：没有预测阳性时精确率定义为 0 而不是崩溃。

~~~python
# 本块中文说明：从零实现混淆矩阵与精确率/召回率/F1。
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
~~~

### 4.4 第 4 步：决策边界分析

把 wx+b=0 整理成 x2 = … 的直线方程打印出来，再在边界附近挑 5 个点看概率的渐变。可手算预期：点越靠近边界，概率越接近 0.5；[3.5,3.5] 这类中点概率在 0.5 附近摆动。

~~~python
# 本块中文说明：打印决策边界直线方程，并观察边界附近点的概率。
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
~~~

### 4.5 第 5 步：softmax 多分类

每类一套权重，softmax 有个数值稳定技巧：先减去最大分数再取指数（分式上下同乘一个常数，概率不变，但 exp 不会溢出）。损失用类别交叉熵，同样在 log 前做 1e-15 保护。数据是三个高斯团，中心 (1,1)、(5,1)、(3,5)。

~~~python
# 本块中文说明：从零实现 softmax 多分类回归。
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
~~~

### 4.6 第 6 步：阈值调优

同一模型不动，只扫阈值 0.3～0.7，重新算四个指标。这是“部署时旋钮”，不是训练的一部分。

~~~python
# 本块中文说明：扫描阈值，观察精确率/召回率如何互换。
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
~~~

可手算预期：阈值升 → 预测的阳性变少 → 精确率趋升、召回率趋降；0.5 附近的准确率/F1 最高（两团数据对称），具体数值以实际训练结果为准。

## 五、把代码运行起来：补充入口与检查

（补充）原课各步连续执行即可：4.3 依赖 4.2 的 `model`、`X_test`、`y_test`；4.6 同样。以下是补充的最小检查：

~~~python
# 本块中文说明：补充的 sigmoid/指标/softmax 手算检查。
assert abs(sigmoid(0) - 0.5) < 1e-12
assert abs(sigmoid(1) - 0.7310585786300526) < 1e-9  # 可手算/查表预期

# 混淆矩阵手算例：TP=3, FP=1, FN=2, TN=4
m = ClassificationMetrics([1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
# 可手算预期：precision=3/4=0.75，recall=3/5=0.6，F1=2×0.75×0.6/1.35≈0.667
assert abs(m.precision() - 0.75) < 1e-12
assert abs(m.recall() - 0.6) < 1e-12
assert abs(m.f1() - 0.6666666666666666) < 1e-9

# softmax 平移不变性：减去最大值不改变概率
sr = SoftmaxRegression(n_features=2, n_classes=3)
p1 = sr.softmax([1.0, 2.0, 3.0])
p2 = sr.softmax([101.0, 102.0, 103.0])
assert all(abs(a - b) < 1e-12 for a, b in zip(p1, p2))
assert abs(sum(p1) - 1.0) < 1e-12
~~~

补充检查方法：训练时画出 `loss_history`，应平稳下降；训练/测试准确率差距过大提示过拟合（本课数据下不应出现明显差距）。

## 六、Use It：使用现成工具

scikit-learn 版。注意这里 StandardScaler 是在训练集上 fit、测试集上只 transform（正确做法）；`classification_report` 一次给出每类的精确率/召回率/F1。

~~~python
# 本块中文说明：scikit-learn 版逻辑回归与完整评估报告。
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
~~~

核对：`lr.fit(X_tr_sc, y_tr)` 用的是训练集标签（y_tr），测试集 y_te 只用于评估；注意别把两者写反。原课说“你的实现和 sklearn 产生相同的决策边界与指标”，并指出 sklearn 额外提供求解器选项（liblinear、lbfgs、saga）、自动正则化、多分类策略与数值稳定优化。理解为“数学等价、工程更稳”。

## 七、失败模式、原文边界与工程注意

1. 类别不平衡时准确率是误导指标：先看混淆矩阵，再选精确率/召回率/F1（本课数据两类均衡，掩盖了这一点；真实数据要先查类别分布，补充）。
2. 阈值 0.5 只是默认，不是天经地义：垃圾邮件（重精确率）和癌症筛查（重召回率）的最优阈值不同，应按业务代价调。
3. 逻辑回归只能画线性边界：同心圆这类数据会直接失败，需加多项式特征（原课练习 1 正是此实验）。
4. 完全可分数据上交叉熵损失没有有限最小点，权重会一直长大；用 L2 正则化解决（原文已指出，此为核对确认）。
5. 预测前忘了模型是在标准化空间训练的：喂原始尺度特征会得到无意义概率（补充，Use It 里 pipeline 的顺序就是为此）。
6. `compute_loss` 里不做 1e-15 截断会直接 log(0) 崩溃；这是教学代码与生产代码都必须处理的数值边界（原课已处理，值得注意）。
7. 概率输出未必校准（calibrated）：逻辑回归在特征假设满足时近似良好，但不要把 0.9 的输出直接当“十次里错一次”的承诺（补充；朴素贝叶斯的校准问题更严重，见[第14课](<14：朴素贝叶斯——条件独立、拉普拉斯平滑与文本分类.md>)）。
8. 训练/测试划分要在任何预处理之前想清楚：本课从零版没有标准化（两个特征同尺度所以没事），真实数据要像 Use It 那样只在训练集上拟合 scaler（补充）。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `code/logistic_regression.py`（带指标的完整实现）。中文学习版自查模板（直接保存在本篇，不另建文件）：

~~~markdown
检查逻辑回归：说明类别是否平衡、用什么阈值、为什么；报告混淆矩阵与精确率/召回率/F1 而不只报准确率；检查概率是否只来自训练集拟合的预处理管道；若数据近似可分，说明是否加了正则化；展示决策边界并说明它是线性的原因；所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Medium：生成不可线性分的数据（如同心圆），训练逻辑回归观察失败；再加多项式特征（x1²、x2²、x1x2）重训，展示准确率提升。验收标准：线性版准确率明显低于多项式版（在同心圆数据上原课预期显著差距，具体数值以实测为准），能解释新增特征如何让边界变弯。
2. 原课 Medium：给三分类 softmax 模型实现多类混淆矩阵，算每类的精确率与召回率，指出最难分的类。验收标准：3×3 混淆矩阵行列对齐“行=真实、列=预测”，能指出混淆最集中的类并联系三个高斯团中心距离解释。
3. 原课 Hard：从零画 ROC 曲线：100 个阈值算 TPR/FPR，用梯形法则算 AUC。验收标准：AUC 在 [0,1] 内、随机猜测对应 0.5；完美分类器趋于 1；能解释 AUC 与阈值选择的区别（AUC 是全阈值的序质量，阈值是单点决策）。

## 十、术语与完成标准

逻辑回归（logistic regression）：线性模型 + sigmoid 输出类别概率的分类器；sigmoid：把实数压到 (0,1) 的 S 形函数；二元交叉熵：惩罚自信错误的对数损失；决策边界：wx+b=0 的线性分界面；softmax：多分类概率化；精确率：报阳中真阳比例；召回率：真阳被抓住比例；F1：两者的调和平均；混淆矩阵：TP/TN/FP/FN 计数表；阈值：判为 1 的概率下限；独热编码：类别 k → 第 k 位为 1 的向量。

完成检查：

- 能手算 σ(0)、σ(1) 与一个 y=1、p=0.5 的交叉熵损失（0.693）。
- 能写出梯度公式并指出与线性回归的“一处不同”（p 是 σ(wx+b)）。
- 能从混淆矩阵手算精确率、召回率、F1（第五节算例）。
- 能解释阈值上调时精确率与召回率怎么动、为什么。
- 能说明 softmax 减最大值的数值稳定作用。
- 能说出两个“准确率失效”的场景。

## 十一、验证范围与来源许可

原课 Build It 共 6 个分段，代码已逐字转录嵌入本篇（仅添加中文说明与注释，未改逻辑）；Use It 的 sklearn 代码亦保留。原课页面未展示运行输出，本篇未虚构“原课参考”数值。写作完成后，正文 python 代码块逐一抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 做静态语法检查，结果见下。

本次验证记录（2026-11-04）：
- raw 文档 502 失败后改用课程页 URL 成功获取原文，以原文为准整理。
- 正文 8 个 python 代码块全部通过 `ast.parse` 静态语法检查；第五节标注"可手算预期"的断言（sigmoid 值、混淆矩阵指标、softmax 平移不变性）另在临时脚本中实际执行复核通过；第四节原课训练演示未执行，其数值仍标注为原课参考。
- 未做：实际训练、ROC/AUC 实验、与 sklearn 的数值比对。

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
