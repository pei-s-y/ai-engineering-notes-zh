# 11：集成方法——Bagging、Boosting与Stacking

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为"原课参考"的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/11-ensemble-methods) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/02-ml-fundamentals/11-ensemble-methods/docs/en.md)。原题：Ensemble Methods。前置：[第10课偏差与方差](<10：偏差与方差——误差分解、学习曲线与欠过拟合.md>)、[第04课决策树与随机森林](<04：决策树与随机森林.md>)。

## 零、初学者先读：一堆"不太行"的模型，凭什么合起来变强

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。

### 0.1 先理解这一课在做什么

看病时找一个医生可能误诊，找五个医生分别诊断再投票，多数意见往往更可靠——前提是五个医生别都犯同一种错。集成方法（ensemble methods）就是这件事的工程版：训练多个"各自都不完美"的模型，用平均或投票把它们合并。有效的关键不是模型多，而是它们**犯的错不一样**。

### 0.2 本课术语先翻成白话

- 集成（ensemble）=多个模型组合成的一个"大模型"
- 基学习器/弱学习器（base/weak learner）=集成里的单个成员模型
- bagging=并行训练多份、各自用不同的抽样数据、最后平均或投票
- bootstrap=有放回地抽 n 次凑成一份新训练集
- OOB（out-of-bag）袋外样本=某棵树这次没抽到的约36.8%样本，可当免费验证集
- boosting=串行训练，每个新模型专门纠正前面模型犯的错
- 决策桩（decision stump）=只有一次分裂的深度1树
- AdaBoost=用"调样本权重"实现纠错的经典boosting算法
- 残差（residual）=真实值−当前预测值，回归里"还没解释掉的误差"
- 学习率/收缩（shrinkage）=每棵新树的贡献打几折再加进模型
- stacking=把几个不同模型的预测当新特征，再训练一个"元模型"
- 多样性（diversity）=成员模型错误互不相关的程度，集成的命根子

### 0.3 用小数字走一遍

三个独立医生，每人准确率0.6。多数投票（至少2人对）正确的概率：

$$3\times0.6^2\times0.4+0.6^3=0.432+0.216=0.648$$

比单人的0.6高。但如果三人总是犯一样的错（完全相关），投票结果还是0.6——这就是"多样性"为什么关键。

AdaBoost小算例：某轮加权错误率 err=0.3，则该模型权重 $\alpha=0.5\ln(0.7/0.3)\approx0.42$。分错的样本权重乘 $e^{0.42}\approx1.53$ 变大，分对的乘 $1/1.53\approx0.65$ 变小——下一轮的模型就会更在意分错的那些点。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：`DecisionStump`（会找"一次分裂里加权错误最小"的那刀）→ `AdaBoostScratch`（循环：训练桩→算err→算α→调权重）→ `SimpleRegressionTree`（原文缺失，第五节补充）→ `GradientBoostingScratch`（循环：算残差→用树拟合残差→小步累加）。核心数据结构：AdaBoost 存两张表 `stumps`（桩列表）和 `alphas`（对应权重）；GB 存 `trees` 列表加一个常数 `initial_pred`。

### 0.5 理解检查（含参考答案）

**问题1：** bagging 和 boosting 分别主要压低误差的哪一项？

**参考答案：** bagging 主要降方差（各自过拟合的方向不同，平均后抵消）；boosting 主要降偏差（每个新模型专门补前面的系统性错误）。

**问题2：** 两个准确率0.6的模型，什么时候集成后还是0.6？

**参考答案：** 两者错误完全相关（总是同时错在同一批样本）时，投票不改变任何预测，集成无效。

**问题3：** AdaBoost里 α 什么时候大？它控制什么？

**参考答案：** err 越小 α 越大（α=0.5·ln((1−err)/err)），α 是这个模型在最终加权投票里的"话语权"；err≥0.5时α≤0，比瞎猜还差的模型话语权为零或为负。

能解释"多样性、α、残差"这三个小例后，进入正文。

---

## 一、这一课解决什么问题？

单个决策树好训练、好解释，但容易过拟合；单个线性模型在复杂边界上欠拟合。与其花几天设计"完美模型"，不如把一堆不完美模型组合起来。原课的动机：集成是表格数据竞赛中最可靠的技术，也是多数生产ML系统的核心；同时它把[第10课](<10：偏差与方差——误差分解、学习曲线与欠过拟合.md>)的偏差-方差权衡变成了看得见的操作——bagging压方差、boosting压偏差、stacking学"什么输入该信哪个模型"。

## 二、核心概念与推导

### 2.1 多数投票为什么有效

设 N 个**相互独立**、准确率均为 p>0.5 的分类器，多数投票的准确率：

$$P(\text{多数正确})=\sum_{k>N/2}\binom{N}{k}p^k(1-p)^{N-k}$$

直觉：每个模型犯的错位置不同，投票时错误相互抵消。原课声称"21个准确率0.6的分类器投票约74%、101个约84%"。按上式逐项求和核对（可手算）：N=21、p=0.6 时约 **0.83**；N=101、p=0.6 时约 **0.98**（正态近似）。原课这两个数值与其自己给出的公式不符（核对），但结论方向——模型越多、独立时投票越准——不受影响。另注意：公式假设独立，真实模型高度相关，实际提升远低于理论值。

多样性从哪来：不同训练子集（bagging）、不同特征子集（随机森林）、串行纠错（boosting）、不同模型家族（stacking）。

### 2.2 Bagging与随机森林

bagging 对每个成员用一份 bootstrap 样本（有放回抽 n 次，与原数据同大小）。约63.2%的不同样本会出现在每份里，剩下36.8%是袋外（OOB）样本，可当免费验证集（[第04课](<04：决策树与随机森林.md>)有 bootstrap 手算）。每棵树各自过拟合自己的样本，但过拟合方向不同，平均后噪声抵消——方差降、偏差几乎不变。随机森林=bagging+每次分裂只考虑随机特征子集（分类常用 `sqrt(n_features)`，回归常用 `n_features/3`，是惯例不是定律）。

### 2.3 Boosting与AdaBoost

boosting 串行训练：新模型专注于前面模型搞错的样本，最终预测是加权求和。AdaBoost 用"样本权重"实现：

1. 初始权重 $w_i=1/N$；
2. 每轮训练弱学习器 $h_t$（通常决策桩），算加权错误率 $err_t=\sum w_i\,\mathbb{I}(h_t(x_i)\neq y_i)/\sum w_i$；
3. 模型权重 $\alpha_t=0.5\ln\frac{1-err_t}{err_t}$；
4. 权重更新 $w_i\leftarrow w_i\,\exp(-\alpha_t\,y_i\,h_t(x_i))$ 再归一化（分错的样本 $y_ih_t(x_i)=-1$，权重变大）；
5. 最终 $H(x)=\mathrm{sign}\left(\sum_t\alpha_t h_t(x)\right)$。

代价：轮数太多会过拟合——它不停拟合"难"样本，而难样本里可能有噪声。

### 2.4 梯度提升

梯度提升把 boosting 推广到任意可导损失：每个新模型拟合当前集成的**负梯度**（伪残差）：

$$r_i=-\frac{\partial L(y_i,F_{t-1}(x_i))}{\partial F_{t-1}(x_i)},\qquad F_t=F_{t-1}+\text{lr}\cdot\gamma_t\cdot h_t$$

对平方误差损失，伪残差就是普通残差 $r_i=y_i-F_{t-1}(x_i)$——每棵树 literally 在拟合前面所有人的误差。学习率（收缩）控制每棵树贡献多少：更小lr要更多树但通常泛化更好，典型0.01～0.3。

### 2.5 XGBoost：工程化的梯度提升

XGBoost = 梯度提升 + 工程优化：叶子权重的 L1/L2 正则；目标函数用一阶+二阶导数（分裂决策更准）；原生处理缺失值（每个分裂学一个缺省方向）；列抽样；加权分位草图；缓存友好的存储结构。原课断言"表格数据上 XGBoost/LightGBM 一贯优于神经网络，且短期内不会改变"——这是经验性倾向而非定理，第七节再讨论边界。

### 2.6 Stacking与投票

stacking 把基模型（随机森林、SVM、逻辑回归……）的预测当作元学习器的输入特征，元学习器学会"哪个区域信谁"。关键纪律：基模型的元特征必须由**训练集上的交叉验证**产生，绝不能用训练数据既训基模型又产元特征，否则泄漏。投票是简化版：hard voting 对类别投票；soft voting 平均概率再取最大——通常更好，因为用上了置信度信息。

### 2.7 用一个小例子把概念串起来（补充）

三类方法各演一遍（补充）：bagging——5棵树每棵只见63%的样本，错得不重样，平均后方差变小；boosting——第1棵桩把一半样本分错，它们的权重翻倍，第2棵桩被迫切开那块区域，如此接力；stacking——森林在密集区准、线性模型在外围稳，元模型在密集区听森林、外围听线性。共同前提始终是成员的错误别完全相关。

## 三、操作前准备

需要 NumPy；对比 sklearn 时需要 scikit-learn；XGBoost/LightGBM 仅在你想跑原课 Use It 延伸时才需要安装。

~~~bash
python -m pip install numpy scikit-learn
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

原课 Build It 共 4 段。第1段实现决策桩，第2段 AdaBoost，第3段梯度提升，第4段（与 sklearn 对比）原文只描述未给代码。注意：第3段用到的 `SimpleRegressionTree` 在原课文档中未定义——原文缺 `SimpleRegressionTree`，第五节补充。

### 4.1 决策桩：只有一个分裂的基学习器

桩在"每个特征的每个候选阈值×两个极性"里暴力搜索，找加权错误最小的那次分裂。标签必须是 {−1, +1}（AdaBoost 的权重更新公式依赖这个约定）。`fit` 不返回 self，与 sklearn 风格不同，教学代码如此，照抄即可。

~~~python
# 本块中文说明：决策桩=深度1的树，枚举所有(特征,阈值,极性)找加权错误最小的分裂。
class DecisionStump:
    def __init__(self):
        self.feature_idx = None   # 选中的特征编号
        self.threshold = None    # 分裂阈值
        self.polarity = 1        # 方向：1表示小于阈值判-1；-1表示大于阈值判-1
        self.alpha = None        # 由 AdaBoost 填写的模型权重

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):               # 逐特征
            thresholds = np.unique(X[:, f])       # 用实际取值当候选阈值
            for thresh in thresholds:              # 逐阈值
                for polarity in [1, -1]:           # 两个方向都试
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])  # 加权错误率
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred
~~~

### 4.2 AdaBoost：串行调权重

每轮：训练桩→算加权err（clip 避免除零和 log(0)）→算 α→按 α 放大分错样本的权重→归一化。最终预测是所有桩的 α 加权和再取符号。

~~~python
# 本块中文说明：AdaBoost 主循环——训练弱学习器、计算α、更新样本权重。
class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)          # 初始均匀权重

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])            # 加权错误率
            err = np.clip(err, 1e-10, 1 - 1e-10)        # 数值保护

            alpha = 0.5 * np.log((1 - err) / err)       # 模型话语权
            weights *= np.exp(-alpha * y * pred)        # 分错的样本权重变大
            weights /= weights.sum()                    # 归一化

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)                # 加权和的符号即最终类别
~~~

### 4.3 梯度提升：逐棵拟合残差

从"全体均值"这个最粗的预测出发，每轮算残差、用回归树拟合残差、乘学习率后累加。预测时把初始均值和所有树的小步贡献加起来。

~~~python
# 本块中文说明：梯度提升(平方误差)——每棵新树拟合当前残差，小步累加。
class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)               # 第0步：常数预测=均值
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_pred            # 平方损失下负梯度=残差
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update        # 收缩后累加
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
~~~

### 4.4 与 sklearn 对比（原文仅描述）

原课说仓库代码会验证自实现与 `AdaBoostClassifier`、`GradientBoostingClassifier` 的准确率相近，并把所有方法并排比较。该段在文档中没有代码，本文不转录；第六节给出 sklearn 等价用法。

## 五、把代码运行起来：补充入口与检查

（补充）原文缺 `SimpleRegressionTree`，这里补一个最小实现——方差减少选分裂的浅层回归树，接口（`fit(X, y)`/`predict(X)`）与 `GradientBoostingScratch` 的调用对齐：

~~~python
# 本块中文说明：补充的最小回归树——按方差减少量贪心选分裂，供梯度提升调用。
class SimpleRegressionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = 0.0                    # 叶子输出：该节点目标均值

    def fit(self, X, y):
        self.value = float(np.mean(y))
        if self.max_depth <= 0 or np.all(y == y[0]):
            return self                     # 停止条件：深度用尽或已纯净
        best_gain, best_f, best_t = 0.0, None, None
        n, parent_var = len(y), np.var(y)
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f])[:-1]:
                mask = X[:, f] <= t
                nl = int(mask.sum())
                if nl == 0 or nl == n:
                    continue
                gain = parent_var - (nl / n) * np.var(y[mask]) \
                       - ((n - nl) / n) * np.var(y[~mask])   # 加权方差减少
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, t
        if best_f is None:
            return self                     # 没有任何分裂能改善
        self.feature, self.threshold = best_f, best_t
        mask = X[:, self.feature] <= self.threshold
        self.left = SimpleRegressionTree(self.max_depth - 1).fit(X[mask], y[mask])
        self.right = SimpleRegressionTree(self.max_depth - 1).fit(X[~mask], y[~mask])
        return self

    def predict_one(self, x):
        node = self
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
~~~

（补充）最小运行入口，本次未执行训练，仅给出结构：

~~~python
# 本块中文说明：补充的最小入口——生成小数据集，验证AdaBoost与梯度提升可跑通。
rng = np.random.RandomState(0)

# 分类：标签必须是{-1, +1}
X_cls = rng.randn(200, 2)
y_cls = np.where(X_cls[:, 0] + X_cls[:, 1] > 0, 1, -1)
ada = AdaBoostScratch(n_estimators=20)
ada.fit(X_cls, y_cls)
assert set(np.unique(ada.predict(X_cls))).issubset({-1.0, 1.0})

# 回归：正弦加噪
X_reg = rng.uniform(-2, 2, size=(100, 1))
y_reg = np.sin(X_reg[:, 0]) + 0.05 * rng.randn(100)
gb = GradientBoostingScratch(n_estimators=50, learning_rate=0.1, max_depth=3)
gb.fit(X_reg, y_reg)
assert gb.predict(X_reg).shape == (100,)
~~~

结构检查（可手算预期）：桩在单特征 `[1,2,3,4]`、标签 `[1,1,-1,-1]` 时应选阈值 2.5 附近使错误为 0；GB 第一轮拟合的是 `y − mean(y)`。

## 六、Use It：使用现成工具

原课的方法选择表（中文转述，"降低误差"一列是方法的主要倾向，不是保证）：

| 方法 | 主要降低 | 适用 | 注意 |
|---|---|---|---|
| Bagging/随机森林 | 方差 | 噪声大、特征多 | 对偏差没帮助 |
| AdaBoost | 偏差 | 干净数据、简单基学习器 | 对离群点和噪声敏感 |
| 梯度提升 | 偏差 | 表格数据、竞赛 | 训练慢、不调参易过拟合 |
| XGBoost/LightGBM | 两者 | 生产表格ML | 超参数很多 |
| Stacking | 两者 | 榨取最后1~2% | 复杂、元学习器易过拟合 |
| Voting | 方差 | 快速组合多样模型 | 模型不多样就没用 |

原课的表格数据生产顺序：先用 LightGBM/XGBoost 默认参数；再调 n_estimators、learning_rate、max_depth、min_child_weight；要最后0.5%再上3~5个多样模型的 stacking；全程用交叉验证。sklearn 等价入口：

~~~python
# 本块中文说明：sklearn 的 AdaBoost 与梯度提升对应接口。
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ada = AdaBoostClassifier(n_estimators=50, random_state=42)
gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, random_state=42)
~~~

## 七、失败模式、原文边界与工程注意

1. 多样性不达标时集成无效：成员错误高度相关（同一数据、同一特征、同一种子）时，加模型只是加计算量。
2. boosting 轮数过多会拟合噪声"难样本"，验证集不涨就该停；early stopping 交给[第12课](<12：超参数调优——网格、随机与贝叶斯搜索.md>)。
3. 原课"神经网络在表格数据上几乎总是不如梯度提升、短期不会改变"过于绝对。梯度提升是强默认选项，但 TabNet、NODE 等偶有打平；具体数据、具体评估说了算。
4. "21个0.6→约74%""101个→约84%"与其公式不符（核对，见2.1）；且公式本身假设独立，真实收益低于理论。
5. stacking 的元特征必须由训练集内的交叉验证产生，否则元学习器看到基模型背下来的答案，验证分数虚高。
6. `DecisionStump` 用 `np.unique` 的实际取值当阈值、三重循环暴力搜索，特征多或取值多时非常慢——教学实现，不是性能参考（补充）。
7. 原课 `SimpleRegressionTree` 未在文档中给出，本文第五节补充；若与仓库 `code/ensembles.py` 的实现不同，以仓库为准。
8. AdaBoost 标签约定 {−1,+1}；换成 {0,1} 后 `exp(-alpha*y*pred)` 的权重更新就不再是"分错变大"（核对，可从公式直接验证）。

## 八、Ship It：本课可复用的中文检查模板

~~~markdown
集成方案检查：说明数据规模、特征类型、噪声水平与类别平衡；列出候选方法（bagging/boosting/stacking/voting）及各自针对的误差成分；说明多样性来源（数据/特征/序列/模型家族）；给出成员数量、学习率与早停策略；报告交叉验证分数与单独最优成员的对比——集成没赢过最优单模型时说明多样性不足；所有未实际运行的数值标注为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 给 AdaBoost 加"每轮记录训练准确率"，画准确率-轮数曲线，找收敛点。验收：曲线可复现（固定种子），能指出何时不再明显改善；不要求精确收敛轮数。
2. 给回归树加随机特征子采样，实现随机森林：100棵树、`max_features=sqrt(n_features)`，平均预测，对比单树方差。验收：多次划分数据重跑，报告方差变化方向；不预设固定降幅。
3. 给梯度提升加 early stopping：验证损失10轮不改善即停。验收：报告实际用掉的树数与验证曲线；"多少棵够"以你的曲线为准。
4. 三基模型（逻辑回归、决策树、KNN）+逻辑回归元学习器的 stacking，用5折CV产元特征。验收：能展示元特征构造过程无泄漏（基模型从不接触自己产出元特征的同折数据），并与单模型比较。
5. 默认参数 XGBoost 对比自实现梯度提升，并计时。验收：报告准确率差与时间差；速度差取决于环境，不预设倍数。

## 十、术语与完成标准

bagging：bootstrap 抽样并行训练再平均，主要降方差；boosting：串行纠错，主要降偏差；AdaBoost：样本权重指数更新的 boosting；梯度提升：新模型拟合当前损失的负梯度；XGBoost：正则化+二阶导+工程优化的梯度提升；stacking：基模型预测作元学习器特征；OOB：bootstrap 没抽到的约36.8%样本；多样性：成员错误不相关的程度。

完成检查：

- 能手算三个0.6独立模型的多数投票准确率（0.648）。
- 能算 err=0.3 时的 α 和分错样本权重的放大倍数（≈1.53）。
- 能解释平方损失下"拟合残差"与"拟合负梯度"是同一件事。
- 能说出 stacking 元特征为什么必须用 CV 产生。
- 能指出 bagging 与 boosting 各自的过拟合风险长什么样。

## 十一、验证范围与来源许可

原课 Build It 共 4 个分段：第1~3段代码逐字转录并加中文注释（逻辑未改），第4段原文仅文字描述未转录；Use It 的方法表转述为中文。原文缺失的 `SimpleRegressionTree` 与最小入口在第五节明确标为补充。本次验证：把正文 python 代码块抽出存临时 .py 文件，逐块做了 ast.parse 静态语法检查（2026-11-04，Python 3.12，临时脚本逐块解析），全部通过；未运行任何训练或模型，正文所有性能数值均为原课参考或可手算预期。数值核对（多数投票概率）由公式手工求和完成。

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
