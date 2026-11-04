# 06：KNN与距离度量——近邻投票、K值与维度灾难

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/06-knn-and-distances) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/02-ml-fundamentals/06-knn-and-distances/docs/en.md)。原题：K-Nearest Neighbors and Distances。

## 零、初学者先读：新来的家伙，问问他邻居是谁

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。距离与范数的直觉对应 Phase 1 第 14 课；分类的评估指标见[第03课](<03：逻辑回归——sigmoid、交叉熵与决策边界.md>)。

### 0.1 先理解这一课在做什么

想判断一个新住户的收入水平？看他住的那栋楼里离他最近的几户人家。KNN（K 最近邻）就是这个思路：不学任何参数，把训练数据整个存起来；预测时算新点到所有训练点的距离，取最近的 K 个，分类就让他们投票、回归就取平均。它“懒”到没有训练阶段，却在中小数据集上出奇地好用——而且它就是今天向量数据库、RAG 检索的算法内核，只是名字换了。

### 0.2 本课术语先翻成白话

- K=投票时请几个邻居，唯一的超参数
- 训练阶段=KNN 没有训练阶段，只是把数据存起来
- 惰性学习（lazy learning）=先不干活，预测时才算（KNN）
- 急切学习（eager learning）=训练时就学好紧凑模型（线性回归、SVM 等）
- L2 / 欧氏距离=直线距离，平时说的“距离”
- L1 / 曼哈顿距离=横平竖直走的距离，对离群点更稳
- 余弦距离=只看向量夹角不看长度，文本/词向量的标配
- Minkowski 距离=带参数 p 的一族距离：p=1 是 L1，p=2 是 L2，p→∞ 是切比雪夫（最大坐标差）
- 距离加权（weighted KNN）=近的邻居票更重，权重取 1/(距离+ε)
- 维度灾难（curse of dimensionality）=维度一高所有点距离趋同，“最近”失去意义
- KD 树=按特征轴反复对半切空间的二叉树，低维时查询快
- 球树=用嵌套超球面切空间的树，中维度更好
- 近似最近邻（ANN）=亿级规模的工程方案（HNSW、IVF 等），牺牲精确换速度
- 特征缩放=距离类方法的前提，否则大数值特征独占“远近”

### 0.3 用小数字走一遍

一维训练点：A(0)、B(1)、C(4)，A、B 是类 0（回归值 1、3），C 是类 1（回归值 10）。查询点 q=0.9。

距离：到 A 是 0.9，到 B 是 0.1，到 C 是 3.1。

- K=1 分类：最近是 B → 判类 0。
- K=3 分类：三票里类 0 拿两票 → 还是类 0。
- K=2 回归（不加权）：平均值 (3+1)/2=2。
- K=2 回归（加权）：权重 1/0.1=10、1/0.9≈1.11，加权平均 (10×3+1.11×1)/11.11≈2.8——离得近的 B 把预测拉向自己的 3。

再感受一下距离族：a=(1,0)、b=(0,1)：L2=√2≈1.414，L1=2，余弦距离=1−0=1；而 a=(1,1)、b=(2,2)：L2=√2，余弦距离=0（方向相同）——选哪种“距离”直接决定谁是谁的邻居。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：四个距离函数（纯数学，无状态）→ `KNN` 类（fit 只存数据，predict 调 `_predict_one`——这个方法原文没给，第五节补齐）→ `KDTree`（原课只给骨架，完整版在仓库 `code/knn.py`，第五节给简化实现）→ `standardize`。核心数据结构：`X_train`/`y_train` 两个平行列表，第 i 个样本 = (X_train[i], y_train[i])；`distance_fn` 作为参数传入，函数本身当值用。

### 0.5 理解检查（含参考答案）

**问题1：** K=1 和 K=N（样本总数）各是什么极端？

**参考答案：** K=1 时边界贴着每个点走，训练误差为零、方差极大（过拟合）；K=N 时永远输出全局多数类，偏差最大（欠拟合）。K 是偏差与方差的旋钮。

**问题2：** 为什么文本要用余弦距离而不是 L2？

**参考答案：** 文本向量的长度往往只反映文档长短或词频总量，方向才携带“讲什么”的信息；余弦忽略长度只看方向。此外高维下 L2 受维度灾难影响更重。

**问题3：** 为什么 KNN 必须标准化特征？

**参考答案：** 距离把各维差异直接相加：0～1000 的特征会把 0～1 的特征完全淹没，“最近邻”实际由那一个大数值特征决定。

三问都清楚了就进正文。

---

## 一、这一课解决什么问题？

来了一个新数据点，要给它分类或估个值。与线性回归、SVM 不同，KNN 不从数据学参数：找到离新点最近的 K 个训练点，让他们投票。没有训练阶段、没有要学的参数、没有要最小化的损失。听起来简单得不像能用的算法，但它在很多中小规模问题上相当有竞争力；而且它把三个基础概念摆上台面：距离度量的选择、维度灾难、惰性与急切学习之分。原课特别指出：现代 AI 里的向量数据库、RAG（检索增强生成）、推荐系统，底层做的就是 KNN 检索，只是规模和数据结构不同。

## 二、核心概念与推导

### 2.1 KNN 的工作流程

给定带标签的训练点和一个查询点：1) 算查询点到每个训练点的距离；2) 按距离排序；3) 取最近 K 个；4) 分类则多数投票，回归则取（加权）平均。就这么多——没有拟合、没有梯度下降、没有轮数。

~~~text
查询点 ? ──▶ 对全部训练点算距离 ──▶ 排序 ──▶ 取 K 个
                                                  ├── 分类：多数投票 ──▶ 预测
                                                  └── 回归：平均 ─────▶ 预测
（补充：原课流程图的文字版）
~~~

### 2.2 K 的选择：唯一的超参数

| K | 行为 |
|---|---|
| K=1 | 边界贴每个点走，训练误差为零，高方差，过拟合 |
| 小 K（3–5） | 对局部结构敏感，能抓复杂边界 |
| 大 K | 边界更平滑，抗噪，但可能欠拟合 |
| K=N | 处处预测多数类，偏差最大 |

经验起点：$K\approx\sqrt{N}$；二分类用奇数 K 避免平票（原课建议，非硬规则）。

### 2.3 距离度量：定义什么叫“近”

**L2（欧氏）**是默认——直线距离：$d(a,b)=\sqrt{\sum_i(a_i-b_i)^2}$。对特征尺度敏感，用前先标准化。

**L1（曼哈顿）**绝对值求和：$d(a,b)=\sum_i|a_i-b_i|$。不平方，对离群点更稳。

**余弦距离**只看夹角：$d(a,b)=1-\frac{a\cdot b}{\|a\|\,\|b\|}$。文本、词向量的标配。

**Minkowski** 一族带参数 p：$d(a,b)=\left(\sum_i|a_i-b_i|^p\right)^{1/p}$；p=1 曼哈顿、p=2 欧氏、p→∞ 切比雪夫（各维最大差）。

选哪种看数据（原课表格）：数值特征同量纲用 L2；有离群点用 L1；文本/嵌入用余弦；高维稀疏用余弦或 L1；混合类型自造组合距离。

### 2.4 距离加权 KNN

等权投票让距离 5.0 的邻居与 0.1 的邻居话语权相同，不合理。加权版给每个邻居 $w_i = 1/(\text{distance}_i+\epsilon)$：分类做加权计票，回归做加权平均 $\frac{\sum_i w_i y_i}{\sum_i w_i}$。ε 防止查询点与训练点重合时除零。加权让结果对 K 不那么敏感——远邻居本来就没什么票。

### 2.5 维度灾难：KNN 的天敌

高维下 KNN 退化，这不是玄学而是数学事实，三个表现：

1. **距离趋同**：维度升高时，最大距离与最小距离之比趋于 1（原课示意：d=100 时约 1.01，d=1000 时约 1.001）。所有点一样远，“最近”就没有意义。
2. **体积爆炸**：要圈住 K 个邻居，搜索半径必须覆盖特征空间的一大块。
3. **角落主导**：d 维单位超立方体的体积集中在角落；内切球占的比例随 d 增大趋于 0。

实践结论（原课）：特征在 20–50 个以内 KNN 表现较好；更多先降维（PCA、UMAP、t-SNE）或用能利用数据内在线维度的树结构。

### 2.6 KD 树：快的近邻搜索

暴力 KNN 每次查询 $O(n\cdot d)$。KD 树递归地按特征轴切空间：每层选一个维度、在median（中位数）处切一刀。查询时先走到查询点所在的叶子，再回溯检查“对面分支有没有可能更近”，多数分支可剪掉。低维平均查询 $O(\log n)$；但 d>20 左右退化为 $O(n)$——能剪的分支越来越少。

### 2.7 球树与近似最近邻

球树用嵌套超球面（球心+半径）而不是轴对齐的盒子切空间：中维度（约 50 以内）更好、对非轴对齐结构更友好、包围更紧剪枝更多。KD 树与球树都是精确算法；真正的亿级、高维检索用近似最近邻（HNSW、IVF、乘积量化）。原课说这些在 Phase 1 第 14 课展开。

### 2.8 惰性学习 vs 急切学习

| 维度 | 惰性（KNN） | 急切（SVM、神经网络） |
|---|---|---|
| 训练时间 | $O(1)$，只存数据 | $O(n\times\text{epochs})$ |
| 预测时间 | 每次查询 $O(n\cdot d)$ | $O(d)$ 或 $O(\text{参数量})$ |
| 预测时内存 | 存整个训练集 | 只存模型参数 |
| 新数据 | 加点即用 | 要重训 |
| 决策边界 | 隐式，查询时现算 | 显式，训练后固定 |

惰性适合：数据频繁增删、查询次数很少、要零训练时间、数据小到暴力搜也不慢。

### 2.9 KNN 回归

预测 = K 个邻居目标值的平均（或 2.4 节的加权平均）。输出是分段常数（加权时分段平滑），且**不能外推**：训练目标都在 0～100 之间时，KNN 永远不会预测出 200——预测值总落在邻居取值范围内（核对：平均不超出被平均的数的最小最大值）。

### 2.10 用一个小例子把概念串起来（补充）

用 0.3 节的 A(0)、B(1)、C(4)：q=0.9 的三个距离是 0.9、0.1、3.1。K=1 判类 0（跟 B 走）；K=3 投票 2:1 仍是类 0，但 C 的一票已把边界拉宽——这就是“K 越大越平滑”。回归 K=2：不加权 2，加权 2.8（近邻 B 权重 10 倍于 A）。再把 C 挪到 4→400：L2 距离变 399.1，但“谁最近”不变；可若把 A 的第一个坐标改成另一个取值 0～1000 的特征，L2 立刻被它主导——这提示标准化不是可选项。最后把点放到 100 维：随机均匀采样时各点到 q 的距离几乎一样（2.5 的趋同现象），K=1 的选择近乎随机。

## 三、操作前准备

Build It 用 Python 标准库（math）；Use It 需要 scikit-learn，FAISS 示例需另装 faiss-cpu：

~~~bash
python -m pip install scikit-learn
python -m pip install faiss-cpu  # 仅 Use It 的 FAISS 示例需要
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

原课 Build It 共 4 个分段；第 2 步的 `KNN` 类不完整（缺 `_predict_one`），第 3 步的 `KDTree` 只给骨架，完整实现原课指向仓库 `code/knn.py`。缺的部分在第五节补齐。

### 4.1 第 1 步：距离函数

四个纯函数。注意余弦距离在任一向量为零时返回 1.0（约定值）；Minkowski 对 p=∞ 特判为最大坐标差。

~~~python
# 本块中文说明：L1、L2、余弦、Minkowski 四个距离函数。
import math

def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def cosine_distance(a, b):
    dot_val = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot_val / (norm_a * norm_b)

def minkowski_distance(a, b, p=2):
    if p == float('inf'):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)
~~~

### 4.2 第 2 步：KNN 分类器与回归器（类框架）

`fit` 只是把数据存进属性——这就是“惰性”。`predict` 对每个查询点调 `_predict_one`（原文未给出其实现，见第五节补充）。`distance_fn` 和 `weighted`、`task` 都是构造参数，一套类同时覆盖分类与回归。

~~~python
# 本块中文说明：KNN 类框架（fit 存数据，predict 逐点预测）。
class KNN:
    def __init__(self, k=5, distance_fn=l2_distance, weighted=False,
                 task="classification"):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_one(x) for x in X]
~~~

### 4.3 第 3 步：KD 树（骨架）

原课只给出结构骨架（省略号处为未展示的实现），说明思路：按当前深度轮换轴、在 median 处切分；查询先下到叶子、再回溯检查兄弟分支。完整实现见仓库 `code/knn.py`，本课程页面未展示。

~~~python
# 本块中文说明：原课给出的 KD 树骨架（省略号为未展示的实现）。
class KDTree:
    def __init__(self, X, indices=None, depth=0):
        # Recursively partition the data
        self.axis = depth % len(X[0])
        # Split on median of the current axis
        ...

    def query(self, point, k=1):
        # Traverse to leaf, then backtrack
        ...
~~~

### 4.4 第 4 步：特征标准化

对每列减均值、除标准差；标准差下限 1e-10 防除零（零方差列全部映射为 0 附近的值）。注意本函数统计量在整份数据上算，严格流程应在训练集上拟合（见第七节）。

~~~python
# 本块中文说明：按列标准化（均值 0、标准差 1）。
def standardize(X):
    n = len(X)
    d = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds = [
        max(1e-10, (sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) ** 0.5)
        for j in range(d)
    ]
    return [[((X[i][j] - means[j]) / stds[j]) for j in range(d)] for i in range(n)], means, stds
~~~

## 五、把代码运行起来：补充入口与检查

（补充）原文缺 `KNN._predict_one` 与 KD 树完整实现。以下补齐 `_predict_one`——分类多数投票（可选加权），回归（加权）平均；K 超过样本数时取全部：

~~~python
# 本块中文说明：补充原文缺失的 _predict_one 实现。
def _predict_one(self, x):
    dists = [(self.distance_fn(x, xi), yi)
             for xi, yi in zip(self.X_train, self.y_train)]
    dists.sort(key=lambda t: t[0])
    neighbors = dists[:min(self.k, len(dists))]
    if self.task == "regression":
        if self.weighted:
            total_w = sum(1.0 / (d + 1e-9) for d, _ in neighbors)
            return sum((1.0 / (d + 1e-9)) * yv for d, yv in neighbors) / total_w
        return sum(yv for _, yv in neighbors) / len(neighbors)
    if self.weighted:
        votes = {}
        for d, yv in neighbors:
            votes[yv] = votes.get(yv, 0.0) + 1.0 / (d + 1e-9)
    else:
        votes = {}
        for _, yv in neighbors:
            votes[yv] = votes.get(yv, 0) + 1
    return max(votes.items(), key=lambda t: t[1])[0]

KNN._predict_one = _predict_one
~~~

（补充）KD 树给出一个可运行的最小实现（每层轮换轴、中位数切分、回溯剪枝），与原课 `code/knn.py` 的完整版相比是简化版：

~~~python
# 本块中文说明：补充的最小可运行 KD 树（简化版）。
class SimpleKDTree:
    def __init__(self, points, indices=None, depth=0):
        self.points = points
        if indices is None:
            indices = list(range(len(points)))
        self.axis = depth % len(points[0])
        indices.sort(key=lambda i: points[i][self.axis])
        m = len(indices) // 2
        self.idx = indices[m]                     # 本节点的样本下标
        self.left = SimpleKDTree(points, indices[:m], depth + 1) if m > 0 else None
        self.right = SimpleKDTree(points, indices[m + 1:], depth + 1) if m + 1 < len(indices) else None

    def query(self, q, k=1):
        best = []                                 # [(距离, 样本下标)]，保持升序

        def visit(node):
            if node is None:
                return
            d = l2_distance(q, self.points[node.idx])
            best.append((d, node.idx))
            best.sort()
            del best[k:]
            diff = q[node.axis] - self.points[node.idx][node.axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            if len(best) < k or abs(diff) < best[-1][0]:
                visit(far)                        # 对面才可能有更近的点

        visit(self)
        return best
~~~

补充的最小检查（全部可手算预期，对照 0.3 节）：

~~~python
# 本块中文说明：补充的手算断言入口。
X = [[0.0], [1.0], [4.0]]

knn_c = KNN(k=1, task="classification")
knn_c.fit(X, [0, 0, 1])
assert knn_c.predict([[0.9]]) == [0]            # 最近邻是 B(1)，类 0

knn_r = KNN(k=2, task="regression")
knn_r.fit(X, [1, 3, 10])
assert abs(knn_r.predict([[0.9]])[0] - 2.0) < 1e-12      # (3+1)/2

knn_w = KNN(k=2, task="regression", weighted=True)
knn_w.fit(X, [1, 3, 10])
assert abs(knn_w.predict([[0.9]])[0] - 2.8) < 0.01       # 权重 10 : 1.11

assert abs(l2_distance([1, 0], [0, 1]) - math.sqrt(2)) < 1e-12
assert l1_distance([1, 0], [0, 1]) == 2
assert abs(cosine_distance([1, 0], [0, 1]) - 1.0) < 1e-12
assert abs(cosine_distance([1, 1], [2, 2])) < 1e-12        # 同向 → 0
assert minkowski_distance([1, 2], [4, 6], p=float('inf')) == 4

tree = SimpleKDTree([[0.0], [1.0], [4.0], [9.0]])
best = tree.query([0.9], k=2)
assert best[0][1] == 1 and best[1][1] == 0       # 最近是 B，次近是 A
~~~

补充检查方法：对同一查询比较 KD 树与暴力排序的前 k 个是否一致（随机小数据上抽查多组）。

## 六、Use It：使用现成工具

scikit-learn 会根据数据规模与维度自动在 KD 树、球树、暴力之间选择（可用 `algorithm` 参数控制）。FAISS 用于大规模向量检索。

~~~python
# 本块中文说明：sklearn 的 KNN 管道（原文示例）。
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
~~~

~~~python
# 本块中文说明：FAISS 亿级向量近邻检索（原文示例，需先安装）。
import faiss

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
distances, indices = index.search(query_vectors, k=5)
~~~

核对：两个示例的 `X_train, y_train, embeddings` 等变量原课未定义，运行需自备数据（原文边界，补充声明）。

## 七、失败模式、原文边界与工程注意

1. 不标准化就跑 KNN 是最常见事故：大数值特征独占距离，其余特征等于没参与（原课强调）。
2. 数据频繁变化时 KNN 的“免重训”是真优势，但预测成本 $O(n\cdot d)$ 会在数据增长后反噬；查询多、数据大时要上树结构或 ANN。
3. 高维下先降维再 KNN：原课“20–50 个特征以内好用”是经验区间，不是硬阈值；正确做法是用实验确认距离是否已趋同。
4. KD 树在 d>20 左右退化到接近暴力搜索；中高维考虑球树，更大规模上 ANN——精确性换速度是有代价的选择（补充）。
5. KNN 回归不能外推：目标范围外的值永远预测不出来（原文已指出）；趋势外推的任务应回归参数模型（[第02课](<02：线性回归——最小二乘、梯度下降与正规方程.md>)）。
6. 4.4 的 `standardize` 在全部数据上算均值方差：严格流程应只在训练集上拟合统计量，测试集只变换，否则测试信息泄漏（补充；与[第03课](<03：逻辑回归——sigmoid、交叉熵与决策边界.md>)Use It 的正确示范对比）。
7. 平票问题：偶数 K 的二分类可能 2:2 平票；本篇补充实现用 `max` 按先到先得打破平局，更稳的做法是加权或奇数 K（补充）。
8. 类别不平衡时多数投票偏向多数类：可考虑加权投票或重采样（补充，呼应[第03课](<03：逻辑回归——sigmoid、交叉熵与决策边界.md>)的不平衡讨论）。
9. 原文“K=√N 起步”“用奇数 K”是经验法则，不是理论最优；K 应由交叉验证选（补充）。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 未给出独立产出，本模板按本课要点整理（补充）：

~~~markdown
检查 KNN：说明特征是否标准化、标准化器是否只在训练集拟合；说明 K 与距离度量的选择依据（交叉验证或实验）；类别不平衡时报告精确率/召回率而不只准确率；高维数据说明是否降维、距离是否趋同（抽查最大/最小距离之比）；数据量大时说明用了暴力、树结构还是 ANN 及理由；所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：二维三分类数据上做 KNN，画 K=1、5、15、N 的决策边界。验收标准：能展示从锯齿状过拟合到平坦欠拟合的过渡，并解释各图差异。
2. 原课 Medium：在 2、5、10、50、100、500 维各生成 1000 个随机点，算两两距离的最大/最小之比并画图。验收标准：比值随维度上升趋近 1，与 2.5 节的趋同现象一致。
3. 原课 Medium：文本分类（TF-IDF 特征）上比较 L1、L2、余弦的 KNN 准确率。验收标准：能报告三者的验证集准确率并解释余弦为何倾向占优（方向携带语义、长度是噪声）。
4. 原课 Hard：实现 KD 树，在 1k/10k/100k 点、2/10/50 维上比较查询时间与暴力搜索。验收标准：能找到 KD 树失去优势的维度拐点，并解释剪枝失效的原因。
5. 原课 Hard：给 y=sin(x)+噪声 做加权 KNN 回归，K=3、10、30 与不加权对比。验收标准：能展示加权在大 K 时预测更平滑（近邻主导），并说明为何（远邻居权重自动衰减）。

## 十、术语与完成标准

KNN：按 K 个最近训练点投票/平均的非参数算法；惰性/急切学习：训练时不干活 vs 训练时学紧凑模型；维度灾难：高维下距离趋同、邻域膨胀；KD 树：按特征轴中位数递归切分的搜索树，低维 $O(\log n)$；球树：嵌套超球面的搜索树，中维度更优；加权 KNN：权重 1/(距离+ε)；特征缩放：距离方法的前提；多数投票：K 邻居里数最多的类；暴力搜索：对全部点算距离，$O(n\cdot d)$；近似最近邻：HNSW/LSH/IVF 等以精度换速度；Voronoi 图：K=1 时的边界划分。

完成检查：

- 能手算一个查询点的 K=1/3 分类与 K=2 加权回归（0.3 节算例）。
- 能写出 L1、L2、余弦公式并各举一个适用场景。
- 能解释维度灾难的三个表现及应对。
- 能说出惰性与急切学习在训练/预测成本上的差别。
- 能说明 KD 树为什么在高维失效。
- 能指出 KNN 回归不能外推的原因。

## 十一、验证范围与来源许可

原课 Build It 共 4 个分段，已逐字转录嵌入本篇（仅添加中文说明与注释，未改逻辑）。原文缺 `KNN._predict_one` 实现与 KD 树完整实现（原课指向仓库 `code/knn.py`，本篇未抓取该文件），已在第五节标注“原文缺 X，第五节补充”并给出补充实现（KD 树为声明的简化版）。原课未展示运行输出，本篇未虚构数值。写作完成后，正文 python 代码块逐一抽出为临时 .py 文件，用 `python -c "import ast; ast.parse(...)"` 做静态语法检查，结果见下。

本次验证记录（2026-11-04）：
- raw 文档 502 失败后改用课程页 URL 成功获取原文，以原文为准整理。
- 正文 9 个 python 代码块全部通过 `ast.parse` 静态语法检查（含第五节补充实现）；第五节"可手算预期"断言（KNN 分类/加权回归、四个距离、补充 KD 树查询）另在临时脚本中实际执行复核通过；sklearn/FAISS 的 Use It 示例未执行。
- 未做：真实数据实验、FAISS 安装与运行、与 sklearn 的数值比对、`code/knn.py` 的核对。

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
