# 07：无监督学习——K-Means、DBSCAN与聚类评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-04 增补「零、初学者先读」、概念讲解与纠错；当日与原课网页版核对，Build It 代码与五步结构一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/07-unsupervised-learning) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/02-ml-fundamentals/07-unsupervised-learning/docs/en.md)。原题：Unsupervised Learning。

## 零、初学者先读：没有标签，算法怎么自己发现分组？

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。前置是[第01课](<01：what-is-machine-learning.md>)的“数据 + 标签”框架；距离计算回看 Phase 1 的 Norms & Distances。

### 0.1 先理解这一课在做什么

前面的课都是“有老师”的学习：每条数据配了标准答案，模型照答案学。这一课把答案抽走——像图书管理员收到一批没人分类过的新书，只能靠“内容相似”把它们摆上不同书架。无监督学习（unsupervised learning）做的就是这件事：没人告诉算法每个样本属于哪类，它自己把相似的归到一起，顺便把特别“不合群”的挑出来。

### 0.2 本课术语先翻成白话

- 无监督学习=没有标签、没有标准答案的学习
- 聚类（clustering）=把相似样本自动分组
- 质心（centroid）=一组的“平均点”，组内所有点取均值
- K-Means=先猜 K 个中心，反复“分配→更新”直到稳定
- 惯性（inertia）=每个点到自己中心距离的平方总和，越小簇越紧
- 肘部法（elbow method）=多试几个 K，画惯性曲线，找“再加一簇就不划算”的拐点
- 轮廓系数（silhouette）=每点比较“自己组内平均距离 a”和“最近别组平均距离 b”的打分
- DBSCAN=按密度聚类的算法：密集处成组，稀疏处算噪声
- eps / min_samples=邻域半径 / 成密度区所需最少点数（本课实现含自身）
- 核心点 / 边界点 / 噪声点=密度够格当簇核心的 / 挨着核心点的 / 都不是的（离群候选）
- GMM（高斯混合模型）=假设数据由 K 个高斯分布混合生成，输出“属于每组的概率”
- EM（期望-最大化）=先算归属概率（E 步）再更新参数（M 步），交替进行
- 异常检测（anomaly detection）=把噪声点或低概率点标为异常

### 0.3 用小数字走一遍

一维 K-Means：6 个点 `[1, 2, 3, 10, 11, 12]`，K=2。任选两个初始中心（比如 1 和 2）：第一轮分完后 1 单独一组，其余跟 2；中心更新为 1 和 7.6；再分配变成 `{1,2,3}` 与 `{10,11,12}`，中心变为 2 和 11；再算一遍分组不变，收敛。惯性 = (1²+0²+1²)+(1²+0²+1²) = 4。对这份数据，无论初始中心怎么抽，最终都收敛到这个划分（见第五节的断言）。

轮廓：对点 3，a = 到同组 {1,2} 的平均距离 = 1.5，b = 到最近别组 {10,11,12} 的平均距离 = 8，s = (8−1.5)/8 ≈ 0.81，说明它待得舒服。

DBSCAN：直线上 4 个点 `[0, 0.5, 1.0, 5.0]`，eps=0.6、min_samples=3。0.5 的邻域内有 {0, 0.5, 1.0} 共 3 个点，够格当核心点；0 和 1.0 各只有 2 个邻居，是边界点，跟着 0.5 进同一簇；5.0 谁也够不着，是噪声点。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按依赖顺序读：`euclidean_distance`（一切距离的基础）→ `kmeans`（分配-更新循环）→ `compute_inertia` 与 `silhouette_score`（评估）→ `find_best_k`（枚举选 K）→ `dbscan`（密度扩散）→ `gmm`（EM 循环）→ `make_blobs` / `make_moons`（造测试数据）。核心数据结构很朴素：点是列表的列表；分配结果是“每样本一个簇编号”的列表；DBSCAN 用 -1 表示噪声；GMM 的责任矩阵是“每样本 × 每分量”的概率表。

### 0.5 理解检查（含参考答案）

**问题1：** 同样的数据、同样的 K，为什么两次运行 K-Means 可能得到不同结果？

**参考答案：** 初始中心是从数据里随机抽的，算法只保证收敛到局部最优；不同起点可能落到不同划分。

**问题2：** DBSCAN 输出里的 -1 是什么意思，能拿来做什么？

**参考答案：** -1 表示噪声点，不属于任何簇；原课直接把它当作异常检测的输出。

**问题3：** GMM 与 K-Means 在“归属”上最本质的区别是什么？

**参考答案：** K-Means 是硬分配（每点恰好属于一簇），GMM 是软分配（每点对每簇有一个概率，即责任）。

能解释这三问后，进入正文“核心概念与推导”。

---

## 一、这一课解决什么问题？

前面所有课默认有标签。原课给的三个现实场景：医院有海量病历但没人逐条标注疾病类别；电商有海量会话但没人手工划分客群；安全团队面对海量网络日志但不可能标出每个异常。标注贵、标签少是常态。

无监督学习在没有“正确答案”的前提下找结构：把相似的点分组、发现隐藏模式、暴露异常。代价也随之而来——没有标签就无法直接算“对错”，你需要另一套工具（惯性、轮廓系数、人工抽查）判断算法找到的结构是否有意义。学完本课你能从零实现 K-Means、DBSCAN、GMM，并知道什么时候用哪个。

## 二、核心概念与推导

### 2.1 聚类：相似的放在一起

聚类把每个点分到一个组，使同组内的点彼此更相似、与组外的点更不相似。所有方法都绕不开同一个问题：“相似”怎么度量？本课用欧氏距离，换一种距离就换一种聚类结果。

### 2.2 K-Means：最常用的主力算法

K-Means 把数据切成恰好 K 个簇，每簇一个质心（组内均值），每点属于最近的质心。Lloyd 算法四步：

1. 随机选 K 个点当初始质心
2. 每点分配给最近的质心
3. 每个质心重算为组内点的均值
4. 重复 2-3 直到分配不再变化

目标函数（惯性）是每点到其质心距离平方的总和：

$$\text{inertia}=\sum_{i=1}^{n}\|x_i-\mu_{c(i)}\|^2$$

K-Means 每轮都在让这个数变小，但只找局部最优：不同初始化可能给出不同结果（补充：这就是正式使用时要多换几个种子比较的原因）。

### 2.3 怎么选 K：肘部法与轮廓系数

**肘部法**：对 K=1,2,3,… 逐个跑 K-Means，画惯性随 K 的曲线，找“再加一簇惯性也不再明显下降”的拐点。拐点有时明显有时模糊，最终要人判断。

**轮廓系数**：对每个点 i，记 a(i) 为它到同组其他点的平均距离（凝聚度），b(i) 为它到最近别组各点的平均距离（分离度）：

$$s(i)=\frac{b(i)-a(i)}{\max(a(i),\,b(i))}$$

取值从 −1（分错组）到 +1（分得合理），全数据平均得全局分。直觉：a 小（离自己人近）且 b 大（离外人远）时 s 接近 1。

### 2.4 DBSCAN：按密度找簇

K-Means 假设簇是球形的、还要预先给 K。DBSCAN 两条都不要：它把簇定义为“被稀疏区域分隔的密集区域”。两个参数：eps（邻域半径）与 min_samples（成密度区所需最少点数）。三种点：

- 核心点：eps 范围内至少 min_samples 个点（本课实现含自身）
- 边界点：在某个核心点的 eps 内，但自己不是核心点
- 噪声点：两者都不是，即离群点

互相在 eps 内的核心点连成同一簇，边界点跟随附近的核心点。优点：任意形状的簇、自动定簇数、顺带识别离群点。弱点：各簇密度差异大时不好用（原文明确写了 struggles with varying densities）。

### 2.5 层次聚类（原课概念部分，Build It 无代码）

自底向上（凝聚式）：每点自成一簇，反复合并最近的两个簇直到只剩一簇；合并过程画成树状图（dendrogram），在想用的层剪开得到 K 个簇。簇间“远近”有四种常见定义：单连接（两簇最近点距离）、全连接（最远点距离）、平均连接（平均距离）、Ward 法（使簇内总方差增加最小的合并）。原课只在概念部分介绍，Build It 没有实现（核对：练习 2 让你自己写）。

### 2.6 GMM 与 EM：从“属于哪簇”到“多大可能属于哪簇”

GMM 假设数据由 K 个高斯分布混合生成，每个分量有自己的均值、协方差和混合权重。EM 算法交替：

- E 步：算每个点属于每个高斯的概率（责任）
- M 步：按责任加权更新各分量的均值、协方差、混合权重

相比 K-Means 的硬分配，GMM 给软分配，能表达椭圆簇和互相重叠的簇。注意：本课实现做了简化——每个分量只有一个跨维度共享的标量方差（球形高斯），不是完整协方差矩阵（核对：见 4.4 的 `variances = [1.0] * k`）。

### 2.7 何时用哪个，以及聚类怎么做异常检测

原课对比表（思路参考，不是硬规则）：

| 方法 | 适合 | 避免 |
|---|---|---|
| K-Means | 大数据、球形簇、K 已知 | 不规则形状、有离群点 |
| DBSCAN | 未知 K、任意形状、离群检测 | 密度不均、很高维 |
| 层次聚类 | 小数据、要树状图、未知 K | 大数据（O(n²) 内存）|
| GMM | 重叠簇、需要软分配 | 很大数据、维度太多 |

聚类天然支持异常检测：K-Means 里离所有质心都远的点、DBSCAN 的噪声点、GMM 里所有分量下概率都低的点，都是异常候选。

### 2.8 用一个小例子把概念串起来（补充）

回到 0.3 的 `[1, 2, 3, 10, 11, 12]`：K-Means（K=2）收敛到惯性 4 的划分。若换成两个交错的月牙形簇，K-Means 会沿中间硬切一刀（它只会按“到质心距离”分），DBSCAN 却能顺着密度把两弯分开——这正是 4.5 的 moons 实验。选 K 时，肘部法看惯性拐点、轮廓系数看平均 s 何时最大；两指标打架时，回到业务上问“分这么多组有什么用”。

## 三、操作前准备

本课 Build It 只用 Python 标准库 `random` 与 `math`，无需安装；Use It 示例需要 scikit-learn（按需）：

~~~bash
python -m pip install scikit-learn
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 K-Means 从零实现

初始中心用 `random.sample` 从数据点里抽 K 个（可能抽到同一自然簇的两个点）；每轮先分配再重算质心；空簇用随机数据点补上（原课补丁写法，保留原样）；所有中心移动量小于 1e-6 就停。返回每个样本的簇编号与最终质心。

~~~python
# 本块中文说明：从零实现 K-Means（Lloyd 算法），只用标准库。
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)                    # 固定种子便于复现
    n_features = len(data[0])

    centroids = random.sample(data, k)   # 初始中心直接从数据点里抽

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            # 每点算到 K 个中心的距离，进最近那组
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                # 空簇补丁：随机挑一个数据点当新中心（原课写法，保留）
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
~~~

### 4.2 肘部法与轮廓系数

`compute_inertia` 把“每点到自己质心距离的平方”加总；`silhouette_score` 对每点算 a 与 b（独自成簇或簇数少于 2 时按 0 处理）；`find_best_k` 枚举 K 打印两种指标，但“肘点”要你看曲线决定，函数不会替你选。轮廓需要 O(n²) 次两两距离（补充：样本上万时会明显变慢）。

~~~python
# 本块中文说明：惯性、轮廓系数与枚举 K 的辅助函数。
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0                       # 少于两个簇，轮廓无定义，按 0 处理

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)            # 独自一簇，按 0 处理
            continue

        # a：到同组其他点的平均距离
        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        # b：到最近“别的组”的平均距离
        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
~~~

### 4.3 DBSCAN 从零实现

`region_query` 数一个点 eps 半径内的邻居（含自身，所以核心点判定含自己）；主循环从任一未访问点出发：邻居不足 min_samples 就标 -1（暂判噪声），够就开新簇并放入种子集合；逐个处理种子，未访问的先访问、密度够的把它的邻居也并入种子集合；曾标 -1 的点若后来挨上了某簇，会被改判为该簇的边界点。

~~~python
# 本块中文说明：DBSCAN 密度聚类，噪声点标 -1。
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        # 数 eps 半径内的邻居（含自身，距离为 0）
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1               # 暂判噪声，后面挨上簇可改判
            continue

        labels[i] = cluster_id          # 核心点，开新簇
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    # q 也是核心点，把它的邻居并入待处理集合
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id  # 噪声改判为边界点

            j += 1

        cluster_id += 1

    return labels
~~~

### 4.4 GMM（EM 算法）

初始化随机抽 k 个样本当均值、方差全 1、权重均匀。E 步：对每点算 k 个“权重 × 高斯密度”并归一化成责任；M 步：按责任加权更新权重、均值、方差（方差下限 1e-6 防塌缩，分量责任和太小则跳过更新）。均值移动量小于 1e-6 判收敛，最后按最大责任给出硬分配。注意 `gaussian_pdf` 所有维度共享同一个标量方差——球形高斯的简化，不是完整协方差 GMM。

~~~python
# 本块中文说明：简化版 GMM（球形方差 + EM 迭代）。
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k               # 每分量一个标量方差，跨维度共享
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)                      # 原文此处重名遮蔽外层 d，无碍，保留
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        # E 步：算每点对每分量的责任
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        # M 步：按责任加权更新参数
        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
~~~

### 4.5 生成测试数据并跑通全部

`make_blobs` 造三个高斯团（中心 [2,2]、[8,3]、[5,8]）；`make_moons` 造两个交错的月牙。主程序依次演示：三团上的 K-Means、肘部法、DBSCAN、GMM；月牙上 DBSCAN（能分开）对比 K-Means（分不开，原文自己说 will fail to separate）；最后往数据里塞 3 个远处的点，用 DBSCAN 把它们标为异常。

~~~python
# 本块中文说明：合成数据生成器与完整演示入口。
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
~~~

## 五、把代码运行起来：补充入口与检查

（补充）原课主程序一次跑全部演示；下面是更小的、全部可手算核对的检查入口，逐块定义后运行：

~~~python
# 本块中文说明：补充的最小检查入口，所有数值可手算预期。
tiny_line = [[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]
assign, cents = kmeans(tiny_line, k=2)
inertia = compute_inertia(tiny_line, assign, cents)
# 无论初始中心怎么抽，都收敛到 {1,2,3} 与 {10,11,12}，惯性恒为 4
assert abs(inertia - 4.0) < 1e-9
assert set(assign) == {0, 1}

tiny_dbscan = [[0.0], [0.5], [1.0], [5.0]]
labels = dbscan(tiny_dbscan, eps=0.6, min_samples=3)
assert labels[0] == labels[1] == labels[2] == 0   # 0.5 是核心点，0 与 1.0 是边界点
assert labels[3] == -1                            # 5.0 是噪声点
~~~

这些都是结构断言与手算等式，不涉及训练。想看原课完整演示就把 4.5 的 `__main__` 块原样运行；打印出的簇数、噪声数等以实际运行为准，本文不代填数值。

## 六、Use It：使用现成工具

~~~python
# 本块中文说明：sklearn 中同样算法的接口调用（data 来自 4.5 的 make_blobs）。
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
~~~

原课观点：从零实现展示了这些库到底在算什么——K-Means 就是“分配-重算”循环、DBSCAN 是密度扩散、GMM 是 E/M 交替；库版本额外提供数值稳定性、更聪明的初始化（K-Means++）与 GPU 加速，但核心逻辑相同。`sklearn_silhouette` 导入了但示例未使用（核对：原样保留）。

## 七、失败模式、原文边界与工程注意

1. K-Means 只保证局部最优、对初始中心敏感；正式使用至少换几个种子比较，或改用 K-Means++（练习 1）。
2. 空簇补丁（`random.choice(data)`）是原课简易处理，不能保证收敛后仍无空簇。
3. K-Means 假设球形、大小相近的簇；moons 实验正是它失败的样子，非球形结构不要硬用。
4. DBSCAN 对 eps 敏感；各簇密度差异大时表现差（原文自己承认），高维空间距离失真后“密度”更难定义。
5. 本课 GMM 是球形简化版；sklearn 的 `GaussianMixture` 默认 full 协方差，两者能力不同，不能逐一对照结果。
6. 惯性与轮廓都是内部指标，只衡量几何紧凑度，不代表聚类“语义正确”；无标签时最终靠人工抽查与领域判断。
7. `random.seed` 是全局种子，`make_blobs`、`kmeans`、`gmm` 各自重置它；复现要以“整段脚本同顺序运行”为前提。
8. 原文“~90 分钟”“库版本更快”等是课程层面的描述，不是本机实测。

## 八、Ship It：本课可复用的中文检查模板

（补充，学习版）把下面模板存进项目随手可用：

~~~markdown
检查聚类结果：写明算法与参数（K，或 eps/min_samples）；报告簇数、每簇大小、噪声点数；用肘部法或轮廓系数说明 K 的选择过程，并注明指标局限；对 DBSCAN 噪声点与“离所有质心都远”的点抽查原样本确认是否真异常；同一数据换种子重复一次，报告结果是否稳定；区分“原课参考”数值与本机实测数值。
~~~

## 九、Exercises：保留原课练习与验收方法

1. （补充定级：Easy）实现 K-Means++ 初始化：第一个中心随机选，后续每个中心按“到最近已有中心距离的平方”为比例的概率选取，与随机初始化对比收敛速度。验收：能复现两种初始化、报告收敛轮数差异；结论基于多次运行，不要求某种初始化绝对更快。
2. （Medium）加层次凝聚聚类：实现 Ward 连接，输出合并过程（嵌套列表形式的树状图），在不同层剪开与 K-Means 对比。验收：能解释每次合并为何发生；对比基于同一数据与同一指标。
3. （Hard）做异常检测流水线：同一数据分别跑 DBSCAN 与 GMM，把“DBSCAN 噪声 ∩ GMM 低概率”的点标为双方一致的异常。验收：报告两法各自的异常数与重合数，讨论分歧样本的可能原因；不预设两法必然一致。

## 十、术语与完成标准

聚类：把相似样本自动分组；质心：簇均值代表点；惯性：到质心距离平方和；轮廓系数：(b−a)/max(a,b)；核心/边界/噪声点：DBSCAN 的三种密度角色；EM：E 步算责任、M 步更新参数的交替；软分配：每点给概率而非唯一归属；异常：不合群的点，DBSCAN 噪声或 GMM 低概率。

完成检查：

- 能手推 0.3 的 K-Means 两轮迭代与惯性 4。
- 能说出 a(i)、b(i) 各量什么，以及 s 的取值范围。
- 能判断一份数据该用 K-Means、DBSCAN 还是 GMM，并说出理由与风险。
- 能指出 4.3 代码里噪声点“改判”边界点发生在哪一行。
- 能说明本课 GMM 相对完整 GMM 简化了什么。

## 十一、验证范围与来源许可

原课 Build It 共 5 个分段，相关代码与演示均嵌入本篇；Use It 代码保留。第五节检查入口为补充，不冒充原文代码。本次验证：与原课网页版逐段核对代码与五步结构；第五节断言为手算可验的等式与结构检查，未执行任何训练或完整演示。正文 Python 代码块抽取后经 `python -c "import ast; ast.parse(...)"` 做静态语法检查，结果见文末报告；含打印输出的 `__main__` 演示块只做语法检查、未运行，相关输出数值不写成实测。

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
