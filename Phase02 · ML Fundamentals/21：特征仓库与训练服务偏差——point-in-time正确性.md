# 21：特征仓库与训练服务偏差——point-in-time正确性

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/21-feature-stores-and-training-serving-skew) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/02-ml-fundamentals/21-feature-stores-and-training-serving-skew/docs/en.md)。原题：Feature Stores and Training-Serving Skew。

## 零、初学者先读：同一个特征，为什么训练和线上会算出两个不一样的值

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。前置：[第08课](<08：特征工程——构造、变换、缩放与泄漏.md>)、[第15课](<15：时间序列——平稳性、滞后特征与滚动验证.md>)、[第19课](<19：ML系统设计——从问题定义到部署模型.md>)。

### 0.1 先理解这一课在做什么

训练像考古：问题是“标签那一刻，这个用户长什么样”，要挖出历史某天的旧状态。线上服务像看直播：问题是“这个用户现在什么样”，永远拿最新值。同一个特征名“距上次购买天数”，两个读者在不同时间问，答案天然不同。更糟的是：如果训练时挖错了层（拿了几周后的新快照），特征里直接混进了标签的答案。这一课造一个小型特征仓库（feature store）：一套定义、两个存储（历史给训练、最新给线上），加上一条防止“时间旅行”的连接规则。

### 0.2 本课术语先翻成白话

- 特征仓库（feature store）=从一套定义出发、同时喂训练和线上的特征系统
- 离线存储（offline store）=带时间戳的完整特征历史，给训练用
- 在线存储（online store）=每个实体只留最新一行，给线上毫秒级读取
- 实体（entity）=特征描述的对象键，如 user_id、card_id
- 事件时间戳（event timestamp）=特征值“变成真的”那一刻，如“截至第14天”就是第14天
- 创建时间戳（created timestamp）=系统把这一行写下来的时刻；回填会让它晚于事件时间
- TTL（time to live）=值的最大年龄，超龄当缺失处理、走默认值
- 物化（materialization）=把离线存储的最新值拷贝进在线存储的定时任务
- 回填（backfill）=让新特征在过去的每个日期上重算一遍、补出历史
- point-in-time join=每个标签只配“标签时刻或更早”的最新特征行
- 训练服务偏差（training-serving skew）=同一个特征在两条路径上算出了不一样的值
- 特征定义/注册表（registry）=特征的一份权威定义：实体、来源、变换、窗口、默认值、TTL、版本

### 0.3 用小数字走一遍

原文的核心小表：某用户，标签在第 16 天，特征快照每周一行——

| 特征行（天） | purchases_30d | point-in-time join | 只按 user_id 的朴素 join |
|---|---|---|---|
| 7 | 4 | | |
| 14 | 3 | **选中** | |
| 21 | 1 | 未来 | |
| 28 | 0 | 未来 | **选中** |

标签问“第 17-46 天用户买没买东西”。第 28 天那行是 0，正是因为用户停止了购买——这正是标签要量的事件。朴素 join 把“最新快照”（第 28 天）接给标签，等于把答案抄进了输入。point-in-time join 选中第 14 天：不早于、也不晚于标签时刻的最近一行。再加 TTL 手算：若 TTL=10 天，第 14 行距标签 16−14=2 天 ≤ 10，保留；若这行在第 5 天，距标签 11 天 > 10，丢弃、填默认值。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

原课 `code/main.py` 只用标准库，时间用整数天数。按依赖顺序读：`FeatureDefinition.compute`（单特征、单时点，绝不读 as_of 之后的事件）→ `backfill`（对每个实体×每个日期调用 compute，填出离线存储）→ `point_in_time_join`（排序 + `bisect_right` 找“不晚于标签日的最后一行” + TTL 检查）→ `OnlineStore.write/read`（每实体一行 + TTL 拒读过期）→ `detect_skew`（按键比对两份视图）。核心数据结构两个：`FeatureRow`（实体、天、特征值字典）与标签行。注意：`Event`、`FeatureRow`、`LabelRow`、`JoinedRow`、`FeatureRegistry`、`group_events` 等定义原文页面未给出，第五节补充最小版本。

### 0.5 理解检查（含参考答案）

**问题1：** 朴素 join 为什么会让离线分数虚高？

**参考答案：** 它给每个标签接上“最新”快照，而最新快照的时间晚于标签、包含标签结果的后果（如用户已停止购买）。特征里含答案，模型在离线轻易得高分，上线遇到真·未知未来的特征就掉分。

**问题2：** 在线存储里 TTL 过期的值怎么处理？

**参考答案：** 当作缺失：`read` 返回 None，读取方按注册表里文档化的默认值填充。默认值必须两条路径同源，否则又是一种偏差。

**问题3：** 训练服务偏差的四种成因是什么？

**参考答案：** 两条代码路径（同一特征两处实现）、特征过期度不同（训练读周快照、服务现算最新）、默认值不同（365 vs −1）、时间旅行（训练读到标签之后的值）。

能答对这三问后，进入正文“核心概念与推导”。

---

## 一、这一课解决什么问题？

一个团队做流失模型：每个标签说“某用户在标签日之后 30 天内是否零购买”。他们按 `user_id` 把标签表 join 到用户特征表上训练一条 `days_since_last_purchase` 规则，离线准确率 0.948。上线后模型反而不如旧的简单版本。原因一：特征表存的是每个用户的**最新**值——来自标签日之后几周；一个第 115 天购买的用户在第 175 天的快照里显示“闲置 60 天”，特征直接包含了答案。原因二：另一个团队在另一个服务里**重写**了服务侧特征：30 天窗口多算一天、新用户默认值是 −1 而不是 365。两个缺陷同根：**训练和服务用两种方式计算“同一个特征”**。这一课从零造特征仓库的零件：注册表、回填、point-in-time join、带 TTL 的在线存储、偏差检测，演示用数字复现两个缺陷。

## 二、核心概念与推导

### 2.1 同一特征的两种读者

训练与服务要的特征形状不同：

| 属性 | 训练（离线） | 服务（在线） |
|---|---|---|
| 问题 | “这 1000 万个（实体，时间）对各是什么值？” | “这一个实体现在是什么值？” |
| 时间 | 过去的很多时点 | 现在 |
| 体量 | 每个任务百万行 | 每请求一行 |
| 延迟 | 分钟级可接受 | 几毫秒 |
| 存储 | 离线库：带时间戳全历史，列式文件或数仓 | 在线库：每实体最新值，键值数据库 |

特征仓库用**一套定义**填两个存储；物化（materialization）就是把离线存储的最新值拷进在线存储的任务。原料事件 → 注册表里的定义 → 离线存储（历史）→ point-in-time join 配标签 → 训练集；同一离线存储 → 物化 → 在线存储 → 服务请求 → 记录线上特征 → 与训练路径做偏差检查——一条闭环。

### 2.2 实体、两种时间戳与 TTL

每一行特征用四个词描述：**实体**（特征描述谁）；**事件时间戳**（值“变成真的”时刻——“截至第14天的购买数”事件时间是第14天）；**创建时间戳**（系统写入这一行的时刻——今天跑的回填写的是“旧事件时间+新创建时间”）；**TTL**（值的最大年龄，超龄视为缺失、走默认值）。同实体、同来源、同 TTL 的一组特征常称一个特征视图（feature view）。

### 2.3 Point-in-time 正确连接

一条训练行 = 一个标签 + 标签时刻的特征值。对每个标签四步：

1. 找同实体的特征行；
2. 留下事件时间 ≤ 标签时间的行；
3. 取其中最新一行；
4. 若标签时间 − 该行事件时间 > TTL，丢弃。

对照 0.3 的表：第 16 天的标签选中第 14 天的行；朴素 join（只按 user_id 取最新）选中第 28 天的行——那行是 0 正因为用户停止了购买，也就是标签本身。[第15课](<15：时间序列——平稳性、滞后特征与滚动验证.md>)按时间切分是同一条原则用在“一个切分点”；point-in-time join 把它用在**每一行**上。

### 2.4 训练服务偏差的四种成因

Google《Rules of Machine Learning》定义训练服务偏差为“训练期表现与服务期表现之差”，并点名三种来源：两条管道数据处理不同、训练与服务之间数据变化、反馈回路。落到特征管道，四种具体形态反复出现：

| 成因 | 发生了什么 | 演示中的例子 |
|---|---|---|
| 两条代码路径 | 训练与服务用不同代码算同一特征 | 服务窗口包含 `now−30` 那天，训练窗口不含 |
| 特征过期度 | 服务读到的值与训练用的值年龄不同 | 训练读周快照，新服务现算最新值 |
| 默认值不同 | 两条路径填缺失的方式不同 | 训练用 365 表示“从未购买”，服务用 −1 |
| 时间旅行 | 训练读到标签之后的值 | 朴素 join 给每个标签取最新快照 |

同一指南第 32 条：尽可能在训练管道与服务管道复用代码——特征仓库就是用“两条路径都调用同一份定义”来落实这条规则。

### 2.5 特征注册表：一份定义，两个读者

注册表是特征定义的目录，每条含：名称、实体、来源、变换、窗口、默认值、TTL、负责人、版本、描述。两条安全规则：**两条路径都从注册表读定义**——任何服务不得私藏一份变换逻辑的副本；**改动定义必须开新版本**——旧版本保留，老模型继续用它训练时的那个特征。模型用 `spend_30d@1` 这样的引用钉住版本：带 `@` 但没写版本号会抛 `ValueError`，防止笔误悄悄读到最新版；不带 `@` 读最新版。引用同时是存储行的键，回填、在线存储、服务读都用被钉住的版本。

### 2.6 回填：给新特征造历史，以及它自带的泄漏

新特征没有历史，但训练需要它在每个旧标签时刻的值。回滚在**每个过去的日期**运行定义、每实体每日写一行，且每次只读事件时间 ≤ 该日期的事件。代价随“实体数 × 日期数”增长。还有一层隐蔽泄漏：今天跑的回填写的是旧事件时间，但它可能读到**晚到或后来被修正**的数据；严格的 point-in-time join 还要检查创建时间 ≤ 标签时间（原课练习3）。

### 2.7 偏差检测：三组对照定位成因

第 29 条规则：在服务时记录特征、用于训练。哪怕一小份记录样本也能做一致性检查：对每个（实体，时间），比较线上记录值与训练路径产出值，报告每个特征的“不一致占比”与“平均绝对差”。要定位成因，比三对：

1. **在线库读取 vs 训练 join**：考整条链路，应当零偏差；
2. **现算最新值 vs 训练 join**：隔离过期度；
3. **第二代码路径 vs 现算**：隔离代码差异（窗口、默认值）。

### 2.8 用一个小例子把概念串起来（补充）

把演示的数字串成故事。数据：299 个用户、25 个快照日（每 7 天，到第 175 天），回填 3 个特征共 7475 行（299×25=7475，可手算）。**缺陷一时间旅行**：在三个标签日（100/120/140）上分别用朴素 join 和 PIT join 造训练集、各拟合一条单特征规则，再在都使用 PIT 特征的第 145 天标签上测试。原课参考结果：朴素 join 离线 0.948（阈值：闲置>53天）、PIT join 离线 0.893（阈值：闲置>19天）；到第 145 天，朴素训练的规则只剩 0.763，PIT 训练的规则还有 0.876。离线更高的那个上线更差——泄漏把阈值从 19 天顶到了 53 天，把模型教坏了。**缺陷二两条路径**：第 141-150 天记录 383 个请求跑三组对照（原课参考）：在线库 vs 训练 join——三个特征不一致率全为 0%；现算 vs 训练 join——`days_since_last_purchase` 83% 不一致（过期度：模型学的是最多 6 天旧的值）；第二路径 vs 现算——窗口计数 12.5% 不一致（多算一天），新用户的 `days_since_last` 差 365 vs −1（均值差 20.07）。三组对照把“系统坏了”定位成“过期度差异 + 代码差异”两个可修的成因。

## 三、操作前准备

原课 `code/main.py` 只用 Python 标准库（`dataclasses`、`bisect`、`collections` 等）；Use It 的 Feast 示例需要 `feast`，本篇未安装：

~~~bash
python -m pip install feast
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

原课 Build It 共 6 步，只用标准库，时间是整数天数，每周批任务在第 7、14、…、175 天写快照。`Event`、`FeatureRow`、`LabelRow`、`JoinedRow`、`FeatureRegistry`、`group_events`、`naive_latest_join` 等定义原文页面未给出（完整实现在仓库 `code/main.py`），下文如实标注。

### 4.1 第一步：特征定义与注册表

`FeatureDefinition` 把变换存成**数据**（聚合方式、窗口、默认值），`compute` 绝不读 `as_of` 之后的事件——这是全课最重要的一行防线。

~~~python
# 本块中文说明：特征定义：聚合+窗口+默认值存成数据，compute 只看 as_of 之前。
@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    entity: str
    aggregate: str
    window_days: int | None
    default: float
    version: int = 1
    owner: str = "growth-team"
    description: str = ""

    def compute(self, events: list[Event], as_of: int) -> float:
        visible = [e for e in events if e.day <= as_of]
        if self.window_days is not None:
            visible = [e for e in visible if e.day > as_of - self.window_days]
        if self.aggregate == "count":
            return float(len(visible))
        if self.aggregate == "sum":
            return round(sum(e.amount for e in visible), 2)
        if self.aggregate == "days_since_last":
            return float(as_of - max(e.day for e in visible)) if visible else self.default
        raise ValueError(f"unknown aggregate {self.aggregate!r}")
~~~

（核对）窗口过滤是 `e.day > as_of - window_days`：30 天窗口覆盖 `as_of-29` 到 `as_of` 共 30 天，这正是演示里“第二路径多算一天”能造成 12.5% 不一致的机制。原文说明：`FeatureRegistry.register` 接受同一定义重复注册、但拒绝同名同版本的不同定义；`fill_defaults` 给两条路径同一个缺失默认值；模型用 `spend_30d@1` 钉版本，注册与解析逻辑原文未给出代码。

### 4.2 第二步：回填离线存储

按实体分组、按日重算，每实体每日产出一条 `FeatureRow`。

~~~python
# 本块中文说明：对每个实体、每个快照日运行注册表定义，产出离线历史行。
def backfill(registry: FeatureRegistry, names: list[str], events: list[Event], days: list[int]) -> list[FeatureRow]:
    rows = []
    for entity, entity_events in sorted(group_events(events).items()):
        for day in days:
            rows.append(FeatureRow(entity, day, registry.compute(names, entity_events, day)))
    return rows
~~~

演示为 299 个用户回填 3 个特征、25 个快照日，共 7475 行（299 × 25 = 7475，可手算核对）。

### 4.3 第三步：Point-in-time join

每个实体的行按天排序一次；对每个标签，`bisect_right` 找“不晚于标签日的最后一行”，再过 TTL 检查。

~~~python
# 本块中文说明：每个标签配“标签日或更早”的最新特征行，TTL 过期则视为缺失。
def point_in_time_join(labels: list[LabelRow], rows: list[FeatureRow], ttl_days: int | None = None) -> list[JoinedRow]:
    by_entity: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        by_entity[row.entity].append(row)
    days_index = {}
    for entity, entity_rows in by_entity.items():
        entity_rows.sort(key=lambda r: r.day)
        days_index[entity] = [r.day for r in entity_rows]
    joined = []
    for label in labels:
        days = days_index.get(label.entity, [])
        pos = bisect.bisect_right(days, label.day) - 1
        match = by_entity[label.entity][pos] if pos >= 0 else None
        if match is not None and ttl_days is not None and label.day - match.day > ttl_days:
            match = None
        joined.append(JoinedRow(
            label.entity, label.day, label.label,
            match.day if match else None, match.values if match else None,
        ))
    return joined
~~~

### 4.4 第四步：量化朴素 join 的泄漏

`naive_latest_join`（原文未给出代码）只按实体 join，每个标签拿到最新行。演示在第 100/120/140 天建标签、各拟合单特征规则，然后都在第 145 天的 PIT 特征标签上测试。运行 `python3 code/main.py` 的原课参考输出：

~~~text
  naive join used a snapshot from after the label day for 100% of rows
  offline accuracy, naive join: 0.948 (rule: idle > 53 days)
  offline accuracy, PIT join:   0.893 (rule: idle > 19 days)
  accuracy on day 145, trained with naive join: 0.763
  accuracy on day 145, trained with PIT join:   0.876
~~~

读法：朴素 join 离线更好看（0.948 vs 0.893），上新数据更差（0.763 vs 0.876）；泄漏把拟合阈值从 19 天顶到 53 天。

### 4.5 第五步：在线存储与物化

在线存储每实体只留一行、拒绝超过 TTL 的行；`materialize`（原文未给出代码）只写“日 ≤ 任务时刻”的行。

~~~python
# 本块中文说明：在线存储：每实体最新一行，读时做“未来行”与 TTL 双重检查。
class OnlineStore:
    def __init__(self, ttl_days: int):
        self.ttl_days = ttl_days
        self._latest: dict[str, FeatureRow] = {}

    def write(self, row: FeatureRow) -> None:
        current = self._latest.get(row.entity)
        if current is None or row.day >= current.day:
            self._latest[row.entity] = row

    def read(self, entity: str, now: int) -> dict[str, float] | None:
        row = self._latest.get(entity)
        if row is None or row.day > now or now - row.day > self.ttl_days:
            return None
        return dict(row.values)
~~~

原文的用法示例：TTL=10 天时，第 145 天的读取拿到第 140 天的行；第 160 天（无新物化）读到 None，走默认值——因为那行已经 20 天大。

### 4.6 第六步：偏差检测

`detect_skew` 按键 `(实体, 天)` 比对两份视图，报告每特征的不一致率与平均绝对差，超过 1% 打标；两视图键不同或没有键都直接抛错——缺请求会掩盖偏差，没有请求的检查什么都说明不了。

~~~python
# 本块中文说明：按（实体, 天）比对训练视图与服务视图，报告每特征的不一致率。
def detect_skew(
    training: dict[tuple[str, int], dict[str, float]],
    serving: dict[tuple[str, int], dict[str, float]],
    features: list[str],
    tolerance: float = 1e-6,
    max_mismatch_rate: float = 0.01,
) -> list[SkewReport]:
    if set(training) != set(serving):
        raise ValueError(f"{len(set(training) ^ set(serving))} keys are in only one view: log and compare the same requests")
    if not training:
        raise ValueError("no requests to compare")
    keys = sorted(training)
    reports = []
    for feature in features:
        diffs = [abs(training[k][feature] - serving[k][feature]) for k in keys]
        mismatches = sum(1 for d in diffs if d > tolerance)
        rate = mismatches / len(keys)
        mean_diff = sum(diffs) / len(keys)
        reports.append(SkewReport(feature, len(keys), rate, mean_diff, rate > max_mismatch_rate))
    return reports
~~~

演示在第 141-150 天记录请求（请求池含 20 个无历史新用户）并跑概念节的三组对照，原课参考输出：

~~~text
=== Skew checks on 383 logged requests, days 141 to 150 ===
  online store read  vs  training join (one definition, same snapshots)
    ok   days_since_last_purchase   mismatch   0.0%  mean |diff|    0.00
    ok   purchases_30d              mismatch   0.0%  mean |diff|    0.00
    ok   spend_30d                  mismatch   0.0%  mean |diff|    0.00
  fresh compute      vs  training join (staleness only)
    SKEW days_since_last_purchase   mismatch  83.0%  mean |diff|    2.77
    SKEW purchases_30d              mismatch  30.5%  mean |diff|    0.38
    SKEW spend_30d                  mismatch  37.1%  mean |diff|   16.77
  second code path   vs  fresh compute (code differences only)
    SKEW days_since_last_purchase   mismatch   5.5%  mean |diff|   20.07
    SKEW purchases_30d              mismatch  12.5%  mean |diff|    0.13
    SKEW spend_30d                  mismatch  12.5%  mean |diff|    5.09
~~~

三段读法：同一定义同一批快照，偏差为零；新鲜值与周快照在多数请求上不同——但**更新不等于更好**，模型本来就是在最多 6 天旧的值上训练的；第二路径在 12.5% 的窗口计数上不一致（多算一天），`days_since_last` 只在新用户上不同、但差得大（−1 对 365）。

## 五、把代码运行起来：补充入口与检查

原文页面未给出 `Event`/`FeatureRow`/`LabelRow`/`JoinedRow` 等数据类，以下补充最小定义，并用 0.3 的小表对 `point_in_time_join` 与 `detect_skew` 做可手算核对（补充，非原文代码；本次未运行回填或训练）：

~~~python
# 本块中文说明：补充最小数据类，用第16天标签核对 PIT join 与偏差检测（非原文代码）。
from dataclasses import dataclass, field
from collections import defaultdict
import bisect


@dataclass
class FeatureRow:
    entity: str
    day: int
    values: dict


@dataclass
class LabelRow:
    entity: str
    day: int
    label: int


@dataclass
class JoinedRow:
    entity: str
    day: int
    label: int
    feature_day: int | None
    values: dict | None


# 检查1：第16天标签应选中第14天的快照（可手算预期，见 0.3 的表）
rows = [
    FeatureRow("u1", 7, {"purchases_30d": 4.0}),
    FeatureRow("u1", 14, {"purchases_30d": 3.0}),
    FeatureRow("u1", 21, {"purchases_30d": 1.0}),
    FeatureRow("u1", 28, {"purchases_30d": 0.0}),
]
joined = point_in_time_join([LabelRow("u1", 16, 1)], rows)
assert joined[0].feature_day == 14 and joined[0].values == {"purchases_30d": 3.0}

# 检查2：TTL=2 天时，第14天行（距标签 2 天）保留
joined = point_in_time_join([LabelRow("u1", 16, 1)], rows, ttl_days=2)
assert joined[0].feature_day == 14

# 检查3：TTL=1 天时，第14天行（距标签 2 天 > 1）被丢弃
joined = point_in_time_join([LabelRow("u1", 16, 1)], rows, ttl_days=1)
assert joined[0].feature_day is None

# 检查4：detect_skew 小例（可手算预期：2 键中 1 个不一致 → 率 0.5、均值差 0.25）
training = {("u1", 1): {"f": 1.0}, ("u2", 2): {"f": 2.0}}
serving = {("u1", 1): {"f": 1.0}, ("u2", 2): {"f": 2.5}}
reports = detect_skew(training, serving, features=["f"])
assert len(reports) == 1
assert abs(reports[0].mismatch_rate - 0.5) < 1e-12
assert abs(reports[0].mean_abs_diff - 0.25) < 1e-12
assert reports[0].flagged is True
~~~

（补充声明）`SkewReport` 的字段名按 `detect_skew` 的构造顺序假定为 `(feature, n, mismatch_rate, mean_abs_diff, flagged)`——原文只给出调用处，未给出类定义；若与仓库版本不一致，以仓库为准。`FeatureDefinition.compute`、`backfill` 等依赖 `Event`/`FeatureRegistry` 的块本地未执行，只做语法检查。

## 六、Use It：使用现成工具

Feast 是开源特征仓库，从零代码的每个零件都有对应物：

| 从零实现 | Feast |
|---|---|
| `FeatureDefinition`、`FeatureRegistry` | 特征仓库库里的 `Entity`、`FeatureView`、`Field`，用 `feast apply` 注册 |
| `FeatureRow` 历史 | 离线存储，如 Parquet 上的 `FileSource` 或数仓表 |
| 带 TTL 的 `point_in_time_join` | `store.get_historical_features(entity_df=..., features=[...])` |
| `materialize` | `feast materialize-incremental <end time>` |
| `OnlineStore.read` | `store.get_online_features(features=[...], entity_rows=[...])` |
| 创建时间检查 | 数据源上的 `created_timestamp_column` 与 `filter_by_created_timestamp=True` |

定义写法（遵循 Feast 快速上手，本课运行不需要）：

~~~python
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

user = Entity(name="user", join_keys=["user_id"])

user_activity_source = FileSource(
    name="user_activity_source",
    path="data/user_activity.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

user_activity = FeatureView(
    name="user_activity",
    entities=[user],
    ttl=timedelta(days=10),
    schema=[
        Field(name="purchases_30d", dtype=Int64),
        Field(name="spend_30d", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
    ],
    source=user_activity_source,
)
~~~

训练与服务读同一个特征视图：

~~~python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=labels_df,
    features=["user_activity:purchases_30d", "user_activity:days_since_last_purchase"],
    filter_by_created_timestamp=True,
).to_df()

online = store.get_online_features(
    features=["user_activity:purchases_30d", "user_activity:days_since_last_purchase"],
    entity_rows=[{"user_id": 42}],
).to_dict()
~~~

`labels_df` 每行含 `user_id`、`event_timestamp` 与标签。Feast 文档明确：TTL 相对**实体表里每个时间戳**、不是相对查询时刻——与 `point_in_time_join` 的 TTL 检查一致；`filter_by_created_timestamp=True` 阻止回填值漏进训练数据，但不是每个离线存储都支持，不支持的会报错而不是静默忽略。Tecton、Hopsworks 和各大云平台的特征仓库都用同样的离线/在线两分。偏差检测可用 TensorFlow Data Validation 对比训练与服务数据；Breck 等人描述了 TFX 的数据验证系统（含训练服务偏差）。

## 七、失败模式、原文边界与工程注意

1. TTL 相对“每个标签的时间戳”而非“查询时刻”计算；把两者搞混会让 TTL 检查时松时紧（核对：Feast 文档与原课代码语义一致）。
2. `filter_by_created_timestamp=True` 依赖数据源真的有创建时间列，且不是所有离线存储都支持；不支持时 Feast 报错——静默降级比报错更危险。
3. 回填的隐蔽泄漏：今天跑的回填可能读到晚到或事后修正的数据；严格 join 还要查创建时间 ≤ 标签时间（原课练习3）。
4. 偏差检测的两个前置：两视图键必须完全一致、必须有请求可查；缺请求的检查不是“通过”，是“没查”。
5. “更新鲜的特征更好”不成立：模型在周快照上训练，线上给现算最新值本身就是过期度偏差；先让两条路径一致，再谈新鲜度。
6. 朴素 join 的离线分数**更高**（0.948 vs 0.893）——离线分数不能当正确性证据，上线后的 PIT 对照才能（原课参考数值）。
7. 一份注册表两读者是原则，但要防“绕过”：任何服务私藏一份变换实现都会重新引入偏差；版本化（`@1`）保证老模型不被定义改动波及。
8. `compute` 的窗口边界（`> as_of - window`）决定“30 天窗口含不含首日”；改窗口=改定义=开新版本，不能悄悄改。
9. 演示数值（7475 行、83%/12.5% 不一致率、0.948/0.893/0.763/0.876）均为原课参考，未在本机复现。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It（特征管道审计技能）的中文学习版，直接保存在本篇，不另建输出文件。

~~~markdown
特征管道审计：先画 join 的数据流，标出每个特征行的“事件时间”与“创建时间”；对每个标签验证只配到“标签时刻或更早”的最新行（point-in-time），抽查三个样本手算核对；确认 TTL 的基准是标签时间戳、过期值走哪条默认值，且默认值在训练与服务同源；列出每条特征路径——是注册表一份定义、还是服务里另写了一份（两条代码路径）；说明离线值与在线值的年龄差是多少（过期度）；检查回填是否只读“事件时间 ≤ 目标日期”的数据、是否查创建时间；确认线上请求记录了服务时刻的特征值、偏差检测的三组对照各定位什么；最后报告每个特征的差异率与均值差，以及“PIT 训练 vs 朴素训练”在保留集上的对照。所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **快照间隔**（原课）：把快照间隔从 7 天改成 1 天，重跑过期度对比并解释结果。验收：能预测不一致率方向并解释“更新鲜 ≠ 更好”何时反转。
2. **TTL 收紧**（原课）：TTL 设为 3 天，数有多少训练行丢特征、拿默认值。验收：给出丢行数与默认值填充的影响面。
3. **创建时间**（原课）：给 `FeatureRow` 加创建日，让 join 忽略标签日之后写入的行。验收：用一两个“今天回填、事件时间很旧”的行演示行为变化。
4. **版本化**（原课）：把 `spend_7d` 注册为 `spend_30d` 的第 2 版，回填并打印两个版本。验收：老引用仍读到 v1、新引用读到 v2。
5. **修第二路径**（原课）：修 `legacy_serving_features` 直到第三组对照通过，列出每处修改。验收：每处修改能对应到四种偏差成因之一。

## 十、术语与完成标准

特征仓库：一套定义喂训练与线上；离线存储：带事件时间戳的全历史；在线存储：每实体最新一行、毫秒级读；实体：特征描述的键；事件/创建时间戳：值变真的时刻 vs 系统写入的时刻；point-in-time join：标签配“不晚于标签时刻的最近行”；TTL：超龄即缺失；物化：离线最新值拷进在线库；回填：在过去的日期重算定义；训练服务偏差：两条路径算出不同的值。

完成检查：

- 能对着一张“标签 + 多行特征”的小表手算 point-in-time 与朴素 join 各选哪行。
- 能说出四种偏差成因并各配一个例子。
- 能解释为什么朴素 join 离线分更高、上线更差。
- 能说出偏差检测三组对照各隔离什么。
- 能解释 TTL 相对谁计算、过期值走什么。

## 十一、验证范围与来源许可

原课 Build It 共 6 个分段（定义与注册表、回填、PIT join、泄漏量化、在线存储、偏差检测），给出的代码均逐字转录；`Event`、`FeatureRow`、`LabelRow`、`JoinedRow`、`FeatureRegistry`、`group_events`、`naive_latest_join`、`materialize`、`fit` 类规则拟合等原文页面未给出代码，本篇如实标注、未臆造。本次验证做的是：正文 Python 代码块抽出后用 `python -c "import ast; ast.parse(...)"` 做静态语法检查（结果见下）；第五节补最小数据类后对 `point_in_time_join`/`detect_skew` 的断言标注“可手算预期”，未执行回填、训练或任何服务；演示输出与不一致率（83%、12.5% 等）均标注“原课参考”。语法检查全部通过（Windows 本机 Python）。

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
