# 13：ML流水线与实验追踪——Pipeline、泄漏防护与记录

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为"原课参考"的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/13-ml-pipelines) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/02-ml-fundamentals/13-ml-pipelines/docs/en.md)。原题：ML Pipelines。前置：[第12课超参数调优](<12：超参数调优——网格、随机与贝叶斯搜索.md>)、[第09课模型评估](<09：模型评估——指标、交叉验证与数据泄漏.md>)。

## 零、初学者先读：为什么"能跑的notebook"不等于"能交付的系统"

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。

### 0.1 先理解这一课在做什么

在notebook里写"填缺失→标准化→训练→打印准确率"像在厨房里做一道菜：做完就吃，不留痕迹。交付到生产是开餐厅：三个月后别人要能按你的配方做出**同一个味道**。流水线（pipeline）就是把所有加工步骤打包成一个带编号的整体对象——训练时它只学训练数据的"加工参数"，预测时原样复用，谁来运行结果都一样。本课两件事：防泄漏的流水线，和记录每次实验的追踪。

### 0.2 本课术语先翻成白话

- 流水线（pipeline）=变换步骤+模型串成的有序整体，一次fit、随处predict
- 变换器（transformer）=有 `fit`/`transform` 的数据加工件（缩放、编码、插补）
- 估计器（estimator）=有 `fit`/`predict` 的模型
- 数据泄漏（data leakage）=测试集或未来数据的信息混进训练，虚高分数
- 拟合（fit）=从训练数据统计学出加工参数（均值、方差、类别表）
- transform 与 fit_transform=用已学的参数变换 / 先学再变换
- ColumnTransformer=不同列走不同加工线（数值列缩放、类别列独热）
- 训练/服务偏斜（training-serving skew）=训练时和上线时数据处理不一致
- 实验追踪（experiment tracking）=每次训练记下参数、指标、数据版本、代码版本
- MLflow=开源的实验追踪+模型注册平台
- DVC=给大文件做版本控制：git里存哈希，远端存数据
- 序列化（serialization）=把训好的对象存成文件（joblib/pickle）
- 可复现性（reproducibility）=同代码+同数据+同配置→同结果

### 0.3 用小数字走一遍

训练集 `[1, 2, 3]`，测试集 `[4]`。只用训练集拟合：均值=2，总体标准差=√(2/3)≈0.816，测试点标准化为 $(4-2)/0.816\approx2.45$。先在全部数据 `[1,2,3,4]` 上拟合再划分（错误做法）：均值=2.5，标准差=√(5/4)≈1.118，同一个4变成 $(4-2.5)/1.118\approx1.34$。同一模型、同一样本，两种做法喂进去的数值不同——这就是泄漏如何悄悄改变结果（可手算预期）。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：`CustomTransformer`（手写缩放器：fit记均值方差、transform套公式）→ `PipelineFromScratch`（把步骤串起来：fit逐个fit_transform、predict逐个transform）→ 第五节的补充模型和泄漏对比。核心数据结构：`steps` 列表，元素是 `(名字, 对象)` 对，最后一个是模型、前面全是变换器。sklearn 的 `Pipeline`/`ColumnTransformer` 是同一思想的工业版。

### 0.5 理解检查（含参考答案）

**问题1：** `pipe.predict(X_test)` 时缩放器做了什么、没做什么？

**参考答案：** 只做 `transform`——用训练时学到的均值方差变换测试数据；**没有**重新 `fit`，测试数据的统计量从未参与学习。

**问题2：** 为什么"先对全数据 fit_transform 再划分"会让准确率虚高？

**参考答案：** 缩放参数里混入了测试样本的均值和方差，模型间接"见过"测试数据的分布信息，评估因此乐观。

**问题3：** 类别列在生产中出现训练时没见过的城市名，怎么不崩溃？

**参考答案：** `OneHotEncoder(handle_unknown="ignore")` 对未见类别输出全零向量而不是报错；没有这个设置，预测时直接抛异常。

理解后进入正文。

---

## 一、这一课解决什么问题？

原课的场景：notebook 能跑、准确率能打印，上线一个月后重训结果却变了——中位数是在含测试数据的全集上算的（泄漏）；缩放参数没保存，推理用了另一套统计量；特征工程代码在训练和 serving 两处复制粘贴后各自漂移；类别列冒出新值编码器直接崩。原课认为这些不是假设，而是ML系统在生产中最常见的失败原因（引用 Sculley et al. 2015 的技术债论文为背景）。流水线的解法：把每一步变换打包进一个有序、可复现、可整体序列化部署的对象。

## 二、核心概念与推导

### 2.1 流水线是什么

有序的变换序列加上末端的模型：`原始数据 → 插补 → 缩放 → 编码 → 模型 → 预测`。整条流水线在训练数据上 fit 一次；推理时同一个已拟合的对象变换新数据并预测。它保证四件事：变换只在训练数据上拟合（防泄漏）、推理应用同一套变换（防偏斜）、整体可序列化为一个制品、交叉验证时按折应用（防细微泄漏）。

### 2.2 数据泄漏：先缩放后划分 vs 先划分后缩放

原课给的错误写法（标注为反面示例，勿照抄到生产）：

~~~python
# 本块中文说明：反面示例——缩放器先在含测试数据的全集上拟合，属于泄漏。
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # 错：此时还没划分，scaler看到了测试数据

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]
~~~

正确写法：先划分，缩放器只 fit 训练部分：

~~~python
# 本块中文说明：正确示例——先划分，再只在训练集上拟合缩放器。
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 只学训练集的均值方差
X_test_scaled = scaler.transform(X_test)         # 测试集只被变换
~~~

用流水线后这层心智负担由框架承担：fit 时每步只接触训练折，predict 时只 transform。

### 2.3 sklearn Pipeline 的 fit/predict 语义

`pipe.fit(X_train, y_train)` 时：缩放器对 X_train 执行 `fit_transform`，模型在缩放后的数据上 `fit`。`pipe.predict(X_test)` 时：缩放器只执行 `transform`（不是 fit_transform），模型在变换结果上 `predict`。测试数据从头到尾没参与任何参数估计。

### 2.4 ColumnTransformer：不同列走不同加工线

真实数据里数值列要"插补+缩放"、类别列要"插补+独热"。`ColumnTransformer` 给不同列组挂不同的小流水线再拼接结果。`OneHotEncoder(handle_unknown="ignore")` 是生产关键：新类别出现时输出全零向量而不是崩溃。

### 2.5 实验追踪与模型/数据版本

流水线让**一次**训练可复现，实验追踪让**所有**训练可比对：MLflow 记录每次运行的参数、指标、制品和完整模型；wandb 提供同类功能的托管面板。模型注册（Model Registry）给每个保存的模型版本号和阶段标签（Staging/Production/Archived），支持审批与回滚。数据版本（DVC）：git 存不了大文件，DVC 把数据放远端（S3等），git 里只留记录哈希的 `.dvc` 小文件；`dvc checkout` 随 git 提交恢复当时的数据——代码和数据被同一个提交钉住。

### 2.6 可复现实验的四要素

固定随机种子（numpy、random、框架）、锁定依赖版本（requirements/poetry.lock）、数据有版本（DVC）、超参数进配置文件而非散落硬编码。原课的 `set_seed` 工具函数见第五节。

### 2.7 从notebook到生产的演进与常见错误

典型路径：notebook探索 → 抽函数进模块 → 组成 Pipeline 对象 → 超参数进YAML/JSON → 接实验追踪 → 加数据校验 → 加测试 → 序列化部署（API+容器）。原课的常见错误表（中文转述）：

| 错误 | 危害 | 修法 |
|---|---|---|
| 划分前拟合 | 数据泄漏 | 用 Pipeline 配 cross_val_score |
| 特征工程在流水线外 | 训练/服务不一致 | 全部变换进 Pipeline |
| 不处理未知类别 | 生产崩溃 | OneHotEncoder(handle_unknown="ignore") |
| 硬编码列名 | 换schema即坏 | 列名来自配置 |
| 无数据校验 | 坏数据静默出错 | 预测前加schema检查 |
| 训练/服务偏斜 | 线上特征不同 | 训练服务共用一个Pipeline对象 |

### 2.8 用一个小例子把概念串起来（补充）

把 0.3 的数字放进流水线视角（补充）：`PipelineFromScratch([("scale", CustomTransformer()), ("model", 某模型)])`。fit 在训练折上记下均值2、方差0.816²；predict 对测试点4输出2.45。如果你绕过流水线先在全集上 fit 再喂给流水线 predict，同一个4会变成1.34——数值差了一倍。整条链路上"谁在什么时候看到了哪些数据"就是流水线要管死的全部事情。

## 三、操作前准备

从零实现部分只需 NumPy；sklearn 部分（Pipeline、ColumnTransformer、SimpleImputer、OneHotEncoder）需要 scikit-learn；实验追踪需要 mlflow 或 wandb；序列化用 joblib（sklearn 自带依赖）；DVC 需单独安装并初始化 git 仓库。

~~~bash
python -m pip install numpy scikit-learn mlflow joblib
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

原课 Build It 共 4 段：第1、2段有代码（逐字转录如下）；第3段（用流水线做交叉验证）与第4段（sklearn 完整生产流水线）原文只描述了 `code/pipeline.py` 的内容，文档中没有代码——本文在第五节补充等价演示。

### 4.1 自定义变换器

最小变换器三件套：`fit` 学参数、`transform` 用参数、`fit_transform` 先学后用。标准差为0的列（常数列）除法保护为1。

~~~python
# 本块中文说明：手写标准化变换器——fit记均值方差，transform套公式。
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)     # 逐列均值
        self.stds = np.std(X, axis=0)       # 逐列标准差
        self.stds[self.stds == 0] = 1.0     # 常数列防除零
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
~~~

### 4.2 从零实现的流水线

`steps` 的最后一个元素是模型（只 `fit`/`predict`），前面都是变换器（`fit_transform`/`transform`）。fit 逐级变换训练数据直到交给模型；predict 用同样的变换链处理新数据。

~~~python
# 本块中文说明：从零流水线——变换器逐级fit_transform，末端模型只fit/predict。
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:          # 除最后一步外的变换器
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]                # 最后一步：模型
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:          # 预测时只transform
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
~~~

### 4.3 用流水线做交叉验证（原文仅描述）

原课说仓库代码演示"交叉验证配合流水线如何防泄漏：缩放器在每个折的训练部分单独拟合"。文档无代码，第五节补充手写循环版。

### 4.4 sklearn 完整生产流水线（原文仅描述）

原课说 `code/pipeline.py` 构建了带 `ColumnTransformer`、多条预处理路径、模型、交叉验证与实验日志的完整流水线。文档无代码；第六节给出等价的 sklearn 写法。

## 五、把代码运行起来：补充入口与检查

（补充）`PipelineFromScratch` 需要一个末端模型才能跑，原文未给；补一个最小最近质心分类器，接口对齐 `fit(X, y)`/`predict(X)`：

~~~python
# 本块中文说明：补充的最小模型——最近质心分类器，供从零流水线调用。
class NearestCentroidScratch:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # 每类的质心 = 该类样本的逐列均值
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        # 每个样本到每个质心的欧氏距离，取最近的类别
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]
~~~

（补充）最小入口与两项检查——"流水线预测=手动变换后预测"的一致性，以及手写二折交叉验证演示"缩放器每折单独拟合"：

~~~python
# 本块中文说明：补充入口——一致性断言 + 手写两折CV演示防泄漏。
rng = np.random.RandomState(0)
X = rng.randn(120, 3)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

pipe = PipelineFromScratch([
    ("scale", CustomTransformer()),
    ("model", NearestCentroidScratch()),
])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

# 检查1：流水线预测 == 手动"变换再预测"（结构一致性，必然相等）
scaler = CustomTransformer().fit(X_train)
manual = NearestCentroidScratch().fit(scaler.transform(X_train), y_train)
assert np.array_equal(preds, manual.predict(scaler.transform(X_test)))

# 检查2：两折CV中每折各自拟合缩放器（防泄漏的写法）
fold_preds = []
half = len(X_train) // 2
for a, b in [(slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))]:
    p = PipelineFromScratch([("scale", CustomTransformer()),
                             ("model", NearestCentroidScratch())])
    p.fit(X_train[a], y_train[a])
    fold_preds.append((p.predict(X_train[b]), y_train[b]))
assert len(fold_preds) == 2
~~~

（补充）原课概念节给出的 `set_seed` 工具（torch 未安装时静默跳过）：

~~~python
# 本块中文说明：原课的统一随机种子设置工具，逐字转录。
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
~~~

可手算预期：0.3 节的例子中，`CustomTransformer().fit([[1],[2],[3]])` 后 `transform([[4]])` 得 ≈2.449。

## 六、Use It：使用现成工具

### 6.1 sklearn Pipeline 与 ColumnTransformer

~~~python
# 本块中文说明：sklearn 版分列预处理 + 整体流水线（原课概念节代码整理）。
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # 数值列：中位数插补
    ("scale", StandardScaler()),                    # 再标准化
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),  # 类别列：众数插补
    ("encode", OneHotEncoder(handle_unknown="ignore")),  # 未见类别输出全零
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
# full_pipeline.fit(X_train, y_train) 后即可 predict/score/序列化
~~~

### 6.2 实验追踪：MLflow 与 wandb

~~~python
# 本块中文说明：MLflow 记录参数、指标并保存整个流水线模型。
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")        # 整个流水线作为一个制品
~~~

~~~python
# 本块中文说明：wandb 托管面板式追踪，功能同类。
import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})
~~~

### 6.3 DVC 数据版本

~~~bash
# 本块中文说明：DVC 把数据放远端，git 里只留哈希文件（原课命令）。
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training data"
dvc push
~~~

序列化用 `joblib.dump(pipe, "pipe.joblib")`；在独立脚本里 `joblib.load` 后预测，同一输入应得到同一输出（原课练习3的验收标准）。

## 七、失败模式、原文边界与工程注意

1. 划分前拟合任何统计量（缩放、插补、编码、目标编码）都是泄漏；放进 Pipeline 后由交叉验证按折处理。
2. 特征工程留在流水线外，训练与 serving 两份代码必然漂移（训练/服务偏斜）；解法是一个对象两处共用。
3. `handle_unknown="ignore"` 只是把未知类别变成全零，不等于模型理解了新类别；类别激增时仍需重训。
4. 硬编码列名在 schema 变化时静默错位；列名应来自配置，并加 schema 校验。
5. `CustomTransformer` 只接受数值矩阵（`np.mean` 对非数值列直接报错）；类别列必须走别的加工线（核对，可从代码直接验证）。
6. `PipelineFromScratch` 没有实现 sklearn 的 clone 协议（`get_params`/`set_params`），因此**不能**直接传给 `cross_val_score`；第五节的 CV 是手写循环（核对，可从代码与 sklearn 接口要求验证）。
7. `set_seed` 里 `cudnn.deterministic = True` 只提高复现概率；多线程、GPU 非确定性运算仍可能造成微小差异（原课自己说"四要素"齐了才谈得上复现）。
8. 原课"这些是ML系统生产中最常见的失败原因"以工程经验与 Sculley 论文为据，属经验性论断（原课参考），不是统计意义上的排位。
9. MLflow、wandb、DVC 涉及外部服务与远端存储，本文均未运行；相关命令仅为转录。

## 八、Ship It：本课可复用的中文检查模板

~~~markdown
流水线检查：列出每个变换步骤及其拟合时可见的数据范围；确认所有统计量只在训练折上学习；类别列声明 handle_unknown 策略；列名来自配置而非硬编码；预测入口与训练共用同一个已拟合对象；序列化制品经过"另脚本加载→同输入同输出"验证；记录实验的参数、指标、数据版本与代码版本；随机种子、依赖、数据、配置四项可复现要素逐项打勾。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 3数值列+2类别列的数据集：数值走"中位数插补+缩放"、类别走"众数插补+独热"，5折CV训练。验收：ColumnTransformer 拼接后的特征维度可解释（独热维度=训练集类别数）。
2. 故意制造泄漏（划分前对全数据拟合缩放器），对比泄漏版与干净版的CV分数。验收：如实报告两者差值——差值可能很大也可能很小，取决于数据，不预设结论。
3. `joblib.dump` 序列化你的流水线，另起脚本加载后预测。验收：两次预测逐位一致。
4. 加一个"对两个最重要数值列生成二阶多项式特征"的自定义变换器，并回答它该插在流水线哪个位置。验收：位置有理由（插补/缩放之后、模型之前），且变换器只依赖训练折统计。
5. 给流水线接 MLflow，跑5组不同超参数的实验。验收：`mlflow ui` 能并排比较5次run，并说明选择依据。

## 十、术语与完成标准

Pipeline：变换+模型的有序整体，防止泄漏与偏斜；数据泄漏：训练外信息进入训练，虚高评估；ColumnTransformer：按列分组应用不同预处理并拼接；实验追踪：记录每次运行的参数/指标/制品/版本；MLflow：开源实验追踪与模型注册平台；DVC：大文件版本控制，git存哈希、远端存数据；模型注册：带阶段标签的模型版本目录；训练/服务偏斜：训练与推理数据处理不一致；可复现性：同代码同数据同配置得同结果。

完成检查：

- 能手算 0.3 节两个标准化数值并解释差异来源。
- 能说出 `pipe.predict` 时每一步用的是 transform 还是 fit_transform。
- 能把一次泄漏事故按"谁在何时看到了什么数据"复述出来。
- 能写出数值列与类别列各走的加工线并解释 `handle_unknown="ignore"`。
- 能列出可复现实验的四要素并各自给一个落地手段。

## 十一、验证范围与来源许可

原课 Build It 共 4 个分段：第1、2段代码逐字转录并加中文注释（逻辑未改）；第3、4段原文仅文字描述、文档无代码，第五节以明确标注"补充"的等价演示代替（最近质心分类器、一致性断言、手写两折CV均为补充，原课 `set_seed` 为逐字转录）。概念节的泄漏/正确两段示例代码与 sklearn/MLflow/wandb/DVC 示例保留（6.1 合并自原课分散的两处示例，仅补全导入，未改逻辑）。本次验证：把正文 python 代码块抽出存临时 .py 文件，逐块做了 ast.parse 静态语法检查（2026-11-04，Python 3.12，临时脚本逐块解析），全部通过；未运行任何训练、追踪或外部服务，0.3 节数值为可手算预期。

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
