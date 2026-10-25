

整理日期：2026-10-01。对应 Phase 00 / 09 Data Management，原课预计约 45 分钟。

按原课八个步骤整理，保留主要数据集与 API；【设备适配】说明你的操作方式，【纠正／补充】指出原文简化或容易误用的地方。以下命令是学习指引，本次没有替你安装软件或下载数据，也没有把示例标为本机已运行通过。

来源：[课程页面](https://aiengineeringfromscratch.com/lesson?path=phases%2F00-setup-and-tooling%2F09-data-management)、[课程英文正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/09-data-management/docs/en.md)、[配套 data_utils.py](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/09-data-management/code/data_utils.py)。正文依据同项目公开源码核对；课程采用 MIT 许可证，许可声明见文末。

## 0. 你的环境与本课约定

此前确认：Windows 11＋WSL 2 Ubuntu 24.04，用户 pei，约 16 GB 内存，RTX 4060 Laptop GPU 约 8 GB 显存。你实际安装的是 WSL 内的 Docker Engine，不是 Docker Desktop。

**本课的数据加载、转换、划分都可以直接在 WSL Python 环境中完成，不需要 GPU，也不要求启动 Docker。** Windows、Ubuntu 和 Docker 容器里的 Python 包与默认缓存目录并不会自动共用。

2026-09-30 检查时 C 盘约剩 20 GiB、D 盘约剩 30 GiB，这是历史记录，不是当前剩余空间。先使用小数据和流式读取；不要下载整套 Wikipedia 或 C4。

约定：bash 代码在 Ubuntu 终端执行；python 代码保存成 .py 文件后用 `python 文件名.py` 执行，或在已选中同一虚拟环境的 Notebook 中运行。每段 Python 示例都包含所需导入，除明确说明外可独立运行。

### 文件放哪里？

为了不依赖尚未确认的课程仓库位置，建立独立练习目录：

```bash
mkdir -p ~/learning/data-management
cd ~/learning/data-management
mkdir -p data outputs
```

下文用 `~` 表示当前 Ubuntu 用户的家目录。工作区示例是：

```text
~/learning/data-management/
├── .venv/        Python 环境
├── data/         自己导出的数据与划分
├── outputs/      记录、结果
└── *.py          练习脚本

~/.cache/huggingface/   Hugging Face 默认缓存
```

在 Linux 工具链中，代码和频繁读取的数据通常优先放 WSL 的 Linux 文件系统；需要与 Windows 交换文件时可用 `/mnt/c`、`/mnt/d`。Windows 资源管理器可以通过 `\\wsl.localhost\Ubuntu-24.04\home\<用户名>` 访问家目录。[Microsoft 文件系统说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

WSL 中 `df -h` 显示的虚拟磁盘可用容量，不等于 Windows 物理磁盘真的还有那么多空间；还要查看承载 WSL 虚拟磁盘的 Windows 分区。

## 1. 原课要解决的问题与学习目标

数据管理不只是“下载一个文件”，而是建立可重复的过程：

```text
找到数据及版本 → 下载或流式读取 → 检查结构
→ 选择格式 → 划分数据 → 保存与记录 → 交给训练程序
```

原课的四个目标：

1. 用 Hugging Face datasets 加载、流式读取与缓存数据。
2. 理解 CSV、JSON、Parquet、Arrow 的区别。
3. 用固定随机种子建立可复现的数据划分。
4. 用 .gitignore、Git LFS 或 DVC 管理大文件。

两个库各有分工：`datasets` 处理数据集及其变换；`huggingface_hub` 处理 Hub 仓库中的文件下载与缓存。数据集仓库、模型仓库都可能有多个文件，不要把“仓库”“一个文件”和“一个 split”混为一谈。

## 2. Step 1：安装 datasets

【原课】安装 `datasets huggingface_hub`。

【设备适配】在刚才的练习目录创建虚拟环境，避免把包写入 Ubuntu 系统 Python：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install datasets huggingface_hub
```

如果创建环境时提示缺少 venv 支持，再执行 `sudo apt-get install python3-venv`，然后重试。若已有专门用于课程的 Python 虚拟环境，可以复用，不必重复创建。不要用 `sudo pip install` 绕过系统环境限制。

检查当前解释器和依赖：

```bash
python -c "import sys, datasets, huggingface_hub; print(sys.executable); print(datasets.__version__); print(huggingface_hub.__version__)"
```

解释器路径应属于你选用的 Ubuntu 虚拟环境，而不是 Windows 的 D 盘 Python。新开终端后需重新进入目录并 `source .venv/bin/activate`。

### 网络：这次看 Python 所在终端的代理

此前你的终端代理为 `http://127.0.0.1:7897`，Docker 服务后来单独配置了同一代理。Docker 下载成功，不等于当前 Python 一定能访问 Hugging Face。

先检查终端：

```bash
env | grep -iE '^(http|https|all|no)_proxy='
curl -I --connect-timeout 10 https://huggingface.co
```

若仍使用此前的代理软件和端口，但新终端没有变量，可以在当前终端设置：

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
```

该地址以你当前 WSL 网络环境仍可访问它为前提，不能推广为所有 WSL 电脑的固定地址。主页能访问也不保证大文件下载端点都通，发生错误时保留具体报错。

## 3. Step 2：加载 IMDB 数据集

保留原课主例，保存为 `01_load.py`：

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
print(dataset["train"].features)
print(dataset["train"][0])
```

运行：

```bash
python 01_load.py
```

读法：`dataset` 通常是包含多个 split 的 DatasetDict；`dataset["train"]` 是一个 Dataset；再写 `[0]` 得到第一条样本。影评例子主要包含 text 和 label，标签的实际含义可通过 features 查看。

只想操作训练集时：

```python
from datasets import load_dataset

train = load_dataset("stanfordnlp/imdb", split="train")
print(len(train))
print(train.column_names)
print(train[0]["text"][:200])
```

区别：`train[0]` 返回一条记录；`train[:5]` 通常返回按列组织的字典；逐行查看可用 `for row in train.select(range(5))`。

首次加载可能经历下载、处理和缓存；再次使用通常复用缓存。普通 Dataset 可利用磁盘上的 Arrow 数据，不等于把全部内容都复制进 Python 内存。不要把大型数据集转换成 `list(dataset)` 来查看。

【补充】`split="train[:100]"` 或加载后 select 100 条，并不保证网络只下载 100 条；底层可能需要先准备完整文件或分片。只预览远程大数据时用下一步的 streaming。

### 默认缓存为什么有两个目录？

| 默认目录 | 主要用途 |
|---|---|
| `~/.cache/huggingface/hub` | Hub 获取的模型文件及数据集源文件等 |
| `~/.cache/huggingface/datasets` | datasets 准备的 Arrow 数据、变换缓存等 |

`HF_HOME` 可调整整体根目录；`HF_HUB_CACHE` 与 `HF_DATASETS_CACHE` 分别调整对应缓存。应在启动 Python 前设置。默认配置足够完成本课；仅修改路径不会自动搬迁已下载内容。[HF 缓存说明](https://huggingface.co/docs/datasets/cache)

查看当前缓存占用：

```bash
du -sh ~/.cache/huggingface
```

缓存、导出文件和划分副本可能同时存在，下载量不等于最终磁盘占用。

## 4. Step 3：流式读取大数据集

原课用英文 Wikipedia，仅查看前五个标题：

```python
from datasets import load_dataset

stream = load_dataset(
    "wikimedia/wikipedia", "20231101.en",
    split="train", streaming=True,
)

for row in stream.take(5):
    print(row["title"])
```

这里把原课的 enumerate＋break 写成 take(5)，读取目标相同。返回的是 IterableDataset，重点是顺序迭代；它不像普通 Dataset 那样适合按任意行号随机访问，通常不能直接用 len 获取总行数。

【纠正】流式读取不是“不走网络”，而是不先把整个数据集下载准备完。底层仍会按块读取、解压和缓冲。原课说内存恒定，应理解为不必随总数据量等比例增长；样本大小、批量和 shuffle 缓冲区都会改变内存占用。[HF Streaming 文档](https://huggingface.co/docs/datasets/stream)

对你约 16 GB 内存的电脑：先取五条，不累计到大列表，不加大量并行 worker。不要对 stream 使用 `list(stream)`，那会继续读取直到耗尽。

## 5. Step 4：CSV、JSON、Parquet 与 Arrow

原课导出完整 IMDB 训练集。首次学习改成同一批 500 条，减少重复占用；保存为 `02_formats.py`：

```python
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb", split="train")
sample = ds.select(range(min(500, len(ds))))
out = Path("data")
out.mkdir(exist_ok=True)

sample.to_csv(str(out / "imdb_sample.csv"))
sample.to_json(str(out / "imdb_sample.jsonl"))
sample.to_parquet(str(out / "imdb_sample.parquet"))

for suffix in ("csv", "jsonl", "parquet"):
    path = out / f"imdb_sample.{suffix}"
    print(path.name, path.stat().st_size, "bytes")

restored = load_dataset(
    "parquet", data_files=str(out / "imdb_sample.parquet"), split="train"
)
assert len(restored) == len(sample)
assert restored[0] == sample[0]
print("行数与第一条内容检查通过")
```

这里的检查只是基本往返验证，并非全文件完整性校验。

| 格式 | 适合什么 | 需要注意什么 |
|---|---|---|
| CSV | 表格交换、人工查看 | 类型、嵌套结构表达较弱；长文本含换行时不便直接查看 |
| JSON／JSONL | API、嵌套记录、逐行处理 | 字段名重复、文本开销通常较大 |
| Parquet | 列式分析、压缩存储、选择部分列读取 | 需要工具读取，不能按普通文本打开 |
| Arrow | 高效列式处理、内存映射、库间交换 | 是数据表示与生态，不能直接等同于“压缩最小” |

【纠正】原课文件扩展名写 .json，但 Dataset.to_json 默认通常输出逐行记录，因此这里明确使用 .jsonl。不要假定文件一定是一个包着所有记录的 JSON 数组。

原课的“小／大、快／慢”是经验性比较，不是保证。Parquet 的压缩、行组、数据内容都会影响大小，Arrow 也不保证在所有操作中最快。比较应使用相同数据，并区分读取、解析和处理耗时。

原课的实用取向是：CSV／JSON 方便交换，Parquet 适合许多数据存储场景，datasets 内部大量使用 Arrow。

## 6. Step 5：划分 train／validation／test

| 集合 | 用途 |
|---|---|
| train | 用来学习模型参数 |
| validation | 选择超参数、比较方案、选择检查点 |
| test | 在方案确定后做最终评估 |

不要通过反复查看 test 结果来挑选模型，否则测试集也参与了决策。

### 原文有一处比例不一致

原文先取 20% 做 test，再从剩余 80% 中取 12.5% 做 validation：

```text
test  = 20%
val   = 80% × 12.5% = 10%
train = 80% × 87.5% = 70%
```

所以原文代码实际是 **70/10/20**，并非前面描述的典型 80/10/10。配套 data_utils.py 则采用另一套通用划分逻辑，默认目标为 80/10/10，不要把正文与脚本混为一谈。

### 正确实现 80/10/10

```python
from datasets import DatasetDict, load_dataset

ds = load_dataset("stanfordnlp/imdb", split="train")
first = ds.train_test_split(test_size=0.2, seed=42)
second = first["test"].train_test_split(test_size=0.5, seed=42)

parts = DatasetDict({
    "train": first["train"],
    "validation": second["train"],
    "test": second["test"],
})
for name, subset in parts.items():
    print(name, len(subset))
assert sum(len(part) for part in parts.values()) == len(ds)
parts.save_to_disk("data/imdb_splits_80_10_10")
```

思路：先留出 20%，再把这 20% 平分。如果输入 25,000 行，预期是 20,000／2,500／2,500。

这是对官方 train 做“划分方法练习”。正式使用 IMDB 基准时，应保留官方 test，通常仅从官方 train 留出 validation；不要把练习里新划出的 test 当作官方测试集。

恢复保存的数据：

```python
from datasets import load_from_disk

parts = load_from_disk("data/imdb_splits_80_10_10")
print(parts)
```

### seed 的作用与边界

seed 固定随机过程，但数据内容、输入顺序、算法与版本也要一致。不能认为“只要 seed 相同，换任何数据都得到相同划分”。实际实验应记录仓库 revision、预处理、包版本和样本分配。

【补充】随机划分也不总适用：时间序列要考虑时间先后；同一用户、病人或文档的近重复片段不能随意跨集合；类别失衡时可考虑分层划分。先决定合理的划分单位，再设置 seed。

## 7. Step 6：下载与缓存模型文件

本课目的是理解下载管理，不要求把模型加载到显卡。

先按原课下载一个很小的配置文件：

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json",
)
print(path)
```

返回的是本地缓存路径。config.json 描述模型配置，不包含完整权重，下载成功不等于模型全部就绪。

原课下一步是 `snapshot_download("sentence-transformers/all-MiniLM-L6-v2")`，会获取整个仓库快照。仓库可能含多种框架／导出格式，同一模型不代表只有一份权重。

【设备适配】先练习过滤下载：

```python
from huggingface_hub import snapshot_download

folder = snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    allow_patterns=["*.json"],
)
print(folder)
```

这只取得匹配的 JSON 文件，**不能因此拿它当完整模型运行**。确实需要整仓时再去掉过滤。精确复现可用 revision 指定实际提交；缓存文件不要直接修改。[HF 下载指南](https://huggingface.co/docs/huggingface_hub/guides/download)

【纠正】原文称缓存后“立即加载”，实际仍有磁盘读取、解析、可能的远程版本检查；加载到 CPU／GPU 也要时间。缓存避免重复传输，并不消除所有耗时。

## 8. Step 7：不要把大文件直接塞进普通 Git

### A. .gitignore：本课优先掌握

在自己的练习项目中，将规则合并进现有 .gitignore，不要覆盖已有规则：

```gitignore
.venv/
__pycache__/
models/
*.bin
*.safetensors
*.pt
*.onnx
data/**/*.parquet
data/**/*.csv
data/**/*.jsonl
data/imdb_splits_*/
```

与原课同一原则：提交代码、环境记录和数据来源说明，忽略可重新下载或生成的大文件。这里补了虚拟环境、JSONL 和划分目录。

【补充】.gitignore 不会删除文件，也不会自动让已经被 Git 跟踪的文件停止跟踪。它不等于备份或版本管理；下载来源与版本仍要记录。

### B. Git LFS：用指针跟踪大文件

原课命令：

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

这些应在需要 LFS 的 Git 仓库内执行，并以前提“已安装 git-lfs”为准。真正数据存于 LFS 存储，Git 中保留指针；仅运行 track 不会自动上传文件。

如果选择 LFS 跟踪某类文件，应调整冲突的 .gitignore 规则，不能一边忽略它一边期待普通 git add 自动加入。原课写的免费额度不是固定规则，本笔记不沿用该额度数字；实际使用时查看所选服务当前政策。

### C. DVC：给数据建立版本记录

原课展示的基本过程是安装 dvc、在 Git 仓库中 dvc init，再执行：

```bash
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
```

上面的 parquet 是示例路径，必须先存在。DVC 会建立小型元数据文件并使用缓存；初始化时产生的 .dvc 配置等也应按提示纳入 Git。

【纠正】执行 dvc add 不代表数据已上传云端。配置 remote 后再 dvc push 才进行远端同步；也能选择本地存储作为 remote。数据版本元信息与数据内容分开管理。[DVC 文档](https://doc.dvc.org/start)

这三个方案是按需要选择，不要求同时启用。本课用 .gitignore＋明确的数据来源记录即可起步。

## 9. Step 8：本地与云存储

原课建议学习阶段本地存储，远程 GPU 或团队协作时再考虑 S3／GCS 等云端方案。

【纠正】原课提到约 10 GB 的分界只是示意，不是硬性界限。是否用云端取决于本机容量、网络、共享需要、预算；大于 10 GB 不一定必须上云，小于 10 GB 也可能需要共享。

对你这台电脑，本课采用：小数据缓存到 WSL；大语料流式预览；导出先用 500 条；模型先下载配置。不要为了学习缓存而下载大量权重。

如果未来将缓存改为 `/mnt/d/...`，要先确认 D 盘实际空间，权衡跨文件系统 I/O。不要因为 WSL 报告虚拟盘还有数百 GB，就认为宿主盘容量充足。

## 10. 原课配套脚本与数据集清单

原课列出的 IMDB、WikiText、SQuAD、Common Crawl 子集、MNIST、COCO 子集分别对应分类、语言建模、问答、网页文本、视觉与多模态。现在不必全部下载，后续课程按需使用。原课尺寸随版本、压缩与子集范围变化，只能作为示意。

若已有课程仓库，在其根目录运行：

```bash
python phases/00-setup-and-tooling/09-data-management/code/data_utils.py
```

原文的 `python code/data_utils.py` 则要求当前目录是本课目录。不要在独立练习目录直接照抄而期待那里已有脚本。

当前源码实际使用 rotten_tomatoes 小数据集，导出 500 条到 `/tmp/data_utils_demo`，划分、回读，再下载 MiniLM 的 config.json。`/tmp` 不适合作为永久成果目录，需要长期保存可在自己的可写副本中改到 data/。

脚本另外有两点限制：缓存统计路径写死为默认 datasets 目录，改了 HF_HOME 后不一定统计到正确位置，也不包括所有 Hub 文件；其 fingerprint 仅基于前 100 条样本，不能当作整个数据集完整性的证明。[配套脚本源码](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/09-data-management/code/data_utils.py)

## 11. 原课四个练习：可直接跟做

### 练习一：GLUE／MRPC 前五条

原课写 glue 与 mrpc；此处使用带组织名的仓库 ID。[GLUE 数据集](https://huggingface.co/datasets/nyu-mll/glue)

```python
from datasets import load_dataset

ds = load_dataset("nyu-mll/glue", "mrpc", split="train")
print(ds.features)
for i, row in enumerate(ds.select(range(5)), start=1):
    print("样本", i)
    print("句子1:", row["sentence1"])
    print("句子2:", row["sentence2"])
    print("标签:", row["label"])
```

MRPC 是句子对释义判断任务。要能解释 repo ID、配置 mrpc、split train 分别选择了什么，并查看标签定义，而不仅是看到代码运行结束。

### 练习二：流式读取 C4，统计约 10 秒处理量

使用官方仓库标识 allenai/c4 和英文配置 en；不要去掉 streaming=True。[C4 数据集](https://huggingface.co/datasets/allenai/c4)

```python
from time import perf_counter
from datasets import load_dataset

setup_start = perf_counter()
stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
iterator = iter(stream)
print("创建数据流耗时:", perf_counter() - setup_start)

start = perf_counter()
count = 0
chars = 0
while perf_counter() - start < 10:
    try:
        row = next(iterator)
    except StopIteration:
        break
    count += 1
    chars += len(row["text"])

elapsed = perf_counter() - start
print(f"读取 {count} 条，{chars} 字符；实际 {elapsed:.2f} 秒")
print(f"平均 {count / elapsed:.2f} 条/秒")
```

这是同步读取的近似 10 秒统计：如果 next 正在等待网络，可能超过 10 秒才返回，计入最后一条；不是严格实时截止。记录实际时间而不是硬写 10 秒。速度受网络、代理、缓存、解压和文本长度影响，不是 GPU 性能测试。

### 练习三：比较 CSV 与 Parquet

运行 Step 4 的 `02_formats.py`，记录同一批 500 条样本的实际字节数，再计算 `CSV字节数 / Parquet字节数`。同时验证 Parquet 可以读回。不要预设一定缩小几倍，文件小的时候元数据开销也会影响结果。

### 练习四：70/15/15，验证可复现和互不重叠

用一个明确的 1,000 行练习子集，让比例容易核对。前 1,000 行只是教学样本，不声称具有总体代表性。

```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb", split="train").select(range(1000))
ds = ds.add_column("row_id", list(range(len(ds))))

def split_data(data):
    first = data.train_test_split(test_size=0.3, seed=42)
    second = first["test"].train_test_split(test_size=0.5, seed=42)
    return {
        "train": first["train"],
        "validation": second["train"],
        "test": second["test"],
    }

a = split_data(ds)
b = split_data(ds)
print({name: len(part) for name, part in a.items()})

assert [len(a[k]) for k in ("train", "validation", "test")] == [700, 150, 150]
for name in a:
    assert list(a[name]["row_id"]) == list(b[name]["row_id"])

train_ids = set(a["train"]["row_id"])
val_ids = set(a["validation"]["row_id"])
test_ids = set(a["test"]["row_id"])
assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)
assert train_ids | val_ids | test_ids == set(range(len(ds)))
print("比例、重复划分一致性、行 ID 互斥和覆盖检查通过")
```

行 ID 不重叠只是基础检查，不能排除原始数据中不同 ID 对应相同或近重复文本。一般数据量不能恰好整除比例时，还会有取整差异。

## 12. 复习与排错

| 现象 | 优先检查 |
|---|---|
| externally-managed-environment | 是否激活了虚拟环境 |
| ModuleNotFoundError | 安装包和运行脚本是否使用同一 Python |
| 下载超时 | 当前 Python 终端代理、具体失败域名；Docker 代理不替代它 |
| 磁盘增长超预期 | hub、datasets、导出数据和保存的划分可能各占一份 |
| IterableDataset 不能按行随机访问 | 使用迭代／take，或明确选择普通加载 |
| JSON 读法不对 | 检查是否为 JSONL，不能一律当单个数组 |
| 同 seed 结果不同 | 核对数据版本、行顺序、预处理与软件版本 |
| .gitignore 后文件还在 Git 中 | 它可能早已被跟踪，忽略规则不追溯历史 |

完成后应能说清：缓存在哪里、streaming 为什么省资源、Parquet 与 Arrow 的区别、两次划分如何计算比例、seed 为什么不是复现的全部、Git 与数据内容分别如何保存。

建议完成顺序：创建环境 → IMDB 检查 → 500 条格式转换 → 数据划分 → 下载 config.json → 大语料流式预览 → 四个练习。把用过的软件版本记录到 `outputs/requirements-observed.txt`：

```bash
python -m pip freeze > outputs/requirements-observed.txt
```

这份记录有助于回溯环境，但不等于包含所有系统依赖与数据版本的完整复现方案。

## 附：课程许可声明

本笔记依照课程结构整理、解释并改编部分示例。设备检查、Windows 适配及纠错为新增内容，不代表课程作者意见。上游采用 [MIT License](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/LICENSE)，声明如下：

```text
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
```

