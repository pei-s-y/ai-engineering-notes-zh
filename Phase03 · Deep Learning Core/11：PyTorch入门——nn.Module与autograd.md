# 11：PyTorch入门——nn.Module与autograd

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/11-intro-to-pytorch) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/11-intro-to-pytorch/docs/en.md)。原题：Introduction to PyTorch。

## 零、初学者先读：从自制引擎换到量产引擎

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第10课](<10：自制迷你框架——拼装前九课的零件.md>)你已经亲手拼出了迷你框架；本课把同一套心智模型搬进工业级引擎，为[第12课](<12：JAX入门——纯函数、jit与vmap.md>)的另一种设计哲学和[第13课](<13：调试神经网络——没有报错的排错法.md>)的排错做地基。

### 0.1 先理解这一课在做什么

上一课你造了一台“能跑的发动机”：自己写 Linear、ReLU、Adam、DataLoader。它的问题不是不对，而是慢——原课说在同样问题上比 PyTorch 慢约 500 倍（原课参考），因为它用嵌套 Python 循环逐样本计算，而 PyTorch 把同样的运算派发给优化过的 C++/CUDA 核在 GPU 上执行。本课要传达的核心事实是：**接口几乎没变，引擎彻底换了**。Module、forward()、backward()、optimizer.step() 这些你造过的零件，PyTorch 里一一对应，只是底下包了十年的系统工程。所以这课不难在概念，难在戒掉“我什么都手写”的惯性，学会信任框架。

### 0.2 本课术语先翻成白话

- 张量（tensor）=多维数组，但带三样“身份证”：形状、数据类型、在哪个设备上算
- 形状（shape）=每个维度多长：标量 ()、向量 (n,)、矩阵 (m,n)、图片批 (批,通道,高,宽)
- 数据类型（dtype）=每个数用多少位存：float32 训练默认、float16/bfloat16 省内存、int8 量化推理
- 设备（device）=数住在哪：cpu 还是 cuda（NVIDIA 显卡）
- 自动微分（autograd）=前向计算时顺带记账，反向时自动算梯度，不用手写 backward()
- 计算图（computational graph）=autograd 记下的“谁经过什么运算变成了谁”的流水账
- 磁带式微分（tape-based autodiff）=像磁带一样记录前向操作，`.backward()` 倒着回放求梯度
- nn.Module=PyTorch 所有网络零件的基类，赋值即自动注册参数
- logits=最后一层输出的“未归一化分数”，softmax 之前的原始值
- 交叉熵损失（CrossEntropyLoss）=分类损失，内部自带 LogSoftmax，**只喂 logits**
- 梯度累加=PyTorch 默认把梯度往 `.grad` 上累加而不是覆盖，所以每步前要 `zero_grad()`
- `torch.no_grad()`=临时关掉梯度记账，评估时省内存提速
- state_dict=参数名到张量的字典，模型的“可携带存档”
- DataLoader=负责分批、打乱、多进程取数的传送带
- 混合精度（mixed precision）=前向/反向用 float16 提速，主权重仍保 float32 保稳定
- 即时执行（eager execution）=写下 `y=model(x)` 立刻算出 y，而不是先攒图再编译

### 0.3 用小数字走一遍

**算例一：autograd 到底替你算了什么。** 设 x=2，y=3·x²=12。手算导数 dy/dx=6x=12。autograd 的“记账—回放”过程是这样的：前向时它记两笔账——t₁=x·x，y=3·t₁；`.backward()` 时倒着走：∂y/∂t₁=3，∂t₁/∂x=2x=4，链式相乘 3×4=12，存进 `x.grad`。你对一个张量做 `x = torch.tensor([2.0, 3.0], requires_grad=True); y = 3 * x**2; y.sum().backward()`，则 `x.grad` 应为 [12.0, 18.0]（6x 在 x=2、3 两点的值）。第[03课](<03：从零实现反向传播——链式法则与计算图.md>)你手推的链式法则，这里全部自动。

**算例二：为什么必须 zero_grad。** 还是 x=2、梯度 12，学习率 0.01。正确流程：更新 x ← 2 − 0.01×12 = 1.88。如果忘了清零，第二次 backward 后 `x.grad` = 12+12 = 24（梯度是**累加**的），更新变成 x ← 2 − 0.01×24 = 1.76——凭空走了一倍步长。这就是[第13课](<13：调试神经网络——没有报错的排错法.md>)“损失振荡”最常见根源之一。

**算例三：参数量自己会算。** 784→256 的 Linear 层有权重 784×256=200,704 个加偏置 256 个，共 200,960；再加 256×128+128=32,896 和 128×10+10=1,290，总参数 235,146——和原课正文给出的数字一致（原课参考的核对）。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按依赖顺序读：第二节先过概念（张量→autograd→nn.Module→训练循环→DataLoader→GPU），其中每个小代码块都值得亲手敲一遍；第四节是原课 Build It，用**纯 PyTorch**（不用 torchvision）从原始二进制文件加载 MNIST、定义 3 层 MLP、跑标准训练循环，四个步骤层层递进；第六节是“生产姿势”（保存/加载、学习率调度）。核心对象只有五个：tensor、nn.Module、loss、optimizer、DataLoader。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 CrossEntropyLoss 要喂 logits 而不是 softmax 之后的概率？

**参考答案：** 它内部已经组合了 LogSoftmax+NLLLoss，你若先做 softmax 再喂进去，等于 log 又套一层 softmax，梯度在数值上和数学上都是错的，而且**不报错**——损失在降，只是降得不对。这是原课点名的“静默错误”典型。

**问题2：** `model.eval()` 和 `torch.no_grad()` 是一回事吗？

**参考答案：** 不是。`eval()` 切换模块行为（Dropout 关闭、BatchNorm 用统计量而非批统计量）；`no_grad()` 关闭梯度记账省内存。评估时两个都要，各管一摊。

**问题3：** `optimizer.zero_grad()` 放循环开头和放 `optimizer.step()` 之后，效果一样吗？

**参考答案：** 等价，只要每个 batch 恰好清一次。原课概念节的标准写法放在每批开头，时序图里画在 step 之后（核对：两种顺序都正确，本篇统一采用开头写法）。

能讲清这三个问题后，进入正文。

---

## 一、这一课解决什么问题？

你的迷你框架能训练 4 层网络，但纯 Python 逐样本循环让它在同样任务上慢约 500 倍（原课参考），而且没有 GPU、没有自动微分（每个模块手写 backward）、没有序列化、没有分布式、没有混合精度。原课给了一个量级感受：单张 A100 上 PyTorch 训 ResNet-50（2560 万参数、128 万张图）约 6 小时，你的框架同样任务约 3000 小时——如果没先爆内存（原课参考）。PyTorch 一次性补齐所有缺口，而且**不要求你换脑子**：Module、forward()、parameters()、backward()、optimizer.step() 与你第10课自己设计的接口一一对应。

## 二、核心概念与推导

### 2.1 为什么 PyTorch 赢了

2015 年的 TensorFlow 要求先定义静态计算图再喂数据，调试像看电路图，改结构等于重建。2017 年 PyTorch 上线，主打即时执行（eager execution）：写 Python、立刻执行、print() 和 pdb 都能用、forward 里能写 if/else。到 2020 年市场投票结束：ML 研究论文中 PyTorch 份额从 2017 年的 7% 涨到 2022 年的 75% 以上，Meta、Google DeepMind、OpenAI、Anthropic、Hugging Face 都以它为主力框架（原课参考）；TensorFlow 2.x 反过来采纳即时执行，等于承认这条路线。教训：开发体验会复利——慢 10% 但调试快 50% 的框架每次都赢。

### 2.2 张量：形状、dtype、device

| 属性 | 管什么 | 常见值 |
|---|---|---|
| shape | 维度结构 | ()、(n,)、(m,n)、(batch,channels,H,W) |
| dtype | 精度与内存 | float32（默认训练）、float16（混合精度）、bfloat16（LLM 训练，范围同 float32 精度更低）、int8（量化推理） |
| device | 在哪算 | cpu / cuda |

新手第一大报错就是设备不匹配：`RuntimeError: Expected all tensors to be on the same device`——所有参与运算的张量必须在同一设备上，先统一搬再算。重塑是常数时间操作（只改元数据不改数据）：`view()` 要求内存连续，`reshape()` 总能用，另有 `permute`、`unsqueeze`、`squeeze`。

### 2.3 autograd：记账与回放

你手写过每个模块的 backward()；PyTorch 不需要。它把前向中对张量做的每个操作记成一张有向无环图（计算图），`.backward()` 时反向遍历自动求梯度：

~~~python
# 本块中文说明：原课的 autograd 最小示例——梯度 dz/dx = 2x + 3。
import torch

x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
~~~

三条规则（原课“autograd 三法则”）：1）只有 `requires_grad=True` 的叶子张量会积累梯度；2）梯度默认累加，每次 backward 前要 `optimizer.zero_grad()`；3）`torch.no_grad()` 关闭记账，评估时用。

### 2.4 nn.Module：赋值即注册

~~~python
# 本块中文说明：原课的 nn.Module 示例——与第10课自制 Module 同构。
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
~~~

在 `__init__` 里把 nn.Module 或 nn.Parameter 赋值为属性，PyTorch 自动注册，`model.parameters()` 递归收齐所有参数——不用再像迷你框架那样手工收集。常用积木（参数量公式）：

| 模块 | 作用 | 参数量 |
|---|---|---|
| nn.Linear(in,out) | Wx+b | in×out+out |
| nn.Conv2d(in_ch,out_ch,k) | 二维卷积 | in_ch×out_ch×k²+out_ch |
| nn.BatchNorm1d(f) | 归一化激活 | 2f |
| nn.Dropout(p) | 随机置零 | 0 |
| nn.ReLU() | max(0,x) | 0 |
| nn.Embedding(v,d) | 查表 | v×d |
| nn.LayerNorm(d) | 逐样本归一化 | 2d |

### 2.5 损失与优化器：现成的积木

损失（torch.nn）：MSELoss（回归）、CrossEntropyLoss（多分类，**喂 logits 不喂 softmax**）、BCEWithLogitsLoss（二分类，同理喂 logits）、L1Loss（稳健回归）。优化器（torch.optim）：SGD+momentum（CNN、精调管线，典型 lr 0.01—0.1）、Adam（默认起点，1e-3）、AdamW（Transformer/微调，1e-4—1e-3）、LBFGS（小规模二阶，1.0）——括号内典型值为原课参考。

### 2.6 训练循环：五行走天下

~~~python
# 本块中文说明：原课的“标准五步”训练循环。
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
~~~

批内五步：zero_grad → 前向 → 算损失 → backward → step。原课的话：训练 GPT-4、Stable Diffusion、LLaMA 的就是这五行，变的是架构和数据，不变的是这五行（原课陈述）。

### 2.7 Dataset 与 DataLoader

`Dataset` 只要实现 `__len__` 和 `__getitem__`；`DataLoader` 包一层就有分批、打乱、多进程取数（`num_workers=4` 起 4 个进程并行备料，磁盘密集任务仅此一项可让训练速度翻倍——原课参考）。

### 2.8 GPU 与混合精度

`model.to(device)` 递归把所有参数和缓冲搬上 GPU，训练时每个 batch 再 `.to(device)`。混合精度用 float16 跑前向/反向、float32 留主权重，在现代 GPU（A100/H100/RTX 4090）上内存减半、吞吐翻倍（原课参考），配套 `torch.amp` 的 autocast 与 GradScaler（见第六节与练习3）。

### 2.9 三框架对照

| 维度 | 迷你框架(L10) | PyTorch | JAX |
|---|---|---|---|
| 自动微分 | 手写 backward() | 磁带式 autograd | 函数式变换 |
| 执行 | 即时（Python 循环） | 即时（C++ 核） | 追踪+JIT 编译 |
| GPU | 无 | CUDA/ROCm/MPS | CUDA/TPU |
| MNIST MLP 速度 | ~300s/epoch | ~0.5s/epoch | ~0.3s/epoch（均为原课参考） |
| 模块系统 | 自制 Module | nn.Module | 无状态函数（Flax/Equinox） |
| 调试 | print() | print、pdb | 较难（追踪破坏 print） |
| 生产用户 | 玩具 | Meta、OpenAI 等 | DeepMind 等 |

## 三、操作前准备

Build It 需要 PyTorch；Use It 另需网络（下载 MNIST 原始文件）：

~~~bash
python -m pip install torch
~~~

先激活你已有的 Python 虚拟环境；以上命令只供按需执行，本次没有替你安装或下载数据。原课 Build It 会从 `storage.googleapis.com/cvdf-datasets/mnist/` 下载 4 个 gzip 文件到 `./mnist_data`，首次运行需联网。

## 四、Build It：原课分步代码与中文讲解

任务：3 层 MLP 训练 MNIST，只用 PyTorch 原语，不用 torchvision.datasets，自己下载并解析原始二进制数据。

### 4.1 第 1 步：从原始文件加载 MNIST

MNIST 是 4 个 gzip 文件：训练图 60000×28×28、训练标签、测试图 10000×28×28、测试标签。解析时读文件头（magic、数量、行、列）再读数据体，像素除以 255 归一到 0—1。

~~~python
# 本块中文说明：下载并解析 MNIST 原始 IDX 二进制格式。
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
~~~

（核对：两个 load 函数里 `data = f.read()` 这一行在原课代码中缺失，`torch.frombuffer(bytearray(data), ...)` 引用了未定义的 `data`。补上 `data = f.read()` 后语义才完整——即读取文件头之后的全部数据体。此处按原样转录正文并在本行说明，未改动上方代码块本体。）另注意 `magic` 读出后未校验值是否为 2051/2049，属教学实现的省略，不是错误。

### 4.2 第 2 步：定义模型

784→256→128→10 的三层 MLP，ReLU 激活，Dropout(0.2) 正则，不用 batch norm 保持简单。输出层给 10 个原始 logits，**不做 softmax**——CrossEntropyLoss 内部处理。

~~~python
# 本块中文说明：3 层 MLP，nn.Sequential 串积木。
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
~~~

参数量 784·256+256 + 256·128+128 + 128·10+10 = 235,146（原课参考；与 0.3 算例三一致）。对比 GPT-2 small 的 1.24 亿，这是极小网络，秒级可训（原课参考）。

### 4.3 第 3 步：训练与评估函数

标准五步的函数化；评估用 `model.eval()` + `torch.no_grad()`——不关记账的话，PyTorch 会为一次你根本不反向传播的前向白白建计算图，费内存又慢。

~~~python
# 本块中文说明：原课的 train_one_epoch 与 evaluate。
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
~~~

### 4.4 第 4 步：串起来

~~~python
# 本块中文说明：主流程——数据、模型、损失、优化器、10 轮训练、存档。
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
~~~

原课预期：10 轮后测试准确率约 97.8%，CPU 训练约 30 秒、GPU 约 5 秒；同架构在迷你框架上约 45 分钟（均为原课参考）。

## 五、把代码运行起来：补充入口与检查

（补充）原课各步按顺序执行即可，Build It 依赖联网下载。以下是**不需要下载数据**的最小检查，数值全部可手算（与 0.3 节算例对应）：

~~~python
# 本块中文说明：补充的最小 autograd/参数量检查入口。
import torch

# 检查1：0.3 算例一——y=3x²，dy/dx=6x。
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = 3 * x ** 2
y.sum().backward()
# 可手算预期：x.grad == [12.0, 18.0]。

# 检查2：0.3 算例二——梯度累加行为。
x2 = torch.tensor([2.0], requires_grad=True)
for _ in range(2):
    (3 * x2 ** 2).sum().backward()
# 可手算预期：x2.grad == [24.0]（两次 12 累加），演示为何要 zero_grad()。
x2.grad.zero_()

# 检查3：no_grad 下不记账。
with torch.no_grad():
    z = (x ** 2).sum()
# 可手算预期：z.requires_grad == False。

# 检查4：参数量。模型构造不依赖数据下载。
model = MNISTModel()
assert sum(p.numel() for p in model.parameters()) == 235146
~~~

## 六、Use It：使用现成工具

**迷你框架 → PyTorch 迁移对照（原课表）：**

| 迷你框架(L10) | PyTorch |
|---|---|
| `Sequential(Linear(784,256), ReLU(), ...)` | `nn.Sequential(nn.Linear(784,256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `criterion.backward()` 再 `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |

接口几乎一致，差异全在引擎盖下面。

**保存与加载**——永远存 state_dict 而不是整个模型对象（后者走 pickle，代码一重构就失效）：

~~~python
# 本块中文说明：state_dict 存档与读档。
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
~~~

**学习率调度**——PyTorch 自带 15+ 种（StepLR、CosineAnnealingLR、OneCycleLR、ReduceLROnPlateau 等），都插在同一个 optimizer 接口上：

~~~python
# 本块中文说明：余弦退火调度示例。
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
~~~

（核对：这两段 Use It 代码假设 `model`、`optimizer`、`train_loader` 等已在内存中，不能单独运行。）

## 七、失败模式、原文边界与工程注意

1. **设备不匹配是新手第一大报错**：模型在 GPU、数据在 CPU（或反过来）直接 RuntimeError。规则：模型搬一次，每个 batch 搬一次，全对齐后再算。
2. **给 CrossEntropyLoss 喂 softmax 输出**：静默产生错误梯度，不报错。同理 BCEWithLogitsLoss 喂 logits。
3. **忘 zero_grad**：梯度累加让有效学习率逐 batch 越来越大，损失振荡（0.3 算例二）。
4. **评估忘 model.eval()/no_grad**：Dropout、BatchNorm 行为不对，测试准确率每次不同；且白白建图费内存。
5. **view 与 contiguous**：`view()` 对非连续内存报错，改用 `reshape()`。
6. **混合精度的 API 版本**：原课用 `from torch.amp import autocast, GradScaler`，这是较新版本的推荐写法（旧版为 `torch.cuda.amp`）；`autocast(device_type="cuda")` 只对 CUDA 有意义，CPU 上收益有限（补充）。
7. **DataLoader 多进程在 Windows 的坑**（补充，原课未提）：`num_workers>0` 在 Windows 上要求入口有 `if __name__ == "__main__":` 保护，否则子进程递归启动报错；本课 Build It 未用多进程，不受影响。
8. **速度数字的边界**：500 倍、6 小时 ResNet-50、97.8% 等均为原课参考，随硬件与版本浮动。
9. **torch.save 只存权重**：结构变化后 `load_state_dict` 会因键名对不上而失败，这是特性不是缺陷——它逼你显式管理结构。

## 八、Ship It：本课可复用的中文检查模板

（补充，改写自原课 Ship It 的两个产物——调试提示词与训练模式速查）

~~~markdown
检查 PyTorch 训练脚本：模型、数据、损失是否都在同一 device；损失函数输入是否为 logits（CrossEntropyLoss/BCEWithLogitsLoss 不吃概率）；每批是否恰好一次 zero_grad；评估是否 model.eval()+torch.no_grad()；保存的是否为 state_dict；若用 GPU，是否报告了吞吐（样本/秒）而不只是准确率；引用的任何速度与准确率数字注明来源与硬件。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：加 batch normalization——每个 Linear 后（激活前）插 `nn.BatchNorm1d`，对比 dropout 版的测试准确率与训练速度；预期更少轮数达 98%+（原课参考）。验收：能给出两版的学习曲线并指出 BN 版第几轮追平 dropout 版最终精度。
2. 原课：实现学习率探查器——一个 epoch 内学习率从 1e-7 指数升到 1.0，画 loss-LR 曲线，取“损失开始爬升前”的值为最优 LR。验收：曲线能标出最陡下降段，并解释为何选在其前一档（与[第09课](<09：学习率调度——warmup与衰减策略.md>)呼应）。
3. 原课：GPU 混合精度移植——加 `torch.amp.autocast` 与 `GradScaler`，测吞吐；A100 上预期约 2 倍加速（原课参考）。验收：报告有/无混合精度的样本/秒，并说明 GradScaler 为何必要（float16 梯度可能下溢）。
4. 原课：自制 Dataset——下载 Fashion-MNIST（与 MNIST 同格式），实现 `FashionMNISTDataset(Dataset)`，训同一 MLP；Fasion-MNIST 更难，预期约 88% 对 98%（原课参考）。验收：`__len__`、`__getitem__` 齐备，能说明二者准确率差异来自任务难度。
5. 原课：把 Adam 换成 `SGD(lr=0.01, momentum=0.9)`，比较收敛曲线，再加 `CosineAnnealingLR` 看 SGD 到第 10 轮能否追上 Adam。验收：两条曲线同一张图，能解释动量+调度如何弥补自适应优化器的起步优势。

## 十、术语与完成标准

张量（tensor）：带形状/dtype/device 的多维数组，操作可被 autograd 记账；自动微分（autograd）：磁带式记录前向、`.backward()` 反向回放求精确梯度；nn.Module：可微计算块基类，自动注册参数、支持嵌套、区分 train/eval；state_dict：参数名到张量的有序字典，可序列化的模型存档；.backward()：反向遍历计算图为所有 require_grad 叶子累积梯度；.to(device)：递归搬运参数与缓冲；DataLoader：分批、打乱、并行取数的迭代器；混合精度：float16 前向/反向+float32 主权重；即时执行：调用即执行，PyTorch 对 TF1.x 静态图的根本区别；zero_grad：清零梯度（默认累加机制的配套动作）。

完成检查：

- 能手推 0.3 算例一（tape 如何算出 12）并说出 x.grad 为何是 [12, 18]。
- 能解释三条 autograd 法则各自防住什么坑。
- 能背写批内五步并说出每步对应对应迷你框架的哪个零件。
- 能算出 784→256→128→10 的参数量 235,146。
- 能说出 CrossEntropyLoss 喂 softmax 后果与原因。
- 能说出 eval() 与 no_grad() 的分工。

## 十一、验证范围与来源许可

原课 Build It 共 4 个分段，代码逐字转录嵌入本篇（仅添加块首中文说明行，未改代码本体）；4.1 缺失的 `data = f.read()` 以“（核对：…）”形式显式标注而非悄悄补入。概念节的 autograd、MLP、训练循环、调度器等短代码同样保留原样。原课页面未展示运行输出，本篇所有性能数字（500 倍、97.8%、30 秒等）均标为原课参考；“可手算预期”数值与 0.3 节算例一致可手工复核。本次未执行训练、未下载 MNIST；正文 python 代码块已抽出为临时文件做 `ast.parse` 静态语法检查，结果见下。

验证记录（2026-11-06）：
- 课程页抓取一次成功，内容以课程页为准整理。
- 正文 python 代码块全部通过 `ast.parse` 静态语法检查（4.1 的缺失变量属原课代码问题，语法本身合法，已标注）。
- 未做：实际训练 MNIST、GPU/混合精度实验、与迷你框架的速度对比。

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
