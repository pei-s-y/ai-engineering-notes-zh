# 09：学习率调度——warmup与衰减策略

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/09-learning-rate-schedules) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/09-learning-rate-schedules/docs/en.md)。原题：Learning Rate Schedules and Warmup。

## 零、初学者先读：步子该多大、什么时候改小

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。上一课（[第08课](<08：权重初始化——零、随机、Xavier与Kaiming.md>)）定了出生点，本课管训练全程的步长；下一课（[第10课](<10：自制迷你框架——拼装前九课的零件.md>)）把前九课的零件全部拼成框架。

### 0.1 先理解这一课在做什么

下山找谷底：一开始离得远，迈大步赶紧接近；快到谷底了，还迈大步就会在两侧来回跨过谷底——永远踩不进去。学习率调度就是“步长随时间变化的计划表”：前期大步赶路，后期小步安家。原课的三个失败场景（原课参考）：学习率 0.1，3 步内损失冲上天（发散）；0.0001，100 轮后模型几乎没挪窝（停滞）；0.01，前 50 轮还行，之后损失在一个永远进不去的极小值附近来回振荡（不衰减）。本课要实现五种调度——常数、阶梯、余弦、warmup+余弦、1cycle——并亲手看到这三种失败长什么样。

### 0.2 本课术语先翻成白话

- 学习率（learning rate）=梯度乘上它得到参数更新量：步长
- 调度（schedule）=把“训练第几步”映射到“此刻学习率”的函数
- 常数调度=从头到尾一个数
- 阶梯衰减（step decay）=每到固定轮数把学习率乘一个因子（常 0.1），像下台阶
- 余弦退火（cosine annealing）=按余弦曲线从 lr_max 平滑滑到 lr_min
- 预热（warmup）=前 N 步从近零线性爬到 lr_max，等优化器统计稳定
- 1cycle 策略=前半程从低爬到高、后半程再降到底，一个来回
- 学习率范围测试（LR range test）=训练几百步同时指数抬高学习率，看损失从哪开始上升来定 lr_max
- SGDR=带热重启的余弦：每隔 T 步把学习率重置回 lr_max 再衰减
- eta_min / 峰值学习率（peak lr）=调度的下限 / 最高点
- step / epoch=一次参数更新 / 全部数据过一遍

### 0.3 用小数字走一遍

设总训练 T=6 个 epoch，lr_max=0.1，lr_min=0.01，warmup=2 步。只用到五个特殊角的余弦值：cos30°≈0.866、cos60°=0.5、cos90°=0、cos45°≈0.707、cos135°≈−0.707。

**常数：** 每轮都是 0.1。

**阶梯衰减（每 3 轮乘 0.1）：** 公式 0.1×0.1^⌊t/3⌋，t 从 0 数：第 1—3 轮 0.1，第 4—6 轮 0.01。

**指数衰减（γ=0.7）：** 0.1×0.7^t：0.1000、0.0700、0.0490、0.0343、0.0240、0.0168。

**余弦：** lr(t)=0.01+0.5×(0.1−0.01)×(1+cos(π t/6))。手算 t=1：0.01+0.045×(1+0.866)=0.01+0.0840≈0.0940；t=3：cos(90°)=0 → 0.01+0.045=0.055；t=5：cos(150°)=−0.866 → 0.01+0.045×0.134≈0.0160。

**warmup+余弦：** t=0 为 0；t=1 为 0.1×(1/2)=0.05；t=2 到顶 0.1；之后在剩余 4 步里余弦下滑：t=4 进度 (4−2)/4=0.5、cos(90°)=0 → 0.055；t=5 进度 0.75、cos(135°)≈−0.707 → 0.01+0.045×0.293≈0.0232。

汇总（t 从 0 起数，epoch 列为 t+1）：

| epoch | t | 常数 | 阶梯(每3轮÷10) | 指数(γ=0.7) | 余弦 | warmup+余弦 |
|---|---|---|---|---|---|---|
| 1 | 0 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0000 |
| 2 | 1 | 0.1000 | 0.1000 | 0.0700 | 0.0940 | 0.0500 |
| 3 | 2 | 0.1000 | 0.1000 | 0.0490 | 0.0775 | 0.1000 |
| 4 | 3 | 0.1000 | 0.0100 | 0.0343 | 0.0550 | 0.0868 |
| 5 | 4 | 0.1000 | 0.0100 | 0.0240 | 0.0325 | 0.0550 |
| 6 | 5 | 0.1000 | 0.0100 | 0.0168 | 0.0160 | 0.0232 |

余弦列可见形状特点：开头降得慢、中间快、结尾又慢（0.1000→0.0940 慢，0.0775→0.0550 快，0.0325→0.0160 又慢）。训练在 t=5 用最后一步，t=6 若继续会恰好到 lr_min=0.01。

**1cycle（同参数，lr 从 0.1/25 爬到 0.1 再降到 0.1/10000）：** t=0..6 依次为 0.0040、0.0360、0.0680、0.1000、0.0667、0.0333、0.00001——前半程升 3 步、后半程降 3 步。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：第 1 步五个调度函数是主角，每个都返回“这一步该用多大的学习率”；第 2 步 `visualize_schedule` 把它们画成文本图；第 3 步 `train_with_schedule` 是实验机——在圆环数据上训一个 2→8→1 的小网络，训练循环里每一步先问调度函数要学习率；第 4 步 `compare_schedules` 五种调度各训一遍对比损失；第 5 步 `lr_sensitivity` 演示学习率过大/过小/正好的三种命运。数据结构全是普通列表，无第三方库。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 warmup 对 Adam 尤其重要？

**参考答案：** Adam 维护梯度均值与方差的滑动估计，第 0 步它们初始化为零——前几步的更新建立在垃圾统计上。此时若学习率很大，模型会迈出方向糟糕的大步。warmup 让学习率从近零线性爬升，等爬到全速时 Adam 的统计已经稳定。

**问题2：** 1cycle 前半程把学习率抬高，这不是更不稳吗，为什么反而有效？

**参考答案：** 高学习率相当于给优化轨迹注入噪声，起正则化作用：模型在爬坡期探索损失曲面的更多区域，找到更好的“盆地”；后半程再降下来，在最佳盆地里精细收敛。代价是必须预知总步数。

**问题3：** 阶梯衰减和余弦都在降学习率，各自的麻烦是什么？

**参考答案：** 阶梯的降点（第几轮降、降多少）依赖数据集与架构，换问题就要重调，且突变时损失可能瞬间弹高；余弦几乎没有额外超参（只要 lr_max 和 lr_min），形状还匹配“训练中段学习最快”的经验，所以是现代默认。

能解释这张 6 轮小表后，进入下面的公式与代码。

---

## 一、这一课解决什么问题？

学习率是单一最重要的超参数——不是架构、不是数据量、不是激活函数（原课观点）。90% 准确率与 95% 的差距常常只差一张调度表。原课给的背景（原课参考）：近三年发表的每个主流模型都用学习率调度——Llama 3 峰值 3e-4、warmup 2000 步、余弦衰减到 3e-5；GPT-3 峰值 6e-4、375M token 的 warmup；这些不是拍脑袋，是花数百万美元超参扫出来的。你需要懂调度，因为默认值不会适合你的问题：微调预训练模型与从头训不同；加大 batch size 时 warmup 期要改；训练在第 10000 步崩了，你需要判断这是不是调度问题。前置：[第06课](<06：优化器——从SGD到Adam.md>)的优化器与[第08课](<08：权重初始化——零、随机、Xavier与Kaiming.md>)的初始化。

## 二、核心概念与推导

### 2.1 常数学习率

$$\text{lr}(t) = \text{lr}_0$$

最简单也几乎从不最优：对训练结尾太大（在极小值附近振荡）或对开头太小（小步浪费算力）。适合小模型与调试；训练超过一小时的任务上是糟糕选择（原课观点）。

### 2.2 阶梯衰减

$$\text{lr}(t) = \text{lr}_0 \cdot \gamma^{\lfloor \text{epoch} / \text{step\_size} \rfloor}$$

ResNet 时代的做法：γ=0.1、step_size=30，即每 30 轮降 10 倍。ResNet-50 的用法：lr=0.1，在第 30/60/90 轮各降 10 倍、无 warmup（原课参考）。问题：最优降点依赖数据集与架构，换问题要重调；切换是突变的，学习率骤变时损失可能弹高。

### 2.3 余弦退火

$$\text{lr}(t) = \text{lr}_{\min} + 0.5(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

t 为当前步、T 为总步数。t=0 时余弦项为 1，lr=lr_max；t=T 时为 −1，lr=lr_min。形状：开头缓、中间快、结尾缓——匹配“大部分学习发生在训练中段”的经验观察，所以中段要保持较大的步长。除 lr_max、lr_min 外无超参，是现代训练的默认。

### 2.4 warmup：为什么要从小开始

自适应优化器（Adam 等）维护梯度均值与方差的运行估计，第 0 步这些估计是零，头几次更新基于垃圾统计；此时学习率若大，模型迈出巨大而方向错误的步。warmup 的修法：从极小学习率（常为 lr_max/warmup_steps 或零）线性爬升，N 步后到达 lr_max：

$$\text{lr}(t) = \text{lr}_{\max}\cdot\frac{t}{\text{warmup\_steps}}\quad (t < \text{warmup\_steps})$$

典型 warmup 占总步数的 1—5%（原课参考；Llama 3 约 1.8 万亿 token 训了 2000 warmup 步，GPT-3 warmup 375M token）。

### 2.5 线性 warmup + 余弦衰减：现代默认

```
if t < warmup_steps:
    lr(t) = lr_max * (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

Llama、GPT、PaLM 与多数现代 transformer 都用它：warmup 防早期不稳定，余弦把模型安进好的极小值。0.3 节的表已把这两个公式在 6 个 epoch 上算过一遍。

### 2.6 1cycle 策略

Leslie Smith（2018）：前半程从低值爬到高值、后半程降回极低——反直觉（训练中途加大步子？）。理论：高学习率作为正则化，给轨迹加噪声，探索更广的损失地形、找到更好的盆地；降速段再在最优盆地里精修。要点位（原课）：阶段一从 lr_max/25 爬到 lr_max，阶段二从 lr_max 降到 lr_max/10000。同样算力预算下 1cycle 常比余弦收敛更快；代价是必须预知总步数。

### 2.7 怎么选（依原课决策图整理）

- 不知道总训练步数 → 常数学习率 + 手动衰减。
- 知道步数、预算大（数天/周）→ warmup+余弦（Llama/GPT 默认），warmup 取 1—5% 步数。
- 预算小（数小时）→ 1cycle（收敛最快），先用 LR range test 找 lr_max。
- 中等预算 → 纯余弦（安全默认），lr_min 取 lr_max/10。

已发表模型的实配（原课参考）：Llama 3（405B）峰值 3e-4、warmup 2000 步、余弦到 3e-5；GPT-3（175B）峰值 6e-4、warmup 375M token、余弦到 0；ResNet-50 峰值 0.1、无 warmup、30/60/90 轮各降 10 倍；BERT（340M）峰值 1e-4、warmup 10K 步、线性衰减。

## 三、操作前准备

Build It 只用 Python 标准库（math、random），无第三方依赖；Use It 部分需要 PyTorch 与 HuggingFace transformers：

~~~bash
python -m pip install torch transformers
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行。补充提醒：第 4 步五种调度各训 300 轮×200 样本、第 5 步五种学习率各训 100 轮，纯 Python 无 numpy，需要跑一阵子。

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：五个调度函数

每个函数签名统一：输入当前步数，返回此刻学习率；`**kwargs` 让训练循环能把 lr、total_steps 等一股脑传进来，不用了的参数自己吞掉。对照 0.3 节的表逐个看：`step_decay_schedule` 的 `step // step_size` 就是表格里的 ⌊t/3⌋；`one_cycle_schedule` 的 25 与 10000 就是 2.6 节说的 lr_max/25 与 lr_max/10000。

~~~python
# 本块中文说明：常数、阶梯、余弦、warmup+余弦、1cycle 五种调度函数。
import math


def constant_schedule(step, lr=0.01, **kwargs):
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress
~~~

### 4.2 第 2 步：调度可视化

取 20 个采样步打印文本柱状图，柱长与当前学习率成正比。

~~~python
# 本块中文说明：把任意调度画成 40 格宽的文本柱状图。
def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n{name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"  Step {s:4d}: lr={lr_val:.6f} {bar}")
~~~

### 4.3 第 3 步：带调度的训练网络

圆环数据集上的两层网络（2→8→1，ReLU+Sigmoid），与前面课的一致，只是训练循环里每一步先调 `schedule_fn(step, ...)` 拿当前学习率再更新。核对两点：其一，参数 `schedule_name` 定义后从未使用（原课冗余参数，保留原样）；其二，`d_out = error * out * (1 - out)` 是“平方误差经 sigmoid”的梯度少乘一个 2（等价于 ½·平方误差的梯度），而 `total_loss` 累计的是普通平方误差——梯度与报告的损失差一个常数因子，最优点相同，相当于有效学习率减半，教学演示无碍。

~~~python
# 本块中文说明：圆环数据 + 两层网络，训练中每一步向调度函数要学习率。
import random


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        accuracy = correct / len(data) * 100
        epoch_losses.append(avg_loss)

    return epoch_losses
~~~

补充：权重用 `std=sqrt(2/2)=1` 采样——对 fan_in=2 的层这是第 08 课的 Kaiming 公式（2/fan_in）取值，前后课在这里接上了。

### 4.4 第 4 步：五种调度对比

同一数据、同一网络，各训 300 轮（总步数=300×200=60000），打印起始/中途/结尾/最优损失。核对：Step Decay 配置 `step_size=15000` 意味着第 30000 步（约第 150 轮、正好半程）降 10 倍；Warmup+Cosine 的 `warmup_steps=3000` 是总步数的 5%——都落在原课“warmup 1—5%”的建议区间内。

~~~python
# 本块中文说明：同任务下五种调度的损失对比。
def compare_schedules(data):
    configs = [
        ("Constant", constant_schedule, {}),
        ("Step Decay", step_decay_schedule, {"step_size": 15000, "gamma": 0.1}),
        ("Cosine", cosine_schedule, {"lr_min": 1e-5}),
        ("Warmup+Cosine", warmup_cosine_schedule, {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("1cycle", one_cycle_schedule, {}),
    ]

    print(f"\n{'Schedule':<20} {'Start Loss':>12} {'Mid Loss':>12} {'End Loss':>12} {'Best Loss':>12}")
    print("-" * 70)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(schedule_fn, name, data, epochs=300, base_lr=0.05, **extra_kwargs)
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"{name:<20} {losses[0]:>12.6f} {losses[mid_idx]:>12.6f} {losses[-1]:>12.6f} {best:>12.6f}")
~~~

### 4.5 第 5 步：学习率过大 vs 过小

常数调度下扫 1.0 到 0.0001 五个学习率各训 100 轮，按结尾损失粗分类：DIVERGED（不降反升或 NaN）、BARELY MOVED（几乎没动）、CONVERGED、LEARNING。核对：判定阈值（`end > start * 0.9` 等）是原课自定的经验规则，不是理论界线。

~~~python
# 本块中文说明：演示发散、停滞、正常三种学习率命运。
def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]

    print("\nLR Sensitivity (constant schedule, 100 epochs):")
    print(f"  {'LR':>10} {'Start Loss':>12} {'End Loss':>12} {'Status':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if end > start or math.isnan(end) or end > 1.0:
            status = "DIVERGED"
        elif end > start * 0.9:
            status = "BARELY MOVED"
        elif end < 0.15:
            status = "CONVERGED"
        else:
            status = "LEARNING"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")
~~~

## 五、把代码运行起来：补充入口与检查

（补充）原课各步是函数定义，页面未给主入口；按第四节顺序执行后需要自己调用。调度函数本身不依赖随机数，可以精确断言——以下预期值全部来自 0.3 节的手算表：

~~~python
# 本块中文说明：调度函数的可手算检查（对应 0.3 节表格）。
assert abs(cosine_schedule(0, lr=0.1, total_steps=6, lr_min=0.01) - 0.1) < 1e-12
assert abs(cosine_schedule(1, lr=0.1, total_steps=6, lr_min=0.01) - 0.0940) < 1e-4
assert abs(cosine_schedule(3, lr=0.1, total_steps=6, lr_min=0.01) - 0.055) < 1e-12
assert abs(step_decay_schedule(3, lr=0.1, step_size=3, gamma=0.1) - 0.01) < 1e-12
assert abs(warmup_cosine_schedule(1, lr=0.1, total_steps=6, warmup_steps=2, lr_min=0.01) - 0.05) < 1e-12
assert abs(warmup_cosine_schedule(5, lr=0.1, total_steps=6, warmup_steps=2, lr_min=0.01) - 0.0232) < 1e-4
assert abs(one_cycle_schedule(0, lr=0.1, total_steps=6) - 0.004) < 1e-12
assert abs(one_cycle_schedule(3, lr=0.1, total_steps=6) - 0.1) < 1e-12

# 入口：实验本体较慢，按需运行
data = make_circle_data()
compare_schedules(data)
lr_sensitivity(data)
~~~

补充检查方法：先用 `visualize_schedule` 五连看形状（阶梯像断崖、余弦像滑梯、warmup+余弦像先爬坡再滑梯）；`lr_sensitivity` 里 lr=1.0 预期 DIVERGED、lr=0.0001 预期 BARELY MOVED（原课参考定性结论）。本篇未运行训练实验，不虚构具体损失数值。

## 六、Use It：使用现成工具

PyTorch 的调度器在 `torch.optim.lr_scheduler`。核对：这段代码里的 `train_step`、`nn` 未定义（原课伪接口示意——`nn` 需另 `import torch.nn as nn`，`train_step` 是你自己的训练一步函数），照抄不能直接运行。

~~~python
# 本块中文说明：PyTorch 调度器的标准用法（示意，含未定义的伪接口）。
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

for step in range(1000):
    loss = train_step(model, optimizer)
    scheduler.step()
~~~

warmup+余弦可用 lambda 调度器，或直接用 HuggingFace 的 `get_cosine_schedule_with_warmup`：

~~~python
# 本块中文说明：HuggingFace 的 warmup+余弦调度器。
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)
~~~

原课建议：拿不准就用 warmup+余弦、warmup 取总步数 3—5%，几乎适用于一切场景。这正是多数 Llama/GPT 微调脚本的用法。

## 七、失败模式、原文边界与工程注意

1. 三种失败对号入座（本课学习目标）：损失飙升/NaN=学习率过高（发散）；损失几乎不动=过低（停滞）；损失在低位来回弹、不再下降=没衰减（振荡）。先跑 `lr_sensitivity` 建立手感再训真模型。
2. 调度需要预知总步数 T：余弦、warmup+余弦、1cycle 都要。训练长度改了，调度要跟着重算——T 写小了余弦会提前贴地。
3. 微调预训练模型与从头训练的合理调度不同（原课明确提醒）；抄 Llama/GPT 的数字前先想 batch size、数据量是否同一量级——那些数字是百万美元扫出来的，不是通用默认。
4. warmup 与 batch size 有耦合：大 batch 常配更长的 warmup（原课引 Goyal 等的线性缩放规则方向）。
5. 第 4 步的对比结论依赖具体数据与轮数（原课页面未展示数值，本篇不虚构）；定性预期是：衰减类调度结尾损失低于常数、warmup 对早期稳定性有帮助。
6. 阶梯的突变可能让损失瞬间弹高；观察到衰减点附近的尖刺时先想到这一点，不是 bug。
7. 调度只调步长，救不了坏的初始化或坏的数据——排错顺序（补充）：先看初始化与数据，再谈调度。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/prompt-lr-schedule-advisor.md`（一个“为你的训练配置推荐学习率调度与超参”的提示词）。此处改写成自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
检查学习率调度：说明峰值学习率与它的确定方式（LR range test 还是沿用默认）；说明调度类型与公式（常数/阶梯/余弦/warmup+余弦/1cycle）；给出 warmup 步数占总步数的比例（经验区间 1—5%）；给出 lr_min 与总步数 T 的来源；若是微调，说明与从头训练的调度差异；训练中途崩坏时，先列出“这可能是调度问题”的判断依据（发散/停滞/振荡三分类）再下结论；所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课练习 1：实现指数衰减 lr(t)=lr₀·γ^t（γ=0.999），与余弦退火在圆环数据上对比。验收标准：两种调度的损失曲线画在同一张图上，能指出指数衰减“早段降得快、末段贴地过早”与余弦“中段保持大步”的差异。
2. 原课练习 2：实现 LR range test——训练几百步，学习率从 1e-7 指数升到 1，画损失 vs 学习率；最优 lr_max 取在损失开始上升之前。验收标准：能从图上指出拐点并给出你自己的 lr_max 候选，并解释为什么不能用“损失最低点”而要用“开始上升前”（最低点常出现在略过拐弯处）。
3. 原课练习 3：warmup+余弦但改变 warmup 长度（0%、1%、5%、10%、20%），找训练最稳的取值。验收标准：五种长度的损失曲线对比图加一句话结论，且结论落在“过短不稳、过长浪费前段算力”的权衡框架里。
4. 原课练习 4：实现带热重启的余弦（SGDR）：每 T 步把学习率重置回 lr_max 再衰减，与标准余弦在更长训练上对比。验收标准：能展示重启时损失弹高再创新低的“热重启”形态，并说出它与 1cycle 的区别（多个周期 vs 一个周期）。
5. 原课练习 5：做“调度外科医生”——监控训练损失，损失稳定后自动从 warmup 切到余弦、损失平台期过长时自动降学习率。验收标准：能写清触发条件与阈值，并讨论误触发风险（噪声导致的假平台）。

## 十、术语与完成标准

学习率：梯度乘它得到参数更新量的标量；调度：步数到学习率的映射函数；warmup：前 N 步从近零线性爬升以稳定优化器统计；余弦退火：按余弦曲线从 lr_max 平滑降到 lr_min；阶梯衰减：固定轮数间隔上乘衰减因子；1cycle：先升后降的单周期策略；LR range test：指数抬学习率找发散点；SGDR：周期性重置的余弦；eta_min：调度下限；峰值学习率：warmup 结束后到达的最高值。

完成检查：

- 能手算 0.3 节那张 6 轮调度表（含 cos 特殊角的代入）。
- 能写出余弦公式并解释 t=0、t=T 两端的取值。
- 能说出 warmup 存在的原因（Adam 统计从零初始化）与典型占比（1—5%）。
- 能按“知不知道总步数 × 预算大小”选出合适的调度（2.7 节决策）。
- 能对发散/停滞/振荡三种症状给出学习率层面的诊断。

## 十一、验证范围与来源许可

原课 Build It 共 5 个分段，代码已逐字转录嵌入本篇（仅按本篇格式在代码块首行添加中文说明，未改逻辑；第 3 步保留原课的冗余参数 `schedule_name` 并显式标注，梯度与报告损失的常数因子差异也以“核对”标注而非修改）；Use It 的 PyTorch/HuggingFace 代码同样保留（其伪接口已标注）。原课页面未展示运行输出，本篇未虚构任何“原课参考”数值；第五节标注“可手算预期”的断言值全部来自 0.3 节手算表，并在临时脚本中用 Python 复核通过（仅调度函数的数值断言，属玩具检查，不构成训练或性能实验）。第 4、5 步的对比与敏感性实验未执行，定性预期以原课参考为准。本次未做：实际训练、超参搜索、与 PyTorch 调度器的数值比对。

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
