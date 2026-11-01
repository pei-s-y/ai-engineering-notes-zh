# 08：LoRA 与 QLoRA——只训练一小部分参数的微调

> 对应课程：Phase 11 / 08-fine-tuning-lora。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/08-fine-tuning-lora/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/08-fine-tuning-lora/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、先区分四件事

提示词把任务写清楚；RAG 把资料取出来放进输入；SFT（监督微调）用输入—目标输出样本训练模型；LoRA 则规定微调时**哪些参数以什么形式可训练**。LoRA 可用于 SFT，也可用于其他训练目标，不能把它当一种新的答案数据集。

**参数**是训练会更新的数值；**冻结**指不为某参数计算/应用更新；**梯度**描述损失如何随参数改变；**优化器**根据梯度更新参数。冻结基础权重不意味着前向计算不再经过基础模型，也不意味着训练完全不需要保存激活。

## 二、矩阵和低秩，先看尺寸

普通线性层 `y=Wx+b`，W 为 `d_out×d_in`。若输入/输出各 4096 维，W 有 `4096²=16777216` 个数。

LoRA 保留 W，同时用两个小矩阵产生增量：

`y=Wx+(α/r)BAx+b`

其中 A 为 `r×d_in`，B 为 `d_out×r`，乘积 BA 与 W 同形状。**秩 r**可理解为增量通过的“较窄中间通道”的宽度，限制这部分变化的表达自由度；它不等于分类标签数。

r=16 时两矩阵共有 `16×4096+4096×16=131072` 个数，为原 W 的 **0.78125%**。这是单个方阵的参数比例，不是整个训练程序显存自动减少到 0.78%。

用小例子想：输入 4 个数，先乘 A 压到 1 个中间数，再乘 B 展成 3 个输出增量。只训练 4+3=7 个参数，而直接训练 3×4=12 个；大矩阵上的节省更明显。

## 三、原代码为什么写成 x@A@B

PyTorch 批量输入常用**行向量**：x 为 `[batch,in]`。`nn.Linear` 保存 W 为 `[out,in]`，计算 `x@W.T+b`。

原 `LoRALayer` 的 A 存成 `[in,r]`，B 存成 `[r,out]`：

```text
x              A             B             输出
[batch,in] @ [in,r] → [batch,r] @ [r,out] → [batch,out]
```

它与论文列向量写法只是转置约定不同。合并回 `linear.weight` 时要加 `(A@B).T`，否则尺寸不对。看公式时先固定行/列约定，不要只背 BA 或 AB。

原代码随机初始化 A，B 初始化为 0，因此起步增量为 0，输出保持基础模型行为。第一次反传时 A 的梯度可能为 0，因为它通过零 B 传播；B 先动，后续 A 也能更新。如果 A/B 全部为零，可能让整个分支无法有效起步。

**alpha** 是缩放系数，常用 α/r；它不是优化器学习率本身。rank 越大容量和开销越大，最佳值要在独立验证集选择。原课的“任务→最佳 rank”表只能作尝试起点。

## 四、注入、训练、保存和合并

**Target modules** 指要加适配器的层，例如 q_proj、v_proj、k_proj、o_proj 和 MLP 投影。不同模型名字与形状不同，不能盲目复制匹配规则。只改 Q/V 或改所有线性层哪个更好取决于模型、任务和数据。

注入顺序：冻结基础参数→找目标线性层→包装为基础分支+LoRA 分支→优化器只接收可训练参数→正常 forward、loss、backward、step。

**Adapter 适配器**是小矩阵及相关配置。单独保存时还需记录基础模型版本、target modules、rank、alpha、分词器/模板等，不能认为它可装在任意基础模型上。

推理可保留适配器便于切换，或计算 `W'=W+ΔW` 后合并，省去单独增量分支。合并不是重新训练；相同输入、相同精度、关闭 dropout 时输出应在浮点误差内一致。量化基础权重的合并可能需要反量化/重新量化，不能直接套浮点公式就保证完全一致。

TIES 合并尝试减轻增量符号冲突，DARE 丢弃/重缩放部分增量，task arithmetic 做任务增量加减。这些方法有兼容性与效果限制，两个适配器相加不保证同时获得两种能力。

## 五、QLoRA 是在 LoRA 基础上进一步省基础权重空间

QLoRA 用低比特存储冻结的基础权重，LoRA 参数仍以较高精度训练。计算时按实现需要解码/反量化，不是“所有乘法、梯度都用 4 bit”。

- **NF4**：针对近似正态分布设计的非均匀 4-bit 编码；16 个码值不是均匀整数格点。
- **Double quantization**：进一步量化每块的缩放等量化常数，降低额外元数据开销。
- **Paged optimizer**：用分页方式管理优化器状态的内存峰值，可能增加传输开销；不是永不 OOM 的保证。

7B 权重按 4 bit 只算原始位数约 3.5 GB，但训练还要量化元数据、适配器、梯度、优化器、激活、缓存和临时空间。**“7B 在 6GB 显存必能训练”不成立**，序列长度与 batch 尤其重要。

原文 8B 全量微调的 Adam 开销算例也不完整：若两份优化器状态都 FP32，仅这两份就是 `8B×4×2=64 GB`（十进制），还未加权重、梯度和可能的主权重副本。不能把某一训练栈的估算当普遍固定配置。

## 六、原文 Build It：八步代码

### 步骤 1：低秩增量层

A为[in,r]、B为[r,out]；x@A@B得到[batch,out]增量，再乘alpha/r。nn.Parameter使张量登记为可训练参数；B为零保证初始增量为零。

````python
# 本块中文说明：低秩增量层
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    # 初始化本实例保存的配置和状态；self 指当前对象。
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    # 前向计算：根据输入和当前参数得到输出，本方法本身不更新权重。
    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
````

### 步骤 2：包装线性层

保存原linear，冻结其参数，forward计算原输出加LoRA输出。两分支输出尺寸必须相同。

````python
# 本块中文说明：包装线性层
class LinearWithLoRA(nn.Module):
    # 初始化本实例保存的配置和状态；self 指当前对象。
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    # 前向计算：根据输入和当前参数得到输出，本方法本身不更新权重。
    def forward(self, x):
        return self.linear(x) + self.lora(x)
````

### 步骤 3：注入模型

先冻结整网，再按名字匹配Linear并替换成包装层。应检查匹配结果是否正好目标层，避免子串误匹配或重复注入。

````python
# 本块中文说明：注入模型
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
````

### 步骤 4：统计参数

numel返回一个张量的元素个数；requires_grad决定是否计入trainable。总参数包括冻结基础与新增LoRA，不能用trainable比例直接推断显存比例。

````python
# 本块中文说明：统计参数
# 按张量元素数量统计总参数、可训练参数与冻结参数。
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
````

### 步骤 5：合并增量

在no_grad下把(A@B).T乘缩放加回原权重，再用原Linear替换包装层。不会再保留额外LoRA分支；合并前后需做输出一致性检查。

````python
# 本块中文说明：合并增量
# 将低秩增量合并到基础权重；行向量代码需注意转置。
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
````

### 步骤 6：量化教学模拟

按块找absmax并均匀缩放、取整、存int8，逆变换再乘scales。虽然函数名有NF4，这不是NF4码本或真实4-bit打包；网页与源码padding签名也不同。

````python
# 本块中文说明：量化教学模拟
# 注意：原函数名虽然叫 NF4，实际为均匀整数范围量化模拟。
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
````

### 步骤 7：真实梯度更新的小网络

优化器只接收requires_grad参数；前向→MSE损失→zero_grad清旧梯度→backward算梯度→step更新。这里确实写了反向传播，但不是LLM的token交叉熵SFT。

````python
# 本块中文说明：真实梯度更新的小网络
# 执行小网络训练循环，只更新允许求梯度的参数。
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
````

### 步骤 8：完整小网络演示

随机生成输入和类别，转one-hot，注入层0/2，训练后合并并返回参数/损失。随机标签实验只验证流程，不证明任务泛化或预训练模型微调质量。

````python
# 本块中文说明：完整小网络演示
# 运行该部分示例并展示结果；函数定义后还需调用才会产生输出。
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
````


## 七、实际运行与教学边界

本课需要 PyTorch。在本课目录执行 `python code/lora.py`；本笔记没有替你下载安装大型模型。源码 demo 用小型全连接网络和随机数据，并不是已经完成一个预训练 LLM 的 SFT。

运行可观察：注入前后可训练参数数目、训练 loss、量化重建误差、合并前后输出差、适配器保存/加载演示。数字随数据、版本和随机种子而变。随机输入配随机标签没有可学习的真实任务关系，训练 loss 下降不证明泛化，也不能用于得出“LoRA 和全量微调质量相同”。

原文函数名 `quantize_to_nf4` 实际做均匀缩放、round、裁到 [-8,7] 并存 int8，**不是真实 NF4，也没有把两个 4-bit 数打包成一个字节**。网页片段要求元素总数可被 block_size 整除；仓库版本额外处理了 padding，返回参数也不同，不能混合两版函数。量化器不等于已把训练循环接成真实 QLoRA。

原 demo 两层 LoRA r=8 有 `8(256+512)+8(512+512)=14336` 个可训练参数；原始网络含偏置共 399370 个，注入后总数 413706，可训练比例约 **3.47%**，不是正文声称的一概约 1%。

Use It 中 PEFT 负责适配器配置/注入，bitsandbytes 提供量化实现，TRL/Trainer 等组织训练。Unsloth、Axolotl、LLaMA-Factory、torchtune 是不同层次的性能优化或训练封装，不是必须叠装的整套依赖。具体配置依据模型与当前库版本；[PEFT 官方 LoRA 参考](https://huggingface.co/docs/peft/main/package_reference/lora)说明配置项和可选缩放策略。

## 八、原文五项练习怎样做才有意义

1. **rank 消融**：2/4/8/16/32/64，固定数据划分、训练步数和随机种子，记录验证 loss、可训练参数、显存/时间。不要以“翻倍 rank 应减半 loss”为定律。
2. **目标层对照**：只选 0、只选 2、只选 4、全部，解释这些是 Sequential 层名，不是实际 Transformer 的 Q/K/V 名字。
3. **量化误差**：block_size 32/64/128/256，计算 MSE、最大误差、相关系数；注明均匀量化模拟，不写成实测 NF4。
4. **多适配器切换**：在同一个固定基础模型上训练两套增量，记录配置，加载切换后比较输出；切换时避免把前一套增量重复叠加。
5. **合并一致性**：model.eval()，固定输入，比较最大差并使用 allclose 容差；再在同硬件预热计时，不能用一次运行证明提速。

交付物可直接是“微调实验卡”：问题是否需要训练、数据与验证划分、基础版本、rank/alpha/目标层、精度、batch/长度、指标、保存和合并检查。能解释每个矩阵尺寸及为何只更新小矩阵，才算掌握 LoRA，而不是只会改配置文件。


## 本次验证记录（2026-11-01）

仅核对公式与参数算术：原网399370，LoRA新增14336。未运行PyTorch训练、显存测量或合并一致性。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<07：进阶RAG——BM25、混合检索、重排与查询改写.md>) · [下一课](<09：函数调用与工具使用——让模型提出请求，让程序执行.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
