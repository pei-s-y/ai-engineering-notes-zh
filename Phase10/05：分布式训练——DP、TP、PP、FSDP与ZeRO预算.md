# 05：分布式训练——DP、TP、PP、FSDP与ZeRO预算

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/05-scaling-distributed) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/05-scaling-distributed/docs/en.md)。原题：Scaling: Distributed Training, FSDP, DeepSpeed。

## 一、这一课解决什么问题？

当模型状态、激活或训练时长超过单卡预算，采用不同切分。原课学习目标提到DDP/FSDP实训，但正文实际上是NumPy/标准库模拟器与估算，不会启动多GPU，也没有配置真正DeepSpeed。你的单4060可以先理解并运行CPU小模拟，无需买集群。

## 二、核心概念与推导

### 2.1 为什么不止权重
训练预算=参数+梯度+Adam一阶/二阶+可能的FP32主权重+激活+通信/临时workspace+碎片。2字节权重、2字节梯度、8字节Adam是12字节/参数；若加4字节master则16。源码不同段落对master不一致，必须列实际策略。GB是10^9字节，GiB是2^30。

### 2.2 三种并行
DP每卡完整模型、数据分片、同步梯度，提升吞吐但不缩每卡参数。TP切单层矩阵：输出列分片拼接，输入行分片求和，依赖互联。PP按层分stage，以microbatch流水，存在填充/排空气泡；优化吞吐也要管在途激活。

### 2.3 集合通信
all-reduce使每rank得到求和/平均；all-gather收集分片；reduce-scatter归约后分发各片。环all-reduce一般有reduce-scatter与all-gather两阶段，不能把N−1轮当完整过程。是否平均由调用/框架约定。

### 2.4 FSDP/ZeRO
ZeRO1分optimizer、2再分grad、3再分params；FSDP full shard与ZeRO3在高层上类似，不意味着所有API/调度完全一样。计算前all-gather临时完整参数，稳态1/N不代表峰值1/N。activation checkpointing只省激活，不能把已超过卡容量的105GB模型状态变成可放80GB。

### 2.5 混合精度与组合
BF16指数范围大、尾数少，FP16范围窄常需scaler；具体master/gradient策略依框架，不是必然84GB。DP×TP×PP只是无其他维度时的world_size分解；MoE还需专家分布与通信，总权重仍要存储，不是仅存active37B。

源文GPU价格、互联速率、训练成本与“线性加速”都需具体硬件/利用率/计费范围，不直接用于采购。

### 2.9 补充算例与理解检查

70B在2字节权重+2字节grad+8字节Adam假设下状态840GB；8卡full shard平均105GB/卡，光状态已超80GB，省激活也不解决。16卡平均52.5GB仍要加临时all-gather与激活。PP的等速toy中4stage、16microbatch气泡为3/(16+3)≈15.8%，源文18.75%是3/16近似而不是本模拟精确值。

### 原课概念部分的公式与结构示例



## 三、操作前准备

NumPy用于DP/TP矩阵；PP排程和预算函数可只用标准库定义后运行。不要先运行原Use It的大8192²矩阵，float64单张约537MB；下方缩到8×8。没有启动torchrun、GPU分布式或云实例。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 模拟数据并行

按batch余数给前rank多一个样本，但返回简单mean各rank loss/grad。若shard大小不等，它不等价于全样本均值，需按样本数加权；num_gpus>样本数会产生空分片。Use It model_fn的grad还与np.mean(output**2)差输出维度归一因子，不能把它当已检验解析梯度。

~~~python
import numpy as np

# 原平均在shard等大时才匹配全数据均值，空shard需拒绝。
def simulate_data_parallelism(data, num_gpus, model_fn):
    batch_size = len(data)
    shard_size = batch_size // num_gpus
    remainder = batch_size % num_gpus

    gpu_losses = []
    gpu_gradients = []

    offset = 0
    for gpu_id in range(num_gpus):
        extra = 1 if gpu_id < remainder else 0
        shard = data[offset:offset + shard_size + extra]
        offset += shard_size + extra

        loss, grad = model_fn(shard)
        gpu_losses.append(loss)
        gpu_gradients.append(grad)

    avg_loss = np.mean(gpu_losses)
    avg_gradient = np.mean(gpu_gradients, axis=0)

    return avg_loss, avg_gradient
~~~

### 4.2 模拟张量并行

按W输出列切，每片算X@Wshard后concatenate。小误差是浮点性质，不保证不同内核逐位相同；输出维必须整除卡数，row切另需sum，不是同一拼法。

~~~python
def simulate_tensor_parallelism(input_data, weight_matrix, num_gpus):
    d_in, d_out = weight_matrix.shape
    assert d_out % num_gpus == 0, f"d_out {d_out} not divisible by num_gpus {num_gpus}"
    shard_size = d_out // num_gpus

    partial_results = []
    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = start + shard_size
        weight_shard = weight_matrix[:, start:end]

        partial = input_data @ weight_shard
        partial_results.append(partial)

    full_output = np.concatenate(partial_results, axis=-1)

    direct_output = input_data @ weight_matrix
    error = np.abs(full_output - direct_output).max()

    return full_output, error
~~~

### 4.3 模拟流水

先排所有forward，再倒排backward，是简化flush式排程，不是1F1B。num_layers//num_stages不能整除时会丢层；0microbatch会max空集，需参数检查。timeline时间是抽象tick不是毫秒。

~~~python
# 抽象flush式forward/backward调度，不是1F1B实装。
def simulate_pipeline_parallelism(num_layers, num_stages, num_microbatches):
    layers_per_stage = num_layers // num_stages

    timeline = {}
    clock = 0

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            start_time = max(
                timeline.get((stage, mb - 1, "fwd"), (0, 0))[1] if mb > 0 else 0,
                timeline.get((stage - 1, mb, "fwd"), (0, 0))[1] if stage > 0 else 0,
            )
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "fwd")] = (start_time, end_time)

    last_fwd_end = max(v[1] for v in timeline.values())

    for mb in range(num_microbatches - 1, -1, -1):
        for stage in range(num_stages - 1, -1, -1):
            deps = [last_fwd_end]
            if mb < num_microbatches - 1 and (stage, mb + 1, "bwd") in timeline:
                deps.append(timeline[(stage, mb + 1, "bwd")][1])
            if stage < num_stages - 1 and (stage + 1, mb, "bwd") in timeline:
                deps.append(timeline[(stage + 1, mb, "bwd")][1])
            start_time = max(deps)
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "bwd")] = (start_time, end_time)

    total_time = max(v[1] for v in timeline.values())
    compute_time = num_microbatches * num_stages * layers_per_stage * 2
    bubble_fraction = 1.0 - compute_time / (total_time * num_stages)

    return timeline, total_time, bubble_fraction
~~~

### 4.4 训练内存估算器

按sharding分状态、保留未分激活。默认激活=params×precision×.5是启发式占位，不是精确模型公式。SGD固定4字节状态不适用于所有实现，unknown optimizer静默0也应改报错。fits_on_80gb只是估算，不是保证可跑。

~~~python
# 启发式预算，不含全部峰值；明确master与activation假设。
def memory_calculator(
    params_billions,
    precision_bytes=2,
    optimizer="adam",
    num_gpus=1,
    sharding="none",
    sequence_length=2048,
    batch_size_per_gpu=1,
    hidden_dim=None,
    num_layers=None,
):
    params = params_billions * 1e9

    weight_memory = params * precision_bytes

    if optimizer == "adam":
        optimizer_memory = params * 4 * 2
    elif optimizer == "sgd":
        optimizer_memory = params * 4
    else:
        optimizer_memory = 0

    gradient_memory = params * precision_bytes

    total_no_activation = weight_memory + optimizer_memory + gradient_memory

    if hidden_dim and num_layers:
        activation_per_layer = (
            sequence_length * batch_size_per_gpu * hidden_dim * precision_bytes * 4
        )
        activation_memory = activation_per_layer * num_layers
    else:
        activation_memory = params * precision_bytes * 0.5

    if sharding == "fsdp" or sharding == "zero3":
        weight_memory /= num_gpus
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero2":
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero1":
        optimizer_memory /= num_gpus

    per_gpu_total = weight_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "params_billions": params_billions,
        "weights_gb": weight_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "per_gpu_total_gb": per_gpu_total / 1e9,
        "total_across_gpus_gb": per_gpu_total * num_gpus / 1e9,
        "fits_on_80gb": per_gpu_total / 1e9 <= 80,
        "num_gpus": num_gpus,
        "sharding": sharding,
    }
~~~

### 4.5 混合精度对照

fp16_with_master把master单算，mixed_bf16却没master；因此不能泛称同样master条件下BF16省25%。比较同一真实实现策略、activation和通信峰值，解释假设后再读数值。

~~~python
# BF16分支不含master，不能和另一分支直接泛化比较。
def mixed_precision_comparison(params_billions):
    params = params_billions * 1e9

    fp32_weights = params * 4
    fp32_optimizer = params * 4 * 2
    fp32_gradients = params * 4
    fp32_total = fp32_weights + fp32_optimizer + fp32_gradients

    fp16_weights = params * 2
    fp16_master = params * 4
    fp16_optimizer = params * 4 * 2
    fp16_gradients = params * 2
    fp16_total = fp16_weights + fp16_master + fp16_optimizer + fp16_gradients

    mixed_weights = params * 2
    mixed_optimizer = params * 4 * 2
    mixed_gradients = params * 2
    mixed_total = mixed_weights + mixed_optimizer + mixed_gradients

    return {
        "fp32_total_gb": fp32_total / 1e9,
        "fp16_with_master_gb": fp16_total / 1e9,
        "mixed_bf16_gb": mixed_total / 1e9,
        "savings_vs_fp32": 1 - mixed_total / fp32_total,
    }
~~~

## 五、把代码运行起来：补充入口与检查

纯标准库先核对排程与模型状态：
~~~python
_, total, bubble = simulate_pipeline_parallelism(32, 4, 16)
assert abs(bubble - 3/19) < 1e-9
print("抽象总时长与气泡", total, bubble)
budget = memory_calculator(70, num_gpus=8, sharding="fsdp")
assert budget["weights_gb"] == 17.5
assert budget["optimizer_gb"] == 70.0
assert budget["gradients_gb"] == 17.5
print(budget)
~~~
源码默认激活还加70GB，因此此budget总数不是只有105GB。NumPy小矩阵检查（本次未执行）：
~~~python
# x = np.arange(16.0).reshape(2, 8)
# W = np.arange(64.0).reshape(8, 8)
# output, error = simulate_tensor_parallelism(x, W, 2)
# assert np.allclose(output, x @ W)
~~~
真DDP验证需独立环境与多个rank、同步/样本数/失败恢复测试，不能由此toy替代。

## 六、Use It：使用现成工具

run_all_demos会顺次打印DP梯度范数、TP误差、PP抽象气泡、内存与精度表；它没创建GPU进程。必须最后手动调用run_all_demos()才有输出。原大矩阵可在自己的练习副本缩小后运行，不修改原仓库。实际框架应按当前PyTorch FSDP/DeepSpeed文档和设备能力配置，本笔记不代启集群。

~~~python
def run_all_demos():
    print("=" * 70)
    print("DATA PARALLELISM SIMULATION")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.randn(64, 32)
    weight = np.random.randn(32, 16)

    def model_fn(batch):
        output = batch @ weight
        loss = np.mean(output ** 2)
        grad = 2 * batch.T @ (batch @ weight) / len(batch)
        return loss, grad

    for n_gpus in [1, 2, 4, 8]:
        loss, grad = simulate_data_parallelism(data, n_gpus, model_fn)
        print(f"  {n_gpus} GPUs: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}")

    print()
    print("=" * 70)
    print("TENSOR PARALLELISM SIMULATION")
    print("=" * 70)

    x = np.random.randn(4, 8192)
    W = np.random.randn(8192, 8192)

    for n_gpus in [1, 2, 4, 8]:
        output, error = simulate_tensor_parallelism(x, W, n_gpus)
        print(f"  {n_gpus} GPUs: output_shape={output.shape}, max_error={error:.2e}")

    print()
    print("=" * 70)
    print("PIPELINE PARALLELISM SIMULATION")
    print("=" * 70)

    for n_mb in [1, 4, 8, 16, 32]:
        _, total_t, bubble = simulate_pipeline_parallelism(32, 4, n_mb)
        print(f"  {n_mb:2d} micro-batches: total_time={total_t:4d}, bubble={bubble:.1%}")

    print()
    print("=" * 70)
    print("MEMORY CALCULATOR")
    print("=" * 70)

    configs = [
        (7, "none", 1),
        (7, "fsdp", 8),
        (70, "none", 1),
        (70, "fsdp", 8),
        (70, "fsdp", 16),
        (405, "fsdp", 64),
        (405, "fsdp", 128),
    ]

    print(f"  {'Model':>8} {'Sharding':>8} {'GPUs':>5} {'Per-GPU':>10} {'Fits 80GB':>10}")
    print("  " + "-" * 50)
    for params, shard, gpus in configs:
        result = memory_calculator(params, num_gpus=gpus, sharding=shard)
        fits = "Yes" if result["fits_on_80gb"] else "No"
        print(f"  {params:>6}B {shard:>8} {gpus:>5} {result['per_gpu_total_gb']:>8.1f}GB {fits:>10}")

    print()
    print("=" * 70)
    print("MIXED PRECISION COMPARISON")
    print("=" * 70)

    for params_b in [7, 13, 70, 405]:
        result = mixed_precision_comparison(params_b)
        print(f"  {params_b}B: FP32={result['fp32_total_gb']:.0f}GB, "
              f"Mixed BF16={result['mixed_bf16_gb']:.0f}GB, "
              f"Savings={result['savings_vs_fp32']:.0%}")
~~~

## 七、失败模式、原文边界与工程注意

1. state-only已超卡容量时checkpoint激活救不了。
2. 参数少于多卡吞吐划算阈值时通信可能更慢，不必为练习采购。
3. 不均DP batch需权重平均，梯度累积也需按有效token数正确归一。
4. 用随机数/RNG、dropout、BatchNorm等时累积与大batch不一定严格等价。
5. master/精度/offload含义不一致会低估显存，80GB门槛不等于OOM免疫。
6. 训练成本是否含数据、实验、研发、后训练各家不同，不能拿报道总数直接反推效率。
7. NVMe/CPU offload受总RAM、每节点分配、带宽和峰值限制，固定30%～50%减速非保证。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
输入模型/序列/batch/精度和硬件，分项列weights、grad、optimizer、master、activation、临时峰值与余量；选择DP/TP/PP/ZeRO/FSDP，说明world_size、通信和调度。估吞吐成本时给利用率/价格来源与区间，并用实际profile验证，不能把模拟值写成多卡实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 加activation checkpointing预算，定义保存边界及重算范围；不要用统一33%替所有网络，且不减少模型状态。
2. 实现1F1B，对比4stage/8microbatch气泡与峰值激活；同样时长假设下吞吐未必变、内存可下降。
3. 梯度累积K步再同步，正确按样本/token归一，检查一次update与等效大batch；记录no_sync、随机层等条件。
4. 给定卡价/总token估GPU-hours与成本，原A100$2/H100$3.5只是题设，不是当前报价；分清报道成本口径。
5. 加CPU/NVMe offload，给RAM容量/带宽/节点数，估70B可行性，不预先保证4卡训练和固定减速。

## 十、术语与完成标准

DP：数据并行；TP：张量并行；PP：流水并行；FSDP：全分片数据并行；ZeRO：分状态冗余优化；all-gather/reduce-scatter：收集/归约分发；bubble：气泡空闲。

完成检查：

- 能算12与16字节/参数两种假设。
- 能解释列切拼接与行切求和。
- 能说明checkpoint救不了105GB状态。
- 能区分模拟tick与真实吞吐。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行纯标准库排程，总抽象时长304，气泡3/19≈0.157895；70B/8卡toy FSDP每卡权重17.5GB、优化器70GB、梯度17.5GB，连同简化激活共175GB。未运行真实分布式或NumPy TP。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 5 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[04：从零搭建Mini GPT——因果注意力、前向计算与真实训练]] · 下一课：[[06：指令微调SFT——聊天格式、响应Mask与真实梯度]]
