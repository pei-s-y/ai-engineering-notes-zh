# 19：DualPipe流水并行——双向调度、通信重叠与气泡模型

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/19-dualpipe-parallelism) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/19-dualpipe-parallelism/docs/en.md)。原题：DualPipe Parallelism。

## 一、这一课解决什么问题？

MoE集群需要协调attention、专家计算、dispatch/combine通信及forward/backward。DualPipe通过调度重叠改善空闲，但原课模拟只有经验公式，没有真实事件依赖/通信内核/多GPU，不能据此量化真实节省GPU-hours。

## 二、核心概念与推导

### 2.1 PP基础
微批前向依赖前stage输出，反向依赖后stage梯度。GPipe会对多个microbatch进行流水，不是原文说的一次只跑一个。1F1B交错前后向降低在途激活，Zero Bubble将反向input-grad与weight-grad拆开安排。

### 2.2 DualPipe思路
将attention、all-to-all dispatch、专家MLP、combine细分，双向流安排互相可重叠的计算/通信。每时隙两个标记不表示同一GPU两份完整计算可无资源冲突同时免费执行，需SM/网络/stream和依赖模型。暖机/排空、负载不均与链路仍有气泡。

### 2.3 固定工作量的比较
简单1F1B toy气泡=(P−1)/(M+P−1)，M增大时比例下降，不是“1F1B气泡随M越来越大”。气泡时长、比例、稳定阶段空闲和整体利用率是不同量。不同公式若forward/backward单位不一致，不能直接把百分比当吞吐提升。

### 2.4 DualPipeV官方口径纠错
[DeepSeek官方DualPipe表](https://github.com/deepseek-ai/DualPipe)在相同PP stage数下，DualPipe与DualPipeV每设备参数都是2×，设备数分别PP与PP/2；V形是cut-in-half调度。原文“V每设备1份、直接减半显存”不是该比较口径。要同时说明stage、device、chunk与参数副本数。

### 2.5 与EP/FSDP结合
MoE专家分布、PP层划分、DP/FSDP状态、all-to-all网络共同决定峰值。更多参数不只看active，需要专家权重位置/复制和梯度状态。原GPU95%利用、245k/340k hours、固定1k卡门槛与调试一周是来源估计，不能归因或保证。

### 2.9 补充算例与理解检查

P8/M16按源码1F1B=7/23≈30.43%；toy DualPipe=((8−1)//2)/(3×16+7)=3/55≈5.45%。这只是两个公式输出，不是实测GPU从69.6%涨到94.5%。M从16增64时1F1B变7/71≈9.86%，直接反驳“随微批增大比例增加”。

### 原课概念部分的公式与结构示例

~~~text
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
~~~

~~~text
bubble_1F1B = (P - 1) * forward_chunk_time
~~~

## 三、操作前准备

无Build It独立节，本课以Learn+公式模拟为主。下面内嵌仓库函数，标准库即可；P/M必须正，奇数P或不支持微批组合在toy没拒绝，但真实DualPipe有调度约束。不会启动分布式集群。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

本课原文没有独立Build It分节。概念部分的必要代码已保留在第二节，下一节另给动手检查；不会把架构算例冒充完整模型训练。

## 五、把代码运行起来：补充入口与检查

原仓库核心经验公式，名称只表示教学对照，不是严格DualPipe性能模拟：
~~~python
"""Pipeline schedule simulator — 1F1B vs Zero Bubble vs DualPipe vs DualPipeV.

Teaching tool. Counts pipeline bubbles per schedule for given (P, micro_batches).
Outputs:
  - bubble fraction per schedule at fixed (P, micro_batches)
  - scaling of bubbles as micro_batches grows

Not a production simulator. Forward/backward chunk costs are unit-normalized.
Comm costs are modeled as overlap windows, not full kernel models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ScheduleStats:
    name: str
    stable_bubble_frac: float
    scales_with_micro_batches: bool
    param_copies: int
    comm_overlap: str


def bubble_1f1b(P: int, M: int) -> float:
    """1F1B: warmup phase has (P-1) forward slots without backward overlap.
    Cooldown mirrors. Stable phase has zero bubble per rank per micro-batch,
    but warmup/cooldown bubble is (P-1) forward + (P-1) backward chunks per
    rank, out of 2 * M + 2 * (P - 1) chunks total.
    """
    total = 2 * M + 2 * (P - 1)
    bubble = 2 * (P - 1)
    return bubble / total


def bubble_zero_bubble(P: int, M: int) -> float:
    """Zero Bubble (Qi 2023) splits backward into B + W. The W part can fill
    the 1F1B bubble. Approximate residual bubble is (P - 1) / 2 chunks of
    warmup plus the same cooldown, out of 3 * M + 2 * (P - 1) sub-chunks.
    """
    total = 3 * M + 2 * (P - 1)
    bubble = (P - 1)
    return bubble / total


def bubble_dualpipe(P: int, M: int) -> float:
    """DualPipe injects micro-batches from both ends of the pipeline. Stable
    phase bubble is zero. Warmup/cooldown has fixed bubble independent of M.
    """
    total = 3 * M + (P - 1)
    bubble = (P - 1) // 2
    return bubble / total


def bubble_dualpipev(P: int, M: int) -> float:
    """DualPipeV uses a V-shape schedule on a single parameter copy. Its
    bubble is slightly larger than DualPipe's at the benefit of halving
    memory. Approximate as 1.2x DualPipe bubble."""
    return bubble_dualpipe(P, M) * 1.2


def summarize(P: int, M: int) -> List[tuple[str, float, int, str]]:
    return [
        ("1F1B",       bubble_1f1b(P, M),        1, "minimal"),
        ("Zero Bubble", bubble_zero_bubble(P, M), 1, "partial"),
        ("DualPipe",   bubble_dualpipe(P, M),    2, "full"),
        ("DualPipeV",  bubble_dualpipev(P, M),   1, "partial"),
    ]


def gpu_hours_recovered(P: int, M: int, total_gpu_hours: float) -> dict:
    b1 = bubble_1f1b(P, M)
    bd = bubble_dualpipe(P, M)
    recovered = (b1 - bd) * total_gpu_hours
    return {
        "1F1B_bubble_frac": b1,
        "DualPipe_bubble_frac": bd,
        "recovered_gpu_hours": recovered,
    }



~~~
运行检查：
~~~python
assert abs(bubble_1f1b(8,16) - 7/23) < 1e-12
assert bubble_1f1b(8,64) < bubble_1f1b(8,16)
assert bubble_dualpipe(8,64) < bubble_dualpipe(8,16)
print(summarize(8,16))
~~~
其中DualPipeV=1.2×DualPipe、param_copies=1是作者人为设定且与官方比较表不一致，保留用于辨析，不采信为真实算法参数。实际恢复GPU-hours必须有同workload/有效吞吐/通信trace，不能直接差气泡比乘总预算当确证。

## 六、Use It：使用现成工具

python code/main.py无参数会打印P8/M16对照、M/P扫描与恢复小时；它不解析原练习那种schedule=命令参数，也不画时间线。要手画P4/M8，应另列每chunk输入/输出/反向依赖与资源，验证无重叠冲突，不能把源码标语当排程证明。真实官方示例需要PyTorch/设备与自定义overlapped_forward_backward，本文不执行。



## 七、失败模式、原文边界与工程注意

1. 公式模拟不验证调度合法性、通信瓶颈或显存峰值。
2. 气泡比例会随M降低，原表“1F1B随M增长”有误导。
3. DualPipeV的1×副本说法需按官方stage/device口径修正。
4. 两方向计算标记并不等于两套矩阵免费同时算。
5. 2×参数副本与EP分片需要节点/RAM、optimizer逐项算。
6. GPU busy率不等于MFU，更不是推导成本因果证据。
7. “bubble-free”通常带条件和阶段，不能理解所有时刻无空闲。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
列PP stage/rank、M、layer/expert分片、副本和forward/input-grad/weight-grad/comm时长。构建资源受限时间线与峰值激活，验证依赖再profile。按官方口径比较DualPipe/V/1F1B，不用经验比例替多卡吞吐/成本或宣称无气泡。
~~~

## 九、Exercises：保留原课练习与验收方法

1. P8/M16比较toy输出，若换GPU-hours/百万token需实测每token耗时与卡数，公式不能独立给出。
2. P4/M8手画合法依赖/双向chunk，标暖机/稳定/排空及资源，不只写F/F两个字。
3. 读V3图示dispatch与计算窗口，说明网络与SM何时可重叠，不必假定全隐藏。
4. dense70B/P8与MoE671B/P16预算，列EP大小/复制策略/optimizer；不能仅active比宣称MoE复制成本小。
5. 与Chimera按原论文逐项对照通信重叠与调度/内存条件，不使用本toy当证据。

## 十、术语与完成标准

PP：按层流水；EP：专家分布；microbatch：微批；bubble：依赖/资源导致空闲；dispatch/combine：专家发送/返回；1F1B：前反交错；DualPipeV：V形切半调度。

完成检查：

- 能区分气泡时长与比例。
- 能从公式看M增大比例下降。
- 能正确说stage/device/副本口径。
- 能说明toy不是真多卡模拟。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行经验公式：P8/M16时1F1B为7/23≈.304348，DualPipe toy为3/55≈.054545；增加M时下降。不是DualPipe真实调度或GPU-hours测量。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 0 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[18：多Token预测MTP——顺序模块、共享参数与辅助损失]] · 下一课：[[20：DeepSeek-V3架构解析——MLA、MoE、MTP与显存预算]]
