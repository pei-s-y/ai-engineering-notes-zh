# 21：Jamba混合架构——SSM状态、注意力比例与长上下文预算

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/docs/en.md)。原题：Jamba — Hybrid SSM-Transformer。

## 一、这一课解决什么问题？

纯attention的KV随上下文增长；SSM把历史压入固定形状状态。Jamba把两者混合，希望兼顾检索能力与缓存成本。本课是架构解释和标准库内存计算器，不训练Jamba或Mamba。

## 二、核心概念与推导

### 2.1 SSM的递推
h_t=Ā_t h_{t-1}+B̄_t x_t，y_t=C_t h_t。状态h的形状不随已经读过的T增长，所以decode持久状态在固定模型/batch下是O(1)关于T；处理T个token仍要O(T)工作。训练反传保留的激活并不因此O(1)。Mamba的selectivity由输入相关的Δ、B、C等实现，不能把A、B、C全部说成任意逐token新参数。

### 2.2 为什么混合
attention可按query读取历史KV；SSM用受容量约束的状态递推整合历史。前者也不保证“精确保留一切”，后者也不必然无法回忆。相同参数/数据/任务下的质量需要实验，不能只按上下文长度划一条绝对选型线。

### 2.3 Jamba的比例与MoE
attention:Mamba=1:7，就是每8层1A+7M，32层有4A+28M；1:3是1A+3M，32层8A；1:15是1A+15M，32层2A。原练习括号中的4M、14M是笔误。交替层MoE与A/M混合是两个轴：一个控制条件计算，一个控制序列状态，不能合并为同一个“稀疏度”。

### 2.4 两笔缓存公式
KV=2×B×L_A×H_kv×d_head×T×bytes。SSM粗估=B×L_M×d_state_channels×state_size×bytes，实际还可能有扩张inner维、卷积状态、更高精度累计；课件用hidden代替channels只是toy。每请求状态、共享模型权重、prefill激活必须分开。

### 2.5 家族与时点
[Jamba1.5论文](https://arxiv.org/html/2408.12570v1)介绍Mini约52B/12B、Large约398B/94B及ExpertsInt8；Large的量化多卡结果不能改写成单80GB卡BF16可装。论文强调混合架构与量化；本次核对不能支持课程“1.5新增RoPE”这一断言，因此不作为已证实差异。
[Mamba3论文](https://arxiv.org/abs/2603.15569)介绍更有表达力的递推、复数状态与MIMO；这是研究架构，不由此断言所有当前生产模型都采用它。

### 2.9 补充算例与理解检查

固定B1,T262144,L32,H_kv32,d128,BF16：纯attention KV128GiB，4个attention层hybrid KV16GiB。若双方都改GQA8，则分别32GiB与4GiB，仍为8倍KV减少。另按课程hidden4096、state16、28个SSM层，toy状态3.5MiB；它不是整个Jamba实际SSM状态精确值。权重52B×2bytes=104GB，光这项就超过80GB。

### 原课概念部分的公式与结构示例

$$
\begin{aligned}
h_t &= A h_{t-1} + B x_t \\
y_t &= C h_t
\end{aligned}
$$

~~~text
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
~~~

## 三、操作前准备

Python标准库即可，先掌握第12课GQA缓存公式、第14课MoE。以下计算器的默认batch为1，不含权重和激活。不要把课件32heads的假设模型直接当成某个官方checkpoint。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

本课原文没有独立Build It分节。概念部分的必要代码已保留在第二节，下一节另给动手检查；不会把架构算例冒充完整模型训练。

## 五、把代码运行起来：补充入口与检查

### 5.1 原仓库必要计算函数
下列摘自code/main.py，去掉打印大量表格的入口；后面的调用是笔记补充。函数ssm_state_bytes用hidden作简化状态通道，务必保留该假设。

~~~python
"""Jamba / Mamba-3 memory calculator — stdlib Python.

Computes KV cache, SSM state, and total attention-layer memory for a range
of hybrid configurations: pure Transformer, Jamba 1:7, 1:3, 1:15, and pure
SSM. Prints the comparison at 8k, 64k, 128k, 256k context.

Numbers are illustrative, not exact production memory budgets. The point is
to show why the hybrid ratio matters and where Jamba's 256k-on-80GB claim
comes from.
"""

from __future__ import annotations

from dataclasses import dataclass


BYTES_BF16 = 2
BYTES_FP8 = 1


@dataclass
class HybridConfig:
    name: str
    total_layers: int
    attn_layers: int
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    ssm_state_size: int


def kv_cache_bytes(cfg: HybridConfig, ctx: int, bytes_per_elem: int) -> int:
    return (2 * cfg.attn_layers * cfg.n_kv_heads * cfg.head_dim * ctx
            * bytes_per_elem)


def ssm_state_bytes(cfg: HybridConfig, bytes_per_elem: int) -> int:
    ssm_layers = cfg.total_layers - cfg.attn_layers
    return ssm_layers * cfg.hidden * cfg.ssm_state_size * bytes_per_elem
~~~

~~~python
# 补充：让纯attention与hybrid用相同GQA配置，做公平比较。
pure = HybridConfig('pure GQA8', 32, 32, 4096, 32, 8, 128, 16)
hybrid = HybridConfig('1A:7M GQA8', 32, 4, 4096, 32, 8, 128, 16)
for cfg in (pure, hybrid):
    print(cfg.name, 'KV GiB =', kv_cache_bytes(cfg, 262144, 2) / 2**30,
          'toy SSM MiB =', ssm_state_bytes(cfg, 2) / 2**20)
~~~

预期分别32GiB和4GiB，toy SSM分别0与3.5MiB。原main末尾概括的67GB/8.4GB与其实际32层、32KV头、256k设定不一致，应以函数公式结果为准；fmt_bytes实际按1024除，却输出GB标签，也应按GiB理解。本次具体执行记录见导航。

## 六、Use It：使用现成工具

原仓库main打印不同context和A/M比例的缓存表。真实部署要逐项核对后端支持的模型版本、SSM缓存分配、量化格式、prefill吞吐和实际任务质量。256k“支持长度”、实际检索表现、单卡“能装下”是三个不同命题；没有本机实测就不写速度或质量保证。



## 七、失败模式、原文边界与工程注意

1. 混合模型KV缩小约8倍，不等于所有运行内存或延迟都缩小8倍。
2. SSM单步状态不随T增长，训练激活、输入数据、输出缓存仍可能增长。
3. 原练习问2024年3月的Jamba为何没有用后来的Mamba2，时间顺序本身不成立，不能编造消融结论。
4. Jamba1.5 Large的94/398≈23.62%，DeepSeek37/671≈5.51%。差异要结合dense/SSM模块和专家数量、top-k解释；仅凭两个总数不能唯一反推MoE开销。
5. 原文“非推理模型绝不能…/当前最佳…”之类广泛结论都要限定模型、数据与时间，不作选型公理。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
你是混合架构选型助手。给定context分布、batch、检索/生成任务、GPU预算，比较纯Transformer、A/M混合、纯SSM。列出A层数、SSM层数、KV heads、state通道及dtype，分别计算权重、KV、SSM状态、激活。说明哪些是toy假设；给出同任务质量和端到端延迟验证方案。不凭最大长度宣称能准确利用上下文，也不凭KV预算宣称整模型可加载。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **复现8倍KV差异。** 用32L、4A对照；统一head数、dtype、batch、T。分别报告KV与KV+toy状态，后者比例不恰好8。
2. **1:3、1:15扫描。** 用8A/24M、2A/30M替代原文错误括号。KV随A层数线性变；求KV=SSM时令两公式相等得到T=L_M×channels×state/(2L_A H_kv d_head)，若解很短要说明toy状态省略项。
3. **阅读原Jamba混合比例消融。** 讨论A/M比例的质量和吞吐折中；原文要求解释“为什么不用Mamba2”是时间错位，应改为对后来架构另作对照，不能声称原论文已有该消融。
4. **398B/94B对照671B/37B。** 计算23.62%与5.51%，区分总/激活；若要算专家参数开销，补齐每类层和专家形状后再统计。
5. **复数与旋转。** 若复系数a=r(cosθ+i sinθ)，对二维实虚部相当于缩放r再旋转θ；输入相关θ可形成数据依赖的旋转。它与外部位置驱动的标准RoPE有数学联系，不能说所有SSM状态更新就是同一种位置编码。

## 十、术语与完成标准

|术语|含义|
|---|---|
|SSM|状态空间模型，通过递推更新固定形状状态|
|selective SSM|部分递推量依赖输入，选择保留信息|
|hybrid ratio|attention与SSM层的比例|
|SSM state|decode递推缓冲，与KV不同|
|MIMO|多输入多输出式状态计算结构|
|ExpertsInt8|专家权重采用INT8的特定部署方案|
|discretization|连续状态方程到离散递推的近似/变换|

完成检查：

- [ ] 能从1:7数出4A+28M。
- [ ] 能解释常数状态为什么不等于常数训练显存。
- [ ] 能公平对照32GiB/4GiB并补计SSM和权重。
- [ ] 能区分最大context长度与实际长程任务质量。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行同GQA8配置：纯attention KV32GiB、1:7混合4GiB，额外toy SSM3.5MiB。不是实际checkpoint总内存，未加载Jamba/Mamba。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[20：DeepSeek-V3架构解析——MLA、MoE、MTP与显存预算]] · 下一课：[[22：异步Hogwild推理——共享KV协作、模拟器与加速边界]]
