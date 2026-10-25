# 20：DeepSeek-V3架构解析——MLA、MoE、MTP与显存预算

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/20-deepseek-v3-walkthrough) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/docs/en.md)。原题：DeepSeek-V3 Architecture Walkthrough。

## 一、这一课解决什么问题？

把第14—19课的模块组合到一个真实模型上：为什么DeepSeek-V3有约671B主模型参数，每个token只激活约37B，却仍需要很大的权重存储？本课重点是读配置、分清计算与存储、推导MLA缓存；不是在4060上训练或完整加载V3。

## 二、核心概念与推导

### 2.1 先分清四笔账
total parameters决定权重常驻总量；active parameters描述一个token走过哪些模块；KV cache随上下文和batch增长；训练还要梯度、优化器、激活、通信缓冲。37B active不能直接用37B×dtype算整模型显存，也不等于每个token的全部FLOPs。

### 2.2 主干与专家
课程给出的主干为61层、hidden7168、前3层dense FFN，后58层MoE；每MoE层256 routed experts，top8，另有1 shared expert。SwiGLU单专家主要矩阵为3hd_ff。共享专家始终工作，路由专家按token选取；专家总数翻倍而top-k不变时，单token专家矩阵工作量大致不变，但router、总权重、分发通信可能增加。

### 2.3 MLA：缓存压缩潜变量
常规GQA每层每token保存2H_kv d_head个元素。MLA把内容相关K/V压到c_KV，再通过上投影生成需要的内容，并把位置相关的RoPE部分分离。实际紧凑逻辑缓存应计r_kv+d_rope，不只r_kv；不同推理后端可能展开、量化或对齐存储，公式必须注明布局。

### 2.4 矩阵吸收为什么可能
qᵀ(W_UK c)=(W_UKᵀq)ᵀc，因此内容分数可在压缩空间算；对值也有Σa_j W_UV c_j=W_UVΣa_j c_j。可利用结合律把固定线性映射与相邻投影融合，减少逐token解压。RoPE位置相关支路要单独处理，不能任意把所有旋转吸收到一个固定矩阵。

### 2.5 必须读实际配置
本次核对[官方config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)：q_lora_rank=1536，kv_lora_rank=512，qk_nope_head_dim=128，qk_rope_head_dim=64，v_head_dim=128，heads=128，tie_word_embeddings=false。原文“输入输出embedding权重绑定”和源码head_dim=7168//128=56，都不符合这些配置。共享input/output词表与MTP共享输出头是不同关系。

### 2.6 其余模块怎样配合
MTP提供额外未来token训练信号，可用于推测推理；参数口径要区分主模型约671B和含MTP附加参数的checkpoint。DualPipe解决流水/通信重叠，FP8解决部分计算和存储精度成本，各自有硬件与数值条件。FP8训练不表示优化器、master weights、所有归约均只占1byte。原文“完全没有额外目标”的辅助损失描述过度简化，本课不据此断言所有训练loss只有语言模型loss。

### 2.9 补充算例与理解检查

按B=1、T=131072、L=61、BF16且缓存为512维latent+64维RoPE：cache=61×131072×576×2=9,210,691,584bytes=8.578125GiB。只计512会得到7.625GiB，少计0.953125GiB。对照H_kv=8、d_head=128的GQA是30.5GiB，约3.56倍，不是笼统“4倍”。这些只计逻辑cache，不含权重和工作区。

### 原课概念部分的公式与结构示例

~~~text
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
~~~

~~~text
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
~~~

~~~text
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
~~~

## 三、操作前准备

架构与预算练习只用标准库，准备第14课参数统计和第12课KV公式。以下配置数值来自官方配置；课程原结构示例保留作对照。不要把修改rank的预算练习当成可以直接加载旧权重的配置修改。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

本课原文没有独立Build It分节。概念部分的必要代码已保留在第二节，下一节另给动手检查；不会把架构算例冒充完整模型训练。

## 五、把代码运行起来：补充入口与检查

### 5.1 补充：纠正维度后的MLA与专家计算器
以下是中文笔记补充，不是完整DeepSeek实现；无下载、无训练。注意最后的attention参数是无bias主要矩阵小计，不声称等于官方逐tensor精确统计。

~~~python
def v3_budget(ctx=131072, batch=1, kv_rank=512, bytes_per_value=2):
    # 官方主模型形状；rank变化仅用于预算敏感性分析。
    h, layers, heads = 7168, 61, 128
    q_rank, q_nope, q_rope, v_dim = 1536, 128, 64, 128
    expert_inner, routed, top_k, shared = 2048, 256, 8, 1
    # Q下投影与上投影、联合KV下投影、内容KV上投影、输出投影。
    attn_matrices = (
        h * q_rank
        + q_rank * heads * (q_nope + q_rope)
        + h * (kv_rank + q_rope)
        + kv_rank * heads * (q_nope + v_dim)
        + heads * v_dim * h
    )
    expert = 3 * h * expert_inner
    return {
        "mla_cache_bytes": batch * layers * ctx * (kv_rank + q_rope) * bytes_per_value,
        "gqa8_cache_bytes": batch * layers * ctx * 2 * 8 * 128 * bytes_per_value,
        "attn_matrix_params_per_layer": attn_matrices,
        "all_expert_params_per_moe_layer": (routed + shared) * expert,
        "active_expert_params_per_token_per_moe_layer": (top_k + shared) * expert,
    }

b = v3_budget()
print(b)
print("MLA GiB:", b["mla_cache_bytes"] / 2**30)
print("GQA8 GiB:", b["gqa8_cache_bytes"] / 2**30)
print("rank减半后的缓存比例:", v3_budget(kv_rank=256)["mla_cache_bytes"] / b["mla_cache_bytes"])
~~~

预期：MLA 8.578125GiB；GQA8 30.5GiB；attention主要矩阵187,105,280参数/层；rank从512到256后比例320/576≈0.555556，减少约44.44%，不是50%。用纸笔核对每一项单位，具体本次执行记录见导航。

### 5.2 怎么理解原仓库main.py
原仓库是参数/显存计算器，不会加载671B模型。源码把head_dim设为h//heads、RMSNorm重复计数、MLA缓存漏位置支路，又把MTP算进与主模型671B比较的总数，都会导致差异。不要靠调一个系数把小计强行凑成发布参数数目；应先统一tensor形状、权重共享和MTP口径。

## 六、Use It：使用现成工具

可在自己仓库的本课目录运行python code/main.py查看原计算表，再对照第五节更正。真实模型部署需核对checkpoint版本、FP8权重格式、MLA内核、专家并行及机器总显存；这里不提供未经实测的单卡启动保证。课程中的R1与后续V4推测只视为写作时的背景，不作为当前产品状态。



## 七、失败模式、原文边界与工程注意

1. GB=10^9bytes，GiB=2^30bytes，原文经常混称。
2. 671B FP8理论权重约671GB，BF16约1342GB；还没加量化元数据和运行时开销。不能按37B active推断可放入普通显卡。
3. 低秩压缩、专家稀疏、FP8、流水并行分别省不同东西，收益不能简单连乘。
4. MLA缓存必须计RoPE支路；把KV rank改256意味着架构/权重形状改变，需重新训练或专门适配。
5. 训练成本是特定数据、token数、硬件利用率、精度和通信下的结果，不能凭参数稀疏比例推出实际费用。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
你是架构预算审阅者。输入模型配置、checkpoint版本、主模型/MTP口径、上下文、batch、dtype与硬件。逐项计算主干、attention、共享与路由专家、输入输出头参数；对active和total分别说明定义。按实际MLA latent与RoPE缓存布局给出KV字节数，区分GB/GiB。列出权重、激活、优化器、通信和量化元数据。所有缺失形状明确标为假设，不用h/heads替代实际head维度，不保证硬件能加载。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **解释参数统计与发布值差异。** 先分开主模型与MTP，再检查Q/内容K/位置K/V维度、norm与输出头。交付按tensor形状的小计表，不能只写“约等于”。
2. **kv_rank512→256。** 用第五节函数算320/576，解释为什么有64维位置项，质量影响需要训练实验。
3. **专家256→512、top8不变。** 专家总矩阵近翻倍，active专家矩阵近不变；router输出维度及通信另算，不能宣称计算完全不变。
4. **推导吸收。** 写出第二节两条矩阵结合律，注明矩阵维度，并说明RoPE为何分离。
5. **FP8与BF16。** 算671GB与1342GB理论权重，再说明这不是训练总内存。固定token预算不因dtype本身自动减少，吞吐与精度影响需另测。

## 十、术语与完成标准

|术语|中文含义|
|---|---|
|MLA|多头潜在注意力，内容KV压缩与位置支路分离|
|latent rank|压缩空间维度，影响权重形状与缓存|
|routed/shared expert|按token选择的专家/每token共用专家|
|active parameters|一个token所用参数口径，区别总权重|
|MTP|多token预测的附加训练模块|
|FP8|8位浮点格式族，训练常混合更高精度状态|

完成检查：

- [ ] 能按官方配置读出五类MLA维度。
- [ ] 能算8.578125GiB缓存并说出前提。
- [ ] 能解释37B active为什么不等于显存只需37B权重。
- [ ] 能定位原计算器误差而不把它当实际模型实现。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行补充官方维度预算：MLA8.578125GiB、对照GQA8为30.5GiB，attention主要矩阵187105280/层，rank减半缓存比例5/9。未加载或训练DeepSeek-V3。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[19：DualPipe流水并行——双向调度、通信重叠与气泡模型]] · 下一课：[[21：Jamba混合架构——SSM状态、注意力比例与长上下文预算]]
