# 14：开源模型架构解析——RMSNorm、RoPE、GQA、MoE与配置预算

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/docs/en.md)。原题：Open Models: Architecture Walkthroughs。

## 零、初学者先读：读模型配置时，怎样把名字变成结构

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

不同模型并不是同一份代码只改参数数量。配置像装配清单：有多少层、每层多宽、怎样计算注意力、词表多大、哪些参数共享。先读清字段再预算。

### 0.2 本课术语先翻成白话

- hidden_size=每个token中间向量长度

- num_layers=模块重复层数

- head=注意力分组

- GQA=若干查询头共享一组KV头

- RoPE=用旋转方式引入位置信息

- RMSNorm=按均方根缩放特征

- MoE=按输入选择部分专家计算。

### 0.3 用小数字走一遍

词表100、向量长8，输入embedding有100×8=800个数。输出若再有一张独立8×100矩阵，再加800；如果明确共享同一张权重，就不能重复计两次。层数翻倍只翻倍重复层部分，不一定翻倍整个模型。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先把config字段翻成中文，再读head_dim和attention_params_per_layer，最后analyze汇总。普通attention里d/head只是常见设定；某些架构有显式head_dim，应读实际字段。GQA省KV头数，但查询头仍可更多。

前置阅读：[第04课](<04：从零搭建Mini GPT——因果注意力、前向计算与真实训练.md>)；[第05课](<05：分布式训练——DP、TP、PP、FSDP与ZeRO预算.md>)；[第12课](<12：推理优化——KV缓存、连续批处理、分页与前缀复用.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 8个查询头共享2组KV，是8组还是2组持久KV？

**参考答案：** 2组，前提是该实现确实采用此布局。

**问题2：** 模型参数量能直接告诉训练显存吗？

**参考答案：** 不能，还要梯度、优化器和激活等。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

从GPT形状读懂Llama/Mistral/Mixtral/Gemma/Qwen/DeepSeek，重点是具体配置与计算，而非背最新排行榜。原文若将所有模型归成六旋钮，属于方便入门的框架，不是架构穷尽分类。只读config也不能精确推出全部激活峰值/性能。

## 二、核心概念与推导

### 2.1 六个设计点
RMSNorm=x/√mean(x²)+ε再乘gamma，不减均值；少的是归约/逐元素操作，不是“少一次matmul”。RoPE成对旋转Q/K，位置可计算不等于无需训练就能任意外推；YaRN等扩展还需实际长文验证。

SwiGLU应含输出投影：(SiLU(xWgate)⊙xWup)Wdown。两张升维+一张降维约3df，标准两层约2df；同预算时f常取8d/3但具体checkpoint不同。GeGLU换门控激活。

MHA有独立KV；GQA多Q共享若干KV；MQA只一组KV；MLA压成latent并按实现吸收投影，不能直接当GQA另一头数。MoE router选top-k专家，加共享/稠密部分，total与active不同，但全部专家权重仍需存储。pre-norm常见不表示所有模型只一种规范；Gemma2还有post norm。

### 2.2 配置字段
hidden_size、intermediate_size、L、Hq、Hkv、head_dim、V、tie_word_embeddings、rope_theta、eps、max_position_embeddings、layer_types/sliding_window都要读。显式head_dim优先，不总为hidden/Hq；max_position_embeddings是配置上限，不足证明训练或有效能力。

### 2.3 计数公式
无bias GQA attention=2d(Hq dh)+2d(Hkv dh)，FFN SwiGLU=3df。untied head再加Vd，Norm按实际次数。稠密/共享专家、路由bias、QKV bias、MLA额外维度需单独加；源码计算器是有假设的估算，某模型匹配不证明所有模型精确。

KV字节=2×Hkv×dh×dtype字节×Σ各层缓存token，局部层用min(T,W)。activation常用B T d L bytes只是边界主项，不是完整峰值；原B1/T8192/d4096/L32/BF16这个乘积是2GiB而不是8GB，真实额外项另计。

### 2.4 原模型表的修正
原Llama3与3.1上下文混用，128k应明确3.1等版本；依据[Meta官方3.1模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)。Spark-X2.5在本篇只作为课程自带配置算例，未独立验证模型卡，不据此推荐。具体405B层/头在课程不同表不一致，应先查该checkpoint。

### 2.9 补充算例与理解检查

Llama式8B算例d4096、Hq32、Hkv8、dh128、L32、V128256，untied head比只计embedding多525336576参数；按文档核心公式可得8030261248。128Ki上下文KV=17179869184字节=16GiB≈17.18GB。能算出来不代表权重和KV装得进你的卡，更不代表延迟合格。

### 原课概念部分的公式与结构示例

$$
\mathop{\mathrm{RMSNorm}}(x) = \frac{x}{\sqrt{\mathop{\mathrm{mean}}(x^2) + \epsilon}} \cdot \gamma
$$

~~~text
q_rotated = rotate(q, angle(pos))
k_rotated = rotate(k, angle(pos))
score = q_rotated . k_rotated
~~~

$$
\mathop{\mathrm{SwiGLU}}(x) = (xW_1) \cdot \mathop{\mathrm{sigmoid}}(xW_1) \cdot xV
$$

~~~text
router_logits = xW_r
indices, weights = top_k(router_logits, k=2)
output = sum_i weights[i] * expert[indices[i]](x)
~~~

~~~text
{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 131072,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5,
  "vocab_size": 128256
}
~~~

~~~text
activation_mem ~ batch_size * seq_len * hidden_size * num_layers * bytes_per_element
~~~

~~~text
kv_cache = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * bytes_per_element
~~~

~~~text
kv_cache = 2 * num_kv_heads * head_dim * bytes_per_element
           * (full_layers * max_seq_len + sliding_layers * min(W, max_seq_len))
~~~

## 三、操作前准备

本课核心计算器是标准库，不下载权重。原Build It只给config没函数，下面内嵌仓库核心。必须补activation/norm/tie字段，否则default GELU/LayerNorm会把Llama算错。MoE/MLA预算依实现仅作近似。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 先读字段再运行计算器

原文输入config只是字段示意，脚本实际还带norm/activation等额外字段。第五节提供核心analyze及相关函数，保留原实现和边界说明；不是加载模型。

~~~python
config = {
    "hidden_size": 4096, "intermediate_size": 14336,
    "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "vocab_size": 128256,
    "max_position_embeddings": 131072,
}
~~~

## 五、把代码运行起来：补充入口与检查

下面摘自原仓库code/main.py的核心计算器，省去重复模型配置和打印排版；没有独立脚本附件：
~~~python
from dataclasses import dataclass

@dataclass
class Breakdown:
    name: str
    total_params: int
    active_params: int
    mlp_params_per_layer: int
    attn_params_per_layer: int
    embedding_params: int
    kv_cache_bytes_bf16: int
    kv_cache_all_full_bytes_bf16: int
    full_attention_layers: int
    mlp_ratio: float
    attention_scheme: str
    verdict: str


def attention_scheme(config: dict) -> str:
    if config.get("attention") == "mla":
        return "MLA"
    q_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    if kv_heads == 1:
        return "MQA"
    if kv_heads == q_heads:
        return "MHA"
    return f"GQA ({q_heads}/{kv_heads})"


def head_dim(config: dict) -> int:
    return config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])


def attention_params_per_layer(config: dict) -> int:
    h = config["hidden_size"]
    q_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    hd = head_dim(config)
    if config.get("attention") == "mla":
        lora = config.get("kv_lora_rank", 512)
        return h * h + 2 * (h * lora + lora * q_heads * hd) + h * h
    q_proj = h * (q_heads * hd)
    kv_proj = 2 * h * (kv_heads * hd)
    out_proj = (q_heads * hd) * h
    gate = h * q_heads if config.get("attn_output_gate") == "headwise" else 0
    return q_proj + kv_proj + out_proj + gate


LAYER_TYPES = ("full_attention", "sliding_attention")


def layer_types(config: dict) -> list[str]:
    n_layers = config["num_hidden_layers"]
    kinds = config.get("layer_types")
    if kinds is None:
        return ["full_attention"] * n_layers
    # Same rule HuggingFace applies when it loads a config: one entry per layer.
    if len(kinds) != n_layers:
        raise ValueError(
            f"layer_types has {len(kinds)} entries but num_hidden_layers is {n_layers}"
        )
    unknown = sorted(set(kinds) - set(LAYER_TYPES))
    if unknown:
        raise ValueError(f"unsupported layer_types: {', '.join(unknown)}")
    if "sliding_attention" in kinds and not config.get("sliding_window"):
        raise ValueError("layer_types has sliding_attention layers but no sliding_window")
    return list(kinds)


def cached_tokens_per_layer(config: dict) -> list[int]:
    max_seq = config["max_position_embeddings"]
    window = config.get("sliding_window")
    return [
        min(max_seq, window) if kind == "sliding_attention" and window else max_seq
        for kind in layer_types(config)
    ]


def mlp_params(h: int, ff: int, activation: str) -> int:
    if activation in ("swiglu", "geglu"):
        gate_and_up = 2 * h * ff
        down = ff * h
        return gate_and_up + down
    return 2 * h * ff


def mlp_params_per_layer(config: dict) -> int:
    return mlp_params(
        config["hidden_size"],
        config["intermediate_size"],
        config.get("activation", "gelu"),
    )


def layer_norm_params_per_layer(config: dict) -> int:
    h = config["hidden_size"]
    norms = config.get("norms_per_layer", 2)
    if config.get("norm") == "rmsnorm":
        return norms * h
    return norms * 2 * h


def analyze(name: str, config: dict) -> Breakdown:
    h = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    activation = config.get("activation", "gelu")
    dense_ff = config["intermediate_size"]

    emb = vocab * h
    attn = attention_params_per_layer(config)
    dense_mlp = mlp_params(h, dense_ff, activation)
    norm = layer_norm_params_per_layer(config)
    final_norm = h if config.get("norm") == "rmsnorm" else 2 * h
    lm_head = 0 if config.get("tie_word_embeddings", False) else vocab * h

    if config.get("moe"):
        n_experts = config["num_experts"]
        experts_per_tok = config["experts_per_token"]
        shared = config.get("shared_experts", 0)
        moe_ff = config.get("moe_intermediate_size", dense_ff)
        expert_mlp = mlp_params(h, moe_ff, activation)
        first_dense = config.get("first_dense_layers", 0)
        n_moe_layers = n_layers - first_dense
        router = h * n_experts

        dense_block_params = attn + dense_mlp + norm
        moe_block_params = (
            attn
            + expert_mlp * n_experts
            + expert_mlp * shared
            + router
            + norm
        )
        active_moe_block = (
            attn
            + expert_mlp * (experts_per_tok + shared)
            + router
            + norm
        )

        total = (
            emb
            + first_dense * dense_block_params
            + n_moe_layers * moe_block_params
            + final_norm
            + lm_head
        )
        active = (
            emb
            + first_dense * dense_block_params
            + n_moe_layers * active_moe_block
            + final_norm
            + lm_head
        )
        mlp = expert_mlp
    else:
        dense_block_params = attn + dense_mlp + norm
        total = emb + n_layers * dense_block_params + final_norm + lm_head
        active = total
        mlp = dense_mlp

    max_seq = config["max_position_embeddings"]
    cached = cached_tokens_per_layer(config)
    full_layers = sum(1 for kind in layer_types(config) if kind == "full_attention")
    if config.get("attention") == "mla":
        per_token_per_layer = config.get("kv_lora_rank", 512)
    else:
        per_token_per_layer = config["num_key_value_heads"] * head_dim(config)
    kv_cache_bytes = 2 * per_token_per_layer * sum(cached) * 2
    kv_cache_all_full = 2 * per_token_per_layer * n_layers * max_seq * 2

    if config.get("moe"):
        mlp_ratio = config.get("moe_intermediate_size", dense_ff) / h
    else:
        mlp_ratio = dense_ff / h

    flags = []
    flags.append(config.get("norm", "layernorm").upper())
    flags.append(config.get("activation", "gelu").upper())
    flags.append(config.get("position", "learned").upper())
    scheme = attention_scheme(config)
    flags.append(scheme)
    if full_layers < n_layers:
        flags.append(f"{full_layers} full / {n_layers - full_layers} sliding")
    if config.get("moe"):
        flags.append(f"MoE {config['num_experts']}e/top-{config['experts_per_token']}")
    verdict = " · ".join(flags)

    return Breakdown(
        name=name,
        total_params=total,
        active_params=active,
        mlp_params_per_layer=mlp,
        attn_params_per_layer=attn,
        embedding_params=emb,
        kv_cache_bytes_bf16=kv_cache_bytes,
        kv_cache_all_full_bytes_bf16=kv_cache_all_full,
        full_attention_layers=full_layers,
        mlp_ratio=mlp_ratio,
        attention_scheme=scheme,
        verdict=verdict,
    )



~~~
补齐原Build It的config后运行：
~~~python
config.update({"activation":"swiglu", "norm":"rmsnorm", "tie_word_embeddings":False})
report = analyze("Llama式8B算例", config)
assert report.total_params == 8030261248
assert report.kv_cache_bytes_bf16 == 17179869184
print(report.total_params, report.kv_cache_bytes_bf16/(1024**3))
~~~
输出应为8030261248和16.0GiB。原MLA分支将latent按2×rank存储，不含解耦RoPE等真实细节，不可用于精确DeepSeek预算；Qwen bias等也未全算。

## 六、Use It：使用现成工具

原脚本会遍历自带configs并打印参数分项/活跃比例/KV。你可在合法公开config上手算并对照模型定义/参数state_dict，而非盲信verdict。不同Llama版本、窗口、精度和是否真截断局部KV都影响部署；读config不用下载数百GB权重。



## 七、失败模式、原文边界与工程注意

1. RMSNorm没省一层matmul；RoPE可计算不等于无限上下文可靠。
2. SwiGLU公式要带down projection，原简写输出仍是ff维。
3. 不同模型norm/head/bias与共享有别，不能每层一律12d²。
4. MLA不是简单缩KV头数，原估算缺额外结构。
5. total/active只涉及专家比例的一部分，不能8/256直接等同37/671。
6. 405B BF16权重约810GB，8×80GB共640GB连权重都不够，不该继续给正并发数。
7. sliding逻辑窗口与实际缓存实现不同，未evict的后端不会自动节省到公式值。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
从具体checkpoint提取norm、RoPE、FFN、Hq/Hkv/head_dim、MoE/shared、tied head、局部层；列参数与KV公式假设，activation作为估算。查版本/语言/许可与工具支持，在目标任务和硬件实测，不按家族名字或未验证表格推荐。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Qwen2.5 72B公开config逐项计数，对照模型实现，寻找QKV bias等819200差额，不能一概叫head rounding。
2. 256/top8=3.125%，8/top2=25%；前者更稀疏而非原文“denser sparse”，总模型active比还含共享/attention。
3. 405B按实际KV配置算FP8/BF16，先检查BF16权重810GB>640GB；原节点假设无法放权重，需量化/更多卡才讨论并发。
4. Gemma2半全局/半W4096，在T8192时KV保留比例(8192+4096)/(2×8192)=.75，即理论省25%，注明缓存策略。
5. 自选新公开模型，读model card/config做六项差异，若新增SSM等应增加维度；不以“六旋钮”限制真实架构。

## 十、术语与完成标准

RMSNorm：均方根规范化；RoPE：旋转位置；SwiGLU：门控MLP；GQA/MQA：共享KV；MLA：latent注意力；MoE：专家混合；active params：每token参与部分；sliding window：局部历史。

完成检查：

- 能读显式head_dim与tied head。
- 能算8B参数和16GiB KV。
- 能指出MLA近似与MoE存储。
- 能发现405B八卡BF16装不下。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行补齐config的架构计算器：8030261248参数，BF16逻辑KV16GiB。没有加载对应checkpoint或测试长上下文能力。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 1 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[13：完整LLM流水线——Manifest、产物契约、门禁与回滚]] · 下一课：[[15：推测解码与EAGLE-3——保分布采样、残差与KV回滚]]
