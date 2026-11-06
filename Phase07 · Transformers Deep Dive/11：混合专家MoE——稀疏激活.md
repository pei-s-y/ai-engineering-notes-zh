# 11：混合专家MoE——稀疏激活

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F11-mixture-of-experts) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md)。原题：Mixture of Experts (MoE)。

## 零、初学者先读：让每个 token 只请少数“专家”看病

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第05课](<05：完整Transformer——编码器与解码器.md>) 与 [第07课](<07：GPT——因果语言建模.md>) 里，Transformer 的前馈层（FFN）是"全员出诊"——每个 token 都要经过同一组参数；本课把这组参数换成一大排"专家"，由一个小小的路由器决定每个 token 只请其中几个。下一课 [第12课](<12：KV缓存与Flash Attention——推理优化.md>) 解决"注意力太贵"，本课解决"FFN 太贵"。

### 0.1 先理解这一课在做什么

稠密（dense）模型有个铁律：参数越多，每个 token 的计算量越大——70B 模型每个 token 都要过完全部 70B 参数。MoE（Mixture of Experts，混合专家）打断这条绑定：把每层的 FFN 换成 E 个独立的专家 FFN，再加一个路由器（router），每个 token 只选其中 k 个。总参数量随 E 涨（存得下多少知识），每个 token 的计算量随 k 涨（每次激活多少参数）——两者不再同步。原课的口号数字：稠密 70B 每个 token 激活全部参数；DeepSeek-V3 总共 671B、每个 token 只激活 37B，却在几乎每个基准上赢过前者（原课参考）。这是"这十年最重要的缩放思想"：想更聪明，不一定要每个 token 付更多的算力。

一个帮助记忆的类比（补充）：医院。稠密模型是"全院医生每次会诊全员到场"——病人多付钱、医院多占人；MoE 是"挂号处分诊"——每个病人只看 k 位专科医生，但医院仍要养全部医生（显存）并在楼里跑腿（通信）。分诊台就是路由器：很小，但决定一切。

### 0.2 本课术语先翻成白话

- 专家（expert）=一个独立的 FFN，参数专门服务某类计算；"专家"只是个 FFN，不是小模型
- 路由器（router）=一个单层线性网络，给每个 token 对每个专家打分
- top-k 路由（top-k routing）=每个 token 只走分数最高的 k 个专家
- 门控（gate）=被选中专家的输出混合权重；选中的分数过 softmax 归一化得到
- 稀疏激活（sparse activation）=激活参数 < 总参数；37B/671B ≈ 5.5%
- 负载均衡（load balancing）=让 token 别扎堆找同一个专家，否则其余专家"饿死"
- 辅助损失（auxiliary loss）=在训练损失里加一项惩罚专家用量不均
- 专家容量/token 丢弃（token dropping）=给每个专家设上限，溢出的 token 直接跳过这层
- 无辅助损失均衡（auxiliary-loss-free）=DeepSeek-V3 的招：给路由分数加可学习的偏置，偏置只影响选择、不影响门控权重
- 共享专家（shared expert）=每个 token 必经的专家，负责通用知识；被路由的专家负责 specialization
- 细粒度专家（fine-grained experts）=每个专家更窄（如 1/8 FFN 宽），E 更大、k 更大
- 专家并行（expert parallelism）=把不同专家放到不同 GPU，token 跨网络找自己的专家
- all-to-all 通信=每台 GPU 与其他所有 GPU 互换 token 的通信模式，MoE 延迟的主要来源

### 0.3 用小数字走一遍

做一个 2 专家、top-1、4 个 token 的迷你 MoE。设 d_model=2，路由器是"hidden·W"打分，每个专家是一个 2 参数的线性层（对应原课 Build It 的"one linear each, for illustration"）。

**第 1 步：路由打分。** 取路由器权重 w_E1=[1.0, 0.0]、w_E2=[0.85, 0.1]，4 个 token：

| token | 打分（E1, E2） | top-1 选择 |
|-------|----------------|-----------|
| A=[1,0] | (1.0, 0.85) | E1 |
| B=[1,1] | (1.0, 0.95) | E1 |
| C=[2,0] | (2.0, 1.70) | E1 |
| D=[0,1] | (0.0, 0.10) | E2 |

（打分就是点积：如 B 的 E1 分数 = 1×1.0+1×0.0 = 1.0。）

**第 2 步：发现问题。** 用量是 E1=3、E2=1，理想均衡是 2/2。路由器有"扎堆"倾向——这就是负载不均。

**第 3 步：偏置修正（无辅助损失均衡）。** 超额的专家偏置减 γ、不足的加 γ，取 γ=0.05：E1 得 −0.05、E2 得 +0.05。选择用"分数+偏置"：

| token | 偏置后 (E1, E2) | 新选择 |
|-------|-----------------|--------|
| A | (0.95, 0.90) | E1 |
| B | (0.95, 1.00) | **E2**（翻转） |
| C | (1.95, 1.75) | E1 |
| D | (−0.05, 0.15) | E2 |

一步之后用量 2/2，均衡了。注意 B 的翻转：偏置只动了"选谁"，门控权重仍由**原始分数**算——模型的表达没有被劫持。

**第 4 步：门控权重。** top-1 时被选集合只有一个专家，单值 softmax 恒为 1.0——门控退化成直通。换成 top-2 才有意义：token A 的两个原始分数是 (1.0, 0.85)，softmax 归一化 = e^0 和 e^−0.15（e^−0.15≈0.861）→ 门控 ≈ 0.537 和 0.463，输出 = 0.537·E1(A) + 0.463·E2(A)。

**第 5 步：激活参数量。** 每个专家 2 个权重 → 总专家参数 4，top-1 每 token 激活 2（50%）。放大到真实规模（原课参考）：DeepSeek-V3 是 256 个被路由专家 + 1 个共享专家、每 token 激活 8 个被路由专家 → 37B/671B ≈ 5.5%。细粒度的组合爆炸：从 256 个里选 8 个的组合数 C(256,8) ≈ 4.1×10^14——同样的总参数，可选"专家组合"多出几个数量级。

以上是用于理解的算例，不是新增的训练或性能实测。（0.3 全部数字已用本机脚本复核，见第十一节。）

### 0.4 怎样连接到正文代码

原课 Build It 的三个部分：先读 `route()` 函数——它同时实现了 0.3 的第 1、3、4 步（打分、加偏置选 top-k、对**原始分数**做 softmax），这是 DeepSeek-V3 均衡法的核心；然后是"100 个 token 过路由器"实验（观察无偏置时用量歪斜、加偏置更新后收敛到均匀）；最后是参数量对比（DeepSeek-V3 形状的配置数出来吓人一跳）。课程页只完整展示了 `route()`，后两部分是文字描述。玩具设定是 n_experts=8、top_k=2，比 0.3 的 2 专家 top-1 更接近真实配置。

### 0.5 理解检查（含参考答案）

**问题1：** MoE 模型的"总参数 671B、激活 37B"，存储和算力分别跟哪个数走？

**参考答案：** 存储跟 671B——所有专家都得放进显存，不管哪个被激活；算力跟 37B——每个 token 只过 k 个专家加共享专家。MoE 打断的是"参数量=每 token 计算量"的绑定，但没打断"参数量=显存需求"。

**问题2：** 为什么偏置只加在"选择"上，门控权重却用原始分数？

**参考答案：** 如果偏置进门控，它就直接改了模型输出（比如人为压低某专家的输出占比），等于往主损失里塞了一个手工调节项，伤质量。只影响选择、不影响权重，就把"负载均衡"与"模型表达"解耦——这正是 DeepSeek-V3 相比辅助损失方案的核心优势。

**问题3：** top-1 路由有什么隐性问题？

**参考答案：** 0.3 第 4 步算过：单专家 softmax 恒为 1.0，门控失去"按置信度混合"的能力，专家之间的软过渡消失；而且 top-1 对路由器噪声最敏感（第二名差 0.01 就完全换人）。所以现代 MoE 多用 top-2 到 top-8。

**问题4：** 为什么小模型（<7B）不适合上 MoE？

**参考答案：** 原课给了经验阈值：MoE 的质量优势要激活参数超过约 6B 才显现。小模型本来就快、显存也放得下，稀疏化只会增加路由开销却换不回质量。

能复算 0.3 表格里任意一格的打分、能说清偏置为什么让 B 翻转，进入正文。

---

## 一、这一课解决什么问题？

稠密 Transformer 的推理 FLOPs 约等于参数量（前向乘 2）。模型越大，每个 token 付的账单越贵。到 2024 年，前沿模型撞上算力墙：要"明显更聪明"，每个 token 需要的算力指数级增长。MoE 拆掉这堵墙：把每个 FFN 换成 E 个独立专家+一个选 k 个的路由器。总参数 = E×FFN 大小，每 token 激活参数 = k×FFN 大小。典型的 2026 配置：E=256、k=8——存储随 E 涨，计算随 k 涨。2026 的前沿几乎全是 MoE：DeepSeek-V3（671B 总 / 37B 激活）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss；在 Artificial Analysis 的独立榜单上，开源前十全是 MoE（原课参考）。本课要回答：**"更聪明"和"更贵"是怎么被拆开的？** 以及拆开的代价是什么（提示：显存与通信）。

## 二、核心概念与推导

### 2.1 FFN 替换：稠密块变稀疏块

稠密 Transformer 块（[第05课](<05：完整Transformer——编码器与解码器.md>) 的标准结构）：

~~~
h = x + attn(norm(x))
h = h + FFN(norm(h))
~~~

MoE 块把第二行的 FFN 换成路由+专家（原课伪代码，原样转录）：

~~~
h = x + attn(norm(h))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
~~~

每个专家是一个独立的 FFN（通常是 SwiGLU 结构）；路由器就是一个单层线性层。每个 token 自己挑 k 个专家，拿到的输出是这 k 个专家输出的门控加权和。注意力子层不动——**MoE 只稀疏化 FFN**，这是最常见的误解点。

（补充）为什么只动 FFN 不动注意力？两个原因。其一，FFN 占了 Transformer 约 2/3 的参数（四倍扩展的中间层，见 [第05课](<05：完整Transformer——编码器与解码器.md>)），是"知识"的主要存放地，稀疏化它的收益最大。其二，注意力的语义是"每个 token 看所有 token"，把它稀疏化就破坏了信息混合的通路；而 FFN 天然是逐 token 独立计算（同一个 token 的向量进去、出来，不跨 token），逐 token 换一组参数不改变数据流形状。这也是 0.3 的迷你例为什么只需要"每个 token 一个向量"就能演完整场戏。

（补充）逐 token 是关键词：MoE 的路由是**每个 token 各自选**，不是每句话或每个 batch 统一选。同一句话里，"的"和"蛋白质"大概率找不同专家——前者走共享/通用路径，后者走领域专家。0.3 表格里 4 个 token 三种选择，就是这个性质的缩影。

### 2.2 负载均衡：三个流派

如果路由器把 90% 的 token 都送给专家 3，其余专家训练不足、显存白占。历史上出现过三种解法：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）：加一个与专家用量方差成正比的惩罚项。有效，但引入超参数和第二路梯度信号。
2. **专家容量+token 丢弃**（早期 Switch）：每个专家最多处理 C×N/E 个 token，溢出的 token 直接跳过这层。伤质量。
3. **无辅助损失均衡**（DeepSeek-V3）：加一个可学习的逐专家偏置，只改变路由器的 top-k 选择；偏置在训练损失之外更新。2024 年的大解锁。

第三种的细节（原课描述）：每步训练后，检查每个专家的用量高于还是低于目标，把偏置拨 ±γ。选择用 `scores + bias`，门控用的专家概率仍是原始 `scores`——路由与表达解耦。0.3 的第 3 步就是这套机制的完整演示。

### 2.3 共享专家

DeepSeek-V2/V3 还把专家分成**共享**与**被路由**两类：每个 token 必过全部共享专家；被路由专家由 top-k 挑选。共享专家负责通用知识（每个 token 都要的语法、常见搭配），被路由专家负责专业化。V3 是 1 个共享专家 + 256 选 8 的被路由专家。

（补充）用 0.3 的迷你例理解共享专家的账：假设给 2 专家的例子再加 1 个共享专家（2 参数），每 token 的激活参数从 2 变成 2+2=4，总参数从 4 变成 6——激活率从 50% 升到约 67%，换来的是"通用成分不必每个专家都学一遍"。极端情形是全共享（k=E）：退回稠密模型。共享专家数量就是"稀疏度换稳定性"的旋钮：共享越多越稳、越不稀疏。

### 2.4 细粒度专家

经典 MoE（GShard、Switch）：每个专家和一个完整 FFN 一样宽，E 小（8—64）、k 小（1—2）。现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 宽度），E 大（256+）、k 大（8+）。总参数不变，但可选组合暴涨——C(256,8) ≈ 4×10^14 种"专家组合"，质量上升、延迟不变（原课参考）。

### 2.5 成本画像（原课表）

| 配置 | 激活参数/token | 总参数 |
|------|----------------|--------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

原课的结论：DeepSeek-V3 在几乎每个基准上赢过 Llama 3 70B（稠密），同时每 token 的激活 FLOPs 更少。**参数多 = 知识多；激活 FLOPs 多 = 每 token 计算多。MoE 把两者解耦。**

（补充）读这张表的正确姿势是横着两列一起看，只看一列都会得出错误结论：只看总参数——"671B 打不过 70B 就是 MoE 不行"（忽略了激活口径不同）；只看激活参数——"37B 的模型赢了 70B，稀疏化白给"（忽略了 671B 的知识容量是赢的原因之一）。工程上汇报 MoE 模型规格时，两个数必须一起报，这也是第八节检查模板的第一条。

（补充）把 0.3 的迷你例接到这张表：2 专家 top-1、每专家 2 参数，激活率 50%；DeepSeek-V3 是 256 选 8+1 共享，激活率约 5.5%。激活率 = k/E（加共享专家修正）——迷你例 1/2=50%，真实配置 8/256≈3.1%（再算上共享专家与注意力等常开部分回到 5.5%）。同为"稀疏"，程度差一个数量级，这正是"细粒度专家"的意义：E 越大、专家越窄，同样的算力能买到越多样的知识组合。

### 2.6 代价：显存与通信

所有专家无论是否触发都在 GPU 上。671B 的模型 fp16 权重需要约 1.3 TB 显存（原课参考）——单卡放不下，前沿 MoE 部署必须做专家并行（expert parallelism）：专家分片到多卡，token 跨网络路由。延迟瓶颈是 all-to-all 通信，不是矩阵乘。

### 2.7 把 0.3 的数字串回架构（补充）

4 个 token、2 专家、top-1：路由器打分（0.3 第 1 步的表格）→ 用量 3/1 偏斜 → 偏置 ±0.05 让 B 翻转、用量 2/2（第 3 步）→ top-1 门控恒为 1（第 4 步）→ 每 token 激活 2/4 参数（第 5 步）。放大到 DeepSeek-V3：256 选 8 + 1 共享 → 5.5% 激活率；偏置 γ 对应原课 Build It 第 2 步的收敛实验。对照 2.2 的三种均衡流派，0.3 演的是第三种。

## 三、操作前准备

Build It 只用 Python 标准库（math、random），无需第三方依赖。Use It 需要 HuggingFace transformers：

~~~bash
python -m pip install transformers torch
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或下载模型（Mixtral-8x22B 约 141B 参数，fp16 需要多卡，仅按原课示例保留）。（补充）运行原课 `route()` 前记得 `import math`——课程页代码块没带这行，直接跑会 `NameError`，本篇第四节已标注。

## 四、Build It：原课分步代码与中文讲解

原课 Build It 三步：路由器 → 100 token 实验 → 参数量对比。课程页只完整展示了第 1 步的 `route()` 函数；第 2、3 步为文字描述。本节先原样转录代码，再按文字描述整理，补充代码一律标注（补充）。三个部分的分工：`route()` 是机制（怎么选、怎么加权），100 token 实验是动态（均衡怎么收敛），参数对比是规模（真实配置长什么样）。先读懂第一个，后两个只是把数字放大。

### 4.1 第 1 步：路由器（原课代码，原样转录）

（核对：函数用了 `math.exp`，原课页未展示 `import math`，运行前需自行补上，此处按原课代码原样转录、不改函数本体。）

~~~python
# 本块中文说明：带偏置的 top-k 路由——打分、加偏置选人、对原始分数做 softmax 门控。
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
~~~

逐行拆开，正好是 0.3 的四步：

- `scores`：hidden 与每个专家权重向量的点积——0.3 第 1 步的打分列。
- `biased`：分数加偏置——0.3 第 3 步；**偏置只出现在这一行**，后面不再出现。
- `top_idx`：按偏置后分数降序取前 k 个的下标——0.3 的"选谁"。`sorted` 稳定排序意味着同分时下标小的专家排前面。
- `chosen = [scores[i] ...]`：注意取回的是**原始分数**——原课特意用大写注释标出"softmax over ORIGINAL scores"。门控权重反映模型真实置信度，不被均衡操作污染（对应 2.2 的解耦思想）。
- 末尾是标准的减最大值 softmax（数值稳定版，与 [第02课](<02：从零实现自注意力——Q、K、V与缩放点积.md>) 同款技巧）。

### 4.2 第 2 步：100 个 token 过路由器（文字步骤）

原课描述：跟踪每个专家被多少 token 触发。没有偏置时用量歪斜；加上偏置更新循环（超用的 −γ、不足的 +γ），几轮迭代后用量收敛到均匀分布。0.3 的 4-token 例子就是这个实验的手算版：γ=0.05 时一步从 3/1 到 2/2。（补充）γ 的选择有讲究：本例若取 γ=0.1，偏置后 A 是 (0.90, 0.95)、B 是 (0.90, 1.05)——两人都翻到 E2，而 C 仍是 (1.90, 1.80) 留在 E1，用量从 3/1 一步反转成 1/3，矫枉过正，下一步又要往回翻，来回振荡。γ 太大振荡、太小收敛慢，这与学习率的处境相同。

### 4.3 第 3 步：数一数 DeepSeek-V3 形状的参数（文字步骤）

原课描述：打印一个 MoE 配置的"稠密等价"参数量。DeepSeek-V3 形状：256 个被路由专家 + 1 个共享、每 token 激活 8 个、d_model=7168。原课结论：总量"令人瞠目"（eye-watering），激活量只有稠密 Llama 3 70B 的七分之一（原课参考）。（补充）粗算量级：每个被路由专家按 SwiGLU 三矩阵约 3×d×(d/8)…不同实现细节不同，精确数字以原课参考 671B/37B 为准，本篇不重算、不虚构。

## 五、把代码运行起来：补充入口与检查

（补充）原课 `code/main.py` 未随页面展示，本篇不虚构其完整内容。以下是本机实际执行过的检查（纯标准库，把 0.3 的手算喂给原课的 `route()`）。运行前先 `import math`（见第三节说明）：

~~~python
# 本块中文说明：（补充）用 0.3 的数字验证原课 route() 的行为。
import math

def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates

W_router = [[1.0, 0.0], [0.85, 0.1]]   # 0.3 的两个专家
tokens = [[1, 0], [1, 1], [2, 0], [0, 1]]

# 无偏置、top-1：A/B/C 都选 E1（0.3 第 1 步的表格）。
assert all(route(t, W_router, 1, [0.0, 0.0])[0] == [0] for t in tokens[:3])
assert route(tokens[3], W_router, 1, [0.0, 0.0])[0] == [1]
# top-1 门控恒为 1.0（单值 softmax）。
assert route(tokens[0], W_router, 1, [0.0, 0.0])[1] == [1.0]
# 偏置后、top-1：B 翻转到 E2，其余不变（0.3 第 3 步的表格）。
biased_pick = [route(t, W_router, 1, [-0.05, 0.05])[0][0] for t in tokens]
assert biased_pick == [0, 1, 0, 1]
# top-2 门控：token A 的两个门控 ≈ 0.537 / 0.463。
g = route(tokens[0], W_router, 2, [0.0, 0.0])[1]
assert abs(g[0] - 0.5374) < 5e-4 and abs(g[1] - 0.4626) < 5e-4
~~~

（补充）补充检查方法：把 γ 换成 0.1 重跑偏置后的选择，4 个 token 的选择变成 [1, 1, 0, 1]——用量从 3/1 反转成 1/3，矫枉过正，下一步还得翻回来。亲眼看一次这种振荡，比读十遍文字更记得住。

## 六、Use It：使用现成工具

HuggingFace 加载（原课代码，原样转录）：

~~~python
# 本块中文说明：加载一个现成的 MoE 因果语言模型。
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
~~~

2026 年的生产推理：vLLM 原生支持 MoE 路由；SGLang 有最快的专家并行路径。两者都自动处理 top-k 选择与专家并行（原课参考）。

（补充）用 `transformers` 直接加载 Mixtral-8x22B 时要有预期：141B 参数 fp16 约 282 GB，单卡放不下——`from_pretrained` 默认会把权重全部拉到本地，没有多卡 `device_map` 就是内存事故。学习阶段更现实的做法是加载小一点的 MoE（如 Mixtral 8x7B 的教学复刻或 Qwen 系小 MoE），或直接读模型配置里的 `num_experts`、`num_experts_per_tok` 字段来对照本课的 E 与 k。真正跑生产负载时，vLLM/SGLang 才是把显存账和路由算清的路径。

原课的选型建议——**选 MoE**：想要更低单 token 推理成本的前沿质量；有显存/专家并行的基建；负载是"token 多"型（聊天、代码）而非"上下文长"型（长文档）。**不选 MoE**：边缘部署——为任何激活 FLOPs 都要付全额存储；延迟敏感的单用户服务——路由有额外开销；小模型（<7B）——质量优势要激活参数过约 6B 才显现。（补充）"token 多型 vs 上下文长型"的分界值得展开：MoE 省的是每个 token 的 FFN 算力，token 越多省得越多；而长上下文的成本大头在注意力与 KV 缓存（[第12课](<12：KV缓存与Flash Attention——推理优化.md>)），MoE 帮不上——两种优化的作用面正好互补，所以前沿模型常常两者都上。

## 七、失败模式、原文边界与工程注意

1. **路由扎堆是默认失败态**：不干预时路由器天然偏科（0.3 第 2 步的 3/1），专家饥饿、显存浪费。三种解法见 2.2，现代默认第三种。
2. **γ 太大振荡、太小收敛慢**：0.3 与 4.2 的演示里，γ=0.05 一步均衡、γ=0.1 会过冲。调法和调学习率一样要看曲线。
3. **显存墙是 MoE 的真代价**：激活参数省了，存储一点没省——671B 需要 ~1.3TB fp16（原课参考）。没有专家并行基建，MoE 的算力优势根本兑现不了。
4. **延迟瓶颈在 all-to-all 通信**，不在 matmul（原课原话）：token 要跨 GPU 找专家，网络一慢整个流水线慢。单用户低延迟场景（路由开销占比高）尤其吃亏。
5. **token 丢弃直接伤质量**：训练期的专家容量上限让溢出 token 跳层，等于给这部分 token 的前向"缺斤短两"。
6. **"MoE 模型更大所以更强"是误读**：强不强看激活参数与训练质量，总参数只决定知识容量上限；混合比较（稠密 70B vs MoE 671B）时要报告激活参数才有意义（补充）。
7. **小模型别上 MoE**（原课边界）：质量优势在 ~6B 激活参数以上才出现；之下只添路由开销。
8. （补充）0.3 的门控演示依赖"原始分数"这一行代码：自己实现时若把 `chosen = [scores[i] ...]` 手滑写成 `chosen = [biased[i] ...]`，均衡操作就悄悄污染了模型表达——这是最容易抄错的一行。
9. （补充）"训练时路由、部署时路由"可能不一致：训练的 batch 统计与推理的单 token 视角有细微差别，个别 token 的选择会漂移。原课 Hard 练习的"推滚对齐路由"就是针对这类不一致的修复；读 DeepSeek 系技术报告时值得留意这一点。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/skill-moe-configurator.md`（一个"给定参数预算、训练 token 量与部署目标，为新 MoE 选 E、k 与共享专家布局"的技能文档）。改写成中文自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
检查 MoE 选型：给出总参数与激活参数两个数（只报一个是误导）；说明 E、k、共享专家数与依据（预算/数据/部署）；说明负载均衡用哪种方案及其超参（γ 或辅助损失系数）；说明显存账（fp16 权重总量）与专家并行拓扑（几卡、all-to-all 带宽）；激活参数是否超过 ~6B 的质量阈值；与稠密基线比较时是否同激活参数口径；所有未实测的基准/速度数字标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **原课 Easy**：运行 `code/main.py`，观察无辅助损失的偏置更新如何在 50 轮迭代内抹平专家用量。验收标准：打印每轮用量直方图或最大-最小用量差，能看到单调收敛；并指出 γ 翻倍后是收敛更快还是开始振荡（自己做实验回答）。
2. **原课 Medium**：把学习型路由器换成哈希路由器（确定性、无学习），比较质量与均衡。验收标准：两种路由的负载均衡度与下游质量各有数字（自己实测），并能解释为什么学习型路由更好——哈希不看内容，token 无法按"需要什么知识"分流。
3. **原课 Hard**：实现 GRPO 式"推滚对齐路由"（DeepSeek-V3.2 trick，原课描述）：推理时记录哪些专家被触发，梯度计算时强制同样的路由。验收标准：在一个玩具策略梯度设定上对比"自由路由"与"对齐路由"的效果，并解释这种做法消除的是哪种训练-推理不一致。

（补充）做 Easy 题时的观察建议：打印每轮的"最大用量−最小用量"，理想情况是这条差值曲线单调趋近 0；若它在两个值之间来回跳，说明 γ 偏大（正是第五节 γ=0.1 的振荡），把 γ 减半再看。这个实验 15 行标准库就能写完，是本课性价比最高的动手题。

（补充）做 Medium 题时的预期：哈希路由的均衡近乎完美（均匀哈希天然均匀），但质量更差——这正是题目想让你撞见的反直觉结论：**均衡是手段，不是目的**。学习型路由器哪怕均衡差一点，只要它把"需要同类知识的 token"送到同一个专家，质量就赢。这也反过来解释了为什么 DeepSeek-V3 费那么大劲做"无辅助损失的均衡"——辅助损失惩罚用量不均，会连带着惩罚"合理的专业化扎堆"。

## 十、术语与完成标准

专家：MoE 里一个独立 FFN；路由器：给 token 对专家打分的单层线性网络；top-k 路由：每 token 只走 k 个专家；门控：选中专家输出的 softmax 混合权重；稀疏性：k×专家宽/(E×专家宽)，DeepSeek-V3 约 5.5%；辅助损失：惩罚专家用量不均的额外损失项；无辅助损失均衡：只影响选择的逐专家偏置，DeepSeek-V3 方案；共享专家：每 token 必经、承载通用知识；细粒度专家：更窄但更多的专家，组合数暴涨；专家并行：按专家分卡、token 跨网络路由；all-to-all：MoE 延迟的主要来源；token 丢弃：专家满员时溢出 token 跳层。

完成检查：

- 能复算 0.3 表格任意一格的打分与偏置后的选择。
- 能说清"总参数 671B、激活 37B"分别对应显存和算力的哪一端。
- 能解释偏置为什么只进选择、不进门控，以及抄错那一行会发生什么。
- 能说出负载均衡三流派及各自代价。
- 能解释共享专家与被路由专家的分工、细粒度专家为什么组合数暴涨。
- 能对"为什么小模型和边缘部署不适合 MoE"给出两条独立理由。
- 能指出 MoE 稀疏化的是 FFN 而不是注意力，并说出原因。
- 能解释"均衡是手段不是目的"——哈希路由均衡完美但质量更差说明了什么。

## 十一、验证范围与来源许可

原课 Build It 共 3 个步骤，其中 `route()` 函数逐字转录嵌入本篇（仅按本篇格式添加中文说明与一处 import 核对说明，未改函数本体逻辑）；第 2、3 步为课程页文字描述，本篇如实转述，未虚构代码。Use It 的 HuggingFace 代码原样保留。课程页未展示运行输出，本篇未虚构"原课参考"数值（671B/37B/141B/1T/1.3TB/C(256,8)≈4×10^14/6B 阈值等均出自原课；C(256,8) 精确值 409,663,695,276,000 为本机按组合公式复核，与原课"约 400 万亿"一致）。

本机验证记录（2026-11-06，纯标准库脚本，不构成训练或性能实验）：

- 抓取课程页成功，以原文为准整理。
- 0.3 节全部手算数字（打分、偏置后选择、top-2 门控 0.537/0.463、2/2 均衡）已用脚本复核，并与原课 `route()` 函数实际运行结果对齐（即第五节的断言全部执行通过）。
- 已转录的 `route()` 通过语法检查（需自行 `import math`，已在正文标注）。
- 未做：加载 Mixtral、任何 MoE 训练或均衡迭代的实际运行、参数量精确统计。

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
