# 08：T5与BART——编码器-解码器模型

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F08-t5-bart-encoder-decoder) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/docs/en.md)。原题：T5, BART — Encoder-Decoder Models。

## 零、初学者先读：编码器管理解，解码器管生成，中间靠交叉注意力搭桥

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第05课](<05：完整Transformer——编码器与解码器.md>)拼好了完整的编码器-解码器骨架，[第06课](<06：BERT——掩码语言建模.md>)只取编码器（理解），[第07课](<07：GPT——因果语言建模.md>)只取解码器（生成）；本课把它们**拼回去**，专攻“输入一条序列、输出另一条序列”的任务。本课是 Learn 型课：以概念为主，Build 部分只有一段 span corruption 片段。

### 0.1 先理解这一课在做什么

很多任务天然是“输入→输出”两段式：翻译（英语→法语）、摘要（5000 词文章→200 词摘要）、语音识别（音频 token→文字）、结构化抽取（正文→JSON）。对这类任务，编码器-解码器最合身：编码器把源序列压成一组稠密表示，解码器一边自回归地生成输出、一边在每一步通过交叉注意力“回头看”那组表示。两篇论文定义了现代打法：**T5**（Raffel et al. 2019，“Text-to-Text Transfer Transformer”）把每个 NLP 任务都改写成“文本进、文本出”——一种架构、一套词表、一个损失；预训练用**跨度损坏**（span corruption，挖掉连续片段再解码出来）。**BART**（Lewis et al. 2019）是去噪自编码器：用多种方式弄脏输入（打乱、遮、删、旋转），让解码器还原原文。2026 年这个架构活在：Whisper（语音→文字）、谷歌翻译栈、Flan-T5 结构化推理任务。原课的判断：解码器-only 抢走了聚光灯，但编码器-解码器从未退场。

学习本课最容易获得的“啊哈”时刻：**交叉注意力就是第05课骨架里那个“仅解码器”的子层第一次真正干活**。第05课它只是接线图上的一根线；本课你会在 0.3 用小数字亲手算一遍信息怎么从编码器流进解码器——从此“翻译模型怎么知道源句说了什么”不再神秘。

### 0.2 本课术语先翻成白话

- 编码器-解码器（encoder-decoder）/ seq2seq=两摞块：双向编码器管输入、带交叉注意力的因果解码器管输出
- 交叉注意力（cross-attention）=解码器的 Q × 编码器的 K/V——编码器信息进解码器的唯一通道（第05课讲过接线，本课讲怎么用）
- 跨度损坏（span corruption）=T5 的预训练招式：挖掉连续片段换成哨兵 token，解码器只负责把挖掉的片段吐回来
- 哨兵 token（sentinel token）=`<extra_id_0>`、`<extra_id_1>` 这类特殊标记，在输入端标记“这里被挖了”、在输出端再标记一次
- 去噪目标（denoising objective）=BART 的玩法：输入加噪声，解码器重建干净原文
- 文本填充（text infilling）=BART 五种噪声之一：遮一段，解码器要推出“多长”和“是什么”
- 句子置换（sentence permutation）=把原文句子顺序打乱
- 文档旋转（document rotation）=把整篇文档从随机位置切开、首尾对调
- teacher forcing（教师强制）=训练时解码器喂真实的上一个输出 token，不是模型采样出来的
- 束搜索（beam search）=每步保留前 k 条完整候选；翻译/摘要的标配（宽度 4—5）
- Flan=T5 在 1800+ 任务上指令微调的版本

### 0.3 用小数字走一遍

**第一步：交叉注意力里 K/V 从哪来（本课的灵魂）。** 源句“the cat sat”，3 个 token、d=2。编码器跑完得到 3 个向量（假装 K=V 就用编码器输出，实际还有投影矩阵）：K/V = [[1,0], [0,1], [1,1]]。解码器此刻的查询向量 q=[1, 0.5]。打分 = q·k_j：对 [1,0] 得 1×1+0.5×0 = **1**；对 [0,1] 得 1×0+0.5×1 = **0.5**；对 [1,1] 得 1×1+0.5×1 = **1.5**。softmax：e^1≈2.718、e^0.5≈1.649、e^1.5≈4.482，和 ≈ 8.849，权重 ≈ [0.307, 0.186, 0.506]。输出 = 加权和：0.307×[1,0] + 0.186×[0,1] + 0.506×[1,1] = [0.307+0.506, 0.186+0.506] ≈ **[0.813, 0.692]**。看清楚唯一的区别：**Q 来自解码器，K/V 来自编码器**——自注意力是三者都来自同一侧。第05课说交叉注意力是信息从编码器进解码器的唯一通道，这组小数字就是那个通道的样子。

**第二步：T5 跨度损坏走一遍。** 句子“the cat sat on the mat”（6 个词），mask_rate=0.15、平均跨度 3：n_mask = max(1, int(6×0.15)) = max(1, 0) = 1 个 token，n_spans = max(1, round(1/3)) = 1 个跨度。设这个跨度落在“sat”上：

- 损坏输入：`the cat <extra_id_0> on the mat`
- 解码目标：`<extra_id_0> sat`

往返检查（round-trip）：把输入里非哨兵的 token 和解码出的片段按哨兵位置拼回去 = 原句 ✓。注意目标**只有被挖的片段**，不是全句——这是 T5 比 BART 便宜的原因（见第七节）。原课自己的示例（9 词句挖 "brown" 和 "over the lazy" 两段）是示意性的，挖掉比例高于 15%，属于教学夸张不是精确配比（核对：9×0.15≈1.35 个 token，示例挖了 4 个；以 2.1 节的规则说明为准）。

**第三步：三家族对照表（把第05—08课串起来）。**

（这张表是本套笔记前四课的总结网：横着读比竖着读收获更大——同一个骨架，三种“注意力可见性 + 损失”的组合，就长出了三种模型。）

| 维度 | BERT（第06课） | GPT（第07课） | T5/BART（本课） |
|------|--------------|--------------|----------------|
| 用了骨架的哪半 | 编码器 | 解码器 | 两半都要 |
| 注意力可见性 | 双向（全部可见） | 因果（只见过去） | 编码器双向 + 解码器因果 + 交叉注意力 |
| 预训练任务 | 遮 15% 猜词（MLM） | 下一词预测（CLM） | 跨度损坏（T5）/ 多噪声去噪（BART） |
| 解码器输出什么 | 不适用（没有解码器） | 整个下一词序列 | 只输出损坏片段（T5）/ 整个原文（BART） |
| 天生擅长 | 理解：分类、检索、标注 | 生成：聊天、代码、推理 | 换序列：翻译、摘要、转写 |
| 代表 | BERT、ModernBERT | GPT、Llama、Claude | T5、BART、Whisper |

以上是用于理解的算例，不是新增的训练或性能实测。

**第四步：BART 的噪声长什么样（在同一句上玩一遍）。** 还用“the cat sat on the mat”：

- token 遮罩：`the cat  ?  on the mat`（中间的 `?` 是遮罩，解码头要还原 sat）
- token 删除：`the cat  on the mat`（注意：**没有占位符**！解码器必须自己发现少了一个词并插回正确位置——这比遮罩难）
- 文本填充：`the cat  ?  mat`（整段“sat on the”被一个遮罩替换，解码器要推出“三个词”+“是哪三个”）
- 句子置换：两句话 “the cat sat. on the mat.” → “on the mat. the cat sat.”
- 文档旋转：`sat on the mat. the cat`（从“sat”处切开、首尾对调）

五种里 token 删除最阴险：输入里没有任何标记告诉你哪里缺词，模型必须对“合理的原句长什么样”有真判断——这也是五种噪声组合训练比单一遮罩强的直觉来源（补充：具体结论以 2.3 的原课参考为准）。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

本课是 Learn 型课，Build 只有一段 `corrupt_spans`（0.3 第二步的代码版，且课程页给的是带省略号的骨架）。读法：先看 2.2 的前向循环图（编码器跑一次、解码器自回归、每步交叉注意力看同一个编码器输出）；再看 `corrupt_spans` 的片段计数逻辑；最后 Use It 的 T5 实际用法——注意“任务名写进输入文本”这个 T5 招式。核心记忆点：**编码器输出算一次、用到底；交叉注意力是唯一的桥；T5 只解码被挖的片段**。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么编码器只需要跑一次？

**参考答案：** 输入（源序列）在生成过程中不变。解码器每生成一个 token 都要交叉注意力看一眼编码器输出，但看的是同一份——所以算一次、缓存复用即可。原课说这对长输入是“免费加速”。

**问题2：** T5 和 BART 的预训练目标差在哪？

**参考答案：** 损坏方式和解码范围都不同。T5：挖连续跨度、换成哨兵，解码器只输出被挖的片段（含哨兵前缀）——信号便宜。BART：五种噪声（遮、删、填充、句序打乱、文档旋转），解码器重建**整个**原文——更贵但信号更全面。原课：BART 的预训练计算量高于 T5。

**问题3：** 为什么翻译用束搜索、聊天不用？

**参考答案：** 翻译的输出分布窄（一段法语译文的合理变体不多），保留前 4—5 条完整候选能换来质量；聊天的输出分布宽，束搜索会产出安全但乏味、甚至高似然却低质量的句子（第07课练习 2 同款结论）。

**问题4：** “文本进、文本出”到底高明在哪？

**参考答案：** 一个架构、一套词表、一个损失吃下所有任务：翻译、摘要、分类（把分类改写成“输出 yes/no”）、问答（输出答案文本）。T5 论文靠这个统一了 NLP 任务接口；2022 年后被指令微调的解码器-only 模型泛化接管，但接口设计是 T5 先定下的。

**问题5：** 编码器-解码器的训练损失和 GPT 一样吗？

**参考答案：** 一样，都是输出侧的 shift-by-one 交叉熵（原课原话：“Same loss as GPT, just conditioned on the encoder output”）。差别只在条件：GPT 的条件是“输入序列的前缀”，这里解码器的每个位置除了看自己已生成的部分，还通过交叉注意力看编码器输出。理解了这一点，第07课的损失知识可以原样搬过来用。

能算出 [0.813, 0.692] 并讲清三种损坏的差别后，进入正文。

---

## 一、这一课解决什么问题？

解码器-only 的 GPT 和编码器-only 的 BERT 各自把 2017 架构裁掉一半，服务于不同目标。但很多任务天然是输入-输出对：翻译、摘要、语音识别、结构化抽取。对它们，编码器-解码器最干净：编码器产出源的稠密表示，解码器生成输出、每步交叉注意那个表示。训练是输出侧的 shift-by-one——**与 GPT 同一个损失，只是条件在编码器输出上**（原课原话）。T5 和 BART 两篇论文定义了现代打法。2026 年它活在输入结构重要的地方：Whisper、谷歌翻译、部分代码补全/修复、Flan-T5。解码器-only 赢了聚光灯，编码器-解码器没退场。

本课承上：第05课的完整骨架在这里第一次“满负荷运转”（编码器、掩码自注意力、交叉注意力、FFN 全用上）；本课启下：KV 缓存与推理优化（[第12课](<12：KV缓存与Flash Attention——推理优化.md>)）里“编码器输出只算一次”正是缓存思想的第一课，Whisper 架构（[第10课](<10：音频Transformer——Whisper架构.md>)）则是本课选型表的音频版展开。

## 二、核心概念与推导

### 2.1 前向循环：一张图看懂

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键点：编码器**每个输入只跑一次**；解码器自回归生成，但每一步交叉注意力看的是**同一份**编码器输出。缓存编码器输出是长输入的免费加速。

### 2.2 T5 预训练：跨度损坏

随机挑输入的跨度（平均长 3 token、总量 15%），每个跨度换成一个唯一的哨兵：`<extra_id_0>`、`<extra_id_1>`……解码器只输出被损坏的跨度，带哨兵前缀。原课示例：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

原课的评价：比预测整个序列便宜，在 T5 论文的消融里与 MLM（BERT）和 prefix-LM（UniLM）有竞争力（原课参考）。

### 2.3 BART 预训练：多噪声去噪

五种噪声函数：1. token 遮罩；2. token 删除；3. 文本填充（遮一段，解码器要推出正确的长度+内容）；4. 句子置换；5. 文档旋转。原课结论：文本填充 + 句子置换组合出最好的下游数字（原课参考）。解码器永远重建**原文**——输出是完整序列而非只有损坏跨度，所以预训练计算量高于 T5。

### 2.4 推理

与 GPT 相同的自回归生成：贪心/束/top-p 都适用。翻译和摘要标配束搜索（宽度 4—5），因为输出分布比聊天窄。

### 2.5 2026 年怎么选

| 任务 | 编码器-解码器？ | 为什么 |
|------|------------------|-----|
| 翻译 | 通常是 | 明确的源序列；输出分布固定；束搜索有效 |
| 语音转文字 | 是（Whisper） | 输入模态≠输出；编码器负责塑形音频特征 |
| 聊天/推理 | 否，解码器-only | 没有持久的“输入”——对话本身就是序列 |
| 代码补全 | 通常否 | 长上下文解码器赢；Qwen 2.5 Coder 等是解码器-only |
| 摘要 | 都行 | BART、PEGASUS 赢过早期解码器基线；现代解码器 LLM 打平 |
| 结构化抽取 | 都行 | T5 干净：文本→文本吃下任何输出格式 |

2022 年以来的趋势（原课的编辑判断）：解码器-only 侵占编码器-解码器的地盘，因为 (a) 指令微调的解码器 LLM 靠提示通吃，(b) 一个架构比两个好扩展，(c) RLHF 假设了解码器。编码器-解码器守住的是：输入模态不同（语音、图像）或束搜索质量重要的场景。

### 2.6 一次完整训练步的时刻表（补充）

把 2.1 的循环和 teacher forcing 拼成训练时的具体时刻表，以“英→法翻译、目标 4 个 token”为例：

1. 源句进编码器，一次前向得到 (N_src, d_model)——之后整个训练步不再动它。
2. 解码器输入（teacher forcing 版目标）：`[BOS, y1, y2, y3]`——注意**右移一位**且首位是起始符，与第07课 shift-by-one 同构。
3. 整条解码器输入一次前向：每个位置经“掩码自注意力（只见更早的目标 token）→ 交叉注意力（看编码器输出）→ FFN”，得到 4 个位置的 logits。
4. 标签 `[y1, y2, y3, y4]`，四个交叉熵求和，反向传播更新**两摞块的全部参数**。
5. 推理时第 2 步换成“从 BOS 开始逐步采样、把采样结果接回输入”，其余不变。

看点：训练时解码器也是**并行**的（teacher forcing 让全部目标位置同时有监督信号）；只有推理才回到逐 token 串行——与第07课“训练并行、推理串行”的结论完全同构，只是多了一座编码器的桥。

## 三、操作前准备

Learn 型课：概念为主。Build 部分课程页只内嵌了 `corrupt_spans` 的**骨架片段**（函数签名 + 省略号，页面原样如此）；Use It 需要 HF `transformers`：

~~~bash
python -m pip install transformers torch
~~~

（核对/说明：本课没有完整 Build It 代码——第五节的往返检查、噪声示例均为（补充）自写，显式标注，非原课代码。）

## 四、Build It：原课分步代码与中文讲解

（说明：本课为 Learn 型，Build 部分短，按原课实际内容处理，不虚构代码。）

### 4.1 第 1 步：跨度损坏

原课片段（逐字转录，仅加中文注释；省略号是课程页原样，非本篇删节）：

~~~python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    # 中文注释：挑总长度约为 mask_rate 比例的跨度；返回（损坏输入, 目标）。
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    # 中文注释：要遮的 token 总数，至少 1 个。
    n_spans = max(1, int(round(n_mask / mean_span)))
    # 中文注释：跨度个数 = 总数 / 平均跨度长，至少 1 个。
    ...
~~~

目标格式是 T5 惯例：`<sent0> span0 <sent1> span1 ...`；损坏输入是未变 token 与哨兵 token 在跨度位置的交错。

### 4.2 第 2 步：验证往返

原课描述：给定损坏输入和目标，重建原句。若损坏是可逆的，前向定义就良定（well-defined）。原课强调这是 sanity check——真实训练永远不做重建，但这个测试便宜、能抓跨度簿记里的差一错误（off-by-one）。（补充解读：这一步本质上是给 `corrupt_spans` 写单元测试——损坏函数是预训练管线的第一环，它错一列，后面全部白训；本课把它列为 Build 步骤，工程直觉值得学。）

### 4.3 第 3 步：BART 噪声

原课描述：五个函数 `token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`；组合其中两个并展示结果。课程页未内嵌代码。0.3 第四步已给出这五种噪声在同一句话上的具体效果，可当本步的纸面版。

## 五、把代码运行起来：补充入口与检查

（补充）自写最小检查，覆盖 0.3 的两块手算（**非原课代码**）：

~~~python
# 本块中文说明：（补充）自写的往返检查与交叉注意力检查，非原课代码。
import numpy as np
import math

# 检查1：T5 式跨度损坏的往返重建（0.3 第二步）。
def corrupt_one_span(tokens, span_start, span_len, sent_id=0):
    # 补充版：只挖一个跨度，哨兵用字符串占位。
    sent = f"<extra_id_{sent_id}>"
    corrupted = tokens[:span_start] + [sent] + tokens[span_start + span_len:]
    target = [sent] + tokens[span_start:span_start + span_len]
    return corrupted, target

tokens = ["the", "cat", "sat", "on", "the", "mat"]
corrupted, target = corrupt_one_span(tokens, span_start=2, span_len=1)
assert corrupted == ["the", "cat", "<extra_id_0>", "on", "the", "mat"]
assert target == ["<extra_id_0>", "sat"]

# 往返重建：非哨兵 token + 解码片段 → 原句。
rebuilt = []
i = 0
for t in corrupted:
    if t.startswith("<extra_id_"):
        seg = target[target.index(t) + 1:]          # 哨兵后面的片段
        nxt = target[target.index(t) + 1:]
        # 片段长度 = 到下一个哨兵或结尾
        end = next((j for j, x in enumerate(nxt) if x.startswith("<extra_id_")), len(nxt))
        rebuilt.extend(nxt[:end])
    else:
        rebuilt.append(t)
assert rebuilt == tokens
print("round-trip ok:", " ".join(rebuilt))

# 检查2：交叉注意力 K/V 来自编码器（0.3 第一步）。
enc = np.array([[1.0, 0], [0, 1], [1, 1]])   # 假装 K=V=编码器输出
q = np.array([1.0, 0.5])                       # 解码器一侧的查询
s = enc @ q                                     # 打分 [1, 0.5, 1.5]
w = np.exp(s) / np.exp(s).sum()                 # ≈ [0.307, 0.186, 0.506]
out = w @ enc                                   # ≈ [0.813, 0.692]
assert np.allclose(w, [0.3072, 0.1863, 0.5065], atol=1e-3)
assert np.allclose(out, [0.8137, 0.6928], atol=1e-3)
print("cross-attn out:", np.round(out, 3))
~~~

（核对/说明：检查 1 的重建逻辑是单跨度简化版，多跨度时每个哨兵的片段要按顺序切分；检查 2 省略了缩放因子 √d_k（d=2 时 ≈1.41，会等比改变权重分布但不改“K/V 来自编码器”的结构），手算值与代码一致以 0.3 为准。）

（补充）为什么 0.3 第一步的交叉注意力不带缩放也说得通：缩放因子 √d_k 的作用是防止点积随维度增大而爆炸，d=2 时点积本来就在 1 上下，除不除都不影响“下标对齐、权重归一”的教学结论；真实维度（如 d_k=64、128）时必须除，否则 softmax 进饱和区、梯度消失——这是第02课缩放点积的老结论，在交叉注意力里同样适用。另外注意权重 [0.307, 0.186, 0.506] 并不均匀：q=[1, 0.5] 与 k₃=[1,1] 的打分最高（1.5），模型“看”得最多的是“sat”——这正是“解码器按当前查询挑相关源位置”的微观机制，翻译时每生成一个词都在做这件事。

## 六、Use It：使用现成工具

原课代码（逐字转录，仅加中文注释）：

~~~python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
# 中文注释：任务名写在输入文本里——同一个模型靠提示切换几十种任务。
~~~

T5 的招式：**任务名放进输入文本**。同一个模型处理几十种任务，因为每个任务都是文本进、文本出。2026 年这个模式被指令微调的解码器-only 模型泛化了，但 T5 先把它定式化（原课判断）。Flan（1800+ 任务微调的 T5 家族）让编码器-解码器在指令跟随上也有一席之地。

## 七、失败模式、原文边界与工程注意

1. 跨度簿记差一错误：哨兵替换时多吃/少吃一个 token，往返重建立即失败——这就是原课 Step 2 存在的意义：真实训练不做重建，但测试便宜、专抓这类 bug。
2. 交叉注意力方向接反：Q 必须来自解码器、K/V 来自编码器输出（0.3 第一步的唯一区别点）。接反后解码器自己看自己，翻译立刻崩——与第05课同一坑，本课它是主角。
3. 忘了缓存编码器输出：每生成一个 token 都重跑一遍编码器，长输入下开销翻倍——原课明说这是“免费加速”，不用白不用。
4. 训练/推理的解码器输入不一致：训练 teacher forcing 喂真实前文，推理喂自己的输出；这在 GPT（第07课）同样存在，seq2seq 里因为多了“编码侧条件”更容易被忽略（补充提醒）。
5. 束搜索用在聊天/开放式生成：输出分布宽的任务上束搜索产出乏味或高似然低质量的序列；它是翻译/摘要的工具，不是通用解码器（第07课练习 2 同款结论）。
6. 原文边界：本课 Build It 只有一段带省略号的骨架代码，Steps 2—3 是文字描述，课程页未内嵌实现——本篇如实标注，补充代码均为自写；“BART+句子置换组合最好”“Flan 1800+ 任务”“PEGASUS 赢过早期基线”均为原课参考；2.5 的趋势判断是原课的编辑观点。
7. 别把“解码器-only 赢了”绝对化（补充）：Whisper（[第10课](<10：音频Transformer——Whisper架构.md>)专题）和谷歌翻译栈证明输入模态不同或束搜索重要的场景里编码器-解码器仍是正解；选型看任务结构，不看流行度。
8. T5 目标里哨兵本身也是 token：解码器的输出词表要包含全部 `<extra_id_N>`，微调换词表时漏掉哨兵会让预训练权重与任务格式对不上（补充，工程提醒）。
9. 混淆“编码器双向”与“解码器也能看全源句”（补充澄清）：编码器内部双向没问题，但**解码器的自注意力仍然因果**——它只能看自己已生成的输出前缀，再加交叉注意力看源句。把解码器也做成双向会重蹈“训练/推理形态不一致”的覆辙：推理时未来的输出还不存在。
10. 忽视听觉/音频输入的预处理成本（补充）：编码器-解码器在跨模态任务（Whisper）里的编码器输入不是文本 token 而是音频特征，[第09课](<09：视觉Transformer——ViT与图像块.md>)、[第10课](<10：音频Transformer——Whisper架构.md>)分别讲这两种输入怎么切块进编码器——选型时编码器的输入管线成本常被低估。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/skill-seq2seq-picker.md`（按输入输出结构、延迟与质量目标为新任务选编码器-解码器或解码器-only）。中文学习版自查模板（直接保存在本篇，不另建文件）：

~~~markdown
为一个新任务选架构：输入和输出是“同一条序列的延续”还是“两条不同序列”？输入模态与输出是否一致（文本→文本？音频→文本？）？输出分布窄还是宽（决定要不要束搜索）？延迟预算（编码器一次前向 + 解码器逐 token，还是解码器-only 一步到位）？质量要求下有没有现成检查点（Whisper/Flan-T5/BART 家族）？结论给出：选型、理由、被否决的备选及否决原因；引用的性能数字标注来源与是否实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：运行 `code/main.py`，对 30 token 的句子做跨度损坏，验证“非哨兵源 token + 解码出的目标片段”能拼回原句。验收标准：往返重建逐 token 相等（可直接跑第五节检查 1 的补充代码；30 token、15%、平均跨度 3 时 n_mask=4、n_spans=1，正好一个 4 token 跨度——先手算预期再对结果，才叫验证而不是碰运气）。
2. 原课 Medium：实现 BART 的 `text_infill` 噪声：随机跨度换成单个 `<mask>` token，解码器要推断正确跨度长度加内容。验收标准：给出至少一个“长度也变了”的示例（比如三词跨度被一个 `<mask>` 替换），说明模型为什么必须学会输出变长序列。
3. 原课 Hard：在 200 对英→猪拉丁语料上微调 `flan-t5-small`，50 对留出集上测 BLEU；同数据同算力微调 `Llama-3.2-1B` 对比。验收标准：两组 BLEU 数字 + 相同算力说明；能解释小语料结构映射任务上两种架构的胜负原因（提示：想想 2.5 的表格——这个任务的输入有明确“源序列”身份吗？）。

（补充）练习之外的延伸：把 0.3 第三步的对照表遮住，凭记忆重画一遍——能画全说明四课的骨架知识已经连成网络；画不全的那格，就是下一个该复习的小节。这句话同样适用于前几课的对照表，是整卷复习的通用方法。

## 十、术语与完成标准

编码器-解码器（encoder-decoder）/seq2seq：双向编码器 + 带交叉注意力的因果解码器；交叉注意力：解码器 Q × 编码器 K/V，源与目标唯一的桥；跨度损坏（span corruption）：T5 的预训练——换哨兵、只解码片段；去噪目标：BART 的预训练——加噪重建全文；哨兵 token：`<extra_id_N>` 占位符；Flan：1800+ 任务指令微调的 T5；束搜索：保留前 k 条部分序列；teacher forcing：训练时喂真实前文。五个噪声函数名也值得记：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`——它们是 BART 的全部武器库。

完成检查：

- 能手算 0.3 的交叉注意力（权重 [0.307, 0.186, 0.506]，输出 [0.813, 0.692]），并说出与自注意力的唯一区别。
- 能对 6 词句走一遍 T5 跨度损坏并完成往返重建。
- 能在“the cat sat on the mat”上分别演示 BART 的五种噪声并说出哪一种不留言占位符（token 删除）。
- 能背出 BART 的五种噪声及最佳组合（原课参考）。
- 能不看表说出三家族（BERT/GPT/T5）的注意力可见性、预训练任务、擅长场景。
- 能解释“编码器只跑一次”和“T5 比 BART 预训练便宜”的原因。
- 能列出 2026 年仍选编码器-解码器的至少三类任务及原因。
- 能不看笔记复述 2.6 的五步时刻表，指出哪几步只在训练发生、哪步换成推理就变串行。
- 能说出 T5 目标格式里哨兵的两侧角色（输入端占位、输出端打头）。
- 能对 30 token 句先手算 n_mask 与 n_spans 再对照程序输出（练习 1 的验收方法）。
- 能各用一句话说清 T5 与 BART 的预训练差异，并指出谁更贵、贵在哪。

## 十一、验证范围与来源许可

课程页抓取成功（2026-11-06）；原课文档 raw.githubusercontent 与 github.com 页均不可达（502/504），与此前阶段相同，以课程页为准整理。本课为 Learn 型：原课页内嵌代码两段——`corrupt_spans` 骨架（含省略号，原样转录）与 Use It 的 T5 生成——仅加中文注释，未改逻辑；Steps 2—3 及 BART 噪声函数课程页只有文字描述，本篇如实标注未虚构。0.3 节手算数字（交叉注意力权重 [0.3072, 0.1863, 0.5065]、输出 [0.8137, 0.6928]、6 词句跨度损坏与往返重建）已于本机 Python 3.12 + numpy 2.5 实际执行复核通过；第五节补充代码通过 `ast.parse` 静态检查并实际执行通过。未执行任何训练、微调或 BLEU 评估、未下载模型。标“原课参考”的数值均来自课程页原文。

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
