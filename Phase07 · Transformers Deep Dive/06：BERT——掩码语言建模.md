# 06：BERT——掩码语言建模

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F06-bert-masked-language-modeling) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/06-bert-masked-language-modeling/docs/en.md)。原题：BERT — Masked Language Modeling。

## 零、初学者先读：先看懂“完形填空”，再学 BERT

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。上一课（[第05课](<05：完整Transformer——编码器与解码器.md>)）搭好了骨架，本课是第一种“减配特化”：只留编码器，训练目标从“预测下一个词”换成“猜被挖空的词”。

### 0.1 先理解这一课在做什么

BERT 做的事一句话讲完：把句子里的 15% 的词挖掉，让模型**看着左右两边**把挖掉的词猜回来。这就是“完形填空”。它为什么重要？2018 年之前，每个 NLP 任务（情感分析、命名实体识别、问答、蕴含）都要用自己的标注数据从头训一个模型。BERT 之后变成两步：先在海量无标注文本上做完形填空预训练出一个“懂语言”的编码器，再在你的任务上接一个小头微调。这套“预训练 + 微调”范式统治了 NLP——原课说 18 个月内 BERT 及其变体横扫了当时所有 NLP 排行榜，到 2020 年地球上每个搜索引擎、内容审核和语义检索系统里都有一个 BERT。本课要学懂三件事：完形填空这个训练信号、80/10/10 的掩码规则、为什么它不能生成文本。学完后你应该能回答“我的任务是该选 BERT 还是 GPT”——这是下一课（[第07课](<07：GPT——因果语言建模.md>)）的入场券。

### 0.2 本课术语先翻成白话

- 掩码语言建模（masked language modeling, MLM）=训练信号：随机遮 15% 的 token，模型预测被遮的原词
- 双向（bidirectional）=编码器注意力没有因果掩码，每个位置看得到所有位置——BERT 的看家本领
- `[MASK]`（BERT 里的特殊遮罩 token）=填空题的“空格”标记；推理时永远不会出现
- `[CLS]`=每个句子开头加的特殊 token，它的最终向量被当作整句表示
- `[SEP]`=分隔符，隔开成对输入（查询/文档、句 A/句 B）
- NSP（next sentence prediction，下一句预测）=BERT 原版第二个预训练任务，后被 RoBERTa 证明有害而淘汰
- 微调（fine-tuning）=编码器基本冻结或小学习率，只训顶部一个小任务头
- 交叉编码器（cross-encoder）=查询和文档拼成一句话一起进 BERT、输出相关性分数的模型，是重排序器
- 双编码器（bi-encoder）=查询和文档各自单独编码再算相似度，是检索的另一半
- 对比损失（contrastive loss）=句子嵌入模型的训练目标：相似的句子对靠近、不相似的远离
- WordPiece / BPE=两种子词分词算法；原版 BERT 用 WordPiece，ModernBERT 换成 BPE
- ModernBERT=2024 年 12 月发布的编码器翻新版：RoPE + GeGLU + pre-norm RMSNorm + 交替局部/全局注意力 + 8K 上下文

### 0.3 用小数字走一遍

**第一步：完形填空怎么做。** 词表就 5 个词：{the:0, cat:1, sat:2, on:3, mat:4}。句子“the cat [MASK] on the mat”把位置 2 的“sat”遮住（下文用 `masked` 代指遮罩 token），输入是 `[the, cat, masked, on, the, mat]`。模型在位置 2 对词表 5 个词各打一个分（logits）：the 0.1、cat 0.2、sat 3.0、on 0.1、mat 0.1。softmax 手算：e^3.0≈20.09、e^0.2≈1.22、e^0.1≈1.11（三个），总和 = 20.09+1.22+3×1.11 ≈ 24.63。于是：

- P(sat) = 20.09/24.63 ≈ **0.816**
- P(cat) = 1.22/24.63 ≈ 0.050
- P(the)=P(on)=P(mat) = 1.11/24.63 ≈ 0.045

五项加起来 = 0.816+0.050+3×0.045 ≈ 1.00（softmax 自带归一）。这个位置的损失 = −ln(0.816) ≈ **0.204**。整句的 MLM 损失只对被遮位置求——没被遮的位置不算损失。

**第二步：80/10/10 规则走一遍。** 拿 20 个 token，15% 选中率期望选中 3 个。对每个选中的位置再掷一次骰子：80% 换成 `masked`、10% 换成随机词、10% 保持原词。假设位置 4、11、17 被选中，三次随机数分别是 0.65、0.85、0.95：0.65<0.8 → 位置 4 变 `masked`；0.85 落在 [0.8,0.9) → 位置 11 换成随机词（比如“dog”）；0.95≥0.9 → 位置 17 原样不动。标签（labels）只在这三个位置记录**原词**，其余位置全是 −100（“忽略此位置”的约定值）。为什么要这么绕？因为 `masked` 这个 token 在下游微调时永远不会出现，若预训练时 100% 都见 `masked`，模型会患上“没空格就不会做题”的病——10% 随机 + 10% 原样把模型逼得对“可能有诈”的输入也保持稳健。

**第三步：看一眼输入的完整形态。** 真实 BERT 的输入是 `[CLS] the cat  [MASK]  on the mat [SEP]`——句首 `[CLS]`、句尾 `[SEP]`；若是句对任务（比如判断两段是否相关），中间再多一个 `[SEP]`：`[CLS] 句A [SEP] 句B [SEP]`。`[CLS]` 位置的最终向量被拿去当整句的表示，因为双向注意力让它“看完全句再总结”。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：先看第四节的 `create_mlm_batch`（0.3 第二步的代码版：选 15%、再掷 80/10/10、labels 记 -100）；再看 Steps 2—4 的流程（20 词小词表、200 句的迷你语料上前向 sanity check，不训练；最后把 MLM 头换成分类头、冻结编码器只训头）。Use It 部分是真实世界的 `transformers` 用法。核心记忆点：**输入被污染，标签在遮罩位置，损失只在遮罩位置算**。读完代码再回头看 0.3 的第三步：`[CLS]`/`[SEP]` 的位置安排在 `create_mlm_batch` 之前就已完成——掩码逻辑只处理普通 token，特殊 token 永不参与被选中的 15%。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么遮 15% 而不是 50%？

**参考答案：** 遮太多，上下文信息不够，任务退化成瞎猜，学不到语言结构；遮太少，每个 batch 里的训练信号（有效损失位置）太稀疏，计算浪费——15% 是原论文调出来的经验折中（原课沿用的数字，原课参考）。

**问题2：** BERT 为什么不能拿来写文章？

**参考答案：** 它的训练目标是“给定了左右上下文，填中间的空”。生成文本需要“从左往右一个字一个字往外蹦”，而 BERT 预测任何一个位置时都偷看了右边未来——推理时右边根本不存在，模型没学过这个任务。生成需要因果结构，见[第07课](<07：GPT——因果语言建模.md>)。

**问题3：** labels 里为什么用 −100 而不是 0？

**参考答案：** −100 是 PyTorch 交叉熵的“忽略索引”约定：该位置不计入损失。如果用 0（0 是个真实词的 ID），没被遮的位置也会被要求“预测自己”，任务就错了——模型会学到“照抄输入”。

**问题4：** NSP 为什么被扔掉？

**参考答案：** RoBERTa（2019）做了消融实验：去掉 NSP 反而更好。原课的说法是“NSP 伤害而非帮助”。所以 2019 年后的现代编码器都不做 NSP。

**问题5：** 掩码语言建模这个名字里的“语言建模”和 GPT 的“语言建模”是一回事吗？

**参考答案：** 不是同一个条件分布：BERT 学 P(被遮词 | 其余所有词)，GPT 学 P(下一个词 | 前面所有词)。都叫“语言建模”是因为都以纯文本为监督信号、不需要人工标注；但一个双向一个单向，决定了前者擅长理解、后者擅长生成（对照 2.6 的表）。

能讲清这道填空题和 80/10/10 后，进入正文。

（补充路线图）本课只动“损失”不动“骨架”：编码器的层数、注意力、FFN 与第05课完全相同——建议学的时候把 0.3 的算例和第05课的维度表并排放着看，体会“同一副骨架换个训练目标就是另一种模型”。

---

## 一、这一课解决什么问题？

2018 年的 NLP 现状：每个任务一个模型、一批标注数据、从零训练，没有一个可以微调的“预训练懂英语”检查点。ELMo（2018）用双向 LSTM 预训练上下文嵌入，有帮助但不能泛化。BERT（Devlin et al. 2018）的问题是：拿一个 Transformer 编码器、用全网文本、逼它从**两侧**上下文预测被遮的词，然后微调一个头到下游任务——参数效率是划时代的。结果：18 个月内 BERT 与变体（RoBERTa、ALBERT、ELECTRA）统治了所有 NLP 排行榜（原课参考）。2026 年，编码器-only 仍是分类、检索、结构化抽取的正确工具——每个 token 比 decoder 快 5—10 倍，其嵌入是现代检索栈的 backbone；ModernBERT（2024-12）用 Flash Attention + RoPE + GeGLU 把架构推到 8K 上下文（原课参考）。一句话总结本课的工程意义：如果你 2026 年在做“给文本打标签”或“找相似文本”的系统，第一反应仍然应该是编码器，而不是大语言模型。

## 二、核心概念与推导

### 2.1 训练信号：完形填空

原课例句 `the quick brown fox jumps over the lazy dog`，随机遮 15%：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the quick brown fox jumps over   the lazy dog
```

训练模型在遮罩位置预测原词。因为编码器是双向的，位置 1 的预测能用上位置 2 之后的 `brown fox jumps`——**这正是 GPT 做不到的事**。GPT 预测 P(x_t | x_{<t})（只看左边），BERT 预测 P(x_masked | x_unmasked)（两侧都看）。损失只发生在被遮位置。

### 2.2 掩码规则：80/10/10

被选中的 15% token 里：

- 80% 替换为 `masked`
- 10% 替换为随机 token
- 10% 保持不变

为什么不全用 `masked`？因为 `masked` 在推理时永不出现——若训练时 100% 的空位都是 `masked`，预训练和微调之间会产生分布偏移（distribution shift）。10% 随机 + 10% 原样让模型“保持诚实”。0.3 节已用 20 个 token 的例子走了一遍这条规则。

### 2.3 NSP 的兴亡

原版 BERT 还有第二个任务：给句 A 和句 B，判断 B 是否紧跟 A。RoBERTa（2019）消融后发现 NSP 有害无益，现代编码器都跳过它。教学价值：不是论文里的每个组件都值得保留——消融实验是裁判。

### 2.4 2024 的 ModernBERT：换零件对照表

| 部件 | 原版 BERT（2018） | ModernBERT（2024） |
|------|----------------------|-------------------|
| 位置 | 可学习绝对位置 | RoPE |
| 激活 | GELU | GeGLU |
| 归一化 | LayerNorm | pre-norm RMSNorm |
| 注意力 | 全量稠密 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 栈不同，它原生支持 Flash Attention。原课给的参考数字：序列长 8K 时推理比 DeBERTa-v3 快 2—3 倍，且 GLUE 分数更好（原课参考）。

### 2.5 2026 年仍选编码器的场景

| 任务 | 编码器为什么赢 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个 token 的嵌入质量更高 |
| 分类（情感、意图、毒性） | 一次前向，无生成开销 |
| NER / token 标注 | 逐位置输出，天然双向 |
| 零样本蕴含（NLI） | 编码器顶上加分类头即可 |
| RAG 重排序器 | 交叉编码器打分，比 LLM 重排快 10 倍 |

共同点：这些任务要的都是“对给定文本的理解表示”，而不是“往下写新文本”。

（补充）用检索的例子把这张表讲活：RAG 系统的第一阶段用双编码器（bi-encoder）给几百万条文档建向量索引、快速召回前 100 条；第二阶段用交叉编码器（cross-encoder）把“查询+文档”成对送进 BERT 精细打分、重排前 10 条。两段都是 BERT，只是用法不同——前者要快（各编各的），后者要准（拼一起互相看）。原课说交叉编码器比 LLM 重排快 10 倍，这就是它在生产 RAG 里的位置。

### 2.6 MLM 与 CLM 放在一起看（补充，衔接下一课）

| 维度 | BERT 的 MLM | GPT 的 CLM（下一课） |
|------|--------------|----------------------|
| 预测什么 | 被遮住的词 P(x_masked \| x_unmasked) | 下一个词 P(x_t \| x_{<t}) |
| 看得见哪边 | 左右两侧（双向） | 只有左边（因果） |
| 每个位置都有损失吗 | 只有约 15% 的位置 | 100% 的位置 |
| 训练与推理形态一致吗 | 不一致（推理时没有 jekt） | 一致（都是“接着往下写”） |
| 能生成文本吗 | 不能 | 能 |
| 擅长 | 理解类任务：分类、检索、标注 | 生成类任务：写作、对话、代码 |

记忆钩子：BERT 是“批改试卷的老师”（看全卷再判题），GPT 是“蒙着眼睛接话的学生”（只听前文往下接）。两者共享同一副第05课的编码器/解码器骨架，差别只在注意力的可见范围和损失落点。

## 三、操作前准备

Build It 用纯 Python（`code/main.py`，课程页内嵌了 `create_mlm_batch` 一个函数，其余步骤是文字描述）。Step 2 明确“无梯度——只做前向 sanity check；完整训练需要 PyTorch”。Use It 需要 HF `transformers`：

~~~bash
python -m pip install transformers torch
~~~

（核对/说明：`create_mlm_batch` 里用到的 `MASK_ID` 和 `rng` 是外部传入的，课程页未给出定义；下文第五节用（补充）的自写版本补齐。）

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：掩码逻辑

原课 `create_mlm_batch`（逐字转录，仅加中文注释）：输入 token ID 列表、词表大小、掩码概率；返回应用掩码后的 input_ids 和 labels（只在遮罩位置有原词 ID，其余 −100——PyTorch 的忽略索引约定）。

~~~python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    # 中文注释：先全填“忽略”，只有被选中的位置才会拿到真值标签。
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
                # 中文注释：80% 换成遮罩 token。
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
                # 中文注释：10% 换成随机词。
            # else: keep original
            # 中文注释：剩下 10% 原样保留——注意此时 labels[i]=t 已记录原词。
    return input_ids, labels
~~~

（核对/说明：`MASK_ID` 未在课程页定义，需外部给定；`rng` 需是带 `random()` 和 `randrange()` 的对象；函数保证“被选中位置才有标签”，与 2.2 的规则逐行对应。）

### 4.2 第 2 步：迷你语料上跑 MLM 预测

原课描述：在 20 个词的词表、200 个句子上训练“2 层编码器 + MLM 头”。无梯度——只做前向 sanity check；完整训练需要 PyTorch。课程页未内嵌这一步的代码。

### 4.3 第 3 步：对比掩码类型

原课描述：展示三分支规则（80/10/10）如何让模型在没有 `masked` token 的情况下仍可用——对一个未遮的句子和一个遮了的句子分别预测，两者都应给出合理的 token 分布，因为训练时两种模式都见过。

### 4.4 第 4 步：微调任务头

原课描述：把 MLM 头换成分類头，接到玩具情感数据集上。只训头，编码器冻结——这是每个 BERT 应用都遵循的模式。课程页未内嵌这一步的代码。

## 五、把代码运行起来：补充入口与检查

（补充）课程页只有 `create_mlm_batch` 一段代码。下面自写最小入口，把 0.3 的两步手算在代码里对上（**非原课代码**）：

~~~python
# 本块中文说明：（补充）自写的掩码与 softmax 检查，非原课代码。
import random, math

MASK_ID = 99  # 补充约定：玩具词表之外的 ID 当遮罩 token。

def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    # 与原课逐字一致的实现（rng 缺省由本补充入口补上）。
    if rng is None:
        rng = random.Random(0)
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
    return input_ids, labels

# 0.3 第二步的可复现版：固定随机数，重现“0.65/0.85/0.95”三种分支。
fake_rng = random.Random(7)
sent = [0, 1, 2, 3, 0, 4] * 4  # 24 个 token 的玩具句子
inp, lab = create_mlm_batch(sent, vocab_size=5, mask_prob=0.125, rng=fake_rng)
print(inp)
print(lab)
# 观察点：labels 里只有少数位置不是 -100；对应 input 位置多数是 MASK_ID，
# 偶尔是随机词（0—4 中某个）或原词——三种分支都能出现。

# 0.3 第一步的 softmax 复核：
logits = [0.1, 0.2, 3.0, 0.1, 0.1]  # 词表 {the, cat, sat, on, mat}
es = [math.exp(v) for v in logits]
z = sum(es)
probs = [e / z for e in es]
print([round(p, 4) for p in probs])  # 预期 [0.0449, 0.0496, 0.8157, 0.0449, 0.0449]
print(round(-math.log(probs[2]), 4))  # 预期 0.2037
assert abs(sum(probs) - 1) < 1e-12 and abs(probs[2] - 0.8157) < 5e-4
~~~

检查方法：多次换随机种子跑 `create_mlm_batch`，统计选中率应围绕 15% 波动、选中者约 80% 是 MASK_ID（这正是练习 1 的内容）。softmax 数值与 0.3 手算一致（本机已验证，见第十一节）。

（补充）再给一个统计口径的小检查，帮你建立“期望 vs 方差”的直觉：对 20 个 token 以 15% 选中，期望选中数 = 20×0.15 = 3 个，但小样本波动很大——单次实验选中 1 个或 5 个都正常；对 10,000 个 token，期望 1,500 个，波动按 √n 缩小，选中率会紧贴 15%。所以练习 1 要求 10,000 个 token 的量级，而不是 20 个：比例统计需要足够大的样本才说得清“约 80%”。80/10/10 的第三分支（10% 原样保留）最隐蔽——input 里完全看不出那个位置被选中过，只有 labels 里的真值暴露它，这也是为什么调试掩码逻辑时必须同时打印 input 和 labels 两个列表。

~~~python
# 本块中文说明：（补充）大样本下掩码比例收敛检查，非原课代码。
import random

rng = random.Random(42)
n_tokens = 10_000
selected = sum(1 for _ in range(n_tokens) if rng.random() < 0.15)
print(f"selected {selected}/{n_tokens} = {selected / n_tokens:.1%}")  # 预期约 15%
branches = [rng.random() for _ in range(10_000)]
mask_branch = sum(1 for r in branches if r < 0.8)
print(f"mask branch: {mask_branch / len(branches):.1%}")  # 预期约 80%
~~~

## 六、Use It：使用现成工具

原课代码（逐字转录，仅加中文注释）：

~~~python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
# 中文注释：输出是每个位置的上下文嵌入；[CLS] 位置的向量即可当句向量用。
~~~

原课点名的三个事实：**嵌入模型就是微调过的 BERT**——`sentence-transformers` 里 `all-MiniLM-L6-v2` 这类模型是拿对比损失训练的 BERT，编码器没变，损失换了。**交叉编码器重排序器也是微调 BERT**——成对分类 `[CLS] query [SEP] doc [SEP]`，查询和文档之间的双向注意力正是交叉编码器质量胜过双编码器的原因。**2026 年什么时候别选 BERT**：任何生成任务——编码器没法自回归地产出 token；还有 1B 参数以下的场景，小 decoder（Phi-3-Mini、Qwen2-1.5B）能以更大灵活性打平质量（原课参考）。

## 七、失败模式、原文边界与工程注意

1. 忘了 −100 约定：labels 全填真值会让每个位置都算损失，模型学成“照抄输入”，MLM 信号被稀释。这是移植 `create_mlm_batch` 时最容易踩的坑（补充提醒，约定本身是 PyTorch 惯例）。
2. 100% 用 `masked`：制造预训练/微调分布偏移——模型在微调时见到没有空格的正常句子就不知如何是好。80/10/10 就是为这个发明的（原课核心论点）。
3. 想拿 BERT 生成文本：方向性错误。生成要因果掩码和自回归，见第07课；用错工具比调参错误更致命。
4. MLM 的计算低效（补充视角）：每个 batch 只有约 15% 的位置产生损失信号，而下一词预测（第07课）是 100% 位置都有信号——这是 ELECTRA 等替换检测类方法在等算力下胜过 MLM 的动机之一（原课在延伸阅读里给了 ELECTRA 论文链接）。
5. 掩码在分词之后按“子词”进行：一个英文词可能被拆成多个子词，随机遮一半会造出怪异输入——练习 2 的全词掩码（whole-word masking）就是修这个的（原课练习设计意图）。
6. 原文边界：“18 个月统治排行榜”“快 5—10 倍”“8K 时快 2—3 倍”“ModernBERT 更好的 GLUE 分数”都是原课陈述的参考数值，非本机实测；课程页 Build It Steps 2—4 只有文字描述、代码在 `code/main.py` 未内嵌，本篇如实标注，不虚构。
7. 微调时冻结策略：原课 Step 4 只训头、冻结编码器。实际工程里常解冻顶层或全网络用小学习率（补充：这是通用实践，原课未展开——别把“只训头”当成唯一模式）。
8. `[CLS]` 不是魔法：它只是预训练时跟着一起更新的特殊位置，句向量质量靠后训练目标（如对比损失）保证，不保证直接可用（补充提醒，与 Use It 段呼应）。
9. 分词器与掩码单位要配套（补充）：80/10/10 是按“子词 token”抽样的；换用 ModernBERT 的 BPE 分词器时，同样的 15% 掩码率在“词”层面的覆盖率会变，做对比实验时要把“分词器”和“掩码策略”两个变量分开，否则结论归因错误。
10. 嵌入模型 ≠ 直接拿 BERT 输出求平均（补充）：`all-MiniLM-L6-v2` 这类模型是用对比损失**后训练**过的；裸 BERT 的 [CLS] 或平均向量在检索任务上通常明显更差——工程上选型时别把“任意 BERT”当嵌入模型用。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/skill-bert-finetuner.md`（为一个新分类/抽取任务规划 BERT 微调：骨干选择、头设计、数据、评估、停止条件）。中文学习版自查模板（直接保存在本篇，不另建文件）：

~~~markdown
规划一次 BERT 微调：任务是什么、输入输出形态（句对？单句？token 级标注？）；骨干选什么（上下文长度、语言、吞吐需求，是否需要 ModernBERT 级新架构）；头怎么设计（[CLS] 分类头还是逐 token 头）；微调策略（只训头 / 解冻顶层 / 全网络小学习率）；数据量和类别分布如何、是否需要处理不平衡；评估指标与验证集划分；早停依据；所有引用的性能数字标注来源与是否实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：运行 `code/main.py`，对 10,000 个 token 打印掩码分布。验收标准：确认约 15% 被选中，其中约 80% 变成 `masked`（统计数字应随样本量收敛到设定比例；可直接复用第五节补充的收敛检查代码，只统计、不训练；别忘了同时看 10% 随机替换和 10% 原样保留两个分支的比例）。
2. 原课 Medium：实现全词掩码：若一个词被分成多个子词，要么全遮要么全不遮。验收标准：在 500 句语料上测 MLM 准确率是否提升，能解释为什么“遮半个词”是坏输入（模型见到不合法的子词序列）。
3. 原课 Hard：在公开数据集的 10,000 句上训练一个微型 BERT（2 层、d=64），把 `[CLS]` 微调到 SST-2 情感分类；与同参数量的 decoder-only 基线对比谁赢。验收标准：两个模型在相同数据与参数预算下的 SST-2 准确率对比，并能从“双向 vs 单向”解释差异方向。

## 十、术语与完成标准

MLM（masked language modeling）：随机替换 15% token 后预测原词的训练信号；双向（bidirectional）：注意力无因果掩码、每位置见所有位置；`[CLS]`：句首特殊 token，最终向量作句级表示；`[SEP]`：成对输入的分隔符；NSP（next sentence prediction）：被 RoBERTa 消融淘汰的第二预训练任务；微调（fine-tuning）：编码器基本冻结、顶上训小头；交叉编码器（cross-encoder）：查询+文档同入一个 BERT 输出相关性分；ModernBERT：RoPE+GeGLU+RMSNorm+交替局部/全局注意力+8K 上下文的 2024 编码器。

完成检查：

- 能手算 0.3 的五词 softmax（P(sat)≈0.816、损失≈0.204）。
- 能说出 80/10/10 各自防什么病，以及 labels 为什么是 −100。
- 能复述“预训练+微调”两步范式在 2018 年之前与之后的差别。
- 能解释为什么 BERT 不能生成而 GPT 能（预测条件不同：两侧 vs 左侧）。
- 能列出 2026 年仍选编码器的至少三类任务及原因。
- 能说出 ModernBERT 相对原版 BERT 的至少三处替换。
- 能列出 0.3 的三步（填空与 softmax、80/10/10、[CLS]/[SEP] 输入形态）并各自复述设计动机。
- 能不查资料说出 cross-encoder 与 bi-encoder 的差别及各自典型用法（重排序 vs 检索召回）。

## 十一、验证范围与来源许可

课程页抓取成功（2026-11-06）；原课文档 raw.githubusercontent 与 github.com 页均不可达（502/504），与此前阶段相同，以课程页为准整理。原课页内嵌代码两段——`create_mlm_batch` 与 Use It 的 ModernBERT 加载——已逐字转录（仅加中文注释，未改逻辑）；Build It Steps 2—4 页面只有文字描述，本篇如实标注未虚构。0.3 节 softmax/损失手算已于本机 Python 3.12 复核（probs=[0.0449, 0.0496, 0.8157, 0.0449, 0.0449]、损失 0.2037）；第五节补充代码通过 `ast.parse` 静态检查并实际执行通过。未执行任何训练、未下载模型或数据集。标“原课参考”的数值均来自课程页原文。

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
