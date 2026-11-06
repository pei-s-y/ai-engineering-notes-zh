# 18：多语言NLP——跨语言迁移、少样本与分语言评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充（zero-shot两层含义、tokenization三税）；当日与原课网页版核对。零节中英对齐小例的余弦已用本机Python标准库复核。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/18-multilingual-nlp) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/18-multilingual-nlp/docs/en.md)。原题：Multilingual NLP。

## 零、初学者先读：一个模型，一百种语言

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课共享子词的直觉来自[第04课](<04：GloVe、fastText与子词表示.md>)的 BPE，句向量对齐则延续[第03课](<03：词向量与从零实现 Word2Vec.md>)的“地图”比喻——只是这次地图上住着一百种语言。

### 0.1 先理解这一课在做什么

现实是残酷的资源不均：英语有几十亿标注样本，乌尔都语只有几千，很多语言几乎为零。不可能每种语言从零训一个模型——于是**多语言模型**把一百种语言塞进同一个词表、同一套参数，让英语上学到的任务能力“顺便”迁移到别的语言。类比：会西班牙语的人学意大利语特别快——不是因为天赋，而是两种语言在大脑里共享了大量结构；多语言预训练就是逼着模型把这种共享结构学进权重里。

本课三个核心动作：**zero-shot 跨语言迁移**（只用英语标注微调，直接测别的语言）；**few-shot**（再补100~500条目标语言标注，通常是最划算的一笔投入）；**分语言评估**（每种语言单独报指标——总分好看不等于每种语言都行）。最后一条是本课的价值观：多语言系统最常见的翻车方式，就是用平均分掩盖了长尾语言的失败。

（补充展开）一个提前打的预防针：“一个模型支持100种语言”里的“支持”只是**预训练语料里见过**，不等于每种语言效果都好。语料占比、文字体系离拉丁字母多远、分词器覆盖，都在暗中决定每种语言的真实待遇——本课反复强调的分语言评估，就是在对抗“支持”这个词带来的错觉。另一个易混点：本课的向量对齐（0.3）说的是**句级**跨语言对齐，与[第04课](<04：GloVe、fastText与子词表示.md>)词级静态向量的跨语言能力是两回事——前者靠多语言编码器，后者需要专门的跨语言映射。

### 0.2 本课术语先翻成白话

- 多语言模型（multilingual model）=一套参数、一个共享词表，服务多种语言
- 共享子词（shared subword）=不同语言里同一词素切成同一token，如英语/意语的 anti- 共用一个编号
- 跨语言迁移（cross-lingual transfer）=在源语言上训练，在目标语言上使用
- zero-shot=无目标语言标注（本课有两层含义，见2.2补充，小心混用）
- few-shot=补少量目标语言标注做微调（原课语境：100~500条）
- mBERT/XLM-R=第一代多语言编码器/当前的跨语言基线（CommonCrawl预训练）
- XLM-V=词表扩到100万token的XLM-R变体，对低资源语言更友好
- mT5/NLLB=多语言生成模型/200语言翻译模型的代表
- 交叉语言一致性（cross-lingual consistency）=同一句话换语言，预测应不变
- byte fallback=生僻字符退回原始字节表示，永不OOV
- 语言亲缘（typological similarity）=语言间类型学/谱系距离，预测迁移效果的先验
- fertility（分词膨胀）=平均每词切出几个token，多则上下文吃紧
- 分语言评估（per-language evaluation）=按语言分别报告指标，不用总平均
- 跨语言一致性=同义不同语言的句子应得到相同预测，差距本身是指标

### 0.3 用小数字走一遍

**中英各3句的跨语言对齐。** 设一个二维玩具“语义空间”：第1维=话题（负=动物，正=科技），第2维=情绪（正=赞美，负=投诉）。3对句子（**直觉化假设值，真实模型是384+维、坐标从数据学出来，没人手写**，道理与[第03课](<03：词向量与从零实现 Word2Vec.md>)“地图是学出来的”相同）：

| 句子 | 向量 | 含义 |
|---|---|---|
| e1 The cat is sleeping. | [0.8, −0.6] | 动物话题、平静 |
| z1 猫在睡觉。 | [0.75, −0.65] | 同上（中文版） |
| e2 This phone is amazing. | [−0.6, 0.8] | 科技话题、赞美 |
| z2 这部手机太棒了。 | [−0.65, 0.75] | 同上（中文版） |
| e3 I want a refund. | [−0.75, −0.6] | 消费话题、投诉 |
| z3 我要退款。 | [−0.7, −0.65] | 同上（中文版） |

手算余弦（以 e1、z1 为例：点积=0.8×0.75+(−0.6)×(−0.65)=0.99；两向量长度≈1、≈0.99，相除后≈0.997）：

~~~text
cos(e1, z1) = 0.99/(1×0.9925) ≈  0.997   ← 翻译对：同义跨语言，几乎同向
cos(e2, z2) ≈  0.997
cos(e3, z3) ≈  0.997
cos(e1, z2) ≈ −0.977                      ← 动物 vs 科技赞美：背道而驰
cos(e1, z3) ≈ −0.178                      ← 动物平静 vs 消费投诉：无关
~~~

**跨语言检索**立刻能用：中文查询 z2“这部手机太棒了”，在英文文档{e1,e2,e3}里找最近邻——e2 的0.997一骑绝尘，召回正确。这就是[第14课](<14：信息检索——BM25、稠密向量、RRF与重排.md>)稠密检索的多语言版：查询和文档**不同语言**也能配上对，前提是编码器把两种语言放进同一坐标系。

**否定句陷阱**（呼应2.9）：“猫没有睡觉。”与“猫在睡觉。”词面只差一个“没有”，真实多语言编码器里两者的向量也常常很近——这是缺点不是功能，评估必须放这类困难对。同理，本课的 zero-shot 分类若英文准确率高、某语言低，总平均会掩盖这个洞——分语言报告是唯一的诚实做法。

以上是用于理解的算例，不是新增的模型实测。读下面的代码时，把每个环节先对应回这里。

### 0.4 怎样连接到正文代码

三个 Build It 分段对应三件事：4.1 的 classify 是“零样本分类”的 NLI 套路——把每个候选标签造成一句假设（“这段文字是关于{}的”），与原文拼对送进模型，读 entailment 分数——注意源码 `logits[0]` 的 softmax 取下标2是**硬编码**，换 checkpoint 必须查 model.config.id2label；4.2 的 pairs/encode 是0.3对齐实验的真身（paraphrase-multilingual-MiniLM-L12-v2 编码、归一化、点积）；4.3 的 few_shot_finetune 与第五节 few_shot_with_padding 是 few-shot 路线（后者补了动态padding这一课明确要求的修正）。第五节的 tokens_per_word 可以自查 fertility——分词膨胀在哪种语言上最严重，一目了然。

### 0.5 理解检查（含参考答案）

**问题1：** “用英语标注微调、直接测印地语”和“给模型一段文字和几个候选标签、不给任何任务标注”——都叫 zero-shot，是一回事吗？

**参考答案：** 不是。前者是**跨语言迁移**的zero-shot（有源语言标注、无目标语言标注）；后者是**NLI零样本分类**（用蕴含关系替代任务标注）。本课两段代码各用一层，混着说会说不清。

**问题2：** 英文准确率90%、阿拉伯语70%，总平均85%——这个系统“多语言合格”吗？

**参考答案：** 不合格。总平均掩盖长尾：阿拉伯语的30%缺口才是要解决的事。分语言报告+跨语言一致性+实体级测试，才能暴露真实覆盖。

**问题3：** 某语言 fertility=3 意味着什么？

**参考答案：** 该语言平均每个词切成3个token：同样的意思要多花约3倍token——上下文窗口、训练效率、推理成本三头受害；且这不是“多练练能解决”的问题，要在选型时就挑对该语言分词覆盖好的模型。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

英文有数据，不代表中文、印地语、阿拉伯语也有相同覆盖。本课学习利用共享模型跨语言迁移，并检验每种语言的效果，而非只看总体均分。

## 二、核心概念与推导

多语言模型共享词表和参数，使不同语言可借助共同表示迁移任务。共享子词并不保证所有同义词天然对齐，跨语言能力取决于预训练、语言覆盖和下游数据。

（补充 2026-11-06）“共享子词”为什么能通向“共享语义”，链条分三步想：一、英语 anti- 和意语 anti- 切成同一个token（[第04课](<04：GloVe、fastText与子词表示.md>)的 BPE 在百语混合语料上学词表，自然会这样切）；二、两种语言的相似句子里反复出现这些共享token，模型的内部表示被迫把它们放在相近位置；三、预训练够久之后，整句的表示也对齐了——0.3 的玩具对齐（cos≈0.997）就是这条链条的终点产物。注意链条的每一步都靠数据分布“顺便”达成，没人手工规定哪个token对应哪个词义，所以“共享”是统计倾向不是保证：预训练里占比小的语言，对齐质量就差——这正是低资源语言三重税的根源。

zero-shot跨语言迁移：在源语言训练，在没有目标语言标注训练的条件下测试目标语言。few-shot：加入少量目标语言标签微调。本课第一段还有“用NLI做零样本类别分类”的另一层zero-shot含义，需与跨语言迁移分开。

（补充 2026-11-06）两层 zero-shot 的对照表（本课最容易混的一对概念）：

| | 跨语言迁移的 zero-shot | NLI 零样本分类的 zero-shot |
|---|---|---|
| “零”掉的是什么 | 目标语言的**标注**（源语言有标注） | 任务的**全部标注**（借蕴含推理顶替） |
| 用的什么模型 | 在源语言微调过的任务头 | NLI/蕴含模型原样使用 |
| 正文代码 | 4.3 few_shot_finetune 的英文版 | 4.1 classify 的 NLI 套路 |
| 典型坑 | 目标语言效果差却只报平均分 | 假设模板别扭（This text is about positive）拉低分数 |

（补充 2026-11-06）few-shot 为什么“最划算”：原课参考——100~500条目标语言标注，通常能把准确率拉到英文基线的95%~98%。直觉是迁移已经把路修到门口，几百条标注只是把最后一段路铺平；而从零标注到第一条标注之间才是最大的跳跃。注意这是原课的经验数字，不是任何任务都保底，低资源+文字体系覆盖差的组合可能差得远。

模型谱系：mBERT/XLM-R偏编码理解；XLM-V扩展词表；mT5偏生成；NLLB偏翻译；BLOOM、Aya等是多语言生成模型示例。课程型号/语言数是源文背景，不当作当前排行榜或可用性承诺。

源语言不必永远英文：可以比较语言亲缘、类型特征、数据量、领域与标注质量。原文LANGRANK与qWALS是来源不同的方法，具体研究结论不应简化成“相近语言必胜”。

（补充 2026-11-06）源语言选择的直觉版：微调数据像教材，目标语言像学生——**教材的语言离学生的母语越近，同样的页数教得越多**。给斯拉夫语系目标训德语/俄语源，常常好于英语源；给印地语目标，印地语的亲缘高资源语言常好于英语。但这只是先验不是定律：领域匹配、标注质量都可能反超亲缘（法律英语源可能好过口语化的近亲语言源），所以原课的建议是“先试亲缘近的，再和英语源对比”而不是直接换。

分词fertility指每词产生多少token，影响上下文与成本。Unicode正规化、变体和稀有文字覆盖也影响表现；byte fallback可避免未知字节丢失，但不保证模型理解。

（补充 2026-11-06）原课把低资源语言的隐性代价叫“tokenization 三税”，一表记全（数字为原课参考量级）：

| 税 | 内容 | 直观后果 |
|---|---|---|
| fertility税 | 每词切成3~5倍token（印地语vs英语） | 同样意思吃3~5倍上下文、算力、延迟 |
| 变体恢复税 | 拼写变体、附加符号、Unicode正规化差异各自成为“生面孔” | 母语用户觉得显然的对应，模型认不出 |
| 容量挤占税 | 前两税吃掉上下文位置与模型容量 | 留给真正推理的预算系统性变少 |

三税合成的典型症状很隐蔽：训练正常、损失曲线正常、评估PPL正常，产出却 subtly 错——形态在中途崩掉、罕见屈折找不回。原课的结论值得背下来：**分词器坏了靠堆数据救不回来**，选型时就要验 fertility（正文第五节的 tokens_per_word 就是自查工具），长尾文字优先选带 byte fallback 的模型。

### 2.9 用一个小例子把概念串起来（补充）

同义英法句向量通常较近，可以英文query召回法语文档；但“猫在睡觉”与“猫没有睡觉”也可能词面很近。分别测每种语言和这类困难对，才能知道向量究竟支持什么任务。英文准确率高、阿拉伯语低时，总平均好看并不代表覆盖合格。

## 三、操作前准备

~~~bash
python -m pip install torch transformers sentence-transformers datasets accelerate
~~~
第一段模型较大，4060能否运行取决于显存、精度与序列长度，不默认保证。三个例子重复使用model变量：分段运行或给分类器/向量器不同名字，避免第二段覆盖第一段模型。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 用多语言NLI做候选分类

为每个候选标签生成假设，将原文与假设作为输入对。源码硬编码第2号logit为entailment，应查model.config.id2label确认；不是所有checkpoint都按同样顺序。对情感任务，This text is about positive并不自然，宜用The sentiment is {}等模板并评估。

~~~python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


# 逐候选构造NLI假设，核对模型标签编号。
def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
~~~

### 4.2 比较跨语言句向量

将英文及其法语、西语、德语翻译分别编码，归一化后点积比较。预期翻译相似度通常高于不相关句，但不是必须达到某个固定值；还应加同主题不同事实的困难负样本。

~~~python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
~~~

### 4.3 目标语言少样本微调

examples需为text与整数label的字典列表，标签要符合分类头定义。原示例未显式给tokenizer/data_collator，变长输入默认拼batch可能失败，应补DataCollatorWithPadding。训练后要在独立目标语言集评估，5epoch/2e-5只是起点。

~~~python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


# 教学训练入口，变长输入需添加动态padding。
def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
~~~

## 五、把代码运行起来：补充入口与检查

补充：训练前使用动态padding，并避免由上一段的SentenceTransformer误传给Trainer。
~~~python
from transformers import DataCollatorWithPadding
# 在few_shot_finetune函数中创建Trainer时增加：
# data_collator=DataCollatorWithPadding(base_tokenizer)
# 保留已编码的labels；为每个batch自动补齐输入长度。

def tokens_per_word(tokenizer, text):
    words = text.split()
    if not words:
        return 0.0
    return len(tokenizer.encode(text, add_special_tokens=False)) / len(words)
~~~
空格分词的指标仅适合采用空格分词的比较协议，中文不能直接把整句当一个词。分别记录语言、样本数、准确率/宏F1和错误例，模型分数与微调本次未运行。

### 补充：将微调片段补成动态padding版

下面是明确修改后的教学版本，不是原文原样代码。修正输入批次长度处理，同时去除不再需要的原文text列；保存模型、验证划分、指标和超参搜索仍需你按数据补充，本次不训练。

~~~python
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

def few_shot_with_padding(base_model, base_tokenizer, examples):
    if not examples:
        raise ValueError("examples不能为空")
    raw = Dataset.from_list(examples)
    def encode_example(example):
        encoded = base_tokenizer(example["text"], truncation=True, max_length=128)
        encoded["labels"] = example["label"]
        return encoded
    encoded = raw.map(encode_example, remove_columns=raw.column_names)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=base_model,
        args=args,
        train_dataset=encoded,
        data_collator=DataCollatorWithPadding(base_tokenizer),
    )
    trainer.train()
    return base_model
~~~

examples形如[{"text":"a sentence","label":0}, ...]。label必须与分类头同义，例如不能把情感的0/1直接塞进原NLI头而不约定含义；真正换任务应建立合适的分类头与标签映射。

## 六、Use It：使用现成工具

原文按分类、零样本、句向量、翻译和生成列工具，可作为候选列表，不是最新唯一选择。先核对语言、许可、显存、token长度，再做本领域对比。few-shot的100～500条可作为试验预算，而非保证达到英文效果95%～98%。模型未预训练覆盖的语言，也不能仅靠共享词表保证零样本质量。



## 七、失败模式、原文边界与工程注意

1. 推理建议model.eval()，并用no_grad；禁用梯度不自动关闭dropout。
2. 三类softmax的entailment分数按标签分别计算，不等于候选标签之间归一化概率。
3. 高学习率可能损伤迁移，但不会必然使模型变成“仅会英文”。
4. byte_fallback是训练/词表设计，不是对现有checkpoint随手打开就能兼容。
5. 原文固定3～5倍token税、性能比例与源语言优势需在具体tokenizer与数据上验证。
6. 分语言、实体、方言、拼写变体及跨语言一致性都要测；总平均会掩盖低资源语言失败。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
列目标语言、领域、任务和标注量；比较英文源与相关高资源源语言；选择支持对应文字的模型，记录tokenizer覆盖与fertility。建立分语言held-out准确率/宏F1、实体测试与同义翻译一致性评价，再决定是否增加目标语言少样本。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：英、法、印地、阿拉伯语各10句手工标签，运行零样本分类，逐语言报告，不预设哪种语言一定更好。
2. Medium：建立混合语言语料，以英文查询检索所有语言文档，明确gold并测Recall@5。
3. Hard：比较英文源微调与印地语源微调，然后在两条路线中使用同一500条目标语言样本；控制初始模型、标签和划分，报告印地语增益。

## 十、术语与完成标准

multilingual：多语言；cross-lingual transfer：跨语言迁移；zero-shot：无目标训练标签；few-shot：少量目标标签；fertility：分词膨胀；language-specific evaluation：分语言评估。

完成检查：

- 能区分两种zero-shot含义。
- 能核查entailment编号。
- 能补动态padding。
- 能按语言汇报而非只报总分。

## 十一、验证范围与来源许可

原课 Build It 共 3 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

