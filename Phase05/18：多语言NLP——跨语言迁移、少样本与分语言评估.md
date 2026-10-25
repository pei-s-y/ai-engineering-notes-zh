# 18：多语言NLP——跨语言迁移、少样本与分语言评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/18-multilingual-nlp) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/18-multilingual-nlp/docs/en.md)。原题：Multilingual NLP。

## 一、这一课解决什么问题？

英文有数据，不代表中文、印地语、阿拉伯语也有相同覆盖。本课学习利用共享模型跨语言迁移，并检验每种语言的效果，而非只看总体均分。

## 二、核心概念与推导

多语言模型共享词表和参数，使不同语言可借助共同表示迁移任务。共享子词并不保证所有同义词天然对齐，跨语言能力取决于预训练、语言覆盖和下游数据。

zero-shot跨语言迁移：在源语言训练，在没有目标语言标注训练的条件下测试目标语言。few-shot：加入少量目标语言标签微调。本课第一段还有“用NLI做零样本类别分类”的另一层zero-shot含义，需与跨语言迁移分开。

模型谱系：mBERT/XLM-R偏编码理解；XLM-V扩展词表；mT5偏生成；NLLB偏翻译；BLOOM、Aya等是多语言生成模型示例。课程型号/语言数是源文背景，不当作当前排行榜或可用性承诺。

源语言不必永远英文：可以比较语言亲缘、类型特征、数据量、领域与标注质量。原文LANGRANK与qWALS是来源不同的方法，具体研究结论不应简化成“相近语言必胜”。

分词fertility指每词产生多少token，影响上下文与成本。Unicode正规化、变体和稀有文字覆盖也影响表现；byte fallback可避免未知字节丢失，但不保证模型理解。

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

