# 06：命名实体识别——BIO、规则、CRF与神经模型

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/06-named-entity-recognition) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/docs/en.md)。原题：Named Entity Recognition。

## 一、这一课解决什么问题？

NER识别文本中的实体边界与类型，例如Apple作为ORG、US作为GPE。并不是所有名词都要标注；search deal是否算实体取决于标注规范。实体体系、字符/token边界和领域约定是模型之前必须确定的内容。

## 二、核心概念与推导

BIO：B-X开始一个X类实体，I-X继续同类实体，O表示非实体。New/B-GPE York/I-GPE City/I-GPE连成一个跨度。BILOU另加U单词实体、L最后位置。

本课span使用token索引[start,end)，end不包含在实体中；它不是字符偏移。要恢复原文字符位置，需保存tokenization offsets。

路线：字典/规则→HMM→CRF→BiLSTM+CRF→Transformer token classifier。CRF对整条标签序列建模，既有每位置emission，也有标签transition；它不是简单给每词独立取argmax。BiLSTM输出上下文特征，CRF再决定序列。

单层BIO只能给每位置一个标签，无法直接表示重叠/嵌套实体。实体级严格F1要求边界和类型同时匹配；token准确率可能被大量O标签抬高。

### 2.9 用一个小例子把概念串起来（补充）

`New York hired Alice`的token编号为0、1、2、3。若New York是地点，则span=(0,2,GPE)，标签是B-GPE、I-GPE、O、B-PER。0到2是token区间，不是字符串字符位置。模型可以逐词预测出非法的O→I-GPE序列，CRF通过序列打分帮助建模转移；仅有Linear输出logits并没有自动完成CRF解码。

## 三、操作前准备

前三个手写步骤仅需标准库；后续按需使用：
~~~bash
python -m pip install sklearn-crfsuite torch pytorch-crf spacy transformers
python -m spacy download en_core_web_sm
~~~
pip包pytorch-crf通常以torchcrf导入，不要与名字近似的其他包混淆。HF调用会下载检查点。不同模型标签集可能是PER/LOC/ORG/MISC或PERSON/GPE/PRODUCT，必须先对齐规范，不应把标签不同直接看作模型错误。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 BIO与跨度转换

spans_to_bio按给定跨度填标签；原函数不检查越界、空跨度和重叠。bio_to_spans在B标签启动，在匹配的I标签延伸；孤立I可能被丢弃，这是原实现的容错策略，不是BIO唯一标准。先定义是否合法，再检查roundtrip。

~~~python
# span使用token索引[start,end)，本函数不校验重叠或越界。
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


# 按当前标签序列收集实体，不匹配的孤立I按原逻辑丢弃。
def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
~~~

~~~pycon
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
~~~

### 4.2 构造词级与上下文特征

lower、大小写、数字、后缀、形状与邻词作为输入。word_shape把每个大写变X、小写变x、数字变d，保留标点。BOS/EOS表示句边界，不是原文真实单词。注意用真正的邻词而不是标签泄漏。

~~~python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


# 逐字符编码大小写与数字，标点保持原样。
def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
~~~

### 4.3 字典规则基线

三个gazetteer按集合精确匹配，一次仅标一个token，全部使用B标签。它不认识New York这样的多token名称，也不能区分水果Apple与公司Apple。规则不是必然零泛化，但这份集合匹配只覆盖已列词。

~~~python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


# 精确匹配三个单token实体字典，不做多词消歧。
def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
~~~

### 4.4 使用CRF库

sentences_tokenized是分词句子列表，bio_labels_train是逐句等长标签列表，源片段未提供训练集，需要补充。c1/c2为L1/L2系数；all_possible_transitions只是允许学习更多转移参数，不会把非法BIO转移硬设为负无穷。严格合法约束需显式实现。

~~~python
import sklearn_crfsuite

# 给CRF构造逐token特征字典，训练与预测必须一致。
def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
~~~

### 4.5 BiLSTM产生发射分数

类名虽然叫BiLSTM_CRF_Head，实际只包含Embedding、双向LSTM与Linear，返回(B,T,L)的emissions，并没有CRF层、CRF损失或Viterbi解码。补上CRF并正确传mask后才是完整BiLSTM-CRF，不能把原代码forward当最终标签。

~~~python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    # 返回发射分数，不包含CRF损失或序列解码。
    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
~~~

## 五、把代码运行起来：补充入口与检查

可先运行不依赖第三方库的闭环测试：
~~~python
tokens = ["New", "York", "hosts", "OpenAI", "."]
spans = [(0, 2, "GPE"), (3, 4, "ORG")]
labels = spans_to_bio(tokens, spans)
print(labels)
assert bio_to_spans(tokens, labels) == spans
print(word_shape("USA-2024"))
print(rule_based_ner(["Apple", "in", "US"]))
~~~
预期labels为['B-GPE','I-GPE','O','B-ORG','O']，形状字符串为XXX-dddd。

补充最小CRF数据以试通接口（只是API演示，不是可靠训练集）：
~~~python
sentences_tokenized = [
    ["Apple", "in", "US"], ["Google", "in", "France"],
    ["OpenAI", "uses", "ChatGPT"]
]
bio_labels_train = [
    ["B-ORG", "O", "B-GPE"], ["B-ORG", "O", "B-GPE"],
    ["B-ORG", "O", "B-PRODUCT"]
]
~~~
在执行原文crf.fit之前定义它们，预测时使用相同to_features。若要试BiLSTM头：
~~~python
head = BiLSTM_CRF_Head(100, 16, 8, 5)
ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
assert head(ids).shape == (1, 3, 5)
~~~
这个形状检查不等于NER效果评估。

## 六、Use It：使用现成工具

spaCy打印ent.text/ent.label_；HF aggregation_strategy把模型子词预测聚合为跨度，不保证模型永远识别出原文示例里的每个实体。spaCy与CoNLL风格模型的类别集合不同。原文LLM零/少样本、多阶段候选→解释→判定→复核、检索相近标注例子和按类型多次提取等方案都可作为实验候选，但F1提升11～12%等未经本次复现，也不能拿它保证医学或法律场景可用。

~~~python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
~~~

~~~text
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
~~~

~~~python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
~~~

~~~text
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
~~~

## 七、失败模式、原文边界与工程注意

1. 预训练模型在新领域可能失效，但“必然差于gazetteer”同样不是定律。
2. 大模型或更大spaCy检查点不保证每种类型都更好；用领域实体级评估判断。
3. 长实体聚合只能整理已有标签，不能自动修正漏识别和错类型。
4. LLM输出JSON合法，不代表实体有原文证据；保存原文跨度并检查类型与偏移。
5. 模型选择不能单凭500样本、50ms或98%F1固定阈值；要测实际延迟和错误成本。
6. 原文提示词提出的CoNLL类别审计要结合目标标签体系，不能不检查直接迁移。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
给任务、语言、领域、标签集合、数据与延迟约束，选择规则/CRF/神经候选；说明BIO或span表示、模型和标签映射；设计实体级严格P/R/F1及类型分解；测试嵌套、长实体、领域词和原文偏移。不可将原文参考输出写成已验证结果。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：对10句合法非重叠跨度做BIO往返；再单独测试孤立I、相邻同类型实体和空句，记录处理约定。
2. Medium：在CoNLL-2003训练CRF，用seqeval报告逐实体类型F1；原课约84是参考，不是验收保证。
3. Hard：用领域NER数据微调distilbert-base-cased，与spaCy小模型比较。按文档划分避免同句/同文泄漏，保存子词与实体跨度对齐规则。

## 十、术语与完成标准

span：半开区间实体跨度；gazetteer：实体词典；emission：每位置标签得分；transition：相邻标签转移得分；CRF：条件序列模型；entity-level F1：边界与类型一起匹配。

完成检查：

- 能在token索引与字符偏移间明确区分。
- 能识别原文神经头尚未包含CRF。
- 能解释软转移学习不等于硬BIO约束。
- 能对合法BIO往返并设计非法输入测试。

## 十一、验证范围与来源许可

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

