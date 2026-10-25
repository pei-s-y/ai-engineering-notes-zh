# 07：词性标注与句法分析——MFT、HMM和依存树

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/docs/en.md)。原题：POS Tagging and Syntactic Parsing。

## 一、这一课解决什么问题？

上一课文本处理需要知道词在当前句子中的角色。本课分别研究“每词是什么词性”和“词与词怎样形成句法关系”。这两项不是同一个任务：给词贴NOUN/VERB标签，并没有自动得到依存树。

## 二、核心概念与推导

PTB是英语细粒度标签，如NN、NNS、VBD；UD提供17个粗粒度类别，如NOUN、VERB、AUX。PTB常说36个主要词性标签，实际还可能包含标点和特殊标签，不要把标签数量当所有工具都相同。

成分句法把NP/VP/PP等短语嵌套成树；依存句法用head→dependent并附关系名。root有特殊根约定；其他词通常有一个head。spaCy的关系名不一定与某份UD树库完全一致。

MFT为每个小写词记训练中最多的标签，未见词返回全局多数类。HMM同时利用转移P(t_i|t_(i−1))与发射P(w_i|t_i)，用Viterbi寻找整句最大联合分数，不是每步贪心。

Viterbi递推：当前位置某标签的最佳分数=所有前一标签候选中“前一最佳分数+转移对数概率”的最大值，再加当前发射对数概率；back记录最优前驱，最后回溯。训练计数不是解码结果。

### 2.9 用一个小例子把概念串起来（补充）

在小语料中the常标DET、cat常标NOUN、runs常标VERB，MFT对三个词独立查表。HMM还比较DET→NOUN→VERB这条路径：把log初始转移、各词发射和相邻转移相加；Viterbi保存每个位置/标签最优分数和回溯指针。表格位置是词，状态是标签，不要把Viterbi当成“逐词选最频标签”。

## 三、操作前准备

MFT与HMM只需Counter/defaultdict/math；spaCy需安装库和语言管线：
~~~bash
python -m pip install spacy
python -m spacy download en_core_web_sm
~~~
训练输入是[(tokens,tags),...]，每句两列表必须等长。原文zip会静默截断长度不一致输入，应在喂数据前断言检查。空训练集和空测试句没有被原函数完善处理。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 最常见词性基线

train_mft统计词→标签计数并返回word_best与default_tag；predict_mft按词查表，不看上下文。小写化会丢部分专名线索。原文Brown约85%只能作参考，标签集、划分和OOV比例都会影响数值。

~~~python
from collections import Counter, defaultdict


# 统计每个小写词的最多标签，以及全局多数标签。
def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


# 未知词使用全局默认类，不读取句子上下文。
def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
~~~

### 4.2 HMM训练与Viterbi解码

train_hmm返回的是计数表，alpha真正用于log_prob与解码分母。加平滑后未见词给出非零发射分数。V是(T,K)分数格，back存前驱编号。原文训练了EOS转移却在最后选路径时没有加终止概率；这不是完整包含结束状态的序列似然。tags由set转列表，平局时结果可能受顺序影响。

$$
P(\text{tags}, \text{words}) = \prod_i P(\text{tag}_i \mid \text{tag}_{i-1}) \cdot P(\text{word}_i \mid \text{tag}_i)
$$

~~~python
import math


# 这里只累计转移与发射计数，alpha在解码平滑时生效。
def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


# 给定计数表和显式分母，返回加alpha平滑的对数概率。
def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


# 对标签格做动态规划；原版未把EOS转移加入最终选路。
def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
~~~

### 4.3 理解现代模型的差异

CRF可混合上下文特征；BiLSTM或Transformer提供更广泛上下文。原文说HMM完全不能对saw消歧过于绝对：它能利用局部序列统计，只是表示受限。原文把超过98%直接判断为测试集过拟合也不成立，需要具体评测协议。



### 4.4 依存句法与现成管线

transition-based通过栈/缓冲区和动作建边；graph-based给候选边打分后进行带树约束的解码。Eisner常用于投射树的动态规划，不应把所有图方法都描述成同一种最大生成树算法。原课没有从零实现parser，下面spaCy调用才是这一步可运行代码。

~~~python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
~~~

~~~text
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
~~~

## 五、把代码运行起来：补充入口与检查

标准库最小练习：
~~~python
train_examples = [
    (["the", "cat", "runs"], ["DET", "NOUN", "VERB"]),
    (["the", "dog", "runs"], ["DET", "NOUN", "VERB"]),
    (["a", "dog", "sleeps"], ["DET", "NOUN", "VERB"]),
]
word_best, default_tag = train_mft(train_examples)
print(predict_mft(["the", "cat", "sleeps"], word_best, default_tag))
transitions, emissions, tags, vocab = train_hmm(train_examples)
print(viterbi(["the", "cat", "sleeps"], transitions, emissions, tags, vocab))
~~~
两者在这个简单例子都应得到DET/NOUN/VERB；这不说明HMM没有用，只说明样本不够区分模型能力。再换歧义词或OOV句子并记录变化。

原文HMM访问tokens[0]，空句会出错；外围调用前先做输入检查。若要严格复现平局，可在补充版本中将tags_list排序，并明确这与原版的差别。

## 六、Use It：使用现成工具

spaCy示例显示text、tag_、pos_、dep_与head.text。列出的是原课参考，不保证模型版本对3pm切分完全相同。Stanza、Trankit、NLTK是其他候选，不应仅凭工具名宣称最高准确率。句法分析可辅助方面情感、查询分解和抽取，但不是所有LLM系统内部都一定调用parser，也不能仅凭语法树验证事实正确。



## 七、失败模式、原文边界与工程注意

1. 词形还原是否必须POS取决于lookup/rule等模式，原文的Always过于绝对。
2. 句法不等于语义；修饰归属、介词附着和被动语态会造成抽取错误。
3. 一棵依存树并不自动输出完整主谓宾：并列、零主语、被动结构都要额外规则。
4. 训练/测试要按句子或文档划分，不能把同句token随机拆到两边。
5. 对精度和参考85/93/97/98的比较，要统一标签集和预处理；不要给自己实验预写结论。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
为下游任务选择PTB或UD标签体系、语言模型与依存属性；给出如何消费pos_/dep_/head的调用；测试saw/book/can歧义与介词附着，抽查20例；说明标注模型错误和抽取规则错误怎样分开统计，不把参考准确率写成本机实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：在Brown子集用留出句评估MFT，报告整体与OOV准确率，约85%只作参考。
2. Medium：训练HMM，计算逐标签P/R与混淆，说明EOS和未知词处理约定。
3. Hard：从1000句依存树中抽主谓宾，在50条人工标注上核对。单列被动、并列和省略主语失败案例，不能只展示挑选的成功例子。

## 十、术语与完成标准

POS：词性标签；PTB：英语细粒度标签；UD：跨语言粗标签和依存规范；head/dependent：支配词/依存词；emission/transition：发射/转移；Viterbi：最大分数路径动态规划。

完成检查：

- 能区分词性、成分树与依存树。
- 能解释HMM的计数、平滑、动态规划和回溯。
- 能指出原实现的EOS与空句边界。
- 能把spaCy输出与后续抽取规则连接起来。

## 十一、验证范围与来源许可

原课 Build It 共 4 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

