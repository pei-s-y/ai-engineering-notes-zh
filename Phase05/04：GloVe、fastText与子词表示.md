# 04：GloVe、fastText与子词表示

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-04 增补「零、初学者先读」、概念讲解、选型表与延伸阅读；当日与原课网页版核对，Build It 代码与三步结构一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md)。原题：GloVe, FastText, and Subword Embeddings。

## 零、初学者先读：三个工具各自接住上一课的哪个问题

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要回看词向量是什么时，先读[第03课](<03：词向量与从零实现 Word2Vec.md>)的零节。

### 0.1 先理解这一课在做什么

[第03课](<03：词向量与从零实现 Word2Vec.md>)留下两个没解决的问题，本课的三件事分别接住它们：

1. Word2Vec是一边读句子一边更新参数（在线学习）。另一派研究（LSA等）直接分解共现统计表。两条路线谁更好？**GloVe**的回答是：设计好损失的矩阵分解能匹敌甚至超过Word2Vec，还更便宜。
2. 训练时没见过的词怎么办？**fastText**的答案：把词拆成字符片段，词向量=片段向量之和；新词只要片段见过就能拼出向量。
3. 到了Transformer时代，词级词表撑不到上百万那么大，语言却是开放的。**BPE**学习“常见片段当单位”的切分规则，任何文本都切得开、永不OOV——现代每个LLM分词器都是它的变体。

注意三者不是同一层面的工具：GloVe和fastText学的是**词向量**，BPE学的是**切分规则**，不产生任何向量。

### 0.2 本课术语先翻成白话

- 共现矩阵 co-occurrence matrix=一张大表，X[i][j]记词j在词i周围出现多少次
- 矩阵分解 factorization=找两组小表，让它们乘起来尽量还原那张大表
- 字符n-gram=连续n个字符的片段，如where里的whe、her
- 词形变化 morphological variants=同一个词的不同形式，play/playing/played
- 合并 merge=BPE把当前最常见的相邻单位对绑定成一个新单位
- `</w>`=词尾标记，让BPE知道一个片段是否顶到词的末尾
- byte fallback=遇到不认识的字符，退回用原始字节表示
- OOV=词表里没有的词

### 0.3 用小数字走一遍

语料只有`cat sat`与`cat slept`、window=1：cat与sat共现一次，cat与slept共现一次；距离1的贡献是1，距离2的贡献是1/2——GloVe先把这些加权计数**全部做完**，再训练向量去拟合。若某对的“点积+偏置”算出1.5而log(1)=0，误差1.5就是要修正的量，更新会把它推低。

fastText：where拆成`<wh`、`whe`、`her`、`ere`、`re>`、`<where>`等片段；未见词whereupon与where共享her、ere等片段，向量因此靠近——共享只代表参数可复用，不证明两词同义。

BPE：lowest初始切成l,o,w,e,s,t,</w>；若(l,o)、(e,s)等对在语料别处高频，会被先合并，lowest最终可能切成low + est</w>。

以上是用于理解的算例，不是新增的训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

三部分互相独立，按顺序读：build_cooccurrence + glove_train（GloVe）；char_ngrams + fasttext_vector（fastText）；learn_bpe + apply_bpe（BPE）。与第03课的关键区别：BPE的输入不是句子列表，是“词→频数”的Counter；GloVe的训练对象是共现计数字典，不是词对列表。

### 0.5 理解检查（含参考答案）

**问题1：** GloVe和Word2Vec最根本的差别是什么？

**参考答案：** Word2Vec在线更新（遇到一个词对更新一次），GloVe先把全局共现统计做完，再让向量拟合对数共现；效果相近，GloVe训练成本更低。

**问题2：** 为什么fastText能处理没见过的词？

**参考答案：** 词向量是字符片段向量之和；新词只要片段训练时见过就能拼出向量，全部片段都没见过才返回None。

**问题3：** BPE学的是词向量吗？

**参考答案：** 不是。BPE学的是“怎样切词”的有序合并规则；向量是后续模型embedding层的事，两者不能混。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

Word2Vec之后还要回答：能否直接利用全局共现统计？没见过的词怎样表示？本课依次实现GloVe、字符n-gram组成向量、带词尾标记的字符级BPE。三者不是同一层面的工具：前两者学词向量，BPE学切分规则。

## 二、核心概念与推导

### 2.1 GloVe：先把全局共现数完，再拟合

原课的动机脉络：Word2Vec之前已有直接分解共现矩阵的路线（LSA、HAL），GloVe证明带精心设计损失的矩阵分解可以匹敌甚至超过Word2Vec，且训练更便宜。

GloVe先统计X_ij：在词i的窗口中，词j出现的加权次数，窗口距离r贡献1/r。补充（原文未展开）：目标是log(X_ij)而不是X_ij本身，因为词频跨度极大——the可能共现百万次而专有名词只共现两次，对数把乘性差异压成加性，回归目标数值才稳定。训练目标可以写成：
$$J=\sum_{i,j:X_{ij}>0}f(X_{ij})(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$$
逐项读：对每个共现过的词对，希望“点积+两个偏置”接近log共现次数；f(X_ij)是权重函数，(x/x_max)^0.75在低频段随共现次数上升、超过x_max后饱和为1，避免(the,and)这类高频对主导损失。偏置吸收频率差异，向量解释共现结构。代码系数少一个平方损失的常数2，可等价吸收到学习率中。最终词向量取W+W_tilde之和，这是原课提到的已发表技巧，通常优于只用一张。

### 2.2 fastText：词是它的片段之和

fastText把词包上边界< >，提取3～6字符片段，词向量=所有片段向量之和，再按Word2Vec的方式训练。未见词若有已知片段可获得向量；所有片段都不认识时，本教学函数返回None。

补充（原文未展开）：这一设计对词形丰富的语言收益最大——芬兰语、土耳其语一个词根能变化出成千上万形式，词级词表存不下，片段级能全部复用。

### 2.3 BPE：学习切分规则，不是学向量

本课BPE输入是“词→频数”的Counter；以字符及</w>开始，不是256字节初始化。算法一句话：数所有相邻片段对的加权频次，把最常见的合并成新单位，重复k次。结果：高频片段（ing、tion、est）成为单token，罕见词拆成熟悉碎片，任何文本都切得开。真实GPT/BERT/T5分词器学习3万～10万次合并，序列长度有界、无OOV。

它学习有序合并规则，不自动学embedding。字符级训练字母表不完整时，不能保证无OOV；真正永不OOV的是字节级BPE（GPT-2方案：从256个字节起步，任何字符至少落到字节上）。不要把本课字符版与完整字节覆盖的BPE混为一谈。

### 2.4 用一个小例子把概念串起来（补充）

假设语料只有 `cat sat` 与 `cat slept`，window=1：cat与sat共现一次，cat与slept共现一次；窗口加大才会引入更远词。GloVe学习的是这些统计约束。fastText则将一个词的字符片段向量合成，例如where与whereupon可共享ere等片段；共享只代表可复用参数，不证明两词同义。BPE与fastText都涉及子词，但前者学习切分规则，后者学习/组合片段向量，任务不同。

## 三、操作前准备

GloVe部分需要NumPy，BPE/字符片段可只用标准库Counter。外部模型示例还涉及fasttext和transformers。
~~~bash
python -m pip install numpy
~~~
只尝试外部示例时，再按环境安装相应库并准备网络与磁盘空间；fastText的cc.en.300.bin不是小玩具文件。不要为了纯BPE练习下载整套模型。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 共现计数与GloVe训练

build_cooccurrence先确定词表，再枚举窗口位置；只保留出现过的词对，所以训练不会对log(0)取值。glove_train有中心表、上下文表与两套偏置；每步根据旧向量计算两侧梯度后再更新，返回W+W_tilde。f(x)随x增加后饱和在1；原文说它专门把高频对降权不够准确：相对低频对，高频对权重更高，只是不无限线性增加。

~~~python
import numpy as np
from collections import Counter


# 返回首次出现顺序的词表，以及按距离倒数加权的有向共现计数。
def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


# 只优化正共现词对；返回两套词向量之和。
def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
~~~

### 4.2 字符n-gram与组合向量

char_ngrams返回集合，去掉重复片段；集合展示顺序不固定。包括整个带边界词及不同长度滑窗。fasttext_vector只组合表中已知片段。where与whereupon会共享一些片段，应用实际集合交集核对，不凭近似拼写保证语义接近。

~~~python
# 集合包含边界词及不同长度字符滑窗，展示顺序不保证固定。
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
~~~

~~~pycon
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
~~~

~~~python
# 只对已知片段求和；一个也不认识时返回None。
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
~~~

### 4.3 BPE学习与编码

每个词的相邻对计数要乘频数；每轮重建当前切分，规则按学习顺序应用。合并仅发生在一个词的当前token边界上。原文10次合并的lowest结果是参考，应按频数和平局策略实测。new_vocab使用赋值而非累加，若扩展成可能碰撞的表示要复查。

~~~python
# 输入词频Counter，每轮按频数累计当前相邻对。
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


# 对新词按已有合并表处理，不重新统计或训练。
def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
~~~

~~~pycon
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
~~~

## 五、把代码运行起来：补充入口与检查

下面是无需NumPy的补充入口：先执行原文char_ngrams、learn_bpe、apply_bpe的定义，并导入Counter。
~~~python
from collections import Counter
a = char_ngrams("playing")
b = char_ngrams("played")
print("shared:", sorted(a & b))
print("Jaccard:", len(a & b) / len(a | b))
corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
merges = learn_bpe(corpus, k_merges=10)
print(merges)
print(apply_bpe("lowest", merges))
~~~
Jaccard是交集大小除以并集大小，不是共同字符串的总长度。若要运行GloVe，可用：
~~~python
docs = ["the cat sat".split(), "the dog sat".split()] * 10
vocab, counts = build_cooccurrence(docs, window=2)
W = glove_train(vocab, counts, dim=8, epochs=20)
assert W.shape == (len(vocab), 8)
assert np.isfinite(W).all()
~~~
后一段需要NumPy，本次不把未执行的优化过程标为实测。

## 六、Use It：使用现成工具

原文fastText调用会下载英文模型，再对已见或未见词组合向量。transformers示例只加载GPT-2分词器，不加载完整生成模型；Ġ是其可视化空白约定，不是原文中的普通字母。任何下方token列表均为原课参考，具体以模型文件与版本为准。预训练模型必须配套分词器；只有从零训练或明确调整模型输入层时才自行设计新词表。

~~~python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
~~~

~~~python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
~~~

~~~text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
~~~

原课给出“何时选哪个”的速查表（思路参考，具体任务仍需自己验证）：

| 场景 | 选择 |
|---|---|
| 要预训练通用词向量，不担心OOV | GloVe 300d |
| 要处理拼写错误、新造词或词形丰富的语言 | fastText |
| 任何要进Transformer的文本 | 用模型自带分词器，绝不更换 |
| 从零训练自己的语言模型 | 先在自己的语料上训练BPE或SentencePiece |
| 生产级文本分类配线性模型 | 仍是TF-IDF（[第02课](<02：词袋模型、TF-IDF与文本表示.md>)） |

## 七、失败模式、原文边界与工程注意

1. fastText不是只靠拼写就保证得到正确含义；未见词组合质量依赖片段和训练语料。
2. GloVe和Word2Vec为静态词向量，仍有多义词问题。
3. BERT的WordPiece、T5常见的Unigram/SentencePiece不应统称完全相同的BPE算法；SentencePiece还是工具框架。
4. 字符级BPE并不自动覆盖新字符；字节覆盖、字节回退和[UNK]策略须单独明确。
5. 词表32k或100k不是跨语言通用最优值；“最终约2～3字符/token”也不是定理。
6. 原文使用GloVe300d等作为候选，不等于所有线性分类都必选TF-IDF或生产都禁用词级分词。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
为给定任务分别说明：词向量方案与切分方案、语言与字符覆盖、词表目标与理由、库调用和模型对应关系。区分从零训练与微调，核对OOV、内存、下载及可复现性。不要给预训练模型随意换词表，不把固定词表大小或某种工具当成所有语言的唯一解。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：比较playing和played的字符集合，计算Jaccard并展示实际共享片段。
2. Medium：追踪每轮BPE合并后token总数与词表增长；明确字符数是否含</w>、空白，不能混用字符/token与token/字符。
3. Hard：在莎士比亚语料训练约1000次合并，比较常用词、专名及留出文本。先保留语料版本，记录平局规则、实际合并次数和未知字符行为。

## 十、术语与完成标准

co-occurrence：窗口内共现统计；subword：字符片段或学习的子词；GloVe：拟合对数共现结构；fastText：片段组合词向量；BPE：高频相邻单位合并；byte fallback：用基础字节补覆盖。

完成检查：

- 能手算距离1与2的共现贡献。
- 能解释GloVe的两张表和偏置。
- 能区分“学词向量”与“学token切分”。
- 能用实际集合而非示例文字核对OOV片段覆盖。

### 延伸阅读（原课推荐）

- [Pennington, Socher, Manning (2014), GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)：GloVe论文，七页，损失函数最好的推导。
- [Bojanowski et al. (2017), Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)：fastText子词论文。
- [Sennrich, Haddow, Birch (2016), Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：把BPE引入现代NLP的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)：BPE、WordPiece、SentencePiece实践差异对照。

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

