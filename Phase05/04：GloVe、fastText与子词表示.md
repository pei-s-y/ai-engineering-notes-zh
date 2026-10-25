# 04：GloVe、fastText与子词表示

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md)。原题：GloVe, FastText, and Subword Embeddings。

## 一、这一课解决什么问题？

Word2Vec之后还要回答：能否直接利用全局共现统计？没见过的词怎样表示？本课依次实现GloVe、字符n-gram组成向量、带词尾标记的字符级BPE。三者不是同一层面的工具：前两者学词向量，BPE学切分规则。

## 二、核心概念与推导

GloVe先统计X_ij：在词i的窗口中，词j出现的加权次数。窗口距离r贡献1/r。训练目标可以写成：
$$J=\sum_{i,j:X_{ij}>0}f(X_{ij})(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$$
偏置吸收频率差异，向量解释共现结构。代码系数少一个平方损失的常数2，可等价吸收到学习率中。

fastText把词包上边界< >，提取3～6字符片段，并组合片段向量。未见词若有已知片段可获得向量；所有片段都不认识时，本教学函数返回None。

本课BPE输入是“词→频数”的Counter；以字符及</w>开始，不是256字节初始化。它学习有序合并规则，不自动学embedding。字符级训练字母表不完整时，不能保证无OOV；不要把它与完整字节覆盖的BPE混为一谈。

### 2.9 用一个小例子把概念串起来（补充）

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

