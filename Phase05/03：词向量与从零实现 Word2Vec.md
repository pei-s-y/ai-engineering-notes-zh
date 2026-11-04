# 03：词向量与从零实现 Word2Vec

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-04 增补「零、初学者先读」、概念讲解与延伸阅读；当日与原课网页版核对，Build It 代码与五步结构一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md)。原题：Word Embeddings — Word2Vec from Scratch。

## 零、初学者先读：为什么要让每个词变成一组数字

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补“向量、点积、余弦”的直觉时，先回看[第02课](<02：词袋模型、TF-IDF与文本表示.md>)的向量部分。

### 0.1 先理解这一课在做什么

上一课TF-IDF把每个词放在自己独立的坐标轴上：dog是一条轴，puppy是另一条轴，两词相似度永远是0。本课换一种思路：给每个词学一组数字（例如16个数），让经常出现在相似上下文里的词，向量自动靠近。就像把词放到一张地图上，意思接近的词住得近；这张地图不是人画的，而是程序读大量句子后自己学出来的。

判断标准只有一句话，语言学上叫**分布假设（distributional hypothesis）**：Firth在1957年说 “You shall know a word by the company it keeps”（看一个词跟谁经常一起出现，就能知道它是什么意思）。cat周围常出现sat、mat、sleeps，kitten周围也常出现这些词，所以两者的向量应该靠近。

### 0.2 本课术语先翻成白话

- embedding（词向量）=一个词对应的一组可训练数字，比如16个
- Skip-gram=给中心词，猜周围词（cat → the, sat, on）
- CBOW=给周围词，猜中心词（the, sat, on → cat）
- 窗口 window=向左向右各看几个词；window=2就是最多各看2个
- one-hot=只有对应位置是1、其余全是0的“编号”向量
- 负采样 negative sampling=每次训练随机抽几个错误答案当陪练，避免对整个词表逐个算分
- sigmoid=把任意实数分数压到0～1之间的函数，σ(0)=0.5
- 梯度下降=沿让损失变小的方向微调参数
- 余弦相似度=只比较向量方向、不管长度的相似度指标
- OOV（out of vocabulary）=训练时没见过、词表里查不到的词
- 静态/上下文向量=一个词永远一个向量 / 同一个词每次出现的向量随上下文变化

### 0.3 用小数字走一遍

句子`a b c`、window=1：a只能看到b，c只能看到b，b两边都能看，所以产生4个有向训练对：(a,b)、(b,a)、(b,c)、(c,b)。

训练开始时假设某正样本对的点积是0：σ(0)=0.5。我们希望正样本的σ接近1，梯度系数是σ−1=−0.5，更新会让点积变大；负样本希望σ接近0，系数是+0.5，更新方向相反。所谓“训练200轮”，就是把这两件小事对每个词对反复做。

类比检索king−man+woman：不是魔法，(king−man)大致捕捉“皇室”这个方向，把它加到woman上，就落在“皇室女性”区域附近——这是大规模语料上观察到的现象，小语料训不出来，也不是恒等式。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

按顺序读五个函数，每个都只做一件事：skipgram_pairs把句子变成训练对 → init_embeddings生成两张随机表 → train_pair做一次梯度更新 → train套上循环 → nearest/analogy在训练结果上查询。核心数据结构两个：词表{词:编号}字典；W与W_prime两张(V,d)矩阵，第i行就是编号i那个词的向量。W[c]是词c当中心词时的向量，W_prime[o]是词o当上下文时的向量——同一个词在两张表里有两个不同的向量。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么一个词需要两个向量（W一行、W_prime一行）？

**参考答案：** 一个词当“被预测的中心”和当“周围的上下文”是两种角色，分开存储让训练更稳定；训练完成后通常只保留W当词向量。

**问题2：** 负采样和softmax是一回事吗？

**参考答案：** 不是。softmax要在整个词表上归一化，词表10万时每个训练对都要算10万次分；负采样把“多分类”换成“一个正样本对几个随机负样本”的二分类，每个训练对只算几次。

**问题3：** 查不到`Zoomer-approved`这个词怎么办？

**参考答案：** 这是OOV。经典Word2Vec没有fallback，直接查不到；[第04课](<04：GloVe、fastText与子词表示.md>)的fastText用字符片段拼向量解决这个问题。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

上一课把每个词放在独立坐标轴上，dog 与 puppy 的相似性不会自动出现。本课学习把词映射到低维稠密向量：上下文相似的词，有机会得到接近的表示。目标是理解“训练任务怎样塑造表示”，而不是用两句话语料证明模型已经理解语言。

原课的类比 king−man+woman≈queen 是某些训练模型上的观察，不是向量恒等式，也不是所有词向量必然满足的性质。

## 二、核心概念与推导

### 2.1 分布假设与两种预测方向

本课所有设计的出发点是分布假设：上下文相似的词，语义大概率相似。把“预测什么”定义成训练任务，词向量只是训练的副产品。

- **Skip-gram**：给中心词，预测周围词。cat → (the, sat, on)。训练慢，但对低频词更好，成为默认选择。
- **CBOW**：给周围词，预测中心词。(the, sat, on) → cat。训练快，低频词表示较弱。

window=2 表示左右各看最多两个位置，不包含中心位置，也不跨文档配对。补充（经验倾向，非硬规则）：窗口偏大捕捉到的多是主题关系，窗口偏小更贴近语法搭配关系。

### 2.2 网络结构与两张表为什么都需要

原课把Skip-gram画成一个两层网络，中间层没有非线性激活：

~~~text
one-hot(中心词) ── W ──▶ 隐藏层(d维) ── W' ──▶ 对全词表的softmax
                              ↑
                        这层的权重就是词向量
~~~

设词表大小V，维度d，W与W_prime都是V×d。W[c]是词c当中心词的向量，W_prime[o]是词o当上下文词的向量。one-hot乘W在数学上等价于“按行查表”，所以代码从不真的创建巨大的one-hot，直接写W[编号]。训练完成后丢掉输出层，只留W当词向量；也可以在明确约定下组合两张表。Skip-gram和CBOW共用这个结构，只是预测方向相反。

### 2.3 为什么要负采样，目标是怎样的

对全词表做softmax，意味着每个训练对都要对V个词各算一次分。V=10万时这是主要计算开销。**负采样（negative sampling）**把多分类改成二分类：只问“这个上下文词是不是真的出现在中心词附近”，再随机抽k个词冒充错误答案。计算量从O(V)降到O(k)，k通常只有5～20。

对正对(c,o)及负样本n，损失为：
$$L=-\log\sigma(v_c^\top u_o)-\sum_n\log\sigma(-v_c^\top u_n)$$

逐项读：第一项希望正对的点积大（σ接近1），第二项希望负对的点积小（σ接近0）。sigmoid把点积变成0～1的得分；它不是整个词表上归一化的softmax分布。对正样本，点积梯度是σ(score)−1（预测不足时接近−1，猛推）；对负样本是σ(score)（预测不足时接近0，几乎不推）。这就是train_pair中的两种系数。

### 2.4 静态向量的能与不能

相近上下文不等于同义词：反义词good/bad也可能出现在相似句式里，向量因此接近——这是分布假设的固有特点，不是实现bug。

一个词只有一个向量（**静态 embedding**）：bank的河岸义和银行义共享表示，这就是原课说的**多义词墙（polysemy wall）**；table的表格义和家具义同理。下游若还读取上下文仍可消歧，不能说整个系统必然无法区分；BERT之后的**上下文向量（contextual embedding）**让同一个词每次出现都有不同向量，那是第07阶段Transformer的内容。另一个限制是OOV：训练没见过的词查不到向量，fastText用子词组合解决（第04课）。

### 2.5 用一个小例子把概念串起来（补充）

以句子 `a b c`、window=1 为例，a只能看b；b能看a和c；c只能看b，因此4个有向训练对。假设正对的点积为0，σ(0)=0.5，正样本梯度系数为−0.5，梯度下降会推动正对点积增加；负样本同为0时系数+0.5，更新方向相反。你需要能解释这个符号，而不必把训练200轮当作理解的证明。

## 三、操作前准备

需要NumPy；现成工具示例还需gensim。
~~~bash
python -m pip install numpy gensim
~~~
原文train依赖上一课build_vocab，但没有在本课重写。先补齐这个函数，再依次执行Build It：
~~~python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
~~~
模型索引必须与词表一致；训练新词表后不能套用旧矩阵。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 构造中心词—上下文正样本

输入是分好词的文档列表。双层位置循环枚举窗口，边界用max/min截断；同一位置跳过。长度5、window=2会产生14个有向词对。词对(c,o)与(o,c)可同时出现，不是重复代码错误。原文带>>>和省略号的块是交互示例，不是完整Python脚本。

~~~python
# 输入预分词文档，返回有向(中心词,上下文词)对。
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
~~~

~~~pycon
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
~~~

### 4.2 初始化中心表与上下文表

随机种子控制初始化；两张表形状都为(V,d)，但数值不同。小随机值打破对称性，不会一开始就具有语义。seed相同有助复现，不保证不同软件、多线程训练绝对逐位一致。

~~~python
import numpy as np


# 固定随机种子初始化两张V×d浮点矩阵。
def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
~~~

### 4.3 计算并更新负采样梯度

先读取旧向量，计算正负得分，再构造中心向量梯度，最后更新两张表。W[context_idx]=W[context_idx]是原文无作用语句，保留以便对照。负样本可重复，重复项会多次贡献梯度。不要把负样本当成“在整个语料中绝不共现”的词，它只是采样训练标签。

~~~python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# 使用当前正词及采样负词，对两张词向量表做一次梯度下降。
def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
~~~

### 4.4 组织训练循环

每个epoch打乱词对，对每对抽负样本。原文均匀抽样，并过滤中心与当前正词，过滤后可能少于k_neg个；这不是经典Word2Vec按词频的3/4次幂采样的完整实现。返回词表及W，不包含分类器或自动打印的准确率。

~~~python
# 先建词表，再逐轮打乱正词对并采样负例。
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
~~~

### 4.5 近邻与类比检索

先把各词向量和查询向量做归一化，再算余弦，按相似度降序排序。exclude存的是编号，不是字符串。analogy(a,b,c)实际构造b−a+c，并排除输入三词。Google News上的queen示例只是原课参考，不是小语料训练结果。

~~~python
# 余弦近邻检索；exclude中的元素是词表编号。
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


# 返回b−a+c向量的近邻，不保证存在语言学上的正确答案。
def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
~~~

~~~pycon
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
~~~

## 五、把代码运行起来：补充入口与检查

定义前述函数后，以下是补充的完整最小入口（需要NumPy，本次未执行训练）：
~~~python
docs = [
    "the cat sleeps on the mat".split(),
    "the dog sleeps on the rug".split(),
    "a cat likes food".split(),
    "a dog likes food".split(),
] * 5
vocab, W = train(docs, dim=16, window=2, epochs=200, seed=0)
print(W.shape)
print(nearest(vocab, W, W[vocab["cat"]], exclude={vocab["cat"]}))
~~~
预期第一项形状为(10,16)；请以实际词表长度核对，不要把维度当准确率。近邻顺序依赖数据和训练，dog未必进前三。先做结构检查：
~~~python
assert W.shape == (len(vocab), 16)
assert len(skipgram_pairs([["a", "b", "c"]], window=1)) == 4
~~~
训练损失下降、邻居合理、下游任务有效是不同层面的检查。

## 六、Use It：使用现成工具

原文gensim示例完整保留在下方：sg=1选择Skip-gram，negative=5选择负采样数量，min_count=1保留低频词。model.wv存训练后的词向量；most_similar返回近邻而不是分类结果。两句语料不足以得到可靠语义。预训练Google News、GloVe和fastText需要另行下载，占用明显内存，本文未下载。

~~~python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
~~~

原课还列出2026年Word2Vec仍占优的四类场景（思路参考，非本机实测）：在小规模专业语料（如医学摘要）上快速自训一套领域向量；用类比差值（如性别方向）做特征工程，公平性研究仍在用；100维小到可以用PCA或t-SNE画图直接观察聚类；以及在无GPU设备上做单行查表式推理。

## 七、失败模式、原文边界与工程注意

1. 未登录词不能直接索引W；应检查词表，不能把KeyError当作训练崩溃。
2. 单词与自己比较通常最相似；评估近邻时排除自己。
3. 原文“范数低于0.01或高于100就是bug”过于绝对。记录范数分布、非有限值与变化趋势，结合初始化和训练规模判断。
4. 小语料不保证学出类比，也不能用PCA/t-SNE图证明原空间距离关系或公平性。
5. 去掉一条所谓性别方向不等于消除所有偏见。偏见实验需说明语料、词表、指标与社会语境。
6. “一小时训练”“某年默认方案”等不是本机实测或普遍保证。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
检查词向量：说明词表与向量维度；做三组明确a、b、c顺序的类比和五组领域词近邻；检查余弦对称性、零/异常向量；报告缺词情况。不要只凭类比成功宣称模型好；同时设计下游评估。所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课Easy：用20句猫狗语料训练约200轮，检查cat的前三近邻。验收为能复现实验、排除自身并解释结果；不是硬性保证dog必出现。
2. Medium：加入高频词下采样，比较低频词近邻。阈值10^-5只是超参数；需明确采用的保留概率公式，不把所有高频词全部删除。
3. Hard：在20 Newsgroups训练，比较he−she、doctor−nurse方向上的职业词投影。记录分词、训练划分、随机种子和样本词；投影是表示偏差探针，不是对个人或职业的价值结论。

## 十、术语与完成标准

embedding：词的可学习向量；Skip-gram：中心预测上下文；CBOW：上下文预测中心；negative sampling：对采样词对做二分类；static/contextual：是否随当前上下文改变；OOV：当前表中没有的词。

完成检查：

- 能从一句话列出窗口词对。
- 能推导正负两类得分梯度的符号。
- 能说明(V,d)矩阵每行是什么。
- 能解释小样本、OOV、多义词和偏见评估的限制。

### 延伸阅读（原课推荐）

- [Mikolov et al. (2013), Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)：负采样目标的出处，篇幅短、可读性好。
- [Rong, X. (2014), word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)：最清晰的梯度推导，觉得原文数学过密时看这份。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html)：生产可用的训练参数设置。

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

