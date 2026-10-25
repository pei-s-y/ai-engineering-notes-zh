# 03：词向量与从零实现 Word2Vec

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md)。原题：Word Embeddings — Word2Vec from Scratch。

## 一、这一课解决什么问题？

上一课把每个词放在独立坐标轴上，dog 与 puppy 的相似性不会自动出现。本课学习把词映射到低维稠密向量：上下文相似的词，有机会得到接近的表示。目标是理解“训练任务怎样塑造表示”，而不是用两句话语料证明模型已经理解语言。

原课的类比 king−man+woman≈queen 是某些训练模型上的观察，不是向量恒等式，也不是所有词向量必然满足的性质。

## 二、核心概念与推导

### 2.1 两种预测方向
Skip-gram：给中心词预测周围词；CBOW：给周围词预测中心词。window=2 表示左右各看最多两个位置，不包含中心位置，也不跨文档配对。

### 2.2 两张表为什么都需要？
设词表大小 V，维度 d，W 与 W_prime 都是 V×d。W[c] 是中心词向量，W_prime[o] 是上下文词向量。one-hot 乘 W 等价于按行查表，不必真的创建很大的 one-hot。训练后常取 W，也可以在明确约定下组合两张表。

### 2.3 负采样目标
对正对(c,o)及负样本n，损失为：
$$L=-\log\sigma(v_c^\top u_o)-\sum_n\log\sigma(-v_c^\top u_n)$$
正样本希望点积大，负样本希望点积小。sigmoid把点积变成0～1的得分；它不是整个词表上归一化的softmax分布。对正样本，点积梯度是σ(score)−1；对负样本是σ(score)，这就是train_pair中的两种系数。

相近上下文不等于同义词：反义词也可能出现在相似句式里。静态词向量每词一个向量，bank的河岸和银行义共享表示；下游若还读取上下文仍可消歧，不能说整个系统必然无法区分。

### 2.9 用一个小例子把概念串起来（补充）

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

