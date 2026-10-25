# 14：信息检索——BM25、稠密向量、RRF与重排

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/14-information-retrieval-search) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/14-information-retrieval-search/docs/en.md)。原题：Information Retrieval and Search。

## 一、这一课解决什么问题？

词面相同和语义相近是两种不同线索。本课将BM25、独立向量编码、排名融合、交叉编码重排接成候选流水线。原文法律条文例子仅是检索文本示意，不应当作现行法律说明。

## 二、核心概念与推导

BM25为query中每项累加IDF×词频饱和×长度校正。k1调重复次数饱和速度；b调文档长度影响。本代码IDF=ln(1+(N−df+0.5)/(df+0.5))。与上一课TF-IDF公式不同，不可直接混数值。

bi-encoder独立编码文档，可离线预存；cross-encoder把(query,doc)放一起打相关分，适合较小候选池。原课dense_search是全矩阵点积，不是FAISS近似索引。

RRF忽略原分数尺度：
$$score(d)=\sum_{\text{list }l}\frac1{k+rank_l(d)}$$
源代码rank从0枚举，故分母用k+rank+1。k=60是平滑常数，不是检索条数。融合不能召回两个列表都没找到的证据。

单相关文档时query命中率@k与常见Recall@k一致；多相关文档时真正Recall应按找回相关数/总相关数计算。MRR只看首个相关位置，nDCG考虑分级相关性。

### 2.9 用一个小例子把概念串起来（补充）

两份排名分别[文档0,文档1]、[文档1,文档2]，k=60时文档1得1/62+1/61≈0.03252，超过只出现一次的0或2。RRF用名次不混分数尺度，但也丢掉了原分差大小。如果正确文档两路都没出现，任何重排都救不回来，应先提高候选召回。

## 三、操作前准备

BM25和RRF只需标准库。dense/rerank需要：
~~~bash
python -m pip install sentence-transformers numpy
~~~
会下载bi-encoder和CrossEncoder；原文reranker在顶层初始化，执行那段定义就可能开始下载。只测BM25时不必运行外部模型片段。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 从零实现BM25

先按小写字母数字正则切分，统计长度和文档频数。score逐词计分，rank为每个文档重新计数排序，属于教学暴力扫描，不是倒排索引性能实现。query中的重复token会重复贡献。空corpus会显式报错；全空文档只得到无意义的并列零分。

~~~python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


# 教学正则只匹配小写ASCII字母与数字，中文需另定规则。
def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    # 本课BM25的IDF，不是上一课TF-IDF公式。
    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    # 逐词累加BM25，返回相关分数而非概率。
    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    # 暴力扫描所有文档，不能据此承诺百万文档低延迟。
    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
~~~

### 4.2 独立编码的稠密检索

建立(N,D)归一化向量表，query编码(1,D)后点积得到N个余弦。model_id必须匹配语言与任务；embedding模型变化后旧索引也要重建。这里返回(score,doc_idx)列表，后面RRF依赖同一文档编号。

~~~python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
~~~

### 4.3 排名融合

每条排名中同一文档应至多一次；遍历时只使用排名位置，不使用元组中的原始分数。不要把BM25分数与cosine直接相加，也别把RRF分数当概率。

~~~python
# 分母用常数k+从1开始的名次，不混合原始分数。
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
~~~

### 4.4 重排融合候选

先两路取候选并融合，再构造(query,原文)对，让CrossEncoder重新打分。pool_size与top_k分别是候选池与最终返回数。重排无法修复未进候选池的文档，重复块和同父文档也需要去重策略。

~~~python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 融合候选后重排，无法找回两路都未召回的文档。
def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
~~~

### 4.5 检索评估

建立查询与相关文档集合的gold，先测retrieval再测回答；Recall、MRR、nDCG用途不同。原文没有给完整实现或gold，不能只凭几条搜索结果宣布完成评估。



## 五、把代码运行起来：补充入口与检查

标准库可先测两路模拟排名与融合：
~~~python
corpus = [
    "cat sat on mat",
    "dog ran outside",
    "cat and dog need water",
]
bm25 = BM25(corpus)
print(bm25.rank("cat", top_k=3))
rank_a = [(10.0, 0), (2.0, 1)]
rank_b = [(0.9, 1), (0.8, 2)]
fused = reciprocal_rank_fusion([rank_a, rank_b])
assert fused[0][1] == 1
print(fused)
~~~
文档1同时进入两份排名会获得两项贡献。真实混合入口需再执行：
~~~python
encoder, dense_embeddings = build_dense_index(corpus)
# 先定义原文reranker及hybrid_search，再运行：
# print(hybrid_search("cat", bm25, encoder, dense_embeddings, corpus))
~~~
后者需要模型下载，本次不执行。排名不等于回答，检索结果还需供reader处理。

## 六、Use It：使用现成工具

原文按规模列内存BM25、FAISS/pgvector、Elastic/OpenSearch、向量数据库、SPLADE/ColBERT等方案。它们是候选组件，不是按文档数自动得出的唯一架构。应实际测索引大小、更新、权限、过滤和并发。原课强调切块、父子文档、HyDE、版本与CI；其80%失败、k=3最优、8k预算和70～90%幻觉减少不是普遍定律。



## 七、失败模式、原文边界与工程注意

1. HyDE可能扩展错意图，不是免费精度提升，需评估查询类型。
2. 文档长、chunk重复和来源权限会影响top-k，不能只换embedding。
3. 原代码BM25分数并列按元组排序偏向较大doc_idx，非语义差异。
4. 模型指标提升不代表延迟/成本符合约束；不能套用原文10ms或100ms。
5. 多路索引必须共用稳定文档ID、版本、删除与权限过滤。
6. query/文档数据不可包含用于评价的答案标签泄漏。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
为语料与查询明确BM25/dense/融合/重排候选，给embedding与reranker版本；说明候选数、过滤与去重；建立Recall@k、MRR、nDCG和延迟基线。对实体/错误码与改写问题分别评估，不凭原文规模表跳过实验。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：500文档、20查询比较BM25/dense/hybrid的Recall或命中@5，说明gold定义。
2. Medium：实现首个相关文档的倒数排名，未命中记0，分别汇报各路线MRR。
3. Hard：500个领域query-doc对微调dense encoder，留出查询且去重近重复；检查MultipleNegativesRankingLoss中的假负样本，再比较召回与泛化。

## 十、术语与完成标准

BM25：带饱和与长度校正的词项分；bi-encoder：独立编码；cross-encoder：成对编码重排；RRF：名次倒数融合；Recall@k：相关证据召回；MRR：首相关倒数排名均值。

完成检查：

- 能解释BM25的df与词频分工。
- 能手算两个排名的RRF。
- 能区分召回池与重排池。
- 能说明原文暴力代码不具备倒排索引性能。

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

