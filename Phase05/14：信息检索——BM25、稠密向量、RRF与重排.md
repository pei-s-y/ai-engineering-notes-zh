# 14：信息检索——BM25、稠密向量、RRF与重排

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充（BM25三因子拆解、bi/cross编码器对照、RRF与指标白话）；当日与原课网页版核对。零节手算例（BM25、余弦、RRF）已用本机Python标准库复核。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/14-information-retrieval-search) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/14-information-retrieval-search/docs/en.md)。原题：Information Retrieval and Search。

## 零、初学者先读：RAG 的地基——把“找证据”拆成一条流水线

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第13课](<13：问答系统——抽取式、检索增强与拒答评估.md>)把“检索+阅读”合在问答流程里讲，本课单独把“检索”那一半拆开吃透；后续[第22课](<22：现代Embedding——稠密、稀疏、多向量与降维评估.md>)与[第23课](<23：RAG切块——固定、递归、语义、父子与上下文增强.md>)都回指这里。

### 0.1 先理解这一课在做什么

用户问“欺骗他人获取财物触犯哪条法律”，正确答案是“第420条 IPC”——两句话没有任何词面重叠，关键词检索抓瞎。反过来，用户搜错误码“ERR_CONN_0x2F”，一个字符都不能差，语义检索反而可能输。**词面相同和语义相近是两种不同的线索**，本课把四件工具接成流水线让它们互相补台：

| 环节 | 工具 | 擅长 | 抓不住 |
|---|---|---|---|
| 第一棒 | BM25（稀疏检索） | 错误码、法条编号、人名等精确匹配 | 换个说法就失效 |
| 第二棒 | 稠密检索（bi-encoder） | 同义改写、语义相近 | 差一个字母的编号可能漏 |
| 接力 | RRF 融合 | 不校准直接合并两份排名 | 两路都没找到的救不回 |
| 裁判 | cross-encoder 重排 | 细粒度相关性最准 | 慢，只能用在小候选池上 |

类比：BM25 像图书馆的卡片目录——书名对得上才找得到；稠密检索像懂意思的管理员——你说“讲猫打盹的书”，他能领你去《小憩的宠物》；RRF 把两位管理员各自开的清单合成一张；重排是最后请资深馆员把三十本候选快速翻一遍，只留最好的五本。这条流水线就是 RAG 系统的地基：阅读器再聪明，证据没捞回来也答不上来。

### 0.2 本课术语先翻成白话

- 检索（retrieval）=给定查询，从文档堆里找出最相关的一小撮
- 稀疏表示（sparse）=向量里绝大多数位置是0；BM25背后就是“数词”的稀疏统计
- 稠密检索（dense retrieval）=把查询和文档都编码成稠密向量再比相似度
- BM25=Okapi BM25，打分=IDF×词频饱和×长度校正三因子
- 词频饱和（term-frequency saturation）=词重复出现的收益递减：出现10次得不了出现1次的10倍分
- IDF（inverse document frequency）=越稀有的词越值钱
- bi-encoder=查询和文档**各自**编码成向量再比相似度；文档向量可离线预存
- cross-encoder=把(查询,文档)**拼在一起**送进模型打一个相关分；准但慢
- RRF（reciprocal rank fusion）=按名次倒数融合多份排名
- 重排（rerank）=对候选池用更准但更慢的模型重新打分
- Recall@k=前k个结果里含相关文档的查询比例
- MRR（mean reciprocal rank）=首个相关文档排得越靠前分越高
- nDCG=考虑“相关程度分级”的排名指标
- 候选池（pool）与重排池=先粗筛一批候选，再在其中精排，两个数量词

### 0.3 用小数字走一遍

**BM25：一个查询词×两篇文档。** 语料只有两篇：D0=`cat cat cat dog`、D1=`cat dog dog dog`（各4词，所以平均长度 avg_dl=4；k1=1.5、b=0.75 用原课默认值）。查询=`cat`。

第一步算 IDF：cat 在两篇都出现，df=2、N=2，IDF(cat)=ln(1+(2−2+0.5)/(2+0.5))=ln(1.2)≈0.182。

第二步逐篇算词频项。两篇长度正好等于平均值，长度校正因子化简为1，分母就是 f+k1：

~~~text
D0（cat 出现3次）：  3×(1.5+1)/(3+1.5) = 7.5/4.5 ≈ 1.667 → 0.182×1.667 ≈ 0.304
D1（cat 出现1次）：  1×(1.5+1)/(1+1.5) = 2.5/2.5 = 1.000 → 0.182×1.000 ≈ 0.182
~~~

D0 胜出——但注意：cat 出现次数是3倍，词频项只有约1.67倍。这就是**词频饱和**。把 f 从1一路加大（长度因子=1时）：

~~~text
f=1 → 1.00    f=2 → 1.43    f=3 → 1.67    f=10 → 2.17    f→∞ → 2.5（=k1+1，天花板）
~~~

十倍的出现次数只换来约2.2倍得分，收益迅速递减；不管堆多少次，这一项永远到不了2.5。没有饱和的话，垃圾文档把关键词抄一百遍就能霸榜。两个旋钮顺带手算：**长度校正**——若把 D0 拉长到8词（cat 仍3次，avg_dl 仍4），分母变 3+1.5×(1−0.75+0.75×8/4)=5.625，词频项从1.667掉到约1.333：长文档被压价，防止“什么都装”的大杂烩靠堆词取胜。**IDF**——若语料扩到3篇、fish 只在其中1篇出现，IDF(fish)=ln(1+2.5/1.5)≈0.981；3篇都有的词只有 ln(1+0.5/3.5)≈0.134，稀有词贵约7倍。

**余弦相似度：稠密检索那一棒。** 设查询向量 q=[0.8, 0.6]（单位向量），两篇文档 d1=[0.6, 0.8]、d2=[−0.8, 0.6]（也都是单位向量）：

~~~text
cos(q, d1) = (0.8×0.6 + 0.6×0.8)/(1×1) = 0.96    → 方向很近
cos(q, d2) = (0.8×(−0.8) + 0.6×0.6)/(1×1) = −0.28  → 基本背向
~~~

向量都归一化后点积=余弦——这正是正文 build_dense_index 里 normalize_embeddings=True 的原因。稠密检索的关键本事：查询“小猫打盹”与文档“年幼的猫在睡觉”**一个词都不重叠**，BM25 给0分，余弦却能给出高分。

**RRF 一分钟版：** 两份排名[文档0,文档1]与[文档1,文档2]、k=60 时，文档1得 1/62+1/61≈0.0325，超过只出现一次的文档0或2——只用名次、不碰原始分数尺度。完整推导见第二节2.9。

以上是用于理解的算例，不是新增的检索或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

四个 Build It 分段正好对应流水线四棒，按顺序读：BM25 类（tokenize 切词→__init__ 统计 df 与 avg_dl→idf→score 逐词累加→rank 暴力扫描排序，0.3 的手算就是 score 的逐步展开）；build_dense_index + dense_search（编码、归一化、点积，与0.3的余弦手算同型）；reciprocal_rank_fusion（0.3 的 RRF 就是它跑两份排名）；hybrid_search + CrossEncoder（注意 pool_size=30 与 top_k=5 的分工——粗筛池大小和最终答案数是两个量）。第五节的标准库入口不装任何模型就能先跑 BM25 和 RRF。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 cat 出现10次的文档，BM25 得分不到出现1次文档的3倍？

**参考答案：** 词频饱和：分子含 f、分母也含 f，该项随 f 递增但上限是 k1+1=2.5；f=10 时约2.17、f=1 时为1.0。设计目的就是让堆关键词的收益递减。

**问题2：** RRF 为什么用名次而不用原始分数相加？

**参考答案：** BM25 分数与余弦相似度是两套互不可比的尺度（一个可以远大于1、一个在[−1,1]之间），直接相加等于拿米和厘米直接做加法；RRF 只取排名位置，天然免校准。代价是丢掉了分差大小信息。

**问题3：** 正确文档在 BM25 和稠密两路的前30名都没出现，重排器能救回来吗？

**参考答案：** 不能。重排只对已进候选池的文档重新打分；两路都没召回的证据，任何重排都无米下锅，应先解决召回（换编码器、改切块、加第三路检索）。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

词面相同和语义相近是两种不同线索。本课将BM25、独立向量编码、排名融合、交叉编码重排接成候选流水线。原文法律条文例子仅是检索文本示意，不应当作现行法律说明。

## 二、核心概念与推导

BM25为query中每项累加IDF×词频饱和×长度校正。k1调重复次数饱和速度；b调文档长度影响。本代码IDF=ln(1+(N−df+0.5)/(df+0.5))。与上一课TF-IDF公式不同，不可直接混数值。

（补充 2026-11-06）把“IDF×词频饱和×长度校正”三个因子各自翻译成一句话：**IDF**回答“这个词稀有吗”（稀有=贵，0.3 里有0.981对0.134的对比）；**词频饱和**回答“文档里出现了几次”（收益递减，上限k1+1，0.3 有饱和表）；**长度校正**回答“这篇是不是靠篇幅堆出来的”（长文档压价）。三者相乘再对查询里的每个词求和，就是score。与[第02课](<02：词袋模型、TF-IDF与文本表示.md>)的TF-IDF对照，差别主要在后两个因子：TF-IDF的TF随次数线性增长，也没有显式的文档长度分母；BM25的k1=1.5、b=0.75是Robertson原论文的推荐值，实践中很少需要调。

（补充 2026-11-06）一张“什么查询谁赢”的对照表：

| 查询类型 | BM25 | 稠密 | 原因 |
|---|---|---|---|
| 错误码、法条编号、产品SKU | 赢 | 可能漏 | 精确串，差一个字符就变义 |
| 同义改写（“骗钱”vs“欺诈获取财物”） | 漏 | 赢 | 零词面重叠、语义相同 |
| 换拼写、大小写、单复数 | 脆 | 稳 | tokenize后精确匹配失败 |
| 预训练语料没覆盖的领域黑话 | 稳 | 可能弱 | 编码器没学过该领域 |

bi-encoder独立编码文档，可离线预存；cross-encoder把(query,doc)放一起打相关分，适合较小候选池。原课dense_search是全矩阵点积，不是FAISS近似索引。

（补充 2026-11-06）两种编码器用招聘类比最直观：bi-encoder像**简历入库**——每份简历（文档）提前独立打好标签存进柜子，来了岗位（查询）也独立打标签，再按标签匹配；快、能撑百万文档，但打标签的两头从没见过对方，细粒度匹配会漏。cross-encoder像**面试**——把岗位需求和简历放在一起逐条对照打分，准得多，但每对都耗时，只请得起前30名面试。这就是“召回靠bi、精排靠cross”的分工：贵的方法只花在小池子上。hybrid_search里 pool_size=30、top_k=5 正是这个预算分配。

RRF忽略原分数尺度：
$$score(d)=\sum_{\text{list }l}\frac1{k+rank_l(d)}$$
源代码rank从0枚举，故分母用k+rank+1。k=60是平滑常数，不是检索条数。融合不能召回两个列表都没找到的证据。

（补充 2026-11-06）k=60的直觉：名次第1贡献1/61、第2贡献1/62……相邻名次的差距本来就小；k越大，各名次贡献越被抹平，头部名次的优势越弱；k越小越奖励霸榜。60是RRF原论文（Cormack et al., 2009）的经验默认值。另一个易错点：融合对象是**名次列表**，同一文档在一份列表里只应出现一次，重复出现会把1/(k+rank)重复累加。

单相关文档时query命中率@k与常见Recall@k一致；多相关文档时真正Recall应按找回相关数/总相关数计算。MRR只看首个相关位置，nDCG考虑分级相关性。

（补充 2026-11-06）三个指标的白话版：**Recall@k**——“该找到的在前k里找到了吗”，是RAG里检索器的第一指标，阅读器救不回没召回的证据（[第13课](<13：问答系统——抽取式、检索增强与拒答评估.md>)的拒答评估接的正是这里）；**MRR**——第一个相关结果排第几，排第3得1/3、没找到得0，对所有查询取平均；**nDCG@k**——相关也分三六九等时用，“完全命中”比“沾边”权重高，还按位置打折。评估前先建gold（哪些文档对哪些查询相关）——没有gold的检索评估只是自我感觉良好。

（补充 2026-11-06）排错小技巧（原课 Debugging 建议）：对失败查询，把 BM25 和稠密两份排名并排 diff。一路找到、另一路没找到，说明是词表失配（补另一路）或语义歧义（换更好的编码器或加重排）。

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

（补充 2026-11-06）规模选型一行版：十万级以下文档，内存BM25＋MiniLM向量＋RRF 不需要任何专门数据库；百万级以上才需要FAISS/pgvector＋Elasticsearch/OpenSearch这类独立索引；重排永远只对top-30左右的候选做。判断依据是延迟预算与召回指标，不是文档数本身。切块细节在[第23课](<23：RAG切块——固定、递归、语义、父子与上下文增强.md>)展开。



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

