# 07：进阶 RAG——BM25、混合检索、重排与查询改写

> 对应课程：Phase 11 / 07-advanced-rag。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/07-advanced-rag/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/07-advanced-rag/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、为什么相似不等于有用

问“错误码 E-4021 怎么处理”，向量检索可能找到泛泛的错误处理文章，却漏掉恰好含 E-4021 的操作指南。问“哪支团队满意度提升最大”，需要多份报告和计算，单个相似段落未必包含最终答案。

本课在基础 RAG 上改进四个地方：**如何切块、怎样召回、怎样重排、怎样把问题变成好查询**。增加组件会增加成本，先定位失败原因再加相应组件。

## 二、BM25：会饱和的关键词评分

BM25 是词项检索算法。一个词在该文档多次出现会提高得分，但提高幅度逐渐变小；稀有词通常权重高；长文档天然更易多次出现词，需要长度修正。

原课形式：

`score(q,d)= Σ IDF(t) × tf(t,d)(k1+1) / [tf(t,d)+k1(1−b+b·|d|/avgdl)]`

- t：查询词；tf：它在该文档的次数；df：包含它的文档数；N：文档数。
- `IDF(t)=ln((N−df+0.5)/(df+0.5)+1)`，罕见词权重大。
- `|d|` 是文档长度，avgdl 是平均长度。
- k1 控制词频饱和速度；b 控制长度归一化强度。原代码用 1.2 和 0.75，是常见起点。

例如相同长度文档中，词频从 1 到 2 通常提高得分，但从 20 到 21 提升已很小。BM25 分数不是概率，也不能直接与余弦 0.8 相加而认为单位相同。

## 三、混合检索和 RRF

**Hybrid search（混合检索）**并行用关键词与向量召回，再融合结果。关键词擅长专名、编号、精确术语，神经语义向量擅长部分改述匹配；教学版向量仍是 TF-IDF，因此互补性不等于真实 BM25+神经嵌入。

**RRF（倒数排名融合）**不直接混不同分数，而是累计名次分：`Σ 1/(K+rank)`。原文常数 K=60，rank 从 1 开始。它与“返回前 k 个”的 k 是不同参数。

例：A 在两个榜单名次 1、3，分为 `1/61+1/63≈0.03227`；B 排 2、1，分为 `1/62+1/61≈0.03252`，所以 B 略高。原始 BM25=10 或余弦=0.8 不直接进入公式。没有出现在某榜单的文档从该榜单得 0。

## 四、重排不是重新搜全库

**Reranker**对候选集合重新算相关性。双编码器可预先存文档向量，适合大规模召回；交叉编码器联合读 `(query,document)`，能看更细致的匹配，但每个候选都要算，因此通常只重排少量候选。

先召回 top-50 再重排取 top-5，与直接召回 top-5 的差别是给精排更多选择。**如果相关文档根本没进入候选，重排无法凭空找回**。

原课 `rerank` 只是词重叠、相邻词组匹配、词在前部加分和初始分数的加权规则，**不是训练好的 cross-encoder**。用途是说明重排接口和前后排序变化。

## 五、查询变换和更好的切块

**HyDE（假设文档嵌入）**：先让模型写一段可能回答问题的假设文档，用这段文字检索真实资料。假设内容不是证据，最终回答只能依真实检回资料。原课生成函数只是套模板；如果生成的假设偏离问题，检索会被带偏。

**Multi-query**：把同一问题改写成多个查询分别检索，再去重融合。**Step-back**：先问更一般的背景问题，再结合原问题寻找答案；过度泛化也会降低精确性。多跳问题要拆成子问题，收齐所需事实后再比较/计算，不能指望一次 top-k 就回答所有关联。

**父子块（parent-child）**：小子块用于精确检索，命中后返回其较大父块作为上下文。例：父块 100 词，每个子块 30 词；一个子块命中“退款 60 天”，返回父块能带上“仅企业版”的条件。需要 child_id→parent_id 映射与父块去重。

**Metadata filter（元数据过滤）**先限定产品、时间、版本、类别或权限，再做检索。过滤错误会把正确答案排除；权限过滤必须在数据访问层落实，不能只靠提示词。

## 六、原文 Build It：七个组件

网页 `hybrid_search` 依赖上一课的 `tfidf_embed` 和 `search`，不是孤立可运行函数；仓库完整文件自带 TF-IDF 和名为 `vector_search` 的实现。代码块解释原网页步骤，运行时使用对应完整脚本。

### 步骤 1：BM25

index统计文档长度和包含每词的文档数；score用饱和TF、IDF和长度修正相加；search逐篇算分。空语料、空文档及分词规则要测试。

````python
# 本块中文说明：BM25
import math
from collections import Counter

class BM25:
    # 初始化本实例保存的配置和状态；self 指当前对象。
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    # 计算候选得分并排序，返回前 top_k 项；返回结果不等于已经生成答案。
    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
````

### 步骤 2：倒数排名融合

每个榜单按rank+1计名次，对同doc_id累加1/(k+rank+1)。原榜单的score未使用，避免不同分数尺度直接相加。

````python
# 本块中文说明：倒数排名融合
# 按各榜单名次累计倒数分，而不是直接相加原始分数。
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
````

### 步骤 3：混合召回

调用上一课TF-IDF与向量搜索，再取BM25候选，用RRF融合。网页search函数来自上一课；完整源码使用vector_search，自带所需定义。

````python
# 本块中文说明：混合召回
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
````

### 步骤 4：规则重排

去停用词，计算词命中、相邻词组和位置加分，再合成分数。初始分数乘5是手设权重，不是学到的cross-encoder。

````python
# 本块中文说明：规则重排
# 用手写词/短语/位置规则重排候选，原实现不是神经交叉编码器。
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
````

### 步骤 5：HyDE模拟

按照what/how套模板生成假设段落，再用它编码检索。返回假设方便检查偏移；这段假设没有事实地位，不能作为最后答案引用。

````python
# 本块中文说明：HyDE模拟
# 用模板模拟假设文档；假设文档只用于检索，不能当事实证据。
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
````

### 步骤 6：父子块

大块先形成parents，再在其范围内生成children，并记录child_to_parent。查询命中子块后要回查、去重父块。

````python
# 本块中文说明：父子块
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
````

### 步骤 7：有据性与召回评价

有据性这里只看内容词出现，不能查数字/否定矛盾；召回使用标注相关索引的集合做分母。无相关标注的查询要另定义指标，不要把默认1.0混进总体质量结论。

````python
# 本块中文说明：有据性与召回评价
# 原文启发式检查：词重叠不等于事实蕴含。
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
````


## 七、运行、输出与判别

在本课目录执行 `python code/main.py`，仅需标准库。会看到关键词/向量/混合候选比较、规则重排、假设文档、父子块映射、有据性/检索召回统计等。它没有调用真实嵌入模型、交叉编码器或大模型。

`create_parent_child_chunks` 的 parent_size 和 child_size 必须为正，否则循环可能不结束。父块结尾不足长度是正常现象。`evaluate_faithfulness` 按词出现情况判断支持，甚至用子串匹配；“退款 60 天”和“退款 6 天”的词高度相似，却有关键事实冲突。空答案可能被它赋 1.0，不能将该结果解释成高质量回答。

补充 RRF 计算：

```python
lists = [[("A", 9.0), ("B", 8.0)],
         [("B", 0.9), ("C", 0.8), ("A", 0.7)]]
scores = {}
for ranking in lists:
    for rank, (doc_id, original_score) in enumerate(ranking, start=1):
        # original_score 只是原榜单分数；RRF 使用的是名次
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)
print(sorted(scores, key=scores.get, reverse=True))  # 预期 ['B','A','C']
```

Use It 中的真实 cross-encoder、Cohere 重排接口和 Weaviate 混合查询分别替换规则重排或手动融合组件。不同数据库的 alpha 混分机制不一定就是 RRF，不能照搬相同数值解释；用真实相关性标注选择参数。

## 八、诊断清单、交付物与原文练习

原课的 debugger/patterns 交付物可以写成一张诊断表：**没检到精确编号→加关键词；候选有答案但名次低→重排；问题含省略→改写；片段丢条件→父子块；跨文档→分解多跳；回答无依据→生成与引用校验**。

1. 五题比较 BM25、向量、混合的第一名相关性。原文预言混合至少赢三题不是验收标准，真实输赢要记录。
2. 加 security/billing/api/product 元数据，查询加密时限定 security；再故意设错类别观察漏检。
3. 对原查询与 HyDE 检索比较 top-3；把假设文本保存，分析是否引入错误关键词，不要求必然提升。
4. 父块 100、子块 30 词，对照普通 50 词切块，比较条件完整性、引用长度与重复父块。
5. 标注 10 个问题的相关块，分别测向量、BM25、混合、混合+重排的 Recall@3/5/10；注明候选池大小，避免把候选池变化与重排效果混淆。

学会的标志：能手算 RRF，解释召回和精排的区别，知道 HyDE 生成内容为什么不能作引用，并能沿检索链定位一个具体失败。


## 本次验证记录（2026-11-01）

完整标准库demo通过；RRF补充例子得到B、A、C。重排与HyDE仍为规则模拟。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<06：RAG——从文档检索到有依据的回答.md>) · [下一课](<08：LoRA与QLoRA——只训练一小部分参数的微调.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
