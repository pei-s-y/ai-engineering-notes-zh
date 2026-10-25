# 22：现代Embedding——稠密、稀疏、多向量与降维评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/docs/en.md)。原题：Embedding Models — The 2026 Deep Dive。

## 一、这一课解决什么问题？

词向量之后，检索通常为句子/段落构造表示。表示模式、语言、截断、维度与算力预算共同决定效果，不能只看排行榜或向量库名称。

## 二、核心概念与推导

dense：每段一个固定长度向量，便于近邻索引。sparse：大词表上多数维度为0的权重，学习式SPLADE与原始词频并非相同。multi-vector：每token一个向量，ColBERT式MaxSim对每个query token选最匹配文档token再累加。

BGE-M3可输出dense、lexical和ColBERT式表示，但生成这些表示不等于自动建立三种可检索索引。

Matryoshka是专门训练让向量前缀也有用的表示学习，截断后通常需重归一化。普通向量切片并不会因此变成Matryoshka。原课首个bge-small-en-v1.5不是该示例所证明的Matryoshka checkpoint，128维练习应称普通截断对照，不能宣传无损。

3072个float32理论原始存储=12288字节，额外索引、元数据、副本另算；按月价格取决于供应商。MTEB/BEIR覆盖不同任务，排行榜与领域、延迟并非同一个指标。原文2026排名、价格和固定1%损失不作为当前事实承诺。

### 2.9 用一个小例子把概念串起来（补充）

同样10万段，384维float32仅向量约153.6MB；1024维约409.6MB，未计索引和元数据。多向量还要乘每段token数量。普通1024维砍到256维虽省空间，但没有训练保证；Matryoshka也应测领域损失，不因名称就宣布无损。

## 三、操作前准备

~~~bash
python -m pip install numpy sentence-transformers FlagEmbedding mteb
~~~
分阶段安装即可；BGE-M3更大，CPU应按后端支持关闭半精度。MTEB的数据/接口随版本变化，先锁版本并查看其官方示例；本次不下载模型或跑评测。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 建立dense基线

emb形状(N,D)，q_emb形状(D,)，归一化后矩阵点积得到N个余弦分数。BGE英文检索query建议加模型卡指令，文档不加；v1.5也支持无指令，不是所有bge都必须加同一前缀。

~~~python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
~~~

### 4.2 截取前缀并重新归一化

源码truncate只是数学操作，需检查0范数、dim正整数及不大于原维数。对任意模型截断都要测损失；不能因为函数名就称其无损Matryoshka。

~~~python
# 普通切片并重归一化，是否有效取决于训练方式；需防零范数。
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
~~~

### 4.3 同时生成三种表示

FlagEmbedding的lexical_weights与colbert_vecs不是普通dense向量的别名。q_lex/d_lex等均需分别生成，dense_score=...是占位不是可计算分数；融合权重是验证集待调超参。

~~~python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
~~~

~~~python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
~~~

### 4.4 运行代表性MTEB任务

ArguAna、SciFact、NFCorpus覆盖不同检索领域。原接口是特定版本，若新版要求task对象请按该版本文档迁移；不应在未知下载量/费用下直接跑全榜。

~~~python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
~~~

### 4.5 哈希向量的标准库教学版

原课仓库用词哈希演示向量化、归一化和排序。没有预训练语义，截断不具备Matryoshka训练保证；下面补一个小入口，让你无需额外脚本看懂形状。



## 五、把代码运行起来：补充入口与检查

补充标准库哈希向量，只用于理解接口，不冒充BGE语义或Matryoshka：
~~~python
import hashlib
import math
import re

def toy_embed(text, dim=64):
    if dim <= 0:
        raise ValueError("dim必须为正数")
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(word.encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % dim] += 1.0
    norm = math.sqrt(sum(x*x for x in vec))
    return [x/norm for x in vec] if norm else vec

a, b = toy_embed("cat mat"), toy_embed("cat dog")
print("维数", len(a), "点积", sum(x*y for x, y in zip(a, b)))
assert abs(sum(x*x for x in a) - 1.0) < 1e-9
~~~
对真实BGE查询可加指令：Represent this sentence for searching relevant passages: 后接query。依据：[BGE官方模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5)。本次真实模型效果和截断损失均未测量。

## 六、Use It：使用现成工具

按语料语言、输入长度、许可、部署、预算选择候选；英文BGE、BGE-M3等是原课示例，不是最新推荐榜。稀疏适合精确词项补充，多向量适合细粒度匹配但索引更大，cross-encoder可重排小候选池。并非所有生产系统都需要三条路线；3分差值不是通用换模型门槛。



## 七、失败模式、原文边界与工程注意

1. query/doc可能需要不同prompt或编码入口，必须按具体模型卡。
2. cosine要求归一化或显式除范数，但若模型设计为原始点积评分就不要擅自归一化。
3. 零向量切片后除零会产生NaN；空输入与过短前缀要测。
4. 模型最大长度与有效能力不同，原文60%～70%容量不能通用。
5. MTEB分数不同任务/版本不直接比较；p99需要足量请求与清楚的并发条件。
6. 小于1万段也可研究ColBERT，原文规模禁令不是技术限制。
7. 仓库toy的词频sparse不是SPLADE，哈希dense不是神经语义。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
记录模型和tokenizer版本、语言/长度、dense/sparse/multi-vector模式、query模板、维度/精度与存储估算。只有支持Matryoshka且经领域验证才采用前缀截断；以领域nDCG/MRR/Recall、p50/p99、成本选择Pareto方案。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：100句用bge-small-en-v1.5得到384维，再截128维，用10查询测MRR变化；明确这是普通截断对照。如要验证Matryoshka，另选明确支持的模型。
2. Medium：500领域段落比较BGE-M3三模式Recall@10与RRF，使用统一gold与文档ID。
3. Hard：3候选模型、2项相关MTEB/领域任务，汇报指标、100查询批次延迟及每百万查询成本估计；p99样本有限需注明不确定性。

## 十、术语与完成标准

dense：稠密表示；sparse：稀疏表示；MaxSim：逐query token的最大匹配累加；Matryoshka：前缀维度可用的训练策略；MTEB：多任务embedding评测；Pareto：无法同时改善所有目标的候选集合。

完成检查：

- 能解释三种索引大小差别。
- 能说明普通截断不是Matryoshka。
- 能读懂(N,D)矩阵点积。
- 能列出模型卡需核查项。

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

