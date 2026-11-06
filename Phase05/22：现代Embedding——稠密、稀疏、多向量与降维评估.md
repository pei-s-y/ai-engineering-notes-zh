# 22：现代Embedding——稠密、稀疏、多向量与降维评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，五步结构与原课一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/docs/en.md)。原题：Embedding Models — The 2026 Deep Dive。

## 零、初学者先读：从“一词一向量”到“一段一向量”的三种形态

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。前置衔接：[第03课](<03：词向量与从零实现 Word2Vec.md>)、[第04课](<04：GloVe、fastText与子词表示.md>)的词向量是“一个词一个静态向量”，本课换成检索场景的“一段话的表示”，并且一次给三种形态；[第14课](<14：信息检索——BM25、稠密向量、RRF与重排.md>)的BM25与混合检索已给过动机，本课补上“表示本身”的细节。

### 0.1 先理解这一课在做什么

RAG检索错了，第一嫌疑人常常不是向量库也不是提示词，而是embedding模型（原课的判断）。但选模型之前，先得弄清“一段话的向量”有几种存法——本课的主角是三种形态：

| 形态 | 一段话怎么表示 | 类比 |
|---|---|---|
| dense（稠密） | 一个固定长度向量 | 一张缩略照片：整体印象 |
| sparse（稀疏） | 大词表上少数维度有权重 | 背面写着关键词的索引卡 |
| multi-vector（多向量） | 每个token一个向量 | 逐帧保存的整场录像 |

[第03课](<03：词向量与从零实现 Word2Vec.md>)是给“词”发身份证，本课给“段落”发身份证、还一次发三种。检索时用哪种，取决于查询要什么：像dense看“整体意思像不像”，sparse像机器筛简历专看“关键词有没有”，multi-vector像逐帧核对录像的每个细节。还有个第四维度的问题——向量太长存不起怎么办？**Matryoshka**（俄罗斯套娃）：专门训练让“前N维也是好向量”，1536维砍到前256维还能用，像套娃一层套一层。注意“砍前缀”和“训练过”缺一不可：只砍不训练就是普通截断（4.2的truncate正是普通截断对照）。

### 0.2 本课术语先翻成白话

- dense embedding（稠密向量）=一段一个固定长度向量（常见384～3072维），近邻用余弦比
- sparse embedding（稀疏向量）=维度=词表大小、大多数为0，只在出现的词项上有权重；学习式（SPLADE）与原始词频不是一回事
- multi-vector / late interaction（多向量/迟交互）=每token一向量，查询来了才逐token比对
- MaxSim=对每个查询token取最像的文档token的分，再对查询token求和
- bi-encoder（双编码器）=查询、文档分别独立编码再比——能预计算、快
- cross-encoder（交叉编码器）=查询和文档拼一起进模型——准但不能预计算，只能重排小候选池（第14课的重排）
- Matryoshka（MRL）=让向量前缀独立可用的训练目标
- BGE-M3=一个检查点同时输出三种表示的模型
- MTEB / BEIR=embedding榜单 / 其中的检索子榜
- RRF=倒数排名融合，第14课混合检索的工具
- normalize（归一化）=向量除以自身长度变成单位向量，之后点积=余弦

### 0.3 用小数字走一遍

两文档两查询，稠密与稀疏各跑一遍。文档：D1“苹果在2007年发布了首款iPhone。”、D2“谷歌开发了Android操作系统。”查询：Q1“iPhone是哪年发布的？”、Q2“哪家公司生产智能手机？”

稀疏（简化TF-IDF式：命中词各计IDF=ln(2/1)≈0.69，教学用）：

| 稀疏得分 | D1 | D2 |
|---|---|---|
| Q1 | 命中iPhone、发布 → 2×0.69≈1.39 | 无命中 → 0 |
| Q2 | 无命中词 → 0 | 无命中词 → 0 |

Q1正确召回D1；Q2在稀疏视角下两文档并列0分——D1里没有“手机”“公司”这些字面，只有“iPhone”。而稠密（余弦为假设值，仅示意形态差别）：

| 稠密余弦 | D1 | D2 |
|---|---|---|
| Q1 | 0.88 | 0.09 |
| Q2 | 0.71 | 0.18 |

Q2的D1=0.71是稠密的看家本领：模型知道iPhone是智能手机，语义桥接了字面鸿沟，正确召回D1。反过来也有稀疏赢的场景：查询含罕见精确串（型号XR-2000、报错码、编号）时字面命中最稳——这正是[第14课](<14：信息检索——BM25、稠密向量、RRF与重排.md>)混合检索加RRF的动机，也是BGE-M3“一个模型三种表示”要解决的问题。

MaxSim顺手小例（multi-vector的打分动作）：查询2个token、文档3个token，余弦矩阵：

~~~text
          文档token  d1   d2   d3
查询 q1              0.9  0.2  0.4   → 每行取max = 0.9
查询 q2              0.3  0.8  0.5   → 每行取max = 0.8
MaxSim总分 = 0.9 + 0.8 = 1.7
~~~

对比dense：一段一个向量、只有一个分数；multi-vector保留了“q1靠d1、q2靠d2”的细粒度信息——代价是存储与计算都乘token数（2.9的规模账）。

以上是用于理解的算例，不是新增的训练或性能实测（稠密余弦值为假设值）。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

4.1 dense基线：SentenceTransformer编码后emb @ q_emb就是0.3稠密列的动作（normalize_embeddings=True让点积=余弦）。4.2 truncate是Matryoshka段的普通截断对照。4.3 BGE-M3一次返回dense_vecs、lexical_weights、colbert_vecs——0.3的三种形态各占一个键；compute_lexical_matching_score对应稀疏打分、colbert_score对应MaxSim。4.4 MTEB评测。第五节toy_embed是哈希dense的教学版（无语义，只看接口形状）。

### 0.5 理解检查（含参考答案）

**问题1：** 稀疏向量“稀”在哪里？

**参考答案：** 维度等于词表大小（几万到几十万），一段话只会在少数词项维度上有非零权重、其余全0；它像学习出来的倒排索引，专管字面精确匹配。

**问题2：** 把1024维向量砍成256维，就是Matryoshka吗？

**参考答案：** 不是。Matryoshka是专门训练目标，“前缀独立可用”是练出来的性质；普通模型砍前缀是普通截断，损失要实测（原课4.2就是普通截断对照，不能宣传无损）。

**问题3：** 为什么bi-encoder能预计算、cross-encoder不能？

**参考答案：** bi-encoder把查询和文档各自独立编码，文档向量可以提前算好入库；cross-encoder必须把“查询+文档”拼在一起进模型，查询不来就没法算——只能等查询到了对小候选池逐对打分，所以它做重排不做召回。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

词向量之后，检索通常为句子/段落构造表示。表示模式、语言、截断、维度与算力预算共同决定效果，不能只看排行榜或向量库名称。

## 二、核心概念与推导

dense：每段一个固定长度向量，便于近邻索引。sparse：大词表上多数维度为0的权重，学习式SPLADE与原始词频并非相同。multi-vector：每token一个向量，ColBERT式MaxSim对每个query token选最匹配文档token再累加。

（补充 2026-11-06）三种形态的账各算一笔就记住了。dense：每段一个384～3072维向量，索引小、比对一次点积，是默认起手式。sparse：维度=词表大小，但只存非零项（像倒排索引），字面精确匹配强——SPLADE与原始词频的差别在权重是模型学出来的（哪个词对这篇更重要、能不能做同义扩展），不是数出来的。multi-vector：存储=段数×每段token数×每token维度，比对复杂度也是“查询token数×文档token数”的乘积——所以只在候选池小或追求细粒度精度时用。0.3的MaxSim小例是它的最小动作；注意“迟交互（late interaction）”迟的是**比对**这步：文档token向量可以提前算好存着，查询来了才逐对算MaxSim——区别于cross-encoder的“提前交互”（查询和文档在模型内部就见面了）。

BGE-M3可输出dense、lexical和ColBERT式表示，但生成这些表示不等于自动建立三种可检索索引。

Matryoshka是专门训练让向量前缀也有用的表示学习，截断后通常需重归一化。普通向量切片并不会因此变成Matryoshka。原课首个bge-small-en-v1.5不是该示例所证明的Matryoshka checkpoint，128维练习应称普通截断对照，不能宣传无损。

（补充 2026-11-06）“截断后需重归一化”的原因拆开看：归一化过的向量长度是1，砍剩256维后长度变成前256个分量的平方和开根号（必然小于1），此时点积不再等于余弦——分母已经不是1了。truncate里那行除以norm就是把长度重新拉回1、恢复“点积=余弦”的等价；零向量时这步除零会产生NaN，正是第七节第3条的坑。Matryoshka前缀为什么可用，直觉是：训练显式要求“只用前64/128/256维算出的相似度排序”也要接近完整向量的排序，梯度被迫把最重要的信息塞进靠前的维度；普通训练没有这个约束，信息均匀摊在各维，砍掉一半就真丢一半。这也是2.9说“Matryoshka也应测领域损失，不因名称就宣布无损”的原因——训练保证的是“前缀可用”，不保证“在你的领域无损”。

3072个float32理论原始存储=12288字节，额外索引、元数据、副本另算；按月价格取决于供应商。MTEB/BEIR覆盖不同任务，排行榜与领域、延迟并非同一个指标。原文2026排名、价格和固定1%损失不作为当前事实承诺。

（补充 2026-11-06）选型时三个“不同指标”别混成一个：MTEB分数测标准任务集上的平均质量，你的领域分布可能完全不同（所以原课反复强调在代表子集上自测）；p50/p99延迟测服务能力（[第14课](<14：信息检索——BM25、稠密向量、RRF与重排.md>)已见过p99对体验的意义）；每百万查询成本测账单。一个模型完全可能三项各占一个名次——“榜单第一”推不出“对你的系统最好也最便宜”。三段式的实用路线（原课Use It的浓缩）：dense起手做召回→加sparse用RRF提召回→小候选池上cross-encoder重排提精度，这正是第14课混合检索思路在表示层的落地。

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

