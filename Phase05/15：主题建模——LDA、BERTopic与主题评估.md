# 15：主题建模——LDA、BERTopic与主题评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充（LDA生成故事、c-TF-IDF与-1离群点、选型对照）；当日与原课网页版核对。零节小例的主题词分布已用本机Python标准库复核行和为1。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/15-topic-modeling) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/15-topic-modeling/docs/en.md)。原题：Topic Modeling — LDA and BERTopic。

## 零、初学者先读：没有标签，让语料自己说出话题

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要回看“词频矩阵”时先读[第02课](<02：词袋模型、TF-IDF与文本表示.md>)；需要回看“句向量”时先读[第03课](<03：词向量与从零实现 Word2Vec.md>)——本课两条路线分别建立在这两块地基上。

### 0.1 先理解这一课在做什么

一万条客服工单、五万篇新闻，没人打过标签，甚至不知道该分几类——主题建模（topic modeling）就是让语料自己开口：给一堆无标注文本，还你一小簇“话题”和每篇文档与各话题的关系。两条主流路线思路完全不同：

- **LDA（2003，概率路线）**：假设每篇文档是几个主题按比例混合的“鸡尾酒”，每个主题又是词的概率分布。输出两张表：文档×主题比例表、主题×词分布表。
- **BERTopic（2020，神经路线）**：先用句向量把文档按语义扎堆（聚类），再给每堆挑代表词。一篇文档只属于一个堆。

类比整理一屋子乱堆的书：LDA 像逐本翻看并写下“这本70%讲宠物、20%讲美食”，允许一本书横跨多个话题；BERTopic 像先把书搬成几堆，每堆写一张标签，一本书只在一堆里。两条路线不是谁淘汰谁：短文本、要语义相似时 BERTopic 占优；长文档、要混合比例时 LDA 仍是基线。注意：得到的话题是“语料里词共现/语义聚集的真实结构”，不一定等于你预设的分类标签。

### 0.2 本课术语先翻成白话

- 主题（topic）=LDA里是词的概率分布；BERTopic里是一簇相似文档
- 混合成员（mixed membership）=一篇文档同时以比例属于多个主题，不是单选题
- doc_topic=文档×主题矩阵，形状(N,K)，每行和为1
- 主题词分布（topic-word distribution）=每个主题在词表上的概率，行和为1
- components_=sklearn LDA 存主题词权重的属性，**未归一化**，要除以行和才是概率
- c-TF-IDF（class-based TF-IDF）=把同一簇的全部文档拼成“一类”再算 TF-IDF，挑簇的代表词
- UMAP=降维方法，把高维句向量压到聚类好用的低维
- HDBSCAN=密度聚类，能找出大小不一的簇，还允许“不要了我”的离群点
- 离群点（outlier，标签-1）=HDBSCAN 不要的文档桶，不是第零个正常主题
- coherence（c_v，主题一致性）=代表词在语料里是否经常一起出现，越高越好
- diversity（主题多样性）=各主题代表词是否重复，越高越好
- NMF（非负矩阵分解）=LDA的快速替代品，输出同样是两张非负表
- 垃圾主题（junk topic）=被停用词或超高频词占满的无效主题，LDA常见失败模式
- 20 Newsgroups=经典新闻组公开数据集，本课 Exercises 的练手场

### 0.3 用小数字走一遍

4篇文档、2个主题、7个词的小世界：D1=`cat dog pet`、D2=`cat pet food`、D3=`football match goal`、D4=`cat football`。直觉化的“文档→主题软分配”（**这是假设的软分配数字，不是真跑 LDA 的输出**）：

| 文档 | 主题A（宠物向） | 主题B（体育向） | 读法 |
|---|---|---|---|
| D1 cat dog pet | 0.9 | 0.1 | 几乎纯宠物 |
| D2 cat pet food | 0.8 | 0.2 | 偏宠物 |
| D3 football match goal | 0.1 | 0.9 | 几乎纯体育 |
| D4 cat football | 0.5 | 0.5 | 一半一半（关键行） |

每行和为1——这就是“混合成员”：D4 不是被迫二选一，而是两个话题各占一半。再给两个主题各配一份词分布（行和为1）：

| 词 | 主题A | 主题B | 归属 |
|---|---|---|---|
| cat | 0.30 | 0.04 | A |
| dog | 0.25 | 0.04 | A |
| pet | 0.20 | 0.01 | A |
| food | 0.15 | 0.01 | A |
| football | 0.04 | 0.35 | B |
| match | 0.03 | 0.30 | B |
| goal | 0.03 | 0.25 | B |

两表合起来就能“算命”：D1 里出现 cat 的概率 = 0.9×0.30 + 0.1×0.04 = 0.274（90%从宠物主题抽、10%从体育主题抽）；D4 里 P(cat)=0.5×0.30+0.5×0.04=0.17、P(football)=0.5×0.04+0.5×0.35=0.195——一篇既讲猫又讲球的文章，两个词都算常见。LDA 训练就是反过来：给定词，反推这两张表。

BERTopic 对照：D4 只能拿**一个**硬标签（猫或球，看聚类怎么分），甚至被扔进-1离群桶。这就是第二节说的“聚类硬标签不能等同LDA的混合比例”。再用 top 词算 diversity：主题A前2词[cat, dog]、主题B前2词[football, match]，4个词不重复，diversity=4/4=1.0；若两主题前2词是[cat,dog]与[cat,football]，则降到3/4=0.75（第五节 topic_diversity 的断言就是它）。

以上是用于理解的算例，不是新增的训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

LDA 一侧：fit_lda 里 CountVectorizer 先把文档变成计数矩阵（[第02课](<02：词袋模型、TF-IDF与文本表示.md>)的老朋友，注意是计数不是TF-IDF），LatentDirichletAllocation 的 fit_transform 返回 doc_topic——就是0.3第一张表的真身，行和为1可以自己验证；print_top_words 读的是 lda.components_——0.3第二张表的**未归一化**版本，要看概率得按行除以总和。BERTopic 一侧：构造函数里 embedding_model→UMAP→HDBSCAN→c-TF-IDF 四步全自动，get_topic_info 看各簇规模（记得 Topic != -1 过滤离群桶），get_topic 看每簇代表词。第五节补充入口有12篇小语料和 topic_diversity，可先跑通再说。

### 0.5 理解检查（含参考答案）

**问题1：** LDA 说“D4 有50%属于主题A”，这个50%和分类器说“D4 有50%概率是A类”是一回事吗？

**参考答案：** 不是。LDA 是混合比例：D4 同时含两个主题、各占一半；分类概率是对“唯一正确类别”的下注。一个能同时为真，一个必须二选一。

**问题2：** BERTopic 输出里的主题-1是什么？能直接删掉吗？

**参考答案：** HDBSCAN 判为离群的文档桶，不是第零个正常主题。离群多可能反映参数、语料或表示问题，应先查原因，不应删除后宣称高质量。

**问题3：** coherence 和 diversity 都很高，主题就一定有用吗？

**参考答案：** 不一定。高分只说明代表词一致、不重复，不保证主题对业务有意义（比如全是由停用词凑成的“垃圾主题”）。还要人工命名、查代表文档、检查换随机种子是否稳定。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

面对没有类别标签的大量文本，自动整理常见话题。本课比较概率主题模型与“句向量聚类后解释主题”的路线；得到的话题不一定等于你预设的分类标签。

## 二、核心概念与推导

LDA把每篇文档视为多个主题的混合，每个主题又是词的概率分布。doc_topic形状(N,K)，每行和为1；sklearn的components_是未归一化的主题词权重，不能直接当概率，需按行除以总和。

（补充 2026-11-06）LDA 的**生成故事**（想象语料是怎么“写”出来的，训练就是反推）：

~~~text
写一篇文档：
  1. 掷“文档骰子”定主题配方，如 [宠物0.9, 体育0.1]
  2. 要写下一个词了：按配方随机挑一个主题（90%挑中宠物）
  3. 从该主题的词分布里随机抽一个词（cat 概率0.30、dog 0.25……）
  4. 回到第2步，直到写完
~~~

反过来，LDA 拿到现成的词流（0.3 的“算命”方向倒过来），用 Gibbs 采样或变分推断把两张骰子表（doc_topic 与主题词分布）估计出来。理解了生成故事，两个常见困惑自动消解：为什么每行和为1（配方是比例）；为什么 components_ 不是概率（它是没归一化的骰子权重）。

（补充 2026-11-06）主题数 K 是自己定的超参数，模型不会告诉你“正确答案”：K 太小，不相干的话题被强行捏在一起；K 太大，一个话题被拆成几个碎瓣。原课 Exercises 的做法是对的——跑多个 K、画 coherence 曲线、再看人工可解释性与稳定性，而不是拍一个数。

BERTopic通常先用SentenceTransformer生成句向量，再用UMAP降维、HDBSCAN聚类，最后用c-TF-IDF挑出代表词。这里的c指class：把同一簇的文本作为一类统计，不等于普通逐文档TF-IDF。主题-1通常表示离群点，不是第零个正常主题。聚类硬标签和可选membership不能直接等同LDA的混合比例。

（补充 2026-11-06）四步流水线每步为什么存在：句向量提供“语义相近”的度量（[第03课](<03：词向量与从零实现 Word2Vec.md>)的地图，搬到句子层面）；UMAP 降维是因为几百维向量直接聚类既慢又不稳，压到约5维后簇间距离才有意义；HDBSCAN 选它是因为不需要预设簇数、能找出大小不一的簇，还允许离群（-1）——这正对主题建模“不知道有几类”的胃口；c-TF-IDF 解决“聚完堆怎么起名”：普通 TF-IDF 按文档统计会淹没簇特征，把整簇文档拼成一个“类文档”再统计，簇内常见、簇外罕见的词才会浮出来当代表词。0.3 的 D4（一半猫一半球）暴露这条路线的代价：硬标签丢失混合信息。

（补充 2026-11-06）两条路线的选型对照（原课 When to pick which 的整理）：

| 场景 | 选 | 一句话理由 |
|---|---|---|
| 短文本（推文、标题、评论） | BERTopic | 词太稀，计数模型饿死；语义向量扛得住 |
| 长文档、天然多话题 | LDA | 混合成员才有意义 |
| 无GPU/算力紧 | LDA 或 NMF | 纯计数矩阵运算 |
| 要“每篇文档各话题占比”输出 | LDA | 聚类硬标签给不了比例 |
| 文档超过编码器上下文长度 | 先切块聚合或用LDA | 长文送 embedding 会被截断 |

c_v coherence衡量代表词的共现一致性；diversity衡量不同主题代表词是否重复。高分仍可能得到对业务无用的主题，因此还需要人工命名、查看代表文档并检查稳定性。

（补充 2026-11-06）coherence 的直觉：一个主题的代表词如果在真实语料里总是成对出现（cat 和 pet 常在同一段落），这些词大概真的指向同一件事；如果代表词彼此从不照面（cat 和 football），这个“主题”多半是模型硬凑的。c_v 用滑动窗口里的共现统计（NPMI）算词对得分再聚合成主题分——分数是“代表词像不像一家人”，不是“主题对业务有没有用”，后一件事指标替代不了人。评估组合拳：coherence（词共现）＋diversity（不重复）＋人工命名＋代表文档抽查＋换种子稳定性，五项缺一不可。

（补充 2026-11-06）两个常见“高分陷阱”：**垃圾主题**（junk topic）——一整桶停用词或超高频词（the、is、and）凑出的主题，coherence 不低但毫无信息量，原课 Ship It 明确把它列为 LDA 要探测的失败模式，对策是 min_df/max_df 过滤＋停用词表＋人工点名；**换种子就变脸**——同一语料换个 random_state，主题数量和代表词大洗牌，说明结构本身不稳固，报告结果时必须附随机种子与稳定性记录（第七节第4条的来源）。

### 2.9 用一个小例子把概念串起来（补充）

假设一篇文章的doc_topic=[0.7,0.2,0.1]，表示在LDA约定下偏主题0、仍混有其他主题，不是三个分类标签的独立yes概率。若两主题前2词为[cat,dog]、[cat,bird]，diversity=3/4=0.75。代表词解释弱时，必须同时看代表文档，不能只让LLM编一个好听主题名。

## 三、操作前准备

~~~bash
python -m pip install numpy scikit-learn bertopic sentence-transformers gensim
~~~
LDA不要求GPU；BERTopic依赖预训练向量模型，首次运行可能下载。原文documents是占位变量，要先提供真实字符串列表。min_df=2要求词至少出现在两篇文档；只有两三篇文本时很容易被过滤为空。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

（补充 2026-11-06）两个安装之外的“数据坑”提前说：一、`min_df=2` 要求词至少出现在两篇文档，只有两三篇文本时词表几乎会被清空——第五节的12篇小语料就是卡着这条线设计的；二、`stop_words="english"` 只适用于英文，中文要先分词（比如 jieba）再换自定义停用词，直接喂整句中文会得到一列“整句”假词。

## 四、Build It：原课分步代码与中文讲解

### 4.1 训练LDA并查看代表词

CountVectorizer得到计数矩阵，不是TF-IDF矩阵。n_components指定主题数，random_state有助复现实验，online是训练更新方式。先fit_lda，再print_top_words；权重高表示该主题偏好该词，不表示该文档一定属于这个主题。

~~~python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


# 先把文档转为词频矩阵，再估计主题分布。
def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


# 按主题词权重排序，仅作主题解释。
def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
~~~

### 4.2 运行BERTopic

先准备documents，再运行示例。min_topic_size=15是聚类规模设置，小样本可能全部成为-1；请用足量语料，或明确记录你为教学调整的参数。get_topic_info查看主题规模，get_topic查看代表词。

~~~python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
~~~

### 4.3 评价主题质量

原课此步主要讲指标，没有提供可直接执行的完整coherence脚本。c_v通常要用一致的分词结果、词典和主题词集合计算；不能把原始字符串直接冒充tokens。不同主题数还要比较重复度、人工可解释性与稳定性。



## 五、把代码运行起来：补充入口与检查

补充运行入口（先收集至少几十篇文本，不用复制同一篇充数）：
~~~python
# documents = ["第一篇真实文本……", "第二篇真实文本……", ...]
# 英文默认停用词仅适用于英文；中文应先分词并更换vectorizer。
# lda, vectorizer, doc_topic, words = fit_lda(documents, n_topics=5)
# print_top_words(lda, words, 10)
# print(doc_topic.shape, doc_topic.sum(axis=1))
~~~
这里第三个位置参数10对应源码的n_top。预期看到每主题前若干词和(N,5)矩阵，不保证主题能自动获得好名字。补充指标：
~~~python
def topic_diversity(top_words):
    flat = [word for topic in top_words for word in topic]
    return len(set(flat)) / len(flat) if flat else 0.0
assert topic_diversity([["cat", "dog"], ["cat", "bird"]]) == 0.75
~~~
本次不执行外部库训练，不提供虚构主题分数。

### 补充：可直接传给LDA的小语料

下面是自编英文教学语料，不是原文数据。先定义Build It的fit_lda与print_top_words，再运行；它需要scikit-learn/NumPy，本次未执行，不保证主题编号与名称。

~~~python
documents = [
    "cat dog pet animal food", "cat pet animal home food",
    "dog pet animal home", "cat dog animal care",
    "football team sport match goal", "football sport team goal",
    "basketball team sport match", "team match sport win",
    "computer code software program", "computer software code system",
    "program software developer code", "computer system program developer",
]
lda, vectorizer, doc_topic, words = fit_lda(documents, n_topics=3)
print_top_words(lda, words, 5)
print("文档主题矩阵形状：", doc_topic.shape)
assert doc_topic.shape == (12, 3)
assert np.allclose(doc_topic.sum(axis=1), 1.0)
~~~

应看到3组代表词与(12,3)。请自己检查能否命名为宠物、体育、软件；不强求模型严格分出这3类。不要把这12篇直接用于原BERTopic的min_topic_size=15并期待3个正常簇。

## 六、Use It：使用现成工具

原文还介绍gensim LDA、sklearn NMF、Top2Vec、FASTopic以及用LLM给主题命名。长文本有明确多主题结构时LDA值得作为基线；短文本可尝试句向量聚类，但并非短文本不能使用LDA。NMF通常搭配非负的TF-IDF矩阵。长文送入embedding可能被截断，必须制定分段/聚合策略。LLM命名是解释层，不应改写原始聚类证据。

（补充 2026-11-06）“用 LLM 给主题命名”的正确姿势：把**代表词＋几篇代表文档**一起给模型，让它起名——LLM 看到的是聚类留下的原始证据。错误姿势是只看几个代表词就让 LLM 编一个好听的名字：它会自信地起一个与语料毫无关系的名字，把解释层变成编故事层。命名之后还要拿代表文档回验：名字说“退款投诉”，抽出来的文档得真是退款投诉才算数。



## 七、失败模式、原文边界与工程注意

1. 主题数不是越大越好，也没有通用“至少5个”规则。
2. BERTopic的nr_topics通常用于主题归并，不能当作与LDA n_components完全相同的控制旋钮。
3. 离群点多可能反映参数、语料或表示问题，不应直接删除后宣称高质量。
4. 随机种子、embedding版本、降维与分词都会改变结果，需保存。
5. 源码英文停用词设置不适合直接处理中文。
6. 不同语料、分词与指标实现下的coherence数值不宜横比。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
输入语料语言、长度、数量和目标，选择LDA/NMF/BERTopic基线；记录向量模型、分词、随机种子、主题数或聚类参数。输出主题词、代表文档、离群比例、coherence、diversity和人工命名依据，不把经验阈值当作质量保证。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：在20 Newsgroups上训练5主题LDA，每主题列前10词，人工命名，并与已知新闻类别对照；不要求一一对应。
2. Medium：在相同子集运行BERTopic，比较主题规模、离群率、代表词与coherence；不只比较主题数量。
3. Hard：比较5、10、20、50主题配置并绘制coherence曲线。对BERTopic说明使用的是归并目标还是聚类参数，不声称它必然精确产生指定数量。

## 十、术语与完成标准

topic：主题；mixed membership：同文档多个主题比例；UMAP：降维方法；HDBSCAN：密度聚类；c-TF-IDF：簇级代表词权重；coherence：主题一致性。

完成检查：

- 能解释两条路线输出的区别。
- 能识别-1离群点。
- 能说明components_归一化与否。
- 能用代表文档检查主题名。

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

