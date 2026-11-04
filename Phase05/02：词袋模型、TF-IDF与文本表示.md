# 02：词袋模型、TF-IDF 与文本表示

> 阅读目标：只看本篇，就能理解 BoW、TF、DF、IDF、L2 归一化的关系，完成原课 Build It，使用 scikit-learn，并解释实际运行结果。
> 2026-11-04 补充：为什么需要词袋（分类器只吃数字、三个子问题）、为什么敢丢顺序、词袋解决与未解决问题的对照、词表成千上万怎么办（裁剪/稀疏存储/统计量）、TF/DF/IDF 公式直觉与手算对照；新增 2.2、2.6、2.7 小节并充实 2.4，原 2.2～2.4 顺延为 2.3～2.5。

本篇沿用原课 **The Problem → The Concept → Build It → Use It → Hybrid → Ship It → Exercises** 的内容顺序。必要代码直接嵌入正文，不另附脚本或代码文件夹。原文的教学简化和不准确表述会明确标注。

- 原课类型：Build；语言：Python；原课时间估计约 75 分钟。
- 先修：上一课文本处理，以及基本列表、字典、循环、函数和向量概念。
- 来源：[课程页面](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf)、[配套英文文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/docs/en.md)。
- 正文五步手写函数及示例已运行核对；scikit-learn 部分依据官方接口与公式整理，未在本次环境中安装运行。文中会分别注明实测和公式参考结果。

## 一、The Problem：字符串怎样变成固定长度的向量？

上一课得到的是字符串 token 列表，例如：

~~~python
["the", "cat", "sat"]
~~~

分类器通常不能直接把这份字符串列表当成数值特征。问题是：

> 句子长短不同，怎样把它们表示成处于同一个坐标系、维数一致的数值向量？

补充（为什么必须先解决这一步）：机器学习分类器的数学形式是 z=Xw+b——输入 X 必须是数字矩阵，每行一篇文档、每列一个特征、所有文档共用同一套列；字符串既不能相加，也不能算距离。具体要回答三个子问题：

1. 每列是什么？（特征怎样定义）
2. 每格填什么数？（怎样把“出现”变成数值）
3. 句子长短不一怎么办？（向量维数怎样统一）

本课给出的基础方案是：先规定词表中每个词对应哪一列，再统计每份文档在各列的值。词袋模型可以一句话概括：把句子扔进袋子摇一摇，顺序全掉出去，只剩下“有哪些词、各出现几次”。

这里的 document（文档）不一定是一整篇文章，也可以是一句话、一条评论或一个客服工单。corpus（语料库）就是一组文档。

~~~text
文本 → 分词 → 词表 → 词频矩阵 BoW
                       ↓
                  TF × IDF
                       ↓
                 L2 归一化
                       ↓
                分类器或相似度计算
~~~

这条流程不会自己生成分类标签。它负责特征表示，分类器训练是后续工作。

## 二、The Concept：先理解 BoW，再理解 TF-IDF

### 2.1 BoW：每一列表示一个词

Bag of Words，中文叫词袋模型。把句子看成一袋词，只统计出现次数，不保留完整顺序。

假设词表是：

~~~text
列序号： 0    1    2   3    4
词：    cat  sat  on  mat  ran
~~~

那么：

~~~text
cat sat on mat → [1, 1, 1, 1, 0]
cat cat ran    → [2, 0, 0, 0, 1]
~~~

两条文档长度分别是 4 和 3，但都得到 5 维向量，因为维数由共同词表决定。

**词表编号与词频不是一回事：** cat 的列编号是 0，不代表它出现 0 次；第二条向量的第 0 列取值是 2，才表示 cat 出现两次。

把上一节的三个子问题对照 BoW 的回答（补充）：

| 子问题 | BoW 的回答 |
|---|---|
| 每列是什么 | 词表中一个词，列有明确含义（cat 就是第 0 列） |
| 每格填什么 | 这个词在这篇文档出现几次 |
| 长短不一 | 维数由全体文档的词表决定，与单句长度无关 |

补充（衔接[第03课](<03：词向量与从零实现 Word2Vec.md>)的视角）：每个词的 one-hot 是一个 V 维向量；一篇文档的 BoW 向量等于把其中每个词的 one-hot **加起来**。cat cat ran = cat 的 one-hot×2 + ran 的 one-hot×1 = [2,0,0,0,1]。词袋不是新东西，就是“数 one-hot 出现的次数”，这也是它零训练、纯计数的原因。

### 2.2 为什么敢把顺序扔掉（补充）

丢顺序是 BoW 看起来最“不对劲”的决定，但有两层理由：

1. 很多任务里，“出现了什么词”比“词的顺序”信息量更大。判断垃圾邮件、判断评论正负面，多数信号来自哪些词出现：winner、free、prize 出现基本就够了；把一段垃圾邮件的词打乱，它看起来还是垃圾邮件。
2. 保留顺序的方案在“统一坐标系”这个要求下反而做不下去。把词换成编号直接排 [3,17,8]，不同句子长度不同、同一位置对不齐，既当不了特征矩阵，也没法算两句话的相似度。

边界（对应 5.4 节的失败例子）：dog bites man 与 man bites dog 是同一个词袋、相反的意思；not good 里 good 的正面信号也还在。[第05课](<05：情感分析——朴素贝叶斯、逻辑回归与评估.md>)的 NOT_ 前缀与 bigram 是把最关键的局部顺序“焊”回特征里——是对 BoW 的修补，不是替换。

### 2.3 为什么原始次数还不够？

常见词可能在很多文档里都出现，仅靠次数难以突出区别。TF-IDF 把两个层面的统计结合起来：

- TF：这个词在当前文档中出现得多不多？
- IDF：这个词在整个语料库中是否普遍？

一个词在当前文档中常见、在其他文档中少见，往往得到较大的 TF-IDF。但这只是统计权重，不保证它一定是有意义的关键词；随机编号、错别字也可能很稀有。

### 2.4 TF、DF、IDF 不要混在一起

三个公式各自回答一个问题（补充总览）：

| 公式 | 回答的问题 | 作用范围 |
|---|---|---|
| TF | 这个词**在这篇文档**里多不多？ | 局部，每篇文档一套 |
| DF | 这个词在**整个语料**里铺得多广？ | 全局，每个词一个数 |
| IDF | 这个词有多**罕见**（= 有多值钱）？ | 全局，DF 的“倒过来” |

TF-IDF 的设计一句话：**一个词对一篇文档的重要性 = 局部存在感 × 全局稀有度**。

设词为 w，当前文档为 d，总文档数为 N。符号含义（补充）：

| 符号 | 含义 |
|---|---|
| w | 一个词（词表中的一项，对应矩阵的一列） |
| d | 当前这篇文档（对应矩阵的一行） |
| N | 语料总文档数 |
| count(w,d) | 词 w 在文档 d 里出现的次数 |
| \|d\| | 文档 d 的总词数（长度） |

手算示例统一使用第六节的三篇语料（the cat sat / the dog sat / the cat ran，N=3）。

$$
TF(w,d)=\frac{\operatorname{count}(w,d)}{|d|}
$$

分子是这个词在这篇文档出现的次数，分母是这篇文档的总词数。**除以长度的唯一目的是让长短文档可比**（补充）：“cat 出现 10 次”在 100 词的短评里是 10% 的存在感，在 10 万词的书里是万分之一；不归一化，长文档所有词的计数都天然偏大，没法放在同一杆秤上。例如 doc1（the cat sat，长度 3）三个词各出现 1 次，TF 均为 1/3。

这是本课手写代码使用的相对词频。TF 还有原始次数、对数缩放等其他定义。

$$
DF(w)=\text{包含词 }w\text{ 的文档数量}
$$

DF 量的是**覆盖面**而不是总量（补充）：对着三篇语料数一遍，the 在 3 篇里都出现 → DF=3；cat 出现 2 次、分布在 2 篇里 → DF=2；sat 同为 2；dog、ran 各只在 1 篇 → DF=1。

一篇文档里同一个词出现 100 次，对 DF 仍只贡献 1；对原始词频才贡献 100。

计算时机（补充，对应 7.2 节）：DF 是**全语料统计**，每个词算一次，所有文档共享同一份；这跟 TF（每篇文档各算各的）正好相反。

原文概念部分介绍的基础 IDF 是：

$$
IDF(w)=\ln\frac{N}{DF(w)}
$$

直觉（补充）：越常见的词越没有区分度——the 每篇都有，看到它猜不出这是哪篇；dog 只在 doc2 有，看到它基本锁定。取倒数 N/DF 就是把“铺得广”翻译成“不值钱”。**外面套 ln 是压缩动态范围**：语料一大，N/DF 从 1 到 10 万都可能，直接当权重会让罕见词碾压一切；ln 把乘性的大跨度压成加性的小刻度。注意 6.4 节的警告：ln 是缓和，不是给 IDF 固定上界。

基础公式手算（N=3，补充）：

| 词 | DF | 基础 IDF = ln(3/DF) |
|---|---:|---:|
| the | 3 | ln(1) = **0** |
| cat / sat | 2 | ln(1.5) ≈ 0.405 |
| dog / ran | 1 | ln(3) ≈ 1.099 |

the 的 IDF 为 0：出现在所有文档的词被完全压掉——行为干净，但埋了个问题（这个词彻底消失），引出平滑版。

但 **Build It 实际实现** 使用平滑版本：

$$
IDF_{\text{smooth}}(w)=\ln\frac{N+1}{DF(w)+1}+1
$$

三个零件各修一个问题（补充）：

| 零件 | 修什么 |
|---|---|
| DF+1 | DF=0 的词不至于除零；本课从当前语料建词表、收录词 DF 至少为 1，更多是保险（见 6.4 节） |
| N+1 | 与 DF+1 配套，保持分数结构 |
| 对数**外**的 +1 | 抬基线：基础公式下 the 为 0（彻底消失），平滑版下 the = ln(4/4)+1 = 1，仍保留小权重 |

因此：

- 不平滑基础公式下，出现在所有文档里的词，IDF 为 0。
- 本课平滑代码下，同样的词，IDF 为 1，而不是 0。
- math.log 默认是自然对数 ln，不是以 10 为底的对数。

平滑版手算见 6.3 节的表格（the=1.000、cat/sat≈1.288、dog/ran≈1.693）：与基础版**排序相同**，只是间距压窄、地板从 0 抬到 1（补充）。

最后：

$$
TFIDF(w,d)=TF(w,d)\times IDF(w)
$$

逐项相乘的含义（补充，完整手算见 7.3 节）：doc1 的 TF=[1/3, 1/3, 1/3, 0, 0]，IDF=[1, 1.288, 1.288, 1.693, 1.693]，乘积 ≈ [0.333, 0.429, 0.429, 0, 0]。the、cat、sat 出现次数一模一样（TF 全是 1/3），但 cat 和 sat 的权重压过 the——**同样的局部存在感，稀有词胜出**，这就是 TF-IDF 的全部意义。另注意 0 的两种来源：dog/ran 为 0 是 TF=0（本篇没有它），不是 IDF 低；重要性永远针对（词，文档）这一对而言。

原文把“概念公式”和“代码公式”放在不同段落中，读的时候要主动区分，不能混用数值。

### 2.5 “稀疏”不是“数值很小”

稀疏向量指大部分位置为 **0**，不是大部分位置接近 0。一个词没有出现在某篇文档中，相应位置才是 0。

BoW、TF-IDF 的每列都有明确词义，适合检查线性分类器的词特征权重；但权重不是因果效应，也不是完整的模型推理过程。

### 2.6 词袋解决了什么、没解决什么（补充总结）

| 问题 | 词袋的回答 | 遗留问题 → 谁来接 |
|---|---|---|
| 文本→固定长度数字 | 解决：统一坐标系 | — |
| 列可解释 | 解决：每列就是一个词 | — |
| 高频虚词干扰 | 没解决：the 与 cat 同样计数 | TF-IDF（本课后半） |
| 词与词的相似度 | 没解决：dog 与 puppy 是两列，余弦恒为 0 | [第03课](<03：词向量与从零实现 Word2Vec.md>) Word2Vec |
| 词序与否定 | 没解决：not good 与 good 的信号分不开 | bigram、NOT_ 前缀（[第05课](<05：情感分析——朴素贝叶斯、逻辑回归与评估.md>)），最终靠上下文表示 |
| 中文没有空格 | 没解决：直接 split() 会失效 | 分词须先做（[第01课](<01：文本处理——分词、词干提取与词形还原.md>)） |

一句话：词袋是“文本→数字”的第一块砖，用最小代价（纯计数、零训练）换来统一坐标系和可解释特征；它故意丢掉的东西，正好划出了 TF-IDF、词向量和上下文表示各自要解决的问题。第04课选型表里“生产级文本分类配线性模型仍是 TF-IDF”一行，说的就是这块砖至今仍然够硬。

### 2.7 词表成千上万怎么办（补充）

先纠正一个想象：没有人手工“做”出这几万列。Build It Step 1 的 build_vocab 遇到新词就自动加一列（`if token not in vocab: vocab[token] = len(vocab)`），词表是语料的自然产物——英文真实语料扫出十几万“词”（含错拼与专名变体）很正常。工程问题不是“怎么造出来”，而是“太长怎么办”。靠三层机制：

**层 1：裁剪——太常见与太罕见都不要。** 这是最反直觉的一点：词袋留下的是“中间段”的词。

| 工具 | 干掉谁 | 为什么 |
|---|---|---|
| max_df=0.95 | 出现在 95% 以上文档的词（the、and） | 每篇都有，没有区分度，纯占列 |
| min_df=2 | 只出现在 1 篇文档的词 | 只见过一次学不到统计，却是词表的大头 |
| max_features=k | 硬性只留词频前 k 名 | 直接给词表设预算上限 |

砍长尾损失不大，因为 **Zipf 定律**：语言里少数词占掉绝大多数出现次数，长尾极长但每条都细，前几千个词已覆盖 90% 以上的 token 出现。注意第十一节自己的警告：情感任务别盲删停用词（not 带信号）；玩具语料上 min_df=3 可能把词删光报空词表错。

**层 2：剩下的几万列，不按几万列存——稀疏存储。** 设 10,000 篇文档、每篇约 100 词、V=50,000：

~~~text
稠密：10,000 × 50,000 = 5 亿格 × 8 字节 ≈ 4 GB   ← 10.5 节 toarray() 危险的原因
稀疏：每篇最多约 100 个非零 → 100 万项 × 约 12 字节 ≈ 12 MB
~~~

差 300 多倍。V 维只是坐标系的名义大小，实际内存跟着非零项走——这正是 2.5 节“稀疏”的含义，也是 CountVectorizer/TfidfVectorizer 默认返回稀疏矩阵的原因（10.4 节）。

**层 3：列多，线性分类器也不怕。** 稀疏矩阵的 X·w 实现上只遍历非零项：一篇 100 词的文档算一次 z=Xw+b 只做约 100 次乘加，不是 5 万次。被裁掉的词在新文档中出现时被 transform 直接忽略（13.3 节的 OOV 处理），不报错。

**真正的代价不是内存，是每列统计量变薄：** 一列（一个词）只在 3 篇文档里出现，它的权重只由 3 个样本决定，容易过拟合——min_df 的深层理由是删掉不可靠的统计，不是省内存。[第04课](<04：GloVe、fastText与子词表示.md>) BPE 词表定 32k 还是 100k 是同一预算问题（覆盖度 vs 每个单位见到的样本量）；词表大小是所有文本表示方法的共同预算，词袋、BPE、现代 tokenizer 都逃不掉。

一句话：词表自动长出来 → min_df/max_df 砍两头（Zipf 长尾）→ 稀疏存储只记非零 → 线性模型只碰非零项。真正的预算约束是“每个词能见到多少样本”，不是内存。

## 三、动手方式与环境

### 3.1 手写部分不需要第三方库

Build It 只用标准库 math。把 Step 1～5 的函数按顺序放在同一个 Python 环境里，再运行示例即可。

你可以在 Python 交互环境、Notebook 或自己的练习文件中执行。本文不在笔记目录另外生成代码附件；完整项目已有的文件继续使用 GitHub 仓库。

### 3.2 在 WSL Ubuntu 中操作

如果沿用上一课创建的虚拟环境：

~~~bash
cd ~/projects/text-processing-lesson
source .venv/bin/activate
python --version
~~~

如果没有这个目录，使用你实际的学习项目及虚拟环境即可。标准库部分即使没有虚拟环境也能运行；独立环境主要用于后面安装 scikit-learn，避免混用依赖。

运行原 GitHub 示例时，应在本地仓库中打开本课对应文件。下文展示的是原课正文函数的输出，不把未读取到的仓库主程序输出当成实测结果。

## 四、Build It · Step 1：建立词表

### 4.1 原课代码的中文注释版

~~~python
def build_vocab(docs):
    # docs 是“文档列表”，每篇文档已经被分成字符串 token 列表
    vocab = {}

    for doc in docs:
        for token in doc:
            # 第一次见到某个词时，给它分配一个新列编号
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab
~~~

### 4.2 输入类型很重要

正确输入：

~~~python
docs = [
    ["cat", "sat", "on", "mat"],
    ["cat", "cat", "ran"]
]
~~~

不要误传原始字符串列表后以为函数会自动分词。这里内部直接遍历 doc；如果 doc 是一个字符串，循环会逐字符处理它。

按首次出现顺序扫描：

~~~text
cat → 0
sat → 1
on  → 2
mat → 3
ran → 4
~~~

再次遇到 cat 时不新增列。

### 4.3 测试与实测输出

~~~python
docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
vocab = build_vocab(docs)
print(vocab)
~~~

~~~text
{'cat': 0, 'sat': 1, 'on': 2, 'mat': 3, 'ran': 4}
~~~

len(vocab) 是当前已有词数，恰好可用作下一个从 0 开始的编号。这个教学实现按首次出现顺序建词表；scikit-learn 自动学习的特征列通常按词项排序，比较矩阵前要先对齐列名。

## 五、Build It · Step 2：生成 BoW 矩阵

### 5.1 原课代码的中文注释版

~~~python
def bag_of_words(docs, vocab):
    # 每篇文档一行，每个词表项一列，初始全部为 0
    matrix = [[0] * len(vocab) for _ in docs]

    for i, doc in enumerate(docs):
        for token in doc:
            # 不在训练词表中的词被忽略，不临时扩展维度
            if token in vocab:
                matrix[i][vocab[token]] += 1

    return matrix
~~~

### 5.2 逐项解释

- enumerate(docs) 同时返回文档编号 i 和这篇文档 doc。
- vocab[token] 查出这个词应该放在哪一列。
- matrix[i][vocab[token]] += 1 在对应文档、对应词的位置加一。
- matrix 的形状是“文档数 × 词表大小”，记为 N × V。

第二条文档中的 cat 出现两次，会对同一位置加两次，而不是创建两列。

**补充 Python 易错点：** 不要把初始化改成 [[0] * V] * N。后者可能让所有行引用同一个内部列表，改一行时其他行也跟着变。原文的列表推导式为每行创建独立列表。

### 5.3 原课例子与实测输出

继续使用 Step 1 的 docs、vocab：

~~~python
bow = bag_of_words(docs, vocab)
print(bow)
~~~

~~~text
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
~~~

也可以按行看：

| 文档 | cat | sat | on | mat | ran |
|---|---:|---:|---:|---:|---:|
| cat sat on mat | 1 | 1 | 1 | 1 | 0 |
| cat cat ran | 2 | 0 | 0 | 0 | 1 |

### 5.4 BoW 丢失了什么？

dog bites man 与 man bites dog 的词计数相同。只依赖相同 unigram BoW 向量的模型无法区分这两个顺序，即使训练数据再多也不能从该向量找回已经丢失的信息。

这是表示能力的限制，不只是分类器没训练好。后面的 n-gram 可以补充部分局部顺序，但仍不保留完整句法。

## 六、Build It · Step 3：计算 TF、DF 和 IDF

### 6.1 原课代码的中文注释版

~~~python
import math

def term_frequency(doc_bow, doc_length):
    # 单篇文档中，各词计数除以文档长度
    # 长度为 0 时返回全 0，避免除以 0
    return [c / doc_length if doc_length else 0 for c in doc_bow]

def document_frequency(bow_matrix):
    # 每个词一项；本实现要求至少有一行
    df = [0] * len(bow_matrix[0])

    for row in bow_matrix:
        for j, count in enumerate(row):
            # 只判断这一篇是否出现，不累加该篇出现的总次数
            if count > 0:
                df[j] += 1

    return df

def inverse_document_frequency(df, n_docs):
    # 平滑 IDF：注意分数内部与对数外部的两个 +1
    return [
        math.log((n_docs + 1) / (d + 1)) + 1
        for d in df
    ]
~~~

### 6.2 使用原文的第二组语料

这里开始更换 docs，并重新建词表，不沿用上一组五列的含义：

~~~python
docs = [
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
    ["the", "cat", "ran"]
]
vocab = build_vocab(docs)
bow = bag_of_words(docs, vocab)

print(vocab)
print(bow)
print(document_frequency(bow))
~~~

~~~text
{'the': 0, 'cat': 1, 'sat': 2, 'dog': 3, 'ran': 4}
[[1, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1]]
[3, 2, 2, 1, 1]
~~~

### 6.3 手算 DF 与 IDF

总文档数 N = 3，列顺序为 the、cat、sat、dog、ran。

| 词 | DF：包含它的文档数 | 平滑 IDF |
|---|---:|---:|
| the | 3 | ln(4/4) + 1 = 1.000000 |
| cat | 2 | ln(4/3) + 1 = 1.287682 |
| sat | 2 | ln(4/3) + 1 = 1.287682 |
| dog | 1 | ln(4/2) + 1 = 1.693147 |
| ran | 1 | ln(4/2) + 1 = 1.693147 |

the 遍布所有文档，得到较低 IDF；dog、ran 只在一篇里出现，得到较高 IDF。

### 6.4 两个 +1 分别在做什么？

- 分数中的 N+1、DF+1 是平滑，能让固定词表中 DF=0 的词也不发生除零。
- 对数外的 +1 提高基线，让遍布所有文档的词仍有权重 1。

对只从当前语料建出的词表，已收录词的 DF 本来至少为 1；不要误以为训练语料里必然有 DF=0 项。

原文说 log 让权重“有界”，不应理解为 IDF 在所有语料规模下都有固定上界。这里 ln 的作用主要是缓和比例变化；IDF 仍随 N、DF 和公式约定改变。

## 七、Build It · Step 4：组合为 TF-IDF

### 7.1 原课代码的中文注释版

~~~python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)

    # DF、IDF 对整个语料计算，所有行共享同一份 IDF
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)

    out = []
    for row in bow_matrix:
        # 本代码把词表中计到的词数之和作为文档长度
        length = sum(row)
        tf = term_frequency(row, length)

        # 按列把 TF 与 IDF 相乘
        out.append([
            tf_j * idf_j
            for tf_j, idf_j in zip(tf, idf)
        ])

    return out
~~~

### 7.2 为什么 IDF 在循环外计算？

IDF 是整个训练语料的共享统计，不是每篇文档自己算一套。TF 才会随着当前文档变化。

zip(tf, idf) 按相同列序把两个列表配对，然后逐项相乘。维度必须一致；不能把两份不同词表顺序的向量直接相乘。

### 7.3 手算第一篇文档

第一行 BoW：

~~~text
[1, 1, 1, 0, 0]
~~~

文档长度为 3，因此：

~~~text
TF  = [1/3, 1/3, 1/3, 0, 0]
IDF = [1, 1.287682, 1.287682, 1.693147, 1.693147]
~~~

逐项相乘：

~~~text
TF-IDF ≈ [0.333333, 0.429227, 0.429227, 0, 0]
~~~

dog 虽然 IDF 大，但没有出现在这篇文档中，TF 为 0，最终仍为 0。**稀有不等于每篇文档都给它非零分数。**

### 7.4 三篇文档的实测结果

~~~python
weighted = tfidf(bow)
for row in weighted:
    print([round(x, 6) for x in row])
~~~

~~~text
[0.333333, 0.429227, 0.429227, 0.0, 0.0]
[0.333333, 0.0, 0.429227, 0.564382, 0.0]
[0.333333, 0.429227, 0.0, 0.0, 0.564382]
~~~

这里 round 只是为了显示易读，后续计算应继续使用未舍入的 weighted，避免过早丢失精度。

### 7.5 原代码的输入边界

- 空列表 bow_matrix=[] 会在 document_frequency 访问第一行时出错，原课没有提供完善输入校验。
- 一篇空文档可以对应一行全 0，TF 函数会避免除零。
- 新文档中的词若不在 vocab 中会被忽略；sum(row) 是“被词表识别到的词数”，不一定等于原始文本 token 总数。
- 这份教学 tfidf 函数同时计算 IDF 和转换；部署时不能对每批测试文档重新调用它来“拟合一套新 IDF”。

## 八、Build It · Step 5：L2 归一化

### 8.1 先理解向量长度

这里的长度是欧氏范数，不是列表里有多少个元素：

$$
\|x\|_2=\sqrt{\sum_j x_j^2}
$$

对于非零向量：

$$
\hat{x}_j=\frac{x_j}{\|x\|_2}
$$

例如 [3,4] 的范数是 5，归一化后得到 [0.6,0.8]。方向没变，欧氏长度变成 1。

### 8.2 原课代码的中文注释版

~~~python
def l2_normalize(matrix):
    out = []

    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))

        # 零向量没有可归一化方向，代码约定继续保留全 0
        out.append([
            x / norm if norm else 0
            for x in row
        ])

    return out
~~~

### 8.3 实测输出

~~~python
normalized = l2_normalize(weighted)
for row in normalized:
    print([round(x, 6) for x in row])
~~~

~~~text
[0.481334, 0.619805, 0.619805, 0.0, 0.0]
[0.425441, 0.0, 0.547832, 0.720333, 0.0]
[0.425441, 0.547832, 0.0, 0.0, 0.720333]
~~~

非零行的元素平方和约为 1；不是元素本身加起来等于 1。元素和为 1 对应另一种归一化思想，不能混淆。

### 8.4 它与文档长度、相似度有什么关系？

L2 归一化去掉每行整体尺度的差异，便于比较向量方向。

原文说“不归一化时长文档的向量就更大”，这是对原始计数常见现象的简化。本课 TF 已经除过行总计数；如果只是把一篇文档中的每个词重复相同倍数，相对 TF 本来就不会改变。因此，L2 不是唯一的长度处理，也不能简单说“文本长就一定 TF-IDF 范数大”。

零向量不会落到单位球面，原代码会保持全 0。

## 九、把五步连起来：运行检查与余弦相似度

定义好前面的全部函数后，执行下面这段补充演示。不需要另外下载训练数据。

~~~python
docs = [
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
    ["the", "cat", "ran"]
]
vocab = build_vocab(docs)
bow = bag_of_words(docs, vocab)
weighted = tfidf(bow)
normalized = l2_normalize(weighted)

print("vocab:", vocab)
print("bow:", bow)
print("df:", document_frequency(bow))
print("idf:", [
    round(x, 6)
    for x in inverse_document_frequency(document_frequency(bow), len(docs))
])
print("tfidf:", [[round(x, 6) for x in row] for row in weighted])
print("normalized:", [[round(x, 6) for x in row] for row in normalized])

# 已经做过 L2 归一化的非零向量，余弦相似度等于点积
score = sum(a * b for a, b in zip(normalized[0], normalized[1]))
print("cosine(d0,d1):", round(score, 6))
~~~

最后一项实测为：

~~~text
cosine(d0,d1): 0.544328
~~~

完整的余弦公式为：

$$
\cos(x,y)=\frac{x\cdot y}{\|x\|_2\|y\|_2}
$$

当两者都已是单位向量时，分母为 1，直接计算点积。

要记住三个边界：

1. 非零向量与自身比较，结果约为 1；同方向但不同长度的向量也可以是 1，不代表文本逐字相同。
2. 在本课非负 TF-IDF 表示下，没有共同非零词项时，相似度为 0。
3. 零向量的余弦在数学上未定义；若程序返回 0，那是明确选择的处理约定，而不是公式自动给出的结果。

### 补充断言检查

~~~python
assert bag_of_words([["cat", "cat"]], {"cat": 0}) == [[2]]
assert document_frequency([[2, 0], [1, 3]]) == [2, 1]
assert inverse_document_frequency([3], 3) == [1.0]
assert term_frequency([0, 0], 0) == [0, 0]
assert l2_normalize([[0, 0]]) == [[0, 0]]
assert abs(sum(x * x for x in normalized[0]) - 1.0) < 1e-12
assert bag_of_words([["unknown"]], vocab) == [[0] * len(vocab)]
print("基础检查通过")
~~~

这些检查把“词次数”“文档次数”“平滑 IDF”“零向量”分别固定下来，能帮助你定位是哪一步理解错了。


## 十、Use It：使用 scikit-learn

### 10.1 安装方式

在你实际使用的虚拟环境中执行：

~~~bash
python -m pip install scikit-learn
~~~

安装包名是 scikit-learn，Python 导入名是 sklearn。当前手写部分不依赖它；本次整理未替你安装。

### 10.2 原课示例的中文注释版

这里使用的是原文 Use It 的另一组原始字符串，不是前面的三词句。向量器会自行处理文本：

~~~python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat ran"
]

# 自动完成预处理、分词、词表学习与计数
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(texts)

# 显示每一列的含义，必须和矩阵一起看
print(bow_vectorizer.get_feature_names_out())
print(bow_matrix.toarray())

# 默认还会执行平滑 IDF 加权与 L2 归一化
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
print(tfidf_matrix.toarray().round(3))
~~~

**说明：** 为避免覆盖前面定义的 tfidf 函数，这里将原文的局部变量名改为 tfidf_matrix；表示方法和调用方式保持一致。

这段代码可以整体复制到自己的 Python 环境执行，不依赖前面的手写函数。

### 10.3 参考列名与计数矩阵

下面是按文档内容和默认公式计算得到的参考，不是本次运行 sklearn 得到的实测：

~~~text
列名：
['cat', 'dog', 'mat', 'on', 'ran', 'sat', 'the']

BoW：
[[1, 0, 1, 1, 0, 1, 2],
 [0, 1, 1, 1, 0, 1, 2],
 [1, 0, 0, 0, 1, 0, 1]]
~~~

对应 TF-IDF 参考值，保留三位小数：

~~~text
[[0.395, 0.000, 0.395, 0.395, 0.000, 0.395, 0.613],
 [0.000, 0.492, 0.374, 0.374, 0.000, 0.374, 0.581],
 [0.548, 0.000, 0.000, 0.000, 0.720, 0.000, 0.425]]
~~~

它与第八节数值不同，首先是因为 **语料不同、列顺序不同**，不是代码自然算错。

### 10.4 手写版与库默认值的区别

当前 scikit-learn 的默认约定包括：

- 将文本转小写；默认词级规则通常保留至少两个词字符组成的项，忽略标点。
- 默认不删除停用词。
- TF 默认使用原始计数，不是先除以文档长度。
- 默认平滑 IDF，与本课代码的 ln((N+1)/(DF+1))+1 一致。
- 默认按行做 L2 归一化。
- 返回稀疏表示，不是 Python 的二维列表。

最后三项分别可从 TfidfTransformer、TfidfVectorizer 的参数核对。[CountVectorizer 文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)、[TfidfVectorizer 文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)、[TF-IDF 公式说明](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

**一个容易误判的地方：** 在同一词表、同一 IDF、普通计数 TF 下，每行先除以同一个正数，再做 L2，整体缩放会抵消。因此本课的相对 TF 与库的原始计数 TF，经过 L2 后可以得到相同方向的结果。未做 L2、启用对数 TF 或分词不同，就不能据此宣称所有配置都相等。

如果要比较，先对齐语料、列顺序、分词、IDF、TF 和归一化，而不是只比较函数名字是否叫 TF-IDF。

### 10.5 为什么不要随便 toarray？

toarray 把稀疏表示转成稠密数组。原课的小示例可以这样打印；真实数据可能把内存用完。

例如 N=100,000 篇文档、V=50,000 个词，若以 8 字节浮点数稠密存储，仅数值就需要：

~~~text
100,000 × 50,000 × 8 = 40,000,000,000 字节
约 40 GB，不含额外开销
~~~

所以原文“100k 文档就一定放不进内存”不够严谨，实际还取决于 V、数据类型和硬件。大数据通常应保留稀疏表示，只查看少量行或非零项。

## 十一、Use It 的五个关键参数

| 参数 | 实际作用 | 不要误解为 |
|---|---|---|
| ngram_range=(1,2) | 同时包含单词和相邻二元词组 | 只保留二元组 |
| min_df=2 | 去掉出现在少于 2 篇文档中的词项 | 词在单篇里出现两次就保留 |
| max_df=0.95 | 去掉出现在超过 95% 文档中的词项 | 删除总出现次数最多的 95% 词 |
| stop_words="english" | 使用内置英语停用词表 | 所有任务都应该删除这批词 |
| sublinear_tf=True | 对正计数使用 1+ln(count) | 对零计数也直接取对数 |

几个细节：

- min_df、max_df 传整数时表示文档个数，传合法比例浮点数时表示文档占比；例如 min_df=2 与 min_df=0.02 不同。
- sublinear_tf 是对非零计数做对数缩放，减少重复次数极大的词的优势。
- ngram_range=(1,2) 能提供 not good 这样的局部组合，但不是完整语法理解。
- 停用词处理需要检查是否丢掉否定、功能状态等信号；这里保留 stop_words=None，不盲删。
- 原课没有要求你在三篇玩具语料上照搬 min_df=3、max_df=0.95。那样可能把所有候选词删光，报空词表错误。

参数应在训练/验证流程中选择，不能反复看测试集分数挑配置。

补充（回链 2.7 节）：这张参数表正是控制词表规模的工具——min_df 与 max_df 裁掉 Zipf 长尾的两头，max_features 直接设预算；词表大小的深层权衡（每列统计量变薄）见 2.7 节。

## 十二、补充：训练、验证、推理怎么保持一致？

这是让原课的教学代码走向实际使用时必须补上的概念。

### 12.1 三个操作的区别

- fit：学习词表、DF/IDF 等训练统计。
- transform：使用已经学到的统计转换新文档。
- fit_transform：学习并转换当前训练数据。

正确示范：

~~~python
from sklearn.feature_extraction.text import TfidfVectorizer

train_texts = ["the cat sat", "the dog sat", "the cat ran"]
new_texts = ["cat spaceship", "spaceship"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_new = vectorizer.transform(new_texts)  # 不重新 fit

print(vectorizer.get_feature_names_out())
print(X_new.shape)
~~~

这里的新词 spaceship 不在训练词表里，会被忽略。第二篇只剩词表外词，可能成为零向量；不会自动获得一个新列。

**此 sklearn 代码本次只做语法检查，运行结果需在安装库后观察。**

### 12.2 不应该做什么？

不要把训练集和测试集合起来建立词表、IDF，再声称测试完全独立；不要对每批新文档重新 fit 一套不同的列和权重。

本课手写 tfidf(bow_matrix) 没有拆分 fit 与 transform，所以对测试矩阵直接再调用它，会重新估计测试语料的 IDF。实际部署应固定训练词表和 IDF，或使用库提供的对象接口。[训练/推理一致性与泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

还要记住：向量器只产生特征，不会自动学习“垃圾邮件”“正面评论”等标签，需要再接监督分类器。

## 十三、When It Works / Fails：原课的应用讨论

### 13.1 TF-IDF 为什么值得作为基线？

原课列举垃圾邮件、主题分类、日志标记等任务，强调速度、词项可解释性和实现成本。当标签与词项出现密切相关时，TF-IDF 加线性模型可以是有竞争力的基线。

但以下原文说法不能当作通用定律：

- “2026 年 TF-IDF 总能打败嵌入”不是对所有任务成立的结论。
- “微秒对 10～100 毫秒”必须有明确硬件、文本长度、批量和模型配置，不能作为你的实测延迟。
- 样本少不意味着预训练表示必然不好；冻结预训练向量与从零训练大模型不是同一回事。

更合适的实验方式是：在同一划分、同一指标和相近调参预算下比较，并同时记录延迟、内存和解释需求。

### 13.2 原课的语义失败例子

~~~text
The movie was not good at all.
The movie was excellent.
~~~

两句一负一正，共享一些普通词。TF-IDF 不会因为 good 与 excellent 在某些语境里语义相关，就自动把它们放到相近维度；它也不直接表示否定的作用域。

**原文纠错：** 只使用 unigram BoW 的分类器可以学到 not 的标签相关性，但不能从计数向量中知道它“靠近 good”，因为相邻关系已被丢弃。要表达局部邻接，需要加入 bigram 等特征，或者使用其他序列表示。

### 13.3 词表外词 OOV

固定词表无法为没见过的完整词临时增加新轴，所以某些新表达可能丢失信号。但不要把它说成“所有 TF-IDF 都绝对无法缓解”：字符 n-gram、不同分词粒度等方案可以改善部分情况。

子词或其他嵌入方法也有各自覆盖条件；本课后面的静态 GloVe 词表仍然可能遇到 OOV，并不是“换成 embedding 就自动覆盖全部词”。

## 十四、Hybrid：TF-IDF 加权词向量

### 14.1 想法：把多个词向量汇总为文档向量

假设每个词已有一个 d 维 embedding。可以把词向量按某些权重加起来，再除以总权重，得到固定 d 维的文档表示。

这与 BoW 的“词表多大就多少维”不同：词向量维度可固定为 100 等数值，文档中各个词共享同一个向量空间。

原文称 TF-IDF 权重像 attention，应理解为“加权汇总”的类比；它不是 Transformer 中学习得到的 Q/K/V 注意力机制。

### 14.2 原课代码的中文注释版

~~~python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0

    # 注意：原文逐次遍历 token，不是遍历不重复词项
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue

        weight = tfidf_scores[token]
        emb = embedding_table[token]

        for i in range(dim):
            vec[i] += weight * emb[i]

        total_weight += weight

    # 所有词都没被匹配时，约定返回零向量
    if total_weight == 0:
        return vec

    return [v / total_weight for v in vec]
~~~

输入约定：

- doc：分词后的文档。
- tfidf_scores：该篇文档“词 → TF-IDF 权重”的字典，不是整个语料的一张统一词权重表。
- embedding_table：“词 → 向量”的字典，向量长度应与 dim 一致。
- dim：最终文档向量维度。

词缺失时跳过；所有词都缺失时返回零向量。代码没有检查词向量维度错误等工程问题。

### 14.3 不下载 GloVe，也能先理解加权过程

下面是补充的二维玩具向量测试，不代表真实词语的语义坐标：

~~~python
doc = ["cat", "cat", "dog"]
scores = {"cat": 2 / 3, "dog": 1 / 3}
embeddings = {
    "cat": [1.0, 0.0],
    "dog": [0.0, 1.0]
}

pooled = tfidf_weighted_embedding(doc, scores, embeddings, dim=2)
print(pooled)
~~~

实测输出：

~~~text
[0.8, 0.2]
~~~

为什么不是 [2/3,1/3]？因为原函数对两个 cat 分别使用了一次已经含有词频的权重：

~~~text
cat 的总权重：2 × (2/3) = 4/3
dog 的总权重：1 × (1/3) = 1/3
归一化后：cat 占 4/5，dog 占 1/5
~~~

这相当于对重复词额外加了一次次数因素。

### 14.4 如果你想按每个不重复词项加权一次

常见的一种定义是：

$$
v_d=\frac{\sum_{w\in U(d)} TFIDF(w,d)e_w}
          {\sum_{w\in U(d)} TFIDF(w,d)}
$$

U(d) 表示文档中不重复的词项集合。TF-IDF 本身已经反映词频，因此不再逐次重复使用同一词权重。

可以这样对照：

~~~python
# 只改变传入函数的遍历项，不改 scores 的定义
unique_doc = list(dict.fromkeys(doc))
pooled_unique = tfidf_weighted_embedding(
    unique_doc, scores, embeddings, dim=2
)
print(pooled_unique)
~~~

结果约为：

~~~text
[0.6666666666666666, 0.3333333333333333]
~~~

**原文与补充的区别：** 原函数保留在上方，没有悄悄修改。这里不是说所有逐位置加权都错误，而是要求先说明权重是否已经包含重复次数，避免以为两种定义相同。

原文声称这种混合在少于约 50k 标注样本时必然胜过两种单独方案，没有给出足以普遍外推的实验条件。请把它当成待比较方案，而不是已保证的结果。

## 十五、Ship It：把选型思路整理成中文提示词

原课建议产出 vectorization-picker 提示词。这里直接放在笔记中，不另外创建输出文件。

下面是中文可用版：保留原文“选择表示、给出参数、指出失败模式”的结构，但把原文固定样本数禁令改为需要说明依据。

~~~markdown
你是一位文本向量化方案顾问。

请根据我提供的任务信息，输出：
1. 表示方法：BoW、TF-IDF、预训练嵌入或混合。
   用一句话说明依据，并指出哪些结论需要实验验证。
2. 可操作配置：说明库名，列出适用的 ngram_range、
   min_df、max_df、sublinear_tf、stop_words 等参数。
3. 上线前至少一个失败测试：如 OOV、否定词丢失、
   稀有类别关键词被过滤、空向量或数据泄漏。

要求：
- 标签很少时，优先建立简单基线，但不要声称固定样本数
  能决定所有任务都不能用预训练嵌入。
- 情感分析不要盲目删除否定信息。
- 类别不平衡还需要指标、训练或采样层面的处理，
  不是换个向量器就能自动解决。
- 说明训练集如何 fit，验证与测试集如何 transform。
- 不把参考延迟或他人的准确率写成当前项目实测。

我的任务：
语言与文本长度：
标注样本数、类别数与类别分布：
输入示例：
解释性要求：
延迟和内存约束：
~~~

### 原文的 30k 工单例子

原文输入：30,000 条英文客服工单、12 类、每条两三句，需要为审计提供解释。

它给出的配置示意是：

~~~python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    stop_words=None
)
~~~

保留 not working 等组合信号，并检查 min_df=3 是否删掉稀有类别的关键词。

**补充说明：** 这只是针对该情境的候选配置，不是已经验证的最佳参数。原文“需要解释就排除稠密嵌入”也太绝对：稀疏词特征更直观，但不能仅凭这个要求证明其他表示绝不可用。前面提示词中的“少于 500 条就拒绝 embedding”同样是教学规则，不是数学结论。

## 十六、Exercises：原课三项练习与操作方向

### 练习 1：实现余弦相似度（Easy）

原要求：对 L2 归一化后的 TF-IDF 向量计算相似度，验证相同文档得 1、不重叠词表得 0。

你可以从第九节的点积表达式封装成函数，检查：

1. 两个非零相同向量，结果接近 1。
2. 不共享非零列的向量，结果为 0。
3. 结果允许浮点误差，用容差比较。
4. 输入若有零向量，明确你的返回约定，不把零向量自相似度当作必须等于 1。

如果函数接收未归一化向量，还需要计算分母，不能直接使用点积。

### 练习 2：给 BoW 加 n-gram（Medium）

原要求：n=2 时，对 ["the","cat","sat"] 生成 the cat 与 cat sat 两个二元组并计数。

建议顺序：

1. 先用滑动窗口生成每篇文档的 n-gram。
2. 用同一种规则为训练语料建立 n-gram 词表。
3. 再调用或扩展计数逻辑。
4. 测试文档长度小于 n、重复二元组及空文档。

注意：原课参数 n=2 表示只取二元组；库的 ngram_range=(1,2) 表示一元和二元都取。建词表与转换时必须使用相同的 n 和同样的连接规则。

### 练习 3：GloVe 100d 混合表示对比（Hard）

原要求：使用 100 维 GloVe，在 20 Newsgroups 上比较：

- 普通 TF-IDF；
- 普通词向量均值；
- TF-IDF 加权词向量。

操作路线：

1. 从 [GloVe 官方项目](https://github.com/stanfordnlp/GloVe) 选择包含 100 维向量的发布包，下载并缓存一次；不要误把 300 维文件当成 100 维。
2. 读取时只保留需要的词项或使用适合内存的存储方式，检查每个向量长度。
3. 准备固定训练/验证/测试数据，统一文本预处理。
4. 只在训练集上学习 TF-IDF 词表与 IDF。
5. 明确混合权重按位置还是按不重复词项使用。
6. 固定或公平选择分类器，比较准确率，并可补充 macro-F1、时间、内存、OOV 和零向量比例。
7. 如实报告各方案的优势与失败案例，不预设混合必胜。

**补充数据准备示意，首次调用可能联网下载，本次未执行：**

~~~python
from sklearn.datasets import fetch_20newsgroups

train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)
test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

train_texts = train_data.data
train_labels = train_data.target
test_texts = test_data.data
test_labels = test_data.target
~~~

去掉邮件头、签名、引用可以减少模型利用某些元数据捷径；这些清理规则并非完美。各方案使用同一配置，并说明它如何影响结果。[数据集接口](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)

不要在训练集上调参后又把同一分数当成测试结论。原课没有给出该练习完整结果，本笔记也没有编造分类准确率。

## 十七、常见错误与排查

| 问题 | 优先检查 |
|---|---|
| 两份 TF-IDF 数值不同 | 语料、列顺序、tokenizer、TF、IDF 平滑、归一化是否一致 |
| 单字符词消失 | sklearn 默认 token_pattern 与你的分词规则是否不同 |
| 全部词项被删掉 | min_df、max_df、停用词及样本量是否匹配 |
| Empty vocabulary 报错 | 输入是否为空、只有标点/停用词，或全被过滤 |
| 新词没有新列 | 固定词表的正常行为，检查是否有信号损失 |
| 零向量的余弦异常 | 是否明确处理分母为 0 |
| toarray 后内存飙升 | 是否把大稀疏矩阵稠密化 |
| 验证效果异常地好 | 是否在全部数据上拟合词表/IDF，或利用元数据捷径 |
| 混合结果不符合手算 | 是否把已有词频的权重又按出现次数重复使用 |

安装或导入错误时，在同一虚拟环境检查：

~~~bash
which python
python -m pip --version
python -m pip show scikit-learn
~~~

## 十八、术语表与本课完成标准

| 术语 | 必须掌握的含义 |
|---|---|
| Corpus | 多篇文档组成的语料库 |
| Vocabulary | 特征词项与列编号的映射 |
| BoW | 按词项计数，通常不保留顺序 |
| TF | 当前文档的词频，存在多种定义 |
| DF | 包含某词的文档数，不是总出现次数 |
| IDF | 根据跨文档普遍程度计算的逆文档频率 |
| Sparse | 大部分位置为 0，不是“大部分数值较小” |
| L2 norm | 元素平方和开根号 |
| Cosine similarity | 非零向量的方向相似度 |
| OOV | 不在固定词表或向量表中的词项 |
| n-gram | 连续 n 个单位组成的特征 |
| Pooling | 把多个向量汇总成固定维度表示 |

完成标准：

- 能从两篇分词文档手写共同词表与 BoW。
- 能解释同一词出现 100 次为何对 DF 只计一篇。
- 能区分本课概念中的基础 IDF 与代码里的平滑公式。
- 能手算第一篇文档的 TF-IDF 并说明哪些位置为 0。
- 能解释 L2 为什么让非零向量的点积等于余弦。
- 能对照五步函数追踪数据类型、矩阵形状和列含义。
- 知道 sklearn 默认与手写版哪里不同。
- 知道何时 fit、何时 transform，避免测试数据泄漏。
- 能解释 unigram 的顺序限制、OOV 和混合代码的重复加权问题。

一句话总结：**BoW 把词变成计数坐标，TF-IDF 用文档内频率与跨文档普遍程度调整权重，L2 统一非零向量尺度；它们是可解释的表示工具，不等于自动理解语义或完成分类。**

## 附录：来源、验证与许可

### 内容与验证范围

- 原文的 Build It 五步函数、两个语料例子、Use It、混合表示、Ship It 和三项练习均已纳入。
- 必要代码只保存在本文中，不在你的笔记目录另外创建脚本、提示词文件或数据文件。
- 标准库示例：实际执行，核对 BoW、DF、IDF、TF-IDF、L2、余弦、OOV、零向量以及混合重复词案例。
- 第三方示例：已核对接口、公式并做语法检查，未安装 scikit-learn、下载 GloVe 或训练 20 Newsgroups 分类器。
- sklearn 的参考数值是按其默认公式计算，不冒充库运行输出。
- 原课程文档与仓库版本可能更新，使用项目自带主程序时以实际版本为准。

### 原代码的 MIT 许可

原项目作者 Rohit Ghumare。原课代码和文档按 [MIT License](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/LICENSE) 发布。代码中的中文注释、运行指引、算例解释及明确标注的纠错属于本笔记的学习整理，下面保留许可声明。

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
