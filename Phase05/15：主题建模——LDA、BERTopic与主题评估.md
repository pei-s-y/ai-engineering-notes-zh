# 15：主题建模——LDA、BERTopic与主题评估

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/15-topic-modeling) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/15-topic-modeling/docs/en.md)。原题：Topic Modeling — LDA and BERTopic。

## 一、这一课解决什么问题？

面对没有类别标签的大量文本，自动整理常见话题。本课比较概率主题模型与“句向量聚类后解释主题”的路线；得到的话题不一定等于你预设的分类标签。

## 二、核心概念与推导

LDA把每篇文档视为多个主题的混合，每个主题又是词的概率分布。doc_topic形状(N,K)，每行和为1；sklearn的components_是未归一化的主题词权重，不能直接当概率，需按行除以总和。

BERTopic通常先用SentenceTransformer生成句向量，再用UMAP降维、HDBSCAN聚类，最后用c-TF-IDF挑出代表词。这里的c指class：把同一簇的文本作为一类统计，不等于普通逐文档TF-IDF。主题-1通常表示离群点，不是第零个正常主题。聚类硬标签和可选membership不能直接等同LDA的混合比例。

c_v coherence衡量代表词的共现一致性；diversity衡量不同主题代表词是否重复。高分仍可能得到对业务无用的主题，因此还需要人工命名、查看代表文档并检查稳定性。

### 2.9 用一个小例子把概念串起来（补充）

假设一篇文章的doc_topic=[0.7,0.2,0.1]，表示在LDA约定下偏主题0、仍混有其他主题，不是三个分类标签的独立yes概率。若两主题前2词为[cat,dog]、[cat,bird]，diversity=3/4=0.75。代表词解释弱时，必须同时看代表文档，不能只让LLM编一个好听主题名。

## 三、操作前准备

~~~bash
python -m pip install numpy scikit-learn bertopic sentence-transformers gensim
~~~
LDA不要求GPU；BERTopic依赖预训练向量模型，首次运行可能下载。原文documents是占位变量，要先提供真实字符串列表。min_df=2要求词至少出现在两篇文档；只有两三篇文本时很容易被过滤为空。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

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

