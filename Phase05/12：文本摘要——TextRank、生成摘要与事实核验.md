# 12：文本摘要——TextRank、生成摘要与事实核验

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/12-text-summarization) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/12-text-summarization/docs/en.md)。原题：Text Summarization。

## 一、这一课解决什么问题？

摘要可从原文挑句子，也可重新生成措辞。前者是排序，后者是条件生成。两者都可能遗漏重要信息，生成式还可能新增没有依据的事实；抽取式即使逐字复制，也会因脱离上下文而误导。

## 二、核心概念与推导

TextRank把句子当节点，词重叠相似度当边，迭代传播重要性；最后选top-k并恢复原文顺序。原代码并非TF-IDF余弦，它用Counter交集计数除以两句独特词数的对数和。

生成摘要如BART需文档—摘要任务训练，输出可压缩和改写，也会发生实体替换、数字漂移、否定翻转及事实捏造。

ROUGE-1/2看一元/二元重叠，ROUGE-L看最长公共子序列。rouge-score的fmeasure通常0～1，论文中40分常是乘100后的写法。它不是事实正确率。BERTScore、BARTScore、MoverScore、QA/NLI与LLM评审各衡量不同侧面，不是互相保证。

### 2.9 用一个小例子把概念串起来（补充）

四句里若两句都谈cats，图中它们可能互相传递较高权重，TextRank会优先选相关代表句。它返回原句列表，生成摘要则可能改写和合并。选出原句仍可能丢限定条件，生成摘要还可能添未出现事实；所以都应对照来源做事实核验，不能只凭ROUGE高下结论。

## 三、操作前准备

TextRank只需标准库。生成与ROUGE按需安装：
~~~bash
python -m pip install torch transformers rouge-score
~~~
原文article是占位文本，reference_summary/generated_summary也未定义；下面补充最小值。不要把占位符送入模型后把输出当新闻摘要。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 从零实现TextRank

sentence_split是简易句末标点+空白切分，不能正确处理所有缩写和语言。边不含自身，迭代到变化小或达到轮数。孤立节点的质量没有按完整PageRank方式回流，是教学简化。最后ranked.sort恢复顺序，避免按分数随意重排原句。

~~~python
import math
import re
from collections import Counter


# 简易英文句边界，空文本和缩写需另外处理。
def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


# 使用词计数交集与独特词数的对数分母，不是余弦。
def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


# 迭代排序后恢复原文句序；孤立节点处理为教学简化。
def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
~~~

### 4.2 调用BART生成摘要

facebook/bart-large-cnn适配新闻域，不保证科学/医疗/对话域可靠。max_length/min_length通常按token数而非汉字/单词数；原代码不是自动输入分块或事实检查。do_sample=False不等于所有环境逐位可复现。

~~~python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
~~~

### 4.3 ROUGE评估

scorer.score顺序为reference、prediction，返回每种指标的precision/recall/fmeasure；示例只打印fmeasure。use_stemmer=True适合某些英语协议，原文“总应该stemming”过于绝对，不适合所有语言或基准。

~~~python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
~~~

### 4.4 超越ROUGE的评价

原文补充语义指标、生成式评分、事实分类器与LLM裁判。其2025/2026排行榜、一致率80%等不在本文复现。要把人工标注的事实支持、覆盖和表达质量分开验证。



### 4.5 事实性错误

实体、数量、极性和新增事实都应检查。仅摘要实体都在原文出现，仍可能把关系说反；仅源实体召回低，也不证明摘要不安全，摘要本来就会选取部分内容。NER指标是探针，不是事实判定器。



## 五、把代码运行起来：补充入口与检查

标准库补充入口：
~~~python
article = (
    "Cats sleep on mats. Cats like quiet rooms. "
    "Dogs sleep near doors. Cats and dogs need water."
)
print(textrank(article, top_k=2))
assert len(textrank(article, top_k=2)) == 2
assert all(s in sentence_split(article) for s in textrank(article, top_k=2))
~~~
先看输出是否确实来自原文，再讨论覆盖度。空字符串在原文sentence_split中可能得到['']，请在外围先处理，不把空句当有效摘要。

ROUGE接口的补充输入：
~~~python
reference_summary = "Cats and dogs need water."
generated_summary = "Cats and dogs need water."
~~~
在原课ROUGE块之前定义这两项。相同非空句子应得到接近1的F值，空二元组等边界需按库约定检查。神经摘要与ROUGE本次未安装运行。

## 六、Use It：使用现成工具

原课把新闻BART、论文Pegasus、对话BART、TextRank/LSA/LexRank与长上下文LLM列作候选。选择要看任务训练、语言和输入长度，不能只看到32k上下文就认为多文档事实归并已解决。超长文本可分块再汇总，但跨块重复、指代和信息丢失要单独检查。



## 七、失败模式、原文边界与工程注意

1. 抽取不是“永不幻觉”的同义词，可能断章取义、引用过时内容或打乱指代。
2. 参考摘要也可能有错；高ROUGE不保证忠实，低ROUGE不一定无用。
3. 将原文实体回收得多不代表摘要更好；摘要目标长度和覆盖要求要明确。
4. 原课30～45 ROUGE等只是参考，不能当5篇自选新闻的预期保证。
5. 生成模型和裁判都可能受同类偏差影响，校准需要独立人工样本。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
给文档类型、长度、读者和约束，选择抽取或生成候选；说明分块和输出预算；设计ROUGE协议及事实/覆盖/流畅度人工rubric；逐项核对实体、数字、关系和否定。未计算的分数不要估成实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：对5篇文章取top3，和参考算ROUGE-L；报告选择与源文本，不强求30～45。
2. Medium：用NER比较源/摘要实体P/R，再构造“同实体但关系相反”反例，说明该探针漏检什么。
3. Hard：同50篇文章比较BART与LLM，保持输出预算，报告ROUGE、人工事实性及实际费用。不要用实体F1替代完整事实评价。

## 十、术语与完成标准

extractive：选原句；abstractive：条件生成；TextRank：句子图传播排序；ROUGE-L：LCS重叠；factuality：摘要断言是否获源支持；coverage：关键信息覆盖。

完成检查：

- 能解释图节点、边与迭代更新。
- 能运行标准库摘要并检查原文包含关系。
- 能区分ROUGE、实体匹配和事实支持。
- 能给实体/数字/极性错误各举一例。

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

