# 12：文本摘要——TextRank、生成摘要与事实核验

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，概念细节与课程页一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/12-text-summarization) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/12-text-summarization/docs/en.md)。原题：Text Summarization。

## 零、初学者先读：摘要的两种做法——挑句子与重写句子

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。第11课把生成机器开进了翻译，本课让它做摘要，同时引入一条不需要训练的路线TextRank；第02课“数词重叠”的老朋友会在这里再次出场。

### 0.1 先理解这一课在做什么

一篇2000词的文章要压成120词（原课情境）。两条路线本质不同：

| | 抽取式（TextRank） | 生成式（BART） |
|---|---|---|
| 做什么 | 给原句打分，挑最重要的几句 | 读懂后用自己的话重写 |
| 像谁 | 拿荧光笔在原文划重点 | 读完复述给你听的人 |
| 输出 | 永远是原文原句，语法必然通顺 | 流畅、可压缩、可改写 |
| 风险 | 划出的句子脱离上下文可能误导 | 可能“顺嘴说错”——捏造事实 |

抽取式是个**排序问题**：句子当节点、词重叠相似度当边，跑一版PageRank（TextRank，Mihalcea & Tarau 2004）给句子排重要度。生成式是**条件生成问题**：第09课那台“读入→逐词写出”的机器，在文档-摘要对上训练（BART）。两条路线的失败模式各配一个：抽取式怕断章取义，生成式怕幻觉——本课用ROUGE量重叠，用事实核验量忠实。

### 0.2 本课术语先翻成白话

- 抽取式摘要 extractive=从原文挑原句
- 生成式摘要 abstractive=重新措辞生成
- TextRank=把句子建图、像PageRank一样迭代传播重要度的排序算法
- 节点／边 node／edge=句子／句子间的相似度
- 阻尼系数 damping=0.85，每轮有15%的分数不沿边走、直接给基础值
- ROUGE-1/2=一元／二元词重叠的指标
- ROUGE-L=最长公共子序列（LCS）版本的ROUGE，词序敏感
- 精确率／召回率／F值=对上了多少／找回来多少／两者的调和平均
- 事实性 factuality=摘要的说法是否被原文支持
- 实体替换 entity swap=John Smith写成John Brown
- 数字漂移 number drift=25,000写成25 million
- 极性翻转 polarity flip=“拒绝”写成“接受”

### 0.3 用小数字走一遍

三句话的TextRank一轮迭代。原文（小写列出的是各自的不同词）：

~~~text
S1: Cats sleep in the sun.       （5个不同词：cats, sleep, in, the, sun.）
S2: Dogs sleep in the rain.      （5个不同词：dogs, sleep, in, the, rain.）
S3: Old cats and old dogs nap.   （5个不同词：old, cats, and, dogs, nap.）
~~~

**第一步：算边（相似度）。** 用正文4.1的公式：交叠词数÷两句词数的对数和。三句都是5个不同词，分母同为log(5+1)+log(5+1)=2×log6≈3.584；S1∩S2共享{sleep, in, the}共3个，S1∩S3共享{cats}共1个，S2∩S3共享{dogs}共1个：

~~~text
        S1      S2      S3
S1     0      0.837   0.279
S2     0.837  0       0.279
S3     0.279  0.279   0
~~~

**第二步：行归一（每句的出边权重和为1）。** S1的两条边是3:1，归一后0.75/0.25；S2同理；S3两条边一样长，各0.5：

~~~text
S1 ─▶ S2: 0.75   S3: 0.25        S3 ─▶ S1: 0.50
S2 ─▶ S1: 0.75   S3: 0.25             S2: 0.50
~~~

**第三步：迭代一轮。** 每句初始1.0分，新分数 = 0.15 + 0.85×(收到的加权和)：

~~~text
S1新 = 0.15 + 0.85×(0.75×1 + 0.50×1) = 0.15 + 0.85×1.25 = 1.2125
S2新 = 0.15 + 0.85×(0.75×1 + 0.50×1) = 1.2125
S3新 = 0.15 + 0.85×(0.25×1 + 0.25×1) = 0.15 + 0.425  = 0.575
~~~

top-2选S1、S2，再按原文顺序排回去。要点：与越多句子共享词，收到的分数越多——S1、S2因为“同伙多”胜出，S3两边各只沾一个词、被甩开。多轮迭代只是把第三步重复到分数不再变化（本机Python已复核全部数字：相似度、归一、一轮更新均与正文4.1代码逐行对应）。

以上是用于理解的算例，不是新增的摘要或评估实测。

### 0.4 怎样连接到正文代码

4.1的三个函数对应0.3的三步：similarity是第一步的交叠/对数公式（注意它不是TF-IDF余弦）；textrank里total_out那行是第二步的行归一，new_scores那行是第三步的更新，ranked.sort()是“恢复原文顺序”；damping=0.85、迭代50轮是PageRank的默认。4.2的summarizer pipeline是生成式路线：facebook/bart-large-cnn只在新闻域可靠。4.3的rouge_scorer给摘要打分，参数顺序是(reference, prediction)。第五节的补充入口给了四句猫狗文章，可以直接跑textrank、对照0.3的机制看输出。

### 0.5 理解检查（含参考答案）

**问题1：** TextRank为什么能“不靠训练”就挑出重要句子？

**参考答案：** 它只依赖“词重叠多的句子互相关联”这个统计事实，用图上的分数传播放大它：与越多句子相连，收到的分数越多——重要度是互相投票投出来的，不需要任何参数学习。

**问题2：** “抽取式摘要永不幻觉”这句话哪里不对？

**参考答案：** 它逐字来自原文，不会捏造原文没有的词；但被选句子可能脱离上下文、断章取义或引用过时内容——不捏造不等于不误导。

**问题3：** ROUGE和BLEU（第11课）都在数n-gram重叠，方向有什么不同？

**参考答案：** BLEU偏精确率（候选里的n-gram有多少在参考中，另用BP防短）；ROUGE偏召回率（参考里的n-gram有多少被找回来）——摘要怕漏重点，翻译怕多废话。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

摘要可从原文挑句子，也可重新生成措辞。前者是排序，后者是条件生成。两者都可能遗漏重要信息，生成式还可能新增没有依据的事实；抽取式即使逐字复制，也会因脱离上下文而误导。

## 二、核心概念与推导

TextRank把句子当节点，词重叠相似度当边，迭代传播重要性；最后选top-k并恢复原文顺序。原代码并非TF-IDF余弦，它用Counter交集计数除以两句独特词数的对数和。

（补充 2026-11-06）**PageRank的“投票”直觉（对照0.3）。** 把每句看成一个网页：相似度是链接，分数是重要度。三个要点：

- 一轮更新=“每人把自己的分数按出边比例分给朋友，再留15%当底薪”——damping=0.85的15%保证孤立节点（谁都不连）也保住0.15的基础分，图不连通也不崩；
- 为什么要迭代：第一轮大家底分相同，投票只反映“连了多少人”；几轮之后朋友的分数本身变高，投出的票更值钱——重要度是逐步收敛出来的；
- 分母取对数是原始TextRank的设计：两句都短时（log值小），同样的交叠占比更高、相似度更大——短句之间的“共享3个词”比长句之间的“共享3个词”更像同伙。

（补充 2026-11-06）**一个必须知道的代码怪癖。** similarity里s.lower().split()不剥标点：句尾的词带着句号进Counter。0.3的例子里特意让所有共享词都出现在非句尾位置，才没被这个坑绊到：

- 若S1以sleep结尾（“sleep.”）而S2句中用sleep，两词不相等、相似度直接少一格；
- 这是教学简化代码的真实边界，工程实现要么先剥标点、要么用带标点处理的分词器（第01课）；
- 同理，孤立节点的0.15是简化处理，完整PageRank会把底薪按全局重新分配。

生成摘要如BART需文档—摘要任务训练，输出可压缩和改写，也会发生实体替换、数字漂移、否定翻转及事实捏造。

（补充 2026-11-06）**生成摘要就是第09课的机器换了个数据集。** encoder读文档、decoder逐词写摘要，cross-attention（第10课）每写一个词都能回看原文——所以它的失败模式也是生成式的：训练目标是“像人写的摘要”，不是“忠于原文”，流畅和忠实不是一回事。四类幻觉各有名字，正好构成一张检查清单：

| 幻觉类型 | 原文说 | 摘要说 |
|---|---|---|
| 实体替换 | John Smith | John Brown |
| 数字漂移 | 25,000 | 25 million |
| 极性翻转 | 拒绝了报价 | 接受了报价 |
| 事实捏造 | （没提CEO） | CEO批准了 |

抽取式“按构造”躲开前四类（词都来自原文），但躲不开“断章取义”——被选句子脱离上下文后含义可能反转。所以两条路线最后都要人工或NLI式的事实核验兜底，这一步没有自动指标能完全替代。

ROUGE-1/2看一元/二元重叠，ROUGE-L看最长公共子序列。rouge-score的fmeasure通常0～1，论文中40分常是乘100后的写法。它不是事实正确率。BERTScore、BARTScore、MoverScore、QA/NLI与LLM评审各衡量不同侧面，不是互相保证。

（补充 2026-11-06）**ROUGE怎么读。** 对一句话：参考摘要有10个词、生成摘要覆盖了其中7个→召回0.7（重点漏没漏）；生成了10个词、其中8个能在参考找到→精确率0.8（废话多不多）；fmeasure是两者的调和平均。与BLEU（第11课）对照着记：

- BLEU是精确率家族（候选里的n-gram对上了多少），ROUGE是召回率家族（参考里的内容找回来了多少）——因为摘要的痛点是漏，翻译的痛点是错译和短译；
- ROUGE-L换成“最长公共子序列”再算同样的三件套——子序列保序，词对但顺序乱的重叠要打折；
- “40 ROUGE-L是good、50是exceptional”（原课参考）只是新闻域的量级感，跨领域没有可比性；
- use_stemmer=True是英语协议的一部分，中文等语言无从谈起——第01课的词干化、第04课的子词在这里都派得上用场；
- 最后一句最重要：ROUGE量的是“和参考像不像”，不是“对不对”——一个忠实但换了说法的摘要可以比一个抄对了原文却漏掉关键的摘要分低。

（补充 2026-11-06）**什么时候选哪条路线。** 原课的倾向（思路参考，非硬规则）：

- 合规、医疗、法律等“错一句话就是事故”的场景：抽取式更安全——不捏造是构造性保证；
- 新闻、消费内容等重流畅与压缩的场景：生成式+事实核验门；
- 超长输入：先分块再汇总，但跨块重复、指代丢失要单独检查（第六节的提醒）；
- 两条路线都要对照原文核验，只是生成式的核验更不可省——这也是第八节模板要求“未计算的分数不要估成实测”的深层原因。

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

