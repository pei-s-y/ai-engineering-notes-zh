# 16：Transformer之前的文本生成——N-gram、平滑与困惑度

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer/docs/en.md)。原题：Text Generation Before Transformers — N-gram Language Models。

## 一、这一课解决什么问题？

不用神经网络也能预测下一个词。本课从计数语言模型理解条件概率、零概率问题和文本采样，为后续神经语言模型建立可解释基线。

## 二、核心概念与推导

N-gram近似P(w_t|全部历史)≈P(w_t|前n−1个词)。bigram看一个前词，trigram看两个。BOS/EOS标记句子边界，训练和评估必须一致。

Laplace加一平滑：(count(context,w)+1)/(count(context)+V)。它给未见词留概率，但会挪走很多常见词的概率。Good-Turing、插值、回退、Katz、绝对折扣都是处理稀疏统计的路线。

Kneser-Ney的核心是低阶分布采用“出现在多少种不同左上下文中”，而非单纯词频。例如固定搭配中高频的词，不一定适合任何上下文。源码仅演示固定折扣bigram，不是完整Modified Kneser-Ney实现。

平均负对数概率是交叉熵：使用自然对数时困惑度PPL=exp(loss)，使用log2时PPL=2^bits。Shannon的字母猜测实验是历史背景；字符与词是不同预测单位，不能仅换对数底就直接比较。原文历史压缩/困惑度数值不代表本次结果。

### 2.9 用一个小例子把概念串起来（补充）

若上下文like后cats出现2次、dogs出现1次，不平滑的概率为2/3和1/3。Laplace在预测词表大小V时分母变3+V；KN则先折扣已有计数，再按词拥有的不同前驱分配剩余质量。文本生成是在这个分布上采样，所以小语料也可能生成语法很奇怪的句子；概率平滑不是语法规则。

## 三、操作前准备

本课Build It仅需标准库。准备“每个句子是一串token”的列表，而不是把所有句子连成一个字符串。固定训练/测试划分、词表和UNK映射，避免测试词偷偷进入训练统计。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 统计N-gram与上下文

train_ngram为每句添加边界并累计计数。raw_probability用联合计数/上下文计数，未见上下文返回0；若测试中出现0，真实负对数损失为无穷。

~~~python
from collections import Counter, defaultdict


# 按句添加边界，避免把句尾与下一句开头当正常转移。
def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
~~~

### 4.2 用Laplace平滑

V必须与预测词表一致，例如是否包含EOS、是否排除BOS。给词表之外的任意字符串也使用同一公式，会破坏归一化语义，因此需要先映射UNK。

~~~python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
~~~

### 4.3 构造bigram Kneser-Ney

continuation看每个词拥有多少种前驱；折扣腾出的质量按continuation重新分配。源码对缺失词给1e-9，是演示兜底，不是严格的完整OOV处理。

~~~python
# 教学版固定折扣bigram模型，不是完整Modified KN。
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
~~~

### 4.4 按概率采样文本

generate要求非空prefix，prob_fn接受前一个词与候选词，所以原接口是bigram，不能直接接trigram。候选词表应包含EOS而不是BOS；max_len限制新增token数，随机种子只保证相同条件可复现。

~~~python
import random


# 按候选概率采样，需非空prefix与正的概率总和。
def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
~~~

### 4.5 计算困惑度

源码逐句加入BOS/EOS，以bigram概率累加自然对数。max(p,1e-12)是数值截断，遇到真实0概率时得到的是截断后的指标，不能冒充严格PPL；空测试集也需先检查。

~~~python
import math


# 用自然对数累加损失，再取指数；注意零概率截断。
def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
~~~

## 五、把代码运行起来：补充入口与检查

补充标准库入口，按上文定义完成后运行：
~~~python
sentences = [["i", "like", "cats"], ["i", "like", "dogs"], ["you", "like", "cats"]]
prob = kneser_ney_bigram_model(sentences)
vocab = sorted({w for sent in sentences for w in sent} | {"</s>"})
# 源码EOS使用字符串</s>，与这里词表保持一致。
print("词表：", vocab)
print("like后的概率和：", sum(prob("like", w) for w in vocab))
print(generate(prob, vocab, prefix=["i"], max_len=10, seed=0))
print(perplexity(prob, sentences))
~~~
上面prob是训练函数返回的bigram概率闭包。概率和应接近1的前提是覆盖了全部训练后继词。训练集PPL只作连通性检查，不能当泛化评估。没有承诺固定生成句子或性能提升幅度。

## 六、Use It：使用现成工具

原文推荐KenLM和nltk.lm实现更成熟的N-gram基线。KenLM适合高效查询/模型存储；nltk.lm便于教学。保持相同切词、大小写、边界、UNK和测试数据后再比较，避免拿词级PPL与字符级PPL或不同语料分数混比。



## 七、失败模式、原文边界与工程注意

1. generate中若全部候选概率为0，原代码可能误选首词，应显式报错或回退，不能视为正常采样。
2. 空prefix会访问tokens[-1]报错；空评估集会除零。
3. 固定seed不表示换seed必然得到不同文本。
4. 原文生成入口仅看最后一个词；要做trigram必须传递完整上下文并相应改写函数。
5. 温度缩放与argmax、beam search不是同一种解码。
6. 在测试集上构词表/调平滑参数会泄漏信息。
7. 底数转换只改变熵单位，不解决不同tokenization的可比性。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
建立计数语言模型基线，说明阶数、边界、分词、UNK、平滑方法和折扣；先验证各已见上下文概率和，再在固定测试集测PPL与采样质量。报告截断零概率的处理，不预先保证比另一方法降低多少。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：收集1000句Shakespeare文本，训练trigram并生成20段。需要补充支持两词上下文的生成函数，不能直接调用原bigram入口冒充trigram。
2. Medium：在相同bigram阶数与词表下比较Laplace和Kneser-Ney的held-out PPL，报告实测，不把原文30%～50%改善当保证。
3. Hard：实现编辑距离候选拼写纠错器，用语言模型给候选上下文打分；在Birkbeck等公开拼写错误数据上评价，区分候选召回与排序准确率。

## 十、术语与完成标准

N-gram：局部计数语言模型；backoff：回退低阶；interpolation：多阶插值；continuation probability：上下文多样性概率；cross-entropy：平均负对数概率；perplexity：困惑度。

完成检查：

- 能手算bigram计数与加一平滑。
- 能解释KN不是按词频回退。
- 能区分训练PPL和测试PPL。
- 能指出原生成器的上下文阶数。

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

