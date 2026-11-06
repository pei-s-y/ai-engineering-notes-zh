# 16：Transformer之前的文本生成——N-gram、平滑与困惑度

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充（平滑阶梯、KN的continuation直觉、困惑度单位）；当日与原课网页版核对。零节手算例（bigram计数、加一平滑、困惑度）已用本机Python标准库复核。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer/docs/en.md)。原题：Text Generation Before Transformers — N-gram Language Models。

## 零、初学者先读：数数也能预测下一个词——语言模型的史前史

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课处在[第19课](<19：子词分词——BPE、WordPiece、Unigram与SentencePiece.md>)（分词，决定“词”是什么）与[第10课](<10：注意力机制——Bahdanau、Luong与QKV.md>)（注意力，让模型看到长上下文）之间的史前史：注意力机制登场之前，语言模型就是一张查表数频次的计数表。但“困惑度”这个词活到了今天——评估 LLM 仍天天说它，源头就在本课。

### 0.1 先理解这一课在做什么

“语言模型”听起来高深，最古老的版本只做一件事：**查表数频次**。见过“the cat”后面跟"sat"47次、"jumped"12次、"refrigerator"0次，就把计数归一化成概率：P(sat|the cat)=47/59。1980到2015年间的语音识别、拼写纠错、机器翻译，底下全是这张表。类比老式手机输入法：打出“今天”就弹出“天气”，因为语料里这么接的最多。

查表的死穴是**没见过的组合**：句子一长，几乎每句话都含至少一个训练时没见过的 n-gram，而计数模型给它们零概率——整句概率连乘直接归零。五十年的平滑（smoothing）研究就是给零概率擦屁股，终点是 Kneser-Ney。本课把这条线从最简单的“加一”走到 KN，最后用**困惑度**（perplexity，PPL）给语言模型打分——这个指标今天仍是评估 LLM 的标准词汇，理解它的最短路径就是本课的手算。

还有一个实用理由：原课 Use It 的建议——宣布神经模型好之前，先算一个 n-gram 基线 PPL；神经模型不大幅超过 KN，说明哪里出了问题。

### 0.2 本课术语先翻成白话

- 语言模型（language model, LM）=给一段历史，给下一个词打概率分布的东西
- N-gram=连续n个token的序列；bigram二元（看1个前词）、trigram三元（看2个前词）
- 马尔科夫假设=只看最近n−1个词、更早的历史全部扔掉
- BOS/EOS（`<s>`/`</s>`）=句首/句尾哨兵，防止把上句结尾当下句开头
- MLE（最大似然估计）=直接用计数除以总数，没见过的就是0
- 零概率问题=未见n-gram概率为0，连乘一句就全盘归零
- 平滑（smoothing）=从见过的事件匀一点概率给没见过的事件
- Laplace加一（add-one）=所有计数加一再归一化，最简单也最粗糙
- Kneser-Ney（KN）=n-gram时代的最强平滑：绝对折扣＋continuation
- continuation probability=按“出现在多少种不同前文里”而非词频算的低阶分布
- 回退（backoff）/插值（interpolation）=计数为零降级用短上下文／多阶加权混合
- 交叉熵（cross-entropy）=平均每个token的负对数概率
- 困惑度（perplexity, PPL）=交叉熵取指数；“等效均匀分支数”，越低越好
- 采样（sampling）=按概率分布随机抽下一个词；贪心=永远挑最大概率

### 0.3 用小数字走一遍

**第一步：数bigram。** 微型中文语料三句（已分好词）：

~~~text
今天 天气 很好
今天 天气 不错
明天 天气 如何
~~~

加边界后数相邻对：`(<s>,今天)`×2、`(今天,天气)`×2、`(天气,很好)`、`(很好,</s>)`、`(天气,不错)`、`(不错,</s>)`、`(<s>,明天)`、`(明天,天气)`、`(天气,如何)`、`(如何,</s>)`。上下文计数：`<s>`3次、`今天`2次、`天气`3次。

**第二步：不加平滑的尴尬。** 问“今天天气___”：bigram 只看最后一个词“天气”，其后3次（很好/不错/如何各1）。原始概率 P(很好|天气)=1/3、P(不错|天气)=1/3、P(下雨|天气)=**0**——“今天天气下雨”整句概率归零，负对数损失无穷大，PPL 根本算不出来。这就是必须平滑的原因。

**第三步：加一平滑。** 词表 V={今天,天气,很好,不错,明天,如何,`</s>`}共7个（含EOS不含BOS，与正文源码约定一致）。每个计数加一、分母加V：

~~~text
P(很好|天气) = (1+1)/(3+7) = 2/10 = 0.2   ← 原来是 1/3≈0.33，被压低
P(下雨|天气) = (0+1)/(3+7) = 1/10 = 0.1   ← 原来是 0，被救活
全部7个候选概率和 = 0.2×3 + 0.1×4 = 1.0   ← 仍然归一化
~~~

代价立刻可见：更狠的是“今天”后面，语料里2次全是“天气”，未平滑 P(天气|今天)=1.0 是必然事件；加一后变成 (2+1)/(2+7)=3/9≈0.33——**十拿九稳的事被打成三成把握**，匀出去的份额全给了从没见过的6个词。这就是“Laplace挪走常见词概率”的手算版。

**第四步：困惑度。** 用加一平滑给句子“今天 天气 很好”打分（含边界共4步转移）：

~~~text
P(今天|<s>)   = (2+1)/(3+7) = 0.30
P(天气|今天)  = (2+1)/(2+7) = 0.333
P(很好|天气)  = (1+1)/(3+7) = 0.20
P(</s>|很好)  = (1+1)/(1+7) = 0.25
PPL = exp( −(ln0.30+ln0.333+ln0.20+ln0.25)/4 ) = exp(1.325) ≈ 3.76
~~~

PPL≈3.76 的读法：这个模型每走一步，等效于在约3.8个词里均匀犹豫。PPL=1 是先知（每步都确定），均匀瞎猜7个词的模型 PPL=7；真实大模型在大赛料上把这个词降到个位数。注意这是**训练集自测**，只能验证“概率表连不连通”，不能当泛化能力。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

五个 Build It 函数与0.3一一对应：train_ngram 就是第一步的数数（自动加 `<s>`/`</s>`）；raw_probability 是第二步的除法（未见上下文返回0）；laplace_probability 是第三步——分母 contexts+vocab_size 正是0.3的 3+7，V 怎么选（含不含EOS）直接改变结果；kneser_ney_bigram_model 是平滑的完全体，三块零件：continuation_prob（0.2的升级版，按“多少种前文”分配）、固定折扣 discount、腾出的质量按 lambda_prev 重新分配；perplexity 是第四步的代码化——逐句加边界、累加自然对数、取指数，与0.3手算完全同型。generate 则是在这张概率表上按分布抽词。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么计数语言模型必须平滑，不能用原始计数直接评估？

**参考答案：** 测试集里几乎必有训练时没见过的 n-gram，原始概率为0；整句概率连乘归零，负对数损失无穷大，PPL 无法计算。平滑把见过的事件匀一点概率给没见过的，保证处处非零。

**问题2：** Kneser-Ney 的 continuation probability 和普通词频差在哪？用“旧金山”说明。

**参考答案：** 词频只看出现次数：“Francisco”出现25次全是跟在“San”后面，词频很高；continuation 看出现在多少种不同前文里：只有1种，continuation 很低。于是新上下文后接“Francisco”的概率被正确压低——高频≠百搭。0.3里Laplace的教训同理：只看次数会错配信任。

**问题3：** PPL=3.76 是什么意思？训练PPL和测试PPL能混用吗？

**参考答案：** 模型每预测一个词，等效于在约3.76个均匀候选里犹豫，是“平均分支数”的度量。不能混用：训练PPL只检验概率表连通性（背书能力），测试PPL才衡量对没见过文本的泛化，后者一定更高。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

不用神经网络也能预测下一个词。本课从计数语言模型理解条件概率、零概率问题和文本采样，为后续神经语言模型建立可解释基线。

## 二、核心概念与推导

N-gram近似P(w_t|全部历史)≈P(w_t|前n−1个词)。bigram看一个前词，trigram看两个。BOS/EOS标记句子边界，训练和评估必须一致。

（补充 2026-11-06）“扔掉更早的历史”是**马尔科夫假设**：预测下一个词只看最近n−1个词，之前的一概不管。这是被迫的精打细算——上下文每加一个词，要数的组合就多一个数量级，小语料根本喂不饱（这就是第二节要讲的稀疏问题）；但代价也很直白：“The bank of the river”里 bank 的意思由很远的 river 决定，bigram 看不见。Transformer 的注意力机制（[第10课](<10：注意力机制——Bahdanau、Luong与QKV.md>)）后来的突破，本质就是打破这条截断线、让每个词直接看全部历史——但预测游戏本身（给历史、猜下一个词）从没变过。BOS/EOS 的作用用0.3的语料看最清楚：不加边界，“很好”后面会偷偷接上下一句的“明天”，数出根本不存在的转移。

Laplace加一平滑：(count(context,w)+1)/(count(context)+V)。它给未见词留概率，但会挪走很多常见词的概率。Good-Turing、插值、回退、Katz、绝对折扣都是处理稀疏统计的路线。

（补充 2026-11-06）平滑方法的阶梯，一行为一个“更聪明的挪法”（0.3 的手算是第一级）：

| 方法 | 一句话白话 | 相对上一级的进步 |
|---|---|---|
| Laplace加一 | 每个计数加一 | 有下限，但一刀切太狠 |
| Good-Turing | 按“频次的频次”整体重分配 | 匀多少由统计决定，不是拍脑袋 |
| 插值 | 高阶低阶加权混合 | 高阶计数少时自动信低阶 |
| 回退（Katz） | 高阶计数为零才降级 | 高阶证据充分时不被稀释 |
| 绝对折扣 | 所有非零计数减固定D | 比“加一”温和得多 |
| Kneser-Ney | 折扣＋continuation低阶 | 低阶看“百搭程度”不看词频 |

（补充 2026-11-06）Laplace 的代价在0.3已经手算过：必然事件 P(天气|今天) 从1.0跌到1/3。根子是它**把匀出去的份额按候选词数平分**——V=7 时每个未见词各拿1/10，哪怕语料里它们从未露面。词表越大（真实任务几十万词），分母被 V 撑得越大，常见词被压得越惨，所以原课说它“simple, terrible on rare events”。

Kneser-Ney的核心是低阶分布采用“出现在多少种不同左上下文中”，而非单纯词频。例如固定搭配中高频的词，不一定适合任何上下文。源码仅演示固定折扣bigram，不是完整Modified Kneser-Ney实现。

（补充 2026-11-06）把“旧金山”例子数化：设语料里 Francisco 出现25次、**全部**跟在 San 后面；city 出现20次、前面跟着 big/small/beautiful/quiet……共20种不同的词。新上下文 "a lovely ___" 要接词时：按词频回退会选 Francisco（25>20）；按 continuation 会选 city（20种前文>1种前文）。城市赢——因为“跟着什么都行”才配得上未知的新上下文。源码里 continuation_prob 的分母 total_unique_bigrams（不同bigram对的总数）正是“多少种前文”的计数，lambda_prev 是每个上下文折扣腾出的总质量，两者相乘把匀出的概率按“百搭程度”分配。

平均负对数概率是交叉熵：使用自然对数时困惑度PPL=exp(loss)，使用log2时PPL=2^bits。Shannon的字母猜测实验是历史背景；字符与词是不同预测单位，不能仅换对数底就直接比较。原文历史压缩/困惑度数值不代表本次结果。

（补充 2026-11-06）PPL 的两种读法，都从0.3的 exp(1.325)≈3.76 出发：**等效分支数**——每步像在3.76个等概率候选里掷骰子；**几何平均概率的倒数**——模型给每个真实词的平均概率约 1/3.76≈0.27。两个读法同一件事，前者好讲，后者好算。今天评估 LLM 报的 PPL 就是这个数（词级、指定分词器下），三个不可比陷阱要背下来：**分词器不同不可比**（token 单位不同）、**对数底不同不可比**（nats 与 bits 只差常数但别混着报）、**语料不同不可比**。原课 Use It 的规矩：比较两个系统，必须同切词、同大小写、同边界、同测试集。

（补充 2026-11-06）Shannon 1951 年的猜字母实验（历史背景）：把英文句子的下一个字母盖住，让人逐个猜直到猜中，记录猜测次数。有趣的是猜测次数序列本身就是文本的无损编码——第二个猜家能按同样顺序还原每个字母。实验测出英文每字母约0.6~1.3比特（27个符号的均匀上限是 log2(27)≈4.75比特），说明四分之三的结构是“强制步”。每个语言模型都是这场猜谜游戏的机械选手：交叉熵是它的得分，困惑度是得分的另一种写法，上下文长度是选手的记忆——trigram 只有两词记忆，Transformer（[第10课](<10：注意力机制——Bahdanau、Luong与QKV.md>)）带着几万词的记忆玩同一个游戏。原课参考：调好的4-gram KN 在 Brown 语料上 PPL 约140，同测试集 Transformer 约15~30，约10倍差距就是领域转向神经的原因。

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

