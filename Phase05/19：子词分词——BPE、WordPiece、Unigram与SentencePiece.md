# 19：子词分词——BPE、WordPiece、Unigram与SentencePiece

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，四步结构与原课一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/19-subword-tokenization) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/19-subword-tokenization/docs/en.md)。原题：Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece。

## 零、初学者先读：四种切词法，其实在回答同一个问题

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课是[第04课](<04：GloVe、fastText与子词表示.md>)2.3节BPE的完整版：那一节只有BPE一把刀，本课把BPE、WordPiece、Unigram、SentencePiece摆在一起对照。切词规则学完就到头了——向量是[第03课](<03：词向量与从零实现 Word2Vec.md>)的事，不要在这里找。

### 0.1 先理解这一课在做什么

[第04课](<04：GloVe、fastText与子词表示.md>)讲过词表的两难：整词词表要么撑爆（百万词）要么大量OOV；字符级永不缺词但太碎、序列太长。子词（subword）是折中方案：常见词保持整词不切，罕见词拆成有意义的片段——`untokenizable`切成`un`+`token`+`izable`，三段都是训练时见过的老朋友。类比还是切菜：常吃的菜整颗下锅（高频词整词），一年买一次的才切开分装（低频词拆件）。

本课要先澄清的类别关系（初学者最容易在这里迷路）：**BPE、WordPiece、Unigram是三种“算法”，SentencePiece和tiktoken是两个“库”**。算法回答“规则怎么学”，库回答“谁替你干活”——SentencePiece能训练BPE或Unigram词表，tiktoken只拿现成编码表做编码、不训练。问“SentencePiece和Unigram哪个好”，和第04课“问BPE和Word2Vec哪个好”一样是类别错误——前者是工具箱，后者是工具箱里的一种算法。在流水线上它们都站在最上游（第04课的图原样适用）：

~~~text
文本 ──子词算法（BPE/WordPiece/Unigram，由SentencePiece等库实现）──▶ token ──embedding层──▶ 向量
~~~

三种算法先立一张总览（细节在0.3和正文展开；SentencePiece和tiktoken是库，不进表）：

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| 训练方向 | 自下而上：从字符开始**合并** | 自下而上：从字符开始**合并** | 自上而下：从大候选词表开始**删除** |
| 挑谁合并/删谁 | 当前最高频的相邻对 | 让语料似然提升最大的对（近似：频次除以部件频率） | 删掉后似然损失最小的片段 |
| 编码新词 | 按学习顺序套规则（确定） | 最长匹配（确定） | 取概率最大的切法，也可按概率**采样** |
| 代表模型 | GPT系列、Llama、Mistral | BERT、DistilBERT、ELECTRA | T5、mBART、Gemma |

### 0.2 本课术语先翻成白话

- 子词 subword=比词短、比字符长的单位，如token、izable
- 合并 merge=BPE/WordPiece把两个相邻单位绑成一个新单位；规则有顺序，编码要按学习顺序套
- `</w>`=词尾标记，让片段知道自己是“顶到词尾”还是“词中间”；WordPiece用`##`前缀表达同一件事（est表示词首或独立，##est表示接在前面的部件）
- byte-level BPE=不按字符、按256种字节当起点，任何UTF-8文本都切得开、永不UNK
- Unigram=给每个片段一个概率的“词表语言模型”，训练靠删除、编码靠比概率
- 似然 likelihood=语料在当前词表下被切出来的概率；裁剪时算“删谁掉得最少”
- subword regularization=Unigram的编码可按概率采样出不同切法，用于数据增强
- SentencePiece=训练BPE/Unigram词表的库，直接吃原始文本、用▁表示空格
- tiktoken=OpenAI的编码库，只编码不训练，对应cl100k_base、o200k_base等预置词表
- character coverage=训练时要求覆盖的字符比例（如0.9995），不是准确率
- tokenizer drift=训练和部署用了不同词表，token编号对不上，模型输出乱码

### 0.3 用小数字走一遍

**A. BPE的前三步（数对→合并）。** 语料用经典教学语料{low×5, lower×2, newest×6, widest×3}，与[第04课](<04：GloVe、fastText与子词表示.md>)2.3完全同一份——那里已有逐步合并的完整过程（数对表、三步合并、生词lowest=low+est</w>的切分演示），这里不重抄，只对齐关键数字：第一轮数对表前几名是(e,s)=9、(s,t)=9、(t,</w>)=9（三者并列，Counter插入顺序决定选谁）、(w,e)=8、(l,o)=7；于是三步合并依次是(e,s)→es、(es,t)→est、(est,</w>)→est</w>，每步都把当前序列重新数一遍、被合并对的新频次又都是9。看懂“每轮重新数、按词频加权、规则有序”这三件事，BPE就懂了。

注意：正文2.9与第五节用的是另一份语料{lower×5, lowest×2, newest×6}，那里(w,e)=13——两份语料不同，别把两套数字混着背。

**B. Unigram的“词表概率”直觉。** 设六片段的Unigram词表（概率和为1）：new=0.4、est=0.1、ne=0.1、west=0.2、low=0.1、er=0.1。词newest有两种切法：

~~~text
切法A：new + est → 0.4 × 0.1 = 0.04   ← 概率大，编码默认选它
切法B：ne  + west → 0.1 × 0.2 = 0.02
~~~

三种用法一次看清：（1）编码时取概率最大的切法；（2）采样时按归一化概率抽——P(A|newest)=0.04÷(0.04+0.02)=2/3，这就是subword regularization做数据增强的原理；（3）训练裁剪时看“删谁损失最小”：语料为newest×6+new×4，对数似然=6×ln(0.04)+4×ln(0.4)≈−22.98；若删掉片段est，newest只能走切法B，对数似然掉到6×ln(0.02)+4×ln(0.4)≈−27.14，损失6×ln(2)≈4.16。对每个片段都算一遍这样的“删除损失”，删掉最不心疼的、重复到词表缩到达标——这就是Unigram的裁剪循环，方向与BPE的合并正好相反。

**C. WordPiece挑合并的标准不同。** BPE第一轮选(e,s)是因为绝对频次9最高；WordPiece看的不是绝对频次，而是“这对一起出现比随机碰上超出多少”，近似实现是：对频次÷(两部件各自频次的乘积)。同一份语料的单符号频次：e=17、s=9、i=3、d=3，于是(i,d)得分3/(3×3)≈0.333，远超(e,s)的9/(17×9)≈0.059——**BPE先合并(e,s)，WordPiece先合并(i,d)**：绝对频次只有3的(i,d)因为部件本身罕见而“惊喜度”最高，高频部件在WordPiece里被除法惩罚。

以上是用于理解的算例，不是新增的训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

四步互相独立，按顺序读：4.1 train_bpe把“词→频数”字典滚成有序合并表（就是0.3 A的循环；注意输入不是句子列表——这是与第03课的关键差别）；4.2 encode_bpe按学习顺序给新词套规则（0.3 A的“规则有序”落到代码）；apply_merge在第五节补齐。4.3 SentencePiece训练把0.3 B的概率词表变成现实（model_type选bpe或unigram）；4.4 tiktoken只编码不训练。第五节有完整入口：corpus={lower:5, lowest:2, newest:6}，断言merges[0]==("w","e")——频次13正是0.3末尾提醒的那份语料。

### 0.5 理解检查（含参考答案）

**问题1：** BPE和Unigram的训练方向差别是什么？

**参考答案：** BPE自下而上，从字符开始每轮合并当前最高频对、攒出越来越大的单位；Unigram自上而下，从超采的大候选词表开始，每轮删掉“删除后似然损失最小”的片段。BPE的词表是攒出来的，Unigram的词表是剪出来的。

**问题2：** “SentencePiece用的是Unigram算法”这句话对吗？

**参考答案：** 不完整。SentencePiece是库，训练时用model_type指定bpe或unigram；它本身的贡献是直接吃原始文本（空格转▁），不依赖预分词。

**问题3：** 为什么预训练模型不能因为“新分词器切得更短”就换掉tokenizer？

**参考答案：** 每个token编号绑定embedding矩阵的一行和输出层的一列；换词表等于换了坐标系，旧权重的行列对不上新编号，输出乱码（tokenizer drift）。扩词表要连带embedding扩展、初始化与继续训练。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

整词词表遇到新词会丢信息，逐字符又让序列太长。子词用高频长片段与低频短片段折中。本课与第04课衔接，重点是完整BPE训练/编码顺序与实际分词工具。

## 二、核心概念与推导

BPE从字符或字节开始，按语料频率累计相邻token对，每轮合并当前最高频对；规则有顺序，编码要遵循学习到的优先级。权重语料lower×5不是只计一次。

（补充 2026-11-06）“规则有顺序”值得再展开一步：合并(es,t)能发生，前提是(e,s)已经把es造出来了——每条规则消耗前面规则的产物，像流水线上每道工序只能加工前一道工序的半成品：

~~~text
字符起点：  e   s
规则1：(e,s)→es             原料：e、s
规则2：(es,t)→est           原料：es（规则1的产物）、t
规则3：(est,</w>)→est</w>   原料：est（规则2的产物）、</w>
~~~

所以encode_bpe必须按学习顺序遍历merges，乱序套规则会漏合并；第五节`assert merges[0] == ("w", "e")`查的正是这条依赖链的第一环。“每轮重新统计”也别漏看：合并(e,s)之后，原本不相邻的(es,t)变成了新的相邻对、频次从无到有变成9——统计对象永远是“当前token序列”，不是原始字符流。

（补充 2026-11-06）`</w>`的存在让est（词中间）和est</w>（顶到词尾）成为两个不同token：widest切出est</w>，estuary只会切出est；不带词尾标记的教材版本会把这两类混为一谈，这正是本课2.9与某些教材结果不同的原因。至于并列时选谁（第一轮的9/9/9），由实现决定——第五节Counter按插入顺序取第一个，不同教材可能不同，这不是谁算错了。

（补充 2026-11-06）字符BPE与字节BPE的差别一句话记住：起跑线不同。字符版以“见过的字符”为基础字表，遇到基础字表外的字符（生僻字、emoji）仍可能UNK；字节版以256种字节为基础，任何UTF-8文本最终都是字节序列、永不UNK——代价是非拉丁文字可能被切成多个token（英文为主的词表里，一个汉字常见要2～3个token），这正是第七节“ASCII或英文平均token不能估算中文场景”的机制根源。两种起点对照：

| | 字符级起点 | 字节级起点 |
|---|---|---|
| 基础单位 | 语料里出现过的字符 | 256种字节 |
| 覆盖范围 | 基础字表外→UNK | 任何UTF-8文本 |
| 代表 | 早期教学实现 | GPT-2、Llama |

GPT-2的50257=256基础字节+50000次合并+1个特殊token（原课参考；正文稍后提醒这个加法只是粗略归属，具体以checkpoint为准）。

（补充 2026-11-06）WordPiece的编码端常用贪心最长匹配：从左到右，每步从词表挑能覆盖当前位置的最长片段（词首用est、非词首用##est），一处匹配不上整个词就标UNK——这是BERT家族“见生词就弃疗”的出处。训练端与BPE的差别已在0.3 C用数字走过：BPE挑绝对频次最高的对，WordPiece挑频次÷(部件频次乘积)最高的对，前者偏爱“到处都有的搭配”，后者偏爱“罕见部件之间的强绑定”。一句话对照：BPE看热度，WordPiece看惊喜度，Unigram看概率账本。

字符BPE不等于字节BPE。字节版本以256种字节为基础可表示有效文本的UTF-8字节；字符版本若基础字表不含新字符仍可能有UNK。Unigram从大候选词表开始，按似然影响逐步裁剪；可采样切分用于数据增强。WordPiece与BPE的训练准则不同，BERT类常见其最长匹配编码。

（补充 2026-11-06）Unigram的“似然影响”用0.3 B的数字再走一遍就具体了。裁剪器给每个候选片段记一笔“删除损失”账：

| 候选动作 | 后果 | 删除损失（对数似然掉多少） |
|---|---|---|
| 删est | newest被迫改走切法B，6次×ln(0.04/0.02) | ≈4.16 |
| 删er | 语料里没有任何词形依赖er | 0 |

于是先删er——这就是“删最不心疼的”。这个循环配上“字符永远保留在候选表里”的兜底，保证任何词最后都切得开。三个算法的差别可以浓缩成一句：**BPE和WordPiece在攒单位（一个按绝对频次、一个按相对惊喜度），Unigram在剪单位（按概率账本）**；SentencePiece和tiktoken只是实现它们的库。选型时先问“要不要训练新词表”，再看语料是否多语言（多语言常配Unigram加character_coverage=0.9995）、是否要对齐GPT系计数（tiktoken），第六节的表就是这个决策的展开。

SentencePiece是可训练BPE/Unigram的库，常用▁表示空格；不是某一种合并算法。tiktoken主要使用既定编码表执行编码/解码，不用于训练新词表。模型必须配套原tokenizer，不能因为新分词更短就直接替换。

（补充 2026-11-06）为什么SentencePiece坚持把空格编码成▁？多数英文工具链先做预分词（按空格切开、词内再子词化），空格信息就此丢掉——"New York"和"NewYork"会切出相同的token，decode回去也无法区分。SentencePiece把空格当普通字符一起进词表，▁un解码时换回空格，round-trip不需要额外规则。这也是它“语言无关、直接吃原始文本”的卖点：中文这类没有空格的语言同样适用，不用先跑一个分词器（0.3 A的英文例子有</w>标记词尾，SentencePiece用▁标记词首，是两种对称的设计选择）。

（补充 2026-11-06）tiktoken的定位再强调一遍：它是“既定编码表的执行器”，对应4.4的get_encoding——想学新词表时它帮不上忙。两个库还有个容易混的细节：SentencePiece的▁是词表里的普通符号，tiktoken的BPE合并直接发生在字节流上、空格也在其中——同一句话在两边的token编号完全是两套，互换没有任何意义。把这一条和“模型必须配套原tokenizer”合起来读，第七节第1条“词表/normalizer/特殊token变化导致训练部署不一致、版本和hash应一起保存”就不是抽象告诫，而是这个事实的工程推论。

原文对不同模型算法的归属、GPT-2词表加法和“所有模型”的表述过于粗略：GPT-2常见50257总词表并不等于256+50000+1。不要据本文固定名称推断所有系列/版本，查具体checkpoint配置。

### 2.9 用一个小例子把概念串起来（补充）

lower×5、lowest×2、newest×6中(w,e)加权计数13，首次合成we。此后原e,s已经不再相邻；应重新统计(we,s)与(s,t)，都为8。本篇4轮还会考虑词尾</w>，因此结果可为we、wes、west、west</w>；词尾标记参与统计，是与没有该标记教材结果不同的原因。

## 三、操作前准备

标准库BPE先运行；外部库可按需安装：
~~~bash
python -m pip install sentencepiece tiktoken tokenizers
~~~
SentencePiece训练需要你自行准备corpus.txt。小语料不一定支持8000或32000词表，降低词表或增大真实语料，不能直接照抄数字。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 按加权频次学习合并

原文train_bpe输入是word→count字典，依赖Counter与apply_merge，下面补齐。注意仓库code/main.py接受原始text并返回(merges,tokens)，与文档接口不同；本篇按文档版本学习，不能混用。

~~~python
# 按词频加权计数，每轮在当前token序列上重新统计。
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
~~~

### 4.2 按规则顺序编码新词

先拆字符加词尾标记，再依次遍历merges。每次只合并相邻且完整匹配的token，不在已合并token内部再次找字符。原文O(n×merge数)仅是粗略描述，反复切片也有额外代价。

~~~python
# 规则按训练顺序应用，不能在已合并token内部再按字符切开。
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
~~~

### 4.3 训练SentencePiece

设置输入、前缀、词表和model_type。normalization会改变文本，decode(encode(text))未必与原始字符串字节完全相同，应与正规化协议比较。character_coverage是字符频率覆盖设置，不是正确率。

~~~python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
~~~

### 4.4 使用预置编码表

get_encoding按名称取得编码表；如需特定API模型应核查模型映射，不能把o200k_base泛指所有GPT模型。下面数字是源文示例，应在本地打印核对；token数也不是byte数。

~~~python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
~~~

## 五、把代码运行起来：补充入口与检查

补充缺失合并函数，与原文train_bpe接口一致：
~~~python
from collections import Counter

def apply_merge(vocab, pair):
    result = Counter()
    for symbols, count in vocab.items():
        output, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                output.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                output.append(symbols[i])
                i += 1
        result[tuple(output)] += count
    return dict(result)

corpus = {"lower": 5, "lowest": 2, "newest": 6}
merges = train_bpe(corpus, num_merges=4)
assert merges[0] == ("w", "e")
print(merges)
print(encode_bpe("lowest", merges))
~~~
并列对频次相同，Counter插入顺序决定本例选谁；因此后续规则不必与另一教材完全一致。对新词也要检查字符覆盖。

## 六、Use It：使用现成工具

训练新词表考虑SentencePiece/HF Tokenizers；使用现有模型先加载其配套tokenizer；tiktoken用于支持的预置编码。领域扩词还涉及embedding矩阵扩展、初始化与训练，并非把两张词表直接拼接即可。词表大小与语言、语料、参数量、输出层成本共同相关，不存在低于16k一律不合格的规则。



## 七、失败模式、原文边界与工程注意

1. 词表/normalizer/特殊token变化都可能导致训练部署不一致，版本和hash应一起保存。
2. 空格前缀、add_special_tokens、emoji、Unicode正规化要单测。
3. byte fallback避免不可表示，不保证切分高效或语义理解。
4. 教学代码对整个合并表重复扫描，并反复创建切片；适合理解过程，不用于大规模低延迟服务。
5. 原文“ASCII或英文平均token”不能估算所有中文场景。
6. 32k词表小语料训练失败是数据/参数不匹配，不应拿空白数据补数。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
明确训练新模型还是使用既有checkpoint；列算法、库、词表大小、特殊token、正规化、覆盖与byte fallback。检查held-out token长度/OOV、空格与emoji、正规化后的round-trip和文件hash；不无依据更换现有模型词表。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：用原仓库小语料训练最多500轮合并，对3个留出词编码，统计单token与多token；语料耗尽可能提前停止。
2. Medium：100句英文比较cl100k_base、o200k_base与自训32k SentencePiece；先确保训练语料足够，明确压缩比单位如字符/token。
3. Hard：同语料训练BPE、Unigram、WordPiece，在小型情感任务比较F1；固定划分、模型预算、随机种子，不用单次差值断言算法优劣。

## 十、术语与完成标准

merge list：有序合并规则；byte-level：字节级；Unigram：概率子词模型；WordPiece：另一子词方案；SentencePiece：训练库；character coverage：字符频率覆盖。

完成检查：

- 能手算第一轮(w,e)频次13。
- 能运行补齐的字符BPE。
- 能区分库和算法。
- 能说明不能直接替换预训练模型tokenizer。

## 十一、验证范围与来源许可

原课 Build It 共 4 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

