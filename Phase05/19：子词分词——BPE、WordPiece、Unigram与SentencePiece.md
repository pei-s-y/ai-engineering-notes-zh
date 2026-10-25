# 19：子词分词——BPE、WordPiece、Unigram与SentencePiece

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/19-subword-tokenization) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/19-subword-tokenization/docs/en.md)。原题：Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece。

## 一、这一课解决什么问题？

整词词表遇到新词会丢信息，逐字符又让序列太长。子词用高频长片段与低频短片段折中。本课与第04课衔接，重点是完整BPE训练/编码顺序与实际分词工具。

## 二、核心概念与推导

BPE从字符或字节开始，按语料频率累计相邻token对，每轮合并当前最高频对；规则有顺序，编码要遵循学习到的优先级。权重语料lower×5不是只计一次。

字符BPE不等于字节BPE。字节版本以256种字节为基础可表示有效文本的UTF-8字节；字符版本若基础字表不含新字符仍可能有UNK。Unigram从大候选词表开始，按似然影响逐步裁剪；可采样切分用于数据增强。WordPiece与BPE的训练准则不同，BERT类常见其最长匹配编码。

SentencePiece是可训练BPE/Unigram的库，常用▁表示空格；不是某一种合并算法。tiktoken主要使用既定编码表执行编码/解码，不用于训练新词表。模型必须配套原tokenizer，不能因为新分词更短就直接替换。

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

