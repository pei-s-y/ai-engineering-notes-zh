> 路径约定：本文的 `~/projects/ai-engineering-from-scratch` 是通用示例，不表示仓库已位于该处。执行命令前，请将它替换为实际克隆目录；以 `phases/` 开头的路径均相对于课程仓库根目录。

课程：[Tokenizers: BPE, WordPiece, SentencePiece](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/01-tokenizers)

本笔记依据你本地仓库的 `docs/en.md`、`code/main.py` 与 `code/bpe.py` 整理，并结合官方资料澄清容易误解的表述。当前 Python 示例主要实现字符级分词和字节级 BPE；WordPiece、SentencePiece/Unigram 主要属于本课的概念介绍或扩展练习。

## 零、初学者先读：为什么模型不能直接读字符串

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

计算机可以保存字符串，但模型的矩阵运算需要数字。分词器先规定怎样把文字拆成片段，再为每种片段编号。编号像图书馆的索书号：便于查找，不表示两本书内容相似。下一课才把这个简单工具组装成处理特殊标记、空白等情况的完整流程。

### 0.2 本课术语先翻成白话

- token=一次处理的片段

- vocab=片段与编号的对照表

- encode=文字转编号

- decode=编号还原文字

- byte=字节，UTF-8把一个字符表示为一个或多个字节。

### 0.3 用小数字走一遍

用abababab做一次BPE：初始是a b a b a b a b；ab相邻出现4次，ba出现3次，于是新增片段ab。整段变成4个ab。新文本abab则变成2个ab。新增ID256只表示查表时能找到b'ab'，并没有学到ab的语义。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先看_get_pairs怎样计数，再用纸笔跟_merge_pair的i走一遍。train保存合并规则；encode使用已经保存的规则。decode先把字节拼完整再转字符串，是因为中文的几个字节可能分在不同token里。

前置阅读：从下面的小例开始即可。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 编号256比97大，是否表示语义更强？

**参考答案：** 不是，编号只是索引。

**问题2：** 为什么每次合并后重数相邻对？

**参考答案：** 旧单位已经改变，原来相邻的两个字符可能不再是两个独立token。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 1. 分词器在大语言模型中负责什么？

Tokenizer 把文本转换为模型使用的 token ID 序列，并提供从 ID 序列还原文本的方式。它负责“表示文本”，不是负责“理解文本”。

```text
输入文本
  → tokenizer.encode：切分并映射为 token IDs
  → embedding：根据 ID 查表，得到向量
  → Transformer：处理向量，预测后续 token
  → tokenizer.decode：把输出 IDs 转回文本
```

三个概念要分开：

- **Token**：分词的基本单位，可能是一个词、一部分词、标点、空白，甚至只是某个字符的一部分字节。
- **Token ID**：这个单位在词表中的编号，是离散索引。
- **Embedding**：模型为该编号学习的向量表示，不是编号本身。

例如本课字符级示例把 `hello` 转成 `[104, 101, 108, 108, 111]`，这些是 Unicode 码点。实际子词分词器可能得到另一组编号。**不同分词器的 ID 不通用，ID 大小也不表示语义大小或相似程度。**

同样的文本，换一个分词器，切分方式和 token 数量都可能变化。

## 2. 为什么不直接按单词或字符切分？

| 方式 | 思路 | 主要代价 |
|---|---|---|
| 词级 Word-level | 整个单词对应一个 ID | 词形、专有名词、代码等太多，词表大，容易出现未知词 |
| 字符级 Character-level | 每个字符对应一个 ID | 序列长，模型要用很多位置组合出常见词 |
| 子词级 Subword | 常见片段合并，少见词拆成较小片段 | 需要学习词表，效果依赖语料与切分规则 |

子词切分是在词表大小与序列长度之间折中。它不一定按词根词缀或人工语言学规则切分：BPE 主要根据统计频率学习片段。

`unhappiness → [un, happi, ness]` 只是说明子词思路的例子，不是所有分词器都会输出这个结果。

“字符级词表很小”也有条件：只覆盖英文字母时确实小，覆盖多种语言的 Unicode 字符时就大得多。只收录训练语料中出现的字符，也可能遇到未知字符。

本课的 `CharTokenizer` 直接使用 `ord()` 与 `chr()`，没有学习紧凑词表。它是教学基线，不能把这些稀疏码点当成一个只有几百行的 embedding 表。

## 3. 字符与字节：理解 Byte-level BPE 的关键

Python 字符串是文本，`text.encode("utf-8")` 才把它转换为字节序列。

```python
list("A".encode("utf-8"))   # [65]
list("你".encode("utf-8"))  # [228, 189, 160]
```

这里 `你` 是一个字符，但对应三个 UTF-8 字节。字符数、字节数和 token 数不是同一个量。

字节级 BPE 以 0～255 这 256 种字节作为基础单位，再学习字节片段的合并。只要完整保留基础字节，就能表示有效 UTF-8 文本中的少见字符，不必因为字符没在训练语料出现过就输出 `[UNK]`。

但是，**能表示中文不等于表示中文很高效**。如果训练数据主要是英文，中文可能需要更多小片段才能表示。tiktoken 也是 BPE 工具；官方资料强调其可逆性和对训练语料之外文本的覆盖能力。[tiktoken 官方说明](https://github.com/openai/tiktoken)

## 4. BPE：反复合并最常见的相邻单位

### 4.1 训练：学到词表和合并规则

Byte Pair Encoding 的核心循环：

1. 把训练文本转成初始字节 ID 序列。
2. 统计当前序列中每一种相邻 ID 对的出现次数。
3. 选择出现次数最多的一对。
4. 为合并后的片段分配新 ID，在当前序列里应用合并。
5. 保存规则与新片段的字节内容，重新统计，继续下一轮。

**每次合并后都要重新统计。** 例如已经把 `w` 和 `e` 合成 `we`，接下来应考虑 `(we, s)`，而不是还按旧序列统计 `(e, s)`。

合并是贪心选择，不保证得到最符合语言学意义的片段；频数相同时，平局处理方式也可能影响结果。

### 4.2 一个能手算的完整例子

训练文本为 `abababab`，只做一次合并：

```text
初始字节 IDs：[97, 98, 97, 98, 97, 98, 97, 98]
相邻对计数：(97,98) 出现 4 次；(98,97) 出现 3 次
选中最多的：(97,98)，也就是 a + b
创建新 ID：256，对应字节片段 b'ab'
合并后的序列：[256, 256, 256, 256]
词表大小：256 个基础字节 + 1 个新片段 = 257
```

之后编码新文本 `abab`，得到 `[256, 256]`；解码时查到两个 `b'ab'`，拼接为 `b'abab'`，再还原成字符串。

### 4.3 编码与解码：使用规则，不重新训练

- `train()`：根据训练语料学习哪些片段值得合并。
- `encode()`：对新文本使用已经学习的规则，返回 IDs。
- `decode()`：查出每个 ID 对应的字节片段，先拼接全部字节，再进行 UTF-8 解码。

本课 `encode()` 按学习顺序遍历合并规则。规则有先后依赖：如果先学 `t+h→th`，后学 `th+e→the`，编码时必须尊重这个优先关系。

这段代码是简化教学实现；生产分词器还包含预切分规则、特殊 token 处理、合并优先级实现和性能优化，不能仅理解为“同一段 Python 代码训练得更久”。

本实现实际完成 M 次合并后，词表大小为 `256 + M`。如果已经没有相邻对，会提前退出，因此请求的 `num_merges` 不一定全部执行完。

## 5. WordPiece、SentencePiece 和 Unigram 如何区分？

### WordPiece：训练选择不同，编码方式也不同

BPE 优先看相邻对的频数；本课介绍的 WordPiece 教学评分是：

$$score(A,B)=\frac{count(A,B)}{count(A)\times count(B)}$$

它更关注两部分相对各自频率的共同出现程度，而不只是绝对频数。BERT 的词中延续片段常带 `##` 标记，比如 `[un, ##happi, ##ness]`；这里是示意，具体结果取决于词表。

编码时，WordPiece 通常使用词表做从左到右的最长匹配，而不是按 BPE 合并表执行。无法完整匹配一个词时可能产生 `[UNK]`，所以“子词分词绝不会有未知词”不成立。

需要注意：上面的训练评分是常见的教学说明。Hugging Face 明确指出，原始 WordPiece 训练实现未公开，其讲解是依据文献的重建，并不保证细节完全一致。[WordPiece 官方课程](https://huggingface.co/learn/llm-course/chapter6/6)

### SentencePiece：是工具，不是单独一种合并算法

SentencePiece 可直接从原始文本训练，不要求先按空格划成单词；支持 BPE 与 Unigram 等模式。它常用 `▁` 表示空白信息，适合处理中文、日文等没有显式空格词边界的语言。

所以应区分“用什么工具”与“采用什么算法”：**SentencePiece 可以使用 BPE，也可以使用 Unigram。** 它的模型文件还包含规范化等规则，不能只记一个词表。[SentencePiece 官方说明](https://github.com/google/sentencepiece)

### Unigram：从大词表里删掉不重要的候选

BPE 从小单位逐渐合并；Unigram 从较大的候选词表开始，学习片段概率，逐步删掉对语料似然影响较小的候选。

编码时比较不同切分的概率，通常选择概率最高的切分，可用动态规划/Viterbi 算法求解。不是简单挑“字符最长的 token”，也不是在执行相邻对合并。[Unigram 官方课程](https://huggingface.co/learn/llm-course/chapter6/7)

## 6. 词表越大，是不是越好？

设词表大小为 V，embedding 维度为 d，输入 embedding 表的参数量为：

$$N_{embedding}=V\times d$$

当 d = 4096 时：

- V = 32,000：131,072,000 个参数，约 1.31 亿。
- V = 128,000：524,288,000 个参数，约 5.24 亿。
- 相差约 3.93 亿参数；这里只计算输入 embedding 表，不是整个模型。

小词表通常让同一文本产生更多 token，序列更长；大词表可能用更少位置表示常见片段，但增加 embedding 与词表输出层的规模，也可能包含训练不足的少见片段。

更长序列增加上下文占用、注意力计算和生成步数。**词表大、token 少，不自动等于模型更好或推理一定更快**：还要考虑模型规模、输出层开销、硬件和实现。

token ID 数字较小，也不等于模型就省计算。模型通常按固定类型存储和查表；真正重要的是序列长度、词表大小及表示质量，而不是编号有几位十进制数字。

## 7. 如何衡量分词效率？

本课主要使用三类统计：

| 指标 | 计算方式 | 注意事项 |
|---|---|---|
| Tokens / bytes | token 数 ÷ UTF-8 字节数 | 本课的压缩比，越低表示同一文本需要的位置越少 |
| Tokens / characters | token 数 ÷ 字符数 | 中英文字符的含义和信息量不同，不能当作语言能力排名 |
| Tokens / words | token 数 ÷ 词数，常称 fertility | 需要明确词边界，中文不能简单按空格计词 |

本课的“压缩比”衡量的是序列单位数，不是压缩文件占用多少字节。Roundtrip 通过说明文本可以还原，不说明模型理解得更好。

比较多语言时，应使用足够多、领域相近的样本；若要比较相同含义需要多少 token，可使用互为翻译的文本。语料中代表不足的语言往往切得更碎，但比例需要实际测量，不能套用固定倍数。

比较代码时应关注缩进、换行、标点和变量名。换成英语、中文、代码三组文本测试，能直观看到训练语料对合并规则的影响。

## 8. 对照本地代码：每个函数在做什么？

- `CharTokenizer.encode/decode()`：用 `ord/chr` 做字符与整数转换。
- `BPETokenizer._get_pairs()`：用 `Counter` 统计相邻 ID 对。
- `_merge_pair()`：从左到右扫描，遇到目标对就替换为新 ID，并前进两格，避免重叠合并。
- `train()`：初始化 256 个字节，重复计数、选对、合并，并更新 `merges` 与 `vocab`。
- `encode()`：文本转字节，再应用保存的合并规则。
- `decode()`：ID 查字节，拼接后解码。
- `compression_ratio()`、`vocabulary_stats()`：统计编码长度、词表使用情况。

这里 `merges` 保存“ID 对 → 新 ID”，`vocab` 保存“ID → 字节片段”。它们不是同一个字典，职责不同。

重点 Python 知识：`Counter`、元组作为字典键、`max(..., key=...)`、`bytes`、字符串编码、列表推导式，以及 `while` 循环中的索引推进。

## 9. 运行本课会看到什么？

`main.py` 的五个步骤如下：

| 演示函数 | 观察目标 |
|---|---|
| `demo_char_tokenizer()` | 字符如何变成整数，编码再解码是否一致 |
| `demo_bpe_training()` | 学习 50 次合并，得到 306 个词表项 |
| `demo_encode_decode(tokenizer)` | 新文本的 IDs、token 数、压缩比和 Roundtrip |
| `demo_tiktoken_comparison(tokenizer)` | 与预训练的 `cl100k_base` 编码比较 token 数及片段 |
| `demo_vocabulary_analysis(tokenizer, corpus)` | 常用 token、未使用词表项、不同文本的压缩比 |

我运行了不依赖外部库的部分，得到这些结果：

| 文本 | UTF-8 字节数 | BPE token 数 | Tokens / bytes |
|---|---:|---:|---:|
| `The cat sat on the mat.` | 23 | 7 | 0.30 |
| `Natural language processing` | 27 | 16 | 0.59 |
| `tokenization pipeline` | 21 | 12 | 0.57 |
| `unhappiness` | 11 | 10 | 0.91 |

这些样本都能还原原文。常见训练模式合并得较好；不熟悉的文本仍能编码，但往往切得较碎。我也检查了 `你好，世界！` 的往返还原，结果正确，这不代表英文训练语料已经让中文切分高效。

这里的未使用 token 数只针对当前测试样本集合，不能据此断言这些词表项在真实应用中都浪费了。256 个基础字节也包含当前英文样本没有用到的字节。

两个 Python 文件的配置不同：`main.py` 合并 50 次，`bpe.py` 合并 30 次；课程正文示例另有 40 次的配置，所以看到 306、286 或 296 的词表大小都可能合理。`bpe.py` 还会逐轮打印合并规则。

在你的 WSL Ubuntu 中运行完整示例：

```bash
python3 ~/projects/ai-engineering-from-scratch/phases/10-llms-from-scratch/01-tokenizers/code/main.py
```

手写部分只依赖标准库。若没有安装 `tiktoken`，代码会提示并跳过比较，其他步骤仍可运行。这次我没有执行外部库比较，也没有安装依赖或修改原仓库。

## 10. 工程上容易踩的坑

1. **不要随意替换预训练模型的分词器。** 词表编号、特殊 token 和模型 embedding/输出层要匹配；换成自己的编号表，模型不会自动理解。
2. **tokenizer 训练不是 LLM 训练。** 本课 `train()` 学的是合并规则，没有训练 Transformer，也没有反向传播。
3. **不要丢掉空白。** 简单 `split()` 后再拼接，可能改变代码缩进和连续空格，破坏精确还原。
4. **特殊 token 是协议的一部分。** `<pad>` 用于填充，`<eos>` 表示结束，`<unk>` 表示未知单位；具体 ID 和行为由实现约定。完整字节覆盖的实现不必因普通未知字符而使用 `<unk>`。
5. **给特殊 token 腾出 ID 时，要同步更新全部引用。** 若占用 0、1、2，不仅基础字节 ID 要移动，合并表中的输入/输出 ID 与词表映射也要一致，不能只改一个字典。
6. **单个 token 可能不是完整 UTF-8 字符。** 单独解码时显示 `�`，不代表完整序列已经损坏；先拼接对应字节，再解码完整序列。
7. **往返还原有实现条件。** 本课对有效输入文本的正常编码结果可以还原；任意拼出的 ID 序列不一定是合法 UTF-8，`errors="replace"` 可能用替代字符掩盖错误。其他分词器若做了规范化或大小写转换，也未必逐字符还原原输入。
8. **教学实例最好一次训练一个新对象。** 本课 `train()` 没有清空旧 `merges`，反复在同一实例上训练可能留下与新词表不一致的规则。

课程的 Use It 部分还展示了 Hugging Face `tokenizers` 训练 BPE，以及 `transformers.AutoTokenizer` 加载模型配套分词器。加载分词器不等于加载完整模型；训练工业级分词器时仍需要明确字节覆盖、规范化、特殊 token 和解码配置。

Ship It 中的分析提示词可辅助组织报告，但 token 数和切分结果应由实际 tokenizer 计算，不要把语言模型猜测的计数当成精确结果。

## 11. 复习标准与原课练习

学完后，你应能解释：token、ID 和 embedding 的区别；字符与字节的区别；BPE 为什么每轮重计相邻对；`train/encode/decode` 的区别；词表大小为何是工程折中；中文为何可能被切得更碎。

保留原课的五项练习，便于你自己安排：

1. 每轮打印合并规则与词表，观察常见片段如何形成。
2. 添加特殊 token 与空白预切分；同时考虑空白能否保留、ID 如何避免冲突。
3. 实现 WordPiece 教学评分，与 BPE 在相同语料上比较。
4. 对英语、西班牙语、中文、韩语、阿拉伯语做分词效率基准测试。
5. 用更大语料训练 BPE，改变合并次数，与 tiktoken 比较。除了同一训练文本，也应保留未用于训练的文本，避免把记住训练语料误认为泛化良好。

本课有助于理解语言输入如何变成模型序列，以及上下文预算和多语言表示。其他模态的表示不一定采用文本 BPE，不要把所有“token”都理解为文本子词。

一句话总结：**分词器决定文本如何被切成模型能处理的编号；BPE 通过合并高频相邻单位学习表示，词表大小和训练语料共同影响序列长度与覆盖效率，而语义能力最终来自模型训练。**


---

## 2026-10-25 完整学习版

原笔记保留在上方。本次按整章统一标准补齐代码、操作入口与练习讲解，以下可独立学习。上方“我运行了”的50次合并数据是旧笔记历史记录；本次验证采用下面明确写出的配置，不混为一次实验。新增代码按定义顺序使用，旧文中的演示数字不作为下面40次合并实例的预设输出。

# 01：分词器——BPE、WordPiece 与 SentencePiece

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/01-tokenizers) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/01-tokenizers/docs/en.md)。原题：Tokenizers: BPE, WordPiece, SentencePiece。

## 一、这一课解决什么问题？

本课把文本变成模型可处理的整数序列，弄清token、token ID和embedding不是同一个概念。学习目标包括BPE机制、WordPiece与Unigram区别、词表大小和跨语言效率；但原课Build It实际只实现CharTokenizer与字节BPE，没有完整实现三种算法，其他算法保留为概念与练习。

## 二、核心概念与推导

### 2.1 三种粒度
词级整词映射容易OOV；字符级序列长，若仅收录训练字符也会OOV；子词让常见片段合并、稀有词拆分，不保证每块是词根。字节BPE基础为256种字节，Python字符串先转UTF-8，例如你→[228,189,160]。覆盖字节不等于模型理解所有语言。

### 2.2 BPE到底学什么
每轮对当前序列计相邻对，选最高频、创建新ID、非重叠替换，再重新统计。merges保存有序(ID对→新ID)，vocab保存(ID→字节片段)。train学习规则，encode应用规则，decode先拼全部字节再解UTF-8。若未满轮数就没相邻对，则实际词表不是请求轮数+256。

### 2.3 WordPiece、SentencePiece、Unigram
WordPiece常用教学评分count(AB)/(count(A)count(B))，编码通常最长匹配，BERT里##表示延续。原始训练细节与教学重建应区分。SentencePiece是库，可训练BPE或Unigram，常用▁表示空格；Unigram从大候选集按似然影响裁剪，并以动态规划求切分。不能把SentencePiece叫另一种BPE。

### 2.4 模型和tokenizer配套
Llama2与Llama3的tokenizer路线不同，不能泛说“Llama都用SentencePiece”。词表ID与embedding/输出层绑定，换tokenizer通常需要适配并训练。merge表也不是完整生产tokenizer：还需预切分、正规化、特殊token、decoder和配置。

### 2.5 词表大小与预算
V×d是输入embedding参数；V=32000、d=4096为131072000，V=128000为524288000，相差393216000。输出权重若共享不能重复算，未共享另算。大词表可能缩短序列，也增加输出层和罕见片段参数，不保证更快更好。

原课列模型词表、平均每词token、价格与多语言倍数属于示例背景，不作为所有版本/语言的定律。tokens/UTF-8字节越低表示该协议下位置更少，不是实际压缩文件字节率；中文每词需要明确分词协议。

### 2.9 补充算例与理解检查

abababab初始97、98交替，(97,98)频数4、(98,97)频数3。一次合并后新ID256表示b'ab'，文本变[256,256,256,256]；新文本abab编码[256,256]。lower×5、lowest×2、newest×6先合并(w,e)频数13，此后(e,s)已不再相邻，应统计(we,s)，不能沿用旧计数。

### 原课概念部分的公式与结构示例

~~~text
Corpus (with word frequencies):
  "lower"  x5
  "lowest" x2
  "newest" x6

Step 0 -- Start with characters:
  l o w e r       (x5)
  l o w e s t     (x2)
  n e w e s t     (x6)

Step 1 -- Count adjacent pairs:
  (e,s): 8    (s,t): 8    (l,o): 7    (o,w): 7
  (w,e): 13   (e,r): 5    (n,e): 6    ...

Step 2 -- Merge most frequent pair (w,e) -> "we":
  l o we r        (x5)
  l o we s t      (x2)
  n e we s t      (x6)

Step 3 -- Recount and merge (e,s) -> "es":
  l o we r        (x5)
  l o we s t      (x2)    <- 'es' only forms from 'e'+'s', not 'we'+'s'
  n e we s t      (x6)    <- wait, the 'e' before 'we' and 's' after 'we'

Actually tracking this precisely:
  After "we" merge, remaining pairs:
  (l,o): 7   (o,we): 7   (we,r): 5   (we,s): 8
  (s,t): 8   (n,e): 6    (e,we): 6

Step 3 -- Merge (we,s) -> "wes" or (s,t) -> "st" (tied at 8, pick first):
  Merge (we,s) -> "wes":
  l o we r        (x5)
  l o wes t       (x2)
  n e wes t       (x6)

Step 4 -- Merge (wes,t) -> "west":
  l o we r        (x5)
  l o west        (x2)
  n e west        (x6)

...continue until target vocab size reached.
~~~

~~~text
BPE merge criterion:      count(A, B)
WordPiece merge criterion: count(AB) / (count(A) * count(B))
~~~

~~~text
"unhappiness" -> ["un", "##happi", "##ness"]
"embedding"   -> ["em", "##bed", "##ding"]
~~~

## 三、操作前准备

字符/BPE部分只需标准库。先按Build It定义类，再执行测试，最后按需运行外部工具：
~~~bash
python -m pip install tiktoken tokenizers transformers sentencepiece
~~~
加载受限模型的tokenizer可能需要授权；不必为了学习BPE下载模型权重。下面各段都用tokenizer这个名称，运行HF工具后会覆盖手写对象，建议分别命名scratch_tok/hf_tok或分会话。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 字符基线

encode逐字符ord，decode逐码点chr。hello是[104,101,108,108,111]。这不是256字节词表，也没有紧凑字符编号；不能直接配一个只有256行的embedding处理中文码点。

~~~python
class CharTokenizer:
    # 将文本转为离散ID；本类具体采用的粒度要看实现。
    def encode(self, text):
        return [ord(c) for c in text]

    # 字节BPE必须先拼全部字节再UTF-8解码。
    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)
~~~

### 4.2 完整字节BPE类

_get_pairs计算相邻ID对；_merge_pair遇到匹配前进2格，否则1格。训练从256个字节开始，new_token=256+i，vocab拼接两个旧片段。字典插入顺序保存规则优先级。本实现再次train没有清空merges，应每次创建新实例或显式重置，不能复用后假定结果可靠。

~~~python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    # 在当前ID序列上计相邻对，每轮合并后重计。
    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    # 从左到右非重叠替换，匹配前进2，否则前进1。
    def _merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    # 学习有序规则，原版不会清空旧merges，建议新建对象。
    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        return self

    # 将文本转为离散ID；本类具体采用的粒度要看实现。
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    # 字节BPE必须先拼全部字节再UTF-8解码。
    def decode(self, tokens):
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")
~~~

### 4.3 编码解码与压缩统计

源文这里40轮，和旧笔记仓库main.py的50轮不是同一配置。每句打印token数、字节数、比率及往返。空文本raw_bytes=0会除零，应单独处理。即使没见过中文也能字节覆盖，但可能更碎。

~~~python
corpus = (
    "The cat sat on the mat. The cat ate the rat. "
    "The dog sat on the log. The dog ate the frog. "
    "Natural language processing is the study of how computers "
    "understand and generate human language. "
    "Tokenization is the first step in any NLP pipeline."
)

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=40)

test_sentences = [
    "The cat sat on the mat.",
    "Natural language processing",
    "tokenization pipeline",
    "unhappiness",
]

for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    decoded = tokenizer.decode(encoded)
    raw_bytes = len(sentence.encode("utf-8"))
    ratio = len(encoded) / raw_bytes
    print(f"'{sentence}'")
    print(f"  Tokens: {len(encoded)} (from {raw_bytes} bytes) -- ratio: {ratio:.2f}")
    print(f"  Roundtrip: {'PASS' if decoded == sentence else 'FAIL'}")
~~~

### 4.4 与预置编码比较

tiktoken编码表来自预训练规则；不仅是“相同Python循环+更多轮”，还有预切分、特殊token和高效merge-rank实现。单独decode一个token可能不是完整UTF-8字符，出现替代符不等于完整序列损坏；检查拼接后解码。

~~~python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "The cat sat on the mat.",
    "unhappiness",
    "Hello, world!",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Geschwindigkeitsbegrenzung",
]

for text in texts:
    our_tokens = tokenizer.encode(text)
    tiktoken_tokens = enc.encode(text)
    tiktoken_pieces = [enc.decode([t]) for t in tiktoken_tokens]
    print(f"'{text}'")
    print(f"  Our BPE:   {len(our_tokens)} tokens")
    print(f"  tiktoken:  {len(tiktoken_tokens)} tokens -> {tiktoken_pieces}")
~~~

### 4.5 观察词表使用

累计测试文本的token频次、字符数与未使用项。未使用是相对这份test_texts，不说明词表全局浪费。所有文本为空时total_chars=0需防护。编号位数小不自动减少固定精度embedding查表成本。

~~~python
# 统计仅针对当前测试集，注意全空文本分母为0。
def analyze_vocabulary(tokenizer, test_texts):
    total_tokens = 0
    total_chars = 0
    token_usage = Counter()

    for text in test_texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        total_chars += len(text)
        for t in encoded:
            token_usage[t] += 1

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Total tokens across all texts: {total_tokens}")
    print(f"Total characters: {total_chars}")
    print(f"Avg tokens per character: {total_tokens / total_chars:.2f}")

    print(f"\nMost used tokens:")
    for token_id, count in token_usage.most_common(10):
        token_bytes = tokenizer.vocab[token_id]
        display = token_bytes.decode("utf-8", errors="replace")
        print(f"  Token {token_id:4d}: '{display}' (used {count} times)")

    unused = [t for t in tokenizer.vocab if t not in token_usage]
    print(f"\nUnused tokens: {len(unused)} out of {len(tokenizer.vocab)}")
~~~

## 五、把代码运行起来：补充入口与检查

不安装外部库，定义Build It类后可直接运行：
~~~python
toy = BPETokenizer().train("abababab", 1)
assert toy.encode("abab") == [256, 256]
assert toy.decode(toy.encode("你好，🌍！")) == "你好，🌍！"
assert toy.decode(toy.encode("")) == ""
assert len(toy.vocab) == 257
print(toy.merges, toy.encode("abab"))
~~~
预期首规则(97,98)→256、编码[256,256]。本课完整语料输出由你实际运行40轮得到，不能把旧笔记50轮结果挪来当同一配置。后续导航会注明本次实际验证范围。

## 六、Use It：使用现成工具

tiktoken使用get_encoding加载既定编码表；HF Tokenizer(BPE())可训练新词表，需真实corpus.txt。源文ByteLevel预切分示例未配ByteLevel decoder/完整字节初始alphabet，不足以证明所有字符覆盖和原样round-trip；应补对应decoder、初始字节字母表和OOV策略。AutoTokenizer只加载tokenizer，不等于加载LLM权重；文本模板、特殊token也要按checkpoint使用。

~~~python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenizers convert text to integers"
tokens = enc.encode(text)
print(f"Tokens: {tokens}")
print(f"Pieces: {[enc.decode([t]) for t in tokens]}")
print(f"Roundtrip: {enc.decode(tokens)}")
~~~

~~~python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<eos>", "<unk>"])
tokenizer.train(["corpus.txt"], trainer)

output = tokenizer.encode("The cat sat on the mat.")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
~~~

~~~python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

text = "Tokenizers are the unsung heroes of LLMs"
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
print(f"Vocab size: {tokenizer.vocab_size}")

multilingual = ["Hello world", "Hola mundo", "Bonjour le monde"]
for text in multilingual:
    ids = tokenizer.encode(text)
    print(f"'{text}' -> {len(ids)} tokens")
~~~

## 七、失败模式、原文边界与工程注意

1. 源文“每个字符级实现无OOV”只在ord/chr或覆盖全部字符前提成立。
2. 字节覆盖针对有效字符串编码；本接口接受str，不是任意损坏二进制文件接口。
3. train未清空规则，重复训练可能污染；decode错误替换也会隐藏非法字节。
4. 词表大不保证更快，token少不保证语义好；固定多语言倍数需要真实数据。
5. 特殊token占0、1、2时必须同步平移基础字节、merges与vocab全部引用。
6. 无损空白预切分不能简单split后丢分隔符。
7. 用1000轮合并压记训练段不等于泛化好，要在留出文本比较。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
分析给定tokenizer与文本，实际计算字节/字符/token数、切分和round-trip；明确词表版本、特殊token、正规化与语言。解释效率与参数的折中，不猜精确token ID/计数，不把示意平均值当实测；报告与模型是否配套。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 逐轮打印规则和词表，观察t+h→th及th+e→the等模式；若该语料先学别的规则也正常。
2. 加pad/eos/unk并分配0、1、2，整体重映射；添加保留空白的预切分，验证代码缩进与多空格。
3. 实现WordPiece教学评分，同一语料/轮数对比BPE；注明这是教学重建，不冒充BERT原训练实现。
4. 英/西/中/韩/阿拉伯语各10句，用cl100k_base统计token/字符；匹配领域、最好使用翻译对，不把字符比当语言智力排名。
5. 用合法可用的维基文章训练，调合并数争取同文本比率距tiktoken在10%内；这是练习目标非保证，并增加留出文本防记忆。

## 十、术语与完成标准

token：离散片段；ID：索引；embedding：学习向量；BPE：相邻对贪心合并；WordPiece：子词训练/匹配方案；Unigram：概率子词；fertility：每词token；pre-tokenization：合并前切分。

完成检查：

- 能手算abab合并。
- 能解释三张概念：token、ID、embedding。
- 能读懂merges与vocab职责。
- 能处理重复train、空文本与多字节字符。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行标准库BPE小例：首规则(97,98)→256，abab→[256,256]，词表257；中文/emoji与空串往返通过。没有重跑旧笔记50次合并语料结果。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

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

返回：[[00：Phase10学习导航与验证说明]] · 下一课：[[02：构建完整分词器——Unicode、预切分、特殊Token与聊天模板]]
