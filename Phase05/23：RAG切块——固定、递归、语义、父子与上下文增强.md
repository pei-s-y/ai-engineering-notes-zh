# 23：RAG切块——固定、递归、语义、父子与上下文增强

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，五步结构与原课一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/docs/en.md)。原题：Chunking Strategies for RAG。

## 零、初学者先读：RAG流水线里最不起眼却最先出错的一刀

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。前置衔接：[第14课](<14：信息检索——BM25、稠密向量、RRF与重排.md>)选好了检索器、[第22课](<22：现代Embedding——稠密、稀疏、多向量与降维评估.md>)选好了embedding，本课在它们上游——切块切坏了，再好的检索器和向量也救不回来（原课引用的结论：切块配置对检索质量的影响不亚于embedding选型）。

### 0.1 先理解这一课在做什么

一份50页的合同塞进RAG，用户问“终止条款是什么”，检索器却递回首封面。为什么？证据在第20页、跨了页边界、块里没有和查询呼应的局部词。修复办法不是“买更好的embedding模型”，是切块（chunking）：决定“证据以什么形状进入索引”。它在流水线里的位置：

~~~text
文档 ──切块──▶ 块 ──embedding──▶ 向量入库 ──检索──▶ 重排 ──▶ 生成回答
  ↑本课        （第22课）        （第14课）
~~~

切块决定三件事：证据能不能被**完整**召回（一条证据被切成两半，哪一半都不够用）、块里的意思**连不连贯**（半句话的向量说不清自己是什么）、以及**成本**（块越多越大，索引和上下文账单越贵）。类比：把一本食谱拍成照片放进检索柜——按固定长度拍（fixed）可能把一道菜拍成两张半截照片；按“一道菜一张”拍（semantic）照片完整但大小不一；父子方案是“卡片记菜名、翻背面看整道菜”（检索小卡片、返回大照片）。

### 0.2 本课术语先翻成白话

- chunk（块）=被embedding、被索引、被检索的文档片段——RAG的最小作战单位
- fixed（固定切块）=每N个字符/token切一刀，不管断在哪
- recursive（递归切块）=先用大分隔符（\n\n）试，不行再降级到小分隔符（\n、句号、空格）
- semantic（语义切块）=给每句算向量，相邻句相似度骤降处下刀
- parent-child（父子）=检索小块（child）、返回大块（parent）
- late chunking（迟切块）=先让整文token在长上下文里“认识彼此”，再按块池化成向量
- contextual retrieval（上下文检索）=给每块前置一段LLM生成的“我在全文哪里”说明
- overlap（重叠）=相邻块共享的长度，像给边界证据上双保险——但2026年的一些基准常发现它没收益（原课参考）
- 字符 vs token=本课源码len()数的是字符；策略表里的512是token——单位不同，数字不能直接对齐（[第19课](<19：子词分词——BPE、WordPiece、Unigram与SentencePiece.md>)）
- hit rate=至少命中一条gold的查询占比（本课recall_at_k的真身）
- 查询分层=事实型/分析型/多跳查询适合的块大小不同，评估集也要按此分层
- min/max enforcement=块长的下限/上限硬约束，防40字碎片和5000字巨块（Ship It的检查项）

### 0.3 用小数字走一遍

一段12句话的文本（S1～S12），设每句恰好20字、全文240字（长度为设定值）；话题边界在S4/S5（行星→恒星）与S8/S9（恒星→彗星）。固定切块size=70、overlap=0，边界落在第70、140、210字：

| 固定块 | 覆盖内容 | 问题 |
|---|---|---|
| 块1（0–70字） | S1、S2、S3完整 + S4的前10字 | S4被腰斩 |
| 块2（70–140字） | S4的后10字 + S5、S6、S7完整 | 开头是半句 |
| 块3（140–210字） | S8、S9、S10完整 + S11的前10字 | S11被腰斩 |
| 块4（210–240字） | S11的后10字 + S12完整 | 只有30字，开头还是半句 |

“腰斩”的代价具体化：设S4讲土星、关键数字“82颗卫星”恰好落在后半句——查询命中这个数字时，块2开头是缺了主语的半句话，embedding对它的编码既缺上下文又不完整；反过来查S4前半句，块1里那半句同样残缺。块1的S1–S4本是同一个话题（行星），却在70字处被硬生生切开——**固定切块对齐的是“长度”，不是“话题”**。尾块只有30字：语义切块要用min_chars防碎片，fixed连“块太短”都察觉不了。

顺带一问：加overlap能救S4吗？size=70、overlap=10时步长变成60，边界挪到60/120/180/240——这次恰好没有句子被腰斩，但救场的其实是“步长60恰好是句长20的整数倍”这个巧合，不是overlap本身；换成size=75、overlap=10（步长65），S4照样在65字处被斩。overlap只是挪动边界、并让相邻块重复存一份内容（0.5问题3的成本账），不是万能保险。

语义切块在同样的文本上下刀：相邻句相似度在S4/S5和S8/S9两处骤降（话题换了），于是切成三块{S1–S4}、{S5–S8}、{S9–S12}——每块80字、句句完整、一块一个话题。代价：每句要过一次encoder（比固定切贵），阈值调太高会碎、太低会连成一坨（4.2的threshold=0.6）。六种策略的关系一句话收拢：recursive是fixed与semantic的折中（尊重段落/句子边界但不看语义），父子是“检索用小块的精度、返回大块的上下文”，late chunking和contextual都是在给块“补上文”——一个用长上下文模型在向量层补，一个用LLM生成的说明在文本层补。

以上是用于理解的算例，不是新增的训练或性能实测（句子长度为设定值）。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

4.1 chunk_fixed/chunk_recursive是0.3固定块的源码（注意：len和切片单位是字符；recursive不支持overlap——第七节第1条）。4.2 chunk_semantic是0.3语义块（embs[i] @ embs[i-1]算相邻句相似度，低于threshold且当前组够长才下刀）。4.3 chunk_parent_child/retrieve_parent是父子方案（检索child、按parent_idx去重后返回parent——top3可能只返回2个父块）。4.4 contextualize_chunks给每块生成“我在全文哪里”的前缀。4.5 recall_at_k评测（实际是hit rate）。第五节split_sentences补中文标点、checked_fixed校验参数；assert checked_fixed("abcdefghij",4,1)==[...]就是2.9那个例子的落地。

### 0.5 理解检查（含参考答案）

**问题1：** 本课源码里的512是字符还是token？

**参考答案：** 字符——chunk_fixed/chunk_recursive用len()和字符串切片；原文512token策略表不能直接当512字符的等价说明，需要目标tokenizer另行计数（第19课）。

**问题2：** top_k=3的父子检索，为什么可能只返回2个父块？

**参考答案：** 多个child可能映射同一个parent，retrieve_parent对parent_idx去重——“检索k个孩子”和“送模型的父块数”不是同一个数（2.9）。

**问题3：** overlap是不是越大越保险？

**参考答案：** 不是。它让边界证据两块各有一份，但索引变大、重复召回变多、上下文成本翻倍；2026年的一些基准发现overlap收益接近零（原课参考）——要不要用必须消融，不能当默认美德。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

长文不能整篇直接塞进短窗口embedding。切块决定答案证据能否被召回、是否被拆断以及上下文成本。本课比较策略，并把字符数和token数严格分开。

## 二、核心概念与推导

fixed按长度切，简单但可能断句；recursive先大分隔符再小分隔符；sentence按句或句窗；semantic在相邻句向量相似度下降处切；parent-child检索小块后返回大父块；late chunking先做整文token级上下文化再按块pool；contextual retrieval为块补充其在全文中的位置说明。

（补充 2026-11-06）六种策略按“下刀依据”排个谱，就不容易混：

| 策略 | 下刀依据 | 一句话 |
|---|---|---|
| fixed | 长度 | 到数就切，不管断哪 |
| recursive | 分隔符层级 | 段落→句子→词，逐级妥协 |
| sentence | 句子边界 | 一句一块或滑窗几块 |
| semantic | 相邻句向量相似度 | 话题变了才切 |
| parent-child | 两级粒度 | 小块检索、大块送模型 |
| late chunking / contextual | 给块补上下文 | 前者在向量层、后者在文本层 |

前四种决定“边界在哪”，后两种决定“块里带多少上文”——它们可以组合（比如recursive切块再加contextual前缀）。选哪个的第一问题是query类型：事实型（CEO是谁）块要小，分析型/多跳要大，整节理解更大——原课的256–512/512–1024/1024–2048是NVIDIA基准的参考值，不是定律。

late chunking不是先对整文求一个单向量再拆分；需模型支持长上下文及token级输出。contextual前缀是索引时生成，可能有错，要保留原文与来源。

本课chunk_fixed/chunk_recursive使用Python字符串len与切片，单位是字符，不是token。原文512token策略表不能直接作为这段代码512字符的等价说明。需要目标tokenizer另行计数。

（补充 2026-11-06）字符和token为什么不能混着数，用[第19课](<19：子词分词——BPE、WordPiece、Unigram与SentencePiece.md>)的数字就有感觉：英文1个token平均约4个字符，512字符大约才128token出头；而一个汉字在英文为主的词表里常要2～3个token，中文512字符可能接近上千token。所以“512字符”和“512token”在中文语料上可能差出一倍的上下文预算——评估切块前先统一单位（数token就用目标tokenizer数），否则块数、边界、成本全对不上账。另一头还要对齐[第22课](<22：现代Embedding——稠密、稀疏、多向量与降维评估.md>)的模型最大长度：那是token长度，切块的size如果是字符，两边的“512”根本不是一回事，超长还会被模型静默截断（第22课第七节的坑）。

chunk应携带doc_id、chunk_id、位置、父ID、版本与权限。评估gold最好定位原文证据区间，改变切块后重新映射，不能沿用旧chunk编号。原文固定2500token悬崖/69%对54%等属于特定研究描述，不是普遍阈值。

（补充 2026-11-06）元数据清单逐项展开成“为什么”：doc_id——跨文档拼块是事故（第七节第5条），先有doc_id才能保证块不出自己的文档；chunk_id+位置——答案要能引回原文，“见第3块”不如“见第14页第2段”可核查；父ID——父子方案里检索的是child、返回的是parent，没有父ID就对不上账；版本——切块参数或分词器变了，同一段文字的块集合就变（与[第19课](<19：子词分词——BPE、WordPiece、Unigram与SentencePiece.md>)tokenizer drift同族的问题）；权限——检索阶段就要能按用户权限过滤，不能等返回父块后才想起来。gold重映射的原因也具体化：gold如果存的是“chunk 7”，换一次切块后chunk 7已经是另一段文字——所以gold要存“原文证据区间”，每次切块后重新映射成新的chunk编号（2.9最后一句的展开）。

### 2.9 用一个小例子把概念串起来（补充）

abcdefghij按size=4、overlap=1会得到abcd、defg、ghij、j，最后一块只有1字符。父子方案若top3孩子有两个来自同一父块，去重后可能仅返回2父块；检索k与送模型的父块数不是同一个数。每次改变切块都要重映射原文证据。

## 三、操作前准备

固定/递归/父子映射仅用标准库；语义检索需numpy和encoder。
~~~bash
python -m pip install numpy sentence-transformers
~~~
需补split_sentences，中文标点要单独处理。contextualize_chunks的llm.batch是自定义适配器，不是任意SDK都支持。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 固定与递归切分

固定步长=size-overlap，必须0≤overlap<size且size>0。递归尝试分隔符、积累buf、超长片段继续细分；源码不保留所有原分隔符和位置，所以不能假定join后逐字还原。原recursive没有overlap参数。

~~~python
# 字符级切片，先校验size和overlap。
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


# 字符级递归，不支持overlap且不保留精确原文offset。
def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
~~~

### 4.2 按相邻句相似度切

split_sentences后编码，在相似度低且当前组长度达到min_chars时新建组，最后超过max_chars再递归拆。min_chars并非严格保证最终每块都够长，当前长度也没计空格；末组可能很短。

~~~python
# 相邻句相似度切组，min_chars不是严格最短块保证。
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
~~~

### 4.3 父子检索与去重

mapping存child、parent_idx、parent。检索top_k个子块后父文档去重，返回父块数可能少于k。原实现每次query重编码全部children，仅为教学，应在索引时预计算。

~~~python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
~~~

### 4.4 生成上下文说明

每chunk将整文与该块交给llm.batch，返回说明+原块。zip会在两列表长度不同时静默截断，要校验；文档需控制长度、隐私、费用与注入风险。

~~~python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
~~~

### 4.5 评价检索

原recall_at_k只要任一gold在topk就加1，是query hit rate/any-hit@k；多个相关块时不是严格Recall@k。空query会除零；不同切块策略要重建gold。

~~~python
# 源码实际是至少命中一条gold的query命中率。
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
~~~

## 五、把代码运行起来：补充入口与检查

补充句子切分和参数检查：
~~~python
import re

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s*", text) if s.strip()]

def checked_fixed(text, size, overlap=0):
    if not (size > 0 and 0 <= overlap < size):
        raise ValueError("需要size>0且0<=overlap<size")
    return chunk_fixed(text, size, overlap)

assert checked_fixed("abcdefghij", 4, 1) == ["abcd", "defg", "ghij", "j"]
print(chunk_recursive("One sentence. Another sentence. Final sentence.", size=20))
print(chunk_parent_child("a b c d e f g h i j", parent_size=10, child_size=4))
~~~
固定切法可能出现极短尾块，检查结果后决定合并策略，而非声称min严格满足。真实语义检索未运行。

## 六、Use It：使用现成工具

先用可解释基线，再按factoid、分析、多跳、整节理解分层评价；512、256～2048与0overlap是候选超参，不是标准答案。对话可按轮次并保留speaker，短文本可整文一块，交叉引用多可试父子/late/contextual。必须衡量索引体积、证据完整性和最终回答，而非只比块数量。



## 七、失败模式、原文边界与工程注意

1. 原练习recursive(512,100)不能直接调用源码：第三个参数是seps不是overlap，传100会报错。需改用支持overlap的库或明确补实现。
2. 固定切分overlap=size时步长0报错，大于size可产生空列表，必须先校验。
3. recursive会改变分隔符，原文精确offset需独立维护。
4. overlap可增大索引、重复召回与上下文成本，也可能帮助边界证据，必须消融。
5. parent返回仍需权限过滤、去重及预算，不能跨文档拼块。
6. 语义相似不是事实关联完整性，合同/表格/代码要结构化处理。
7. contextual前缀不能替换原文，生成内容不应冒充证据。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
列文档结构、query类型与证据粒度；明确字符或token单位、策略、size、overlap和尾块政策。保存来源位置与父子ID，建立分层gold，比较hit/Recall/MRR、答案支持度、索引与生成成本，并做overlap消融。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：20页文档比较fixed(512,0)、recursive无重叠与支持100重叠的递归实现；说明源码原函数不支持overlap，统一长度单位后对比块数与边界。
2. Medium：5文档30查询，比较recursive/semantic/parent的hit@5及严格Recall@5；为每策略重新映射原文证据。
3. Hard：实现contextual retrieval，对比递归基线MRR与答案支持度，记录索引LLM调用数/token成本、失败、生成前缀错误及隐私处理。

## 十、术语与完成标准

chunk：检索单元；overlap：相邻块重叠；parent-child：检索子块返回父块；late chunking：上下文化后pool；contextual：生成索引说明；hit rate：至少命中一条的查询占比。

完成检查：

- 能指出源码长度是字符。
- 能解释父块返回数少于k。
- 能区分hit与Recall。
- 能发现recursive不支持overlap。

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

