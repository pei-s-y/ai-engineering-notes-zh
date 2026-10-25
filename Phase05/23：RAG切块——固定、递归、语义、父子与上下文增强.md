# 23：RAG切块——固定、递归、语义、父子与上下文增强

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/docs/en.md)。原题：Chunking Strategies for RAG。

## 一、这一课解决什么问题？

长文不能整篇直接塞进短窗口embedding。切块决定答案证据能否被召回、是否被拆断以及上下文成本。本课比较策略，并把字符数和token数严格分开。

## 二、核心概念与推导

fixed按长度切，简单但可能断句；recursive先大分隔符再小分隔符；sentence按句或句窗；semantic在相邻句向量相似度下降处切；parent-child检索小块后返回大父块；late chunking先做整文token级上下文化再按块pool；contextual retrieval为块补充其在全文中的位置说明。

late chunking不是先对整文求一个单向量再拆分；需模型支持长上下文及token级输出。contextual前缀是索引时生成，可能有错，要保留原文与来源。

本课chunk_fixed/chunk_recursive使用Python字符串len与切片，单位是字符，不是token。原文512token策略表不能直接作为这段代码512字符的等价说明。需要目标tokenizer另行计数。

chunk应携带doc_id、chunk_id、位置、父ID、版本与权限。评估gold最好定位原文证据区间，改变切块后重新映射，不能沿用旧chunk编号。原文固定2500token悬崖/69%对54%等属于特定研究描述，不是普遍阈值。

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

