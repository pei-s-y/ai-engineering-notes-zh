# 03：预训练数据流水线——清洗、MinHash去重、打包与批次

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/03-data-pipelines) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/03-data-pipelines/docs/en.md)。原题：Data Pipelines for Pre-Training。

## 一、这一课解决什么问题？

模型前向只是一部分，数据还要清洗、去重、tokenize、按上下文长度打包和提供批次。本课Build It是内存中的教学流水线，不是已经能流式处理TB数据的工程实现。你要分清目标与源码实际做到的范围。

## 二、核心概念与推导

### 2.1 来源与配比
Web、百科、代码、书籍、论文、问答和清洗Web质量/许可不同。原文给出语料规模及具体模型百分比属于来源背景，不可直接当已核实配方。任何下载都需许可、隐私、去重与评测污染审查；本次不抓网页。

### 2.2 清洗与质量
HTML/脚本/导航、URL、垃圾重复、PII与语言混杂需分别处理。正则剥标签不是完整HTML正文抽取，困惑度高只说明不像参考语言模型，不一定低质量，会误杀方言、代码和稀有领域。对分组保留率做审查。

### 2.3 MinHash与LSH
文档→word shingles集合→多哈希最小值签名→分band桶→候选对→精确Jaccard验证。J=|A∩B|/|A∪B|；签名近似相似，LSH减少候选，不保证最坏线性，也可能漏掉近重复。完全重复可先SHA256，近重复再用MinHash。

### 2.4 Packing与mask
串接文档并加EOS，再切固定长度块。padding mask标识有效位置；causal mask防看未来；document mask可隔离不同文档，这三者不是同一张mask。是否允许后文档看前文档是一种训练政策，EOS不会自动隔离；源码只返回1D padding mask。

### 2.5 规模法则与吞吐
Chinchilla讨论固定训练算力预算下参数/数据配比，常见约20 token/参数是近似背景，不是绝对上限。更长训练换小模型推理收益与计算最优不冲突，也不保证“10倍参数+10倍数据得到相同loss”。记录tokens/s、数据等待、过滤比例、shuffle与断点状态，不能仅用总数据量表示质量。

### 2.9 补充算例与理解检查

两文档IDs为[10,11]与[12]，EOS=2，串接[10,11,2,12,2]。seq_length=4得到[10,11,2,12]和[2,0,0,0]，对应padding mask[1,1,1,1]、[1,0,0,0]。第二文档的12是否能看10、11，不由这两行0/1决定，还需要训练时的注意力策略。

### 原课概念部分的公式与结构示例

$$
\begin{aligned}
N_{\text{opt}} &\sim C^{0.5} \\
D_{\text{opt}} &\sim C^{0.5}
\end{aligned}
$$

## 三、操作前准备

清洗、MinHash、打包和toy DataLoader只用标准库。tokenizer接口必须有encode与eos_id，不能直接拿上一课没有该属性的实例就运行。源码pack默认pad_id=0，若0也是合法普通字节，不能靠token!=0统计有效位；应靠mask。本次不下载百科或模型。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 清洗与启发式质量筛选

clean_text删除非ASCII，中文/emoji等会丢失！只能作为针对ASCII英文的示意，不是通用Unicode清洗。过滤器检查词数、大写词比例、特殊符号比例，代码/数学可能被误杀；PII和语言模型过滤并未实际实现。

~~~python
import re

# 原课仅保留ASCII，中文和emoji会被删，不能当通用清洗。
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

# 启发式阈值须按领域测误杀，不等于完整质量识别。
def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
~~~

### 4.2 MinHash近重复

get_shingles默认5词，短文本得空集；空签名全inf进入同桶但最终被跳过，所以短文本完全重复也不一定去掉。bands必须正且整除num_hashes；候选集合遍历不稳定可能影响保留对象，需排序/固定保留政策。

~~~python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

# 每个哈希种子取shingle集合最小值，空集会得到inf。
def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

# LSH先召回候选，再精确算Jaccard；短文本空shingle被跳过。
def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
~~~

### 4.3 tokenize并打包

tokenize_corpus把全部IDs装列表，并在每文档后追加eos_id，不是流式。pack_sequences按步长切分，末块补pad并给padding mask；seq_length必须>0。文档边界信息未保留，无法从此直接构造精确块对角mask。

~~~python
# 原版装入完整列表而非流式，需tokenizer.eos_id。
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

# 返回的是padding mask，不是块对角文档隔离mask。
def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
~~~

### 4.4 批量迭代器

shuffle洗的是序列编号，批次仍在内存，random状态未独立保存。最后一批可小于batch_size；需校验mask长度对齐、batch_size>0。分布式rank分片、worker、prefetch、重启恢复均未实现。

~~~python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
~~~

### 4.5 统计数据与利用率

本课compression_ratio=字符/token，与上一课token/字节方向相反，别直接比数值大小。non_pad_tokens硬编码t!=0会误把真实ID0当pad，应该累加mask；total_tokens=0与词表大小0也要校验。EOS计入总token，doc词数用split仅是空格协议。

~~~python
from collections import Counter

# 原版按ID!=0数利用率，仅在0专为pad时正确。
def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
~~~

## 五、把代码运行起来：补充入口与检查

补充一个无需下载模型的tokenizer接口和打包测试：
~~~python
class ToyTokenizer:
    eos_id = 2
    def encode(self, text):
        # 普通字符统一加3，避开本例pad=0/eos=2；仅用于接口教学。
        return [ord(c) + 3 for c in text]

toy = ToyTokenizer()
token_ids = tokenize_corpus(["ab", "c"], toy)
sequences, masks = pack_sequences(token_ids, seq_length=4)
assert token_ids == [100, 101, 2, 102, 2]
assert masks == [[1, 1, 1, 1], [1, 0, 0, 0]]
assert sum(map(sum, masks)) == 5
print(sequences, masks)

# 补充：参数检查，避免静默分片错误。
def checked_buckets(signature, bands):
    if bands <= 0 or not signature or len(signature) % bands:
        raise ValueError("bands必须为正且整除非空签名长度")
    return lsh_buckets(signature, bands)
~~~
真实ID0可用时，有效位置数应=sum(sum(mask) for mask in masks)，而不是查token值。source clean_text("你好")为空可直接验证，正是为什么Unicode清洗不能照搬。

## 六、Use It：使用现成工具

HF datasets.map使用Rust tokenizer及多进程，但源文truncation=True会丢每条超2048部分，并不等价于上述完整EOS打包，也不是streaming=True示例；必须统一“处理了哪些输入”后才能比较速度。多进程数量、磁盘缓存与启动开销都影响结果，不能保证固定10～50倍。

~~~python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
~~~

## 七、失败模式、原文边界与工程注意

1. 原文所谓TB流式目标并未由列表实现，训练前不可把所有数据硬塞内存。
2. 去重前统一正规化协议，训练/验证按文档与近重复簇划分，防止泄漏。
3. 0/1 padding mask不隔离文档，EOS也不是attention控制符。
4. 所有ASCII以外删除会破坏多语言和隐私字段位置，不可用于原文可逆溯源。
5. 数据源的公开不等于允许任意再分发/训练，PII检测和删除要专门做。
6. 高PPL不是垃圾充分条件；Hard练习“移除bottom20%”应明确为低质量侧，若按PPL数值排序通常应审查最高PPL而非最低。
7. 小样本过滤率和具体模型数据配比不应冒充行业标准。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
检查来源/许可、语言与领域配比、PII、去重阈值和污染；明确streaming/内存、tokenizer版本、EOS/PAD、packing与三种mask政策。报告过滤/保留率、tokens/s、mask利用率、空样本和可恢复状态，不把原课统计硬编码当生产正确性。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：加字符集语言启发式，统计移除文档；强调不是准确语言识别器，检查多语言/代码误杀。
2. Medium：SHA256精确去重+MinHash近重复，分别记录新增捕获、候选对、Jaccard与短文本；使用合法小语料，不需自动抓站。
3. Hard：以百科训练bigram质量模型，按明确定义的低质量20%移除；保留独立数据对比训练效果，检查高PPL领域误杀，不能只看过滤后训练loss。

## 十、术语与完成标准

shingle：连续词片段；Jaccard：集合重合比；MinHash：最小哈希签名；LSH：候选桶；packing：固定长度打包；padding/causal/document mask：有效位置/未来/文档边界控制；throughput：吞吐。

完成检查：

- 能用5个ID手算packing。
- 能解释LSH候选还要精确核验。
- 能指出原clean_text删中文。
- 能分清三种mask与利用率计算。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行打包：[[100,101,2,102],[2,0,0,0]]，mask为[[1,1,1,1],[1,0,0,0]]；长文重复删1条、两个short重复不删，参数非法时拒绝。未运行大语料流水线。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[02：构建完整分词器——Unicode、预切分、特殊Token与聊天模板]] · 下一课：[[04：从零搭建Mini GPT——因果注意力、前向计算与真实训练]]
