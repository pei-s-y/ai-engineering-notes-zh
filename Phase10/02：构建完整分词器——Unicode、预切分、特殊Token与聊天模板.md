# 02：构建完整分词器——Unicode、预切分、特殊Token与聊天模板

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/02-building-a-tokenizer) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/02-building-a-tokenizer/docs/en.md)。原题：Building a Tokenizer from Scratch。

## 零、初学者先读：会做BPE后，还缺哪些环节

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

上一课像一台只会压合零件的机器。本课给它补进料规则：文字先统一形式，识别有特殊含义的标记，再划成小块，最后在块内做BPE。每一步都可能改变输入，因此先检查有没有丢文字，再谈模型效果。

### 0.2 本课术语先翻成白话

- Unicode=字符编号标准

- UTF-8=把字符编号编码为字节的方法

- normalize=按约定统一文字形式

- 正则表达式=描述哪些字符属于一类的匹配规则

- 聊天模板=把角色与消息组织成模型约定的序列。

### 0.3 用小数字走一遍

NFKC可以把全角Ａ变成A。假设输入是Ａ猫，规范化后是A猫；正确往返目标可能是A猫而不是原始Ａ猫。若预切分却只抓到A、漏掉猫，后面的BPE再完整也无法恢复猫。特殊标记如<|end|>则应按协议独占一个ID。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

按normalize→split_with_specials→pre_tokenize→apply_merge→ID的顺序追踪一个短输入。re.compile先生成规则，finditer找所有匹配；不要把这种规则字符串当普通句子阅读。ProductionTokenizer是类名，不自动代表代码已适合所有真实场景。

前置阅读：[第01课](<01：分词器——BPE、WordPiece 与 SentencePiece.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 规范化后往返正确，是否保证保留原样？

**参考答案：** 不保证，规范化本身可能改写字符。

**问题2：** 特殊token能和某个字节使用同一ID吗？

**参考答案：** 不能，否则解码或控制含义发生冲突。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

从“能合并字节”走向可组合的tokenizer流水线：正规化→识别特殊串→预切分→BPE→ID。原文称ProductionTokenizer不意味着它已达到生产安全/性能标准；尤其降级正则会丢中文、正规化改变原文，是本课必须理解的边界。

## 二、核心概念与推导

### 2.1 五个环节
NFKC做兼容分解后规范组合，如ﬁ→fi、全角Ａ→A，因此目标是decode(encode(s))=normalize(s)，不保证与原s逐字相同。代码、标识符和安全语境下是否允许这种改变要单独决定。可选小写/去重音不是所有任务都适用。

预切分规定BPE在哪些块内学习/应用，禁止跨块合并是一种设计，不是所有分词器都必须禁止跨词。GPT2正则中\p{L}/\p{N}需第三方regex；fallback只支持部分ASCII字符类别，不是Unicode等价替代。

### 2.2 特殊Token
固定ID、不可拆分，须与普通字节/合并ID避免冲突。源文先训练、再用next_id添加特殊串；不同添加顺序会改变ID，应保存完整协议。用户文字里的特殊标记是否允许当控制符，需要信任边界，不能无条件把输入字符串当系统角色。

### 2.3 聊天模板
消息list最后被格式化成序列，但具体模型格式不同。Llama3的begin_of_text、start_header_id/end_header_id、eot_id与泛用user/assistant标记不同；原文表格是混合示意，不能直接当实现。优先使用模型配套apply_chat_template，避免重复加BOS与EOS。

### 2.4 字节、速度和来源修正
字节BPE可覆盖普通有效Unicode编码，中文不一定全是3字节，某些扩展汉字是4字节。GPT2空间显示标记常见Ġ，不是原文写的普通G；它是字节到可打印符号的映射，不是算法语义。原文“Llama都无regex”与上一课Llama3路线矛盾，应按版本。

Rust/C++库可更快，但10～100倍、万亿token处理天数只是容量算例，不保证当前机器速度。

### 2.9 补充算例与理解检查

你好的UTF-8是6字节，🔥是4字节；ﬁ经NFKC成为fi，2字符而非原1字符。预切分先要求拼回chunks与正规化文本相同，再谈BPE；若中文没被正则捕获，后面256字节词表再完整也无法找回已经丢掉的数据。

### 原课概念部分的公式与结构示例

~~~text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
~~~

~~~text
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
~~~

~~~text
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
~~~

## 三、操作前准备

核心除完整Unicode正则外用标准库。建议安装regex后使用完整模式：
~~~bash
python -m pip install regex
~~~
若不装，请先看下一节覆盖检查，别把fallback例子当多语言成功。各代码块按顺序定义；get_byte_pairs只是计原始块字节对，真正训练在ProductionTokenizer内部每轮对已更新chunk_bytes重计。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 UTF-8编码与还原

bytes_to_tokens输入str，得到0～255整数；tokens_to_text默认replace非法编码，调试可改strict。英语5、你好6、🔥4仅是给定文本的可手算结果；接口并不接受任意二进制文件。

~~~python
# 字符串先UTF-8编码，再取得基础字节ID。
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

# replace会掩盖非法UTF-8，调试可用strict。
def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
~~~

~~~python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
~~~

### 4.2 Unicode预切分

完整regex覆盖字母、数字、标点和空白；标准re的fallback可能漏掉中文等属于\w但不是ASCII字母的部分。原注释Hello前面多一个空格，Don't大小写也不保证分成源文展示结果。必须打印真实结果并检查join。

~~~python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

# 必须检查拼回结果等于输入，源码ASCII降级可能漏中文。
def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
~~~

~~~python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
~~~

### 4.3 在各块内部合并

get_byte_pairs从字符串块重新编码，不能拿来统计已经包含新ID的序列。apply_merge操作ID列表，非重叠扫描；边界不会被跨过。训练和推理必须用相同预切分规则。

~~~python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

# 只在单块内合并，不越过预切分边界。
def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
~~~

### 4.4 特殊串识别器

add_token按长度倒序构造转义正则，避免前缀短串抢先；split_with_specials用(text,is_special)标记。未检查重复token串/ID、空特殊串与越权输入，生产需加校验。

~~~python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    # 保留原文先后顺序，标明哪些段是特殊串。
    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
~~~

### 4.5 完整类拼装

normalize先于特殊串切分，因此特殊串若含兼容字符可能变化。next_id递增创建词表；decode遇未知ID静默忽略，是数据损坏风险，应显式报错。训练并未先跳过special parts，所以“特殊串从不参与训练合并”并非源码完全兑现。

~~~python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    # NFKC可能改变原字符，round-trip应与正规化结果比较。
    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    # 原版会静默跳过未知ID，生产应改显式校验。
    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
~~~

### 4.6 多语言与特殊标记测试

英文语料50轮后加begin/end；能否还原中文取决于上一节regex是否完整覆盖。比较normalize(text)而非一概原字符串。示例未打印PASS/FAIL，应加断言。

~~~python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
~~~

## 五、把代码运行起来：补充入口与检查

补充覆盖检查，无第三方库时也能发现降级错误：
~~~python
samples = ["Hello, world! Don't stop.", "你好世界", "Hello 🌍 World", "ﬁ Ａ", "  x\t\ny"]
for text in samples:
    normalized = unicodedata.normalize("NFKC", text)
    chunks = pre_tokenize(normalized)
    print(repr(text), chunks)
    if "".join(chunks) != normalized:
        print("FAIL：预切分丢字符，先安装regex或修正fallback，不能继续宣称可逆")

# 补充一个确保覆盖的标准库fallback（行为不同于GPT2，需重新训练）：
SAFE_PATTERN = re.compile(r"\s+|\w+|[^\w\s]+", flags=re.UNICODE)
def safe_pre_tokenize(text):
    pieces = SAFE_PATTERN.findall(text)
    assert "".join(pieces) == text
    return pieces
~~~
要用fallback完整训练，可在新会话将pre_tokenize定义为safe_pre_tokenize，再新建ProductionTokenizer；不要对同一已训练模型切换规则。这个补充不冒充GPT2精确分词。新tokenizer训练后验证所有samples的decode等于NFKC结果，特殊串独占一个ID且不会碰撞。

## 六、Use It：使用现成工具

tiktoken与HF模型tokenizer可做对照，必须使用相同add_special_tokens约定；Llama/Mistral的授权、模板、EOS和前缀不同。比较同文token数不等于比较语言能力，也不能保证大词表每段都更短。本次不会下载受限tokenizer/模型。

~~~python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
~~~

~~~python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
~~~

## 七、失败模式、原文边界与工程注意

1. fallback会漏中文，覆盖测试优先于速度测试。
2. NFKC有损于原字形；不是任意任务都应该正规化。
3. decode静默跳未知ID，add_special_token重复调用也需定义行为。
4. 无条件识别用户提供的角色特殊串可能破坏协议隔离；数据与控制标记分开处理。
5. 重复train不清空已有规则，保存/加载也未实现；类名不能当生产完整性证明。
6. 原文空格映射、Llama泛称与示例首空格存在问题，正文已修正。
7. 聊天模板匹配错误会退化，但并非任何多一个空格都必然“垃圾输出”，要测试实际影响。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
设计tokenizer流水线，记录正规化、Unicode覆盖、pre-token规则、合并表、特殊token/ID、decoder与聊天模板。先测空串、多空格、代码、CJK、emoji、兼容字符及未知ID；保存配置/hash并与模型匹配，性能另测，不仅检查单句英语。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：加get_token_bytes(id)，对无效ID报错，打印常见合并片段与单字节，注意单个片段可能UTF-8不完整。
2. Medium：按原练习实现保留前空格的词/数字分割，与GPT2正则对比。注明这只是自定“Llama-style”教学规则，不等于Llama3真实tokenizer。
3. Hard：为role/content消息实现Llama3格式，和配套HF apply_chat_template比token IDs；核查BOS、eot、换行、add_generation_prompt，不把聊天JSON直接交模型。

## 十、术语与完成标准

NFKC：兼容正规化；pre-token：BPE前的块；special token：结构控制标记；chat template：消息序列协议；byte fallback：最小字节覆盖；fertility：每词token数。

完成检查：

- 能区分可逆原文与可逆正规化结果。
- 能发现fallback漏中文。
- 能验证特殊ID独立且有效。
- 能解释聊天模板为何必须配套。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已复现无regex时ASCII fallback丢中文（预切分结果[]）；新建tokenizer改用补充safe fallback后，NFKC文本往返、空串、空白和独占特殊ID通过。没有运行tiktoken/AutoTokenizer比较。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 6 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[01：分词器——BPE、WordPiece 与 SentencePiece]] · 下一课：[[03：预训练数据流水线——清洗、MinHash去重、打包与批次]]
