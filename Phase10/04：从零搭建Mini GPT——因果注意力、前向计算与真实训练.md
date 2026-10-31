# 04：从零搭建Mini GPT——因果注意力、前向计算与真实训练

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/04-pre-training-mini-gpt) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/04-pre-training-mini-gpt/docs/en.md)。原题：Pre-Training a Mini GPT (124M Parameters)。

## 零、初学者先读：一个小GPT究竟怎样猜下一个片段

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

先把语言模型看成反复做选择题的函数：输入已经出现的token，输出下一个token的概率。它不是一次吐出整篇文章，而是选一个、接到末尾，再算下一次。训练则让正确答案的概率逐渐变大。

### 0.2 本课术语先翻成白话

- embedding=按ID查到的可训练向量

- hidden=中间表示

- logit=尚未变成概率的分数

- softmax=把分数转换为和为1的正数

- attention=按当前需要从上下文加权取信息

- FFN=逐位置加工特征的网络。

### 0.3 用小数字走一遍

词表仅有猫、狗、。三个候选。某位置输出分数[ln2,0,0]，softmax后是[.5,.25,.25]。正确答案若为猫，loss=-ln(.5)≈.693。训练前预测正确答案概率.25时，loss≈1.386；把概率提高到.5才算这个样本的loss降低。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先追Embedding查表，再追TransformerBlock，再追输出层与cross_entropy_loss，最后才读generate。输入形状(B,T)里的B是样本条数、T是每条长度；输出(B,T,V)为每条、每个位置、每个词表候选各给一个分数。attention里的Q/K/V分别承担查询、被查询的特征、取回内容，都是向量而非原句文本。

前置阅读：[第01课](<01：分词器——BPE、WordPiece 与 SentencePiece.md>)；[第03课](<03：预训练数据流水线——清洗、MinHash去重、打包与批次.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 第一位置能看未来正确答案吗？

**参考答案：** 不能，所以用因果mask遮住未来。

**问题2：** 算了200次loss就学会了吗？

**参考答案：** 不一定，必须由正确梯度更新参数；原课前向示例缺少这一步，正文有具体纠错。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

本课目标是看清ID→embedding→Transformer→logits→下一token的完整路径。但必须先纠正原文：所给NumPy train_mini_gpt只做forward和loss，没有backward、梯度或参数更新，因此它不能训练模型；运行200步不代表学会语言。标题124M也是参考配置，不能用教学实现冒充GPT2精确复刻。

## 二、核心概念与推导

### 2.1 完整路径及形状
输入(B,T)→token查表(B,T,d)+位置表(T,d)→L个pre-norm block→最终LN→共享embedding转置→(B,T,V)。每block为x+Attn(LN(x))，再x+FFN(LN(x))。残差提供梯度捷径，不代表无限深也无优化问题。

### 2.2 注意力
Q=XWq、K=XWk、V=XWv；A=softmax(QKᵀ/√dh+mask)，输出AV。d必须能整除head数h，(B,T,d)拆(B,h,T,dh)。mask屏蔽j>i，不能把上三角设0后仍让softmax看未来；被屏蔽位置需足够负或−∞。

### 2.3 FFN、LN与共享
LN沿最后特征维求均值方差，gamma/beta可学习。FFN升维到4d再降维。原np.maximum(0,h)是ReLU，不是数学GELU近似函数。输入/输出embedding共享减少一张V×d表，仍需对应同一词表。

### 2.4 下一词监督与采样
序列[3,4,5,6]用输入[3,4,5]预测[4,5,6]。CE=−log正确词概率，均匀V=256时ln256≈5.545。真实训练需loss.backward/解析梯度→optimizer.step，随机窗口导致loss波动不证明学习。温度>0缩放logits；temperature=0需单独argmax分支，原函数直接除0会坏。

### 2.5 缓存与阶段
prefill批处理prompt、decode逐步生成。KV缓存减少重复K/V计算，但每步仍需对历史做注意力，不能笼统所有计算O(1)。GPT2式缓存每token约2Lhdh个数，单序列T还要乘T和dtype字节。prefill偏算力、decode偏带宽是常见情形，batch/长度/内核会改变瓶颈，不是硬定律。

### 2.6 参数表的边界
源码漏注意力bias、用ReLU且无反传，与GPT2不同。按本代码计数公式Vd+Td+L(12d²+9d)+2d，默认为124402944，而原文声称124438272不吻合。原课“几小时训练完整124M”“GPU毫秒数”和GPT4层数猜测不作为本机/已知架构事实。

### 2.9 补充算例与理解检查

B=1,T=3,d=8,h=2，则每头维4，Q/K/V为(1,2,3,4)，score为(1,2,3,3)，输出拼回(1,3,8)，词表256时logits=(1,3,256)。保持前两个token相同、只改未来第三token，前两个位置logits应不变，这比仅检查输出形状更能验证causal mask。

### 原课概念部分的公式与结构示例

~~~text
Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # causal mask: -inf for future positions
attention_weights = softmax(attention_scores)
output = attention_weights @ V
~~~

## 三、操作前准备

原代码只需NumPy，但本次未安装/执行该库。先用小配置，不要直接在CPU构造默认124M的float64矩阵。
~~~bash
python -m pip install numpy
~~~
numpy.random.randn默认float64，不是源文FP32/FP16显存预算。只看教学前向不代表利用了4060；GPU训练需要支持GPU的框架及真实反传。定义全部类后再运行入口。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 embedding查表

token_embed是(V,d)，pos_embed是(maxT,d)，tok_emb+pos_emb靠广播加位置。token ID需0≤id<V，T≤maxT。位置超界不是自动滑窗，要由生成器截断上下文。

~~~python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    # 沿本类描述检查形状，所有NumPy类都只有前向、没有自动反传。
    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb
~~~

### 4.2 单头注意力

原attention中K.transpose参数不适用于4维数组，3维时也可能重复轴。通用修正是np.swapaxes(K,-1,-2)。这段没有被后面MultiHeadAttention调用，所以可能示例整体跑时没暴露，但不能照抄作正确单头实现。softmax先减max防溢出。

~~~python
# 原transpose有维度错误，正文给出swapaxes修正。
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
~~~

### 4.3 多头注意力

reshape按最后维分头，transpose把头移到batch后。K.transpose(0,1,3,2)交换最后两轴得到每头QKᵀ，处理后逆序变回(B,T,d)。需显式断言embed_dim%num_heads==0。

~~~python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    # 沿本类描述检查形状，所有NumPy类都只有前向、没有自动反传。
    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out
~~~

### 4.4 pre-norm block

两个LN、注意力、FFN与两条残差。ReLU只为教学简化，无法据此声称GELU/训练行为相同；本类没有任何backward方法。

~~~python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    # 沿本类描述检查形状，所有NumPy类都只有前向、没有自动反传。
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    # 沿本类描述检查形状，所有NumPy类都只有前向、没有自动反传。
    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU approximation: ReLU for simplicity
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    # 沿本类描述检查形状，所有NumPy类都只有前向、没有自动反传。
    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x
~~~

### 4.5 组合模型与参数计数

逐层因果前向，最后x@token_embed.T。count_parameters按数组size统计共享表一次。默认配置参数量与源文表格不同，原因上文说明；训练前先用几十维排错。

~~~python
class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    # 沿本类描述检查形状，所有NumPy类都只有前向、没有自动反传。
    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total
~~~

### 4.6 所谓训练循环：实际不训练

train_mini_gpt构造随机模型、抽窗口、前向、打印loss后直接下轮。lr从未使用，参数不更新！所以“loss逐步下降”的叙述没有代码依据；不能为练习画出预期三段曲线并称已观测。

~~~python
# 稳定log-softmax后取真实下一token的负对数均值。
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


# 重要：原版仅forward+loss，lr没用，完全没有参数更新。
def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model
~~~

### 4.7 自回归生成

截最后maxT个输入，每步取最后位置logits采样并追加。这个版本没有KV cache、EOS终止、top-k/top-p；温度必须正且prompt非空。生成乱码是未训练/小数据均可能原因，本代码首先是未训练。

~~~python
# 原版temperature必须>0，未实现EOS/KV/top-k/p。
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens
~~~

## 五、把代码运行起来：补充入口与检查

先做前向/因果检查，不宣称学习。需要NumPy，本次未运行：
~~~python
np.random.seed(0)
model = MiniGPT(vocab_size=256, embed_dim=8, num_heads=2,
                num_layers=1, max_seq_len=8, ff_dim=32)
a = np.array([[3, 4, 5]])
b = np.array([[3, 4, 6]])
ya, yb = model.forward(a), model.forward(b)
assert ya.shape == (1, 3, 256)
assert np.allclose(ya[:, :2], yb[:, :2])
assert np.isfinite(cross_entropy_loss(ya, a))
print(ya.shape, model.count_parameters())
~~~
单头函数若要调用，把score那一行改为Q @ np.swapaxes(K,-1,-2) / np.sqrt(Q.shape[-1])。真实训练最低检查链：保存参数副本→计算loss→反传→step→确认参数改变→相同留出batch再次测loss；只显示参数变了仍不能证明梯度正确。后续SFT/RL练习都需要先有真正可训练模型。

## 六、Use It：使用现成工具

原文Full Training and Generation Demo会打印“Training”和生成文本，但它复用无更新循环，不能当预训练成功。可用它观察前向与采样。若要真实训练，应使用已验证的PyTorch实现或补齐所有层解析反传；不要只给FFN加噪声或只调整输出头后称整个Transformer已训练。原课建议Adam、warmup、梯度累积/裁剪是训练配置，不是源码已实现功能。

~~~python
corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")
~~~

## 七、失败模式、原文边界与工程注意

1. 原attention.transpose错误；完整多头另实现了正确的4轴交换。
2. lr未用、无backward、无optimizer，不会学。
3. NumPy float64和默认124M数组可占大量RAM，参数×2字节估算不适用。
4. temperature=0需argmax，负温度也应拒绝；空prompt无法取最后logit。
5. 生成窗口截断重置位置是简化，真实缓存与位置偏移要设计。
6. 没有训练/验证拆分，loss波动不能证明泛化；平台曲线也不能独立证明过拟合。
7. 没有EOS、top-k/p或KV实现，不能按Learning Objectives声称全完成。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
分析GPT配置与实现：列V/T/d/h/L/FFN、参数来源、pre-norm/残差、位置与causal mask、权重共享。核查是否真实反传并更新、精度、optimizer与数据；分开报告前向正确、训练拟合、留出loss和生成质量，不用标题参数/示意曲线冒充实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 改24层16头，对比深度/宽度参数；保证d整除h，用公式或小数组先算，避免内存爆。
2. 用erf形式GELU替换ReLU，先建立真正反传，再同数据同种子训练500步比较；原无更新循环不能完成该实验。
3. 加每层KV缓存，验证相同前缀logits一致后测生成200token延迟，分别记录prefill/decode。
4. 实现top-k=50与top-p=.95，温度.8，检查概率归一、边界和可复现，不把主观几个例子当评价。
5. 真训练1000步画train/validation loss，检查参数更新和梯度；不能预设必须有原文所说三阶段，更不能从曲线外观推断GPT4。

## 十、术语与完成标准

autoregressive：条件于历史预测下一token；causal mask：屏蔽未来；logits：未归一分数；weight tying：共享输入输出表；prefill/decode：prompt处理/逐步生成；CE：负对数正确概率。

完成检查：

- 能追踪(B,T,d)到(B,h,T,dh)。
- 能发现只算loss不更新的循环。
- 能验证改变未来不影响过去。
- 能区分原模型结构与GPT2精确实现。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

仅核对参数公式得到124402944；本环境没有NumPy，未运行attention/完整forward，更未训练模型。原代码无真实优化更新这一点由代码审查确认。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 7 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[03：预训练数据流水线——清洗、MinHash去重、打包与批次]] · 下一课：[[05：分布式训练——DP、TP、PP、FSDP与ZeRO预算]]
