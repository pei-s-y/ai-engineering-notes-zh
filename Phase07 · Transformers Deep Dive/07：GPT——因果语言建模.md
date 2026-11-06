# 07：GPT——因果语言建模

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F07-gpt-causal-language-modeling) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md)。原题：GPT — Causal Language Modeling。

## 零、初学者先读：只看过去，一个字一个字往下写

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第06课](<06：BERT——掩码语言建模.md>)的 BERT 双向看句、填中间的空；本课的 GPT 只能看左边、从左往右一次一个 token 地往外写——两课是同一副骨架的两种“注意力可见性”，对照着学最快。

### 0.1 先理解这一课在做什么

语言模型只回答一个问题：给定前 t−1 个 token，第 t 个 token 的概率分布是什么。用这个信号（下一词预测，next-token prediction）训练，就得到一个能一次一个 token 生成任意文本的模型。要在整条序列上端到端并行训练，你必须保证每个位置的预测只依赖更早的位置——否则模型会直接偷看答案。因果掩码（causal mask）就是干这个的：一个加在 softmax 之前注意力打分上的上三角 −∞ 矩阵。原课的说法很煽情但基本准确：这是现代 AI 里影响最深的一行代码。GPT-1（2018）到 GPT-5（2025）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——全是解码器-only 的因果 Transformer，同一个核心循环；区别在数据质量、规模、架构微调和后训练（SFT、RLHF、DPO 及后继者）。

### 0.2 本课术语先翻成白话

- 因果掩码（causal mask）=上三角全 −∞ 的矩阵，加到注意力打分上，让位置 i 只能看到 ≤i 的位置
- 下一词预测（next-token prediction）=每个位置对“下一个 token”算交叉熵的训练损失
- 自回归（autoregressive）=把自己的输出接回输入继续生成；训练时并行、生成时串行
- logits=softmax 之前的原始打分；采样都在这上面做文章
- 温度（temperature）=logits 除以 T 再 softmax：T→0 就是贪心，T 大分布变平
- top-k / top-p（nucleus，核采样）=只在前 k 个 / 累积概率达到 p 的最小集合里采样
- min-p=只保留概率 ≥ min_p × 最大概率 的 token，2024+ 的默认候选
- 投机解码（speculative decoding）=小模型草拟 N 个 token、大模型并行验证，同质量下 2—3 倍加速
- 束搜索（beam search）=每步保留前 k 条完整候选序列（练习 2 出现）
- teacher forcing（教师强制）=训练时喂真实的上一个 token，不是模型自己的预测
- KV 缓存（KV cache）=生成时缓存已算过的 K/V，避免重复计算（第12课专题）
- 困惑度（perplexity, ppl）=对数损失的指数，越低越好（解码策略对比的度量）

### 0.3 用小数字走一遍

**第一步：看清这张三角形。** N=5 时因果掩码长这样（0=可见，−∞=屏蔽）：

```
        j=0   j=1   j=2   j=3   j=4
i=0  [   0,  -inf,  -inf,  -inf,  -inf ]
i=1  [   0,    0,   -inf,  -inf,  -inf ]
i=2  [   0,    0,     0,   -inf,  -inf ]
i=3  [   0,    0,     0,     0,   -inf ]
i=4  [   0,    0,     0,     0,     0  ]
```

每行是“第 i 个位置能看到谁”：第 3 行只有 j=0—3 有 0。−∞ 加到打分上再过 softmax，e^−∞=0，屏蔽位权重精确为 0。

**第二步：验证 softmax 真的把右边清零。** 拿 2 个 token 的打分矩阵 [[1,2],[3,4]]，加掩码后第 0 行是 [1, −∞]：softmax 给出 [1, 0]；第 1 行 [3,4] 不受影响：e³/(e³+e⁴) = 1/(1+e) ≈ 0.269、4/(3+4) 的比例即 ≈ 0.731。结果 [[1,0],[0.269,0.731]]——下三角、每行和为 1。

**第三步：三角形的“祖先”是前缀平均。** 序列 X=[2,4,6]（3 个标量），最朴素的因果摘要 = 每个位置取前缀均值：out[0]=2，out[1]=(2+4)/2=3，out[2]=(2+4+6)/3=4。写成矩阵：A = [[1,0,0],[0.5,0.5,0],[1/3,1/3,1/3]]，out = A·X = [2,3,4]。对角线以上的 0 就是因果性——不是“把未来遮住”，是求和范围里根本没有未来。

**第四步：用 “The cat sat on the” 预测下一个词。** 词表 {The, cat, sat, on, the, mat}。模型在最后一个位置输出 logits [0.1, 0.1, 0.1, 0.1, 0.1, 2.4]（mat 的分最高）。softmax 手算：e^2.4≈11.02，e^0.1≈1.11（五个），总和 = 11.02+5×1.11 ≈ 16.55。P(mat) = 11.02/16.55 ≈ **0.666**，其余每个 ≈ 0.0668（六项加起来恰好 1）。这一步的损失 = −ln(0.666) ≈ **0.406**。

**第五步：温度把分布摊平。** 同样 logits 除以 T=2 再 softmax：变成 [0.05×5, 1.2]，e^1.2≈3.32、e^0.05≈1.05，P(mat) = 3.32/(3.32+5×1.05) ≈ **0.387**，其余每个 ≈ 0.123。分布从“六四开”变成“四六开”——温度越高越接近均匀随机。

**第六步：shift-by-one（错位一位）。** 训练时输入和标签错开一格：输入 [The, cat, sat, on, the]，标签 [cat, sat, on, the, mat]。每个位置算一个 −log P(标签|前缀)，整句加总。一次前向 = 五道“预测下一词”的题同时作答，这就是并行训练的秘密。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

按顺序读：先看第四节的 `causal_mask`（0.3 第一步的一行代码版）；再看概念节的两段 numpy 代码（前缀平均 → 学到的权重 → 内容相关权重，对应 0.3 第三步的推广）；然后是 Steps 2—4 的文字描述（2 层解码器、20 词玩具词表上算 shift-by-one 交叉熵、实现五种采样）。Use It 是 HF `generate()` 的真实用法。核心记忆点：**掩码加在 softmax 之前的打分上；训练并行、推理串行；损失错位一位**。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么掩码要加在 softmax **之前**、用 −∞，而不是把 softmax 之后的权重直接置 0？

**参考答案：** 两种做法最终权重都是 0，但先置 0 再归一化会改变剩余权重的相对比例——softmax 分母里少了一项，等于给没被遮的位置重新分配概率；用 −∞ 加在打分上则等价于“e^−∞=0 自然从分母里消失”，一步到位且实现上只是一次矩阵加法。原课三阶段推导还给了更深的理由：−∞ 就是第一阶段里“求和范围之外的 0”翻译到 softmax 输入域的写法。

**问题2：** 训练能并行、推理为什么必须串行？

**参考答案：** 训练时整条序列已知，N 个位置的“下一词”题可以一次前向同时作答（teacher forcing：每个位置喂真实的前文）。生成时第 t+1 个 token 依赖第 t 个 token 的生成结果，输出多长就得串行多少步——原课称之为“自回归税”，也是所有 LLM 延迟的瓶颈。KV 缓存（第12课）只能省重复计算，省不掉串行本身。

**问题3：** 温度设成 0 会发生什么？

**参考答案：** 数学上除以 0 未定义；工程上“T→0 等价于贪心 argmax”，所以实现里 T=0 直接走 argmax 分支。温度是“创造力旋钮”：0 确定性最强，越大越接近均匀随机。

**问题4：** BERT 和 GPT 一句话的区别？

**参考答案：** 原课原话：GPT 预测 P(x_t | x_{<t})，BERT 预测 P(x_masked | x_unmasked)。损失决定模型能不能生成——GPT 的训练形态和推理形态一致（接着往下写），所以能生成；BERT 不一致，所以不能。

能默画出 5×5 三角形并算完上面六步后，进入正文。

---

## 一、这一课解决什么问题？

语言模型的任务定义（预测下一词分布）本身很简单，难点有二：一是怎么在整条序列上**并行**训练而不泄露未来信息——因果掩码解决；二是生成阶段天然串行——这是延 迟瓶颈，衍生出 KV 缓存（第12课）、投机解码（[第16课](<16：投机解码——草拟、验证、重复.md>)）等一整套优化。本课把因果掩码的机制、三阶段推导、shift-by-one 损失和采样策略一次讲清。原课的历史叙事：GPT-1（2018）到 GPT-5（2025）、Claude、Llama 全是同一核心循环，分野发生在数据、规模与后训练。

## 二、核心概念与推导

### 2.1 掩码的数学定义

对长度 N 的序列，构造 N×N 矩阵：M_ij = 0（若 j ≤ i），−∞（若 j > i）。把 M 加到原始注意力打分上再 softmax：e^−∞ = 0，屏蔽位贡献零权重，注意力矩阵每一行变成“只在过去位置上的概率分布”。实现成本：一次 `torch.tril()` 调用；计算耗时：纳秒级；对整个领域的影响：一切（原课原话的意译）。

### 2.2 三角形从哪来：注意力是前缀平均的第三次改良

原课这段推导是全课最精彩的部分，建议逐行读代码。通常因果掩码被讲成“焊在注意力上的补丁”；反过来推导，它就不神秘了：注意力是**前缀平均**的第三次改良，三角形只是那个平均的循环边界写成了矩阵。

**阶段一：前缀平均。** 最笨的因果摘要：位置 i 变成 0…i 的均值。一个下三角全 1 矩阵、每行除以个数，一次矩阵乘法完成（原课代码，逐字转录）：

~~~python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
~~~

A 的第 i 行是 [1/(i+1), …, 1/(i+1), 0, …, 0]。对角线以上的 0 就是因果性——不是未来被遮住了，是求和里从来没有未来（0.3 第三步手算过 [2,4,6]→[2,3,4]）。

**阶段二：学到的权重。** 均值把每个过去的 token 一视同仁；把 1 换成学出来的打分矩阵 S。行不再天然和为 1，改用 softmax 归一。但 softmax 不输出精确的 0，因果性破功——除非未来打分以 −∞ 进场（原课代码，逐字转录）：

~~~python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
~~~

同一个三角形、同样的行随机矩阵、同样一次矩阵乘法。−∞ 掩码不是新机器，是阶段一的 0 翻译到 softmax 输入域。

**阶段三：内容相关的权重。** 阶段二里 S 训练完就固定了：位置 7 永远以同一个权重看位置 3，不管 token 说了什么。让打分依赖 token 本身：S = Q·Kᵀ/√d_k。其他一切不变——掩码、softmax、矩阵乘，全部照旧。

三个阶段一个不变量：**下三角行随机矩阵 × 序列**。均匀平均 → 静态学习权重 → 内容相关权重。掩码从来不是加到注意力上的，它是从前缀平均里存活下来的。

### 2.3 并行训练，串行推理

训练：整条 (N, d_model) 序列一次前向，算 N 个交叉熵损失（每位置一个），求和、反向传播——沿序列维并行，这就是 GPT 训练可扩展的原因：一个 GPU pass 处理一个 batch 里的 1M token。推理：逐 token 生成——喂 [t1,t2,t3] 得 t4，喂 [t1,t2,t3,t4] 得 t5……KV 缓存（第12课）保存 t1…tn 的隐状态免得重算，但推理的串行深度 = 输出长度。这是自回归税，也是每个 LLM 解码延迟的瓶颈。

### 2.4 损失：shift-by-one

给定 [t1,t2,t3,t4]：输入 [t1,t2,t3]，目标 [t2,t3,t4]。每个位置 i 算 −log P(target_i | inputs[:i+1])，求和即整句交叉熵。你听说过的每个 Transformer 语言模型都在这个损失上训练：预训练、微调、SFT——同一损失，不同数据（原课原话）。0.3 第六步的 6 词例子就是它的最小版。

### 2.5 解码策略

训练完之后，采样的选择比多数人以为的更重要（原课对照表）：

| 方法 | 做什么 | 何时用 |
|--------|--------------|----------------|
| Greedy（贪心） | 每步取 argmax | 确定性任务、代码补全 |
| Temperature（温度） | logits 除以 T 再采样 | 创意任务，T 高多样性高 |
| Top-k | 只在前 k 个 token 里采样 | 砍掉低概率长尾 |
| Top-p（nucleus） | 在累积概率 ≥ p 的最小集合里采样 | 2020+ 默认；适应分布形状 |
| Min-p | 只留 p > min_p × max_p 的 token | 2024+；比 top-p 更会拒绝长尾 |
| 投机解码 | 草稿模型提 N 个、大模型验证 | 同质量 2—3 倍降延迟 |

2026 年原课的建议：开源权重模型用 min-p + 温度 0.7 是合理默认；投机解码是生产推理栈的标配。0.3 第五步手算了温度的效果。

### 2.6 “GPT 配方”为什么灵

1. **解码器-only**：没有编码器开销，每层一遍注意力 + FFN。
2. **规模**：124M → 1.5B → 175B → 万亿级；Chinchilla 缩放定律（[第13课](<13：缩放定律——算力、数据与参数的预算.md>)）告诉你怎么花钱。
3. **上下文学习（in-context learning）**：约 6B—13B 涌现（原课参考）；模型能照 few-shot 示例做题，不用微调。
4. **RLHF**：人类偏好后训练把原始预训练文本变成聊天助手。
5. **pre-norm + RoPE + SwiGLU**：大规模训练稳定。

原课的判断：GPT-2 以来核心架构没大变，有趣的事都发生在数据、规模和后训练里。

### 2.7 一图串起本课（补充）

把 2.1—2.6 压缩成一段“从一句训练语料到一次生成”的完整流程：语料 `[The, cat, sat, on, the, mat]` → 右移一位得到标签 → 嵌入 + 位置编码 → 每个解码器块里“打分 → 加因果掩码 → softmax → 加权”→ 末位置（训练时是全部位置）得到 logits → 与标签算交叉熵 → 反向传播更新参数。生成时同一套前向，只是每轮只关心最后一个位置的 logits，采样、拼接、再来一遍。训练和推理用的是同一个函数，差别只在“关心哪些位置”和“下一个输入从哪来”——这就是 2.3 说的形态一致性，也是 GPT 能生成、BERT 不能的根本原因。

## 三、操作前准备

概念推导用 numpy（2.2 的两段代码）；Build It 是纯 Python 一行掩码 + 文字描述的模型组装；Use It 需要 torch + transformers：

~~~bash
python -m pip install numpy torch transformers
~~~

（核对/说明：课程页 Build It 只内嵌 `causal_mask` 一个函数；Step 2 的 2 层模型、Step 3 的玩具词表前向、Step 4 的五种采样只有文字描述，代码在 `code/main.py` 未内嵌；本篇按页面实际内容整理，补充代码均显式标注。）

## 四、Build It：原课分步代码与中文讲解

### 4.1 第 1 步：因果掩码

原课代码（逐字转录，仅加中文注释）——一行列表推导：

~~~python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
    # 中文注释：i 行 j 列：j<=i 可见（0），j>i 屏蔽（-inf）。
    # 加到注意力打分上、softmax 之前——这就是全部机制。
~~~

### 4.2 第 2 步：2 层“GPT 味”模型

原课描述：堆两个解码器块（掩码自注意力 + FFN，无交叉注意力——纯解码器没有第05课那个中间子层），加 token 嵌入、位置编码、unembedding（与 token 嵌入矩阵共享权重——GPT-2 以来的标准技巧，叫 weight tying：嵌入表既是“词→向量”也是“向量→词”的查表，省一份 V×d 参数）。

### 4.3 第 3 步：端到端下一词预测

原课描述：20 token 玩具词表上，每个位置产出 logits，对 shift-by-one 的标签算交叉熵。无梯度——这是前向 sanity check（与第06课同样的克制：不假装训练）。

### 4.4 第 4 步：采样

原课描述：实现 greedy、temperature、top-k、top-p、min-p，各自跑同一个固定 prompt 对比输出。原课原话：“一个采样函数 10 行。”（补充）这五种里 top-p 和 min-p 最容易写错截止条件，第五节给了可运行的对比例子。

## 五、把代码运行起来：补充入口与检查

（补充）课程页只内嵌了 `causal_mask`。下面自写最小检查，把 0.3 的六步手算逐项对上（**非原课代码**）：

~~~python
# 本块中文说明：（补充）自写的因果掩码与采样检查，非原课代码。
import numpy as np
import math

def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]

def softmax_rows(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# 检查1：5x5 掩码形状（0.3 第一步）。
M = np.array(causal_mask(5))
assert M[3][3] == 0.0 and M[3][4] == float("-inf")
assert M[0][1] == float("-inf") and M[4][0] == 0.0

# 检查2：softmax 后严格下三角、每行和为 1（练习 1 的内容）。
S = np.array([[1.0, 2.0], [3.0, 4.0]]) + np.array(causal_mask(2))
A = softmax_rows(S)
assert np.allclose(A, [[1.0, 0.0], [0.2689, 0.7311]], atol=1e-4)
assert np.allclose(A.sum(axis=1), 1.0)

# 检查3：前缀平均（0.3 第三步）。
X = np.array([[2.0], [4.0], [6.0]])
Ap = np.tril(np.ones((3, 3)))
Ap = Ap / Ap.sum(axis=1, keepdims=True)
assert np.allclose(Ap @ X, [[2.0], [3.0], [4.0]])

# 检查4：下一词分布与温度（0.3 第四、五步）。
logits = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 2.4])
p = np.exp(logits) / np.exp(logits).sum()
assert abs(p[5] - 0.6661) < 5e-4 and abs(-math.log(p[5]) - 0.4063) < 5e-4
logits2 = logits / 2.0
p2 = np.exp(logits2) / np.exp(logits2).sum()
assert abs(p2[5] - 0.3871) < 5e-4

# 检查5：（补充）min-p 与 top-p 的截止对比——同一组 logits 上的两种截法。
probs = sorted(p, reverse=True)
top_p_set, acc = [], 0.0
for v in probs:
    top_p_set.append(v)
    acc += v
    if acc >= 0.9:
        break
min_p_set = [v for v in probs if v >= 0.1 * probs[0]]
print("top-p(0.9) 保留数:", len(top_p_set), " min-p(0.1) 保留数:", len(min_p_set))
~~~

检查要点：掩码之后注意力矩阵“softmax 后下三角、行和为 1”是黄金不变量；温度只改变分布平坦度、不改变排名。

## 六、Use It：使用现成工具

原课代码（逐字转录，仅加中文注释），2026 年的 PyTorch 惯用法：

~~~python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
~~~

引擎盖下：`generate()` 跑一次前向，取最后位置的 logits，采样下一个 token，追加，重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）实现的是同一个循环，外加重度优化——批量 prefill、连续批处理、KV 缓存分页、投机解码。

**GPT 对 BERT，各一句话**：GPT 预测 P(x_t | x_{<t})；BERT 预测 P(x_masked | x_unmasked)。损失决定模型能不能生成。

## 七、失败模式、原文边界与工程注意

1. 忘加掩码训练语言模型：损失迅速塌到接近 0——模型学会了“直接看答案位”，不是学会了语言。这是本课第一失败模式，且症状是“好得可疑”。
2. 在 softmax **之后**置零：等价于对剩余位置重新归一化，看似可行实则改变了模型语义；正确做法是打分上加 −∞（0.5 问题1 的结论）。
3. 温度 T=0 直接除零：实现上要特判走 argmax；同样，logits 全相等时 top-k=1 与采样无差别，别把边界情况混进主路径（补充提醒）。
4. 束搜索不是万能：练习 2 的答案——翻译这类窄分布任务常赢，开放式聊天常输（原课提示“usually for translation, not for open-ended chat”）。
5. 自回归税：输出多长就串行多少步，解码是每个 LLM 的延迟瓶颈；KV 缓存省的是重复计算不是串行（原课反复强调，专题在第12课）。
6. shift-by-one 错位对不齐：输入多一位或少一位，标签全错、损失异常大——检查方法是把输入和标签按 0.3 第六步那样并排打印（补充）。
7. 原文边界：“影响最深的一行代码”是原课的修辞判断；上下文学习“6B—13B 涌现”、“min-p + 0.7 是合理默认”、“投机解码 2—3 倍”都是原课参考，非本机实测；Build It Steps 2—4 的代码课程页未内嵌，本篇如实标注、未虚构。
8. weight tying 的边界：共享嵌入与 unembedding 省参数且通常无害，但词表极大时（如 128K）有的实现选择不共享（第05课参数账里 untied 的例子），不是铁律（补充）。
9. 采样参数不是免费旋钮（补充）：同一模型换温度/top-p 可以让输出质量天差地别；上线前要在真实 prompt 集上做 A/B，而不是凭感觉抄 0.7/0.9。原课 Ship It 技能（第八节模板）的存在本身就在提示这件事的工程分量。
10. 教师强制的隐含缺口（补充视角）：训练时模型永远见到“完美的前文”（真实 token），推理时见到的是“自己生成的前文”，一旦早期生成走偏，后续条件分布进入训练从未覆盖的区域——原课没展开此点（teacher forcing 只在术语表出现），此为通用文献常识，供延伸阅读时留意，不作为本课结论。

## 八、Ship It：本课可复用的中文检查模板

原课产出 `outputs/skill-sampling-tuner.md`（为新生成任务选采样参数、标记何时必须确定性解码）。中文学习版自查模板（直接保存在本篇，不另建文件）：

~~~markdown
为一个生成任务选采样参数：任务要确定性还是多样性（代码/抽取类必须 greedy 或 T→0；创意类才放开）？温度取多少、是否需要 top-p/min-p 截断长尾、截断值依据是什么？输出长度上限与停止条件？是否需要 seed 固定以便复现？若延迟敏感，是否该上投机解码（草稿模型怎么选）？每个参数给出：取值、理由、超出默认值的风险。所有引用的效果数字标注来源与是否实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：运行 `code/main.py`，验证 softmax 后的因果注意力矩阵是下三角。验收标准：抽查第 3 行——权重只出现在第 0—3 列且和为 1（可直接跑第五节检查 2 的代码核对 [[1,0],[0.269,0.731]] 一例）。
2. 原课 Medium：实现宽度 4 的束搜索，在 10 个短 prompt 上比较束-4 与贪心的困惑度。验收标准：能回答“束搜索总是赢吗”——通常翻译赢、开放式聊天输，并能解释原因（束搜索优化的是整体似然，不优化人类偏好）。
3. 原课 Hard：实现投机解码：2 层小模型当草稿、6 层模型当验证器，在 100 条长度 64 的补全上测墙钟加速比。验收标准：输出与验证器的 greedy 完全一致（接受-拒绝机制保证），加速比 >1 且能分解时间花在哪。

## 十、术语与完成标准

因果掩码（causal mask）：上三角 −∞、位置 i 只见 ≤i；下一词预测：每个位置对真实下一 token 的交叉熵；自回归（autoregressive）：输出接回输入；logits：softmax 前的原始分；温度：logits ÷ T，T→0 贪心、T→∞ 均匀；top-p/nucleus：累积概率 ≥ p 的最小集合内采样；min-p：保留 p ≥ min_p×max_p；投机解码：草稿+验证；teacher forcing：训练喂真实前文。

完成检查：

- 能默写 5×5 因果掩码并解释 −∞ 为什么加在 softmax 之前。
- 能手算 0.3 的六步（三角形、softmax 清零、前缀平均 [2,3,4]、P(mat)≈0.666、T=2 摊平、shift-by-one）。
- 能复述三阶段推导：前缀平均 → 学习权重 → 内容相关权重，不变量是“下三角行随机矩阵 × 序列”。
- 能说清训练并行/推理串行的原因和“自回归税”。
- 能给出六种解码策略的适用场景。
- 能背出 GPT 配方五条并说出哪几条在架构里、哪几条在架构外。
- 能把 0.3 六步对应到 2.2 三阶段：掩码三角形（第一步）、前缀平均（第三步）、内容相关权重（第四步的 Q·Kᵀ 打分）。
- 能说出 teacher forcing 是什么、以及它为什么让训练能并行。

## 十一、验证范围与来源许可

课程页抓取成功（2026-11-06）；原课文档 raw.githubusercontent 与 github.com 页均不可达（502/504），与此前阶段相同，以课程页为准整理。原课页内嵌代码四段——2.2 的两段 numpy 推导、4.1 的 `causal_mask`、Use It 的 `generate()`——均已逐字转录（仅加中文注释，未改逻辑；2.2 阶段二代码里的 `S = S + ...` 依赖外部传入 S，原样保留）。0.3 节全部手算数字（5×5 掩码、softmax [[1,0],[0.269,0.731]]、前缀平均 [2,3,4]、P(mat)=0.6661/损失 0.4063、T=2 时 0.3871）已于本机 Python 3.12 + numpy 2.5 实际执行复核通过；第五节补充代码通过 `ast.parse` 静态检查并实际执行通过。未执行任何训练、未下载模型。标“原课参考”的数值均来自课程页原文。

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
