# 12：KV缓存与Flash Attention——推理优化

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F12-kv-cache-flash-attention) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention/docs/en.md)。原题：KV Cache, Flash Attention & Inference Optimization。

## 零、初学者先读：让每个新 token 只算一次，而不是重算整段历史

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第02课](<02：从零实现自注意力——Q、K、V与缩放点积.md>) 讲了注意力怎么算，[第07课](<07：GPT——因果语言建模.md>) 讲了怎么一个 token 一个 token 地生成；本课问的是：生成的时候，怎么不算废话、不搬废数据？这是推理侧的两大优化，与 [第16课](<16：投机解码——草拟、验证、重复.md>) 的投机解码共同构成"让大模型跑得动"的全部基础。

### 0.1 先理解这一课在做什么

训练和推理的瓶颈完全不同：训练是并行的、卡在算力（FLOP-bound）上；推理是串行的、卡在读写带宽（memory-bound）上。朴素的生成循环里，每写一个新 token 都要把之前所有 token 的 K、V 向量**重新算一遍**——第 1 步算 1 份、第 2 步算 2 份……写到第 N 个 token 共算了 1+2+…+N = N(N+1)/2 份，绝大多数是重复劳动。KV 缓存（KV cache）的思路朴素到近乎废话：算过的 K、V 存起来，新 token 只算自己的那一份。Flash Attention 解决的是另一个问题：标准注意力要把 N×N 的注意力矩阵在显存（HBM）里搬进搬出三个来回，而片上高速缓存（SRAM）比 HBM 快 10 倍——那就把注意力切块（tile）算，全程不落地。原课的口号：**训练并行且吃算力，推理串行且吃带宽。瓶颈不同，药方不同。**

### 0.2 本课术语先翻成白话

- 自回归解码（autoregressive decoding）=每次用全部已生成 token 预测下一个，逐 token 串行
- KV 缓存（KV cache）=把每个前缀 token 的 K、V 向量存起来，新 token 只当"提问方"
- 预填充（prefill）=把整段提示词一次并行算完、填满缓存的过程；与逐 token 的解码（decode）阶段相对
- HBM（High Bandwidth Memory）=显卡主显存，H100 上 80 GB、约 3 TB/s
- SRAM=片上高速缓存，每个 SM 约 256 KB、约 30 TB/s——快但小
- 分块/平铺（tiling）=把大矩阵切成 128×128 左右的小块逐块算，工作集塞进 SRAM
- 在线 softmax（online/running softmax）=分块累加时维护"当前最大值 m 与当前和 s"，最后一步归一化，结果与一次性 softmax 完全一致
- GQA（Grouped-Query Attention）=多个查询头共享一组 KV 头，缓存直接缩小数倍
- MLA（Multi-head Latent Attention）=DeepSeek 系对 KV 的进一步压缩
- 前缀缓存（prefix caching）=多请求共享同一段提示词（系统提示、少样本例子）的 KV
- PagedAttention=把 KV 缓存按 16-token 块分页管理，像操作系统的虚拟内存
- 连续批处理（continuous batching）=完成的请求立刻换出、新请求立刻换入，不等整批
- 投机解码（speculative decoding）=小模型起草、大模型一次验证多个——详见 [第16课](<16：投机解码——草拟、验证、重复.md>)
- memory-bound / FLOP-bound=瓶颈在数据搬运 / 瓶颈在乘加计算

### 0.3 用小数字走一遍

**第一问：一个迷你 KV 缓存有多大？** 设 2 层、2 头、每头维度 d_head=4、批大小 1、fp16（每数 2 字节）：

- 每 token 每层缓存 = 2（K 和 V）× 2 头 × 4 维 × 2 字节 = **32 字节**。
- 2 层合计：每 token **64 字节**。
- 生成到第 10 个 token 时：10 × 64 = **640 字节**；按浮点数数是 2×2×2×10×4 = **320 个数**（K 存 160 个、V 存 160 个）。

用原课的公式验算：`bytes_per_token_per_layer = 2 * n_kv_heads * d_head * dtype_size` = 2×2×4×2 = 32 ✓。

**第二问：缓存到底省了什么？** 生成 10 个 token（1 个提示 token + 逐个生成 9 个，即第 t 步注意力看 t 个 token）：

- 两种方案的 q·k **比较次数相同**：1+2+…+10 = **55 次**——缓存不是省比较。
- 省的是 **K/V 投影（token 状态计算）**：无缓存时第 t 步要给全部 t 个 token 重算 K、V，共 55 份 token 状态；有缓存时每个 token 只算一次自己的，共 **10 份**。原课的放大版：100 个 token 的生成，无缓存 5,050 份对有缓存 100 份。

**第三问：在线 softmax 的分块合并。** q=[1,0]，三个键的分数是 [1, 0, 2]（对应值向量 V 按行）。一次性 softmax：m=2，指数 = [e^−1, e^−2, 1] = [0.368, 0.135, 1]，和 = 1.503。分块算（前两键一块、第三键一块）：

- 块 1：分数 [1, 0]，块内最大 m₁=1，未归一化累加 o₁ = 1·V₁ + 0.368·V₂，s₁ = 1.368。
- 块 2：分数 [2]，全局最大变 m₂=2。旧累加整体乘 e^(1−2)=0.368 缩放，加上新贡献 1·V₃；s = 1.368×0.368 + 1 = 1.503 ✓ 与一次性完全一致。

归一化后输出也逐位一致（本机已验证，见第十一节）。这就是 Flash Attention "不是近似、是精确" 的全部秘密：**缩放因子 e^(m_old − m_new) 随时把旧的局部和折算到新的全局最大值上**。

**第四问：外推到真实模型。** 7B 模型（32 层、32 头、d_head=128、fp16）：每 token 每层 2×32×128×2 = 16,384 字节（16 KiB）；每 token 512 KiB；32K 上下文 = **16 GiB**。Llama 3 70B（80 层、GQA 8 个 KV 头）：每 token 每层 4 KiB，32K 上下文 = **10 GiB**（原课数字，本机按公式复核一致）。10 GiB 在 128K 上下文涨到 40 GiB——一张 40 GB A100 批大小 1 就装满了。GQA 是缓存的大赢家：64 个 KV 头的 MHA 在 32K 要约 80 GiB，是 GQA 的 8 倍。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

原课 Build It 三个部分按序读：`KVCache` 类（两个嵌套列表，K、V 各一层一头一条）对应 0.3 第一问的"320 个数"；`tiled_softmax_dot` 对应第三问的在线 softmax（它的 m、s、out 三个变量就是手算里的"当前最大、当前和、未归一化输出"）；最后的 naive vs cached 对比对应第二问——两解码器都做 55 次比较，差别在 token 状态计算次数。原代码只有 `tiled_softmax_dot` 完整展示在页面上（`KVCache` 也有完整代码），第三节准备好 `import math` 即可跑。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么缓存的是 K、V 而不是 Q？

**参考答案：** 因果注意力里，历史 token 的任务是"被看"：它们的 K、V 会被后面每个新 token 反复查询。而 Q 属于"看别人的"那一方——每个新 token 只需要自己的 Q，看完就丢，没有复用价值。存 K、V 是存"会被反复查询的索引和内容"。

**问题2：** Flash Attention 为什么说是"精确"而不是"近似"？

**参考答案：** 它只是换了计算顺序：分块累加时用缩放因子 e^(m_old−m_new) 把旧的局部和折算到新的全局最大值（0.3 第三问的手算），数学上与一次性 softmax 严格相等（浮点结合律差异除外）。省的是 HBM 往返，不是精度。

**问题3：** 16 GiB 的 KV 缓存意味着什么成本？

**参考答案：** 模型权重之外还要再占 16 GiB 显存，且随上下文线性增长——批大小 1 的 32K 对话就能吃掉一张 A100 的一半。这正是 GQA/MLA（把缓存缩小数倍）、PagedAttention（消灭碎片）和前缀缓存（共享重复前缀）存在的原因：权重省不了多少，缓存才是长上下文推理的主战场。

**问题4：** 有缓存后，生成第 100 个 token 的一步要做多少次 q·k 比较？

**参考答案：** 100 次（与 100 个缓存键各比一次）。注意每步的比较次数就是当前序列长度——生成长度从 10 涨到 100，单步成本也涨 10 倍；总比较次数 1+2+…+100=5,050，这就是"总代价仍随 N²"的含义，缓存省的是其中重复的投影计算。

能算出 640 字节、55 次比较、10 份对 55 份这三组数后，进入正文。

---

## 一、这一课解决什么问题？

朴素的自回归解码器生成 N 个 token 要做 O(N²) 的注意力计算：每一步都对完整前缀重算一遍注意力。4K 的回答就是 1,600 万次注意力操作，绝大多数是冗余的——前缀 token 的隐藏状态一旦算过就是确定的，新 token 只需要拿自己的 Q 去查缓存里所有人的 K、V。除此之外，注意力本身搬运数据很凶：标准实现要把 N×N 分数矩阵、N×d 的 softmax 输出、N×d 的最终输出在 HBM 里写进读出——N≥2K 之后注意力先撞上带宽墙再撞上算力墙，经典注意力核让现代 GPU 的算力闲置 4—10 倍。两项优化（都出自 Dao 等人）把前沿推理从"慢"推到"快"：KV 缓存把每步生成从 O(N²) 降到 O(N)；Flash Attention 用分块让 N×N 矩阵全程不落 HBM，A100 上 2—4× 墙钟加速、H100 配 FP8 达 5—10×（原课参考）。到 2026 年两者已是标配：每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认假设它们存在。

## 二、核心概念与推导

### 2.1 KV 缓存的数学

每个解码器层、每个 token、所有 KV 头合计：

~~~
bytes_per_token_per_layer = 2 * n_kv_heads * d_head * dtype_size
                          ^
                          K 和 V 两份
~~~

7B 模型（32 层、32 头全作 KV 头、d_head=128、fp16，原课算例）：

~~~
per token per layer = 2 * 32 * 128 * 2 = 16384 字节（16 KiB）
per token（32 层）  = 512 KiB
per 32K context     = 16 GiB
~~~

Llama 3 70B（80 层、d_head=128、GQA 8 个 KV 头）：

~~~
per token per layer = 2 * 8 * 128 * 2 = 4096 字节（4 KiB）
per 32K context     = 10 GiB
~~~

10 GiB 在 32K 涨到 128K 就是 40 GiB——批大小 1 就吃掉整张 40 GB A100。**GQA 是 KV 缓存的关键赢家**：64 个 KV 头的 MHA 在 32K 要约 80 GiB，是 GQA 的 8 倍；MLA 压得更狠（原课参考，细节在后续课）。0.3 的迷你例（2×2×4×2=32 字节/层）与这两组数字是同一个公式，只是量级不同。

### 2.2 Flash Attention：分块平铺

标准注意力的三趟 HBM 往返（原课伪代码）：

~~~
S = Q @ K^T          （读 HBM，N×N，写 HBM）
P = softmax(S)       （读 HBM，写 HBM）
O = P @ V            （读 HBM，写 HBM）
~~~

H100 上 HBM 带宽 3 TB/s、SRAM 30 TB/s——每趟 HBM 相对片上是 10 倍减速。Flash Attention 的做法（原课伪代码）：

~~~
for each block of Q（块大小约 128 × 128）:
    把 Q 块载入 SRAM
    for each block of K, V:
        把 K 块、V 块载入 SRAM
        计算 S 块 = Q 块 @ K 块^T     （SRAM 内）
        在线 softmax 累加              （SRAM 内）
        累加进 O 块                   （SRAM 内）
    把 O 块写回 HBM
~~~

每块只出一趟 HBM。总内存占用从 O(N²) 降到 O(N)。反向传播不存中间矩阵、由前向重算——又一个内存上的赢。数值技巧：在线 softmax 跨块维护 (max, sum)，最终归一化是精确的——**Flash Attention 与标准注意力输出逐位一致（fp16 结合律差异除外），不是近似**。0.3 第三问的 [1,0]、[2] 两块合并就是这个算法的完整手算。

### 2.3 版本演进（原课表）

| 版本 | 年份 | 关键变化 | 参考硬件上的加速 |
|------|------|---------|-----------------|
| Flash 1 | 2022 | 分块 SRAM 核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行、因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5—2×（FP16 约 740 TFLOPs） |
| Flash 4 | 2026 | Blackwell 五级流水线、软件 exp2 | 推理优先（初期仅前向） |

Flash 4 发布时只有前向：训练仍用 Flash 3；GQA 与变长序列支持在 2026 年中尚在推进（原课边界）。

### 2.4 投机解码：另一条延迟路线

小模型起草 N 个 token，大模型一次并行验证。验证接受 k 个，就等于用一次大模型前向买了 k 个 token。典型 k=3—5（代码与普通文本，原课参考）。2026 默认：EAGLE 2 / Medusa（草稿头与验证器共享隐藏状态，2—3× 无损加速）、草稿模型式（消费级硬件 2—4×）、前瞻解码（Jacobi 迭代，免草稿模型，小众但免费）。细节留给 [第16课](<16：投机解码——草拟、验证、重复.md>)。

### 2.5 连续批处理

经典批处理要等最慢的序列完成才能换新批，短回复提前结束就浪费 GPU。连续批处理（Orca 首创，现已是 vLLM、TensorRT-LLM、SGLang 的标配）：有请求完成立刻换入新请求。典型聊天负载吞吐 5—10×（原课参考）。

### 2.6 PagedAttention：KV 缓存的虚拟内存

vLLM 的招牌特性。KV 缓存按 **16-token 块**分配；页表把逻辑位置映射到物理块。好处：并行采样/束搜索可共享 KV、系统提示等热前缀可以热切换（提示缓存）、内存免碎片化。相比朴素的连续分配吞吐 4×（原课参考）。（补充：与 [第04课](<04：位置编码——正弦、RoPE与ALiBi.md>) 的类比——虚拟内存的页表思想原样搬进显存管理，工程里好点子常常是跨领域搬运的。）

### 2.7 把 0.3 的数字串回架构（补充）

迷你缓存（2 层 2 头 4 维）→ 2.1 的公式（乘上 32 层 32 头 128 维就是 7B 的 16 GiB）；55 次比较两解码器相同 → 2.2 的"缓存省的是投影与搬运，不是比较本身"；在线 softmax 的 1.503 → 2.2 的"逐位一致"；10 份对 55 份 → 第一问的 O(N²)→O(N)。三个代码块各认领一组数字，读代码时就不会迷路。

## 三、操作前准备

Build It 只用 Python 标准库（math；随机数据可选 random），无需第三方依赖。Use It 需要 HuggingFace transformers + PyTorch（示例模型），生产示例需要 vLLM：

~~~bash
python -m pip install torch transformers
python -m pip install vllm          # 生产部署按需
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或下载模型。（核对：原课代码用了 `math.exp`，课程页代码块未展示 `import math`，运行前需自行补上；本篇按原样转录，不改代码本体。）

## 四、Build It：原课分步代码与中文讲解

原课 Build It 三步：KV 缓存 → 分块 softmax → naive 与 cached 对比。前两步的课程页代码原样转录如下，第三步为文字描述。

### 4.1 第 1 步：KV 缓存（原课代码，原样转录）

~~~python
# 本块中文说明：用嵌套列表存每层每头的 K、V 向量，逐 token 追加。
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
~~~

逐行看：`K` 是"层 → 头 → 向量列表"的三层嵌套，`V` 同构——0.3 第一问的 320 个数就住在这两个结构里（2 层 × 2 头 × 10 个向量 × 4 维 × 2 个结构）。`append` 每生成一个 token 调一次（每层每头各一次）；`read` 在算注意力时取整条历史。原课的评语：简单——就是让每层每头的 K、V 列表不断变长。（补充：真实框架里这是预先分配的大张量加一个长度游标，避免 Python 列表的追加开销；此实现只示范语义。）

### 4.2 第 2 步：分块 softmax（原课代码，原样转录）

（核对：函数用 `math.exp`，课程页未展示 `import math`，运行前需补；按原课代码原样转录。）

~~~python
# 本块中文说明：Flash Attention 风格的 softmax(qK^T)V——跨块维护运行最大值与运行和。
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
~~~

逐行拆开，变量正好对应 0.3 第三问的手算：`m` 是"当前全局最大分数"（块 1 后为 1、块 2 后为 2）；`exp_old = e^(m−new_m)` 就是把旧累加折算到新最大值的缩放因子（手算里的 0.368）；`s` 是运行和（1.368 → 1.503）；`out` 一直存**未归一化**的加权和，循环结束才统一除以 `s`——这就是"归一化推迟到最后一步"的精确性来源。原课的评语：与一次算完的 `softmax(qK)V` 逐位一致，但任意时刻的工作集只有 tile×d_head 一块，不是完整 N×d_head。`exp_old` 的 `-inf` 特判处理第一块（还没有"旧累加"可折算，缩放因子取 0）。

### 4.3 第 3 步：naive 与 cached 对比（文字步骤）

原课描述：数注意力操作次数。两个解码器做**同样的** q·k 比较：第 t 步看 t 个 token，合计 1+2+…+N = N(N+1)/2。代码取 N=10，两个解码器都打印 **55**。100 个 token 的生成两边都是 5,050 次比较；差别在 K、V 投影：无缓存每步为整个前缀重算 K、V，5,050 份 token 状态；有缓存每个 token 只算一次，100 份（原课数字）。0.3 第二问的 55 对 10 就是 N=10 的版本。

## 五、把代码运行起来：补充入口与检查

（补充）原课 `code/main.py` 未随页面展示，本篇不虚构。以下是本机实际执行过的检查（纯标准库）：

~~~python
# 本块中文说明：（补充）0.3 手算的落地检查。
import math

# 第一问：迷你缓存的字节数与浮点数个数（2 层、2 头、d_head=4、fp16、序列 10）。
per_tok = 2 * 2 * 2 * 4 * 2            # 2(K,V) x 2 层 x 2 头 x 4 维 x 2 字节
assert per_tok == 64 and per_tok * 10 == 640
assert 2 * 2 * 2 * 10 * 4 == 320        # 浮点数总数（K 160 + V 160）

# 第二问：比较次数与 token 状态数。
assert sum(range(1, 11)) == 55           # 两个解码器相同的比较次数

# 第四问：7B 与 70B 的公式复核（原课数字）。
assert 2 * 32 * 128 * 2 == 16384                 # 16 KiB / token / layer
assert 2 * 32 * 128 * 2 * 32 * 32768 / 1024**3 == 16.0   # 32K 上下文 16 GiB
assert 2 * 8 * 128 * 2 * 80 * 32768 / 1024**3 == 10.0    # 70B：10 GiB

# 在线 softmax 与一次性 softmax 的一致性（0.3 第三问 + 6 键随机例）。
def ref(q, K, V):
    scores = [sum(a * b for a, b in zip(q, k)) for k in K]
    m = max(scores); exps = [math.exp(x - m) for x in scores]; t = sum(exps)
    return [sum(e * v[j] for e, v in zip(exps, V)) / t for j in range(len(V[0]))]
# 对 tiled_softmax_dot(q, K, V) 与 ref(q, K, V) 逐位断言（数值见第十一节）。
~~~

（补充）补充检查方法：给 `tiled_softmax_dot` 喂 6 个键、tile=4（两块，第一块 4 个第二块 2 个，专门检验"块尾不满"的分支）；再喂 tile=3（整除分块）对照——两种切块结果都应与一次性计算逐位一致。`KVCache` 则用 2 层 2 头各 append 10 次后检查 `len(cache.K[1][0]) == 10`，并数出总浮点数 320。

## 六、Use It：使用现成工具

HuggingFace（原课代码，原样转录）：

~~~python
# 本块中文说明：transformers 的 generate() 默认启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # 用 Hopper 就换 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
~~~

vLLM 生产部署（原课命令，原样转录）：

~~~bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
~~~

（补充）读参数清单：`--tensor-parallel-size 4` 把权重分到 4 卡；`--max-model-len 32768` 同时约算了 KV 缓存预算（2.1 的 10 GiB）；`--enable-prefix-caching` 打开 2.6 的前缀缓存；`--kv-cache-dtype fp8` 把缓存再压一半。原课的 2026 实践要点：跨请求的前缀缓存是大事——同一段系统提示、少样本例子或长文档的 KV 跨调用复用；对重复工具提示的智能体负载，前缀缓存 routinely 5× 吞吐（原课参考）。

## 七、失败模式、原文边界与工程注意

1. **KV 缓存吃光显存是长上下文的第一死法**：0.3/2.1 的账——批 1 的 128K 上下文光缓存就 40 GiB。先算 `2 × n_kv_heads × d_head × dtype × 层数 × 序列长 × 批`，再谈部署。
2. **缓存不是万能提速**：它省的是重复的 K/V 投影（55 份对 10 份），不省 q·k 比较（两边都是 55）。序列越长，单步成本依然线性涨；总比较次数仍 O(N²)。
3. **fp8 KV 缓存有精度权衡**（补充）：`--kv-cache-dtype fp8` 换一半显存，但缓存值经过量化，长上下文末端可能有细微质量回退——开之前做回归测试。
4. **Flash 4 的边界**（原课）：发布时仅前向、训练仍用 Flash 3、GQA 与变长支持在 2026 年中待补——追新版本前先查自己需要的功能位。
5. **投机解码不是免费午餐**（补充）：验收通过率低（草稿模型与目标分布不匹配，如高度专业文本）时退化为 1×，还要倒贴草稿开销；[第16课](<16：投机解码——草拟、验证、重复.md>)展开。
6. **批大小 1 的延迟与吞吐是两件事**（补充）：连续批处理提高吞吐（多请求拼批），但单请求的首 token 延迟还取决于预填充；评估服务时分开测量。
7. **朴素实现里"没缓存反而更慢"的假象**（补充）：0.3 的第二问提醒你，比较次数相同——所以玩具基准里缓存版可能只快一点点（省的是投影），别因此低估它：真实模型里投影是 2×d_model×(层数×前缀长度) 级别的矩阵乘，才是大头。
8. **KVCache 类是教学语义，不是性能代码**（核对边界）：Python 嵌套列表逐 token append 在真实规模下完全不可用；真实框架用预分配张量+页表（PagedAttention）。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/skill-inference-optimizer.md`（一个"给新推理部署选注意力实现、KV 缓存策略、量化与投机解码"的技能文档）。改写成中文自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
检查推理部署：给出 KV 缓存的显存预算算式（2 × n_kv_heads × d_head × dtype × 层数 × 序列长 × 批）与实际数字；说明注意力实现与硬件的匹配（FA3 需 Hopper、FA4 的前向限制是否影响本场景）；是否启用前缀缓存、共享前缀占比多少；量化策略（权重与 KV 缓存分开说明）及其验证方法；投机解码的预期接受率与失败退化路径；吞吐与首 token 延迟分开测量；所有未实测的加速数字标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **原课 Easy**：运行 `code/main.py`，确认 naive 与 cached 解码器输出一致、注意操作次数之差。验收标准：两个解码器各打印 55（N=10）；能解释"输出一致"为什么是必然（缓存不改变数学，只消除重算）。本篇第五节的断言已覆盖数字部分。
2. **原课 Medium**：实现前缀缓存：给定提示 P 与若干个续写，对 P 只做一次前向填充 KV 缓存，然后每个续写从缓存分支。验收标准：与"每个续写重编码 P"的基线比速度（自己实测），并解释省下的计算正比于 P 的长度乘续写个数。
3. **原课 Hard**：实现玩具 PagedAttention：KV 缓存按 16-token 定长块分配、维护空闲块链表，序列结束把块还回池子。模拟 1,000 个变长对话补全，对比连续分配的碎片率。验收标准：给出两种分配的碎片/浪费字节统计，并解释页表如何支持多序列共享前缀（对照 2.6）。

## 十、术语与完成标准

KV 缓存：存下每个前缀 token 的 K、V，新查询直接查；预填充：提示词一次性并行进缓存；HBM/SRAM：显存（大而慢，~3 TB/s）/片上缓存（小而快，~30 TB/s）；Flash Attention：分块在 SRAM 算注意力，N×N 不落 HBM，输出精确；在线 softmax：跨块维护 (max, sum)，最后归一化；GQA：多查询头共享 KV 头，缓存缩小数倍；MLA：DeepSeek 的进一步 KV 压缩；PagedAttention：16-token 块+页表的 KV 虚拟内存；前缀缓存：跨请求共享公共前缀的 KV；连续批处理：完成即换入换出，吞吐 5—10×；投机解码：草稿+一次验证多个（第16课）；memory-bound：瓶颈在数据搬运而非计算。

完成检查：

- 能手算迷你缓存的 32 字节/层、64 字节/token、640 字节@序列 10、320 个浮点数。
- 能复算 7B 的 16 GiB 与 70B 的 10 GiB（含 GQA 头数变化的解释）。
- 能说清"缓存省的是投影不是比较"，以及 55=10 的具体含义。
- 能手算在线 softmax 的两块合并（0.3 第三问），并解释为什么精确。
- 能指出 Flash 4 的使用边界（前向 only、GQA/varlen 待补）。
- 能解释 PagedAttention 与前缀缓存分别解决缓存的什么问题（碎片、重复）。

## 十一、验证范围与来源许可

原课 Build It 共 3 个步骤：`KVCache` 与 `tiled_softmax_dot` 两个代码块逐字转录嵌入本篇（仅按本篇格式添加中文说明与 import 核对说明，未改代码本体逻辑）；第 3 步为课程页文字描述，本篇如实转述。Use It 的 HuggingFace 代码与 vLLM 命令原样保留。课程页未展示运行输出，本篇未虚构"原课参考"数值（16 GiB/10 GiB/40 GiB/80 GiB、A100 2—4×、H100 5—10×、吞吐 5—10×/4×/5×、k=3—5 等均出自原课）。

本机验证记录（2026-11-06，纯标准库脚本，不构成性能实验）：

- 抓取课程页成功，以原文为准整理。
- 0.3 节全部手算数字（32/64/640 字节、320 浮点、55 次比较、16 GiB、10 GiB）已用脚本按公式复核；第五节对应断言实际执行通过。
- `tiled_softmax_dot`（原样转录）与一次性 softmax·V 在一个 6 键随机测试向量上逐位一致（差异 < 1e-12）；0.3 第三问的两块手算（s 从 1.368 折算到 1.503）与脚本输出一致。
- 已转录的两个代码块通过语法检查（`tiled_softmax_dot` 需补 `import math`，已在第三、四节标注）。
- 未做：加载 Llama、运行 vLLM、任何真实模型的加速测量。

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
