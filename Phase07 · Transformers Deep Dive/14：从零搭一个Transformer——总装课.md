# 14：从零搭一个Transformer——总装课

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/14-build-a-transformer-capstone) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/14-build-a-transformer-capstone/docs/en.md)。原题：Build a Transformer from Scratch — The Capstone。

## 零、初学者先读：把 13 课的零件装成一台能写“莎士比亚”的机器

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第13课](<13：缩放定律——算力、数据与参数的预算.md>)讲了“训多大、用多少数据”的预算；本课把预算花出去——13 课的零件（注意力、多头、位置编码、RMSNorm、SwiGLU、GPT 损失）在这里第一次装成一台完整的解码器（decoder-only）生成模型。之后的[第15课](<15：注意力变体——滑动窗口、稀疏与差分注意力.md>)与[第16课](<16：投机解码——草拟、验证、重复.md>)都在这台机器上做改装。

### 0.1 先理解这一课在做什么

前面每一课都在造一个零件：第 2 课造了缩放点积注意力，第 3 课把它掰成多头，第 4 课给 token 编上位置，第 5 课组装了编码器与解码器块，第 7 课定义了“预测下一个 token”的训练目标。零件都验过收了，但它们从没在同一台机器上一起转过——总装课就是拧最后一圈螺丝：端到端训练一个小的解码器 Transformer，做字符级（character-level）语言建模，语料是莎士比亚全集的小节选（tinyshakespeare，约 1.1MB）。它小到笔记本 10 分钟内能训完，又正确到“换更大的数据集、训更久，就是一台真正的语言模型”。

原课明说这不是原创：它是本课程的"nanoGPT"——Karpathy 2023 年的 nanoGPT 教程是每个学生都至少要写一遍的参考实现，本课借用其形状、并围绕本课程已覆盖的内容重造。类比：前面是学做菜的单个技法（切菜、火候、调味），这节课从头到尾炒出一盘完整的菜——菜谱还是经典家常菜（nanoGPT），但每一刀都是你自己下的。

### 0.2 本课术语先翻成白话

- 解码器 Transformer（decoder-only transformer）：只保留 GPT 那半边（带因果掩码）的架构，生成式任务的标准形状
- 字符级分词器（char-level tokenizer）：直接把每个字符当一个 token；tinyshakespeare 只有 65 个不同字符，词表就是 65
- GPTConfig：把所有超参数（层数、头数、维度、词表大小）收进一个配置对象，改实验只改一处
- 因果注意力（causal attention）：第 i 个位置只能看 ≤i 的位置，保证“预测下一个 token”时不偷看答案
- RMSNorm：LayerNorm 的简化版，只除以均方根、不减均值，参数只有一组缩放
- SwiGLU：前馈层（FFN）的现代配方——门控分支乘上激活分支，三块矩阵
- pre-norm：先归一化再进子层，残差主干保持干净；比 post-norm 好训
- 残差连接（residual connection）：子层输出加回输入，梯度的高速公路
- 绑定嵌入（tied embeddings）：输出层（lm_head）的权重就是词嵌入矩阵的转置，省参数还提质
- shift-by-one 交叉熵（shift-by-one cross-entropy）：输入 x 的目标是右移一位的 y——每个位置都在预测下一个 token
- AdamW：带解耦权重衰减的 Adam 优化器，Transformer 训练默认
- 余弦学习率（cosine LR schedule）：先线性热身再按余弦衰减到峰值 10%
- 梯度裁剪（gradient clipping）：全局梯度范数超过 1.0 就按比例压回，防训练爆炸
- bf16 autocast：前向/反向用 bfloat16、优化器状态留 fp32 的混合精度训练
- 验证损失（val loss）：模型没见过的那 10% 文本上的交叉熵，过拟合探测器
- MFU（模型算力利用率）：实际 FLOPs / 理论峰值；2026 年稠密模型 40% 算强
- 采样（sampling）：按 logits 的概率分布挑下一个 token；本课用 top-p 截断后再随机挑

### 0.3 用小数字走一遍：参数量估算与总装清单

先做参数量估算。设一个比原课更小的配置：d_model=64、4 层、4 头、词表 V=1000、上下文 256、SwiGLU 隐层 h≈8d/3≈170（补充：h 取 8d/3 是为了让 SwiGLU 的 3dh 与传统 2 层 4d 的 8d² 参数量对齐，是常见工程惯例）。参数逐项数（嵌入与 lm_head 绑定，位置嵌入可学习）：

| 部件 | 公式 | 数值 | 出处课 |
|---|---|---|---|
| 词嵌入（绑定 lm_head） | V×d = 1000×64 | 64,000 | 第07课 |
| 位置嵌入 | 256×64 | 16,384 | 第04课 |
| 每层注意力 Wq/Wk/Wv/Wo | 4×d² = 4×4096 | 16,384 | 第02/03课 |
| 每层 SwiGLU FFN | 3×d×h = 3×64×170 | 32,640 | 第05课 |
| 每层两个 RMSNorm | 2×d | 128 | 第05课 |
| 单层小计 | — | 49,152 | — |
| 4 层合计 | 4×49,152 | 196,608 | — |
| 最终 RMSNorm | d | 64 | — |
| **总计** | — | **277,056 ≈ 0.28M** | — |

手算自洽性检查：注意力占每层 16,384/49,152=1/3，FFN 占 2/3——与“FFN 是参数大户”的直觉一致；词嵌入 64,000 占全模型约 23%，说明小词表下嵌入不可忽略（换 BPE 词表 50,000 后嵌入会反超一切，这正是原课 Use It 提醒“词表变大后容量要跟着扩”的原因）。再用原课配置验证同一套公式：d=128、4 层、词表 65、上下文 256、h≈8×128/3≈341 → 嵌入 8,320 + 位置 32,768 + 每层（65,536+130,944+256）×4=786,944 + 最终 128 = 828,160 ≈ 0.83M，与原课"~800K"对上（原课参考；我的公式复算见第十一节）。注意头数（4 头）不改变参数量——4 个 16 维头拼起来仍是 d×d 的投影。

再列总装清单（每个零件都能对应回前面某课）：

1. 数据与分词：读文本 → 65 字符词表 → encode/decode 函数 → 90/10 训练/验证切分（第07课）
2. 嵌入：token 嵌入 + 可学习位置嵌入，相加进入第 1 层（第04课）
3. ×L 个 Block：pre-norm 因果多头注意力 + 残差 → pre-norm SwiGLU + 残差（第02/03/05/07课）
4. 出口：最终 RMSNorm → 绑定的 lm_head → logits（第05/07课）
5. 损失：shift-by-one 交叉熵（第07课）
6. 训练循环：随机取批 → 前向 → 反向 → 梯度裁剪 → AdamW 步进（余弦学习率）
7. 生成：给个开头，反复“前向 → top-p 采样 → 追加”，直到 500 个 token

以上是用于理解的算例（参数量复算），不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

原课页面展示了三个代码片段（数据准备、训练循环骨架）和一张架构图，完整实现指向 `code/main.py`（页面未全文展示，第四节如实标注）。读的顺序：先看第二节的架构 ASCII 图（把 13 课零件对号入座），再看 4.1 的数据代码（5 行，建立“字符→整数”的映射），然后是训练循环片段（4.3，注意 shift-by-one 交叉熵藏在 `F.cross_entropy(logits, y)` 里），最后看采样与输出（4.4/4.5）。核心数据只有四个：`text`（全文）、`stoi`/`itos`（字符↔整数）、`(x, y)` 批次（x 与 y 就是同一窗口错开一格）。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 y 是 x 右移一位，而不是另找一批标签？

**参考答案：** 语言建模的“标签”就是下一个 token 本身。同一段文本错开一位就同时是输入和标签（x=“ROMEO: ” 对应 y=“OMEO: ”），不需要人工标注——这是 GPT 类模型“自监督”的含义（第07课）。

**问题2：** lm_head 与词嵌入“绑定”是什么意思，省了多少参数？

**参考答案：** 输出投影的权重矩阵直接用嵌入矩阵的转置（两者形状天然互为转置：V×d 与 d×V）。本课 0.3 的小配置省 64,000 个参数（约总量 19%）；更重要的作用是嵌入空间里相近的 token 在输出端也天然相近，质量通常更好。

**问题3：** 头数从 4 改成 8，参数量怎么变？

**参考答案：** 不变。多头是把 d_model 切成 h 份分别做注意力再拼接，投影矩阵始终是 d×d；改头数改的是“视角的粒度”，不是参数量。要加参数得改 d_model 或层数。

**问题4：** 验证损失和训练损失贴得很近，说明什么？贴太近又是坏事吗？

**参考答案：** 贴得近说明在这个尺寸/预算下没有过拟合（原课的目标指标之一）。它不必然是坏事，但意味着模型可能“还没吃饱”——容量或数据还有富余，可以加大模型或训更久；真正的过拟合信号是训练损失继续降、验证损失开始回升。

能回答这四个问题后，进入正文。

---

## 一、这一课解决什么问题？

你读完了所有论文，实现了注意力、多头切分、位置编码、编码器解码器块、BERT 和 GPT 的损失、MoE、KV 缓存。现在让它们在一台真实机器上协同工作，完成一个真实任务。总装项目：在字符级语言建模任务上端到端训练一个小型解码器 Transformer——读莎士比亚，生成“新的莎士比亚”。它小到笔记本上 10 分钟内训得完，又正确到“换上更大的数据集、训得更久，就得到一台真正的语言模型”。

这就是本课程的"nanoGPT"。原课坦诚它不是原创——Karpathy 的 2023 nanoGPT 教程是每个学生至少要写一遍的参考实现——本课沿用其形状、按本课程已覆盖的内容重新组织。

交付内容（原课 "What we ship"）：`GPTConfig`（超参数集中一处）、`MultiHeadAttention`（因果、批处理、可选 Flash 式通路 PyTorch `scaled_dot_product_attention`）、`SwiGLUFFN`、`Block`（pre-norm、残差包裹的注意力+FFN）、`GPT`（嵌入、堆叠块、lm_head、generate()）、带 AdamW/余弦学习率/梯度裁剪的训练循环、莎士比亚字符级分词器。

明确不交付（原课 "What we don't ship"，都是练习题）：RoPE（第04课概念实现过，这里用可学习位置嵌入求简，练习要求换装）；生成时的 KV 缓存（每步重算全前缀，慢但简单，练习要求补上，见[第12课](<12：KV缓存与Flash Attention——推理优化.md>)）；Flash Attention（PyTorch 2.0+ 输入匹配时自动派发，代码里直接用 `F.scaled_dot_product_attention`）；MoE（每层单个 FFN，见[第11课](<11：混合专家MoE——稀疏激活.md>)）。

目标指标（原课参考，Mac M2 上 4 层/4 头/d_model=128、2000 步）：训练损失约 6 分钟从 4.2（随机）收敛到 1.5；采样输出“像莎士比亚”：古体词、换行、人名（如 "ROMEO:"）开始出现；验证损失（文本末尾留出的 10%）紧跟训练损失，该尺寸/预算下无过拟合。

## 二、核心概念与推导

### 2.1 整机架构图（原课转录）

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

逐行导读（补充）：(B, N) 是批次×序列长度的整数下标；嵌入层把每个下标变成 d 维向量并加上位置向量；block × L 重复 L 次，每次内部两个子层（注意力、FFN）各自 pre-norm 并走残差；出口先整体归一化再用绑定的 lm_head 投到词表维度 (B, N, V)；损失把每个位置的 logits 与“右移一位”的目标对齐。图里每一行右侧的课号就是零件的出处——总装课没有新零件，只有装配。

### 2.2 为什么是这套零件

- **pre-norm 而非 post-norm**：残差主干上没有归一化层阻挡，梯度可以无衰减地直达浅层，深层模型不用热身也能训稳（第05课）。
- **RMSNorm 而非 LayerNorm**：省掉减均值和偏置，参数减半，2021 后的主流选择。
- **SwiGLU 而非两层 GELU**：门控机制让 FFN 学“哪些信息通过”，同等参数下质量更好（第05课）。
- **因果掩码**：训练时整段文本一次前向，靠掩码保证位置 i 看不到 >i 的 token——并行训练、串行生成的两不误（第07课）。
- **绑定嵌入**：省参数、共享语义空间（0.3 问题 2）。
- **余弦学习率 + 热身**：初期学习率小步升（避免 Adam 二阶矩未热身时的震荡），随后余弦衰减到峰值的 10%，是 2020 后的默认。
- **梯度裁剪（范数 1.0）**：单个坏批次的尖峰梯度被整体压回，防止损失突然爆掉。

### 2.3 一个批次的形状旅行（补充）

设 B=32、N=256（块大小）、V=65、d=128：文本先被 encode 成整数流（形状无，就是一串 Python int）；`get_batch` 切出 x:(32, 256)、y:(32, 256)（同一窗口错开一位）；嵌入后 (32, 256, 128)；每个 block 原样进出（残差保持形状）；lm_head 后 logits:(32, 256, 65)；`view(-1, V)` 摊平成 (8192, 65) 与 y 的 (8192,) 做 `F.cross_entropy`，得到标量损失。反向传播沿同一形状原路返回。记住“形状不变、只在进出口变形”这条规则，读任何 Transformer 代码都不会迷路。

### 2.4 训练动力学：从 4.2 到 1.5 意味着什么（补充）

原课说损失从 4.2 降到 1.5，这两个数用“每个位置的困惑”来读最直观：交叉熵 L 对应的平均选择数≈e^L。初始 e^4.17≈65——“在全部 65 个字符里瞎猜”，这正是均匀分布的理论值；训完 e^1.5≈4.5——“平均每个位置只在 4 个半字符里犹豫”。也就是说模型 6 分钟学会了三件事：这里该换行、这里该是元音、这里大概是人名开头的大写字母。想象一个只见过 1.1MB 文本的人脑能达到的“下一字符预测”水平，你对 1.5 这个数就不会失望——也不会对“换成 10B token 后会发生什么”失去想象（这正是[第13课](<13：缩放定律——算力、数据与参数的预算.md>)的预测曲线管的地方）。

### 2.5 为什么选莎士比亚、为什么选字符级（补充）

tinyshakespeare 约 1.1MB，2015 年以来每个字符级语言建模教程都用它（原课 Key Terms 表）——小到 CPU/笔记本可训，又“有风格到一眼能看出模型学没学到东西”：古体词（thou、hath）、固定的剧本格式（人名+冒号+换行）给了模型清晰的统计结构。字符级而不是 BPE，是为了把“分词器”这个变量从总装课上拿掉（原课：No BPE, no tokenizer drama）——65 个字符的词表让嵌入矩阵小到可以忽略，把注意力集中在装配本身。代价是同样的文本变成 5—6 倍长的 token 序列（英文平均每字符≈0.25 个 BPE token），总装课愿意付这个代价换取简单。

## 三、操作前准备

本课用 PyTorch。安装 `torch`（CPU 版即可，原课原话）：

~~~bash
python -m pip install torch
~~~

原课脚本 `code/main.py` 负责处理：下载 `tinyshakespeare.txt`（缺失时）或读取本地副本；字节级字符分词；90/10 训练/验证切分；支持硬件上的 bf16 autocast 训练循环；训练完成后采样。先激活你已有的 Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装 PyTorch、没有下载数据集或运行训练。（补充）字符级分词不需要任何第三方库，但训练循环与张量操作全部依赖 torch——这与第 02 课的纯标准库风格不同，是本阶段第一次引入深度学习框架。

## 四、Build It：原课分步代码与中文讲解

> （核对）原课页面对模型主体只给出文字描述（"See `code/main.py`"），页面实际展示的代码片段为 4.1 与 4.3 两段、加架构图与输出示例；本节逐字转录这些片段，其余步骤按原文描述如实转述，不虚构未展示的类实现。

### 4.1 第 1 步：数据（原课代码）

读入全文，收集 65 个不同字符建词表，`stoi`/`itos` 是双向字典，encode/decode 一行一个。65 个字符的微型词表，一个 4 字节整数都装得下——没有 BPE，没有分词器玄学。

~~~python
# 本块中文说明：字符级分词器——把每个字符映射为整数下标。
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
~~~

（核对）本段假设 `tinyshakespeare.txt` 已在当前目录；原课 main.py 会在文件缺失时自动下载（见第三节）。`set(text)` 去重后排序保证词表顺序稳定，同一份文本任何时候 encode 的结果一致——可复现性的小细节。

### 4.2 第 2 步：模型（原课文字描述）

原文：见 `code/main.py`。block 就是第 05 课的教科书实现——pre-norm、RMSNorm、SwiGLU、因果多头注意力。4/4/128 配置的参数量约 ~800K（原课参考；0.3 节用公式复算得 828,160，基本对上）。

对照 0.3 的估算表，这里可以自查模型实现是否漏了东西（补充）：数一下你的 `model.parameters()` 总数，若与表算的差了整块的量（比如少了位置嵌入 32,768），多半是哪个模块忘了注册；若只差几个参数，多半是 RMSNorm 或绑定的差。这是总装课上最快的“装配正确性”检查。

### 4.3 第 3 步：训练循环（原课代码）

随机取长度 256 的 token 窗口批次，前向，shift-by-one 交叉熵，反向，AdamW 步进。日志。重复。

~~~python
# 本块中文说明：训练循环骨架——注意损失就藏在 cross_entropy(logits, y) 里。
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
~~~

三个易错点（补充）：

1. `get_batch("train")` 返回的 y 已是 x 错开一位的版本——shift-by-one 是在取数时完成的，不是在损失函数里；
2. `opt.zero_grad()` 放在 `opt.step()` 之后（原课顺序），梯度不会跨步累积；放前面也对，但别两处都放或都不放；
3. 原课正文说“训练循环带 AdamW、余弦学习率、梯度裁剪”，此骨架里只见裁剪与步进——优化器构造与学习率调度在页面未展示的 main.py 里（核对：`opt` 与 `max_steps` 在本片段外定义）。

### 4.4 第 4 步：采样（原课文字描述）

原文：给一个开头（prompt），反复前向、从 top-p 截断后的 logits 里采样、追加、继续，生成 500 个 token 停止。注意每步都对全前缀重新计算注意力（没有 KV 缓存）——慢但简单，练习 3 要求补上缓存，提速预期 5—20 倍（原课参考）。

### 4.5 第 5 步：读输出（原课转录）

2000 步之后的输出长这样（原课参考，原课页面转录）：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚，但“有莎士比亚的形状”。对 0.8M 参数、笔记本上 6 分钟的训练来说，是明确的胜利。

### 4.6 （补充）总装自查对照表

装完之后逐项打勾，每项都能追溯到前课：

| 检查项 | 通过标准 | 追溯 |
|---|---|---|
| 词表 | `len(chars)==65`，encode/decode 互为逆函数 | 第07课 |
| 因果性 | 手改 x 的最后一个字符，前面位置的 logits 不变 | 第02/07课 |
| 残差 | 把 block 换成恒等映射（跳过子层），模型仍可训 | 第05课 |
| 形状 | 2.3 节的形状表逐一核对 | 本课 |
| 参数量 | `sum(p.numel() for p in model.parameters())` 与 0.3 表对上 | 本课 |
| 损失初值 | 随机初始化时应约等于 ln(65)≈4.16 | 第07课 |
| 绑定 | `model.lm_head.weight is model.tok_emb.weight` 为真 | 本课 |

其中“损失初值≈ln(词表)”值得记住（补充）：未训练模型对 65 个字符均匀猜测，交叉熵就是 ln(65)≈4.16——原课说的“从 4.2 开始收敛”正是这个理论值附近（原课参考 4.2 与我的 ln(65)≈4.16 一致）。

## 五、把代码运行起来：补充入口与检查

（补充）原课完整代码在 `code/main.py`（页面未全文展示），以下是我不依赖 torch 的纯 Python 最小检查，用来复核 0.3 的参数量与损失初值。本机已实际运行：

~~~python
# 本块中文说明：参数量复算与“随机模型损失=ln(V)”检查（补充，非原课代码）。
import math

def count_params(d, V, ctx, L, h):
    emb = V * d                      # 词嵌入（与 lm_head 绑定，只算一次）
    pos = ctx * d                    # 可学习位置嵌入
    per_block = 4 * d * d + 3 * d * h + 2 * d  # 注意力+SwiGLU+两个 RMSNorm
    return {"emb": emb, "pos": pos, "per_block": per_block,
            "total": emb + pos + L * per_block + d}

small = count_params(d=64, V=1000, ctx=256, L=4, h=170)
print(small)  # 本机输出 total=277056（0.3 节表格）

course = count_params(d=128, V=65, ctx=256, L=4, h=341)
print(course)  # 本机输出 total=828160，与原课 ~800K 对上（原课参考）

# 随机初始化模型的初始损失 = ln(词表大小)
print(math.log(65))   # 本机输出 4.174 —— 原课“从 4.2 开始”的理论依据
print(math.log(1000)) # 本机输出 6.908
~~~

两个更贵的检查（需 torch 与数据，本次未执行，补充为建议）：训练前先过一遍“恒等测试”——把所有 block 设为恒等映射，模型退化为“嵌入+线性出口”，损失应仍能降到远低于 4.2 的水平，证明装配通路本身无短路；以及 0.3 的参数量断言 `abs(sum(p.numel() for p in model.parameters()) - 828160) < 1e4`（绑定嵌入时会略去重复计数，容差放 1e4）。

（补充）由于原课页面没有展示模型代码，下面给出一份**我自己的参考装配骨架**——不是原课代码、本机也未运行（无 torch 环境），只用于对照你按 4.2 描述写的实现，帮你确认没有漏装零件。形状注释与 2.3 节一一对应：

~~~python
# 本块中文说明：（补充，非原课代码，未运行）总装参考骨架，对照用。
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:                     # 原课 What we ship 清单第 1 项
    def __init__(self, vocab_size, d_model=128, n_layer=4, n_head=4,
                 block_size=256, ffn_h=None):
        self.vocab_size, self.d_model = vocab_size, d_model
        self.n_layer, self.n_head, self.block_size = n_layer, n_head, block_size
        self.ffn_h = ffn_h or round(8 * d_model / 3)   # SwiGLU 惯例（0.3 节）

class RMSNorm(nn.Module):            # 第05课：只除均方根、不减均值
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g

class SwiGLUFFN(nn.Module):          # 三块矩阵：gate / up / down
    def __init__(self, d, h):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MultiHeadAttention(nn.Module): # 第02/03/07课：因果、批处理
    def __init__(self, d, n_head, block_size):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.d_head = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.register_buffer("mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    def forward(self, x):
        B, N, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, N, self.n_head, self.d_head).transpose(1, 2)   # (B,h,N,dh)
        k, v = [t.view(B, N, self.n_head, self.d_head).transpose(1, 2) for t in (k, v)]
        # 生产写法：F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)        # (B,h,N,N)
        att = att.masked_fill(self.mask[:, :, :N, :N] == 0, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, N, d)
        return self.proj(y)

class Block(nn.Module):              # pre-norm + 残差（第05课）
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), MultiHeadAttention(cfg.d_model, cfg.n_head, cfg.block_size)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLUFFN(cfg.d_model, cfg.ffn_h)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)     # 第04课简化版
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                    # 绑定（0.3 问题2）
    def forward(self, idx):
        B, N = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(N, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm_f(x))                          # (B, N, V)
~~~

用 `count_params`（第五节第一段）核这份骨架：`GPT(GPTConfig(65))` 的参数量应落在 828,160 附近（d=128、h=341）。骨架刻意省略了训练循环（见 4.3 原课片段）与 `generate()`（见 4.4 描述）——那是练习让你自己装的部分。

## 六、Use It：使用现成工具

这台总装机是参考架构。原课给出三条通往“真实模型”的扩展路线：

1. **换分词器**：用 BPE（如 `tiktoken.get_encoding("cl100k_base")`），词表从 65 跳到约 50,000——模型容量必须跟着扩容补偿（0.3 的嵌入行会从 64,000 涨到数百万，正是原课提醒的原因）。
2. **换语料**：用 OpenWebText 或 fineweb-edu（HuggingFace）。单张 A100 上 10B token 训 125M 参数的 GPT 约需 24 小时（原课参考）。
3. **加 RoPE + KV 缓存 + Flash Attention**：练习逐个带你走。

终点是一台 125M 参数、能生成流畅英文的 GPT——不是前沿模型，但“同一条代码路径、只是更大”就是 Karpathy、EleutherAI、Allen Institute 在 2026 年训练研究检查点的方式（原课）。

从 0.83M 到 125M 的换算（补充，用 0.3 的公式感受一下）：保持词表 50,000（BPE）、上下文 1024，设 d=768、12 层、h=2048，则嵌入 38.4M + 位置 0.79M + 每层（4×768²+3×768×2048+1536≈7.1M）×12≈85.0M + 出口 768 ≈ 124M——正好落在 GPT-2 124M 的经典配置上。同样的公式从 0.28M 一路算到 125M 再到 70B（第13课的 Chinchilla），这就是“同一条代码路径、只是更大”的定量含义。

（补充）与现成工具的关系：HuggingFace `transformers` 的 `GPT2LMHeadModel`/`LlamaForCausalLM` 与本课的 `GPT` 是同一形状的工业版——读懂本课的自制版，再看那些几千行的实现，你会发现能认出 90% 的结构；剩下的 10% 是工程（分片、混合精度、检查点、张量并行）。

## 七、失败模式、原文边界与工程注意

1. 最常见的装配错误是“忘了因果掩码”：训练损失掉得飞快、采样却一团糟——模型在训练时偷看了未来。第 4.6 节的因果性检查（改末位字符、看前面 logits）能在 30 秒内确诊（补充）。
2. shift-by-one 做反了（y 与 x 对齐）损失会异常地低（在预测自己），生成质量为零。自查：打印 `decode(x[0])` 与 `decode(y[0])`，后者应是前者去掉第一个字符、多出一个新字符（补充）。
3. 原课目标指标（4.2→1.5、6 分钟、~800K）是 Mac M2 + 2000 步的参考值（原课参考）；换 CPU 训练时间会长一个数量级，CPU 版若 1 小时内训不完 2000 步属正常，不是你装错了。
4. 页面未展示完整 main.py（原文边界）：`opt`、`max_steps`、`get_batch`、top-p 采样与余弦调度只在文字里出现；自己补实现时优先按第 05/07 课与 4.3 注释的顺序，不要凭空发明接口。
5. 无 KV 缓存的生成是 O(N²) 的：500 token 的采样在笔记本上肉眼可见地变慢，这是练习 3 的动机，不是 bug（原课）。
6. 学习率是总装课最敏感的旋钮：太大损失发散（NaN），太小 2000 步内到不了 1.5。若发散，第一反应永远是调小学习率或确认热身生效，而不是改架构（补充）。
7. 训练/验证切分按字符位置 90/10（原课脚本行为），不是随机打乱——语言建模的验证集必须与训练文本在时间上不相交，随机打乱会造成“见过上下文”的泄漏（补充边界，原课未展开）。
8. bf16 autocast 在不支持的硬件上自动退回（原课：supported hardware）；CPU 训练直接用 fp32 即可，不要手写半精度（补充）。
9. 损失变 NaN 的排查顺序（补充）：先看梯度范数日志（裁剪前是否爆到 1e3+）→ 确认热身是否生效 → 学习率减半再试 → 最后才怀疑架构。90% 的 NaN 是学习率问题，不是数值精度问题。
10. 绑定嵌入的一个副作用（补充）：词嵌入同时接收“输入端”与“输出端”两路梯度，等效学习率约翻倍；若训练不稳，可考虑对绑定矩阵单独调学习率（nanoGPT 系实现的常见做法）——总装课尺寸下通常感知不到，先不用管。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/skill-transformer-review.md`（按前 13 课逐项审查一个从零实现的 Transformer 的正确性）。改写成中文自查模板（补充）：

~~~markdown
审查自建 Transformer：核对因果掩码（改末位、看前 logits）；核对 shift-by-one（打印 x/y 各自 decode）；参数量手算 vs numel() 双向核对（差整块=漏模块）；初始损失应约 ln(V)；残差与 pre-norm 是否在子层外加（主干干净）；嵌入是否绑定（is 同一对象）；训练曲线是否先降后稳、验证损失是否跟上；生成是否用 top-p；无缓存生成慢是预期不是 bug；所有未实测指标标注原课参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课 Easy：运行 `code/main.py`，验证最终验证损失 <2.0；把 `max_steps` 从 2000 改成 5000，验证损失是否继续改善？验收标准：能报告两组 val loss 并解释变化方向（继续下降=模型还没吃饱；走平=容量/数据到顶）。
2. 原课 Medium：把可学习位置嵌入换成 RoPE，在 `MultiHeadAttention` 内对 Q、K 施加旋转，训练并验证 val loss 至少不差于原版。验收标准：两版 val loss 并排报告；能指出 RoPE 的外推性质让更长上下文的生成不崩（对照第04课）。
3. 原课 Medium：在采样循环里实现 KV 缓存，有无缓存各生成 500 token 对比墙钟时间，笔记本上应有 5—20 倍改善（原课参考）。验收标准：两段输出文本质量一致（缓存不改变结果，只改变速度——若文本变了，缓存写错了）。
4. 原课 Hard：给模型加第二个预测 t+2 的头（DeepSeek-V3 的 MTP，多 token 预测），联合训练，看是否有帮助。验收标准：能报告主头 val loss 的变化并给出解释（辅助任务正则化主干的效应）。
5. 原课 Hard：把每层单 FFN 换成 4 专家 MoE（路由 + top-2），在匹配激活参数下看 val loss 变化。验收标准：能说明“匹配激活参数”怎么控制变量（见第11课），并报告路由负载是否均衡。

## 十、术语与完成标准

解码器 Transformer：只含因果掩码半边的生成式架构；字符级分词器：每字符一个 token、词表=不同字符数；GPTConfig：超参数集中管理；因果注意力：位置 i 只见 ≤i；RMSNorm：只除均方根的归一化；SwiGLU：门控+激活两分支的 FFN；pre-norm：子层前归一化、残差主干干净；绑定嵌入：lm_head 权重=词嵌入转置；shift-by-one 交叉熵：错开一位的自监督目标；AdamW：解耦权重衰减的 Adam；余弦学习率：热身后余弦衰减；梯度裁剪：全局范数封顶 1.0；bf16 autocast：混合精度训练；验证损失：留出文本上的交叉熵；MFU：实际/理论 FLOPs；top-p 采样：累计概率 p 内截断后随机挑；nanoGPT：Karpathy 的约 300 行参考实现。

完成检查：

- 能默画 2.1 的架构图并给每个部件标注出处课。
- 能用 0.3 的表手算任意 (d, 层数, 词表, 上下文) 配置的参数量。
- 能解释为什么初始损失≈ln(V)、为什么头数不影响参数量。
- 能说出“绑定嵌入”省了多少参数、为什么顺带提质。
- 能诊断三大装配错误：无因果掩码、shift-by-one 反向、残差断路。
- 能把“val loss <2.0”翻译回语言：平均每个字符的困惑度 e^2≈7.4 个字符——即模型在每个位置大约在 7 个字符里犹豫。
- 能列出 4.6 对照表的每一项并亲手执行。

## 十一、验证范围与来源许可

原课页面展示了 2 个 Python 代码片段（4.1 数据、4.3 训练循环）、1 张架构图、1 段采样输出，均逐字转录；模型实现与采样细节为原课文字描述，页面未展示代码本体，本节如实转述并标注，未虚构类实现。第五节的参考装配骨架是我自己的补充实现（明确标注“非原课代码”），本机无 torch 环境未运行，仅通过 `ast.parse` 静态语法检查；其参数量断言用 0.3 的公式手算核对（GPTConfig(65) → 828,160）。原课的目标指标（4.2→1.5、6 分钟、~800K、ROMEO 输出、5—20 倍缓存加速、A100 24 小时）均为原课参考。0.3 与第五节的参数量复算（小配置 277,056、课程配置 828,160、GPT-2 类比 124,140,288）与 ln(65)≈4.174、e^4.17≈64.7、e^1.5≈4.48、e²≈7.39 已于 2026-11-06 在本机用 Python 标准库实际运行复核通过（属公式复算，不构成训练实验）。未做：安装 PyTorch、下载数据、训练、采样、运行骨架。

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
