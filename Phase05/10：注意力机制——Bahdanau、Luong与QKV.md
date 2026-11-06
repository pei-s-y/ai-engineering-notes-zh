# 10：注意力机制——Bahdanau、Luong与QKV

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，概念细节与课程页一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/10-attention-mechanism) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/10-attention-mechanism/docs/en.md)。原题：Attention Mechanism — The Breakthrough。

## 零、初学者先读：给解码器一双能随时回看原文的眼睛

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。第09课结尾留下“便签瓶颈”：解码器只能拿到编码器末态。本课的修正只有三步——让解码器每写一个词，都对源句每个位置打一次分、按分数加权取材。这是全阶段通往Transformer的门，值得慢慢读。

### 0.1 先理解这一课在做什么

第09课的译员只带一张便签上场；注意力（attention）给他发回原稿：每写一个目标词之前，先看一遍源句所有位置，问“此刻我最需要哪个词的信息”，按相关程度加权取材。图书馆检索是最好的类比——

- 你提的问题=**查询（query）**；
- 每本书的书脊标签=**键（key）**，用来和问题比对打分；
- 书里的内容=**值（value）**，按分数高低借来混合。

打分用什么？正是[第03课](<03：词向量与从零实现 Word2Vec.md>)2.2用过的**点积**——两个向量点积大=方向合拍=“相关”。分数经softmax（第03课2.3那个softmax）变成一组和为1的权重，再加权平均所有value，得到这一步专属的上下文向量。所以注意力=“打分（点积）→归一（softmax）→加权平均”三步，没有更多魔法。2014年Bahdanau用它救活了长句翻译；后来Transformer把同一套三步用在句内（self-attention）并砍掉RNN——但那是第07阶段的事，本课先把这三步吃透。

### 0.2 本课术语先翻成白话

- 注意力 attention=按相关度对一组向量做加权平均的机制
- 查询 query（Q）=“我现在要找什么”，经典注意力里来自解码器当前状态
- 键 key（K）=“每个位置拿什么来应征”，用来与query打分
- 值 value（V）=“每个位置实际贡献的内容”，被加权平均
- 对齐权重 alignment weights=softmax后的α，这一步看每个源位置的比重，和为1
- 加性注意力 additive attention=Bahdanau式打分：v_aᵀtanh(W_a s + U_a h)
- 乘性注意力 multiplicative attention=Luong式打分：dot（直接点积）或general（中间乘个W）
- 上下文向量 context vector=加权平均的结果，形状与单个编码器状态相同
- cross-attention（交叉注意力）=query来自解码器、K/V来自编码器，两条序列之间
- self-attention（自注意力）=序列自己问自己，Q/K/V是同一序列的不同投影
- 多头注意力 multi-head=同一机制并行跑多组投影，各看各的模式
- mask=softmax前把无效位置（PAD、未来词）设成负无穷，排除出权重
- 对齐矩阵 alignment matrix=(T_dec×T_enc)的权重网格，每行和为1，可画成热图
- 缩放点积 scaled dot-product=Transformer的打分版本：QKᵀ/√d_k

### 0.3 用小数字走一遍

一个查询对两个键。设query=[1,0]，源句两个位置的key₁=[1,0]、key₂=[0,1]，value就用key本身（经典注意力里K=V）：

~~~text
打分（点积）：  q·k₁ = 1×1 + 0×0 = 1
               q·k₂ = 1×0 + 0×1 = 0
softmax：      e¹ = 2.718，e⁰ = 1，总和 = 3.718
               α₁ = 2.718/3.718 ≈ 0.731    α₂ = 1/3.718 ≈ 0.269
加权平均：     c = 0.731×[1,0] + 0.269×[0,1] = [0.731, 0.269]
~~~

三个读数：第一，q与k₁完全同向，分数1比0，但softmax只给0.731——注意力是“软”的混合，不是硬性二选一；第二，c=[0.731, 0.269]是两个value按比例的勾兑，偏向k₁但不等于k₁；第三，打分这一步和第03课2.2的W[c]·W_prime[o]是同一个点积、归一是同一个softmax——注意力把第03课的“给全词表打分再归一”搬过来，只是打分对象从“词表里的词”换成了“源句的位置”（本机Python已复核：softmax(1,0)=(0.731,0.269)；正文4.3三个分数0.94/0.52/0.24→[0.464,0.305,0.231]同样复核无误）。换成加性打分，只是把第一行换成v_aᵀtanh(W_a s + U_a h)，后两步原封不动。

再把query挪一挪，看权重的“注意”跟着走：取q=[0.6, 0.4]（不再完全偏向k₁）：

~~~text
q·k₁ = 0.6，q·k₂ = 0.4
α₁ = e^0.6/(e^0.6+e^0.4) = 1.822/3.314 ≈ 0.550    α₂ ≈ 0.450
c = [0.550, 0.450]
~~~

分数从“1比0”变成“0.6比0.4”，权重只是从“0.731比0.269”温和地挪到“0.550比0.450”——query的每一次移动，权重连续地跟着移动，这正是“对齐”的含义：翻译Je时权重压在Je的位置，写到pas时权重整体挪向pas的位置。正文4.3“移动query后检查分布是否移动”说的就是这件事。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

4.1的additive_attention逐步对应0.3的三步：projected_enc+projected_dec进tanh是打分，softmax(scores)是归一，weights @ encoder_states是加权平均；v_a的作用是把(d_attn,)的中间量压成一个标量分数。4.2的dot_attention是0.3的直接代码化——三行就是打分、归一、平均；general_attention多乘一个W接维度（双向encoder的d_s≠d_h时用它）。4.3的H与s_close_to_cat是三个键的版本，移动query可看权重移动。第六节MultiheadAttention是同一机制的批量多头版：Q(2,5,128)对K/V(2,10,128)，输出(2,5,128)、权重(2,5,10)——相当于5个查询各自对10个键做了一遍0.3，返回的weights是平均到各头的对齐图。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么不直接选分数最高的那个value，而要softmax加权平均？

**参考答案：** argmax不可导、无法训练；softmax把分数变成和为1的权重，梯度能流回打分参数。软混合还允许“主要看这里、余光看那里”，信息不丢。

**问题2：** K和V都是编码器状态时为什么不算同一个东西？“拿什么应征”和“实际给什么”分开有什么好处？

**参考答案：** 经典注意力里数值上恰好相同，但角色不同；Transformer给它们各自的可学习投影，让“用来匹配的特征”和“被取用的内容”学出不同的东西——与第03课W/W_prime拆两张表同一个道理：一个问题拆成两个向量，避免自匹配退化。

**问题3：** Bahdanau用s_(t−1)、Luong用s_t，这个下标差异要紧吗？

**参考答案：** 要紧。前者是“写这个词之前”的解码器状态、后者是“之后”的，混用会产生难以察觉的错误梯度；看论文或抄实现时先核对用的是哪套约定（原课Ship It也专门提醒这一点）。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

基础seq2seq只给decoder一个末态。本课让每个输出步骤以当前查询对全部源状态打分，再做加权汇总。关键不在“看哪里”的比喻，而在分数、归一化轴、维度和mask。

## 二、核心概念与推导

令H形状(T_enc,d_h)，query=s形状(d_s)。加性打分：
$$e_i=v_a^\top\tanh(W_as+U_ah_i),\quad \alpha=\operatorname{softmax}(e),\quad c=\sum_i\alpha_i h_i$$
W_a为(d_attn,d_s)，U_a为(d_attn,d_h)，v_a为(d_attn,)。加性内部表示逐位置相加，投影后从d_attn压为一个标量。

（补充 2026-11-06）**这个公式逐块拆开读。** s是“我现在要找什么”，h_i是“源句第i个位置”；但两者维度可能不同、即使相同也不宜直接比，所以各乘一个矩阵（W_a、U_a）都投到d_attn维的“谈判桌”上相加，tanh把数值压进±1防止打分爆炸，最后v_aᵀ把整桌信息压成一个标量——第i个位置的分数。三样东西各司其职：

- W_a／U_a管“各自带什么特征上桌”；
- tanh管稳数值；
- v_a管“桌上信息怎么合成一个分”。

对照0.3：点积打分是q·k一步出分，加性打分是先投影、再相加、再压缩——多花参数换来“维度不匹配也能打分”和一点非线性。softmax那行与第03课2.3的softmax是同一个函数：一列分数→一组和为1的权重，e的指数放大分差（0.94对0.24约四倍的分差，权重是0.464对0.231约两倍）。最后一行c=Σα_i h_i就是0.3的加权平均行。

（补充 2026-11-06）**“加权平均”才是主干，打分只是配角。** 全部公式的产出只是一个新向量c——它不是任何单一位置，而是所有位置的加权和。类比调色：query说“我要偏暖的灰”，α就是每种颜料的滴数（0.731滴k₁、0.269滴k₂），c是调出来的颜色。由此立刻理解两件事：

- 第09课的瓶颈怎么被拆掉：上下文不再是一张写死的便签，而是每写一个词就重新调一次、永远按需取材；
- mask为什么要放在softmax**之前**：先归一再把PAD清零，权重和就不为1，context的尺度被破坏（第五节的中文说明正是这个意思）；正确做法是把无效位置的分数先设成负无穷，softmax后自然近似0。

dot直接H@s，要求d_s=d_h；general用W把维度接起来，W为(d_s,d_h)。双向encoder不意味着一定不能dot：若decoder状态维度匹配或增加投影，同样可用。

（补充 2026-11-06）**三种打分对照表。** 打分函数回答“query和key有多匹配”，三条路线：

| 打分 | 公式 | 参数量 | 适用 |
|---|---|---|---|
| dot | sᵀh | 0 | 要求d_s=d_h，最便宜 |
| general | sᵀWh | d_s×d_h | 维度不匹配时用W接桥 |
| additive | vᵀtanh(Ws+Uh) | 最多 | 非线性、最灵活，也最贵 |

Luong论文的结论（原课参考）：多数任务上乘性与加性精度相当、代码短得多——正文4.2每版只有三行，这就是“Luong的论文一炮而红”的原因。两个延伸：

- 双向encoder把d_h撑成2×d_hidden后dot不再直接可用，要么general接桥、要么把decoder状态投影过去；
- “不能dot”是维度约束，不是能力判决。

（补充 2026-11-06）**Bahdanau与Luong的时间下标。** 打分用的解码器状态一个取“写词前”（s_(t−1)，Bahdanau：先看再看写）、一个取“写词后”（s_t，Luong：先写再回头核对）。两套都能工作，但实现里混用会产生方向微妙错误的梯度、极难调试——原课Ship It把它列为新手第一大坑。读2014～2017年的论文（含很多经典复现）前，先确认它用的是哪套约定。

Q是查询，K用于打分，V用于汇总。cross-attention来自不同序列；self-attention来自同一序列的不同投影。Transformer常用softmax(QKᵀ/√d_k)V，并需位置表示、多头与其他层，不能把二者说成只是换字母。

（补充 2026-11-06）**从经典注意力到Transformer的一步之遥。** 把0.3的机制换一套记号：query排成矩阵Q，key排成K，分数不再逐个算而是QKᵀ一次算完——softmax(QKᵀ)V就是“所有查询同时对所有键做一遍0.3”。真正的概念跳跃只有两个，其余都是工程件：

- /√d_k是数值稳定件：维度大时点积的方差随之变大，softmax会被个别大分数垄断，除以√d_k把方差压回去（经典注意力维度小，常省略）；
- self-attention把Q/K/V换成同一序列的三份不同投影（“自己问自己”，K与V从此分离，呼应第03课两表的角色分工）；
- 多头把d维切成几段并行各做各的，有的头看语法、有的头看主题；
- 再剩下的（位置编码、残差、层归一化）是组装——所以原课说从Bahdanau到scaled dot-product“大部分是记号问题”。

（补充 2026-11-06）**对齐图不是解释。** 权重和为1、能画成热图，“高亮=看这里”的诱惑很大；但Jain & Wallace（2019）指出：某些任务里把注意力权重换成别的分布，预测可以不变。权重是计算的一部分，不是因果证据：

- 报“模型关注了X”需要消融或反事实检验（第七节第1条的出处）；
- 调试形状、看对齐趋势可用它，写结论不能只靠它；
- 这条边界在第13课RAG里会再次出现：引用存在不等于引用支持。

### 2.9 用一个小例子把概念串起来（补充）

点积示例3个score为0.94、0.52、0.24，softmax约为0.464、0.305、0.231，总和1。context不是选第一个向量，而是三个encoder向量按权重加权和。Q/K用于决定看哪里，V提供被聚合内容；真实模型由可学习投影得到它们，不应把Q/K/V想成三个独立文本文件。

## 三、操作前准备

NumPy用于手写单样本注意力；PyTorch用于多头例子：
~~~bash
python -m pip install numpy torch
~~~
这里的手写函数没有batch维，不能直接把(B,T,H)张量塞进去。先核对单样本，再扩展batch与mask。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 加性注意力

projected_dec为(d_attn,)，projected_enc为(T,d_attn)；广播将同一query投影加到所有位置。scores/weights为(T,)，context为(d_h,)。softmax先减最大值提高稳定性，作用轴是全部源位置。合法广播是设计的一部分，不是一律禁用。

~~~python
import numpy as np


# 单样本输入，广播query投影后对源位置softmax。
def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


# 向量版本稳定化；未提供mask、空输入和batch处理。
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
~~~

### 4.2 Luong点积与general

dot需要query与状态末维匹配；general先W.T@s得到(d_h,)。原代码返回(context,weights)，不要把顺序接反。此处没有除√d，保留经典注意力形式，与Transformer缩放版本区分。

~~~python
# query末维必须与encoder_states末维一致。
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


# W形状为d_s×d_h，先映射query再与源状态打分。
def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
~~~

### 4.3 数值例子

三个源向量和query得到分数[0.94,0.52,0.24]，softmax约[0.464,0.305,0.231]。这是数学可核对的例子，不需要训练。移动query后检查分布是否移动，但不要把权重最高等同于模型真正的因果推理。

~~~python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
~~~

~~~text
weights: [0.464 0.305 0.231]
~~~

### 4.4 连接到Transformer

原课用decoder为Q、encoder为K/V建立cross-attention直觉，再讨论self与multi-head。Bahdanau前态、Luong后态的时间下标要与具体架构保持一致，不能只改变量名就保证实现正确。



## 五、把代码运行起来：补充入口与检查

NumPy定义完成后做以下补充测试（本次仅做独立标量数学核对，未安装NumPy）：
~~~python
ctx, weights = dot_attention(s_close_to_cat, H)
assert weights.shape == (3,)
assert ctx.shape == (3,)
assert np.allclose(weights.sum(), 1.0)
assert np.allclose(ctx, weights @ H)
W = np.eye(3)
ctx2, weights2 = general_attention(s_close_to_cat, H, W)
assert np.allclose(ctx, ctx2)
~~~
若加入padding mask，应在softmax前把无效位置设为负无穷，并明确“全被屏蔽”的行如何处理，否则可能NaN。只在softmax后把PAD权重清零而不重归一化，会改变context尺度。

## 六、Use It：使用现成工具

MultiheadAttention例子输入Q为(2,5,128)，K/V为(2,10,128)。默认返回平均各头的weights，形状(2,5,10)，不是(2,8,5,10)；若要逐头权重需另设参数。输出投影回128维，不能只凭它称整个模块就是完整Transformer层。batch_first与mask约定应参照实际库版本。

~~~python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
~~~

~~~text
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
~~~

## 七、失败模式、原文边界与工程注意

1. attention权重不自动构成解释，需要消融或反事实检验。
2. 单样本softmax代码未处理空序列、mask或batch。
3. K/V长度需一致，Q长度可以不同；多头embed_dim应能被num_heads整除。
4. 加性/乘性效果和成本受具体配置影响，原文“相同准确率”不是保证。
5. 经典attention缓解压缩瓶颈，不意味着RNN已变成并行Transformer。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
给定实现，列出B、Tq、Tk、dq、dk、dv和各投影矩阵；逐行推导分数、权重和输出；检查mask轴、全mask与padding；给形状和权重和测试。只修真正的维度错误，不用reshape掩盖语义不匹配。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：扩展masked softmax，在变长batch检查PAD权重为0、有效行和为1，并测试全PAD约定。
2. Medium：实现general形式的多头，说明拆分/拼接/输出投影，验证单头与基线一致。
3. Hard：给第9课GRU加入Bahdanau注意力，比较复制任务长度曲线。保持参数量、训练预算与数据划分可比较，不预写提升幅度。

## 十、术语与完成标准

query/key/value：查询、匹配键、被汇总值；alignment：输出位置到源位置的权重矩阵；additive：非线性投影打分；general：双线性打分；mask：排除无效或未来位置。

完成检查：

- 能逐行推导单样本与batch形状。
- 能手算源例softmax。
- 能区分多头权重平均与逐头权重。
- 能说明注意力图不是自动解释。

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

