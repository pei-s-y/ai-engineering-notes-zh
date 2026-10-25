# 10：注意力机制——Bahdanau、Luong与QKV

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/10-attention-mechanism) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/10-attention-mechanism/docs/en.md)。原题：Attention Mechanism — The Breakthrough。

## 一、这一课解决什么问题？

基础seq2seq只给decoder一个末态。本课让每个输出步骤以当前查询对全部源状态打分，再做加权汇总。关键不在“看哪里”的比喻，而在分数、归一化轴、维度和mask。

## 二、核心概念与推导

令H形状(T_enc,d_h)，query=s形状(d_s)。加性打分：
$$e_i=v_a^\top\tanh(W_as+U_ah_i),\quad \alpha=\operatorname{softmax}(e),\quad c=\sum_i\alpha_i h_i$$
W_a为(d_attn,d_s)，U_a为(d_attn,d_h)，v_a为(d_attn,)。加性内部表示逐位置相加，投影后从d_attn压为一个标量。

dot直接H@s，要求d_s=d_h；general用W把维度接起来，W为(d_s,d_h)。双向encoder不意味着一定不能dot：若decoder状态维度匹配或增加投影，同样可用。

Q是查询，K用于打分，V用于汇总。cross-attention来自不同序列；self-attention来自同一序列的不同投影。Transformer常用softmax(QKᵀ/√d_k)V，并需位置表示、多头与其他层，不能把二者说成只是换字母。

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

