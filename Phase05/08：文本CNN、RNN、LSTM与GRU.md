# 08：文本CNN、RNN、LSTM与GRU

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，概念细节与课程页一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md)。原题：CNNs and RNNs for Text。

## 零、初学者先读：让网络学会按顺序读句子

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。第03、04课给了每个词一组数字，但“一整句话该怎么读”从这里才开始；本课是第09课“怎么写出序列”、第10课“怎么对齐”的起点。

### 0.1 先理解这一课在做什么

第02课的词袋把句子当一袋无序的词，dog bites man与man bites dog的表示一模一样；第03、04课的词向量只回答“每个词是什么”，仍没回答“词与词的先后”。本课把一串词向量交给两种会“读顺序”的网络（补充总览）：

| | TextCNN | RNN／LSTM／GRU |
|---|---|---|
| 怎么读 | 拿几个“短语模板”并行扫全句 | 从第一个词读到最后一个词，边读边记 |
| 像谁 | 扫关键词的阅卷老师 | 逐字读书、边读边记笔记的读者 |
| 输出 | 每种模板在全句的最高分 | 读完时笔记本里的隐藏状态 |
| 长短处 | 快、可并行，只看局部窗口 | 能带长程信息，但串行、难训练 |

1. **TextCNN**：宽度3的滤波器就是一个“可学习的三连词探测器”，沿句子滑动打分；最大池化只留最高分——“not good”出现在句首还是句尾，同一个探测器都能抓住（原课对“位置无关”的说明）。
2. **循环网络**：隐藏状态像一本笔记本，每读一个词更新一次；读完，笔记本就是全句摘要。朴素RNN的笔记会“越翻越淡”（梯度消失），LSTM加三个门管理记忆，GRU用两个门做简化版。
3. 本课结尾点出三类问题（串行慢、末态压缩、远距离衰减），正是第09、10课逐个接住的伏笔。

### 0.2 本课术语先翻成白话

- TextCNN（文本卷积网络）=在词向量序列上做一维卷积的分类网络，Kim 2014
- 滤波器 filter=一个可学习的“n连词模板”，宽度k就对连续k个词打分
- 卷积 convolution=模板沿句子滑动、逐位置打分
- 最大池化 max-pooling=只保留每个模板在全句的最高分，丢掉“出现在哪”
- 隐藏状态 hidden state=循环网络的“笔记本”，读到哪里记到哪里
- 时间步 time step=读到第几个词，第t步
- 梯度消失 vanishing gradient=训练信号往回传时被反复乘小于1的数，传到早期几乎为零
- 门 gate=LSTM／GRU里取值0～1的开关，决定忘多少、记多少、说多少
- 细胞状态 cell state=LSTM单独一条“记忆传送带”，加法更新、信号不易淡
- 双向 bidirectional=一个从左读、一个从右读，两边信息拼接
- logits=softmax之前的类别分数，不是概率
- BPTT（沿时间反向传播）=训练信号沿时间步逐格往回传，梯度消失就发生在这条路上
- dropout=训练时随机屏蔽一部分连接的正则手段，本课用在分类头之前

### 0.3 用小数字走一遍

**RNN两个时间步。** 把状态更新简化到一维（真实是向量）：h_t = tanh(0.7·x_t + 0.5·h_(t−1))。设h_0=0，第一个词x_1=1，第二个词x_2=0：

~~~text
h_1 = tanh(0.7×1 + 0.5×0)      = tanh(0.7)   ≈ 0.604
h_2 = tanh(0.7×0 + 0.5×0.604)  = tanh(0.302) ≈ 0.293
~~~

x_2明明是0，h_2却不是0——0.293全是x_1的残留，这就是“状态在传信息”。但每传一步残留都被乘一次0.5，越传越淡；对照正文4.3，权重0.9连乘100步只剩约2.7×10^-5（原课参考）。梯度消失就是这件事发生在训练信号上。

**LSTM三个门的直觉数字例。** 门都是0～1的系数。设遗忘门f=0.9（九成旧记忆放行）、输入门i=0.1（一成新信息写入）、候选g=0.5、旧记忆c_(t−1)=1.0，则传送带上：c_t = 0.9×1.0 + 0.1×0.5 = 0.95——九成旧的加一成新的，是加法不是连乘；若输出门o=0.8，对外读数h_t = 0.8×tanh(0.95) ≈ 0.592。连乘会指数衰减，“旧的乘0.9、新的加一点”能撑很久——这就是LSTM缓解梯度消失的机制（缓解，不是根治）。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

4.1的TextCNN按“嵌入→转置→多宽度卷积→ReLU→全局max→拼接→线性层”走：transpose(1,2)是因为Conv1d把中间那维当通道，要把(B,T,E)换成(B,E,T)；宽度k的卷积输出长度是T−k+1，池化后回到固定(B,F)。4.2的LSTMClassifier对应0.3的循环读法，out.max(dim=1)在时间维取最大，对应0.1“位置无关”的池化思想；双向时每位置输出是两个方向拼接的2H维。4.3的vanishing_gradient_sim就是0.3第二个观察的代码版。第六节BertCNN只是把“嵌入”换成冻结的BERT编码，卷积流水线原样。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么max-pooling让TextCNN不关心“not good”出现在句首还是句尾？

**参考答案：** 滤波器在所有位置都打分，池化只取最高分、丢掉位置信息；模板在任何位置命中一次，激活就被保留。

**问题2：** x_2=0时h_2为什么不是0？

**参考答案：** h_2由当前输入项（0.7×0=0）和状态项（0.5×h_1≈0.302）相加决定，后者带着x_1的残留进tanh，结果≈0.293。循环网络的“记忆”就是这条状态项。

**问题3：** LSTM的细胞状态和朴素RNN的隐藏状态，更新方式差在哪？

**参考答案：** 朴素RNN每步整体重算，长程信号经过连乘衰减快；LSTM的c_t = f⊙c_(t−1) + i⊙g，旧记忆按门缩放后加上新信息，梯度沿这条加法通路传得更远。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

词袋丢失顺序，静态词向量单独查表也不组织整句结构。本课把词向量序列送入TextCNN或循环网络，为分类得到固定维表示；借此理解局部模式、递归状态和长距离信息的问题。

## 二、核心概念与推导

输入token_ids形状(B,T)，Embedding后为(B,T,E)。TextCNN要求Conv1d输入(B,E,T)，所以transpose不是随意调维：E是通道，T是滑动位置。宽度k的卷积输出长度T−k+1，每种宽度F个滤波器，最大池化后(B,F)，拼接m种宽度得到(B,mF)，再映射类别logits。

（补充 2026-11-06）**滤波器就是“可学习的短语探测器”。** 一个宽度3的滤波器有E×3个权重加一个偏置，作用在任意连续3个词的向量拼接上，输出一个分数——分数高表示“这3个词长成了我负责的那个模式”。训练前它对什么敏感是随机的，训练中分类损失会把它调成对任务有用的模式：情感分类里，某些滤波器会自然学会对not good这类组合给高分。

形状全链路用一个可手算的例子走一遍（与2.9同一例子）：

~~~text
token_ids (2,6) ──嵌入──▶ (2,6,8) ──转置──▶ (2,8,6)
  ──宽度3卷积、4个滤波器──▶ (2,4,4)    （6−3+1=4个滑动位置）
  ──时间维max──▶ (2,4) ──m种宽度拼接──▶ (B,mF) ──线性层──▶ (B,C)
~~~

- ReLU在池化前：把“负相关”统一截成0，池化选的是最强正激活；
- dropout放在拼接之后、分类头之前：正则的是最终表示而不是卷积特征；
- 最终维度只与滤波器个数有关、与句子长度无关——这就是“任意长度输入、固定长度输出”的实现方式，也是第02课TF-IDF以外另一种把变长文本变成定长向量的路线。

（补充 2026-11-06）**padding_idx的边界。** 代码里nn.Embedding(..., padding_idx=0)只保证PAD编号的词向量恒为零向量、且它的梯度不再更新。它**不会**自动屏蔽包含PAD的卷积窗口：

- 一个跨到填充区的窗口照样打分，池化也可能选中它；
- 变长batch要么按长度分组，要么在池化处用mask把PAD位置设成负无穷（第06课的masked思想）；
- 否则短句的真实特征可能被邻居句子的PAD污染——这也是第七节第4条要求按长度分层检查准确率的原因。

RNN共享时间参数，h_t=f(Wx_t+Uh_(t−1)+b)。LSTM增加细胞状态与门；GRU使用更新/重置机制，减少参数但不是必然同精度。双向网络可读左右上下文，不能直接用于严格因果实时生成。

（补充 2026-11-06）**三个门各管什么（对照表）。** 把LSTM想成一个带三条规则的档案员，c_t = f⊙c_(t−1) + i⊙g、h_t = o⊙tanh(c_t)：

| 门 | 0～1的系数管什么 | 白话 |
|---|---|---|
| 遗忘门 f | 旧记忆c_(t−1)保留几成 | 过时的划掉 |
| 输入门 i | 新候选g写入几成 | 重要的记下 |
| 输出门 o | 记忆对外说几成 | 需要时才讲 |

数字例见0.3：f=0.9、i=0.1时，旧记忆乘0.9、新信息加0.1——0.95里九成是旧的。“共享时间参数”指W、U、b在读第1个词和第100个词时是同一套：学会的“更新规则”全句通用，这正是与TextCNN的本质分工——卷积学“哪些模式存在”，循环学“怎样边读边更新状态”。

- GRU把三条并成两条：更新门z同时扮演“遗忘+写入”（两者互补的绑定设计），重置门r决定算新候选时参考多少旧状态；参数更少、常与LSTM同精度，但“必然同精度”不成立（原课边界）；
- 双向网络给每个位置拼接左右两个方向的读数，适合分类与标注（第06课BiLSTM-CRF正是它）；
- 但“从右往左读”要求右边先存在，严格因果的实时生成（边收词边出词）用不了它。

（补充 2026-11-06）**为什么4.2用max池化而不是取末状态。** 原课给出的经验理由：长序列末端的信息会主导最后一个隐藏状态，max在时间维取“全程最强激活”，不被位置偏置。两种池化的分工：

- max／mean池化：适合分类——要的是“全句有没有出现过这个信号”；
- 取末态：适合生成——第09课的解码器必须拿到“读完整句后的总结”当便签。

这是待验证的经验而非定理（4.2的中文说明也标注了）；两种用法各归其位，不能互相替代。

0.9^T只是在标量线性链上的梯度衰减直觉；真实梯度还包含雅可比、激活和门。LSTM有缓解作用，不保证任意长度训练稳定。

（补充 2026-11-06）**“权重小于1”不是判据。** 0.9^T把梯度衰减压缩成一个数，便于建立直觉，但真实网络里梯度是雅可比矩阵的连乘，混合了激活函数导数与各门系数——单个recurrent weight是0.95还是1.1，推不出训练稳定与否。LSTM的传送带是加法更新，等于给梯度修了一条“高速公路”，缓解长程衰减；但门控不是免死金牌：

- 超长序列仍需梯度裁剪、层归一化等工程手段配合（原课边界：不保证任意长度稳定）；
- 原课4.4点出的三类问题里（串行、末态压缩、远距离），LSTM真正缓解过的只有第三类；
- 前两类要等第10课的注意力、第07阶段的Transformer逐条接手。

### 2.9 用一个小例子把概念串起来（补充）

若输入token_ids为(2,6)，embed_dim=8，则embedding为(2,6,8)，CNN转置为(2,8,6)。宽度3、4个filter得到(2,4,4)，max-pool后(2,4)；多种宽度拼接才送分类头。双向LSTM hidden_dim=16会在每位置得到32维输出。本课结构输出类别logits，不会自动输出一段文本或准确率。

## 三、操作前准备

原文神经模型需PyTorch；BertCNN另需transformers与检查点下载。
~~~bash
python -m pip install torch
~~~
CPU可做下面形状检查，不需要为本课先配置CUDA。训练任务还需词表、PAD编号0、等长批次或变长处理、标签、损失和优化器。模型类本身没有训练数据和训练循环。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 TextCNN的形状路径

多宽度Conv1d→ReLU→全局max→拼接→dropout→线性层。输入长度至少达到最大filter_width，否则卷积报错。padding_idx=0只让PAD词向量固定为零，不会自动屏蔽包含PAD的卷积窗口；池化仍可能选中这些位置。

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    # 依照本节说明核对batch、时间、通道维，原版未完整mask PAD。
    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
~~~

### 4.2 LSTM分类器

batch_first让输出为(B,T,H×方向数)。max(dim=1)在时间维池化，不是取最后一层hidden。它比last-state更好只是待验证的经验，非保证。padding位置同样要mask；循环经过零词向量后状态未必为零。

~~~python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    # 依照本节说明核对batch、时间、通道维，原版未完整mask PAD。
    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
~~~

### 4.3 用标量幂理解梯度衰减

这段函数只计算recurrent_weight^seq_len，不训练RNN。0.9^100约2.656×10^-5；1.1^100会放大。真实网络还取决于激活导数、参数谱与门控，不能仅检查单个权重是否小于1。

~~~python
# 只计算标量幂作为梯度传播直觉，不是RNN训练实验。
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
~~~

### 4.4 为什么继续研究注意力

原课指出串行时间步、encoder末态压缩及远距离信息三类问题。经典RNN加attention主要缓解压缩瓶颈，本身仍有递归串行性；Transformer去掉递归才改变跨位置并行方式。不能说加入任意attention就自动解决全部问题。



## 五、把代码运行起来：补充入口与检查

先执行两类模型定义，再做不训练的形状检查（本次未安装PyTorch运行）：
~~~python
torch.manual_seed(0)
ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
cnn = TextCNN(vocab_size=20, embed_dim=8, n_classes=3)
rnn = LSTMClassifier(20, 8, 6, 3)
cnn.eval()
rnn.eval()
with torch.no_grad():
    assert cnn(ids).shape == (2, 3)
    assert rnn(ids).shape == (2, 3)
print(vanishing_gradient_sim(100))
~~~
输出(B,C)是类别分数，不是概率。训练时可直接交给CrossEntropyLoss；先加softmax再交给该损失是常见错误。训练与评估应切换train()/eval()并保留独立验证集。

## 六、Use It：使用现成工具

原文BertCNN先下载BERT，并冻结参数；这不等于只加载静态词表，而是每次仍需运行完整上下文编码器。requires_grad=False或no_grad不等于eval，训练模式的dropout仍可能随机。源例的卷积在池化列表中重复计算两次，可在补充实现中缓存，注意说明改动。外部调用未执行。

~~~python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    # 依照本节说明核对batch、时间、通道维，原版未完整mask PAD。
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
~~~

## 七、失败模式、原文边界与工程注意

1. “CNN小10～100倍”“CPU5分钟”需给具体配置，不是本机验证。
2. 因果Transformer也可用KV缓存增量生成，并非所有Transformer都要每次重算完整序列。
3. 卷积局部建模并不保留所有全局位置关系；max-pool丢弃出现位置。
4. 变长batch要处理PAD；检查按文本长度分层的准确率。
5. 未在500样本上跑过CNN基线不等于禁止使用预训练模型，数据量阈值不是定理。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
根据任务、输入长度、延迟和部署约束选择TextCNN、BiLSTM或预训练编码器；明确词表、embedding、PAD mask和输出维度；给优化器/损失/验证方案；检查长短文本、OOV和显存。没有跑训练时，不填写虚构F1。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：自建3类玩具数据，比较多宽度与单宽度，多种子报告macro-F1；不强求多宽度必胜。
2. Medium：实现masked max、mean与last-state池化，比较同一划分结果并解释PAD处理。
3. Hard：结合第6课补齐BiLSTM-CRF，在CoNLL-2003与CRF和BERT比较F1、训练时间及内存。

## 十、术语与完成标准

TextCNN：词向量序列上的一维卷积；hidden：递归状态；cell：LSTM细胞状态；bidirectional：双向上下文；pooling：序列到固定维汇总；logits：分类归一化前分数。

完成检查：

- 能逐步写出(B,T)到(B,C)形状。
- 能指出池化维度与PAD风险。
- 能区分形状检查和训练效果。
- 能解释注意力与去掉递归不是同一件事。

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

