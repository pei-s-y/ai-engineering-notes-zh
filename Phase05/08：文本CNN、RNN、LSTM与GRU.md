# 08：文本CNN、RNN、LSTM与GRU

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md)。原题：CNNs and RNNs for Text。

## 一、这一课解决什么问题？

词袋丢失顺序，静态词向量单独查表也不组织整句结构。本课把词向量序列送入TextCNN或循环网络，为分类得到固定维表示；借此理解局部模式、递归状态和长距离信息的问题。

## 二、核心概念与推导

输入token_ids形状(B,T)，Embedding后为(B,T,E)。TextCNN要求Conv1d输入(B,E,T)，所以transpose不是随意调维：E是通道，T是滑动位置。宽度k的卷积输出长度T−k+1，每种宽度F个滤波器，最大池化后(B,F)，拼接m种宽度得到(B,mF)，再映射类别logits。

RNN共享时间参数，h_t=f(Wx_t+Uh_(t−1)+b)。LSTM增加细胞状态与门；GRU使用更新/重置机制，减少参数但不是必然同精度。双向网络可读左右上下文，不能直接用于严格因果实时生成。

0.9^T只是在标量线性链上的梯度衰减直觉；真实梯度还包含雅可比、激活和门。LSTM有缓解作用，不保证任意长度训练稳定。

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

