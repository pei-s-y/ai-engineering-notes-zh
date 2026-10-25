# 09：序列到序列——编码器、解码器与Teacher Forcing

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/docs/en.md)。原题：Sequence-to-Sequence Models。

## 一、这一课解决什么问题？

分类输出一个标签，翻译/改写输出另一条可变长序列。本课用GRU编码器和GRU解码器搭出seq2seq，重点是状态、训练输入错位与停止条件，而不是几段类定义就拥有翻译能力。

## 二、核心概念与推导

Encoder输入(B,S)，输出逐位置状态(B,S,H)与末态(1,B,H)。基础Decoder仅由末态初始化，每步输入(B,1)的上一步token，输出(B,1,V_tgt)的下一词logits及新hidden。源词表与目标词表可以不同。

Teacher forcing：用目标上一个真实token而非上一步预测作为下一步输入。第0步输入BOS，对齐预测第0个目标token；tgt应含EOS但不再把BOS当首个监督目标。PAD=0，BOS/EOS应另分编号。

训练损失为逐步交叉熵，推理不再知道真实前缀，形成exposure bias。scheduled sampling改变教师输入比例，是可研究的缓解，不是必然消除偏差的标准答案。

基础模型把所有信息压进固定H的末态；attention允许每一步查看全部encoder states。长序列失败受模型容量、数据、优化共同影响，不能从维度固定推出“40词绝不可能记住”的定理。

### 2.9 用一个小例子把概念串起来（补充）

src=[3,4,5]、tgt=[3,4,5,EOS]时，第0步decoder输入BOS，监督目标3；第1步若teacher forcing则输入3、监督4；若不用则输入上一步argmax，不保证是3。把tgt写成[BOS,3,4,5]又不调整训练循环，会把目标整体错位。训练损失低但自由生成差，可能正是两种输入分布不一致。

## 三、操作前准备

需要PyTorch：
~~~bash
python -m pip install torch
~~~
原课没有准备src/tgt、词表或optimizer，下面提供一条最小复制任务的补充入口。不要用无关的词ID直接对照两份不同词表。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 编码器

Embedding→GRU。基础版本返回outputs但训练函数只取hidden。padding_idx只屏蔽embedding梯度，不自动让末态停在最后有效词；真实变长输入需pack/mask等处理。

~~~python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    # 编码器和解码器的返回形状见本节说明，注意两者时间维不同。
    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
~~~

### 4.2 逐步解码器

hidden跨步传递；GRU一次接收一个token。fc输出目标词表维度。这里没有attention，也没有输出概率校准。

~~~python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    # 编码器和解码器的返回形状见本节说明，注意两者时间维不同。
    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
~~~

### 4.3 带Teacher Forcing的训练

optimizer必须包含encoder和decoder参数。每个时间步抽一次use_teacher，这份代码对整批共同使用该决策。loss按时间累加后反传，返回时才除tgt_len；与直接优化平均损失的梯度尺度不同。input_token默认建在CPU，GPU训练需补device=src.device。所有目标都为PAD的时间步用默认mean损失可能出现NaN，应处理有效mask。

~~~python
# 原版按时间累加损失；CPU建BOS，真实GPU/变长训练需补device与mask。
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
~~~

### 4.4 贪心解码

no_grad关闭梯度但不自动eval。原代码仅在同一步全部样本都输出EOS时停止；若不同样本在不同步结束，需要逐样本finished标志，否则已完成样本仍会继续生成。max_len=0时cat空列表也会失败。

~~~python
@torch.no_grad()
# 原版仅检查同一步全批EOS，实际应追踪每个样本的finished状态。
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
~~~

### 4.5 复制任务与瓶颈观察

原文表格中的98/91/62/23%是参考示意，不是本笔记实测，也未由上方自动输出。需训练数据、长度控制和准确率定义：token准确率与整句完全正确率差别很大。

~~~text
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
~~~

## 五、把代码运行起来：补充入口与检查

定义上述类和训练函数后，可在CPU试通一批复制任务：
~~~python
torch.manual_seed(0)
encoder = Encoder(10, 8, 16)
decoder = Decoder(10, 8, 16)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=0.01
)
bos_id, eos_id = 1, 2
src = torch.tensor([[3, 4, 5], [5, 4, 3]])
tgt = torch.tensor([[3, 4, 5, 2], [5, 4, 3, 2]])
for _ in range(100):
    loss = train_batch(
        encoder, decoder, src, tgt, bos_id, optimizer,
        teacher_forcing_ratio=1.0
    )
encoder.eval()
decoder.eval()
print(greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=6))
~~~
需要PyTorch，未在本次运行。两条训练样本只能检查能否拟合和形状对齐；不能据此报告泛化复制能力。增加不同长度的训练及留出样本再评估。

## 六、Use It：使用现成工具

原文BART示例能演示AutoTokenizer、Seq2Seq模型和generate，但facebook/bart-base是基础去噪模型，不保证听从英文“Translate...”指令产生正确法语。实际任务需选择经过相应训练且语言覆盖匹配的检查点。beam保留多个候选，不保证绝对优于greedy；greedy也不是所有用户输出都必然不可靠。

~~~python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
~~~

## 七、失败模式、原文边界与工程注意

1. 不能用训练时teacher forcing的准确率代替自由生成评估。
2. EOS/PAD/BOS和目标错位错误比模型选择更常见。
3. beam比较序列对数分数，需说明长度惩罚；只加beam宽度不保证质量。
4. 原课提到minimum-risk、RL奖励等属于后续训练方法，不是本函数已经实现。
5. “不到百万平行句禁止从零训练”只是一种选型建议，不是实验许可门槛。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
为翻译/摘要/改写列出输入输出格式、配套tokenizer与检查点、BOS/EOS/PAD、训练损失与mask、解码策略和逐样本停止；检查长输出漂移与长度分层表现。不把示意复制准确率当实测。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：训练复制任务，比较5/10/20长度，并分别报告token与整句准确率。
2. Medium：实现beam=3，与greedy在同一平行语料上比较，记录长度与失败样本。
3. Hard：在约10k改写对上微调BART，按原文建议比较beam=4和基础模型。保持独立测试与相同生成参数，不只挑10个成功例子。

## 十、术语与完成标准

encoder/decoder：读入与生成网络；context：本课为编码末态；teacher forcing：真实前词输入；exposure bias：训练/推理前缀分布差异；beam：多候选序列搜索。

完成检查：

- 能写出src、tgt、hidden和logits形状。
- 能解释第0步为什么输入BOS。
- 能指出GPU device与批量EOS边界。
- 能区分拟合两条样本与泛化。

## 十一、验证范围与来源许可

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

