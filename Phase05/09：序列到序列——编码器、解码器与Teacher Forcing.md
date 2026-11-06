# 09：序列到序列——编码器、解码器与Teacher Forcing

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，概念细节与课程页一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/docs/en.md)。原题：Sequence-to-Sequence Models。

## 零、初学者先读：让模型学会写出一串话

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。第08课的网络读完句子只能吐一个分类标签；本课让它学会“写”：读进一句话，逐词产出另一句长度都不固定的话——这是翻译、摘要等一切生成任务的骨架，也是第10课注意力的出发点。

### 0.1 先理解这一课在做什么

分类是“读一段话，贴一个标签”；翻译是“读一句话，写出另一句连词表都不同的、长度也不固定的话”。seq2seq的解法出人意料地朴素：把第08课的两个循环网络接起来——

- **编码器（encoder）**：一个RNN把源句从头读到尾，读完后交出“一张便签”——最终隐藏状态，即上下文向量（context vector）。
- **解码器（decoder）**：另一个RNN拿这张便签初始化，之后每一步吃进上一个词，吐出对整个目标词表的下词分数，直到写出EOS停止。

类比：译员听完一整段话，在心里压成一张便签，然后凭便签逐词口译。麻烦也在这儿——便签只有固定大小，原文越长丢得越多（4.5的复制任务将量化这一点），这个“末态瓶颈”正是第10课注意力要解决的问题。

训练时还有个专门技巧：**Teacher forcing**——学徒练口译时，每写完一个词，下一步喂给他的不是他自己写的词，而是标准答案里的词。不这样，早期的错误会滚雪球。但考试（推理）时没有标准答案可喂，只能喂自己写的词——训练与推理的这种落差叫exposure bias。

### 0.2 本课术语先翻成白话

- seq2seq（sequence-to-sequence，序列到序列）=输入一条序列、输出另一条
- 编码器 encoder=读源句的RNN
- 解码器 decoder=写目标句的RNN
- 上下文向量 context vector=编码器的最终隐藏状态，源句的“压缩便签”
- BOS／EOS／PAD=句首标记／句尾标记／填充标记，应各占一个编号
- teacher forcing（教师强制）=训练时用真实上一个词当输入
- exposure bias（曝光偏差）=训练喂真词、推理喂自己的词，两者分布不一致
- greedy decoding（贪心解码）=每步都选分数最高的词
- beam search（束搜索）=同时保留前k条候选序列，最后挑总分最高的
- logits=对目标词表每个词的分数（softmax之前）
- scheduled sampling（计划采样）=训练中按概率掺入模型自己的预测当输入
- 复制任务 copy task=源句等于目标句的诊断任务，专测“背了再默写”

### 0.3 用小数字走一遍

设源句src=[[3,4]]（1个样本、2个词），嵌入维E=8、隐藏维H=16、目标词表V=10（与第五节补充入口的参数一致）。维度流转：

~~~text
src (1,2) ──嵌入──▶ (1,2,8) ──GRU──▶ outputs (1,2,16) ＋ hidden (1,1,16)
                                                  │
                                  hidden 就是交给解码器的“便签”
~~~

目标句tgt=[3,4,EOS]（设EOS=2）。解码器一次走一步，teacher forcing全程打开：

| 步 | 输入（喂给解码器的词） | 监督目标（这一步该输出谁） |
|---|---|---|
| 0 | BOS | 3 |
| 1 | 3（真词，teacher forcing） | 4 |
| 2 | 4（真词） | EOS |

第0步必须喂BOS：句子的第一个词“之前”没有词，BOS就是那个占位输入。若不用teacher forcing，第1步的输入是第0步自己预测的argmax——可能是3，也可能不是；错一步，后面的输入就再也不是“真正的上一词”。tgt含EOS是为了教模型“何时停”；tgt不含BOS，因为BOS只当输入、从不该被预测。每一步logits形状是(1,1,V)，与tgt当前步的词做交叉熵。若把tgt写成[BOS,3,4,5]又不改训练循环，监督目标会整体错一位——训练损失看着在降，学的却是错位的对应关系。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

4.1 Encoder对应0.3的编码半段：forward返回outputs形状(B,S,H)、hidden形状(1,B,H)，本课训练函数只取hidden当便签。4.2 Decoder对应解码半段：一次只接收(B,1)的一个token，GRU的hidden跨步传递。4.3 train_batch把0.3的表格变成循环——input_token先填BOS，loss_fn(logits.squeeze(1), tgt[:, t])就是“监督目标”那一列，use_teacher那两行决定下一步喂真词tgt[:, t]还是喂argmax。4.4 greedy_decode是推理版：没有任何真词可喂，全部用argmax，见EOS停止。4.5的复制任务（98/91/62/23%为原课参考）用实验展示末态瓶颈：长度越长，一张便签越装不下。第六节BART示例是现代版seq2seq——编码器解码器换成Transformer，但“读入→逐词生成”的外形与2014年原论文相同。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么第0步解码器的输入是BOS而不是tgt的第一个词？

**参考答案：** 第0步要在“什么都没写”的条件下预测第一个词，BOS是“句首之前”的占位输入；若喂tgt第一个词，监督目标就错位成了第二个词。

**问题2：** teacher forcing=1.0训练到损失很低，为什么自由生成还是可能一团糟？

**参考答案：** 训练时每步的输入都是正确的真词，模型从没练过“从自己的错误里恢复”；推理时只能吃自己上一步的输出，输入分布变了——这就是exposure bias，低训练损失不保证生成质量。

**问题3：** 编码器outputs里明明有每个位置的隐藏状态，为什么基础版只把hidden交给解码器？

**参考答案：** 基础seq2seq把整个源句压进末态hidden这张“便签”，逐位置outputs在本课被丢掉；第10课的注意力恰恰要把它们全部捡回来加权使用。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

分类输出一个标签，翻译/改写输出另一条可变长序列。本课用GRU编码器和GRU解码器搭出seq2seq，重点是状态、训练输入错位与停止条件，而不是几段类定义就拥有翻译能力。

## 二、核心概念与推导

Encoder输入(B,S)，输出逐位置状态(B,S,H)与末态(1,B,H)。基础Decoder仅由末态初始化，每步输入(B,1)的上一步token，输出(B,1,V_tgt)的下一词logits及新hidden。源词表与目标词表可以不同。

（补充 2026-11-06）**源/目标词表为什么可以不同。** 编码器的Embedding查源语言词表，解码器的Embedding查目标语言词表，两张表各自随机初始化、各自训练：

- 大小、嵌入维都可以不一样；
- 唯一的耦合点是GRU的hidden_dim必须一致——便签只有H格，读方写方都得按H格交接；
- 第03课的W是“一词一行”的表，这里只是“每张表各管一种语言”；
- 复制任务里两张表恰好相同（源=目标），才允许直接比对——第08课正文里“不要用无关的词ID直接对照两份不同词表”警告的正是把这个特例当通例。

（补充 2026-11-06）**“错位”是最常见的bug。** 0.3的表格是本课最容易写错的地方：第t步的输入是“t−1时刻的词”、监督目标是“t时刻的词”，一进一出正好错一位。

- 把tgt写成[BOS,...]再原样循环，等于每步都把答案提前喂了进去——损失降得飞快，模型什么都没学会；
- 这类错误比选错模型常见得多（第七节第2条的由来）；
- 排查方法就是手写一遍0.3的表格，逐行核对代码里的输入列和目标列。

Teacher forcing：用目标上一个真实token而非上一步预测作为下一步输入。第0步输入BOS，对齐预测第0个目标token；tgt应含EOS但不再把BOS当首个监督目标。PAD=0，BOS/EOS应另分编号。

（补充 2026-11-06）**Teacher forcing的力气从哪来。** 想象不用它：第0步预测错了一个词，第1步就被迫拿这个错词当输入，在错误的前缀上继续学——错误滚雪球，梯度信号一团乱。喂真词等于把“生成”暂时拆成“单步预测”：

- 每次都站在正确前缀上只练“下一词”，训练稳定得多；
- 代价是模型从没见过自己的错误分布，推理时第一次遇到自己写的词就可能跑偏——exposure bias不是bug，是这套训练法的固有副作用；
- scheduled sampling按概率逐步掺入模型自己的预测（本课代码用torch.rand逐时间步抽签），让模型提前练习“从错误中恢复”，是缓解而非根治。

（补充 2026-11-06）**PAD=0与ignore_index的配合。** tgt按最长句对齐后短句补PAD，CrossEntropyLoss(ignore_index=0)让这些位置不计损失：

- 但4.3的代码对“整批该时间步全为PAD”的极端情况仍可能算出NaN（分母为0），真实变长训练要用有效长度mask——这是中文说明里提醒补mask的原因；
- BOS/EOS/PAD各占独立编号（如1/2/0），混用一个编号会让“停止条件”和“填充”互相干扰；
- 三个编号的具体值可自定，但词表、训练循环、解码停止条件全流程必须用同一套约定。

训练损失为逐步交叉熵，推理不再知道真实前缀，形成exposure bias。scheduled sampling改变教师输入比例，是可研究的缓解，不是必然消除偏差的标准答案。

基础模型把所有信息压进固定H的末态；attention允许每一步查看全部encoder states。长序列失败受模型容量、数据、优化共同影响，不能从维度固定推出“40词绝不可能记住”的定理。

（补充 2026-11-06）**瓶颈的正确理解。** 便签比喻的准确版：信息在每个encoder位置都还在（outputs里逐位置都有），只是decoder看不到——不是“记不住”，是“交接时只准递一张纸”。

- 4.5的23%（原课参考）演示的是交接损失，不是记忆极限；
- H=16装40个词当然紧张，但把H加大也只是缓解，结构性出路是第10课让decoder每一步都翻看全部outputs；
- “不能从维度推出定理”同样要记住：40词23%是该玩具实验的参考数字，不是普适规律。

（补充 2026-11-06）**greedy与beam的分界。** greedy每步选最高分词，一旦选错无法反悔，长句容易重复打转；beam同时养k条候选（k=3～5常见），按整句总分收尾。但beam不是免费午餐：

- beam比较的是序列对数分数，需要长度惩罚配合（第七节第3条）；
- “beam一定优于greedy”在生成任务里并不总成立；
- 第11课翻译的num_beams、length_penalty正是这两个旋钮的现成接口。

（补充 2026-11-06）**为什么用复制任务考机器。** 复制任务（源句=目标句）是诊断末态瓶颈的最小实验：

- 它不需要语义、不需要两套词表对齐——机器只需“背下来再默写”，背不下来的部分就是便签容量的硬上限；
- 4.5把它按长度分层（5/10/20/40），正是把“最多记得住多长”变成一条可测曲线；
- 工程里拿到新的seq2seq实现，也常先用复制任务做冒烟测试：复制都学不会，翻译更不用谈。

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

