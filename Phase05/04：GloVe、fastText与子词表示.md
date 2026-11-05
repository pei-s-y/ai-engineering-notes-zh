# 04：GloVe、fastText与子词表示

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-04 增补「零、初学者先读」、概念讲解、选型表与延伸阅读；当日与原课网页版核对，Build It 代码与三步结构一致。
> 2026-11-05 补充：0.1 增“三个洞”总览表、在线vs全局两条学习路线类比、fastText片段拆解展开、BPE两难与流水线层级图；0.2 增GloVe/fastText/Word2Vec名字来历；2.2 增“片段向量不是迷你词义”与“参数共享=家族绑定”机制、形近≈义近赌注（united/untied反例）、求和的取舍；2.1 增“猜-乘-比-调”循环与迷你手算、“乘积格=03课点积”洞察、损失三件套、路线之争终结与“先压缩再学习”；2.3 增BPE合并全过程手算（含生词切分演示）、名字来历、</w>作用与“频次即预算”。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md)。原题：GloVe, FastText, and Subword Embeddings。

## 零、初学者先读：三个工具各自接住上一课的哪个问题

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要回看词向量是什么时，先读[第03课](<03：词向量与从零实现 Word2Vec.md>)的零节。

### 0.1 先理解这一课在做什么

[第03课](<03：词向量与从零实现 Word2Vec.md>)留下几个没解决的问题，本课的三件事分别接住它们（补充总览）：

| 03课留下的洞 | 谁来接 |
|---|---|
| 训练方式只能“边读边学”吗？ | GloVe |
| OOV：没见过的词查不到向量 | fastText |
| 词表撑不到百万那么大 | BPE |

1. Word2Vec是一边读句子一边更新参数（在线学习）。另一派研究（LSA等）直接分解共现统计表。两条路线谁更好？**GloVe**的回答是：设计好损失的矩阵分解能匹敌甚至超过Word2Vec，还更便宜。

（补充展开）“在线学习”就是03课0.3手算的那种：读到哪学到哪、路过一次学一次，从不真的去数cat和sat共现了多少次。另一派相反：先把整个语料的共现数完，做成一张大表X[i][j]（词j在词i周围出现的加权次数），再找两张瘦表——形状正好就是03课的W和W'——让它们“乘起来”尽量还原大表，这就是**矩阵分解**。类比：在线学习像边看小说边记笔记，看一页记一页；矩阵分解像先通读全书做完整索引、再回头总结规律。GloVe的贡献是证明：把损失设计好（拟合共现次数的对数、按频次加权，见2.1），这条老路线能匹敌甚至超过Word2Vec，还更便宜——统计数一遍就完，不用反复过语料。

2. 训练时没见过的词怎么办？**fastText**的答案：把词拆成字符片段，词向量=片段向量之和；新词只要片段见过就能拼出向量。

（补充展开）即不给“整个词”学向量，给“字符片段”学向量：where = <wh + whe + her + ere + re> + <where>，片段们各有各的向量，加起来就是where的向量。好处马上显现：来了没见过的whereupon，拆开发现her、ere这些片段训练时都见过——向量拼得出来。附赠一个好处：词形变化play/playing/played共享大量片段，向量天然相近，对低频词形尤其救命。边界（见0.3）：whereupon靠近where只是“参数可复用”，不证明两词同义。

3. 到了Transformer时代，词级词表撑不到上百万那么大，语言却是开放的。**BPE**学习“常见片段当单位”的切分规则，任何文本都切得开、永不OOV——现代每个LLM分词器都是它的变体。

（补充展开）BPE面对的两难：词级词表要么撑爆（百万词）要么大量OOV；字符级永不缺词但太碎、几乎不含语义。它的答案是学一套切分规则让单位大小自动适配——从字符开始，每次把当前最常见的相邻对绑成新单位（0.3有lowest的例子），常见块成为整单位、罕见块拆成小块，最差也能退回字符。

注意三者不是同一层面的工具：GloVe和fastText学的是**词向量**，BPE学的是**切分规则**，不产生任何向量。（补充）用流水线看最清楚：

~~~text
文本 ──BPE规则──▶ 基本单位（token） ──embedding层──▶ 每个单位的向量
      （本课第三部分）              （03课Word2Vec / 本课GloVe、fastText）
~~~

BPE在流水线上游，只决定“基本单位是什么”，一个向量都不产生。问“BPE和Word2Vec哪个好”是类别错误——像问“菜刀和红烧肉哪个好吃”。

### 0.2 本课术语先翻成白话

- 共现矩阵 co-occurrence matrix=一张大表，X[i][j]记词j在词i周围出现多少次
- 矩阵分解 factorization=找两组小表，让它们乘起来尽量还原那张大表
- （补充）名字来历：GloVe=**Glo**bal **Ve**ctors（全局向量），名字直接宣传它“用全局共现统计”的路线；fastText=fast+Text（快速文本），宣传它轻量快速；Word2Vec=Word to Vector（词→向量），见[第03课](<03：词向量与从零实现 Word2Vec.md>)0.2；BPE=Byte Pair Encoding（字节对编码），原是1994年的数据压缩算法名（来历详见2.3）
- 字符n-gram=连续n个字符的片段，如where里的whe、her
- 词形变化 morphological variants=同一个词的不同形式，play/playing/played
- 合并 merge=BPE把当前最常见的相邻单位对绑定成一个新单位
- `</w>`=词尾标记，让BPE知道一个片段是否顶到词的末尾
- byte fallback=遇到不认识的字符，退回用原始字节表示
- OOV=词表里没有的词

### 0.3 用小数字走一遍

语料只有`cat sat`与`cat slept`、window=1：cat与sat共现一次，cat与slept共现一次；距离1的贡献是1，距离2的贡献是1/2——GloVe先把这些加权计数**全部做完**，再训练向量去拟合。若某对的“点积+偏置”算出1.5而log(1)=0，误差1.5就是要修正的量，更新会把它推低。

fastText：where拆成`<wh`、`whe`、`her`、`ere`、`re>`、`<where>`等片段；未见词whereupon与where共享her、ere等片段，向量因此靠近——共享只代表参数可复用，不证明两词同义。

BPE：lowest初始切成l,o,w,e,s,t,</w>；若(l,o)、(e,s)等对在语料别处高频，会被先合并，lowest最终可能切成low + est</w>。（补充）完整合并过程与生词切分演示见2.3。

以上是用于理解的算例，不是新增的训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

三部分互相独立，按顺序读：build_cooccurrence + glove_train（GloVe）；char_ngrams + fasttext_vector（fastText）；learn_bpe + apply_bpe（BPE）。与第03课的关键区别：BPE的输入不是句子列表，是“词→频数”的Counter；GloVe的训练对象是共现计数字典，不是词对列表。

### 0.5 理解检查（含参考答案）

**问题1：** GloVe和Word2Vec最根本的差别是什么？

**参考答案：** Word2Vec在线更新（遇到一个词对更新一次），GloVe先把全局共现统计做完，再让向量拟合对数共现；效果相近，GloVe训练成本更低。

**问题2：** 为什么fastText能处理没见过的词？

**参考答案：** 词向量是字符片段向量之和；新词只要片段训练时见过就能拼出向量，全部片段都没见过才返回None。

**问题3：** BPE学的是词向量吗？

**参考答案：** 不是。BPE学的是“怎样切词”的有序合并规则；向量是后续模型embedding层的事，两者不能混。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

Word2Vec之后还要回答：能否直接利用全局共现统计？没见过的词怎样表示？本课依次实现GloVe、字符n-gram组成向量、带词尾标记的字符级BPE。三者不是同一层面的工具：前两者学词向量，BPE学切分规则。

## 二、核心概念与推导

### 2.1 GloVe：先把全局共现数完，再拟合

原课的动机脉络：Word2Vec之前已有直接分解共现矩阵的路线（LSA、HAL），GloVe证明带精心设计损失的矩阵分解可以匹敌甚至超过Word2Vec，且训练更便宜。

（补充）这句话四个词组值得逐个拆开：**矩阵分解**、**设计好损失**、**匹敌甚至超过**、**更便宜**。先看矩阵分解在做什么：找两张瘦表（W和W'，形状就是03课的那两张），让它们的**乘积**尽量还原共现大表X。玩法是个循环：

~~~text
① 猜两张瘦表（随机数）
② 相乘，得到“预测的共现表”
③ 和真表对比，逐格算误差
④ 按误差微调瘦表里的数
⑤ 回到②，直到误差够小
~~~

（补充，迷你手算一轮）设迷你语料数出的X如下（行=中心词，列=邻居词）：

~~~text
            sat   slept
   cat  [    4      2  ]
   dog  [    8      1  ]
~~~

取d=1维（真实是100～300维），猜W：cat→0.9、dog→1.5；W'：sat→1.2、slept→0.4。回归目标是log(X)：cat–sat为log4≈1.39、cat–slept为log2≈0.69、dog–sat为log8≈2.08、dog–slept为log1=0。乘出来：cat–sat=0.9×1.2=1.08（比1.39低，要推高）；dog–slept=1.5×0.4=0.60（比0高，要压低）。微调、再乘、再比——这就是训练的全部动作，也是0.3里“误差就是要修正的量”的出处。

（补充，关键洞察）乘积表的第(i,j)格=行i向量·列j向量=**v_i·u_j，正是03课的打分点积**。所以矩阵分解不是新数学——是把03课的点积排成一张网格，回归到log共现次数上。

GloVe先统计X_ij：在词i的窗口中，词j出现的加权次数，窗口距离r贡献1/r。补充（原文未展开）：目标是log(X_ij)而不是X_ij本身，因为词频跨度极大——the可能共现百万次而专有名词只共现两次，对数把乘性差异压成加性，回归目标数值才稳定。训练目标可以写成：
$$J=\sum_{i,j:X_{ij}>0}f(X_{ij})(w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$$
逐项读：对每个共现过的词对，希望“点积+两个偏置”接近log共现次数；f(X_ij)是权重函数，(x/x_max)^0.75在低频段随共现次数上升、超过x_max后饱和为1，避免(the,and)这类高频对主导损失。偏置吸收频率差异，向量解释共现结构。代码系数少一个平方损失的常数2，可等价吸收到学习率中。最终词向量取W+W_tilde之和，这是原课提到的已发表技巧，通常优于只用一张。

（补充）“设计好损失”的三件套在公式里各管一件事：①取log压动态范围（上段已述）；②f是音量旋钮——低频对降权（共现一两次的对多半是噪声，不该和共现上万次的对一样大声说话），高频封顶（出现9000次还是9100次已无新信息），且f(0)=0：从没共现过的对直接跳过，零格根本不进损失；③偏置先把“这个词天生高频”记走，向量不浪费容量在频率上——即上句“偏置吸收频率差异，向量解释共现结构”。

（补充）“匹敌甚至超过”的背景：2013年Word2Vec效果惊艳，在线预测派压过数表分解派；2014年GloVe论文用实验证明老路线配上好损失不输、部分任务（词类比）更好，终结了路线之争。后来的共识是两边调好超参后成绩相当——两条路通向几乎同一个终点：都在让点积反映共现统计，只是一个边走边学从不建表，一个先建完表再校准。

（补充）“更便宜”一句话：**先压缩，再学习**。语料里the cat相邻共现一万次，Word2Vec要过一万张卡片，GloVe压成X[the][cat]=10000一个条目，一万这个数直接当回归目标用；语料只读一遍（建X用），之后训练只碰X的非零格，而Word2Vec通常要在全语料上跑多轮epoch；也不需要负采样——那是Word2Vec绕开softmax的补丁，GloVe从没用过softmax。边界：X是V×V的稀疏大表，构建和存储要工程处理（0.4说的“训练对象是共现计数字典”即此意）；“便宜”指典型设置下的训练开销，不是没有成本。

### 2.2 fastText：词是它的片段之和

fastText把词包上边界< >，提取3～6字符片段，词向量=所有片段向量之和，再按Word2Vec的方式训练。未见词若有已知片段可获得向量；所有片段都不认识时，本教学函数返回None。

（补充）最容易误解的一点：**片段向量不是“迷你词义”**——没有任何环节要求过片段“有自己的意思”。与Word2Vec对比着看，fastText只改了一个零件：

| | Word2Vec | fastText |
|---|---|---|
| 词的向量是 | W表里**一整行自由参数** | **若干片段参数之和**（本身不是自由参数） |
| 训练任务 | 猜邻居 | 猜邻居（一模一样） |
| 判卷对象 | 词向量 | **片段之和** |

判卷只看总和、从不看单个片段——那“和”怎么变得有意义？看训练动力（补充，小例子）。设片段表有pl、lay、er、ing等：

~~~text
player  的向量 = v<pl> + v<lay> + v<er>
playing 的向量 = v<pl> + v<lay> + v<ing>
rain    的向量 = v<ai> + v<n> + ...
~~~

卡片(player, game)的误差说“player的**总和**该靠近game的上下文向量”，梯度分摊回pl、lay、er；卡片(playing, game)又推pl、lay（共同片段）。于是pl、lay被play家族的所有词反复拉扯，收敛到“对这一族都有用的平均贡献”；rain的片段不受这些更新影响。**参数共享=家族绑定**：play/player/playing的向量大部分项相同，自然挤在一起。类比合唱团：每个话筒单独听都是噪声，没有一个话筒“代表这首歌”；调音师（训练）只听混音台的总信号（和），把每个推子调到让整场合唱都过得去的位置——推子位置（片段向量）有没有“意思”，从头到尾没人检查。

（补充）所以OOV的本质是一个赌注：**形近≈义近**。whereupon与where共享her、ere等大部分片段，和的大部分项相同，向量靠近——对词形变化家族（where/whereupon、play/played）这个赌注赢面大；对任意词就露馅：united与untied片段几乎全同、意思毫不相干；生僻乱码词的片段大多没见过，拼出来是垃圾（全部片段没见过才返回None）。

补充（原文未展开）：这一设计对词形丰富的语言收益最大——芬兰语、土耳其语一个词根能变化出成千上万形式，词级词表存不下，片段级能全部复用。（补充）代价与收益对偶：高频词的向量被无关片段掺入噪声，fastText有时略逊于Word2Vec；罕见词、OOV、形态丰富的语言上大幅更好——这正是它存在的理由。至于为什么用“求和”这么简单的组合：快（名字里的fast），训练和查询都只是查表加法；更讲究的字符组合（如对字符跑小网络）存在但慢。

### 2.3 BPE：学习切分规则，不是学向量

本课BPE输入是“词→频数”的Counter；以字符及</w>开始，不是256字节初始化。算法一句话：数所有相邻片段对的加权频次，把最常见的合并成新单位，重复k次。结果：高频片段（ing、tion、est）成为单token，罕见词拆成熟悉碎片，任何文本都切得开。真实GPT/BERT/T5分词器学习3万～10万次合并，序列长度有界、无OOV。

（补充）留意输入形态：词频表而不是句子列表——与GloVe同一个哲学：先压缩统计，再在统计量上工作，不逐句过语料。

（补充，把算法完整走一遍——经典例子，Sennrich et al. 2016）语料压成词频表：

~~~text
low ×5    lower ×2    newest ×6    widest ×3
~~~

第0步全拆成字符（</w>为词尾标记）：

~~~text
l,o,w,</w>          ×5
l,o,w,e,r,</w>      ×2
n,e,w,e,s,t,</w>    ×6
w,i,d,e,s,t,</w>    ×3
~~~

第1步数相邻对、按词频加权（newest有6个，它内部的每个对都计6次）：

| 相邻对 | 次数 |
|---|---|
| (e,s) | 6+3=9（newest、widest里各一） |
| (s,t) | 9 |
| (t,</w>) | 9 |
| (w,e) | 2+6=8（lower、newest里各一） |
| (l,o) | 5+2=7 |
| (o,w) | 7 |
| 其余 | ≤6 |

三并列9，平手任选其一（实现按固定顺序打破，保证可复现）。选(e,s)→es。第2步在合并后的新表上重新数对，(es,t)=9最高→est；第3步(est,</w>)=9→est</w>；之后(l,o)=7→lo、(lo,w)=7→low。设k=5，停手。

（补充）学到的不是一个词表，是一张**有序规则清单**：(e,s)→es、(es,t)→est、(est,</w>)→est</w>、(l,o)→lo、(lo,w)→low。切新词=按学习顺序逐条套、套无可套为止。切两个语料里没有的词：

~~~text
lowest:   l,o,w,e,s,t,</w>
        → l,o,w,es,t,</w>      （套(e,s)→es）
        → l,o,w,est,</w>       （套(es,t)→est）
        → l,o,w,est</w>        （套(est,</w>)）
        → lo,w,est</w>         （套(l,o)→lo）
        → low,est</w>          （套(lo,w)→low）

slowest:  s,l,o,w,e,s,t,</w> → … → s,low,est</w>
~~~

0.3的草图至此闭环：lowest正是low + est</w>——而且它是生词，规则照样切得开。

它学习有序合并规则，不自动学embedding。字符级训练字母表不完整时，不能保证无OOV；真正永不OOV的是字节级BPE（GPT-2方案：从256个字节起步，任何字符至少落到字节上）。不要把本课字符版与完整字节覆盖的BPE混为一谈。

（补充，三个为什么）
- 名字来历：Byte Pair Encoding（字节对编码）本是1994年的数据压缩算法——文件里最常见的字节对，用一个闲置字节代替，文件就变短；2016年被借来学词表——最常见的字符对，用一个新词表条目代替。“把高频搭配缩写成符号”就是压缩，名字没骗人。
- </w>的作用：切完之后空格就没了，词边界必须活在符号本身里。est</w>只匹配顶到词尾的est——widest里的est（词尾）与establish里的est（词中间）是两个不同对象，不标就会混。
- 单位大小为什么自动适配：频次就是预算——low出现得多，挣到整词符号（序列短）；slowest没见过，拆成熟悉碎片（覆盖全）。k是总旋钮：词表大小≈初始符号数+合并次数k。

（补充，边界）合并只认频次、不认词法：lowest切成low+est</w>是巧合的漂亮，大量词的切分毫无词法道理——模型数strawberry里有几个r数不对的梗，根源就是token与字母不一一对应；平手时任选也意味着规则对语料细节敏感，所以实现必须用固定的打破顺序保证可复现。

### 2.4 用一个小例子把概念串起来（补充）

假设语料只有 `cat sat` 与 `cat slept`，window=1：cat与sat共现一次，cat与slept共现一次；窗口加大才会引入更远词。GloVe学习的是这些统计约束。fastText则将一个词的字符片段向量合成，例如where与whereupon可共享ere等片段；共享只代表可复用参数，不证明两词同义。BPE与fastText都涉及子词，但前者学习切分规则，后者学习/组合片段向量，任务不同。

## 三、操作前准备

GloVe部分需要NumPy，BPE/字符片段可只用标准库Counter。外部模型示例还涉及fasttext和transformers。
~~~bash
python -m pip install numpy
~~~
只尝试外部示例时，再按环境安装相应库并准备网络与磁盘空间；fastText的cc.en.300.bin不是小玩具文件。不要为了纯BPE练习下载整套模型。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 共现计数与GloVe训练

build_cooccurrence先确定词表，再枚举窗口位置；只保留出现过的词对，所以训练不会对log(0)取值。glove_train有中心表、上下文表与两套偏置；每步根据旧向量计算两侧梯度后再更新，返回W+W_tilde。f(x)随x增加后饱和在1；原文说它专门把高频对降权不够准确：相对低频对，高频对权重更高，只是不无限线性增加。

~~~python
import numpy as np
from collections import Counter


# 返回首次出现顺序的词表，以及按距离倒数加权的有向共现计数。
def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


# 只优化正共现词对；返回两套词向量之和。
def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
~~~

### 4.2 字符n-gram与组合向量

char_ngrams返回集合，去掉重复片段；集合展示顺序不固定。包括整个带边界词及不同长度滑窗。fasttext_vector只组合表中已知片段。where与whereupon会共享一些片段，应用实际集合交集核对，不凭近似拼写保证语义接近。

~~~python
# 集合包含边界词及不同长度字符滑窗，展示顺序不保证固定。
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
~~~

~~~pycon
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
~~~

~~~python
# 只对已知片段求和；一个也不认识时返回None。
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
~~~

### 4.3 BPE学习与编码

每个词的相邻对计数要乘频数；每轮重建当前切分，规则按学习顺序应用。合并仅发生在一个词的当前token边界上。原文10次合并的lowest结果是参考，应按频数和平局策略实测。new_vocab使用赋值而非累加，若扩展成可能碰撞的表示要复查。

~~~python
# 输入词频Counter，每轮按频数累计当前相邻对。
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


# 对新词按已有合并表处理，不重新统计或训练。
def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
~~~

~~~pycon
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
~~~

## 五、把代码运行起来：补充入口与检查

下面是无需NumPy的补充入口：先执行原文char_ngrams、learn_bpe、apply_bpe的定义，并导入Counter。
~~~python
from collections import Counter
a = char_ngrams("playing")
b = char_ngrams("played")
print("shared:", sorted(a & b))
print("Jaccard:", len(a & b) / len(a | b))
corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
merges = learn_bpe(corpus, k_merges=10)
print(merges)
print(apply_bpe("lowest", merges))
~~~
Jaccard是交集大小除以并集大小，不是共同字符串的总长度。若要运行GloVe，可用：
~~~python
docs = ["the cat sat".split(), "the dog sat".split()] * 10
vocab, counts = build_cooccurrence(docs, window=2)
W = glove_train(vocab, counts, dim=8, epochs=20)
assert W.shape == (len(vocab), 8)
assert np.isfinite(W).all()
~~~
后一段需要NumPy，本次不把未执行的优化过程标为实测。

## 六、Use It：使用现成工具

原文fastText调用会下载英文模型，再对已见或未见词组合向量。transformers示例只加载GPT-2分词器，不加载完整生成模型；Ġ是其可视化空白约定，不是原文中的普通字母。任何下方token列表均为原课参考，具体以模型文件与版本为准。预训练模型必须配套分词器；只有从零训练或明确调整模型输入层时才自行设计新词表。

~~~python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
~~~

~~~python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
~~~

~~~text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
~~~

原课给出“何时选哪个”的速查表（思路参考，具体任务仍需自己验证）：

| 场景 | 选择 |
|---|---|
| 要预训练通用词向量，不担心OOV | GloVe 300d |
| 要处理拼写错误、新造词或词形丰富的语言 | fastText |
| 任何要进Transformer的文本 | 用模型自带分词器，绝不更换 |
| 从零训练自己的语言模型 | 先在自己的语料上训练BPE或SentencePiece |
| 生产级文本分类配线性模型 | 仍是TF-IDF（[第02课](<02：词袋模型、TF-IDF与文本表示.md>)） |

## 七、失败模式、原文边界与工程注意

1. fastText不是只靠拼写就保证得到正确含义；未见词组合质量依赖片段和训练语料。
2. GloVe和Word2Vec为静态词向量，仍有多义词问题。
3. BERT的WordPiece、T5常见的Unigram/SentencePiece不应统称完全相同的BPE算法；SentencePiece还是工具框架。
4. 字符级BPE并不自动覆盖新字符；字节覆盖、字节回退和[UNK]策略须单独明确。
5. 词表32k或100k不是跨语言通用最优值；“最终约2～3字符/token”也不是定理。
6. 原文使用GloVe300d等作为候选，不等于所有线性分类都必选TF-IDF或生产都禁用词级分词。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
为给定任务分别说明：词向量方案与切分方案、语言与字符覆盖、词表目标与理由、库调用和模型对应关系。区分从零训练与微调，核对OOV、内存、下载及可复现性。不要给预训练模型随意换词表，不把固定词表大小或某种工具当成所有语言的唯一解。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：比较playing和played的字符集合，计算Jaccard并展示实际共享片段。
2. Medium：追踪每轮BPE合并后token总数与词表增长；明确字符数是否含</w>、空白，不能混用字符/token与token/字符。
3. Hard：在莎士比亚语料训练约1000次合并，比较常用词、专名及留出文本。先保留语料版本，记录平局规则、实际合并次数和未知字符行为。

## 十、术语与完成标准

co-occurrence：窗口内共现统计；subword：字符片段或学习的子词；GloVe：拟合对数共现结构；fastText：片段组合词向量；BPE：高频相邻单位合并；byte fallback：用基础字节补覆盖。

完成检查：

- 能手算距离1与2的共现贡献。
- 能解释GloVe的两张表和偏置。
- 能区分“学词向量”与“学token切分”。
- 能用实际集合而非示例文字核对OOV片段覆盖。

### 延伸阅读（原课推荐）

- [Pennington, Socher, Manning (2014), GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)：GloVe论文，七页，损失函数最好的推导。
- [Bojanowski et al. (2017), Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)：fastText子词论文。
- [Sennrich, Haddow, Birch (2016), Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：把BPE引入现代NLP的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)：BPE、WordPiece、SentencePiece实践差异对照。

## 十一、验证范围与来源许可

原课 Build It 共 3 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

