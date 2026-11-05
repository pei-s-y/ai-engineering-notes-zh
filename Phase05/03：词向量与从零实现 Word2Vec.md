# 03：词向量与从零实现 Word2Vec

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-04 增补「零、初学者先读」、概念讲解与延伸阅读；当日与原课网页版核对，Build It 代码与五步结构一致。
> 2026-11-05 补充：0.1 增“词向量本体”（W的一行、与TF-IDF三区别、重训换数字而相对位置不变）；0.2 增Word2Vec名字来历；0.3 重写为“造卡片→学一下→看终态”三步式，补打分/sigmoid/初始化三前提、逐次手算更新与力气直觉；2.2 增W'=W_prime记法、逐站拆图与查表手算、两表角色分工与自匹配退化、脚手架类比、与0.3符号对照；2.3 增softmax手算与单选/是非题对照、负样本防坍塌、log与近似边界、“使用履历压缩记忆”的几何解读；2.4 增反义词句式模板、bank混合点与能/不能收拢表；2.5 增全课概念闭环检查表。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md)。原题：Word Embeddings — Word2Vec from Scratch。

## 零、初学者先读：为什么要让每个词变成一组数字

> 2026-11-04 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补“向量、点积、余弦”的直觉时，先回看[第02课](<02：词袋模型、TF-IDF与文本表示.md>)的向量部分。

### 0.1 先理解这一课在做什么

上一课TF-IDF把每个词放在自己独立的坐标轴上：dog是一条轴，puppy是另一条轴，两词相似度永远是0。本课换一种思路：给每个词学一组数字（例如16个数），让经常出现在相似上下文里的词，向量自动靠近。就像把词放到一张地图上，意思接近的词住得近；这张地图不是人画的，而是程序读大量句子后自己学出来的。

先给一个不浪漫的答案：词向量就是矩阵 W 的第 i 行，一行 d 个数，比如 cat 对应 [0.21, −0.83, 0.45, …] 这样16个数。单看任何一个数字都说不出含义——第3维是0.45，不代表“可爱度”也不代表“动物性”；有意义的只有几何关系：谁离谁近、方向差多少。与第02课的三个本质区别：

| | TF-IDF（第02课） | 词向量（本课） |
|---|---|---|
| 数怎么来的 | 数的（数出现次数，公式固定） | 学的（随机初始化，训练调出来） |
| 轴有名字吗 | 有：每列就是“词X出现的次数”，可解释 | 没有：16根轴全是无名轴 |
| 向量挂在谁身上 | 文档（一篇文档一行） | 词（一个词一行） |

第二行是关键的观念跳跃：TF-IDF的坐标系是人定的，词向量的坐标系是模型自己发明的，那16根轴人类甚至无法命名。重新训练一次，同一个cat会得到完全不同的16个数，但词与词之间的距离关系大致还在——相当于整张地图被旋转、拉伸，城市间的相对位置不变。“地图是学出来的”，准确含义是**画地图的方法学出来了，不是某套固定坐标**。

判断标准只有一句话，语言学上叫**分布假设（distributional hypothesis）**：Firth在1957年说 “You shall know a word by the company it keeps”（看一个词跟谁经常一起出现，就能知道它是什么意思）。cat周围常出现sat、mat、sleeps，kitten周围也常出现这些词，所以两者的向量应该靠近。

### 0.2 本课术语先翻成白话

- Word2Vec=“Word to Vector（词→向量）”的谐音简写，2=to（同B2B、P2P的缩写习惯），Google的Mikolov团队2013年发布该工具时起的名字。日常说它指一整包东西：Skip-gram/CBOW两种模型＋负采样等训练技巧＋一张“一行一词”的W表
- embedding（词向量）=一个词对应的一组可训练数字，比如16个
- Skip-gram=给中心词，猜周围词（cat → the, sat, on）
- CBOW=给周围词，猜中心词（the, sat, on → cat）
- 窗口 window=向左向右各看几个词；window=2就是最多各看2个
- one-hot=只有对应位置是1、其余全是0的“编号”向量
- 负采样 negative sampling=每次训练随机抽几个错误答案当陪练，避免对整个词表逐个算分
- sigmoid=把任意实数分数压到0～1之间的函数，σ(0)=0.5
- 梯度下降=沿让损失变小的方向微调参数
- 余弦相似度=只比较向量方向、不管长度的相似度指标
- OOV（out of vocabulary）=训练时没见过、词表里查不到的词
- 静态/上下文向量=一个词永远一个向量 / 同一个词每次出现的向量随上下文变化

### 0.3 用小数字走一遍

这一段把“训练”拆到最小零件：3个词、4张训练卡片、1次更新，用手算看清“学”到底是什么动作。三步分别对应：造卡片 → 学一下 → 看终态。

**第一步：训练数据怎么造。** 想象拿荧光笔逐个词划过句子，每次划一个词当**中心词**，window=1 表示可以看它左右各 1 个词，看到的每个词记作它的**邻居**。句子`a b c`、window=1：

~~~text
句子：  a   b   c
   a 当中心：左边没词，右边是 b           →  记 (a, b)
   b 当中心：左边 a，右边 c               →  记 (b, a)、(b, c)
   c 当中心：左边 b，右边没词             →  记 (c, b)
~~~

a只能看到b，c只能看到b，b两边都能看，所以产生4个有向训练对：(a,b)、(b,a)、(b,c)、(c,b)。“有向”指(a,b)是“a当中心、b当邻居”、(b,a)是“b当中心、a当邻居”，是两条不同样本，不能合并。到这里还没有任何学习发生——只是把句子切成一堆（中心，邻居）小卡片，共4张。

**第二步：一张卡片的学习动作。** 先补齐三个前提：

1. **每张卡片先打分**：中心词向量与邻居向量做**点积**，分数高=模型认为“这俩经常一起出现”。
2. **sigmoid 是打分器**：σ 把任意分数压到 0～1。分数很大→σ≈1（“是真邻居”）；分数为 0→σ=0.5（**完全没主见**）；分数很小→σ≈0（“不是邻居”）。
3. **刚初始化时分数≈0**：向量是随机数，设某正样本对的点积为 0，模型对“a 和 b 是不是邻居”弃权——这就是起点。

训练开始时假设某正样本对的点积是0：σ(0)=0.5。我们希望正样本的σ接近1，梯度系数是σ−1=−0.5，更新会让点积变大；负样本希望σ接近0，系数是+0.5，更新方向相反。梯度系数就是“推的力气”：正样本力气=1−σ，负样本力气=σ——**错得越离谱推得越狠（差距大），已经答对就几乎不推**（σ=0.99 时力气只剩0.01）。

用手算验证一次（学习率取 0.1）。设 v_a=[0.5, 0.5]、u_b=[0.5, −0.5]，点积=0.25−0.25=0，σ=0.5。正样本更新 v_a ← v_a + 学习率×(1−σ)×u_b：

~~~text
v_a = [0.5, 0.5] + 0.1 × 0.5 × [0.5, −0.5] = [0.525, 0.475]
新分数 = 0.525×0.5 + 0.475×(−0.5) = 0.025
~~~

分数从 0 涨到 0.025——**真的变大了**，σ 微升到约 0.506。学习就是这一步；负样本同力气、反方向（v_a ← v_a − 学习率×σ×u_负样本），分数变小。所谓“训练200轮”，就是把这两件小事对每个词对反复做——一轮=4张卡片各做一次上面的更新，200轮=翻来覆做200遍；真实训练里卡片是几十亿张、向量几百维，但每次动作与上面一模一样。

**第三步：微小移动的累积。** 单次更新只挪 0.025 这种量级，毫无用处；但 cat 的每张卡片都被 sat、mat、sleeps 拉一下，kitten 的卡片被**同样的**词拉。这些微小移动累积成什么样？跳到训练结束后的假想终点（二维示意，真实是100～300维，数字为可手算的假设值）：cat=[0.8, 0.2]、kitten=[0.7, 0.3]、car=[−0.6, 0.8]。余弦相似度 cos(cat, kitten)=(0.8×0.7+0.2×0.3)÷(0.825×0.762)≈**0.99**，几乎同方向；cos(cat, car)=(−0.48+0.16)÷0.825≈**−0.39**，背向而行。对比第02课的惨状：cat与kitten是两根不同的轴，余弦恒为0，任何公式都算不出“相似”。词向量第一次让“意思接近”变成一个可以计算的数。cat与kitten的靠近不是谁设计出来的，是亿万次拉近/推开堆出来的。

类比检索king−man+woman：不是魔法，“皇室”是地图上一个**方向**，(king−man)大致捕捉从“男人”指向“皇室男人”的那段箭头，把它整体平移加到woman上，就落在“皇室女性”区域附近——这是大规模语料上观察到的现象，小语料训不出来，也不是恒等式。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

按顺序读五个函数，每个都只做一件事：skipgram_pairs把句子变成训练对 → init_embeddings生成两张随机表 → train_pair做一次梯度更新 → train套上循环 → nearest/analogy在训练结果上查询。核心数据结构两个：词表{词:编号}字典；W与W_prime两张(V,d)矩阵，第i行就是编号i那个词的向量。W[c]是词c当中心词时的向量，W_prime[o]是词o当上下文时的向量——同一个词在两张表里有两个不同的向量。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么一个词需要两个向量（W一行、W_prime一行）？

**参考答案：** 一个词当“被预测的中心”和当“周围的上下文”是两种角色，分开存储让训练更稳定；训练完成后通常只保留W当词向量。

**问题2：** 负采样和softmax是一回事吗？

**参考答案：** 不是。softmax要在整个词表上归一化，词表10万时每个训练对都要算10万次分；负采样把“多分类”换成“一个正样本对几个随机负样本”的二分类，每个训练对只算几次。

**问题3：** 查不到`Zoomer-approved`这个词怎么办？

**参考答案：** 这是OOV。经典Word2Vec没有fallback，直接查不到；[第04课](<04：GloVe、fastText与子词表示.md>)的fastText用字符片段拼向量解决这个问题。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

上一课把每个词放在独立坐标轴上，dog 与 puppy 的相似性不会自动出现。本课学习把词映射到低维稠密向量：上下文相似的词，有机会得到接近的表示。目标是理解“训练任务怎样塑造表示”，而不是用两句话语料证明模型已经理解语言。

原课的类比 king−man+woman≈queen 是某些训练模型上的观察，不是向量恒等式，也不是所有词向量必然满足的性质。

## 二、核心概念与推导

### 2.1 分布假设与两种预测方向

本课所有设计的出发点是分布假设：上下文相似的词，语义大概率相似。把“预测什么”定义成训练任务，词向量只是训练的副产品。

- **Skip-gram**：给中心词，预测周围词。cat → (the, sat, on)。训练慢，但对低频词更好，成为默认选择。
- **CBOW**：给周围词，预测中心词。(the, sat, on) → cat。训练快，低频词表示较弱。

window=2 表示左右各看最多两个位置，不包含中心位置，也不跨文档配对。补充（经验倾向，非硬规则）：窗口偏大捕捉到的多是主题关系，窗口偏小更贴近语法搭配关系。

### 2.2 网络结构与两张表为什么都需要

原课把Skip-gram画成一个两层网络，中间层没有非线性激活：

这张图用了不少神经网络行话，但底下只有**两张查找表加两次乘法**（补充）。先说记法：图里的 W' 读作“W prime”，本课代码里写作 W_prime——**同一个东西**，别当成两个未知符号。

~~~text
one-hot(中心词) ── W ──▶ 隐藏层(d维) ── W' ──▶ 对全词表的softmax
                              ↑
                        这层的权重就是词向量
~~~

**站点1——one-hot：编号的正式写法（补充）。** 词表 {a, b, c} 里 b 的 one-hot 是 [0, 1, 0]：只有自己的位置是1，其余全是0。它就是用V个数写一个编号，本身没有任何含义。

**站点2——乘W等于查表（手算一遍，补充）。** 设W是3×2的表（V=3行，每行d=2个数）：

~~~text
W =  a: [1, 2]
     b: [3, 4]
     c: [5, 6]

[0, 1, 0] × W = 0×[1,2] + 1×[3,4] + 0×[5,6] = [3, 4]   ← 正好是b那一行
~~~

one-hot里那个1像探照灯，只照亮一行，其余乘0蒸发。

设词表大小V，维度d，W与W_prime都是V×d。W[c]是词c当中心词的向量，W_prime[o]是词o当上下文词的向量。one-hot乘W在数学上等价于“按行查表”，所以代码从不真的创建巨大的one-hot，直接写W[编号]。两张表的角色分工（补充）：

| | W（输入表） | W'（输出表） |
|---|---|---|
| 大小 | V×d | V×d |
| 第o行是什么 | 词o**当中心**时的向量 | 词o**当邻居**时的向量 |
| 干什么用 | 中心词进来时查这张表 | 拿去和它逐行点积、给全词表打分 |
| 训练完 | 留下（这就是词向量） | 扔掉（脚手架） |

**关键：同一个词在两张表里的数字不同**——cat在W里有一行，在W'里另有一行，各是各的数。W'的机械角色是“题库”：前半程中心词查W得到W[cat]（d个数）；后半程拿W[cat]和W'的每一行做点积，词表多大做几次，得到一列分数。手算（d=2，词表{cat, sat, car}，补充）：

~~~text
W  （中心表）        W' （邻居表）
cat: [ 1, 2]        cat: [ 0, 1]
sat: [ 3, 0]        sat: [ 2, 1]
car: [-1, 0]        car: [ 0,-1]

cat当中心，W[cat]=[1,2] 与 W' 三行各点积：
score(cat) = 1×0 + 2×1    =  2
score(sat) = 1×2 + 2×1    =  4   ← 最高
score(car) = 1×0 + 2×(−1) = −2   ← 最低
~~~

模型的回答：sat最可能是cat的邻居，car最不可能——这就是图上“对全词表的softmax”一站干的事。

**为什么非要两张表（补充）？** 如果只留一张表、一个词一个向量，会发生**自匹配退化**：打分变成 v_c·v_o，取o=c自己时 v_c·v_c=|v_c|²，而 v_c·v_o ≤ |v_c||v_o|——只要各词向量长度差不多，**每个词的最高分永远是它自己**。模型会学出偷懒答案“cat旁边最可能出现cat”，但真实语言里cat的邻居是sat、mat，不是cat cat cat。两张表把这个死循环拆开：v_cat·u_cat 没有任何理由最大。直觉上这也是两种角色：cat被解释时（当中心，用W的行）和cat去解释别人时（当邻居，用W'的行），统计规律不同。这个“一个问题拆成两个向量”的技巧在Transformer的注意力里原样重现（Query与Key分开，第07阶段）。

**“没有非线性激活”是什么意思（补充）？** 普通神经网络每层之后过一个弯曲函数（如ReLU）才能学曲线关系；这里隐藏层原样直通，骨子里是线性模型——图比内容唬人，全部内容就是查表和点积。

训练完成后丢掉输出层，只留W当词向量；也可以在明确约定下组合两张表。（补充类比）整个预测机器——W'和softmax——是**脚手架**：盖楼时必须有，盖完就拆。像考试：出题（“猜邻居”）是为了逼你把知识装进脑子（W），考完试卷扔掉，留下的是脑子——这就是2.1说“词向量只是训练的副产品”的具象版。注意两张表训练时都在被更新，不是只有W在学。

Skip-gram和CBOW共用这个结构，只是预测方向相反。

与0.3的符号对照（补充）：

| 0.3的符号 | 2.2的位置 |
|---|---|
| v_a（a当中心的向量） | W的第a行 |
| u_b（b当上下文的向量） | W_prime的第b行 |
| 卡片打分“点积” | W[c]·W_prime[o] |

看其他教材时的形状差异（补充）：有的书把 W' 写成 d×V、说“第o**列**是上下文向量”——同一张表转置了一下，别被绕晕；本课代码两张都是 (V, d)，行对行点积。

### 2.3 为什么要负采样，目标是怎样的

**先看softmax是什么（补充）。** 2.2手算过：cat当中心时，词表三词的分数是 cat=2、sat=4、car=−2。softmax把一列分数变成“每个词当邻居的概率”（总和为1）：

$$\text{softmax}(s_i)=\frac{e^{s_i}}{\sum_j e^{s_j}}$$

手算（e²≈7.39、e⁴≈54.6、e⁻²≈0.14，分母≈62.1）：

| 词 | 分数 | e^分数 | 概率 |
|---|---:|---:|---:|
| sat | 4 | 54.6 | 0.88 |
| cat | 2 | 7.39 | 0.12 |
| car | −2 | 0.14 | 0.002 |

读法两条：分高的概率大；e的指数还放大差距——分数4只比2大一倍，概率却是0.88对0.12的碾压。这就是图上“对全词表的softmax”一站输出的东西。

**softmax的问题是贵（补充）。** 注意分母要把全词表**每个词**的 e^分数 加起来：词表3个无所谓；词表10万时，每张训练卡片都要先算10万次点积、10万次e^、再求和，而卡片有几十亿张——数学没错，账单付不起。

对全词表做softmax，意味着每个训练对都要对V个词各算一次分。V=10万时这是主要计算开销。**负采样（negative sampling）** 把多分类改成二分类：只问“这个上下文词是不是真的出现在中心词附近”，再随机抽k个词冒充错误答案。计算量从O(V)降到O(k)，k通常只有5～20。

换一种理解，这是把考法换了（补充）：

| | 原考法（softmax） | 新考法（负采样） |
|---|---|---|
| 题型 | 单选题：cat的邻居是10万个选项中的谁？ | 是非题：cat和这个词是真邻居吗？ |
| 判卷要算多少 | 全部10万个选项的分数 | 只算题目里出现的那几个 |
| 每张卡的成本 | O(V) | O(k)，k=5～20 |

一张卡片(cat, sat)变成1道“是”的题（cat–sat真邻居）加k道“否”的题（cat–随机词，冒充的）；每道题只需一次点积加一次σ——0.3的打分器原封不动就是这里的判卷工具，0.3里“正样本力气1−σ、负样本力气σ”正是这套考法的评分标准。

**为什么必须有负样本（补充，理解负采样的钥匙）。** 只拉近真邻居行不行？不行，会**坍塌**：只拉近时模型有完美作弊解——把所有词的向量都变成同一个点，任意两点积都最大、每道“真邻居”题满分、损失为0，向量全废（所有词在同一位置，“相似度”全是1，等于什么都没学），像全班搬进同一间教室再宣布“我们都住得很近”。负样本的职责就是逼模型保留区分度：sat靠近cat，但airplane、quantum这些冒充者要被推开。名字也由此而来：“负”=假配对；“采样”=10万个错词不逐个推开（那又回到老问题），而是随机抽几个当代表。

对正对(c,o)及负样本n，损失为：
$$L=-\log\sigma(v_c^\top u_o)-\sum_n\log\sigma(-v_c^\top u_n)$$

逐项读：第一项希望正对的点积大（σ接近1），第二项希望负对的点积小（σ接近0）。sigmoid把点积变成0～1的得分；它不是整个词表上归一化的softmax分布。对正样本，点积梯度是σ(score)−1（预测不足时接近−1，猛推）；对负样本是σ(score)（预测不足时接近0，几乎不推）。这就是train_pair中的两种系数。外面套log是单调变换（补充），不改变“推的方向”，只让优化在数值上更好算。再立一个边界（补充）：softmax是“10万个选项分一块蛋糕、总和为1”；负采样的每道是非题独立判分、互不影响——所以负采样是softmax目标的**近似替换**，两者不等价，工程上换来的是O(V)到O(k)的账单。

翻译成几何语言（补充）：正样本的更新近似“中心词往它真实邻居的方向挪一小步”，负样本则把两者推开一小步。cat和kitten因为总被同样的上下文向量（sat、mat、sleeps）反复吸引，几十亿次微小移动后挤到同一片区域。所以词向量的本质可以一句话说清：**它是模型对一个词“见过哪些邻居”的压缩记忆——不是“词的意思”，是“词的使用履历”**。

### 2.4 静态向量的能与不能

相近上下文不等于同义词：反义词good/bad也可能出现在相似句式里，向量因此接近——这是分布假设的固有特点，不是实现bug。

（补充展开）具体看这个句式模板：“The movie was ___”。填good和填bad，周围词几乎一样（the、movie、was），训练时两个词被同样的上下文反复拉近；在真实预训练词向量里查good的最近邻，bad排在前列是常见观察。要分褒贬，靠的不是换更好的词向量，而是下游任务——第05课情感分析的分类器从**标签**学极性，标签才是褒贬信息的来源。

一个词只有一个向量（**静态 embedding**）：bank的河岸义和银行义共享表示，这就是原课说的**多义词墙（polysemy wall）**；table的表格义和家具义同理。下游若还读取上下文仍可消歧，不能说整个系统必然无法区分；BERT之后的**上下文向量（contextual embedding）** 让同一个词每次出现都有不同向量，那是第07阶段Transformer的内容。另一个限制是OOV：训练没见过的词查不到向量，fastText用子词组合解决（第04课）。

（补充）bank的静态向量其实是两种用法按出现频率**加权平均出来的混合点**：假设语料里七成是银行义、三成是河岸义，它的向量就落在银行词簇的边缘偏外——哪个簇都不完全属于。静态向量的能与不能，一张表收拢（补充）：

| 问题 | 静态词向量的回答 | 遗留 → 谁来接 |
|---|---|---|
| 词相似度可计算 | 解决：余弦/最近邻 | — |
| 类比方向（king−man+woman） | 大语料上常成立 | 不是恒等式，不能当API用 |
| 反义词的褒贬 | 没解决：good/bad一样近 | 下游分类器学极性（第05课） |
| 多义词 | 没解决：bank一个混合点 | 上下文向量（第07阶段） |
| OOV | 没解决：查不到 | fastText子词（第04课） |
| 词组合成句子 | 没解决：词级向量不含词序 | 句子级/上下文表示 |

一句话：静态词向量把“意思接近”变成可计算的数，但**分辨“好与坏”“此义与彼义”“生词”超出了分布假设的能力范围**——这三样分别交给任务、交给上下文、交给子词。

### 2.5 用一个小例子把概念串起来（补充）

以句子 `a b c`、window=1 为例，a只能看b；b能看a和c；c只能看b，因此4个有向训练对。假设正对的点积为0，σ(0)=0.5，正样本梯度系数为−0.5，梯度下降会推动正对点积增加；负样本同为0时系数+0.5，更新方向相反。你需要能解释这个符号，而不必把训练200轮当作理解的证明。

（补充）现在全课的概念都能在 `a b c` 里对上号了——一张闭环检查表：

| 概念 | 在 `a b c` 例子里的化身 |
|---|---|
| 分布假设 | 训练对只来自真实共现：a 的邻居是 b，不发明没出现过的搭配 |
| 两张表 | v_a=W[a]（a当中心）、u_b=W_prime[b]（b当邻居），同一个b在两表各一行 |
| σ打分 | 点积0 → σ(0)=0.5，模型弃权 |
| 正样本更新 | 力气 1−σ=0.5，往真邻居方向挪（2.2的拉近） |
| 负样本更新 | 随机抽词冒充，力气 σ=0.5，推开（2.3的防坍塌） |
| 多轮训练 | 200轮 = 把4张卡片翻来覆做 |
| 终态检验 | nearest 用余弦查“谁离a最近”——它是副产品，见下 |

最后一行值得多看一眼：训练任务从头到尾只在做“猜邻居”的是非题，**没有任何一步直接要求过“意思相近的词向量要靠近”**——相似度结构是自己冒出来的。这正是“词向量是训练的副产品”（2.1）最准确的意思。理解了这个链条，就可以进入第四节的Build It代码：skipgram_pairs对应造卡片、init_embeddings对应两张随机表、train_pair对应学一下、train套循环、nearest对应终态检验。

## 三、操作前准备

需要NumPy；现成工具示例还需gensim。
~~~bash
python -m pip install numpy gensim
~~~
原文train依赖上一课build_vocab，但没有在本课重写。先补齐这个函数，再依次执行Build It：
~~~python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
~~~
模型索引必须与词表一致；训练新词表后不能套用旧矩阵。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 构造中心词—上下文正样本

输入是分好词的文档列表。双层位置循环枚举窗口，边界用max/min截断；同一位置跳过。长度5、window=2会产生14个有向词对。词对(c,o)与(o,c)可同时出现，不是重复代码错误。原文带>>>和省略号的块是交互示例，不是完整Python脚本。

~~~python
# 输入预分词文档，返回有向(中心词,上下文词)对。
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
~~~

~~~pycon
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
~~~

### 4.2 初始化中心表与上下文表

随机种子控制初始化；两张表形状都为(V,d)，但数值不同。小随机值打破对称性，不会一开始就具有语义。seed相同有助复现，不保证不同软件、多线程训练绝对逐位一致。

~~~python
import numpy as np


# 固定随机种子初始化两张V×d浮点矩阵。
def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
~~~

### 4.3 计算并更新负采样梯度

先读取旧向量，计算正负得分，再构造中心向量梯度，最后更新两张表。W[context_idx]=W[context_idx]是原文无作用语句，保留以便对照。负样本可重复，重复项会多次贡献梯度。不要把负样本当成“在整个语料中绝不共现”的词，它只是采样训练标签。

~~~python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# 使用当前正词及采样负词，对两张词向量表做一次梯度下降。
def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
~~~

### 4.4 组织训练循环

每个epoch打乱词对，对每对抽负样本。原文均匀抽样，并过滤中心与当前正词，过滤后可能少于k_neg个；这不是经典Word2Vec按词频的3/4次幂采样的完整实现。返回词表及W，不包含分类器或自动打印的准确率。

~~~python
# 先建词表，再逐轮打乱正词对并采样负例。
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
~~~

### 4.5 近邻与类比检索

先把各词向量和查询向量做归一化，再算余弦，按相似度降序排序。exclude存的是编号，不是字符串。analogy(a,b,c)实际构造b−a+c，并排除输入三词。Google News上的queen示例只是原课参考，不是小语料训练结果。

~~~python
# 余弦近邻检索；exclude中的元素是词表编号。
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


# 返回b−a+c向量的近邻，不保证存在语言学上的正确答案。
def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
~~~

~~~pycon
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
~~~

## 五、把代码运行起来：补充入口与检查

定义前述函数后，以下是补充的完整最小入口（需要NumPy，本次未执行训练）：
~~~python
docs = [
    "the cat sleeps on the mat".split(),
    "the dog sleeps on the rug".split(),
    "a cat likes food".split(),
    "a dog likes food".split(),
] * 5
vocab, W = train(docs, dim=16, window=2, epochs=200, seed=0)
print(W.shape)
print(nearest(vocab, W, W[vocab["cat"]], exclude={vocab["cat"]}))
~~~
预期第一项形状为(10,16)；请以实际词表长度核对，不要把维度当准确率。近邻顺序依赖数据和训练，dog未必进前三。先做结构检查：
~~~python
assert W.shape == (len(vocab), 16)
assert len(skipgram_pairs([["a", "b", "c"]], window=1)) == 4
~~~
训练损失下降、邻居合理、下游任务有效是不同层面的检查。

## 六、Use It：使用现成工具

原文gensim示例完整保留在下方：sg=1选择Skip-gram，negative=5选择负采样数量，min_count=1保留低频词。model.wv存训练后的词向量；most_similar返回近邻而不是分类结果。两句语料不足以得到可靠语义。预训练Google News、GloVe和fastText需要另行下载，占用明显内存，本文未下载。

~~~python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
~~~

原课还列出2026年Word2Vec仍占优的四类场景（思路参考，非本机实测）：在小规模专业语料（如医学摘要）上快速自训一套领域向量；用类比差值（如性别方向）做特征工程，公平性研究仍在用；100维小到可以用PCA或t-SNE画图直接观察聚类；以及在无GPU设备上做单行查表式推理。

## 七、失败模式、原文边界与工程注意

1. 未登录词不能直接索引W；应检查词表，不能把KeyError当作训练崩溃。
2. 单词与自己比较通常最相似；评估近邻时排除自己。
3. 原文“范数低于0.01或高于100就是bug”过于绝对。记录范数分布、非有限值与变化趋势，结合初始化和训练规模判断。
4. 小语料不保证学出类比，也不能用PCA/t-SNE图证明原空间距离关系或公平性。
5. 去掉一条所谓性别方向不等于消除所有偏见。偏见实验需说明语料、词表、指标与社会语境。
6. “一小时训练”“某年默认方案”等不是本机实测或普遍保证。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
检查词向量：说明词表与向量维度；做三组明确a、b、c顺序的类比和五组领域词近邻；检查余弦对称性、零/异常向量；报告缺词情况。不要只凭类比成功宣称模型好；同时设计下游评估。所有未实际运行的数值标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课Easy：用20句猫狗语料训练约200轮，检查cat的前三近邻。验收为能复现实验、排除自身并解释结果；不是硬性保证dog必出现。
2. Medium：加入高频词下采样，比较低频词近邻。阈值10^-5只是超参数；需明确采用的保留概率公式，不把所有高频词全部删除。
3. Hard：在20 Newsgroups训练，比较he−she、doctor−nurse方向上的职业词投影。记录分词、训练划分、随机种子和样本词；投影是表示偏差探针，不是对个人或职业的价值结论。

## 十、术语与完成标准

embedding：词的可学习向量；Skip-gram：中心预测上下文；CBOW：上下文预测中心；negative sampling：对采样词对做二分类；static/contextual：是否随当前上下文改变；OOV：当前表中没有的词。

完成检查：

- 能从一句话列出窗口词对。
- 能推导正负两类得分梯度的符号。
- 能说明(V,d)矩阵每行是什么。
- 能解释小样本、OOV、多义词和偏见评估的限制。

### 延伸阅读（原课推荐）

- [Mikolov et al. (2013), Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)：负采样目标的出处，篇幅短、可读性好。
- [Rong, X. (2014), word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)：最清晰的梯度推导，觉得原文数学过密时看这份。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html)：生产可用的训练参数设置。

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

