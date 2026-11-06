# 09：视觉Transformer——ViT与图像块

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F09-vision-transformers) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/09-vision-transformers/docs/en.md)。原题：Vision Transformers (ViT)。

## 零、初学者先读：把一张图变成“一句话”，再交给 Transformer

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第05课](<05：完整Transformer——编码器与解码器.md>)搭好了标准编码器，[第07课](<07：GPT——因果语言建模.md>)让它生成文本；本课把同一台机器的输入换成图像。下一课 [第10课](<10：音频Transformer——Whisper架构.md>)（Whisper）会复用同一个思路：先把非文本数据变成“图块序列”。

### 0.1 先理解这一课在做什么

卷积神经网络（CNN）天生带两个"先验"：局部性（每个卷积核只看一小块邻域）和平移不变性（猫挪到照片左下角，特征检测器照样响应）。2020 年之前所有视觉基准的第一名都是 CNN。ViT（Vision Transformer）的赌注是：**这两个先验都可以不要**——把图像切成固定大小的小块，每块拉平成一个向量，当作"词"排成序列，后面接的就是 [第06课](<06：BERT——掩码语言建模.md>) 里那个一模一样的标准编码器。数据够多时（ImageNet-21k 级别以上的预训练），模型自己能学会局部性。原课的口号值得背下来：图像是块的网格，句子是词的网格，同一个 Transformer 都能吃。

本课你真正要带走的只有一个新技能：切块（patchify）以及围绕它的三个数字——块数、块维度、序列长度。其余部分（注意力、编码器、[CLS]）你都在前几课学过了。

### 0.2 本课术语先翻成白话

- 图像块（patch）=图像里一个 P×P 的小方格，是视觉 Transformer 的"词"
- 切块（patchify）=把整张图切成互不重叠的小块，每块拉平成一维向量
- 线性嵌入（linear embedding）=一个可学习矩阵，把拉平的像素向量投影到 d_model 维
- [CLS] token（classification token）=额外拼接的可学习"汇总位"，最后用它做分类
- 位置嵌入（positional embedding）=告诉模型每个块原本在图像哪个位置的可学习向量
- 归纳偏置（inductive bias）=模型内置的先验假设；ViT 比 CNN 少，所以要更多数据补
- 蒸馏（distillation）=让小模型学生向大模型老师学习；DeiT 靠它把训练成本降下来
- 移位窗口（shifted window）=Swin 的招：注意力只在局部窗口内算，开销低于二次方
- 自监督（self-supervised）=不用人工标签，从图像自身的增广/遮罩里学特征；DINOv2 的路线
- 对比损失（contrastive loss）=拉近配对的图文、推远不配对的；SigLIP 用 sigmoid 版本
- register tokens=几个额外的可学习 token，替模型"吸收"多余注意力，改善特征质量
- ImageNet-1k / ImageNet-21k=约 128 万 / 约 1,400 万张标注图像的两个数据集；ViT 的"够大"指后者起步
- ViT-B/16 这种记号=ViT-Base 结构、patch 大小 16；斜杠后的数字就是 P
- 冻结骨干（freeze backbone）=预训练的编码器参数不更新，只训上面的小分类头

### 0.3 用小数字走一遍

一张 224×224 的 RGB 图，取 patch 大小 P=16：

1. 每行切出 224/16=14 块，每列也是 14 块，一共 14×14=**196 个块**。
2. 每块是 16×16×3 的像素立方体，拉平后是 16·16·3=**768 个数**——正好等于 ViT-Base 的 d_model。
3. 拼上 1 个 [CLS] token，序列长度变成 **197**。这就是"图像变句子"。

再感受 patch 大小这个"杠杆"：同样 224×224 的图，P=32 时只有 7×7=49 块。注意力矩阵的元素个数从 196²=38,416 掉到 49²=2,401，差 16 倍——块越小分辨率越高，但注意力开销按平方涨。

序列长度的量级对比（补充，串起整个阶段）：一句英文约 30 个词；ViT-B/16 一张图 197 个 token；下一课 Whisper 处理 30 秒音频要 1,500 个 token。注意力开销随 N² 增长，所以"视觉/音频的句子"天生比文本贵。

参数量粗算（对照原课 ~86M，ViT-Base：12 层、d=768）：每个编码器块 = 注意力 4d²（Q、K、V、O 四个投影）+ MLP 8d²（768→3072→768）= 12d² ≈ 7.08M，12 块约 84.9M；再加 patch 投影 768×768、位置嵌入 197×768、[CLS]、分类头，合计约 86M。原课给的 ViT-Base ~86M、ViT-Large ~307M、ViT-Huge ~632M 为原课参考。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

原课 Build It 用纯 Python 标准库搭"最小前处理流水线"：先看 `patchify()`（切块+拉平，全课唯一展示的函数）；然后是三段文字描述的步骤——随机矩阵做线性嵌入、拼 [CLS] 后检查形状 (N+1, d_model)、数一数真实 ViT 的参数量。玩具设定是 24×24 的假图、6×6 的块 → 16 块、每块 108 维，正好对应 0.3 的手算逻辑，只是数字缩小到能在脑内跑。

读的时候带着三个对应关系：`patchify` 的双重循环 ↔ 0.3 的 14×14 网格；线性投影的形状检查 ↔ 197×768 的序列；参数量统计 ↔ 86M 的粗算。第二节的概念编号（第 1 步到第 5 步）与原课一致，可以两邊对照着读。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么每个 patch 拉平后是 768 个数，而不是 16×16=256 个？

**参考答案：** 图像是 RGB 三通道，一个 16×16 的块有 16·16 个像素位置，每个位置 3 个数（R、G、B），所以是 16·16·3=768。灰度图才会是 256。768 恰好等于 ViT-Base 的 d_model 是历史巧合，不是必须。

**问题2：** [CLS] token 在前向计算时"看到"了什么？

**参考答案：** 它自己不含图像信息，只是一个可学习的占位向量；但每层注意力里所有 patch 都会与它交换信息，12 层之后它的隐藏状态被"喂"成了整图的汇总表示，分类头只读它。这正是 [第06课](<06：BERT——掩码语言建模.md>) BERT 的同款设计。

**问题3：** 同样的任务、同样的数据量，直接用小数据训 ViT 和训 CNN，谁更可能赢？

**参考答案：** CNN。ViT 缺少局部性与平移不变性这两个内置先验，等于让模型自己从数据里学这些规律，数据不够就学不出；原课说没有 1 亿级标注图像或强自监督预训练时，同等算力下 CNN 仍占优。DeiT（蒸馏）和 DINOv2（自监督）后来分别缓解和解决了这一点。

**问题4：** 把 patch 从 16 改成 8，token 数和注意力矩阵元素各变成几倍？

**参考答案：** token 数变 4 倍（224/8=28，28×28=784，是 196 的 4 倍），注意力矩阵元素变 16 倍（784²=614,656 对 196²=38,416）。规律：patch 减半=分辨率加倍，token 数×4、注意力开销×16。这就是密集预测任务愿意付、通用任务要掂量的账。

能解释 196 块、768 维、197 个 token 这三个数字的来历后，进入正文。

---

## 一、这一课解决什么问题？

2020 年之前，计算机视觉就是卷积的天下，ImageNet、COCO、检测榜的第一名清一色 CNN 骨干。Dosovitskiy 等人 2020 年的论文《An Image is Worth 16x16 Words》（一张图值 16×16 个词）证明：把卷积整个扔掉，图像切块、线性投影、送进普通 Transformer 编码器，在足够规模（ImageNet-21k 预训练起步）下 ViT 就能追平甚至超过 ResNet 系模型。这一课要回答的问题是：**Transformer 处理图像，到底需要改动多少？** 答案是几乎不改——这正是它教学上最漂亮的地方：到 2026 年，Whisper 吃音频、机器人吃动作 token、视频吃像素 token，"一个架构、多种模态"的全部起点就是本课的切块 trick。ViT 的后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了视觉的大半壁江山；CNN 仍守住边缘设备和低延迟场景。

本课在阶段里的位置：它是 [第09—12课](<12：KV缓存与Flash Attention——推理优化.md>) 这组"模态与效率"课程的开头——先证明 Transformer 能吃图像（本课），下一课吃音频（Whisper），再往后两课解决"模型变大之后怎么算得起"的问题（MoE 稀疏激活、KV 缓存与 Flash Attention）。学完本课你应该相信一句话：只要能把数据变成 token 序列，剩下的机器你已经会了。

## 二、核心概念与推导

### 2.1 第 1 步：切块（patchify）

把 H×W×C 的图像切成 N×(P·P·C) 的序列：N=(H/P)·(W/P) 个不重叠的块，每块拉平成 P·P·C 维向量。典型设定就是 0.3 节算的 224×224×3 → 196 个 768 维向量。patch 大小是核心杠杆：块小 → token 多、分辨率高、注意力开销平方涨；块大 → 粗糙但便宜。

（补充）为什么"拉平"就够了、不需要更花哨的编码？因为位置信息马上会由位置嵌入补上，而块内像素的二维排列对全连接投影来说只是"哪几个数配哪个权重"的对应关系，行优先展平是一个约定俗成的固定顺序，投影矩阵自己会学到该怎么组合它们。真正不能丢的是"块与块之间的相对位置"，那是位置嵌入的职责。

### 2.2 第 2 步：线性嵌入

一个可学习的矩阵把每个拉平的块投影到 d_model 维。数学上等价于 kernel=P、stride=P 的卷积——PyTorch 里真的就是一行 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`，实现只要两行。这也解释了为什么有人说"ViT 的 patch 投影是披着全连接外衣的卷积"。两种理解方式对照：

| 视角 | 说法 | 计算 |
|------|------|------|
| 全连接 | 每个拉平的 768 维块乘一个 768×768 的矩阵 | 一个矩阵乘法 |
| 卷积 | kernel=16、stride=16 的卷积核在图上滑动 | 14×14 次不重叠的窗口乘加 |

结果完全一样——因为窗口不重叠（stride=kernel），卷积在每个位置做的事恰好就是"取出窗口、拉平、乘同一个矩阵"。

### 2.3 第 3 步：拼 [CLS]、加位置嵌入

- 拼接一个可学习的 [CLS] token，最终隐藏状态做整图表示，供分类使用。
- 加可学习位置嵌入（ViT 原版），或 2D 正弦位置编码（后续变体）。原课提到 2024 年以后 RoPE 也被扩展到 2D，有时不再用显式嵌入（衔接 [第04课](<04：位置编码——正弦、RoPE与ALiBi.md>)）。

### 2.4 第 4 步：标准 Transformer 编码器

L 层 `LayerNorm → Self-Attention → 残差相加 → LayerNorm → MLP → 残差相加`，与 BERT 完全相同，没有任何"视觉专用层"。这是原课强调的教学重点（pedagogical punchline）：论文的贡献不是新结构，而是证明了结构不需要改。逐项对回前面课程：

- 自注意力本身就是 [第02课](<02：从零实现自注意力——Q、K、V与缩放点积.md>) 的缩放点积 + [第03课](<03：多头注意力——并行的视角子空间.md>) 的多头拼接；
- "LayerNorm 先行 + 残差"是 [第05课](<05：完整Transformer——编码器与解码器.md>) 的 pre-norm 编码器块；
- MLP 就是两层全连接加激活，视觉里没有任何特殊化。

换句话说：**切块之后，视觉与语言共用同一台机器**。这也是为什么 [第10课](<10：音频Transformer——Whisper架构.md>) 只需把音频变成"图"，就能直接套用同一套编码器。

### 2.5 第 5 步：输出头

分类任务：取 [CLS] 隐藏状态 → 线性层 → softmax。DINOv2、SAM 这类任务则丢掉 [CLS]，直接用各 patch 的嵌入——因为分割、检测需要"每个位置一个答案"，而 [CLS] 只有整图级的一个汇总。（补充：这正是第 0.5 节问题 1 想让你建立的直觉——输出头的选择由任务的粒度决定，不由模型决定。）

### 2.6 变体速览（原课表）

| 模型 | 年份 | 关键改动 |
|------|------|---------|
| ViT | 2020 | 原版：固定 patch、全局注意力 |
| DeiT | 2021 | 蒸馏；只用 ImageNet-1k 就能训好 |
| Swin | 2021 | 层级化+移位窗口，开销严格低于二次方 |
| DINOv2 | 2023 | 自监督（无标签）；最佳通用视觉特征 |
| ViT-22B | 2023 | 220 亿参数；缩放定律照样适用 |
| SigLIP | 2023 | ViT+文本编码器，sigmoid 对比损失 |
| SAM 3 | 2025 | 可提示分割；ViT-Large+掩码解码器 |

读这张表的正确姿势（补充）：前三个解决"ViT 太贵/太吃数据"（训练成本、数据量、注意力开销），DINOv2 解决标签依赖，后三个说明 ViT 已经变成"视觉积木"——ViT-22B 把它当缩放定律的试验场，SigLIP 把它当图文对齐的一侧，SAM 3 把它当分割系统的骨干。你在 2026 年遇到的视觉模型，大概率是这张表里某一项的后代。

### 2.7 为什么 ViT 晚了三年才追上 CNN

ViT 没有卷积的归纳偏置（平移不变性、局部性），没有 1 亿+ 标注图像或强自监督预训练时，同等算力下打不过 CNN。2021 年 DeiT 用蒸馏技巧把数据需求降到 ImageNet-1k；2023 年 DINOv2 用自监督"永久性"解决了这个问题——不靠标签也能得到最好的通用特征。

用一张表把两种架构的差别对齐（补充，据原课要点整理）：

| 维度 | CNN | ViT |
|------|-----|-----|
| 基本单元 | 卷积核（只看局部邻域） | patch token + 全局注意力 |
| 平移不变性 | 内置（权重共享） | 无，需自己学 |
| 局部性 | 内置（感受野逐层扩大） | 无，第一层就看全图 |
| 数据需求 | 中小数据可用 | 要么大数据要么强预训练 |
| 序列长度 | 与分辨率成线性关系 | (H/P)·(W/P)，注意力开销随其平方涨 |
| 典型强项 | 边缘设备、低延迟 | 大规模预训练后的质量上限 |

### 2.8 用 0.3 的小数字把概念串起来（补充）

224×224×3 的图切成 14×14 网格：196 个块，每块 16·16·3=768 个数。线性投影把每块变成 d_model=768 的 token（这里输入输出维度相同纯属巧合）；加 1 个 [CLS]、197 个位置嵌入。序列进入 12 层标准编码器后，[CLS] 的最终隐藏状态过线性层输出 1000 类得分。参数量按 12d²/块 粗算约 86M，与原课参考一致。如果嫌 197 个 token 太贵，把 P 换成 32 立刻降到 49+1=50 个 token，代价是分辨率变粗。

## 三、操作前准备

Build It 只用 Python 标准库（random 用于演示线性投影），无第三方依赖。Use It 需要 PyTorch、HuggingFace transformers 和 Pillow：

~~~bash
python -m pip install torch transformers pillow
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或下载模型。原课明确说明：不在标准库实现里训练 ViT——任何真实规模的 ViT 训练都需要 PyTorch 和数小时 GPU 时间。

## 四、Build It：原课分步代码与中文讲解

原课 Build It 分四步：假图 → 切块 → 线性嵌入 → 数参数。课程页只完整展示了第 2 步的 `patchify` 函数，其余步骤是文字描述；本节先原样转录代码，再把描述性步骤讲清，补充代码一律标注（补充）。

### 4.1 第 1 步：造假图（文字步骤）

24×24 的 RGB 假图，表示成"行组成的列表，每行是 (R, G, B) 元组"。取 6×6 的块 → 16 个块，每块 6·6·3=108 维向量。这个玩具设定就是 0.3 手算的缩小版：24/6=4，4×4=16。原课选 24×24 而不是 224×224 的原因很直接：标准库里逐像素的纯 Python 循环是 O(像素数)，224×224×3 有 15 万个数，演示没有必要；数据流的形状逻辑在 24×24 上一模一样。

### 4.2 第 2 步：切块（原课代码，原样转录）

栅格顺序（raster order）：按行优先遍历整个网格。原课强调每一种 ViT 用的都是这个顺序。

~~~python
# 本块中文说明：把图像切成 P×P 的块并逐块拉平，返回块的列表。
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
~~~

逐行看：外两重循环按 P 为步长跳到每个块的左上角 (i, j)；内两重循环在块内走 P×P 个像素；`extend` 把每个像素的 (R, G, B) 三个数追加进一维列表。所以每个 patch 的长度是 P·P·C，patch 总数是 (H/P)·(W/P)——正是 0.3 手算的两个公式。

（补充）用一个能脑内跑的 4×4 灰度微缩例子走一遍循环：图有 4 行 4 列，取 P=2。外循环 i 取 0、2，内循环 j 取 0、2，共 4 个块；对块 (0,0)，内层 di、dj 各取 0、1，依次取像素 (0,0)、(0,1)、(1,0)、(1,1)——注意是行优先的"Z 字形"顺序。最终 `len(patches)=4=(4/2)²`、每块长度 `2·2·1=4`（灰度单通道）。把 4×4 换成 224×224、P 换成 16、通道换 3，就是 196 块、768 维，逻辑一行都不用改。

### 4.3 第 3 步：线性嵌入（文字步骤，附补充代码）

原课描述：每个拉平的块乘一个随机的 (patch_flat_size, d_model) 矩阵；拼上 [CLS] 后验证输出形状是 (N_patches + 1, d_model)。（补充）最小实现：

~~~python
# 本块中文说明：（补充）按原课文字步骤写的最小线性嵌入与形状检查。
import random

def linear_embed(patches, d_model, seed=0):
    random.seed(seed)
    d_in = len(patches[0])
    W_proj = [[random.uniform(-0.02, 0.02) for _ in range(d_model)] for _ in range(d_in)]
    cls = [random.uniform(-0.02, 0.02) for _ in range(d_model)]
    tokens = [cls] + [[sum(p[i] * W_proj[i][j] for i in range(d_in)) for j in range(d_model)]
                       for p in patches]
    return tokens

# 24x24 假图、6x6 块 → 16 块、108 维；d_model 取 32：
# len(linear_embed(patches, 32)) == 17，且每个 token 长度 32。
~~~

注意真实 ViT 的投影矩阵是可学习的，这里随机初始化只是验证数据流形状。

### 4.4 第 4 步：数一数真实 ViT 的参数（文字步骤）

原课要求打印 ViT-Base（12 层、12 头、d=768、patch=16）的参数量，并与 ResNet-50 的约 25M 对比。结果：ViT-Base 约 86M，ViT-Large 约 307M，ViT-Huge 约 632M（原课参考）。0.3 节的 12d²/块 粗算给出 86M 量级，本机按公式复核为 86.4M——这只是参数量公式验算，不是训练实测。

## 五、把代码运行起来：补充入口与检查

（补充）原课 `code/main.py` 未随页面展示，本篇不虚构其完整内容；按 4.2—4.4 的步骤拼起来即可运行。以下是本机已验证过的最小检查（只验数字，不训练）：

~~~python
# 本块中文说明：（补充）切块数量与维度公式检查，可手算复核。
# 224x224、P=16：块数 = (224//16)**2 = 196，每块维度 = 16*16*3 = 768。
assert (224 // 16) ** 2 == 196 and 16 * 16 * 3 == 768
# 24x24、P=6：块数 = 16，每块维度 = 108。
assert (24 // 6) ** 2 == 16 and 6 * 6 * 3 == 108
# patch 大小的平方杠杆：196**2 与 49**2 的注意力矩阵元素比是 16:1。
assert 196 ** 2 // 49 ** 2 == 16
~~~

（补充）再给一个把 4.2 的 `patchify` 真正跑起来的微缩检查：4×4 的网格、P=2。注意一个真实坑：原课 `patchify` 里用 `extend` 展开"像素"，所以每个像素必须是可迭代的 (R, G, B) 元组——传纯数字会直接报 `TypeError`。手算预期：块 (0,0) 按 Z 字形收集像素 (0,0)、(0,1)、(1,0)、(1,1)：

~~~python
# 本块中文说明：（补充）4x4 图实跑 patchify，每个像素是 (R, 0, 0) 三元组。
img4 = [[(c, 0, 0) for c in row]
        for row in [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
patches = patchify(img4, 2)
# 手算预期：len(patches) == (4/2)*(4/2) == 4；每块 2*2*3 == 12 个数；
# patches[0] == [1,0,0, 2,0,0, 5,0,0, 6,0,0]（块内先右后下，Z 字形）；
# patches[1] == [3,0,0, 4,0,0, 7,0,0, 8,0,0]；patches[2]、[3] 依此类推。
~~~

补充检查方法：对 `patchify` 的输出验证 `len(patches) == (H//P)*(W//P)`、`len(patches[0]) == P*P*C`——这正是原课 Easy 练习的验收标准。

## 六、Use It：使用现成工具

原课的 HuggingFace 用法（原样转录）：

~~~python
# 本块中文说明：用预训练 ViT 提取图像表示。
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768)：[CLS] + 196 个 patch
cls_emb = out[:, 0]                       # 图像表示
~~~

输出形状 (1, 197, 768) 正是 0.3 手算的 197 个 token、d_model=768。原课的 2026 实践建议：图像特征默认用 DINOv2——冻结骨干、只训一个小头，分类/检索/检测/描述生成都行；Meta 的 DINOv2 检查点在所有非文本视觉任务上超过 CLIP（原课参考）。patch 大小选择：小模型用 16×16（ViT-B/16）；密集预测（分割）用 8×8 或 14×14（SAM、DINOv2）；超大模型用 14×14。

（补充）使用预训练 ViT 的两个实用提醒：一是 `last_hidden_state` 的第 0 列是 [CLS]、其余 196 列是 patch，做检索/分类取第 0 列、做密集任务用后 196 列，别搞反；二是换分辨率输入时处理器会自动缩放并插值位置嵌入（这正是第七节第 5 条失败的来源），但极端分辨率（如 1024×1024 输入 224 训练的模型）仍会明显掉点。

## 七、失败模式、原文边界与工程注意

1. **数据不足是第一失败模式**：小数据集上直接训 ViT 通常输给 CNN——没有内置先验可依赖。要么用预训练检查点，要么换 DeiT/DINOv2 路线。
2. **patch 太小的平方陷阱**：分辨率翻倍（等效于 patch 减半）会让 token 数变 4 倍、注意力矩阵元素变 16 倍。0.3 节 196 vs 49 的对比就是提醒。
3. **[CLS] 不是万能表示**：分割、检测这类密集任务直接用 patch 嵌入（DINOv2、SAM 的做法）；只有整图级分类才读 [CLS]。
4. **patch 顺序不能乱换**：原课强调所有 ViT 都用行优先的栅格顺序切块；如果你自己实现时改成列优先，预训练位置嵌入与块的对应关系就全错了（补充：对从头训练影响不大，但加载预训练权重时是致命的）。
5. **输入分辨率与预训练不一致需要插值位置嵌入**（补充，原课本课未展开）：直接换分辨率会让位置嵌入对不上，需要双三次插值等处理，原课未提，此处仅作边界提醒。
6. **register tokens 的教训**（原课 Key Terms）：DINOv2 特征图上的伪影后来被发现是注意力"沉淀"在个别 patch 上，加几个 register token 吸收即可改善——自监督特征质量问题的排查思路值得记住。
7. **"等数据够了再上 ViT"的门槛在移动**（补充，据原课要点归纳）：DeiT 之后 ImageNet-1k 级别（约 128 万张）能训出有竞争力的 ViT，DINOv2 之后连标签都可以不要；但"从零在几千张图上训 ViT"仍不是合理预期。
8. **边缘与低延迟场景仍是 CNN 主场**（原课原话）：ViT 后代赢的是质量上限，不是每瓦特延迟。
9. （补充）本课纯标准库实现只覆盖前处理（切块+投影），不含注意力与训练；不要误以为跑完 `main.py` 就"实现了一个 ViT"。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/skill-vit-configurator.md`（一个"给新视觉任务选 ViT 变体和 patch 大小"的技能文档）。改写成中文自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
检查视觉任务选型：说明数据规模（<1k / 1k-100k / >100k）与来源（有无标签）；有标签大数据选 ViT 微调，小数据选 DINOv2 冻结骨干+小头，无标签选 DINOv2；说明 patch 大小的取舍（分辨率需求 vs 序列长度的平方代价，给出 N=(H/P)(W/P) 的实际数字）；说明输出表示用 [CLS] 还是 patch 序列；边缘部署时说明为何仍考虑 CNN；所有未实测的精度/速度数字标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **原课 Easy**：运行 `code/main.py`，验证块数等于 (H/P)·(W/P)、每块维度等于 P·P·C。验收标准：对玩具设定 24×24、P=6 报出 16 块、108 维，并能对 224×224、P=16 报出 196、768；本篇第五节的检查已覆盖公式本身。
2. **原课 Medium**：实现 2D 正弦位置嵌入——对每个块的行号和列号各做一套独立的正弦编码再拼接；接入一个小 PyTorch ViT，在 CIFAR-10 上与可学习位置嵌入比准确率。验收标准：能展示两种位置嵌入的最终准确率（自己实测），并解释行/列分离编码为何比把位置压成一个标量更合理。（补充提示：[第04课](<04：位置编码——正弦、RoPE与ALiBi.md>)的正弦公式可以直接复用，只是要分别对行、列各算一份。）
3. **原课 Hard**：搭 3 层 PyTorch ViT，用 4×4 patch 在 1,000 张 MNIST 上训练并测测试准确率；再在同一千张图上做简化版 DINOv2 预训练（遮住部分块、让编码器预测被遮块的嵌入），看准确率是否提升。验收标准：给出预训练前后的两组测试准确率，并解释自监督预训练为什么在数据极少时帮助最大。

## 十、术语与完成标准

图像块（patch）：P×P×C 区域拉平成的向量，视觉 Transformer 的 token；切块（patchify）：切图+拉平；线性嵌入：把 patch 投影到 d_model 的可学习矩阵（等价于 stride=P 的卷积）；[CLS] token：拼在序列开头的可学习汇总位，最终隐藏状态作整图表示；位置嵌入：编码块在图中位置的向量；归纳偏置：模型内置先验，ViT 比 CNN 少、更吃数据；DeiT：靠蒸馏降低数据需求的 ViT；Swin：移位窗口层级 ViT，亚二次开销；DINOv2：自监督预训练 ViT，2026 默认图像特征；SigLIP：sigmoid 对比损失的图文模型；register tokens：吸收注意力沉淀的额外 token。

完成检查：

- 能手算 224×224、P=16 时的块数（196）、每块维度（768）、序列长度（197）。
- 能解释 patch 投影为什么等价于 kernel=P、stride=P 的卷积。
- 能说出 [CLS] token 从哪来、最后怎么用、什么时候不用它。
- 能解释为什么小数据上 ViT 输给 CNN，以及 DeiT/DINOv2 分别怎么解决。
- 能用 N² 的注意力开销解释"patch 大小是分辨率与算力的杠杆"。
- 能把 0.3 节的三个数字（196、768、197）逐一对应到 `patchify` 代码的具体行。
- 能说清 Swin、DINOv2、SigLIP 各自解决的是表里的哪一行问题。

## 十一、验证范围与来源许可

原课 Build It 共 4 个步骤，其中 `patchify` 代码逐字转录嵌入本篇（仅按本篇格式添加中文说明，未改逻辑）；第 1、3、4 步在课程页为文字描述，本篇如实按描述整理，补充代码均已标注（补充）并明确区别于原课。Use It 的 HuggingFace 代码原样保留。原课页未展示运行输出，本篇未虚构"原课参考"数值（86M/307M/632M/25M、DINOv2 与 CLIP 的比较均为原课参考）。原课 Further Reading 列出的论文（ViT、DeiT、Swin、DINOv2、register tokens 五篇）见来源链接，本篇不重复展开。

本机验证记录（2026-11-06，纯标准库脚本，不构成训练或性能实验）：

- 抓取课程页成功，以原文为准整理。
- 已用脚本复核：196 块、768 维、P=32 时 49 块、196²=38,416 与 49²=2,401（16 倍差）、玩具设定 16 块 108 维、ViT-Base 参数公式粗算 86.4M（与原课参考 ~86M 量级一致）。
- 已转录的原课 `patchify` 与（补充）代码通过语法检查；第五节的三个断言实际执行通过。
- 未做：下载预训练模型、运行 HuggingFace 推理、任何训练。

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
