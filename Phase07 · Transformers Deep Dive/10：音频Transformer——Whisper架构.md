# 10：音频Transformer——Whisper架构

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases%2F07-transformers-deep-dive%2F10-audio-transformers-whisper) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/07-transformers-deep-dive/10-audio-transformers-whisper/docs/en.md)。原题：Audio Transformers — Whisper Architecture。

## 零、初学者先读：把声音先变成“图”，再交给 Transformer 转述

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第09课](<09：视觉Transformer——ViT与图像块.md>)刚证明"图像切块就能喂 Transformer"，本课把同一招用在音频上：先把声音画成"频率×时间"的图（mel 谱），再让一个编码器-解码器（[第05课](<05：完整Transformer——编码器与解码器.md>)、[第08课](<08：T5与BART——编码器-解码器模型.md>)）把它转述成文字。

### 0.1 先理解这一课在做什么

Whisper（OpenAI，2022）干的事：听 30 秒音频，输出这段话的文字。它没有发明新架构——编码器读"声音的图"，解码器自回归地写文字，就是 [第05课](<05：完整Transformer——编码器与解码器.md>) 的教科书结构。真正特别的是三件事：一，音频先变成 log-mel 频谱图，这张"图"就是编码器的输入序列（第09课思路的音频版）；二，解码器的开头几个特殊 token（任务 token）告诉模型"这次是转写还是翻译、什么语言"；三，一个模型同时学会了转写、翻译、语种识别、时间戳，靠 68 万小时的弱标注音频喂出来。原课的口号：**音频是"频率随时间变化的图像"，Whisper 就是一个吃 mel 谱图、然后开口说话的 ViT。**

整条流水线用一行伪代码概括（数字全部来自 0.3 的手算）：

~~~
30秒音频 → 480,000采样点 → 3,000帧×80mel格（图） → 卷积茎 → 1,500 token
        → 24层编码器 → 1,500个隐藏状态
        → 解码器（提示：<|startoftranscript|><|en|><|transcribe|><|0.00|>）→ 逐token输出文字
~~~

看不懂中间的数字没关系，0.3 节会一个一个算给你看。

### 0.2 本课术语先翻成白话

- 采样率（sample rate）=每秒记录多少个空气压强数值；语音标准是 16 kHz，即每秒 16,000 个数
- 帧（frame）=一个 25 毫秒的采样窗口；步进（hop/stride）=相邻帧起点相隔的 10 毫秒
- mel 频率格（mel bin）=纵轴上的一格频带；Whisper 用 80 格，mel 刻度按人耳分辨力压扁高频
- mel 频谱（mel spectrogram）=一张二维图：横轴时间帧、纵轴频率格、格子里是能量
- log-mel=对 mel 谱取对数，更贴近人耳对响度的感知；"Whisper 看到的图"就是它
- 卷积茎（convolutional stem）=编码器前的两层一维卷积，把 3,000 帧降采样到 1,500
- 自回归（autoregressive）=每次用已生成的全部 token 预测下一个 token，见 [第07课](<07：GPT——因果语言建模.md>)
- 多任务学习（multi-task learning）=一个模型同时学转写、翻译、语种识别等，靠任务 token 区分
- 任务 token（task token）=解码器开头的特殊词，如 `<|transcribe|>`、`<|en|>`，告诉模型干什么
- BPE（Byte Pair Encoding）=把文本切成子词单元的词表方案；Whisper 的词表是 GPT-2 的超集
- 束搜索（beam search）=每步保留 5 个候选句子的解码策略（beam 宽度 5）
- VAD（Voice Activity Detection，语音活动检测）=先找出"有人在说话"的段落，把静音切掉
- CTC=经典 ASR 的对齐无关损失函数；**Whisper 不用它**，这是它与老式 ASR 的分水岭
- 说话人分离（diarization）=标注"谁在什么时候说"；Whisper 不做，要配 pyannote
- WER（Word Error Rate，词错误率）=转写结果与标准答案的编辑距离百分比
- 弱标注（weakly-labeled）=从互联网上抓的、质量参差的字幕/配文，不是人工精标

### 0.3 用小数字走一遍

Whisper 的输入约定：16 kHz 采样、25 ms 窗、10 ms 步进、80 个 mel 频率格、固定 30 秒。

1. **30 秒有多少个采样点？** 16,000 × 30 = 480,000 个。
2. **一帧多长？** 25 ms × 16,000/ms = 400 个采样点；步进 10 ms = 160 个采样点。
3. **有多少帧？** 按"每 10 ms 一个帧起点"的口径：480,000 ÷ 160 = **3,000 帧**（原课的 ~3,000 即此）。若严格按原课 `frame_signal` 的切片规则（起点从 0 到 480,000−400，步进 160，且不许最后一帧越界），则是 floor((480,000−400)/160)+1 = **2,998 帧**——差的 2 帧是尾部边界处理，量级不变。
4. **输入"图"有多少格？** 3,000 帧 × 80 mel 格 = **240,000 个输入数值**。这就是 Whisper 眼里的"一张 3,000×80 的图"。
5. **编码器实际处理多少 token？** 卷积茎两层 stride-2 卷积把 3,000 帧砍半再砍半 → **1,500 个 token** 进入 24 层编码器（large 模型，d_model=1,280）。

序列长度的量级感（补充，接 [第09课](<09：视觉Transformer——ViT与图像块.md)）：一句话约 30 个词；ViT 一张图 197 个 token；**Whisper 30 秒音频 1,500 个 token**——比 ViT 还长 7 倍多，这就是音频任务天然更贵的原因。

再验一下 1 秒的小情形（原课 Easy 练习）：16,000 个采样点，严格切片是 floor((16,000−400)/160)+1 = **98 帧**，原课口径"约 100 帧"=16,000÷160。手算对得上。

时间戳粒度：0.02 秒一个 → 30 秒共 30/0.02 = 1,500 个可能的时间戳位置，恰好与编码器 token 数同量级。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

这是 **Learn 型课**：原课没有完整 Build It 代码，只有两个短函数——`frame_signal`（分帧）和 `whisper_prompt`（任务 token 拼接），其余步骤是文字描述。读的顺序：先看 `frame_signal` 对应 0.3 的第 2、3 步（400/160 的由来）；再看 `whisper_prompt` 对应第二节的"任务 token"（Step 5）；架构本身（mel 谱、卷积茎、编码器、解码器）在第二节按概念理解，不敲代码。原课明说：生产里你真正会碰的就是这两段——分帧流水线和提示词格式。Learn 型课的学法与 Build 型不同：动手的分量减少，把时间花在"每个架构决策为什么这么做"上——本节的 0.5 检查题就是为此设计的。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 mel 频谱说是一张"图"？它的"行、列、像素值"分别是什么？

**参考答案：** 列是时间帧（每 10 ms 一帧），行是 80 个 mel 频率格，像素值是那一帧里该频带的能量（取过对数）。3,000 列 × 80 行 = 240,000 个格子——一张长条形的灰度图，所以原课说 Whisper 是"吃 mel 谱图的 ViT"。

**问题2：** 同一段法语音频，想要英语文字输出，解码器开头要放什么 token？

**参考答案：** `<|startoftranscript|>`、`<|fr|>`、`<|translate|>`、`<|0.00|>`——语种 token 标音频的语言（fr），任务 token 选"翻译"而不是"转写"。模型在训练时就见过这种组合，等于用前缀控制任务。

**问题3：** 为什么 Whisper 处理 1,500 个 token 而 30 秒音频明明有 3,000 帧？

**参考答案：** 编码器前的卷积茎（两层 Conv1D）把帧数从 3,000 降到 1,500，1,500 是进入编码器的数量。这个降采样很便宜，却让注意力开销降到四分之一。（核对：原课正文写"两层 kernel 3、stride 2"，但两层都取 stride 2 应得 3,000→1,500→750；真实 Whisper 的卷积茎是第一层 stride 1、第二层 stride 2，结果 3,000→1,500。原课的 1,500 与真实模型一致，"两层都是 stride 2"应为其笔误，以 1,500 为准。）

能自己算出 480,000 → 3,000 帧 → 1,500 token 这条链后，进入正文。

---

## 一、这一课解决什么问题？

Whisper 之前，最先进的自动语音识别（ASR）是 wav2vec 2.0 和 HuBERT 路线：先自监督学特征提取器，再微调一个输出头。质量高，但数据管线昂贵、换个领域就脆，多语言要按语系分别建模型。Whisper 押了三个赌注：一，**用一切数据训练**——从互联网抓的 68 万小时、覆盖 97 种语言的弱标注音频，不用干净学术语料、不用音素标注；二，**多任务单模型**——一个解码器联合学转写、翻译、语音活动检测、语种识别、时间戳，全靠任务 token 区分；三，**标准编码器-解码器 Transformer**——编码器吃 log-mel 谱，解码器自回归出文本，不要声码器、不要 CTC、不要 HMM。

三个赌注里最反直觉的是第一个。传统 ASR 的信仰是"数据质量决定一切"：音素级标注、安静录音棚、语言学家人工校对。Whisper 的发现（原课参考）是：互联网规模的弱标注（视频字幕、播客配文，带噪声、带错）在规模面前足够好——模型在 68 万小时里自己学会了过滤口音和噪声。这与 [第09课](<09：视觉Transformer——ViT与图像块.md>) ViT 的故事同构：砍掉手工先验（音素标注↔卷积归纳偏置），用规模换回来。结果（原课参考）：Whisper large-v3 对口音、噪声、以及零标注语言都很稳，是 2026 年开源语音助手和多数商业产品的默认语音前端。本课要回答的问题是：**这套架构为什么这么普通，却这么能打？** 答案不在结构里，在数据和对齐方式里。

## 二、核心概念与推导

### 2.1 流水线总览（原课六步）

原课把 Whisper 的前向过程拆成六步，对应 0.3 的手算链：

1. **重采样+定窗**：音频统一到 16 kHz，裁剪/补零到 30 秒，算 log-mel 谱：80 个 mel 格、10 ms 步进 → 约 3,000 帧 × 80 特征。这就是"Whisper 看到的输入图"。
2. **卷积茎**：两层 kernel=3、stride=2 的 Conv1D，把 3,000 帧降到 1,500——序列减半，参数增加很少。（核对：两层都取 stride 2 会得到 750 而非 1,500；真实 Whisper 是第一层 stride 1、第二层 stride 2。原课结果 1,500 与真实模型一致，步长描述以其结果为准。）
3. **编码器**：24 层（large）Transformer 编码器处理 1,500 个时间步，正弦位置编码 + 自注意力 + GELU FFN，输出 1,500 × 1,280 的隐藏状态。结构与 [第05课](<05：完整Transformer——编码器与解码器.md>) 的编码器一致，只是序列更长、没有 [CLS]。
4. **解码器**：24 层 Transformer 解码器，从 BPE 词表（GPT-2 词表的超集，外加几个音频专用特殊 token）里自回归地生成文本 token。交叉注意力每次都看编码器全部 1,500 个隐藏状态——这是"边听边写"的连接点。
5. **任务 token**：解码器提示以控制 token 开头，告诉模型做什么（下详）。
6. **输出**：beam 宽度 5 的束搜索加对数概率阈值；没有 `<|notimestamps|>` token 时，每 0.02 秒预测一次时间戳。

（补充）把六步分成两半记：第 1—3 步是"把声音变成编码器能吃的序列"，全是第09课见过的把戏（非文本数据 → 二维图 → token 序列）；第 4—6 步是"按指令写文字"，全是第07—08课见过的把戏（自回归解码器+提示控制）。Whisper 的"新"，从头到尾只有数据和对齐方式。

### 2.2 任务 token：用前缀控制任务

解码器提示的两种典型开头（原课例子，原样转录）：

~~~
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
~~~

~~~
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
~~~

第一组：英语音频、转写成英语文字。第二组：法语音频、翻译成英语文字。模型训练时就见过这些约定，你靠前缀控制任务——原课称这是"指令微调的 2026 等价物，只是用在语音上"。（补充：与 [第07课](<07：GPT——因果语言建模.md>) 的差别在于，GPT 的"指令"是自然语言，Whisper 的指令是保留的特殊 token，更短且无歧义。）

为什么 4 个 token 就够（补充，拆开看每一格）：

| 位置 | token | 回答的问题 |
|------|-------|-----------|
| 1 | `<\|startoftranscript\|>` | "开始了"，同时充当解码序列的起始符 |
| 2 | `<\|en\|>` / `<\|fr\|>` | 音频是什么语言（97 选 1） |
| 3 | `<\|transcribe\|>` / `<\|translate\|>` | 原样转写还是译成英语 |
| 4 | `<\|0.00\|>` / `<\|notimestamps\|>` | 要不要带时间戳，从哪个时刻起 |

关键在于：这些组合在 68 万小时训练数据里出现过足够多次，模型把它们当成"任务的语法"学了下来——所以同一个解码器能身兼数职，而不会互相打架。

### 2.3 模型家族尺寸表（原课表）

| 模型 | 参数量 | 层数 | d_model | 头数 | 显存（fp16） |
|------|--------|------|---------|------|--------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

表内数字均为原课参考。注意 turbo 的行：参数量 809M、层数 32——因为它的**解码器只有 4 层**（原课表格备注），编码器仍是 large-v3 的 32 层。2024 年的 turbo 把解码器从 32 层砍到 4 层，解码快 8 倍，词错误率回退不到 1 个点（原课参考）——这个解码速度的解锁正是它在 2026 年成为实时语音助手默认选择的原因。（补充：解码是逐 token 的串行过程，砍解码器层数直接砍每步延迟；编码器只需对 30 秒窗跑一遍，且可以被 KV 缓存复用，见 [第12课](<12：KV缓存与Flash Attention——推理优化.md>)。）

### 2.4 Whisper 不做的事（原课边界，重要）

- **不做说话人分离**（谁在说）：要配 pyannote。
- **不原生支持实时流式**：30 秒窗口是固定的。现代封装（faster-whisper、WhisperX）用 VAD+重叠分块外挂了流式。
- **超过 30 秒没有长程上下文**：要外部切块。原课说实践上问题不大——人类语音的转写很少需要超长程上下文。

（补充）把这三条放在一起读，能看到一个共同根源：Whisper 的训练单位是"30 秒窗+窗内文本"。说话人信息、跨窗上下文、逐样本递进都不在这个目标函数里，所以都不是模型的能力——要么外挂工具补，要么接受边界。选型时先问"我的任务能不能塞进 30 秒窗"，答案决定后面全部工程。

### 2.5 2026 年的任务格局（原课表）

| 任务 | 模型 | 备注 |
|------|------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘端快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可达 150 ms 延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但形状同 Whisper |
| 音频+语言 | AudioLM、SeamlessM4T | 文本 token 与音频 token 同处一个 Transformer |

（补充）读表方法：注意每个"替代品"赢的维度都不同——Moonshine 赢在边缘端速度，faster-whisper 赢在工程化延迟，pyannote 赢在 Whisper 没有的能力。没有任何一行是"全面替代 Whisper"，所以选型题永远是"你的约束落在哪一行"。TTS 一行还提示一个有趣的事实：文字转语音（Piper 等）与语音转文字（Whisper）是同一副编码器-解码器骨架的两个方向。

### 2.6 把 0.3 的数字串回架构（补充）

480,000 个采样点 →（400/160 分帧）→ 3,000 帧 × 80 mel 格 →（两层 stride-2 卷积茎）→ 1,500 token × 1,280 维 →（24 层编码器）→ 1,500 个隐藏状态；解码器每步生成一个文本 token，提示以 `<|startoftranscript|> <|en|> <|transcribe|> <|0.00|>` 开头。0.02 秒的时间戳粒度 × 1,500 = 30 秒，正好铺满整个窗口。每个数字都能在 2.1 的六步里找到落点。

## 三、操作前准备

Build It 部分只用 Python 标准库（math、random），无需第三方依赖——原课的简化版分帧不要求 librosa。Use It 部分需要 OpenAI 官方包或 faster-whisper：

~~~bash
python -m pip install openai-whisper        # 官方参考实现
python -m pip install faster-whisper        # 生产推荐封装
~~~

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供按需执行，本次没有替你安装或下载模型（模型权重在首次调用时才下载，体积从 Tiny 的约 39M 到 large-v3 的约 1.5B 参数不等）。（补充）两个包不必都装：学原理、跑对照选官方包；做任何实际转写任务直接选 faster-whisper——int8 量化、CTranslate2 后端、VAD 集成都是现成的。Medium 练习的完整 mel 实现另需 `numpy`（FFT）与 `librosa`（对照），也按需安装。

## 四、Build It：原课分步代码与中文讲解

（核对）本课为 **Learn 型**：原课 Build It 只展示了两个代码块（`frame_signal` 与 `whisper_prompt`），第 1、3 步为文字描述，未随页展示完整 `main.py`。本节如实转录这两个代码块并讲解，文字步骤用中文转述，不虚构代码。

### 4.1 第 1 步：合成音频（文字步骤）

生成 1 秒、440 Hz 的正弦波，16 kHz 采样，共 16,000 个采样点。440 Hz 是标准音 A4——每秒 440 个完整振动周期，16,000 个采样点里约有 440 个周期。（补充：为什么用正弦波？它是纯音，mel 谱上会在对应的频率格亮起一条横线，是检查特征提取管线的"已知答案"信号——你若做 Medium 练习实现完整 FFT 版 mel 谱，应该能在 440 Hz 附近看到能量集中，这是天然的验收标准。）

### 4.2 第 2 步：简化版 log-mel 谱（原课代码，原样转录）

完整的 mel 频谱需要 FFT。原课做一个简化的"分帧+逐帧能量"版本，展示流水线而不依赖 librosa：

~~~python
# 本块中文说明：把一维采样序列切成重叠的帧，帧长 25ms（400 样本）、步进 10ms（160 样本）。
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
~~~

逐行看：`range(0, len(x) - frame_size + 1, hop)` 枚举所有不越界的帧起点——0、160、320……；每帧切出连续的 400 个采样点。原课说明：帧=25 ms、步进=10 ms，与 Whisper 的窗口参数一致；逐帧能量在教学中代替 80 个 mel 频格（核对：真实 Whisper 用 FFT 把每帧分解到 80 个频带，本函数只保留"分帧"这一步，是教学简化，不是可用的特征提取器）。

0.3 节的手算可以直接验证：1 秒（16,000 样本）→ floor((16,000−400)/160)+1 = 98 帧；30 秒（480,000 样本）→ 2,998 帧，约等于原课口径的 3,000。

### 4.3 第 3 步：补齐到 30 秒（文字步骤）

Whisper 永远按 30 秒块处理：把频谱补零（或裁剪）到 3,000 帧。短音频不会报错，只是后半段全是用零填充的帧。

### 4.4 第 4 步：构造提示 token（原课代码，原样转录）

~~~python
# 本块中文说明：拼出 Whisper 解码器开头的任务控制前缀。
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
~~~

原课的原话：这就是任务控制的全部表面——一个 4 token 的前缀。（补充：默认参数对应"英语转写带时间戳"；法语翻译就是 `whisper_prompt(lang="fr", task="translate")`，与 2.2 节的第二组例子一致。）

## 五、把代码运行起来：补充入口与检查

（补充）按 4.2 + 4.4 的两个函数即可在本机跑最小检查。以下是本机实际执行过的验证（纯标准库，不下载模型）：

~~~python
# 本块中文说明：（补充）分帧数量检查——0.3 节手算的落地。
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames

one_sec = [0.0] * 16000        # 1 秒静音（16 kHz 采样）
thirty_sec = [0.0] * 480000    # 30 秒静音
# 手算预期：1 秒 → 98 帧；30 秒 → 2998 帧（严格切片口径，原课说 ~100 / ~3000）。
assert len(frame_signal(one_sec)) == 98
assert len(frame_signal(thirty_sec)) == 2998
# 每 10ms 一个起点的口径：16000/160 = 100，480000/160 = 3000。
assert 16000 // 160 == 100 and 480000 // 160 == 3000
~~~

（补充）补充检查方法：给 `frame_signal` 喂 0 到 999 的整数序列、frame_size=100、hop=50，手算预期 `len(frames) == 19` 且 `frames[1]` 从 50 开始——验证"起点枚举不越界"的边界行为。`whisper_prompt` 则检查三种组合的 token 数（3 或 4）。（补充）三个组合的手算预期：`whisper_prompt()` → 3 个 token（默认英语转写带时间戳）；`whisper_prompt(timestamps=False)` → 4 个（多了 `<|notimestamps|>`）；`whisper_prompt(lang="fr", task="translate")` → 3 个。`frame_signal` 的 19 帧来自 floor((1,000−100)/50)+1 = 19。

## 六、Use It：使用现成工具

原课给出两段用法（原样转录）。官方参考实现：

~~~python
# 本块中文说明：官方 whisper 包的最小转写。
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
~~~

更快、与 OpenAI 兼容的 faster-whisper：

~~~python
# 本块中文说明：faster-whisper 的 int8 量化推理，带 VAD 过滤。
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
~~~

（补充）读这两段代码时注意三个对应：`language=` 和 `task=` 参数最终会被拼成 2.2 节那 4 个 token 的前缀——你在使用 4.4 的 `whisper_prompt` 所做的事的现成版；`vad_filter=True` 就是先跑 VAD 切静音，直接对应第七节的降本手段；`segments` 里的 `start/end` 是 0.02 秒粒度时间戳的聚合成句版本。两段代码做同一件事，差别是后者用 CTranslate2 重实现、int8 量化，比官方参考实现快约 4 倍（原课参考）。

原课给的 2026 选型建议——**选 Whisper**：多语言 ASR 只想维护一个模型；嘈杂、多样音频的稳健转写；研究/原型最快起点。**选别的**：边缘端超低延迟流式（Moonshine 同质量下更快）；实时对话 AI 要 <200 ms（专用流式 ASR）；说话人分离（Whisper 不做，外挂 pyannote）。

## 七、失败模式、原文边界与工程注意

1. **30 秒窗口是硬边界**：长音频必须外部切块；有长程依赖的转写（跨段指代、术语）会在切块处断掉。faster-whisper/WhisperX 的 VAD+重叠方案是补丁，不是架构能力。
2. **没有说话人分离**：多人对话场景直接转写会混在一起；先过 pyannote 再对每段跑 Whisper 是标准组合（原课建议）。
3. **简化分帧不是 mel 谱**（核对边界）：4.2 的 `frame_signal` 只展示"切窗"这一步，逐帧能量只是教学的替身；真实输入必须走 FFT+mel 滤波器组，Medium 练习就是让你补这一块。
4. **任务 token 拼错不报错**（补充）：`whisper_prompt(lang="english")` 这类写法不在 97 个语言 token 里，行为未定义——错误前缀只会让输出变差，不会有显式报错，排查时要先检查前缀。
5. **流式延迟不是 Whisper 的原生能力**（原课边界）：150 ms 目标靠的是 faster-whisper + VAD 的工程封装；直接循环调用官方 `transcribe` 达不到。
6. **词表是 GPT-2 超集**（补充）：意味着解码侧的文本处理（分词、特殊 token）与 GPT-2 工具链兼容，但也继承其多语言分词粒度不均的问题。
7. **turbo 的取舍**（原课参考）：<1 个 WER 点的回退换 8× 解码加速，对实时场景几乎总是划算；对离线高精度场景，large-v3 完整解码器仍是首选。
8. **时间戳粒度 0.02 秒**（补充）：需要更细对齐（如字幕逐字）时要用专门的对齐工具，Whisper 原生粒度到此为止。
9. **mel 谱是"有损照片"**（补充）：相位信息在取能量时就丢了——Whisper 永远无法从输出恢复原始波形，它只做"听写"，不做"录音"。这也是 TTS 需要另一套模型（2.5 表）的原因之一。
10. **编码器 1,500 token 的推理成本不可忽略**（补充，接 [第12课](<12：KV缓存与Flash Attention——推理优化.md>)）：每 30 秒窗的编码器前向是 1,500 个 token 的自注意力，长音频分块处理时这是主要算力开销之一；好在编码器输出在同一窗内可复用，真正逐 token 增长的是解码器一侧。

## 八、Ship It：本课可复用的中文检查模板

原课 Ship It 产出 `outputs/skill-asr-configurator.md`（一个"给新语音应用选 ASR 模型、解码参数与预处理流水线"的技能文档）。改写成中文自查模板，直接保存在本篇，不另建输出文件：

~~~markdown
检查 ASR 选型：说明音频语言数量与是否需要翻译（多语言→Whisper，仅英语边缘端→Moonshine/小模型）；说明延迟目标（离线/近实时/<200ms 流式）并指出流式靠 VAD+分块封装而非原生能力；是否需要说话人分离（需要则加 pyannote）；预处理是否包含 16kHz 重采样与 VAD；解码参数（beam 宽度、温度回退、时间戳开关）是否说明依据；所有未实测的 WER/速度数字标为参考。
~~~

## 九、Exercises：保留原课练习与验收方法

1. **原课 Easy**：运行 `code/main.py`，确认 1 秒、16 kHz、10 ms 步进的信号约 100 帧，30 秒约 3,000 帧。验收标准：能报出严格切片口径的 98 与 2,998，并解释与 100/3,000 的差异来自尾部边界处理（本篇第五节的断言已实际执行覆盖此题）。
2. **原课 Medium**：用 `numpy.fft` 实现完整的 log-mel 频谱，验证 80 个 mel 格与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差内一致。验收标准：给出两者输出形状一致（帧数 × 80）且逐格误差小于你声明的阈值，并指出差异来源（mel 滤波器组实现、对数底数）。
3. **原课 Hard**：实现流式推理：音频切成 10 秒窗、2 秒重叠，逐窗跑 Whisper 后合并转写，在 5 分钟播客样本上与单次整段推理比词错误率。验收标准：报告两种方式各自的 WER 与耗时（自己实测），并解释重叠区去重的策略（如按时间戳中点切分）。

（补充）做 Hard 题时的一个已知坑：重叠区同一个词会被两个窗口各转写一次，直接拼接会出现重复词——先按时间戳丢弃重叠区的一半再拼，WER 才有意义。这正好复现了第七节"30 秒窗是硬边界"的工程后果。

## 十、术语与完成标准

mel 频谱：时间×频率的能量图，音频的"图像"表示；log-mel：取对数的 mel 谱，Whisper 的实际输入；帧：25 ms 窗口，10 ms 步进；卷积茎：编码器前的两层 stride-2 Conv1D，序列 3,000→1,500；任务 token：解码器开头的控制前缀（语种+任务+时间戳开关）；BPE：子词词表，Whisper 的是 GPT-2 超集；束搜索：每步保留 5 个候选的解码；VAD：切掉静音的预处理；CTC：老式 ASR 损失，Whisper 不用；弱标注：互联网字幕级质量的训练数据；说话人分离：标注谁在说，需外挂；WER：转写词错误率；turbo：4 层解码器的 large-v3 变体。

完成检查：

- 能手算 30 秒音频 → 480,000 采样 → 3,000 帧 → 1,500 token 这条链。
- 能说清 98 与 100、2,998 与 3,000 两组数的差异来源。
- 能写出"法语音频翻成英语"的正确任务 token 前缀。
- 能解释 Whisper 为什么不用 CTC、不做流式、不做说话人分离，以及各自的补法。
- 能说出 turbo 砍掉的是解码器还是编码器，以及为什么砍它最划算。
- 能解释"弱标注 + 大规模"为什么能赢"精标 + 小规模"，以及这个结论与第09课 ViT 的哪个教训同构。
- 能把 80 个 mel 格、1,280 维、0.02 秒这三个数各自挂到流水线的哪一步上。

## 十一、验证范围与来源许可

原课 Build It 共 4 个步骤，其中 `frame_signal` 与 `whisper_prompt` 两个代码块逐字转录嵌入本篇（仅按本篇格式添加中文说明，未改逻辑）；第 1、3 步为课程页文字描述，本篇如实转述。Use It 的 whisper / faster_whisper 代码原样保留。本课为 Learn 型，课程页未展示训练或运行输出，本篇未虚构任何"原课参考"数值（模型表、8× 加速、<1 WER 点回退、68 万小时、97 语言、faster-whisper 4× 等均为原课参考）。另：本篇在 2.1/0.5 处显式标注了原课"两层 stride-2 卷积把 3,000 帧降到 1,500"的自相矛盾（按真实 Whisper 应为 stride 1+2），以原课结果数字 1,500 为准。

本机验证记录（2026-11-06，纯标准库脚本，不构成性能实验）：

- 抓取课程页成功，以原文为准整理。
- 第五节标注"手算预期"的断言（98 帧、2,998 帧、100、3,000）已实际执行通过；0.3 节全部算例与之互证。
- 已转录的原课两个函数通过语法检查。
- 未做：下载 Whisper 模型、真实音频转写、FFT/mel 实现、WER 测量。

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
