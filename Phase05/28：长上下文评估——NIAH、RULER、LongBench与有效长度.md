# 28：长上下文评估——NIAH、RULER、LongBench与有效长度

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对概念与示例。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/28-long-context-evaluation) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/docs/en.md)。原题：Long-Context Evaluation — NIAH, RULER, LongBench, MRCR。

## 零、初学者先读：宣传的窗口不等于能用的窗口

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课是“评估三部曲”的第二课：[第27课](<27：LLM评估——RAGAS、DeepEval、G-Eval与人工校准.md>)评“答得好不好”，本课评“文档一长还行不行”，[第29课](<29：对话状态追踪——槽位、修改、清除与JGA.md>)评“多轮对话追得对不对”。与第27课同源的道理：LLM能力是行为不是公式，[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)那种一个指标定胜负的做法在这里不够用。

### 0.1 先理解这一课在做什么

模型卡片写“支持1M token上下文”，你把200页合同贴进去问“终止条款是什么”，它答了——但答的是封面页的内容，因为终止条款埋在12万token深处，超出了模型**实际注意**的范围。这就是2026年的“上下文容量差”（原课参考）：规格书说10M，实测约60%～70%可用，且“可用”随任务变——单针检索接近满窗，多跳推理在128k附近就开始塌。本课的任务：设计测试，把“宣传窗口”和“任务可用长度”分开量。

类比：手机电池标5000mAh，实际亮屏续航取决于你干什么——待机一天半，打游戏三小时。“标称容量”是宣传窗口，“干这件事能撑多久”是有效长度。NIAH就是“待机测试”（最容易的检索），RULER的多跳变量链是“游戏测试”（最难的多步推理）——待机吓人地长的手机，打游戏照样歇菜。“大海捞针”这个中文名字说的就是这项测试：干草堆（filler干扰文本）里埋一根针（一个事实），问模型针是什么。

### 0.2 本课术语先翻成白话

- NIAH（needle in a haystack，大海捞针）=干扰文本中埋一个事实、问出来；最原始的长上下文基准
- 针（needle）=被埋的事实，如“魔法单词是pineapple”；干扰堆（haystack/filler）=其余无关文本
- 深度（depth）=针埋的位置比例：0=最开头，0.5=正中间，1=最末尾
- 多针（multi-needle）=埋多条针、问“三个魔法单词都是什么”——考注意力分配，不是单纯检索
- 丢失在中间（lost in the middle）=模型对中间内容的注意力系统性弱于两端——所以深度要覆盖0、0.25、0.5、0.75、1.0
- RULER=13种任务的合成基准（检索/多跳/聚合/问答四类）；原课参考：2024版中宣称32k+的17个模型只有一半在32k保持质量
- LongBench v2=503道多选题、8k～2M词真实长文的生产级基准（原课参考）
- MRCR=多轮共指消解的规模化测试（8/24/100针变体）
- NoLiMa=针与问题零字面重合，逼模型走一步语义推理
- BABILong=把推理链埋进干扰堆，考“干扰中的推理”而非“干扰中的检索”
- 有效长度（effective context）=指定任务、指定阈值下仍达标的长度——不是一个数，是“任务×阈值”的二维量
- TTFT=首token耗时（time to first token）
- 宣传窗口=规格书允许输入的长度上限

### 0.3 用小数字走一遍

一个100词的干草堆、一根2词的针、三个深度，把“构造NIAH”和“读有效长度”走一遍（算例，参数与正文第五节的build_haystack调用一致）：

**第一步：埋针。** 预算100个空格词，针是“SECRET_CODE ZXQ731”（2词），所以干扰体=100−2=98词。插入位置=int(98×depth)：

~~~text
深度0.0：[针][干扰×98]              ← 开头：针在第0词
深度0.5：[干扰×49][针][干扰×49]      ← 中间：针在第49词
深度1.0：[干扰×98][针]              ← 结尾：针在第98词
~~~

手算复核：int(98×0.5)=49——前49个词、2词的针、后49个词，总计100。为什么必须测三个深度：lost in the middle效应——不少模型开头结尾能捞到、正中间反而漏（原课参考）。正文第五节的断言`len(haystack.split()) == 100`和`index("SECRET_CODE") == 49`就是这两个数字。

**第二步：加干扰强度。** 单针加“SECRET_CODE”这种任何filler里都不会出现的词，等于在干草堆里放了块磁铁——检索太容易。NoLiMa的对策：让针与问题零字面重合（问“神奇单词”，针里写的是“菠萝”），逼语义检索。

**第三步：读有效长度。** 假设对某模型测得（数字为算例）：

| 上下文长度 | 检索型（单针）通过率 | 推理型（变量链）通过率 |
|---|---|---|
| 4k | 100% | 95% |
| 16k | 100% | 82% |
| 64k | 96% | 50% |
| 128k（宣传上限） | 91% | 30% |

按“检索≥90%、推理≥70%”两个阈值读表：有效检索长度=128k（91%≥90%仍达标），有效推理长度=16k（82%≥70%，64k的50%不达标）。注意推理有效长度只有宣传窗口的1/8，比原课经验范围（25%～50%）更糟——这正是要自己测的原因：经验比例是参考，退化曲线才是证据。报告要写“检索有效128k、推理有效16k”，而不是笼统一句“支持128k”。

变量链顺带手算（推理型的最小例，正文4.3）：X1=42；X2=X1+10=52；X3=X2×2=104——答X3必须同时找到两条赋值并串起来，这是“多跳”。

以上是用于理解的算例，不是新增的模型实测。

### 0.4 怎样连接到正文代码

四段代码对应四类测试：4.1的build_haystack是0.3第一步的构造器（tokenize在第五节补齐为空格分词——注意“空格词”不是模型token，正文反复强调的坑），score_niah用子串匹配评分，正文指出它会把“不是ZXQ731”误判通过；4.2的build_multi_needle是多针变体——原草图混用字符和token单位，正文列了问题；4.3的变量链就是0.3末尾的X1→X2→X3；4.4的LongBench接入有数据卡不符的纠错（split与字段名都要改）。第五节是可直接跑的入口：100词断言、index==49就是0.3的数字。

### 0.5 理解检查（含参考答案）

**问题1：** “模型支持1M上下文”和“1M上下文可用”差在哪？

**参考答案：** 前者是规格（允许输入的长度），后者是有效长度（指定任务、指定阈值下仍达标的长度），后者随任务和阈值变：单针检索的可用长度远大于多跳推理。原课参考：推理有效长度通常只有宣传窗口的25%～50%，且要用退化曲线自己实测。

**问题2：** 为什么NIAH必须覆盖开头、中间、末尾多个深度？

**参考答案：** lost in the middle效应：模型对中间内容的注意力系统性偏弱。只测一个深度（很多实现偷懒只测0.5）会漏掉位置敏感的退化，0.3第一步的三个深度就是为它准备的。

**问题3：** NIAH满分能推出多跳也没问题吗？

**参考答案：** 不能。NIAH只考“找到一根针”（检索），多跳/聚合/推理是不同的能力轴（RULER/BABILong考的）；原课明确说NIAH是必要不充分的基线——0.3的表里检索128k仍91%，推理已塌到30%。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

模型允许输入很长，不代表能在同样长度完成检索、推理与多事实聚合。本课按长度×证据深度×任务类型设计测试，区分宣传窗口与任务可用长度。

## 二、核心概念与推导

NIAH在干扰文本中埋事实并问出；multi-needle要找多条；RULER扩展到追踪/聚合/问答；LongBench v2包含真实长文多选任务；MRCR侧重多轮相似请求与指代；NoLiMa减少字面匹配捷径；BABILong把推理链放进干扰上下文；HELMET涉及多类长上下文能力，不能只理解为单一文档拼接题。

（补充 2026-11-06）一串基准按“考什么能力”排序，就不再是一堆名词：NIAH最底（检索一根针，前沿模型已饱和——必要不充分）；多针进一阶（同时攥住几根针，考注意力分配）；RULER把任务面铺开成四类13种（检索/多跳/聚合/问答）——原课参考：2024版里宣称32k+的17个模型只有一半在32k保持质量，说明NIAH饱和≠长上下文没问题；LongBench v2用8k～2M词真实长文多选，考生产形态；MRCR专攻多轮共指（对话场景，与[第24课](<24：共指消解——提及、先行词、聚类与跨句实体.md>)的共指是同一能力的规模化压测）；NoLiMa把字面捷径堵死（针与问题零词面重合，逼一步语义推理）；BABILong把推理链埋进干扰堆。选型原则一句话：先NIAH确认“能找到”，再上至少一种推理任务确认“能想通”——本课Use It的原文建议“NIAH+1个推理任务”才算数。

报告模型允许窗口、检索有效长度、推理有效长度和退化曲线，同时标注阈值、样本数与置信区间。90%/70%只是可选业务阈值，不是“有效长度”的统一标准。原文10M、26.3%、25%～50%、60%～70%等型号/比例是来源中的描述，未在本地核实，不作为选型结论。

（补充 2026-11-06）“有效长度”不是一个数，是“任务×阈值”的二维量——这是本课最容易被忽略的观念。同一个模型，检索阈值90%下有效长度128k，推理阈值70%下可能只有16k（零节0.3的表）；把阈值从90%收紧到95%，检索有效长度也会缩水。所以报告必须写清三件事：什么任务、什么阈值、多少样本与置信区间——只报“有效长度64k”等于没说。这与[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)的教训同源：一个不带条件的指标数字，无法支持任何决策；那边丢的是“阈值和业务代价”，这边丢的是“任务和阈值”。

（补充 2026-11-06）本课最容易踩的计量坑：词数≠字符数≠模型token数。“100词”的干草堆用不同tokenizer切出的token数不同（中英文差异更大），LongBench数据卡的word长度也不能直接当目标模型的token数（正文第七节6条）。一张表收拢：

| 单位 | 谁的尺度 | 何时可信 |
|---|---|---|
| 空格词（split） | 你的脚本 | 教学演示（正文第五节的tokenize） |
| 字符数 | len(text) | 与模型容量无直接关系 |
| 模型token | 目标模型的tokenizer | 正式实验唯一合法单位 |

原课4.2的多针草图就是把total_tokens当字符数切片——单位一错，“128k测试”实际可能只测了几十个词。正式实验先用目标tokenizer验证实际长度（Exercises Easy的第一步）。

lost in the middle指位置敏感；测试应覆盖开头、中间、末尾，而非只测中点。单次成功不是通过率，要多种随机needle、不同背景和重复试验。

（补充 2026-11-06）“单次成功不是通过率”值得展开：一次捞到针可能是运气（恰好在开头）或巧合（filler里混进了相似词）。正式测法：同一配置换不同针内容、不同filler、多个随机种子重复试验，报通过率与置信区间——单点的0/1要聚合成百分比才有意义，这正是[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)“单次观测与统计结论”的区别在长上下文场景的翻版。同理，厂商自报分数要打折：OpenAI、Google、Anthropic都发布自己的长上下文成绩单，原课Use It明确要求“永远在自己的用例上重跑”——自己业务里文档是中文合同还是英文代码，退化曲线可能和厂商的英文小说测试完全不同。

### 2.9 用一个小例子把概念串起来（补充）

在100空格词预算中插入2词needle，背景需98词，深度0.5插在第49词位置。这不是100个模型token，真正tokenizer可能切成不同长度。3个needle还需减去全部needle预算、保证各条存在；原三固定depth草图不保证这些性质。

## 三、操作前准备

NIAH教学可用标准库；真实模型调用需自定义model.complete适配器并由你确认权限/费用。本次不调用服务、不下载LongBench。用目标tokenizer计数，而不是把split后的词数叫模型token数。完整预算还包括问题、模板、输出与模型特殊token。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 构建单needle干扰文本

原tokenize未定义，补充用空格词作教学单位。若needle长度大于total_tokens，源码仍保留整条needle，所以最终会超预算；先显式校验。score_niah只检查expected是答案子串，可能误判否定/顺序错误，应改任务解析器。

~~~python
# 教学长度依赖tokenize定义，needle超预算时需额外检查。
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


# 子串匹配只是简化评分，不能识别否定或错误归属。
def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
~~~

### 4.2 多needle变体

原实现用字符串切片，把total_tokens当字符数；只zip三个固定depth，超过3条needle会静默忽略；不重复filler也不保证总长。它是草图，不能据此宣称做到了指定token长度。

~~~python
# 原草图混用字符和token单位，正式实验前需重写。
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
~~~

### 4.3 多跳变量追踪

X1=42，X2=X1+10=52，X3=X2*2=104。要改变值/依赖/干扰位置并确保无歧义；一次固定答案可被记住而非推理。

~~~python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
~~~

### 4.4 LongBench v2接入

原示例split=test、字段task与真实数据卡不符：当前数据是train split，使用domain/sub_domain；还应把choice_A至choice_D放进提示，并按A/B/C/D评分。只拼question遗漏选项无法复现正式任务。

~~~python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
~~~

## 五、把代码运行起来：补充入口与检查

补充标准库入口，长度单位明确为“空格分隔词”，不是模型token：
~~~python
def tokenize(text):
    return text.split()

needle = "SECRET_CODE ZXQ731"
haystack = build_haystack("irrelevant background text", needle, 0.5, 100)
assert len(haystack.split()) == 100
assert "ZXQ731" in haystack
assert 2 * (42 + 10) == 104
print(len(haystack.split()), haystack.split().index("SECRET_CODE"))
~~~
预期100词，插入位置49；这只测构造器，不是LLM通过NIAH。

补充实际数据结构接入骨架（不执行下载）：
~~~python
# from datasets import load_dataset
# dataset = load_dataset("zai-org/LongBench-v2", split="train")
# print(dataset.column_names)
# print(sorted(set(dataset["domain"])))
# row = dataset[0]
# choices = "\n".join(f"{letter}. {row['choice_' + letter]}" for letter in "ABCD")
# prompt = row["context"] + "\n" + row["question"] + "\n" + choices
# 输出只允许一个A/B/C/D，并使用对应解析器和gold row["answer"]。
~~~
依据：[LongBench v2官方数据卡](https://huggingface.co/datasets/zai-org/LongBench-v2)，核对日期2026-10-25。分组筛选需读取真实domain值，不沿用猜测字符串。

## 六、Use It：使用现成工具

快速检查做NIAH网格；多跳任务加RULER/变量链，真实业务加LongBench或自有长文，长对话加MRCR。先小预算确认提示构造、token数与评分器，再扩大长度。记录TTFT、总延迟、吞吐和token费用；不因为厂商分数高就省略自有任务。



## 七、失败模式、原文边界与工程注意

1. 输入被SDK或模型偷偷截断会污染测量，记录实际发送长度与截断策略。
2. filler重复、needle唯一且易检索可能使任务过简单；NoLiMa不只是随便删掉共同词，还要保持可解。
3. 子串匹配会把“不是ZXQ731”误判通过，需结构化答案和否定检查。
4. 多needle应要求全对/逐项召回，并报告两者，不用一个答案出现算全部成功。
5. 海量token测试可能费用高，先列预算与次数，禁止本次自动跑付费测试。
6. 数据集word长度不等于目标模型token长度。
7. 原multi-needle构造器和LongBench骨架有上述问题，不能直接作为正式评测。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
指定模型版本、目标任务、输入/输出预算；设计多深度多长度NIAH、多needle及至少一种推理任务。固定种子/样本/评分，记录实际token、有效检索/推理长度、准确率置信区间、TTFT与成本；模型升级做相同回归测试。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：0.25/0.5/0.75深度×1k/4k/16k长度，先以目标tokenizer验证实际长度，再在授权模型上重复采样，画3×3通过率热图。
2. Medium：修正多needle构造器，保证3条全插入与总预算，比较全对率、逐条召回与单needle成绩。
3. Hard：在64k干扰中放变量链，对3模型实测；不必使用付费模型，先确认预算。原X1→X2→X3只有两次派生运算，若要求三跳，需再加X4，统一定义后比较。

## 十、术语与完成标准

NIAH：干扰中找埋入事实；effective context：指定任务/阈值下有效长度；depth：证据相对位置；TTFT：首token耗时；multi-hop：多步依赖推理；MRCR：多轮共指相关评测。

完成检查：

- 能区分词数/字符数/模型token数。
- 能算出变量链104。
- 能指出multi-needle草图的问题。
- 能解释单needle不等于多跳能力。

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

