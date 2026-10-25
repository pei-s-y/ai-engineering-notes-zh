# 28：长上下文评估——NIAH、RULER、LongBench与有效长度

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/28-long-context-evaluation) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/docs/en.md)。原题：Long-Context Evaluation — NIAH, RULER, LongBench, MRCR。

## 一、这一课解决什么问题？

模型允许输入很长，不代表能在同样长度完成检索、推理与多事实聚合。本课按长度×证据深度×任务类型设计测试，区分宣传窗口与任务可用长度。

## 二、核心概念与推导

NIAH在干扰文本中埋事实并问出；multi-needle要找多条；RULER扩展到追踪/聚合/问答；LongBench v2包含真实长文多选任务；MRCR侧重多轮相似请求与指代；NoLiMa减少字面匹配捷径；BABILong把推理链放进干扰上下文；HELMET涉及多类长上下文能力，不能只理解为单一文档拼接题。

报告模型允许窗口、检索有效长度、推理有效长度和退化曲线，同时标注阈值、样本数与置信区间。90%/70%只是可选业务阈值，不是“有效长度”的统一标准。原文10M、26.3%、25%～50%、60%～70%等型号/比例是来源中的描述，未在本地核实，不作为选型结论。

lost in the middle指位置敏感；测试应覆盖开头、中间、末尾，而非只测中点。单次成功不是通过率，要多种随机needle、不同背景和重复试验。

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

