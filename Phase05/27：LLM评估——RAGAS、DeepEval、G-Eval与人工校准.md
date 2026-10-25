# 27：LLM评估——RAGAS、DeepEval、G-Eval与人工校准

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/docs/en.md)。原题：LLM Evaluation — RAGAS, DeepEval, G-Eval。

## 一、这一课解决什么问题？

答案措辞不同不一定错，流畅也不一定对。LLM评估要先定义质量维度，再选择自动指标和人工校准，最后决定哪些退化应阻止发布，而不是把一个分数当真理。

## 二、核心概念与推导

faithfulness检查回答claim是否被给定context支持；answer relevance检查是否回答问题；context precision检查检索内容的相关性/排序质量；context recall检查必要参考证据是否被覆盖。具体库实现不一定等于简单比例，尤其context precision可采用排名敏感的计算。

RAGAS是框架，DeepEval是另一评测/测试框架，G-Eval是可按自定义rubric打分的方法及相关实现，不是三个必须全装的独立模型。不同指标可能需reference或外部judge，不能将全部指标叫reference-free。

LLM-as-judge用模型按规则评分，会受长度、自家模型风格、位置和提示注入影响。锁定judge、提示、数据、参数，并与人工评分比较。Spearman衡量排序相关，不等于绝对分值校准；还要误报/漏报、一致性和关键失败。

案例“June 29th, 2007”与“June 29, 2007”说明表面匹配可能苛刻，但F1具体值取决于正规化与切词，原文75%不是通用结果。评测价格、同模型偏置10%～20%、rho0.7都是原文示例/经验，不当保证。

### 2.9 用一个小例子把概念串起来（补充）

10个case里9个0.9、1个0，均值0.81仍可能掩盖一次严重失败。应查看底部样本、关键case、失败调用数与真实错误类别。judge与人工Spearman高也可能所有分数系统偏高，因此相关性之外还要校准门槛和检查误报/漏报。

## 三、操作前准备

~~~bash
python -m pip install ragas deepeval pytest transformers sentence-transformers numpy
~~~
不必一次安装所有库。RAGAS/DeepEval某些指标默认使用外部judge，会产生费用、上传数据并需要凭证；先读配置，使用脱敏/合成数据。本次不安装或发起评分请求，下面纯标准库先演示CI门禁。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 原子claim与NLI支持度

atomic_claims只是splitlines，空行/列表标记应清理；每claim分数阈值0.5需校准。与第21课一样要检查pipeline输出嵌套和标签大小写，避免next拿错层。它是RAGAS-style教学版，不是当前RAGAS所有实现的精确复现。

~~~python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


# 逐行拆分是教学近似，还需清理空行与校验claim质量。
def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


# 按claim计算支持比例，检查NLI返回结构与标签。
def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
~~~

### 4.2 由答案生成问题并比较

llm为answer生成若干可能问题，与原query向量余弦比较。相似度不是正确率，可能小于0；答案也可相关但事实全错。需要剔除空输出、处理不足n个问题，记录调用失败。

~~~python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
~~~

### 4.3 G-Eval自定义评分

criteria/evaluation_steps定义rubric。expected_output不完整时，额外正确事实也可能被错判不支持；评分要明确是否允许补充。示例metric.measure可能调用付费judge，未执行。

~~~python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
~~~

### 4.4 把评估接入测试

load_regression_cases未实现，case.id也不是所有LLMTestCase版本自带。要使用自己的case_id并将其与测试对象关联。0.85/0.7是示例门槛；先建立基线、置信区间与关键样本策略。

~~~python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


# 原代码是测试骨架，数据加载与case ID需要补齐。
def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
~~~

### 4.5 标准库toy评估

词重合可演示聚合与门禁，但不识别否定/角色关系，不能替代faithfulness。下面补纯数值回归门禁以检验失败是否被正确暴露。



## 五、把代码运行起来：补充入口与检查

补充标准库门禁，不会请求LLM：
~~~python
import math

def gate_scores(scores, minimum=0.7):
    if not scores:
        raise ValueError("评估集不能为空")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("包含失败或越界分数，不允许静默丢弃")
    ordered = sorted(scores)
    count = max(1, math.ceil(0.1 * len(ordered)))
    bottom_mean = sum(ordered[:count]) / count
    return {"mean": sum(scores)/len(scores), "bottom_10pct_mean": bottom_mean,
            "passed": bottom_mean >= minimum}

assert gate_scores([0.9] * 10)["passed"]
assert not gate_scores([0.0] + [0.9] * 9)["passed"]
print(gate_scores([0.0] + [0.9] * 9))
~~~
第二组均值0.81却仍有严重失败，这就是只报平均的问题。真实门禁可选择逐case、底部分位、配对退化与业务关键项，不建议机械照搬这一toy政策。

RAGAS context recall可能需要reference或gold ID，依据：[官方Context Recall文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)。

## 六、Use It：使用现成工具

RAGAS可用于RAG质量分析，DeepEval可结合pytest，G-Eval可加入领域rubric；LangSmith/Phoenix等可协助人工标注，Promptfoo可测试配置与安全。不必三框架全用；先用少量清楚指标。reference-free不等于不需要人工验证，也不等于不调用付费服务。



## 七、失败模式、原文边界与工程注意

1. 解析失败、NaN、超时要作为失败/缺失显式报告，并给分母；不能删掉后抬高均值。
2. judge升级会改变分数，先并行测旧新版本，冻结数据和rubric。
3. 不同家族judge可减轻部分偏置，但不会自动无偏，同模型评估也非绝对不可研究。
4. 两两比较交换A/B顺序，检查位置偏置。
5. 人工gold也需标注规范、双人一致性、仲裁和版本。
6. retrieved context中的恶意指令可能攻击judge，隔离待评文本与评分指令。
7. 不能用高faithfulness代替正确检索或完整回答；忠实地复述错误来源仍可能错。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
定义任务与失败成本，列faithfulness/relevance/precision/recall等指标及所需字段。冻结judge版本、rubric和样本，人工校准相关与绝对误差；CI报告成功/失败数、均值、底部样本、关键错误和配对退化。记录token/成本，升级前做并行比较。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：10条含已知无支持claim的RAG样本运行所配置的RAGAS指标，检查漏报，不预设每条一定能抓到。
2. Medium：50回答人工0～1评分，与G-Eval测Spearman及误差；常数评分无法定义有效相关，需检查分布。
3. Hard：建立pytest门禁，故意移除关键检索证据，确认门禁失败；加入最低10%与关键case检查，不把失败case静默删除。

## 十、术语与完成标准

rubric：评分规则；judge：评分模型；calibration：与人工校准；Spearman：秩相关；regression gate：退化门禁；golden set：版本化评估集。

完成检查：

- 能区分四个维度。
- 能指出哪些指标需reference。
- 能解释均值掩盖灾难样本。
- 能区分评分失败与低质量答案。

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

