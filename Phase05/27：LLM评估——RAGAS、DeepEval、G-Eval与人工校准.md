# 27：LLM评估——RAGAS、DeepEval、G-Eval与人工校准

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对概念与示例。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/docs/en.md)。原题：LLM Evaluation — RAGAS, DeepEval, G-Eval。

## 零、初学者先读：答案对不对，谁来判卷

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课是“评估三部曲”的第一课：视角从“造东西”切换到“验收东西”——前面所有课产出的输出怎么判断好坏。本课评单轮答案质量，[第28课](<28：长上下文评估——NIAH、RULER、LongBench与有效长度.md>)评长文档能力，[第29课](<29：对话状态追踪——槽位、修改、清除与JGA.md>)评多轮对话状态。三课都在回答同一件事：LLM输出是自由文本，[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)那种“一个准确率走天下”的做法不够用了。

### 0.1 先理解这一课在做什么

Phase02的[第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)评估分类器：标签是枚举，精确率/召回率/混淆矩阵是可复算的算术。本课面对的输出是**自由文本**，麻烦立刻出现（原课参考的例子）：参考答案“June 29, 2007.”，模型答“June 29th, 2007.”——Exact Match给0分、F1约75%、人类判卷100%。措辞不同不等于错，流畅不等于对：传统指标既苛刻（惩罚无害改写）又宽松（骗人的漂亮话能拿高分）。人工评最准但不扩展——一万条测试、每次改检索器/切块/提示词都要重评。本课的答案：**LLM当判卷老师（LLM-as-judge）+ 一层人工校准的信任**。

类比：传统指标像机读答题卡（涂对格子得分，字迹稍变就零分）；人工评像好老师（看意思给分，但贵且慢）；LLM-as-judge像雇了一位AI助教——便宜、快、大致像人，但有自己的毛病：偏爱长答案、偏爱自家文风、看A/B两份卷子时偏爱先看到的那份（位置偏置）、偶尔交白卷（JSON解析失败→NaN被静默剔除）。所以不能盲信助教：先拿100份人工判过的卷子对一对它的分数（校准，Spearman rho≥0.7才可信，原课参考），并且每次都翻一翻最差的那批（底部样本）——均值0.85可能藏着5%的灾难性错误。

### 0.2 本课术语先翻成白话

- LLM-as-judge=让一个LLM按评分规则给输出打0～1分
- rubric=评分规则（判卷标准），步骤写得越具体越稳定
- 忠实度（faithfulness）=答案的每条断言能否被检索到的上下文支持（接地不接地）
- 答案相关性（answer relevance）=答没答所问
- 上下文精确率（context precision）=检索回来的块里相关的占多少、排序好不好
- 上下文召回（context recall）=该有的证据检索回来了没有（需要gold答案）
- RAGAS／DeepEval／G-Eval=指标包／测试框架／评分方法——不是三个必须全装的独立模型
- reference-free=不需要参考答案的指标（忠实度、答案相关性）；context recall不在此列
- 校准（calibration）=judge分数与人工分数对表，Spearman rho<0.7则rubric返工（原课参考）
- Spearman rho=秩相关：judge给人排的序和老师排的序像不像
- 自评偏置=同一模型既生成又判卷，分数虚高10%～20%（原课参考）
- 位置偏置=两两比较时judge偏爱先出现的选项
- 回归门禁（regression gate）=分数低于阈值就阻止合并的CI关卡
- 底部样本（bottom quantile）=最差的那10%——平均值的天敌
- golden set=版本化的评估集，防止“数据烂掉”破坏纵向对比

### 0.3 用小数字走一遍

一个问题、三条答案、两个维度，把“忠实度”和“相关性”为什么必须分开走一遍（算例；忠实度的算法与正文4.1一致：答案拆原子断言→逐条问上下文支不支持→支持比例）：

~~~text
问题：第一部iPhone何时发布？
上下文（检索到的）：iPhone于2007年6月29日发布，售价499美元。
A1：iPhone于2007年6月29日发布。
A2：iPhone于2007年6月29日发布，售价599美元。
A3：iPhone的售价是499美元。
~~~

| 答案 | 拆出的断言 | 被上下文支持 | 忠实度 | 回答问题了吗 |
|---|---|---|---|---|
| A1 | 1条（发布日期） | 1/1 | **1.0** | ✓ |
| A2 | 2条（日期+“售价599”） | 1/2（价格是编的） | **0.5** | ✓（但夹带幻觉） |
| A3 | 1条（售价499） | 1/1 | **1.0** | ✗（答非所问） |

三个读数，三个教训：A3证明**忠实≠切题**——完全接地的话也可以不回答问题（相关性归相关性管）；A2证明**流畅≠忠实**——最自然的句子里夹一条编造，忠实度立刻掉到一半；A1也不是“满分答案”的终点——若检索到的上下文本身是错的（过期网页），忠实复述它的A1照样错。忠实度只管“有没有依据”，不管“依据真不真”（正文第七节7条）。

**没有参考答案时怎么评。** A1～A3的打分全程没用到“标准答案”：忠实度只需要(context, answer)——把答案拆成原子断言，逐条问“上下文蕴含它吗”（NLI，正是[第21课](<21：自然语言推断——蕴含、矛盾、中立与忠实性检查.md>)的蕴含判断）；相关性反着来——从答案“生成它可能回答的问题”，与真实问题比相似度（正文4.2）。这两个是reference-free。但context recall要判断“该检索到的证据是什么”，必须有gold答案做参照（原课依据：RAGAS官方文档）；G-Eval的Correctness也要expected output。所以“reference-free”不等于“不需要任何人工”——judge本身要跟人工校准，rubric和judge版本要冻结。

### 0.4 怎样连接到正文代码

四段代码按维度读：4.1的atomic_claims+faithfulness就是0.3的算法——llm先把answer按行拆断言，nli逐条打分，faithfulness返回支持比例（原代码是splitlines的教学近似，空行要清理）；4.2的answer_relevance是“反向生成问题”的实现；4.3的GEval把criteria+evaluation_steps写成rubric（自定义维度的标准做法）；4.4的pytest门禁把评估接入CI（0.85/0.7是示例阈值）。第五节的gate_scores是纯标准库的门禁实现——10个0.9加1个0的那组数字，就是零节0.1说的“均值掩盖灾难”的可运行版本。

### 0.5 理解检查（含参考答案）

**问题1：** 忠实度=1.0的答案一定是对的吗？

**参考答案：** 不。忠实度只衡量断言是否被检索上下文支持：上下文本身错了，忠实复述的答案照样错；答案答非所问（0.3的A3）忠实度也是1.0。它管接地，不管正确和切题。

**问题2：** reference-free的指标为什么还需要人工？

**参考答案：** 指标不需要每条样本的标准答案，但judge本身要先与人工评分校准（Spearman rho≥0.7），否则是噪声；judge版本、rubric、数据也都要冻结，否则分数漂移无法对比。

**问题3：** 均值0.81的评估集为什么不能直接放行？

**参考答案：** 可能是9条0.9加1条0：均值掩盖灾难样本。要看底部10%、失败调用数、关键case——正文第五节的gate_scores就实现了这种门禁（bottom_10pct_mean不达标就不放行）。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

答案措辞不同不一定错，流畅也不一定对。LLM评估要先定义质量维度，再选择自动指标和人工校准，最后决定哪些退化应阻止发布，而不是把一个分数当真理。

## 二、核心概念与推导

faithfulness检查回答claim是否被给定context支持；answer relevance检查是否回答问题；context precision检查检索内容的相关性/排序质量；context recall检查必要参考证据是否被覆盖。具体库实现不一定等于简单比例，尤其context precision可采用排名敏感的计算。

（补充 2026-11-06）四个维度用“考试”串起来最直观：faithfulness问“你写的是不是都来自发给你的讲义”——抄没抄别的书；answer relevance问“你答的是不是这道题”——答没答非所问；context precision问“发给你的讲义有几页真有用”——检索的效率与排序；context recall问“该发的讲义发全了没”——漏没漏证据。四个维度**互相不可替代**（零节0.3的A2/A3说明忠实度和相关性可以一高一低），所以RAGAS一次报四个，而不是合成一个总分。注意各维度的“证人”不同：faithfulness的证人是NLI模型（[第21课](<21：自然语言推断——蕴含、矛盾、中立与忠实性检查.md>)的蕴含判断），context recall的证人是gold答案——证人不同，成本和可信度也不同，正文说“不能将全部指标叫reference-free”正是这个意思。

RAGAS是框架，DeepEval是另一评测/测试框架，G-Eval是可按自定义rubric打分的方法及相关实现，不是三个必须全装的独立模型。不同指标可能需reference或外部judge，不能将全部指标叫reference-free。

（补充 2026-11-06）三者的关系容易误解：RAGAS是**指标包**（四个RAG维度，研究背景），DeepEval是**测试框架**（把LLM评估变成pytest，CI/CD原生），G-Eval是**打分方法**（rubric+思维链步骤+0～1分，在DeepEval里是一个metric类）。它们不是三选一的竞争关系——原课Use It的典型组合是RAGAS做监控、DeepEval做CI门禁、G-Eval补自定义维度。类比：RAGAS是体温计套装，DeepEval是体检流程，G-Eval是你自己发明的专项检查。

LLM-as-judge用模型按规则评分，会受长度、自家模型风格、位置和提示注入影响。锁定judge、提示、数据、参数，并与人工评分比较。Spearman衡量排序相关，不等于绝对分值校准；还要误报/漏报、一致性和关键失败。

（补充 2026-11-06）judge的四种已知毛病各配一个对策：①自评偏置——同一模型生成+判卷，分数虚高10%～20%（原课参考）→换不同家族的模型当judge；②位置偏置——两两比较偏爱先出现的→A/B顺序交换、两个方向都跑；③解析失败——JSON坏了出NaN，静默剔除会抬高均值→失败必须显式计数并给分母（正文第七节1条）；④版本漂移——judge升级分数全变→冻结judge模型+版本，升级时并行跑新旧对照。与传统指标的最大区别在此显形：

| | 传统指标（[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)） | LLM评估（本课） |
|---|---|---|
| 被测对象 | 枚举标签 | 自由文本 |
| 打分者 | 确定性公式（可复算） | 一个LLM（自身会错） |
| 表面不匹配 | EM=0、F1≈75%（原课参考） | 语义等价可识别 |
| 同数据同分数？ | 是 | 换judge/版本分数就变 |
| 信任来源 | 公式正确 | 与人工的Spearman rho |

精确率是算出来的，judge分数是另一个模型的行为——它自己也是被测对象。这就是为什么“校准”在传统评估里不存在、在这里是生命线。

案例“June 29th, 2007”与“June 29, 2007”说明表面匹配可能苛刻，但F1具体值取决于正规化与切词，原文75%不是通用结果。评测价格、同模型偏置10%～20%、rho0.7都是原文示例/经验，不当保证。

（补充 2026-11-06）“均值掩盖灾难”值得手算一遍（正文第五节gate_scores的数字）：10个case，9个得0.9、1个得0。均值=(9×0.9+0)/10=0.81，看起来体面；但把最差的10%（1个）单独拎出来是0分——如果那个0是“退款金额多写一个零”，均值0.81毫无安慰。这就是Ship It模板要求“底部样本必报”的原因：门禁不只看平均，还看bottom_10pct_mean（正文第五节的实现）。反过来也成立：只有底部好、整体差同样不行——两个统计量一个管下限、一个管水平。这与传统回归测试的思路同源（[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)的“看错误类型不只看总错数”），只是这里的“错误”要先经过一层会出错的judge才能看见。

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

