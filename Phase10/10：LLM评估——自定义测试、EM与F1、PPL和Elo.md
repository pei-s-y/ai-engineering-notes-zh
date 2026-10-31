# 10：LLM评估——自定义测试、EM与F1、PPL和Elo

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/10-evaluation) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/10-evaluation/docs/en.md)。原题：Evaluation: Benchmarks, Evals, LM Harness。

## 零、初学者先读：怎样判断模型真的变好，而非看起来不错

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

评估先规定题目、标准和计分方法，再比较模型。考试改成了什么分数才有意义：精确抄对、包含答案词、自然语言质量、推理延迟，分别需要不同指标。

### 0.2 本课术语先翻成白话

- benchmark=固定评测题集合

- metric=计分规则

- EM=按约定规范化后完全相同

- precision=预测中正确部分占比

- recall=应有内容被找回多少

- F1=精确率与召回率的调和平均

- PPL=由真实token概率计算的困惑度。

### 0.3 用小数字走一遍

预测a a b、标准a b b，若把词当集合，两边都是{a,b}，会得满分；若计重复次数，正确重叠为2个、两边各3个，F1=2/3。不是任意叫F1的函数都遵循相同计数协议。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先看EvalCase是一道题、EvalSuite是一组题加评分器；run接收一个回答函数，并不要求必须接在线模型。再看exact_match、token_f1和补充multiset_f1。PPL需真实正确token的条件概率；随机生成分数只能验证统计程序，不能证明语言能力。

前置阅读：[第04课](<04：从零搭建Mini GPT——因果注意力、前向计算与真实训练.md>)；[第06课](<06：指令微调SFT——聊天格式、响应Mask与真实梯度.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 模型在训练题上满分能证明泛化吗？

**参考答案：** 不能，需要未用于调参的题。

**问题2：** 模拟judge是另一个大模型吗？

**参考答案：** 本课该函数只是词重合规则，应按实现判断。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

模型好不好要落到你的任务、数据分布和错误成本。本课搭一个小型eval harness，理解指标会如何改变排名。原文的judge与模型概率都是模拟，不是真LLM评分；“所有实验室都刷榜”“只有你的eval有意义”也不应当无证据定论。

## 二、核心概念与推导

### 2.1 三层证据
公开benchmark给可比基线；领域自定义eval测实际任务；人工盲评校准自动指标。MMLU/HumanEval/SWE-bench/MATH/ARC/HellaSwag覆盖不同能力，版本/设置/污染会影响成绩。不能仅凭分数差小就宣布统计噪声，需配对检验/置信区间；人工也有分歧与标注误差。

### 2.2 指标定义
EM按协议完全匹配，适合封闭输出，不识别同义/单位换算。token F1通常可按多重集合计重复，本源码却先set，实际是“去重词集合F1”。BLEU/ROUGE是表面重叠不是事实证明。judge需明确rubric、人工校准、位置/长度偏置及失败率。

### 2.3 PPL
PPL=exp(平均负log真实下一token概率)，自然对数对应exp。同测试文本还不够，tokenizer/单位/上下文和mask必须匹配；不能拿不同分词PPL直接比。概率由手工model_quality构造不是测模型能力。

### 2.4 Elo
Ea=1/(1+10^((Rb−Ra)/400))，更新Ra+=K(Sa−Ea)。只表相对比较网络，受对局安排/顺序/样本/平局定义影响。原toy不是现代Arena全部统计实现，也不是最新排行榜。等级分非概率、非绝对能力。

### 2.5 自定义流程
定义任务成功→收集代表/困难/边界案例→gold与评分→冻结版本→自动回归→人工复核。50/200样本是来源建议，关键按分层覆盖、风险和不确定性预算。按case定位退化，不只报总分。

补充依据：[OpenAI官方评估最佳实践](https://developers.openai.com/api/docs/guides/evaluation-best-practices)强调任务相关与人工校准；这里不使用其付费接口。

### 2.9 补充算例与理解检查

预测a a b与参考a b b，本源码set后都变{a,b}，F1=1；若保留频次，重合2、两边3词，F1=2/3。两方1500分，A赢一局K=32后A1516/B1484。100°C与212°F在适当压力条件下可等价，EM判错不代表事实错。

### 原课概念部分的公式与结构示例

$$
\text{PPL} = \exp\left(-\frac{1}{N} \sum_i \log P(\text{token}_i \mid \text{context})\right)
$$

## 三、操作前准备

EvalCase/字符串评分/Elo只需标准库；PPL随机模拟与聚合依赖NumPy。本次不安装或运行付费judge，promptfoo YAML里的provider可能触发外部费用，不是本地安全模拟。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 评估接口

case保存输入/gold/metadata，suite对每case调用model_fn和scorer。原run没有异常、timeout、id、seed或保存失败，生产要补，不能让出错case从分母消失。

~~~python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
~~~

### 4.2 三种打分

EM忽略首尾空白和大小写；token_f1使用set，不含频次/标点正规化；llm_judge_simulated是overlap+.3长度项，完全没有LLM，不检测事实/否定。空-空F1在本源码为0，协议可不同但须注明。

~~~python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

# 本源码先set去重，非保留频次的常见QA F1。
def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# 这是词重合/长度启发式，没有调用LLM。
def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
~~~

### 4.3 Elo更新

预期得分与实际胜负/平局更新，ratings总和对每局保持；未知outcome原版默认tie，应显式校验以防拼写错误。它不是完整Arena Bradley-Terry/区间估计管线。

~~~python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
~~~

### 4.4 PPL与模拟概率

perplexity接log概率列表，不是logits。if not log_probs对NumPy数组可能真值歧义。hash(text)受进程随机化，不是跨进程稳定seed，且np.random.seed改全局RNG。

~~~python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

# 手工质量参数与随机噪声，不是模型真实token概率。
def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
~~~

### 4.5 汇总

mean/median/std/min/max/passrate按指标统计，需统一missing/NaN处理。print_summary却把门槛文字硬写.8，若调用threshold变动会显示错。多指标不能不校准就同权平均。

~~~python
# 真实评估需显式处理NaN、失败与分母。
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")
~~~

### 4.6 运行示例模型

good/bad是查字典，bad中改写和华氏表达不全错误；标签只是作者命名。EM偏向gold逐字输出，说明指标协议，不证明“简短模型普遍更好”。

~~~python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
~~~

### 4.7 模拟锦标赛

胜负由同一token_f1决定，因此Elo只是该评分的另一个汇总，不是独立人类偏好证据；要随机化顺序/标注条件。

~~~python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
~~~

### 4.8 PPL对照

手工调model_quality自然会改变采样概率，不是加载了strong/medium/weak实际模型。保持相同词序才有这个模拟意义，不报告为LLM测试。

~~~python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
~~~

## 五、把代码运行起来：补充入口与检查

标准库入口，先定义相关函数和ELOTracker：
~~~python
assert exact_match(" Paris ", "paris") == 1.0
assert token_f1("a a b", "a b b") == 1.0  # 展示set去重的边界
tracker = ELOTracker(k=32)
tracker.record_match("A", "B", "a")
assert tracker.ratings == {"A":1516.0, "B":1484.0}
print(tracker.leaderboard())

# 补充保留重复token的F1，与原set版本作区分。
from collections import Counter
def multiset_f1(prediction, expected):
    a,b = prediction.lower().split(), expected.lower().split()
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    return 2 * overlap / (len(a)+len(b))
assert abs(multiset_f1("a a b","a b b") - 2/3) < 1e-12
~~~
补充版空-空为1，与原协议不同已显式说明，真实任务先固定协议再比较。

## 六、Use It：使用现成工具

lm-evaluation-harness按当前版本选择checkpoint/任务/模板/batch，原命令会下载模型；promptfoo配置可调用外部服务，需权限/密钥/预算；RAGAS部分指标需要reference/LLM并非都免费无gold。来源中的包名/API示意可能变化，先看安装版本与--help，不直接大规模跑。

~~~python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
~~~

~~~yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
~~~

~~~python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
~~~

## 七、失败模式、原文边界与工程注意

1. simulated judge不是语义裁判，不能称80%人工一致。
2. PPL跨tokenizer/数据/上下文不可直接比，模拟分布不是实测。
3. 事实等价、单位、合法多答案要定义gold，不只字符串。
4. 失败/NaN/空集要显式记录，不能静默剔除抬高均值。
5. 少量Elo对局与固定F1胜负不支持普遍能力排名。
6. 原文历史模型得分/价格/平台地位不当当前事实。
7. 污染检测只能查已知训练文本，未查到不等于从未看过。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
定义任务与高风险错、版本化case_id/gold/scorer/prompt/checkpoint；组合结构/语义/人工指标，报告失败分母、分层成绩、配对退化与不确定性。为judge做人工校准，为PPL固定token协议；任何模拟器标模拟，不把排行榜替实际验收。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 同输入5次测一致性，分原字符串与语义一致；相同错误也可一致，不能当正确率。
2. 多judge权重对Elo排名影响，说明同一证据重用、尺度和tie协议。
3. 邮件5分类100条，含歧义/空/多语言，合成或授权脱敏数据，报告各类F1与混淆。
4. 用精确/近重复匹配查eval与可见训练语料重合，说明检测范围和误报，不声称证明完全无污染。
5. 按稳定case_id列新旧改善/退化/不变，保留失败与gold版本，避免只看均值。

## 十、术语与完成标准

eval suite：版本化测试集与评分；EM：完全匹配；F1：重合调和指标；PPL：真实token负log概率指数；Elo：成对等级分；contamination：训练测试污染；rubric：评分规则。

完成检查：

- 能区分set与多重集合F1。
- 能手算一局Elo。
- 能识别模拟judge/PPL。
- 能按case解释版本退化。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行EM、set F1、多重集F1和Elo：重复词例分别1与2/3，A获胜后1516/B1484。未运行模拟PPL、外部评测集、API judge或真实模型比较。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 8 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[09：宪法AI与自我改进——规则奖励、组优势和GRPO边界]] · 下一课：[[11：模型量化——位宽、缩放、PTQ与QAT及GPTQ-AWQ边界]]
