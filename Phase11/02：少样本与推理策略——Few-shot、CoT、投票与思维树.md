# 02：少样本与推理策略——Few-shot、CoT、投票与思维树

> 对应课程：Phase 11 / 02-few-shot-cot。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/02-few-shot-cot/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/02-few-shot-cot/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、先把这一课放进整体流程

第 01 课解决“任务怎么说清楚”，这一课解决“复杂任务怎样组织示例、分步求解与检查”。这里通常不训练模型：模型参数不变，变化的是输入内容、调用次数和结果选择方法。

想象你请人解决一道应用题。你可以直接出题，也可以先给一道类似题的解法，还可以让几个人独立解答后比较结果。这对应零样本、少样本、自一致性。类比帮助理解流程，但模型投票并不等于多个独立专家作证。

## 二、概念从零讲起

### 2.1 Zero-shot、Few-shot 与上下文学习

**Zero-shot（零样本）**：提示中没有任务示范。例如“把下面评价分类为正面或负面：这东西很好用”。“零”指本次提示的示范数量，不是模型从来没训练过。

**Few-shot（少样本）**：在待处理输入前放几组输入—输出示例。例如：

```text
任务：判断评价情感，只输出正面、负面或中性。
示例一：很好用 → 正面
示例二：到货就坏了 → 负面
示例三：今天收到了 → 中性
现在判断：包装完整，但还没使用 →
```

模型通过上下文中的模式来完成当前任务，称为 **in-context learning（上下文学习）**。这不会执行反向传播，也不把这些例子永久写入权重。

示例选择要兼顾：与问题相似、标签覆盖充分、难度接近、答案确实正确。只展示正面评价，模型可能偏向正面。示例多会增加输入长度和费用；原课建议从 3～5 个试起，这是实验起点，不是最佳数量定律。测试题不能同时放进示例，否则是在泄漏答案。

### 2.2 CoT：中间步骤如何帮到答案

**Chain-of-Thought（思维链）提示**在研究中指示范或引导模型生成中间推理步骤。原课例子：Roger 有 5 个球，又买 2 罐，每罐 3 个。先算 `2 × 3 = 6`，再算 `5 + 6 = 11`。

**Zero-shot CoT** 不给解题示例，只要求分步解决；**Few-shot CoT** 给出带中间步骤的示例。原课另一例：每天 16 个蛋，吃 3 个、做糕点用 4 个，剩下 `16−3−4=9`，每个卖 2 元，收入 18 元。

新增 token 会成为后续生成的输入，可能帮助分解问题，但模型写出的解释不是内部计算的可靠记录。步骤多也可能只是错误累积。对于已有内置推理能力的模型，优先清楚说明目标、约束与验收要求，要求可核查的关键计算或简短说明；不要把“必须输出冗长思维链”当作通用优化。

### 2.3 Self-consistency：把多个结果统一后投票

重复采样 N 次，提取最终答案，按出现次数最多者选择结果。

例：`18、18、16、18、18`，18 获得 `4/5=80%` 的票。这个 80% 是**样本中的票数占比**，不是“正确概率为 80%”。同一个模型可能五次犯同一错误。

必须先统一答案格式：`18`、`18.0`、`$18` 应在题目允许时归一化成相同答案；解析失败要单独统计。假如五次只有两次解析成功且答案相同，成功样本内的一致率是 100%，但有效输出率只有 40%。

temperature 控制部分模型的采样随机性。较高值可以增加输出多样性，但不是越高越好；temperature=0 也不是跨平台、跨运行绝对确定的承诺，部分推理模型也不支持原课的 temperature 参数。

### 2.4 ToT：不满意的分支可以放弃

**Tree-of-Thought（思维树）**把候选解法组织成搜索树：提出几个方向→评价各方向→保留较好的→继续展开→检查最终结果。**节点**是一段候选状态，**边**是一次延伸，**剪枝**是放弃不值得继续的分支。

广度优先 BFS 先探索同一层；深度优先 DFS 先走到底再回溯。原课程序每层只扩展前两个候选，更接近保留有限候选的束搜索，不是完整枚举全部树。

若每节点分 3 枝，扩展 3 层，完整树有 `3+9+27=39` 个非根节点，但实际 API 次数还取决于“一次生成几个候选”和“每个候选是否单独评分”。不能把 39 直接当该程序的调用次数。

### 2.5 ReAct、结构化提示与提示链

**ReAct**：模型选择下一步动作→程序执行计算/搜索工具→把工具结果交回模型→继续判断。真实依据来自工具返回；模型“写了一段搜索描述”不等于已经搜索。

**结构化提示**：用 XML 标签、Markdown 标题或分隔符区分规则、资料、问题、输出格式。例如 `<资料>...</资料>`；分隔符使内容更清楚，但不构成安全隔离。

**Prompt chaining（提示链）**：第一个调用提取事实，第二个分析事实，第三个形成建议。每一步可单独检查。它和 CoT 不同：CoT 可以只在一次调用中完成，提示链显式包含多次调用。错误也会向后传递，所以中间结果要验证。

| 方法 | 主要用途 | 需要付出的成本 |
|---|---|---|
| Few-shot | 示范格式、标签、术语 | 更多输入 token |
| CoT 提示 | 多步骤任务的可检查解答 | 可能增加输出和推理开销 |
| 自一致性 | 比较多次候选答案 | 多次调用、答案解析 |
| ToT | 可评价的规划/搜索问题 | 分支生成、评分、剪枝 |
| ReAct | 需要外部事实或计算 | 工具接口和循环控制 |
| 提示链 | 有清晰子任务的大任务 | 中间结果契约和失败处理 |

## 三、原文 Build It：阅读顺序与必要代码

原课代码有真实 API 调用；本笔记不会把接口输出冒充已验证结果。先读提示拼接和投票，再看搜索树。以下各块是原文的核心片段，**不是每块都能单独运行**：`extract_answer`、候选生成/评分函数等完整定义在仓库 `code/advanced_prompting.py`。不要把带 `...` 的示例列表直接当正式数据集。

### 步骤 1：示例库

每条示例有 question、reasoning、answer 三个字段。原文用省略号表示还有其他样本；这里展示的是结构，不能将 Ellipsis 当字典使用。示例答案需先人工校验。

````python
# 本块中文说明：示例库
GSM8K_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every egg at the farmers' market for $2. How much does she make every day at the farmers' market?",
        "reasoning": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4, using 3 + 4 = 7 eggs. So she has 16 - 7 = 9 eggs left. She sells each for $2, so she makes 9 * 2 = $18 per day.",
        "answer": "18"
    },
    ...
]
````

### 步骤 2：构建少样本提示

system 写总体规则；examples[:num_examples] 取前几个示例，不是按语义自动选最相似的；逐条拼接问答后追加当前 question。返回两个字符串，尚未调用模型。

````python
# 本块中文说明：构建少样本提示
# 将示范题及可核查解答填入提示，再追加当前题目。
def build_cot_prompt(question, examples, num_examples=3):
    system = (
        "You are a math problem solver. "
        "For each problem, show your step-by-step reasoning, "
        "then give the final numerical answer on the last line "
        "in the format: 'The answer is [number]'."
    )

    example_text = ""
    for ex in examples[:num_examples]:
        example_text += f"Q: {ex['question']}\n"
        example_text += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"

    user = f"{example_text}Q: {question}\nA:"
    return system, user
````

### 步骤 3：多次采样并投票

循环调用模型，提取最终答案，再用 Counter 统计。reasonings 保存输出方便复查；confidence 实为成功解析答案中的多数占比。依赖 Counter 与 extract_answer，空样本/答案0处理见正文。

````python
# 本块中文说明：多次采样并投票
# 采样多个答案并投票；票数占比不是真实正确概率。
def self_consistency_solve(question, examples, client, model, n_samples=5):
    system, user = build_cot_prompt(question, examples)

    answers = []
    reasonings = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7
        )
        text = response.choices[0].message.content
        reasonings.append(text)
        answer = extract_answer(text)
        if answer is not None:
            answers.append(answer)

    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0] if vote_counts else None
    confidence = vote_counts[best_answer] / len(answers) if best_answer else 0

    return best_answer, confidence, reasonings, vote_counts
````

### 步骤 4：展开候选树

先生成候选并评分，再仅扩展前两个。scored 元素是(候选,分数)；sort 的 reverse=True 表示高分在前。generate/evaluate/extend 由完整脚本定义，评估本身也可能调用模型并花费token。

````python
# 本块中文说明：展开候选树
# 扩展候选、评分、保留较优分支，涉及多次模型调用。
def tree_of_thought_solve(question, client, model, breadth=3, depth=3):
    thoughts = generate_initial_thoughts(question, client, model, breadth)
    scored = [(t, evaluate_thought(t, question, client, model)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)

    for current_depth in range(1, depth):
        next_thoughts = []
        for thought, score in scored[:2]:
            extensions = extend_thought(thought, question, client, model, breadth)
            for ext in extensions:
                ext_score = evaluate_thought(ext, question, client, model)
                next_thoughts.append((ext, ext_score))
        scored = sorted(next_thoughts, key=lambda x: x[1], reverse=True)

    best_thought = scored[0][0] if scored else ""
    return extract_answer(best_thought), best_thought
````

### 步骤 5：组合升级流程

先单次求解，再五次投票；只有单次与多数一致且票比达到0.8才接受，否则调用树搜索。注意它不是先判断失败才启动投票，也不能证明多数答案正确。

````python
# 本块中文说明：组合升级流程
def solve_with_escalation(question, examples, client, model):
    single_answer, _ = few_shot_cot_solve(question, examples, client, model)

    sc_answer, confidence, _, _ = self_consistency_solve(
        question, examples, client, model, n_samples=5
    )

    if confidence >= 0.8 and single_answer == sc_answer:
        return sc_answer, "self_consistency", confidence

    tot_answer, _ = tree_of_thought_solve(question, client, model)
    return tot_answer, "tree_of_thought", None
````


## 四、怎样运行，以及会得到什么

1. 先用纸笔核对两个算术示例，并查看 `build_cot_prompt` 返回的 `(system, user)` 两个字符串。
2. 不连模型时可以先执行下面的投票例子，理解 Counter；这是补充的离线演示。

```python
from collections import Counter

answers = [18, 18, 16, 18, 18]  # 已经解析和归一化后的答案
counts = Counter(answers)       # {18: 4, 16: 1}
winner, votes = counts.most_common(1)[0]
agreement = votes / len(answers)
print(winner, agreement)        # 预期：18 0.8
```

3. 运行仓库完整脚本需要 `openai` 包、账户可访问的模型和环境变量 `OPENAI_API_KEY`。安装与运行位置见本章导航；具体命令为 `python code/advanced_prompting.py`（在本课目录内）。先检查文件末尾的测试题数量和默认模型，以免一次执行多轮比较。
4. 成功调用后会看到各策略答案、与参考答案的匹配和比较报告。答案文字、票数、耗时及成本不固定；本次未执行付费调用。

**代码核对：**原文 `if best_answer` 在答案为数值 0 时可能错误判断，应写 `best_answer is not None`。升级流水线不是“只在必要时才投票”：它先运行单次求解，又总是运行五次投票，然后才决定是否进入 ToT。因此最低已包含多次调用。原文的准确率表、78%/91%/95%等数字没有一致的可复现实验设置，且段落有自相矛盾，不能当自己的预期得分。

## 五、Use It 与交付物如何理解

LangChain 的 PromptTemplate 负责填变量，FewShotPromptTemplate 负责拼示例，ExampleSelector 负责选示例；这些功能仍然需要模型与数据。DSPy 用声明式模块表达任务，并可在给定训练/验证样本和指标后优化提示；创建 `ChainOfThought` 模块本身不等于已经完成优化。`dspy.majority` 对候选答案投票。框架不会自动保证推理正确。

原文交付物是“推理提示模板”和“策略选择指南”。可以直接在笔记中保存这样的模板：**任务→示例→当前问题→关键可核查步骤→最终答案格式→信息不足怎么办**。选择时记录质量要求、最多调用次数、时间预算和判分方法，不需要另外建输出文件夹。

## 六、原文练习与验收

1. 取 10 道未放进示例的 GSM8K 题，比较零样本、少样本、零样本 CoT、少样本 CoT。记录题目、最终答案、正确性、token 数；10 题仅用于熟悉实验，不能得出稳健的模型排名。
2. 固定题目与示例数量，比较随机示例和相似示例。保证没有把测试题答案放入示例，分析究竟是格式还是题型匹配带来的变化。
3. 20 题分别采样 1、3、5、7、10 次，画准确率—总费用曲线。不要只统计成功解析的样本而漏掉失败样本。
4. 加计算器工具，将明确运算符和数值参数交给白名单函数执行；不直接对模型文本使用 `eval()`。比较工具算术和纯文本求解。
5. 用 ToT 生成“同时有趣且悲伤的六词故事”。先确定按空格计词的规则，再比较单次生成、分支搜索和人工偏好，不能把模型自评分当唯一答案。

## 七、学到什么程度算理解

能解释：少样本为什么不是微调；五个答案都一样为什么仍可能全错；ToT 比单次生成多了哪些程序步骤；ReAct 中到底是谁执行工具。能自己写一个示例拼接函数和 Counter 投票函数，就掌握了本课的核心程序结构。


## 本次验证记录（2026-11-01）

本次仅执行提示构造、答案提取和离线投票小例子，投票输出18与0.8；没有执行模型API或思维树求解。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<01：提示词工程——技巧与模式.md>) · [下一课](<03：结构化输出——JSON、Schema、验证与约束解码.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
