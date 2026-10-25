# 09：宪法AI与自我改进——规则奖励、组优势和GRPO边界

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/09-constitutional-ai-self-improvement) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/09-constitutional-ai-self-improvement/docs/en.md)。原题：Constitutional AI and Self-Improvement。

## 一、这一课解决什么问题？

区分按原则批评修订和按可验证奖励优化。原critic是手写规则，self_improvement_round仅汇总固定sampler，没有更新模型，不是完整CAI/GRPO训练。reward_math用eval执行输入有安全风险，不应对不可信prompt调用。

## 二、核心概念与推导

### 2.1 CAI两步
SL-CAI是初稿→按原则critique→revise→以修订回答SFT；RLAIF用AI产生偏好，可训练RM+RL，也可直接偏好优化，DPO不需先RM。原则不能把永不拒绝/永不犹豫当普遍准则，应含事实不确定性和安全。

### 2.2 GRPO
同prompt采G回答，Ai=(ri−组均值)/(组std+ε)。ratio=πθ/πold用于clip，KL另对πref；old与ref不是一个角色。省掉特定价值估计模型不表示无额外采样成本；仍需真实反传、response mask和optimizer。

### 2.3 ORM/PRM
结果奖励看最终，过程奖励看步骤；规则验证也有盲区/漏洞。数学/代码/证明判题需正确gold、沙箱和防作弊。不能将某私有模型使用PRM的猜测当事实，也没有所有任务ORM必优的定律。

### 2.4 循环不是自动学习
生成→评分→筛选→真训练→再采样才改策略。固定sampler重复汇总不会变强。自judge共同盲区、分布收窄与过拟合需要外部核验和独立集。std=0可能全答对/全错/奖励太粗，不必然mode collapse；还测文本多样、pass率、熵和题难度。

原文人类数据5%～10%、3～5轮退化、所有前沿系统用四方法、R1等泛化描述不作为已核实普遍事实。

### 2.9 补充算例与理解检查

[0,0,1,1]均值.5、总体std=.5，优势近[−1,−1,1,1]。全[1,1,1,1]优势0，可能已经全正确；只凭reward方差不能知道4条文本是否相同。

### 原课概念部分的公式与结构示例

$$
L_{\text{PPO}} = \mathbb{E}\left[\min\left(r(\theta) \cdot A, \mathop{\mathrm{clip}}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A\right)\right]
$$

$$
A_i = \frac{r_i - \mathop{\mathrm{mean}}(r_1, \ldots, r_G)}{\mathop{\mathrm{std}}(r_1, \ldots, r_G)}
$$

$$
L_{\text{GRPO}} = \mathbb{E}\left[\min\left(r(\theta) \cdot A_{\text{group}}, \mathop{\mathrm{clip}}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A_{\text{group}}\right)\right] - \beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
$$

## 三、操作前准备

原代码需re/NumPy，不运行含eval的reward_math。下方安全示例限制为单个整数运算，不支持任意表达式；生成程序验证需另加沙箱/超时，本次不执行任何生成代码。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 本课动手范围

原课先概述本课要实现的环节；以下逐段区分任务目标、可执行代码与未实现的训练功能。



### 4.2 原则

四原则只是示例，事实未知应允许说明不确定，安全边界仍保留。源文原则是学习数据，不是用户对真实系统权限的指令。

~~~python
CONSTITUTION = [
    "The response must directly answer the question asked, without hedging.",
    "The response must not include unnecessary filler or padding.",
    "If the question has a single numeric answer, state the number plainly.",
    "The response must not refuse a reasonable, benign request.",
]
~~~

### 4.3 批评与修订替身

按词数、逗号、拒绝前缀评分，不是模型自批评。revise在缺句点长文上可能IndexError，删句也会丢正确事实；修订后需要再次审核。

~~~python
def critique(response: str, principle: str) -> dict:
    problems = []
    if len(response.split()) > 40 and "plainly" in principle:
        problems.append("answer buried in extra prose")
    if response.strip().lower().startswith(("i can't", "i cannot", "as an ai")):
        problems.append("unwarranted refusal")
    if response.count(",") > 4:
        problems.append("too much hedging")
    return {"principle": principle, "problems": problems}

# 规则替身，缺句点可能越界，修改后要复核事实。
def revise(response: str, critique_result: dict) -> str:
    if "answer buried" in " ".join(critique_result["problems"]):
        return response.split(".")[-2].strip() + "."
    if "unwarranted refusal" in " ".join(critique_result["problems"]):
        return "Here is the answer: " + response.split(":")[-1].strip()
    return response
~~~

### 4.4 规则奖励的安全问题

eval可执行任意Python，禁止接不可信输入。最后整数规则会误读小数/多个数字，format search只说明有标签不保证唯一正确。用下方受限解析替代，不演示危险执行。

~~~python
import re

# 安全警告：原eval可执行输入，不运行不可信文本，使用补充受限判题。
def reward_math(prompt: str, response: str) -> float:
    try:
        expected = eval(prompt.replace("What is ", "").replace("?", "").strip())
    except Exception:
        return 0.0
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return 0.0
    return 1.0 if int(numbers[-1]) == expected else 0.0

def reward_format(response: str) -> float:
    return 1.0 if re.search(r"<answer>.*</answer>", response) else 0.0
~~~

### 4.5 组优势

总体std小返回0，无区分不是自动崩溃；空组应拒绝，G=1通常无有用相对信号。

~~~python
import numpy as np

def group_relative_advantage(rewards: list[float]) -> np.ndarray:
    r = np.array(rewards, dtype=float)
    if r.std() < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / (r.std() + 1e-8)
~~~

### 4.6 GRPO草图错用ref

原ratio使用ref_logprobs而非旧采样策略，kl=mean(ref−policy)也不保证目标KL；函数只返回指标，没有梯度或更新。不能称“symbolic gradient”完整实现。

~~~python
# 原版old/ref和KL含义有误，且不更新任何参数。
def grpo_step(policy_logprobs: np.ndarray, ref_logprobs: np.ndarray,
              advantages: np.ndarray, beta: float = 0.01, clip_eps: float = 0.2) -> dict:
    ratios = np.exp(policy_logprobs - ref_logprobs)
    unclipped = ratios * advantages
    clipped = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -np.minimum(unclipped, clipped).mean()
    kl = (ref_logprobs - policy_logprobs).mean()
    total_loss = policy_loss + beta * kl
    return {
        "policy_loss": float(policy_loss),
        "kl": float(kl),
        "total_loss": float(total_loss),
        "mean_ratio": float(ratios.mean()),
    }
~~~

### 4.7 采样汇总

self_improvement_round只生成/选best/算advantages，不调用优化器或grpo_step。空prompt/零组需要校验。reward变动也可能只是采样随机性，不能假定学会。

~~~python
# 只采样评分汇总，不是完整训练。
def self_improvement_round(prompts: list[str], policy_sampler, group_size: int = 8) -> dict:
    metrics = []
    for prompt in prompts:
        responses = [policy_sampler(prompt) for _ in range(group_size)]
        rewards = [reward_math(prompt, r) + 0.1 * reward_format(r) for r in responses]
        advantages = group_relative_advantage(rewards)
        best = responses[int(np.argmax(rewards))]
        metrics.append({
            "prompt": prompt,
            "mean_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "std_reward": float(np.std(rewards)),
            "best_response": best,
            "advantages": advantages.tolist(),
        })
    return {"per_prompt": metrics,
            "overall_mean": float(np.mean([m["mean_reward"] for m in metrics]))}
~~~

## 五、把代码运行起来：补充入口与检查

补充安全有限算术奖励，不使用eval：
~~~python
import re
import operator
import math

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
def safe_math_reward(prompt, response):
    if len(prompt)>200 or len(response)>1000:
        return 0.0
    match = re.fullmatch(r"What is (-?\d{1,8})\s*([+*-])\s*(-?\d{1,8})\?",prompt.strip())
    answer = re.fullmatch(r"<answer>\s*(-?\d{1,16})\s*</answer>",response.strip())
    if not match or not answer:
        return 0.0
    a,op,b = match.groups()
    return float(int(answer.group(1)) == OPS[op](int(a),int(b)))

assert safe_math_reward("What is 3+4?","<answer>7</answer>") == 1.0
assert safe_math_reward("What is 3+4?","<answer>8</answer>") == 0.0
assert safe_math_reward("What is __import__('os')?","<answer>7</answer>") == 0.0
print("受限算术判题通过")
~~~
只支持单个加减乘，拒绝括号/浮点/除法和任意程序。运行原汇总前先显式替换reward_math并确认prompt格式，不把格式奖励当真实证明。

## 六、Use It：使用现成工具

源码最多展示批评/修订与组统计，不会训练policy。真实CAI/GRPO需可靠可训练模型、old/ref分离、采样logprob、mask、优化器、数据筛选与held-out验证。只改原则文字不会自动改已经部署的权重。



## 七、失败模式、原文边界与工程注意

1. eval不安全，不能对不可信文本执行。
2. old/ref不可混，均值log比不是随手就能叫精确KL。
3. reward std=0不等于模式崩溃。
4. self-improvement函数没有训练，固定sampler不会自行改进。
5. 格式奖励可作弊，判题规则需独立测试。
6. 原文InstructGPT/Anthropic归属有混淆，模型训练泛称不当完整历史。
7. 真代码reward需沙箱，控制网络/文件/超时与权限。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
审计critique/revise、AI偏好、规则reward与实际训练各步骤；列安全判题范围、old/ref、响应mask、clip/KL/熵与optimizer。测pass率、奖励分布、文本多样与人工/外部gold。如果只有固定sampler汇总，明确标为统计模拟，不声称自我提升。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 用本地chat替代手写critic，人工盲评改好/不变/变坏，不需付费API。
2. 加事实原则，查首都/日期gold，统计纠错与新增错。
3. 20prompt各两回答，critic偏好后真正DPO，先修正08噪声loop，与真实GRPO对照。
4. 加−αentropy、α=.01做5轮试验，文本多样与奖励分别测，别用std独断崩溃。
5. 两步算术分中间/最终，定义可解析规则，PRM/ORM做10轮真实更新比较，禁止不可信eval和任意程序执行。

## 十、术语与完成标准

CAI：原则反馈与修订；RLAIF：AI反馈学习；GRPO：组相对策略优化；ORM/PRM：结果/过程奖励；rejection sampling FT：筛优再训练；mode collapse：输出分布收窄。

完成检查：

- 能手算组优势。
- 能指出ratio的old/ref错误。
- 能识别统计并未训练。
- 能用受限解析替代eval。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

只执行无eval的safe_math_reward：3+4正确1、错误0、程序形状输入0。原课eval奖励函数及自改进训练没有执行。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 7 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[08：DPO直接偏好优化——推导、响应对数概率与算法边界]] · 下一课：[[10：LLM评估——自定义测试、EM与F1、PPL和Elo]]
