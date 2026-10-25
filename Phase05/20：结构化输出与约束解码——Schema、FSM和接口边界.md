# 20：结构化输出与约束解码——Schema、FSM和接口边界

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/docs/en.md)。原题：Structured Outputs & Constrained Decoding。

## 一、这一课解决什么问题？

让模型“只输出JSON”只是提示，不是可靠接口。结构化输出要同时区分格式正确、字段符合schema、内容事实正确和操作获准。后两项不能只靠语法约束保证。

## 二、核心概念与推导

模型每步产生词表logits。约束器根据已生成前缀与目标语法，屏蔽非法后继token为−∞，再从合法集合采样。FSM适合有限状态约束，CFG更适合递归结构；真实词表token可能对应多字符，不能只检查单个字符。

三条机制：提示词请求；服务端原生结构化输出；本地约束解码。Instructor侧重schema解析/校验与重试，是否底层也使用原生约束取决于provider和模式，不能一律理解为同一种FSM。

Outlines、XGrammar、llguidance、vLLM是不同库/引擎/服务集成，版本会改变接口与后端。原文“Outlines总是FSM、不支持递归”“任何本地模型都100%成功”不是跨版本定律。

格式合法不等于语义正确。未知日期应允许null，而非迫使编造。字段顺序可能影响生成行为，但并非永远必须reasoning在前；可只设计简短证据/依据，不要求公开内部思维过程。

### 2.9 用一个小例子把概念串起来（补充）

情感schema只允许positive/negative/neutral，能拦住任意第四种字符串，却拦不住把负面评论判成positive。confidence:float还允许2.0，必须加范围约束；evidence_span:str允许杜撰引文，必须切片核对。这样你就能分清语法、类型、业务与事实四道检查。

## 三、操作前准备

本课mask_logits与补充FSM演示只用标准库；Pydantic校验需安装pydantic；Outlines/Instructor/厂商例子需对应库、模型授权或API密钥，本文不执行。
~~~bash
python -m pip install pydantic
~~~
原文Outlines的outlines.models.transformers/outlines.generate.json属于特定版本调用，需匹配该版本或按当前官方文档迁移。不要为了学schema先下载受限大模型或付费调用。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 屏蔽非法logits

原文是机制伪接口：model.next_token_logits、sample、fsm、tokenizer均需实现。没有最大步数/无合法token处理；is_accept到达接受态立即停，未处理可继续的接受态。返回decode(ids)还包含prompt，不是只含新输出。

~~~python
# 只保留合法token分数，调用前需验证编号与非空集合。
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


# 教学接口非完整实现，需补步数、EOS与错误处理。
def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
~~~

### 4.2 Outlines与Pydantic

Review定义情感枚举、浮点置信值和证据。float不自动限制0～1，证据也没有自动验证来自原文。原输出0.93是示意；“永远零错误”不覆盖截断、拒绝、schema不支持与实现故障。

~~~python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
~~~

### 4.3 Instructor校验与重试

Invoice的total_usd要求非负，line_items是字符串列表。格式通过仍可能金额/币种提取错误。默认重试次数与model名称需查安装版本和账户可用模型，不能把课程型号当可调用保证。

~~~python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
~~~

### 4.4 原生结构化输出

原文把responses.create与output_parsed混用，schema也未明确strict/additionalProperties。下面保留来源以便理解，再在下一节给出经官方文档核对的parse写法；仅供阅读，不会请求服务。

~~~python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
~~~

## 五、把代码运行起来：补充入口与检查

补充：按官方文档修正SDK使用关系，未执行的API示例：
~~~python
from pydantic import BaseModel, ConfigDict
from typing import Literal
from openai import OpenAI

class SentimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentiment: Literal["positive", "negative", "neutral"]

# 需要用户自行配置密钥、确认模型权限及费用后才能运行。
# client = OpenAI()
# response = client.responses.parse(
#     model="gpt-5",
#     input="Classify the sentiment: The food was cold.",
#     text_format=SentimentResult,
# )
# result = response.output_parsed
~~~
responses.parse+text_format用于类型化解析；若用responses.create+text.format应处理output_text并解析。严格schema的对象需additionalProperties:false，字段要求、schema子集、拒绝和不完整输出都要按接口处理。依据：[OpenAI官方Structured Outputs文档](https://developers.openai.com/api/docs/guides/structured-outputs)，核对日期2026-10-25。

补充标准库验证mask本身：
~~~python
masked = mask_logits([0.1, 2.0, -1.0], [0, 2])
assert masked[0] == 0.1 and masked[1] == float("-inf")
assert masked[2] == -1.0
~~~
这只验证屏蔽逻辑，不等于完整约束解码端到端成功。

### 补充：不下载模型，跑完整的字符级电话格式约束

这个版本把“模型”替换成固定的11个候选字符分数，只演示FSM与mask怎样衔接；不是LLM推理，也不代表大词表的工程性能。字符表是0～9和连字符，状态是已输出的字符数，位置3、7只能输出连字符，其余位置只能输出ASCII数字。12字符后进入终态。

~~~python
import math
import random
import re

def toy_phone(seed):
    rng = random.Random(seed)
    alphabet = "0123456789-"
    output = []
    for position in range(12):
        valid_ids = [10] if position in {3, 7} else list(range(10))
        # 这里用固定分数替代模型输出，机制仍是先mask再采样。
        logits = mask_logits([0.0] * len(alphabet), valid_ids)
        weights = [math.exp(score) for score in logits]
        token_id = rng.choices(range(len(alphabet)), weights=weights, k=1)[0]
        output.append(alphabet[token_id])
    return "".join(output)

samples = [toy_phone(seed) for seed in range(1000)]
assert all(re.fullmatch(r"[0-9]{3}-[0-9]{3}-[0-9]{4}", s) for s in samples)
print("合法样本数：", len(samples))
~~~

预期打印合法样本数1000。合法格式不代表电话号码存在、属于谁或已取得联系授权。真实tokenizer的一个token可跨多个字符，必须检验整个token展开后能否留在可接受路径，不能直接套用这里的字符下标。

## 六、Use It：使用现成工具

简单支持schema可用服务原生机制；需要跨provider/Pydantic可比较Instructor；本地推理结合具体Outlines版本或CFG引擎；自托管服务按当前vLLM接口配置。不要照抄已改名的guided_*参数。比较有效完成率、schema通过率、语义准确率、拒绝率和p50/p99延迟，而非只看解析成功。



## 七、失败模式、原文边界与工程注意

1. 空合法集合导致全−∞，softmax无效，要显式终止/回退。
2. max_tokens截断、拒绝、服务错误都应独立分支处理，不能计为合法业务结果。
3. 枚举过大应先检索候选，检索漏掉正确项也会强迫错选。
4. 更快并非约束解码必然效果：编译、掩码、缓存与后端会影响开销。
5. JSON mode与JSON Schema约束不同；schema通过仍需事实/业务校验。
6. 课程带注释的BAD/GOOD JSON是讲解，不是有效JSON文件。
7. 格式约束不提供工具调用授权，不绕过确认和权限。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
按provider、本地模型、延迟和schema复杂度选择机制；定义必填、枚举、null、额外字段和业务校验。规定拒绝、不完整、schema不支持、重试上限与回退，分别报告完成率、schema合规和语义准确率。字段顺序通过任务评估决定，不要求输出内部推理。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：用本地小模型在100条评论上仅提示输出Review，统计JSON可解析与schema合规，二者分别计数。
2. Medium：同数据使用匹配版本的Outlines，比较schema合规、语义准确、完成率与延迟，报告失败而非删除失败样本。
3. Hard：实现电话号码ASCII格式[0-9]{3}-[0-9]{3}-[0-9]{4}的FSM，1000次生成用re.fullmatch校验；若用\d说明其Unicode数字语义，并测试空合法集合和步数限制。

## 十、术语与完成标准

logits：采样前分数；mask：屏蔽非法候选；FSM：有限状态机；CFG：上下文无关文法；schema：结构契约；semantic validation：语义/业务验证。

完成检查：

- 能解释每步mask而非生成后删字。
- 能区分parse和create输出。
- 能列举格式正确但内容错误的例子。
- 能处理拒绝、截断与无合法后继。

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

