# 20：结构化输出与约束解码——Schema、FSM和接口边界

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对，四步结构与原课一致。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/docs/en.md)。原题：Structured Outputs & Constrained Decoding。

## 零、初学者先读：把“拜托输出JSON”换成“只能输出JSON”

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。前置衔接：[第17课](<17：聊天机器人——从规则、检索到神经模型与工具循环.md>)的聊天机器人已经见过“模型输出要接进下游程序”的接口问题，[第19课](<19：子词分词——BPE、WordPiece、Unigram与SentencePiece.md>)的token是本课“为什么约束要落在token上”的基础。

### 0.1 先理解这一课在做什么

对一个自由生成的模型说“请只输出JSON”，它大多数时候会照做——但“大多数时候”在产品里就是事故率。原课的例子：要它返回 {positive, negative, neutral} 三选一，它回你一段“The sentiment is positive — this review is overwhelmingly favorable because...”，下游解析器直接崩，F1归零。**提示词是请求，不是合同**；本课的核心动作是把请求换成合同：约束解码（constrained decoding）。类比一条单行道：不是等司机开错路口再吊销驾照（生成完再检查、重试），而是提前把所有逆行的岔路口用栏杆焊死（每个生成步骤之前，把非法token的分数设为−∞）——司机再怎么想违规，物理上开不出去。

“每一步”是关键：此刻合法的下一个token取决于已经生成了什么（前缀），所以栏杆要随位置移动。三层实现的强度递增（正文第二节展开）：

| 层 | 做法 | 可靠性 |
|---|---|---|
| 提示词 | “请只输出JSON” | 看模型心情 |
| 服务端原生结构化输出 | API按schema约束 | 支持的schema上可靠 |
| 本地约束解码 | mask logits | 构造上100%格式合法 |

但“合法”只保证格式——内容对不对、字段是不是真的，是另一回事（0.3、2.9和第七节会反复敲这条边界）。

### 0.2 本课术语先翻成白话

- logit=模型对词表里每个token打的原始分数，softmax之前的数
- mask（屏蔽）=把非法token的logit改成−∞，softmax之后概率精确为0
- FSM（finite-state machine，有限状态机）=把“格式长什么样”编译成一张“在哪个状态、哪些后继合法”的表
- CFG（context-free grammar，上下文无关文法）=能表达递归结构（嵌套数组、嵌套对象）的语法，比FSM表达力强
- 接受态 accept state=到达即“输出完成”的FSM状态（本课is_accept检查的就是它）
- schema（结构契约）=有哪些字段、什么类型、什么取值范围
- JSON mode=只保证“是合法JSON”，不保证符合你的schema——和JSON Schema约束是两回事
- 枚举 enum=只允许从固定清单里选，如positive/negative/neutral
- 语义校验 semantic validation=格式之外检查内容对不对（金额真伪、证据出处）
- Pydantic=Python里用类定义schema并自动校验的库

### 0.3 用小数字走一遍

任务：让模型输出只能是字符串`yes`或`no`（正文第九节Hard练习的电话号码格式[0-9]{3}-[0-9]{3}-[0-9]{4}的极简版）。设词表只有5个token：y、e、s、n、o。FSM把“合法”画成一张状态图：

~~~text
       y            e            s
  S0 ─────▶ S1 ─────▶ S2 ─────▶ S3 ✓接受（"yes"）
   │
   │ n          o
   └─────▶ S4 ─────▶ S5 ✓接受（"no"）

  图上没有的转移一律不存在：想走？logit被置−∞
~~~

第1步在S0（什么都还没输出），合法后继只有{y, n}。假设模型原始logits：e=3.0、o=2.5、y=2.0、n=1.5、s=0.5——**模型自己最想输出的是e**，但e不在{y,n}里，被mask成−∞，softmax后概率精确为0。剩下两个合法候选按softmax分概率：

~~~text
P(y) = e^2.0/(e^2.0+e^1.5) = 7.39/(7.39+4.48) ≈ 0.62
P(n) = e^1.5/(e^2.0+e^1.5) = 4.48/11.87        ≈ 0.38
~~~

一条被拦下的路径就这样死亡：e的logit是全场最高的3.0，第一步就被栏杆拦死，"e…"整个分支不存在。再看第2步：假设第1步采样选了y、到达S1，合法后继只剩{e}；此时模型其实更想输出o（logit 2.5，比如想拼"yo"）——但S1只有去S2的栏杆，"yo…"路径同样死亡。走通的路只有y→e→s到S3、或n→o到S5，任何别的拼写都到不了接受态。

两个工程事实对照：（1）第五节的toy_phone把这个机制做成可运行的完整版（11字符电话号码、1000次采样全部通过re.fullmatch）——它是本例的工程放大版；（2）真实词表10万+token，而且一个token可能对应多个字符（[第19课](<19：子词分词——BPE、WordPiece、Unigram与SentencePiece.md>)），所以“合法”要检查整个token展开后是否仍留在可接受路径上，不能像本例按单字符查表——这就是正文说的“不能只检查单个字符”。

以上是用于理解的算例，不是新增的训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

4.1 mask_logits是本例“置−∞”的两行（第五节有标准库验证：masked[1]==float("-inf")）；generate_constrained是外层循环“问FSM要合法集→mask→采样→状态转移”。4.2 Outlines把Pydantic的Review类编译成FSM。4.3 Instructor是另一条路线：不碰logits，解析失败就重试。4.4 原生API是服务端替你做mask。第五节的toy_phone把“模型”换成固定分数，不下载任何模型就能跑通“FSM+mask+采样”全流程——本课0.3的迷你版和它一一对应。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么约束解码是“每步mask”而不是“生成完再删掉非法部分”？

**参考答案：** 合法性取决于前缀（当前位置在哪），所以必须每步问一次；更根本的是mask发生在采样之前——非法token概率精确为0，构造上不可能被采出来。事后删字做不到：已经生成的内容删不干净，重试又引入延迟和成本。

**问题2：** 格式100%合法，就等于结果可信吗？

**参考答案：** 不等于。sentiment枚举拦得住第四种字符串，拦不住把差评判成positive；confidence:float允许2.0这种越界值；evidence_span可以是杜撰的引文——语法、类型、业务、事实是四道不同的检查（见2.9）。

**问题3：** Instructor和Outlines机制上差在哪？

**参考答案：** Outlines把schema编译成FSM、在采样前mask logits，非法输出构造上不可能；Instructor把schema写进提示、解析输出、失败重试，不改logits——跨provider可移植，但付出重试的延迟和费用。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

让模型“只输出JSON”只是提示，不是可靠接口。结构化输出要同时区分格式正确、字段符合schema、内容事实正确和操作获准。后两项不能只靠语法约束保证。

## 二、核心概念与推导

模型每步产生词表logits。约束器根据已生成前缀与目标语法，屏蔽非法后继token为−∞，再从合法集合采样。FSM适合有限状态约束，CFG更适合递归结构；真实词表token可能对应多字符，不能只检查单个字符。

（补充 2026-11-06）−∞为什么有这么大威力值得拆开看：softmax的分母是所有e^logit之和，而e^−∞=0——被mask的token在分子分母里都贡献0，概率精确为0，剩下的合法token重新分掉全部概率（0.3的P(y)=0.62、P(n)=0.38就是重新分配后的结果）。这也是“约束解码常常反而更快”的机制根源之一（原课的反直觉结论）：合法集小，采样空间就小；更极端的是强制token（如JSON开头被`{`锁死的部分）可以整段跳过采样。CFG与FSM的分工给一个直观判据：你的输出会不会“自己套自己”（数组的元素还是数组、对象的字段还是对象）？会，就是递归结构——FSM要拍平到固定深度才能处理（Outlines的做法），CFG引擎（XGrammar、llguidance）原生支持。token多字符的坑用一个例子记牢：若词表里有整词token"yes"，第1步的合法集就不是{y}而是{"yes"}（也许还有"y"）——合法集长什么样取决于具体词表把文本切成什么形状，这正是第五节末尾“必须检验整个token展开后能否留在可接受路径上”的意思。

三条机制：提示词请求；服务端原生结构化输出；本地约束解码。Instructor侧重schema解析/校验与重试，是否底层也使用原生约束取决于provider和模式，不能一律理解为同一种FSM。

（补充 2026-11-06）三条机制的取舍展开成一张决策表（第六节Use It的浓缩版）：

| 场景 | 推荐 | 一句话理由 |
|---|---|---|
| 用厂商API、schema简单 | 原生结构化输出 | 服务端mask，可靠性等同本地 |
| 要跨provider、能容忍重试 | Instructor | 不碰logits，靠重试换可移植性 |
| 本地模型、schema扁平 | Outlines（FSM） | 构造上100%合法 |
| 本地模型、递归schema | XGrammar/llguidance（CFG） | FSM拍平递归有深度上限 |
| 自托管推理服务 | vLLM guided_* | 约束集成进推理服务 |

注意Instructor那一行的成本结构：每次重试都是一次完整的API往返，延迟和费用随失败率线性上涨；这与mask路线“一次成型”是两种完全不同的账单——选型时算的是这笔账，不是“哪个更先进”。

Outlines、XGrammar、llguidance、vLLM是不同库/引擎/服务集成，版本会改变接口与后端。原文“Outlines总是FSM、不支持递归”“任何本地模型都100%成功”不是跨版本定律。

格式合法不等于语义正确。未知日期应允许null，而非迫使编造。字段顺序可能影响生成行为，但并非永远必须reasoning在前；可只设计简短证据/依据，不要求公开内部思维过程。

（补充 2026-11-06）“未知日期应允许null”是本课最容易被低估的工程经验，机制值得讲透：如果用正则把date字段锁死成YYYY-MM-DD，模型在“真不知道”时没有合法出口，只能在合法集里挑一个**编造的日期**——约束越硬，编造越隐蔽（格式完美、内容虚构）。出路是在schema层面给“不知道”留位置：date: string|null，或一个约定的sentinel值。字段顺序那条同理：JSON字段顺序会影响生成行为（先写的字段先被commit），所以原课建议reasoning在前、answer在后；但正文边界说得对——这不是铁律，本质是“先生成依据、后生成结论”通常更稳，你也可以只设计简短的evidence字段而不强求模型公开内部推理。最后把四道检查按位置排清楚：

| 检查 | 查什么 | 谁负责 |
|---|---|---|
| 语法 | 是不是合法JSON | JSON mode |
| 类型/结构 | 字段、枚举、数值范围 | Schema约束（本课主角） |
| 事实 | 内容是否为真 | 检索核对、NLI（[第21课](<21：自然语言推断——蕴含、矛盾、中立与忠实性检查.md>)）、人工 |
| 权限 | 操作是否获准 | 业务规则、确认流程（[第17课](<17：聊天机器人——从规则、检索到神经模型与工具循环.md>)的工具循环） |

本课的全部工具只覆盖第二行；第三、四行是别的课的领地——这也是第一节“后两项不能只靠语法约束保证”的展开。

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

