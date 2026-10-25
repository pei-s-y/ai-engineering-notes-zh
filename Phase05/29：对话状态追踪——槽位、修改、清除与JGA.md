# 29：对话状态追踪——槽位、修改、清除与JGA

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking/docs/en.md)。原题：Dialogue State Tracking。

## 一、这一课解决什么问题？

用户说“北边便宜的意大利餐厅”，下一轮改“价位中等”，系统应保留其他条件。本课维护结构化状态，区分未提及、明确修改、清除与不限制，避免把聊天误当执行授权。

## 二、核心概念与推导

domain是任务领域，slot是参数如cuisine/area/price，state是到当前轮的有效值。封闭槽位可分类，开放槽位可生成；ontology-free能自由生成槽名但仍需要后续业务映射。

JGA=所有槽都完全正确的轮数/总轮数，是严格整体指标。还应看slot级P/R、修改轮、否定轮和多领域轮表现。MultiWOZ标注版本影响成绩，原文80%～83%不是本项目验收指标。

更新政策需要至少三种事件：UNCHANGED未提不动；SET指定/修改；CLEAR删除原条件。any/dontcare表示用户不限制，不一定等于None未知。否定“不要意大利，改中餐”不能在更新后把新值又清空。

规则可解释但覆盖窄；TripPy/BERT-DST、LoRA语言模型、结构化提示是候选路线。Pydantic保证类型约束不等于槽位含义正确，几行代码不能自动变成生产系统。

### 2.9 用一个小例子把概念串起来（补充）

初始{}；说cheap italian in the north后填3槽；说actually moderate只改price；clear cuisine只清cuisine，area仍north。若用户说any cuisine，应记录不限而非不知道。即使槽已全齐，预订也应由独立确认/授权流程决定，不是DST自动执行。

## 三、操作前准备

规则与JGA先用标准库；LLM结构化例子需pydantic/instructor、模型适配器和render(history)，不是原样即可运行。people应增加正整数/合理上限校验，day需结合日期/时区，不把任意字符串等同有效预约日期。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 提取菜系

源码any(syn in utterance)是子串查找，含否定也照样命中；noodles不一定代表中餐，pizza也可能只是讨论而非选择。多个菜系同时出现时按字典顺序返回第一个。

~~~python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


# 子串规则只作基线，不能理解否定与真正选择意图。
def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
~~~

### 4.2 更新状态

先复制原字典，extractor返回非None才覆盖，所以未触及槽保留。SLOT_EXTRACTORS、NEGATION_CLEARS和is_negated未定义；最后clear可能误清用户刚给的新值，必须区分清除与替换的优先级。

~~~python
# 复制后更新，源码先填后清可能冲突，需明确事件优先级。
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
~~~

### 4.3 LLM结构化DST

RestaurantState定义合法结构，render(history)与llm是待实现接口；全历史重新生成仍可能忘记旧槽或猜测，不能据schema声称无幻觉。加入轮次来源、未确定状态与业务校验。

~~~python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
~~~

### 4.4 JGA

原代码zip默默截断长度不等列表，分母却用pred长度；空列表除零。要先断言两边非空且等长，再统一大小写、缺失值约定和槽集合，不能任意normalize掩盖实错。

~~~python
# 正式评价前校验非空、等长与统一状态schema。
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
~~~

### 4.5 处理修改

is_correction只检查提示词，不定位具体槽。“改到7点”不能机械覆盖最后更新的price。全历史与增量patch各有成本/错误模式，通过数据选择，不保证全历史自然解决一切。

~~~python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


# 只识别提示词，不确定具体被修改槽。
def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
~~~

## 五、把代码运行起来：补充入口与检查

补充最小可运行的规则配置，明确仅覆盖示例，不做真实预订：
~~~python
import re

def extract_area(text):
    return next((x for x in ("north", "south", "east", "west", "center")
                 if re.search(r"\b" + x + r"\b", text.lower())), None)

def extract_price(text):
    return next((x for x in ("cheap", "moderate", "expensive")
                 if re.search(r"\b" + x + r"\b", text.lower())), None)

SLOT_EXTRACTORS = {"cuisine": extract_cuisine, "area": extract_area, "price": extract_price}
NEGATION_CLEARS = ("cuisine", "area", "price")

def is_negated(text, slot):
    # 只演示明确的“clear <slot>”，不声称理解一般否定。
    return f"clear {slot}" in text.lower()

state = {"cuisine": None, "area": None, "price": None}
state = update_state(state, "cheap italian in the north")
assert state == {"cuisine": "italian", "area": "north", "price": "cheap"}
state = update_state(state, "actually moderate")
assert state["price"] == "moderate" and state["area"] == "north"
state = update_state(state, "clear cuisine")
assert state["cuisine"] is None and state["price"] == "moderate"
print(state)

def checked_jga(predictions, gold):
    if not predictions or len(predictions) != len(gold):
        raise ValueError("需要非空且等长的预测与gold")
    return joint_goal_accuracy(predictions, gold)
assert checked_jga([state], [dict(state)]) == 1.0
~~~
预期最后菜系为空、地区north、价位moderate。本次不调用LLM，不创建预约或执行付款。

## 六、Use It：使用现成工具

窄域可用规则基线；广域可标注训练或schema引导LLM；语音增加ASR与正规化。合规/高风险槽应确定性校验与用户确认，非必须把“规则永远主、LLM永远辅”当统一政策。槽值全齐不等于用户已经授权调用后端操作。



## 七、失败模式、原文边界与工程注意

1. “未提及”与“明确清空”若都用None表示，patch更新时无法区分，要用事件或field presence。
2. 状态保留、跨domain隔离、撤销与重开任务需要明确规则。
3. “好”“就那个”需关联上一轮系统候选，单看用户文本无法定位。
4. 全历史每轮重发在固定轮长假设下累计输入近二次增长；摘要可能丢关键槽，需对照测。
5. 否定不能简单清空所有同名槽，更不能覆盖合法的新值。
6. schema需版本；新增槽对旧数据/旧API的兼容要测试。
7. 原Hard练习按<2槽触发LLM仅是启发式：用户合法地只修改1槽不代表失败，改用明确不确定性与冲突条件评估。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
定义domain、槽、枚举/自由值、unknown/dontcare与SET/CLEAR/UNCHANGED更新语义。比较全历史生成和增量patch，记录每槽来源轮与撤销；评估JGA、slot P/R及修改/否定子集。后端执行前校验参数、权限与用户确认，版本化schema。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：3槽规则系统测试10段对话，逐轮人工gold，覆盖新增、保留、覆盖和清除，算JGA。
2. Medium：同一数据用Instructor/Pydantic+授权本地或外部模型，比较JGA并分析最难轮，不只看合法JSON率。
3. Hard：规则优先、明确不确定时LLM fallback；与原文<2槽规则做对照，汇报JGA、误路由、调用比例与每轮成本。所有预约/支付用mock，不实际执行。

## 十、术语与完成标准

DST：对话状态追踪；slot：槽位；JGA：整轮全部槽正确比例；ontology：槽与候选值体系；correction：修改；dontcare：不限；patch：增量更新。

完成检查：

- 能让未触及槽保持不变。
- 能区分CLEAR与UNCHANGED。
- 能解释JGA严格在哪里。
- 能指出状态正确不等于操作已获授权。

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

