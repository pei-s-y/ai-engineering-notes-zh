# 29：对话状态追踪——槽位、修改、清除与JGA

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对概念与示例。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking/docs/en.md)。原题：Dialogue State Tracking。

## 零、初学者先读：把用户东一句西一句的话，攒成一张能下单的表

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课是“评估三部曲”的收官：[第27课](<27：LLM评估——RAGAS、DeepEval、G-Eval与人工校准.md>)评单轮答案质量，[第28课](<28：长上下文评估——NIAH、RULER、LongBench与有效长度.md>)评长文档能力，本课评**多轮对话**——系统是否始终记得“用户到底要什么”。与[第17课](<17：聊天机器人——从规则、检索到神经模型与工具循环.md>)的工具循环衔接：那里系统决定“做什么”，本课管“用户条件攒齐了没”。

### 0.1 先理解这一课在做什么

用户说“北边便宜的意大利餐厅”，下一句“价位改中等”，再一句“菜系算了不限了”。三句话零零碎碎，但后端API要的是一张完整的参数表：{cuisine: any, area: north, price: moderate}。对话状态追踪（DST）就是**维护这张表的人**：每听一句、更新一次——没提到的槽原样保留，改了的覆盖，明确放弃的清空。表错一个槽，订错餐厅、付错卡——DST是“用户嘴上说的”和“后端实际执行”之间的铰链。

类比：DST像跟在你身后记菜的服务员——你说“不吃辣”他记下，你说“算了要微辣”他划掉重写，你说“再也不要辣了”他擦掉整行；**唯独你随口聊天气，他一个字都不动**。三个动作对应本课的三种事件：SET（填/改）、CLEAR（清）、UNCHANGED（不动）。最难的不是记，是判断这句话属于哪种动作——“改成中等”是SET price，不是CLEAR cuisine，更不是把整张表重置。

与[第17课](<17：聊天机器人——从规则、检索到神经模型与工具循环.md>)的分工：那课的机器人决定“该检索还是该调工具”（行为层），本课决定“调工具时参数填什么”（状态层）。状态对了不等于可以下单——**授权**（用户确认）是另一道闸（正文原话：槽值全齐不等于用户已经授权调用后端操作）。

### 0.2 本课术语先翻成白话

- DST（对话状态追踪）=跨轮维护“槽-值”表
- 领域（domain）=任务区（餐厅/酒店/打车），决定有哪些槽
- 槽（slot）=后端要的参数：cuisine、area、price、people、day
- 状态（state）=到当前轮为止所有槽的有效值——不是最后一句话的翻译，是**整段对话的累积**
- 封闭槽/开放槽=值只能从枚举里选（price: cheap/moderate/expensive）/自由文本（名字、地址）
- SET／CLEAR／UNCHANGED=填或改／清空／没提就不动
- dontcare（any）=用户明确说“随便、不限”——是一种**表态**，不是“不知道”
- None（未知）=用户从没提过——与dontcare是两码事，不能塞同一个空值
- 修改（correction）=“actually……改到7点”：覆盖旧值，不能追加、也不能把别的槽清了
- JGA（联合目标准确率）=**整轮所有槽全对**才算对的轮数占比——全有或全无
- MultiWOZ=标准DST评测集（原课参考：2.4榜单2026年约80%～83%）
- ontology-free=不预设槽名表，让模型自由生成槽和值——灵活，但后续还得映射回业务字段

### 0.3 用小数字走一遍

三句话、四张状态表、一次JGA判分，把“填→改→清（不限）”走一遍（算例，与正文第五节可运行例子同构）：

~~~text
轮1 用户：“我想找北边一家便宜的意大利餐厅。”
轮2 用户：“actually，改成中等价位。”
轮3 用户：“菜系不限制了，什么都行。”
~~~

**gold状态（每轮的正确答案）：**

| 轮次 | cuisine | area | price | 发生了什么 |
|---|---|---|---|---|
| 初始 | None | None | None | 什么都没说 |
| 轮1后 | italian | north | cheap | 三个SET |
| 轮2后 | italian | north | moderate | 一次修改：只覆盖price，**其余原样保留** |
| 轮3后 | any | north | moderate | dontcare：cuisine=“不限”，**不是None** |

轮2是本课的灵魂：用户只动了price，area和cuisine绝不能跟着抖——把没提到的槽重置，是最经典的DST错误。轮3是另一个灵魂：用户“不限制”是明确表态（记any），和“从没提过”（None）要分开——两种“空”混在一个值里，后面就没法查“用户到底说过口味偏好吗”。

**假设系统的预测与JGA判分：**

| 轮次 | 预测状态 | 与gold逐槽对比 | 整轮全对？ |
|---|---|---|---|
| 轮1后 | {italian, north, cheap} | 3/3对 | ✓ |
| 轮2后 | {italian, north, cheap}（没听出“改成”） | price错（cheap≠moderate） | ✗ |
| 轮3后 | {None, north, moderate}（把“不限”当“没提”） | cuisine错（None≠any） | ✗ |

JGA=1/3≈0.33——**从轮2开始错**，一错到底。但逐槽看：cuisine对1轮、area对3轮、price对2轮，槽级正确率=7/9≈0.78。两个数字差出一倍多，这就是JGA“严格”的含义：单槽错误会传染整轮。与[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)的分类准确率对照还有一层：分类器一个样本错只算一个样本，DST一轮里一个槽错就废掉整轮——所以MultiWOZ榜上80%～83%（原课参考）看起来不如分类任务的95%体面，但难度完全不同。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

五段代码正好是0.3的零件：4.1的extract_cuisine是槽提取器（子串规则，正文指出含否定也照样命中的坑——0.3轮2的“改成中等”它就认不出）；4.2的update_state是状态机本体——先复制旧字典再覆盖，0.3轮2“其余原样保留”就是`new_state = dict(state)`那一行的功劳；4.3的RestaurantState+llm_dst是LLM路线（Pydantic锁类型，但schema保证合法不保证值对）；4.4的joint_goal_accuracy就是0.3的JGA公式（zip截断、分母取pred长度、空列表除零——正文三条纠错）；4.5的is_correction认“actually”这类提示词（0.3轮2开头那个词就是它要抓的cue）。第五节把4.1/4.2拼成可运行最小系统并加了checked_jga——0.3的三轮对话可直接换成英文喂给它验证。

### 0.5 理解检查（含参考答案）

**问题1：** “价格改中等”之后，cuisine和area应该是什么？为什么？

**参考答案：** 不变（italian、north）。修改事件只覆盖被提到的槽；未触及的槽必须原样保留（UNCHANGED）。把整个状态重置或顺带清空其他槽，系统就会“失忆”。

**问题2：** dontcare和None的区别是什么？为什么不能都写成None？

**参考答案：** dontcare=用户明确表态“不限”（有信息量的答案）；None=用户从未提及（未知）。都写None会把“用户说过随便”和“系统不知道”混淆——后续要追问确认口味时，前者不该再问。要靠事件或field presence区分（正文第七节1条）。

**问题3：** 为什么JGA比槽级准确率严格？0.3里两者差多少？

**参考答案：** JGA要求一轮内**所有**槽同时正确才算对（全有或全无），单槽错误废掉整轮；槽级逐个统计。0.3里JGA=1/3≈0.33，槽级=7/9≈0.78——JGA暴露“修改轮必错”这种系统性问题，槽级会把它平均掉。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

用户说“北边便宜的意大利餐厅”，下一轮改“价位中等”，系统应保留其他条件。本课维护结构化状态，区分未提及、明确修改、清除与不限制，避免把聊天误当执行授权。

## 二、核心概念与推导

domain是任务领域，slot是参数如cuisine/area/price，state是到当前轮的有效值。封闭槽位可分类，开放槽位可生成；ontology-free能自由生成槽名但仍需要后续业务映射。

（补充 2026-11-06）“状态是累积不是翻译”值得钉死：状态表不是最后一句话的解析结果，而是**从对话开始到当前轮所有信息的合并**。零节0.3的例子里轮3只说菜系，但area=north、price=moderate都还在表上——它们来自轮1和轮2。两种实现路线由此分野（正文4.3/4.5的原文）：增量patch每轮只更新变化（省输入、但错误会累积）；全历史重生成每轮从头看一遍对话（天然处理修改，但正文第七节4条指出固定轮长假设下累计输入近二次增长）。与[第17课](<17：聊天机器人——从规则、检索到神经模型与工具循环.md>)对照：那课的循环每轮问“现在做什么”，本课的表回答“现在用户要什么”——行为层每步都可以变，状态层必须忠实累积。

JGA=所有槽都完全正确的轮数/总轮数，是严格整体指标。还应看slot级P/R、修改轮、否定轮和多领域轮表现。MultiWOZ标注版本影响成绩，原文80%～83%不是本项目验收指标。

（补充 2026-11-06）JGA的严格性用“点菜”类比：服务员把你的菜记对九成，上桌错一道——客人体验是“这单废了”，不是“九成好”。JGA按“单”判分，槽级P/R按“菜”判分，两个都要报，但门禁用JGA更接近业务后果（订错餐厅=整单失败）。与[Phase02第09课](<../Phase02 · ML Fundamentals/09：模型评估——指标、交叉验证与数据泄漏.md>)的对照还有一层：分类指标里每个样本独立，DST的轮**不独立**——轮3的状态建立在轮2之上，轮2一错、轮3几乎必错（零节0.3的传染现象）。所以看JGA还要看“修改轮”“否定轮”这些**困难子集**（正文原话）：一个在修改轮上塌掉的系统，平均分再高也不敢上生产——平均会把最要命的轮稀释掉，这与[第27课](<27：LLM评估——RAGAS、DeepEval、G-Eval与人工校准.md>)“均值掩盖灾难样本”是同一个教训在两个课的回响。

更新政策需要至少三种事件：UNCHANGED未提不动；SET指定/修改；CLEAR删除原条件。any/dontcare表示用户不限制，不一定等于None未知。否定“不要意大利，改中餐”不能在更新后把新值又清空。

（补充 2026-11-06）三种事件的边界情形最值得背下来：①“随便，都可以”→SET为any（dontcare），不是CLEAR；②“不要意大利菜了”（不给新值）→CLEAR该槽或标记待定，但**绝不能**把用户随后给的新值又清掉（正文原话：更新后不能把新值清空——4.2源码“先填后清”的顺序坑）；③“好”“就那个”→UNCHANGED，但可能隐式确认了系统上一轮的提议——单看用户文本无法定位，要读系统话语（正文第七节3条）。三种“空”的对照表：

| 值 | 含义 | 例子 | 后端行为 |
|---|---|---|---|
| any（dontcare） | 用户明确说不限 | “菜系随便” | 该槽不做筛选 |
| None（未知） | 用户从未提及 | — | 该槽保持开放、可追问 |
| 旧值 | 用户提过且未撤销 | price=moderate | 按值筛选 |

people这类数值槽还要加正整数与上限校验（正文第三节的提醒），day要结合日期/时区——schema里的`Optional[int]`挡不住“订-3个人”。

规则可解释但覆盖窄；TripPy/BERT-DST、LoRA语言模型、结构化提示是候选路线。Pydantic保证类型约束不等于槽位含义正确，几行代码不能自动变成生产系统。

（补充 2026-11-06）“Pydantic保证类型不保证含义”展开一句：RestaurantState能保证cuisine字段只能是五个枚举之一（结构合法），但把“便宜的”填进area、把“周二”填进day，schema照单全收——**结构正确≠语义正确**。这与[第27课](<27：LLM评估——RAGAS、DeepEval、G-Eval与人工校准.md>)的教训同构：结构化输出能锁格式，锁不住内容质量；也和[第26课](<26：关系抽取与知识图谱——三元组、锚点、方向与溯源.md>)的span验证同构：锚点一致不等于语义成立。每道“格式闸门”后面都得再配一道“语义闸门”（业务校验、确认流程或人工规则）。三条路线的选择与[第06课](<06：命名实体识别——BIO、规则、CRF与神经模型.md>)的NER路线一脉相承：窄域规则基线可调试、广域要学出来或用LLM——结构化提示（Pydantic+约束解码）是2026年的新默认，几行代码就能上线，但“生产就绪”指的是结构层面，语义层面仍要走完校验。

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

