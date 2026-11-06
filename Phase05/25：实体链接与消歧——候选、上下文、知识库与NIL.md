# 25：实体链接与消歧——候选、上下文、知识库与NIL

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。
> 2026-11-06 增补「零、初学者先读」与正文概念补充；当日与原课网页版核对概念与示例。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/25-entity-linking) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/25-entity-linking/docs/en.md)。原题：Entity Linking & Disambiguation。

## 零、初学者先读：同一个名字，世界上有几十万人叫

> 2026-11-06 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。本课是“信息抽取三部曲”的第二课：[第24课](<24：共指消解——提及、先行词、聚类与跨句实体.md>)把文档内的提法归成“一个实体”，本课负责问“这个实体在知识库里到底是哪一个”，[第26课](<26：关系抽取与知识图谱——三元组、锚点、方向与溯源.md>)再在挂上ID的实体之间抽关系连成图。

### 0.1 先理解这一课在做什么

NER（[第06课](<06：命名实体识别——BIO、规则、CRF与神经模型.md>)）告诉你“Jordan是个人名”；共指（第24课）告诉你“这段里的Jordan和他是一个人”。但到底是**哪个**Jordan？——Michael Jordan（篮球）、Michael B. Jordan（演员）、Michael I. Jordan（伯克利教授，原课说ML论文里这个混淆真实存在）、还是约旦（国家）？实体链接（EL）回答这个问题：把每个提及映射到知识库（Wikidata/Wikipedia/你的领域库）里**唯一的稳定ID**，查不到就明说“没有”（NIL），不硬猜。

类比：NER像机场安检看证件类型（“这是护照”），共指像把同一个人的行李归到同一条传送带，EL像海关拿护照号对系统——“张伟”这个名字全国有几十万人，必须看护照号（稳定ID）加行程单（上下文）才能确认是哪个张伟。工作分两步：**候选生成**（这个名字可能是哪几十个人？）和**消歧**（从名单里挑出对的那个人）。原课参考：Wikipedia别名索引约1800万个（别名，实体）对，一个提及典型返回10～30个候选。

### 0.2 本课术语先翻成白话

- 实体链接（EL, entity linking）=把提及映射到知识库唯一条目
- 知识库（KB, knowledge base）=Wikidata、Wikipedia或你自己的领域库
- 别名索引（alias index）=“表面名→候选实体列表”的倒排表，如apple→[Apple Inc., 苹果（水果）]
- 候选生成（candidate generation）=查别名表拿短名单；追求“正确答案在名单里”（覆盖）
- 消歧（disambiguation）=用上下文从名单里挑人；追求“挑得对”（精度）
- 先验（prior）=这个名字历史上通常指谁；流行实体不等于领域正确答案
- NIL=知识库里没有匹配项，系统应明说“无匹配”而不是硬选
- Jaccard=两集合交集÷并集，本课的最简消歧打分器
- AIDA-CoNLL=EL标准评测集（原课参考：1393篇路透社文章、3.4万个提及）
- 候选召回×消歧正确率=0.8×0.99=0.792——两层都好才有端到端的好

### 0.3 用小数字走一遍

一个提及、两个候选、三个判定，把“查名→对证据→判NIL”走一遍（算例，正文4.2的Jaccard可纸笔复核）：

~~~text
提及：Apple
上下文：Apple released new computers and phones
别名表：apple → [Q312 (Apple Inc.), Q89 (apple, fruit)]
描述：  Q312: "Apple company designs computers and phones"
       Q89:  "apple fruit grows on trees and is food"
~~~

**第一步：候选生成。** 查别名表“apple”，得到两个候选。这步只管“名单里有没有对的”，不做选择。

**第二步：上下文对证据（Jaccard手算）。** 上下文词集={apple, released, new, computers, and, phones}，共6个词：

| 候选 | 描述词集 | 交集 | 并集 | Jaccard |
|---|---|---|---|---|
| Q312（公司） | {apple, company, designs, computers, and, phones} | {apple, computers, and, phones}＝4 | 6+6−4＝8 | 4/8＝**0.5** |
| Q89（水果） | {apple, fruit, grows, on, trees, and, is, food} | {apple, and}＝2 | 6+8−2＝12 | 2/12≈**0.17** |

证据偏向“computers、phones”这些公司词，选Q312（Apple Inc.）。注意Jaccard只数词面重合，不认语序、不认否定——“Apple no longer makes phones”会得到同样的高分，这是词重合基线的固有盲区（正文有同款提醒）。

**第三步：判NIL。** 换个上下文“It tastes nice”：词集={it, tastes, nice}，与两个描述的交集都是空、Jaccard都是0.0。正文4.2的原代码best_score从−1起步，**零分也会让第一个候选胜出**——返回“Apple Inc., 0.0”。分数为0的意思是“没有任何证据”，正确行为是判NIL（或转人工），而不是默默选了哪个要看字典顺序。NIL还有另一面：提及是“zorpix”、别名表查无此名，返回(None, 0.0)——这是“候选都没召回”的NIL，与“候选在但证据为零”的NIL要分开统计。

**乘法账。** 候选召回R=0.8、消歧正确率A=0.99时，端到端≈0.8×0.99=0.792——消歧器再强，正确ID不在候选名单里它就选不到。这与[第24课](<24：共指消解——提及、先行词、聚类与跨句实体.md>)的“提及漏检”同构：上游的洞，下游补不上。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

四个零件按数据流读：4.1的alias_to_entities是第一步的别名索引（玩具版，3个键）；4.2的disambiguate是第二步的Jaccard打分器——0.3的表就是它逐候选算的东西，注意它“零分选首项”的危险默认（0.3第三步）；4.3的embed_mention/embed_entity把Jaccard换成句向量余弦（编码“提及+上下文”对“实体描述”，[MENTION]只是普通文本标记，不是训练过的特殊符号）；4.4的prompt走生成路线。第五节补齐了tokenize和entity_desc，0.3的两个描述就取自那里——可以直接跑通并核对0.5与0.17两个分数（原字典只给apple候选配了描述，查jordan前要自己补，否则KeyError，正文第五节的提醒）。

### 0.5 理解检查（含参考答案）

**问题1：** 候选生成和消歧分别负责什么？为什么必须分开测？

**参考答案：** 候选生成负责“正确答案在不在短名单里”（召回），消歧负责“在名单里时选不选得对”（精度）。分开测是因为错误责任不同：前者要改索引、扩候选，后者要改打分器；且端到端正确率≈两者相乘（0.8×0.99=0.792），混在一起看不出哪层在漏。

**问题2：** 什么情况应该判NIL？判了NIL之后怎么办？

**参考答案：** 别名表查无此名，或候选虽在但上下文证据为零、候选间难分时，应输出NIL而非硬选。之后可转人工、扩充KB，或按业务建“新实体”档案。把“函数总返回某个ID”当可靠是错的。

**问题3：** 共指（第24课）和实体链接（本课）有什么区别？

**参考答案：** 共指在**文档内部**把提法聚成实体（不问这个人是谁），EL把实体接到**外部知识库**的稳定ID上（不问文档里还有没有别的说法）。两者互相帮助——簇内只要链接一次、全簇共享ID——但任务定义不同。

能解释这个小例后，继续阅读下文“核心概念与推导”。

---

## 一、这一课解决什么问题？

NER告诉你Jordan是一个提及，实体链接进一步判断指哪位Jordan或哪个地点，并映射到稳定知识库ID。候选召回与消歧要分别测，任何一层漏掉答案都会影响最终结果。

## 二、核心概念与推导

alias index把表面名映射到若干候选ID；candidate generation先求覆盖，disambiguation结合上下文、实体描述和先验选择。prior表示这个别名通常指谁，但流行实体不一定是领域正确答案。

（补充 2026-11-06）“先求覆盖”值得展开成“地板”直觉：候选召回是整条流水线的上限，消歧、重排、LLM都只能在名单内挑。工程上宁可名单长一点（候选30个、召回99%）再让消歧去粗取精，也不要名单太短（召回80%）让再好的消歧器也无米下锅。这与[第14课](<14：信息检索——BM25、稠密向量、RRF与重排.md>)里“召回层松、重排层严”是同一个设计哲学：粗筛管覆盖，精排管正确。三条消歧路线一张表收拢：

| 路线 | 怎么打分 | 好处 | 短板 |
|---|---|---|---|
| 先验+上下文（Milne & Witten, 2008） | P(实体\|提及)×上下文相似度 | 快、免训练 | 先验偏置强 |
| 向量（BLINK等，2020～2024默认） | 编码提及+上下文，与候选描述比余弦 | 语义匹配、抗同义改写 | 要预编码整个KB，更新要重算 |
| 生成（GENRE/LLM） | 模型逐token写出实体标题，trie约束 | 端到端、灵活 | 合法≠正确，需白名单验证（正文4.4） |

词重合基线用Jaccard=交集词数/并集词数；embedding路线编码mention+context与实体描述，候选中按相似度排序；生成路线用trie约束合法标题/ID，但合法不等于正确，标题也要映射到稳定ID。

（补充 2026-11-06）Jaccard的三个盲区，正好说明为什么2020～2024年的默认做法换成了向量：一是不认否定——“Apple no longer makes phones”与“Apple makes phones”算出同样的词面重合；二是不认语义——“电脑”和“计算机”零重合、换种说法就归零；三是对短上下文太脆——零节0.3第三步里上下文稍不相关就全零。embedding路线把“提及+上下文”和“实体描述”都编码成向量、用余弦比较，等于把词面匹配换成语义匹配（正是[第03课](<03：词向量与从零实现 Word2Vec.md>)以来“意思接近=向量接近”的直接应用）；生成路线再进一步，直接让模型“写出”实体的规范标题，用trie/白名单约束输出必是合法ID——合法不等于正确，标题最终仍要映射到稳定ID（正文4.4的纠错）。

NIL表示知识库没有匹配项，必须允许拒绝硬猜。候选召回R与“正确候选已在列表时”的消歧正确率A相乘，才近似得到理想上游条件下端到端正确率：0.8×0.99=0.792，不是精确80%，更不是99%。还未计NER漏检。

（补充 2026-11-06）NIL可以拆成三种情形分别统计，混在一起就无法改进：其一，**别名缺**——实体在KB里存在，但索引没收这个写法（“C罗”没进Cristiano Ronaldo的别名表）；其二，**实体缺**——实体本身不在KB里（新兴公司、你司新员工），这是真正的NIL；其三，**证据缺**——候选都在，但上下文不足以裁决（零节0.3的“It tastes nice”）。三种情形的处方不同：扩别名、扩KB/建新实体、转人工或澄清。原课还提醒流行偏置：ML论文里的“Michael I. Jordan”常被链到篮球乔丹——先验在高频实体上过强，正是第三种情形被判错的样子。

共指聚合同一文档内提及；EL连接外部KB，两者互相帮助但不是同一任务。BLINK等并不全是一次前向同时完成NER的端到端系统，应按具体架构区分。

（补充 2026-11-06）用一条数据流把三部曲接起来（原课Use It的生产模式，补充翻译）：

~~~text
文本 ──NER（第06课）──▶ 提及 ──共指（第24课）──▶ 文档内簇 ──EL（本课）──▶ 每簇一个KB稳定ID
~~~

注意方向：是“先在文档内归簇、再对簇做一次链接、全簇共享ID”，而不是每个提及各链各的——同一簇里“Apple/该公司/它”若链到不同ID，说明共指或链接至少一个出了错。这样设计还省事：簇内几十个提及只要消歧一次，其余照抄结论。原课同时提醒：BLINK等系统并非都一次前向同时完成NER+EL，“端到端”与“流水线”要按具体架构区分——流水线的红利是每个零件可单独替换、单独验收（[第26课](<26：关系抽取与知识图谱——三元组、锚点、方向与溯源.md>)接着用这些ID连边）。

### 2.9 用一个小例子把概念串起来（补充）

候选表把Apple映射公司与水果。context含company/phones时，描述重合会偏公司；若上下文只有一个不相关词，Jaccard可能全0，源码仍选第一候选。你应当把低证据/并列情形纳入NIL或澄清，不能把“函数总返回一个ID”当可靠性。

## 三、操作前准备

Jaccard只用标准库；句向量需：
~~~bash
python -m pip install sentence-transformers
~~~
原文alias条目把ID与名称拼在同一字符串，本篇补充入口保持完全相同key；真实系统应拆id/name/description/version字段。不得在未授权情况下下载公司员工数据来做练习。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 建立别名候选表

原代码是小型手工字典，不是真从Wikipedia dump解析的完整倒排索引。真实数据量与版本不同，需处理重定向、别名正规化、大小写、同名与增量更新。

~~~python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
~~~

### 4.2 用上下文Jaccard选候选

先tokenize，再算context与description的集合重合，忽略词频与词序。原文缺tokenize和entity_desc，下一节补齐。零分仍会选第一个候选；这不是有效NIL策略。

~~~python
# 候选内Jaccard基线，零分仍会选首项，需另加NIL政策。
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
~~~

### 4.3 句向量消歧

mention_span按Python字符[start,end)切；[MENTION]标签不是该encoder天然训练过的特殊符号，只是普通文本。称BLINK-style是架构借鉴，不等于重现BLINK的训练与准确率。

~~~python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 以字符跨度标出提及，标记不等于已训练特殊token。
def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
~~~

### 4.4 生成合法知识库标题

prompt仅要求JSON，未实际连接trie或choice约束；须把候选列表与ID白名单交给约束/验证层，检查输出确实存在且适配上下文。GENRE按模型token解码，不是简单逐字符循环。

~~~python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
~~~

### 4.5 AIDA与领域评价

分开候选Recall@K、候选内top1、端到端链接P/R/F1与NIL检测。AIDA子集/标注协议影响结果，不把单一benchmark成绩直接套到中文或企业新实体。



## 五、把代码运行起来：补充入口与检查

补充缺失输入，纯标准库运行：
~~~python
import re

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

entity_desc = {
    "Q312 (Apple Inc.)": "Apple company designs computers and phones",
    "Q89 (apple, fruit)": "apple fruit grows on trees and is food",
}
entity_id, score = disambiguate(
    "Apple", "Apple company designs phones", alias_to_entities, entity_desc
)
print(entity_id, score)
assert entity_id == "Q312 (Apple Inc.)"
assert disambiguate("unknown-name", "nothing", alias_to_entities, entity_desc) == (None, 0.0)
~~~
注意：这里只给apple候选补了描述，不要直接拿jordan查询而不补描述，否则KeyError。未来把低分/候选过近也交给NIL/人工复核，但阈值要校准。

## 六、Use It：使用现成工具

BLINK/REL/mGENRE等是原课候选路线，中文到英文KB需多语言表示或适当翻译；低量可用候选内生成但仍要NIL。生产KB需版本、更新、删除、权限与来源管理。pipeline容易替换单个组件；端到端也要诊断候选漏召回。



## 七、失败模式、原文边界与工程注意

1. 正确ID不在候选中时，重排器再强也选不到。
2. 对所有候选零重合仍返回首项，是源码的危险默认行为。
3. 实体名称相同不表示同一个对象；不同提及可共享ID但要保留offset。
4. [MENTION]标记不自动获得专门实体链接能力。
5. span错误、流行偏置、KB过期与语言不匹配需单独统计。
6. 不要把EL结果自动视为事实确认；KB也可能不全或有错。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
定义KB稳定ID、版本和刷新；设计alias/embedding候选召回，再做context消歧，保留NIL。分别测候选Recall@K、条件top1、端到端F1、NIL F1与同名冷门实体；生成输出必须经过候选ID验证。
~~~

## 九、Exercises：保留原课练习与验收方法

1. Easy：10个Paris/Jordan/Apple歧义样本手标ID，对比先验与上下文，报告候选漏召回和消歧错分。
2. Medium：50个歧义提及，对候选描述建向量，比较Jaccard与embedding；相同候选池、同一gold，校准NIL。
3. Hard：建1000实体的合成/授权领域KB，串联NER+EL，在100留出句上测P/R；不要将真实隐私员工数据上传外部模型。

## 十、术语与完成标准

alias：别名；candidate recall：候选中包含正确ID的比例；disambiguation：上下文消歧；NIL：KB无匹配；canonical ID：规范ID；KB：知识库。

完成检查：

- 能区分NER、coref与EL。
- 能运行Jaccard补充例。
- 能解释候选召回上限。
- 能发现全零分硬选首项。

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

