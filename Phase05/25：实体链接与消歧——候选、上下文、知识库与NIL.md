# 25：实体链接与消歧——候选、上下文、知识库与NIL

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/25-entity-linking) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/05-nlp-foundations-to-advanced/25-entity-linking/docs/en.md)。原题：Entity Linking & Disambiguation。

## 一、这一课解决什么问题？

NER告诉你Jordan是一个提及，实体链接进一步判断指哪位Jordan或哪个地点，并映射到稳定知识库ID。候选召回与消歧要分别测，任何一层漏掉答案都会影响最终结果。

## 二、核心概念与推导

alias index把表面名映射到若干候选ID；candidate generation先求覆盖，disambiguation结合上下文、实体描述和先验选择。prior表示这个别名通常指谁，但流行实体不一定是领域正确答案。

词重合基线用Jaccard=交集词数/并集词数；embedding路线编码mention+context与实体描述，候选中按相似度排序；生成路线用trie约束合法标题/ID，但合法不等于正确，标题也要映射到稳定ID。

NIL表示知识库没有匹配项，必须允许拒绝硬猜。候选召回R与“正确候选已在列表时”的消歧正确率A相乘，才近似得到理想上游条件下端到端正确率：0.8×0.99=0.792，不是精确80%，更不是99%。还未计NER漏检。

共指聚合同一文档内提及；EL连接外部KB，两者互相帮助但不是同一任务。BLINK等并不全是一次前向同时完成NER的端到端系统，应按具体架构区分。

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

