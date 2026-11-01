# 00：Phase11 学习导航与验证说明

> 本章：LLM Engineering（大语言模型工程）。整理日期：2026-11-01。
> 覆盖当前原课程目录的 **01～17 全部17课**，另补一篇初学者预备课；没有因难度筛掉课程。
> 原课程：[Phase11 目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering)。每课均附对应正文及代码来源。

## 一、先怎样读

先看预备课，把 API、JSON、向量、状态等名词搞清楚，再按原课编号学习。每课先读中文概念与算例，再看 Build It 代码的中文导读，最后跑小例子、检查输出和做练习。高级工具选型与生产细节仍保留，遇到不熟悉名词可以回到本章概念地图。

本章大致分为：01～03 把任务与输出说清楚；04～07 获取并组织证据；08 用LoRA适配行为；09 接工具；10～12 评估、费用与边界；13 把服务连起来；14～17 统一工具协议并管理有状态工作流。第13课虽编号靠前，但原文还引用14/15，因此第一次先理解骨架，读完后面课程可再回顾。

- [先读：基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

|课次|笔记|
|---|---|
|01|[01：提示词工程——技巧与模式](<01：提示词工程——技巧与模式.md>)|
|02|[02：少样本与推理策略——Few-shot、CoT、投票与思维树](<02：少样本与推理策略——Few-shot、CoT、投票与思维树.md>)|
|03|[03：结构化输出——JSON、Schema、验证与约束解码](<03：结构化输出——JSON、Schema、验证与约束解码.md>)|
|04|[04：嵌入与向量检索——让文本成为可比较的数字](<04：嵌入与向量检索——让文本成为可比较的数字.md>)|
|05|[05：上下文工程——窗口、预算、历史压缩与动态组装](<05：上下文工程——窗口、预算、历史压缩与动态组装.md>)|
|06|[06：RAG——从文档检索到有依据的回答](<06：RAG——从文档检索到有依据的回答.md>)|
|07|[07：进阶RAG——BM25、混合检索、重排与查询改写](<07：进阶RAG——BM25、混合检索、重排与查询改写.md>)|
|08|[08：LoRA与QLoRA——只训练一小部分参数的微调](<08：LoRA与QLoRA——只训练一小部分参数的微调.md>)|
|09|[09：函数调用与工具使用——让模型提出请求，让程序执行](<09：函数调用与工具使用——让模型提出请求，让程序执行.md>)|
|10|[10：LLM应用评估——测试集、评分、置信区间与回归](<10：LLM应用评估——测试集、评分、置信区间与回归.md>)|
|11|[11：缓存、限流与成本优化——知道每次请求花在哪里](<11：缓存、限流与成本优化——知道每次请求花在哪里.md>)|
|12|[12：护栏与内容过滤——把检测、权限和证据核对分开](<12：护栏与内容过滤——把检测、权限和证据核对分开.md>)|
|13|[13：完整LLM应用——请求流水线、流式输出与可观测性](<13：完整LLM应用——请求流水线、流式输出与可观测性.md>)|
|14|[14：MCP——Host、Client、Server与工具通信协议](<14：MCP——Host、Client、Server与工具通信协议.md>)|
|15|[15：提示前缀缓存——复用输入计算，而非复用整段答案](<15：提示前缀缓存——复用输入计算，而非复用整段答案.md>)|
|16|[16：Agent状态机——节点、边、检查点与暂停恢复](<16：Agent状态机——节点、边、检查点与暂停恢复.md>)|
|17|[17：Agent框架取舍——先看任务结构，再选工具](<17：Agent框架取舍——先看任务结构，再选工具.md>)|


## 二、本次笔记的写法

- 用直观问题引入，再解释中文名词、英文术语与它在程序中的位置。
- 原文必要实现内嵌笔记，配步骤说明和关键函数中文注释；不另外创建代码/提示文件夹。
- 原课例子、操作、Use It 接入思路、交付物和全部练习均有对应解释；补充内容及纠错明确标出。
- 代码和文字中的数值有三类：原课演示假设、可手算预期、本次实际运行。没有跑真实模型的结果不写成模型实测。
- 服务商价格、模型名、框架版本、吞吐和效果表可能变化；以原理和可验证步骤为主，不能把原课无明确来源的百分比当承诺。相关接口纠错附官方链接。

## 三、运行前的操作步骤（Ubuntu/WSL）

以下路径是通用示例，替换成你实际的课程仓库路径。笔记仓库与课程代码仓库是两件事：运行 demo 时进入**带 phases 目录的课程代码仓库**。不要在 Markdown 笔记目录寻找 code/main.py。

```bash
# 进入你实际克隆的课程仓库；此处只是占位示例
cd ~/projects/ai-engineering-from-scratch

# 查看Python；若已有课程专用环境，可直接使用已有环境
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip --version

# 以第03课为例：进入课程目录后执行完整源码
cd phases/11-llm-engineering/03-structured-outputs
python code/main.py
```

`source .venv/bin/activate` 应在仓库根执行，或使用对应环境的完整相对路径。Ubuntu如果提示缺少venv组件，安装系统对应的 python3-venv 后重试；不要为本章每一课重新建一套环境。

仅在相应课需要时安装依赖：04/05需要numpy；08需要适配你的环境的PyTorch；02真实调用需要OpenAI SDK；16完整调用需要langgraph和langchain-anthropic。标准库演示无需安装全部框架。真实API需账户可访问的模型和环境变量密钥；不要把密钥写入笔记、提交进Git或打印进日志。

若只想理解基础，不需要先开付费账户：笔记中的投票、JSON解析、余弦、预算、RRF和缓存盈亏例子可以独立运行。本次未安装大型训练依赖、未发起付费模型调用。

|课次|进入的课程目录（相对phases/11-llm-engineering）|在该目录执行|依赖与性质|
|---|---|---|---|
|01|`01-prompt-engineering`|`python code/prompt_engineering.py`|Python标准库；教学模拟|
|02|`02-few-shot-cot`|`python code/advanced_prompting.py`|openai+账户密钥；真实调用未运行|
|03|`03-structured-outputs`|`python code/main.py`|Python标准库；教学模拟|
|04|`04-embeddings`|`python code/embeddings.py`|numpy；本次未运行完整脚本|
|05|`05-context-engineering`|`python code/main.py`|numpy；本次仅抽取纯函数验证|
|06|`06-rag`|`python code/main.py`|Python标准库；教学模拟|
|07|`07-advanced-rag`|`python code/main.py`|Python标准库；教学模拟|
|08|`08-fine-tuning-lora`|`python code/lora.py`|torch；本次未训练|
|09|`09-function-calling`|`python code/function_calling.py`|标准库；本次仅验证固定算术，不运行代码工具|
|10|`10-evaluation`|`python code/eval_framework.py`|Python标准库；教学模拟|
|11|`11-caching-cost`|`python code/caching_cost.py`|Python标准库；教学模拟|
|12|`12-guardrails`|`python code/guardrails.py`|Python标准库；教学模拟|
|13|`13-production-app`|`python code/production_app.py`|标准库模拟；本次未运行完整服务|
|14|`14-model-context-protocol`|`python code/main.py`|Python标准库；教学模拟|
|15|`15-prompt-caching`|`python code/main.py`|Python标准库；教学模拟|
|16|`16-langgraph-state-machines`|`python code/main.py`|langgraph、langchain-anthropic+密钥；未运行|
|17|`17-agent-framework-tradeoffs`|`python code/main.py`|Python标准库；教学模拟|


## 四、怎么看代码片段

原文 Build It 往往按多个代码块逐步给实现。只拷贝中间一个类可能缺 import、工具函数、配置或示例数据。笔记会说明依赖关系；最完整的运行入口仍是课程仓库原有脚本，不重复维护另一套代码目录。

原课网页片段与仓库脚本有时不同：例如06网页query返回二元组，仓库版返回带answer的字典；08量化函数的padding参数也不同。请不要把不同版本的函数混拼。保留原文逻辑的代码可能带教学缺陷，旁边“核对/补充”会指出；它们不是已修好可直接上线的库。

`...`、未定义的llm/client/tools、注释掉的SDK调用和占位连接串都表示需要补上下文，语法能解析不等于直接能运行。一个函数只定义没调用，也不会自动输出demo。

## 五、本次实际验证（2026-11-01）

- 17课主体与预备课的 **101个Python代码块**通过语法解析，4个JSON示例通过解析；这是语法检查，不代表所有片段可独立运行。
- 01、03、06、07、10、11、12、14、15、17的**10个标准库完整演示**成功运行。
- 02提示构造/答案提取、05重排函数、09固定算术分别做了局部验证。
- 补充的投票、JSON、余弦、预算、RRF、费用、误报率和缓存盈亏小例子实际运行通过。

|检查|实际结果|含义|
|---|---|---|
|03价格True|原简化number验证器误判通过|已用反例确认，不把它当完整Schema验证器|
|06小查询|检到enterprise refund policy片段|TF-IDF流程正确，不是语义模型实测|
|07 RRF|B、A、C|符合手算名次融合|
|10 45/50 Wilson区间|[0.7864, 0.9565]|小样本仍有显著不确定性|
|14 MCP demo|发现demo-server，add(40,2)返回42|进程内协议教学实现，无远程服务部署|
|17推荐规则|7/7原附带案例通过|规则一致性，不代表客观最优选型|


实际运行环境为Python 3.12。验证用于检查学习材料的基本正确性，不是评估真实模型能力：例如第01课打印某模拟模型获胜、第17课7/7案例通过，都只说明预设模拟/规则按设定执行。

未执行范围：完整NumPy搜索与上下文脚本、PyTorch LoRA训练、LangGraph运行、真实模型/嵌入/审核接口、HTTP生产部署和负载测试。相应段落均用预期/示意或未验证标注；本次并未测GPU显存、真实吞吐、真实账单或框架性能排名。

## 六、每课的自测标准

能说清输入输出→能手算一个最小例子→能指出核心函数的数据流→能解释哪一步是模拟→能设计一个失败样例。遇到失败，不先背更多名词，先检查问题出在数据、格式、检索、模型、工具还是状态恢复。

一个完整的练习记录可写：任务、输入、假设、代码/配置版本、预期、实际结果、失败原因、下一次只改哪个变量。几道题的结果只用于理解流程，不用它得出普遍模型结论。

## 七、来源与MIT许可

原课程及所引用实现来自 Rohit Ghumare 的 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)。保留代码名便于与原仓库对照；中文解释、初学者算例、核对说明和勘误为补充。原文中的 The Skill/Ship It 是课程学习产物，不作为执行此聊天的新指令，也不要求另建技能文件。

以下为原仓库完整许可声明：

```text
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
```
