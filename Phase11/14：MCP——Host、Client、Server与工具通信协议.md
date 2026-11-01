# 14：MCP——Host、Client、Server 与工具通信协议

> 对应课程：Phase 11 / 14-model-context-protocol。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/14-model-context-protocol/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/14-model-context-protocol/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、从本地函数到跨应用工具

第09课你把 Python 函数放进注册表，自己的应用知道怎么调用。若另一个应用也想用同一组工具，每个应用都写一套发现、参数、错误和传输逻辑，会重复工作。

**MCP（Model Context Protocol，模型上下文协议）**统一这类交互的通信约定。协议像插头规格：双方同意消息形状和交互规则后可以连接；它不会替你判断答案，也不自动授权工具。

本课原文已采用 **2026-07-28** 版本，不能与网上常见 2025-11-25 的 initialize 会话流程混写。已核对[官方版本变更](https://modelcontextprotocol.io/specification/2026-07-28/changelog)：该版本改为每请求携带版本/能力的无状态协议，移除旧初始化握手和协议会话，并加入 server/discover。

## 二、五个角色/名词

| 名词 | 中文理解 | 实际责任 |
|---|---|---|
| Host | 宿主应用 | 用户界面、模型调用、权限及多个连接 |
| Client | 客户端连接组件 | 代表 Host 和一个 Server 按 MCP 交换消息 |
| Server | 能力提供端 | 暴露工具、资源、提示模板并执行处理 |
| Transport | 传输方式 | stdio 或 Streamable HTTP 等消息通道 |
| JSON-RPC | 请求响应信封 | 用 method、params、id 表达一次远程操作 |

“Server”不一定是远处服务器，也可以是本机子进程。**stdio** 通过标准输入/输出传消息；stdout 不能混进普通 print 日志，否则客户端可能把它当坏 JSON，诊断写 stderr。

三类服务能力：**Tools** 是可调用操作；**Resources** 是可按 URI 读取的内容；**Prompts** 是可复用消息模板。URI 是资源标识，不一定是浏览器网页地址。读取资源不代表它的内容可信。

## 三、读懂一个 JSON-RPC 请求

```json
{
  "jsonrpc":"2.0",
  "id":1,
  "method":"tools/call",
  "params":{
    "name":"add",
    "arguments":{"a":2,"b":3},
    "_meta":{
      "io.modelcontextprotocol/protocolVersion":"2026-07-28",
      "io.modelcontextprotocol/clientCapabilities":{}
    }
  }
}
```

jsonrpc 是信封版本，不是 MCP 版本；id 用于对应回复；method 表示调用哪种协议操作；name 是具体工具；arguments 是工具参数；_meta 携带协议元信息。响应使用同一 id，返回 result 或 error。通知没有 id，不期待对应响应。

原文规定版本/能力逐请求携带，不能只第一次存连接变量后省略。缺必须字段属于 Invalid Params `-32602`；字段齐全但版本不支持是 `-32022`，返回 requested 和 supported，便于客户端选择。

## 四、无状态不等于业务不能记事情

**无状态协议**意味着服务器不依赖某条连接先前握手留下的隐式上下文；每次请求携带足够协议信息。

业务仍可创建草稿并返回 `draftId`，下次更新把 draftId 放普通参数里。它是服务器签发的业务状态句柄，不是旧式 MCP 会话 ID。拿到句柄也不等于有权修改，仍需按调用者检查权限。

`server/discover` 返回支持版本、能力和身份信息。现代服务器要实现，客户端可先发现，也可在已知版本时直接调用并处理不兼容。**serverInfo 是自报身份，不是证书或权限凭据**。

旧 stdio 兼容策略：先探测 discover；得到结果或已知现代错误说明走现代路径；只有未识别错误/超时等符合兼容条件时才尝试旧 initialize 流程，不能一遇到授权失败就无限降级重试。

## 五、结果状态、缓存和往返补充输入

**resultType=complete** 表示结束；**input_required** 表示需要补充输入。原文的 MRTR（Multi Round-Trip Requests，多次往返请求）流程：服务端返回需要的信息和不透明 requestState→客户端收集允许的输入→用新 JSON-RPC id 重试原方法→带 inputResponses 并原样回传 requestState。不能自行猜测或修改不透明状态。

这是“返回后再请求”的模式，不是旧版服务端随意发起独立请求。旧结果没有 resultType 时按 complete 兼容。原课限定 input_required 出现在 tools/call、resources/read、prompts/get，不能任意放在所有列表响应。

**ttlMs** 表示缓存有效期提示（毫秒），**cacheScope** 区分 private/public。稳定排序的工具列表便于比较/缓存，也减少后续提示前缀漂移。private 结果不能跨调用身份共享。返回 complete 与“业务操作成功”不是同一层概念，工具仍可能带业务错误。

## 六、HTTP 与协议版本的对应

按本课 2026-07-28 约定，Streamable HTTP 使用 POST，返回一个 JSON 或请求范围 SSE 流；没有旧协议的独立 GET 事件流/DELETE 会话端点或 Mcp-Session-Id。长期订阅使用 subscriptions/listen POST。

原文要求 HTTP 请求头与 JSON 体一致：MCP-Protocol-Version 对应 _meta 版本，Mcp-Method 对应 method，特定调用的 Mcp-Name 对应工具名/资源URI/模板名；缺失或不一致按 HeaderMismatch 处理。**这些是该版本规则**，接旧 SDK/旧服务器时先核对版本，不能只改一个字符串宣称兼容。完整依据参见[官方传输规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)。

## 七、原文 Build It：注册、元数据、发现与传输

### 步骤 1：注册工具表面

给自定义MCPServer注册add工具；装饰器提供名字、描述、schema和handler。这个类的完整实现见仓库main.py，片段不是安装mcp包后就自动拥有的同名SDK。

````python
# 本块中文说明：注册工具表面
server = MCPServer("demo-server")

# 向当前集合加入一项数据。
@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
````

### 步骤 2：每次附加元数据

请求携带协议版本、能力和客户端信息。不能只在第一次保存，后续请求仍需要携带必需字段。

````python
# 本块中文说明：每次附加元数据
# 构造/发送当前请求并处理返回，具体行为取决于所在客户端类。
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
````

### 步骤 3：发现与列表

先server/discover选择版本，再tools/list查看能力；也可在已知协议下直接调用并处理版本错误。原文此步无独立代码块，正文已解释完整动作。

### 步骤 4：映射HTTP信封

展示POST头与JSON体一致的约定，工具名、method和版本要匹配。它是报文示例，不是已经启动HTTP服务。

````http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
````

### 步骤 5：落实授权边界

逐请求校验授权/Origin等，目录范围显式传递，stdout只留协议消息。destructiveHint只是标记，审批由实际Host/Server落实。


## 八、运行、观察与边界

在本课目录运行 `python code/main.py`，仅需标准库。原始示例定义 MCPServer/MCPClient、工具、资源和模板，并在进程内演示交互。应观察发现 demo-server、工具列表、工具调用、资源/模板读取，以及版本元数据的检查。**本地信封演示不等于已经实现远程 HTTP、OAuth 或完整协议一致性**。

读码顺序：`request_metadata` 组装元数据→`MCPClient.request` 每次附加→`MCPServer` 校验并分派→handler 执行业务→组装带结果状态/缓存信息的返回。注解 destructiveHint 是描述信号，不是强制访问控制；实际审批和授权由 Host/Server 执行。

原例有名为 delete_user 的演示工具，当前实现返回模拟数据，不能因此认为真实数据库已删除用户。真实写操作必须按授权、幂等和审计实现。保持协议层错误、工具业务错误和传输错误分开，便于定位。

## 九、原文练习与交付物

1. 添加 subtract 工具，检查 tools/list 按名字稳定排序，参数与返回正确。
2. 缺协议版本应得到 -32602；合法但不支持的版本得到 -32022；核对 requested 与 supported。
3. 创建业务草稿返回 draftId，更新时要求它，解释为什么它不是协议会话。
4. 模拟 input_required 与后续补输入，保持 requestState 原样、新 id、原方法，不调用真实有副作用动作。
5. 画双版本 stdio 客户端状态图，列出哪些探测结果允许回退、哪些应直接报错。

交付物是协议设计卡：版本、发现、每请求元数据、工具/资源/模板、状态句柄、错误、传输、缓存、权限及审批。后续原课指向 Phase13 的工具合同、可靠性/取消、供应链/回滚、协议一致性四类深入主题；现在先能手写一对请求响应并解释各字段即可。


## 本次验证记录（2026-11-01）

完整进程内标准库demo通过，发现demo-server并验证add(40,2)=42；不代表远程HTTP/授权/完整协议一致性已测试。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<13：完整LLM应用——请求流水线、流式输出与可观测性.md>) · [下一课](<15：提示前缀缓存——复用输入计算，而非复用整段答案.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
