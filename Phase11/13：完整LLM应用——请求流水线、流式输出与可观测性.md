# 13：完整 LLM 应用——请求流水线、流式输出与可观测性

> 对应课程：Phase 11 / 13-production-app。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/13-production-app/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/13-production-app/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、把前十二课连起来

此前每课练一个部件，本课把提示、检索接口、缓存、护栏、模型调用、费用和日志放进同一请求流程。原文称 production-ready，但源码主要是**生产架构教学骨架**，含模拟模型和未完成的集成；跑通 demo 不等于已具备上线条件。

应用的输入不再只是一段字符串，还包括用户身份、请求 ID、配置、上下文和预算；输出不再只有答案，还要有状态、来源、错误和必要的使用统计。

## 二、先认识服务端基础概念

- **HTTP API**：客户端通过地址和请求参数调用服务；endpoint（端点）是一类具体入口，如 `/chat`。
- **FastAPI**：Python Web 框架，组织端点和数据验证；**Uvicorn** 是运行 ASGI 应用的服务器。
- **API gateway**：请求入口，处理鉴权、路由、限流等，不等于网页前端。
- **Authentication 身份认证**确认你是谁；**Authorization 授权**确认能做什么。
- **Async/await 异步等待**：等待网络时让事件循环处理别的任务，不会自动让所有 CPU 计算变快。异步函数里调用同步阻塞 SDK 仍可能卡住循环。
- **CORS**：浏览器跨域访问规则，不是服务端鉴权，限制域名也不能代替用户验证。
- **Health check 健康检查**：检查服务能否提供功能。只返回 `healthy` 不证明模型、缓存、数据库都可用；存活检查与依赖就绪检查有区别。

## 三、一条请求如何流经各组件

认证与范围→输入校验/限流→选择提示及版本→检查带权限/版本的缓存→检索必要资料/执行获准工具→模型调用→输出校验→记录 usage/费用/trace→交付。缓存命中也应遵守当前输出政策与数据权限。

**Prompt router** 根据任务类型选模板；**A/B 实验**给不同用户或请求分配版本。用稳定哈希保证同一用户在同一实验期间一致；不要用默认 Python hash 当跨进程稳定分桶。先离线评估，再少量灰度，监控关键指标，保留可回滚的版本。

**Shadow mode** 是在后台比较新版本而不把其答案显示给用户。它仍有费用与数据处理，且不得让影子运行执行真实副作用操作。

## 四、流式输出到底是什么

**Streaming** 是生成过程中把片段逐步发给客户端。**TTFT** 是首个 token/片段延迟，**TPOT** 是后续 token 时间。流式改善等待感受，但不保证总生成时间减少。

**SSE（服务器发送事件）**是基于 HTTP 的单向事件流，常见文本帧为 `data: {...}\n\n`；WebSocket 是双向连接，适合交互更密集的场景；长轮询则不断等待/重发请求。

浏览器原生 EventSource 通常使用 GET，不能直接拿它发送任意 POST JSON；POST 聊天流常用 fetch + ReadableStream 或相应客户端。一个 SDK chunk 也不一定恰好一个模型 token。

输出审核与流式有取舍：完整审核后再发更容易控制，但增加首段等待；边生成边发则已经发出的内容无法收回。必须设计分段缓冲/审核与出错结束事件，不能声称“先全部审核且同时即时直通”没有成本。

## 五、失败、重试和降级

原文分三层：服务商错误（429/5xx/超时）、模型结果错误（格式/参数）、应用组件错误（数据库/检索等）。

**Exponential backoff** 按 1、2、4 等递增间隔重试；**jitter** 加随机扰动，避免很多请求同时重试。遵守 Retry-After，设置总期限和尝试上限；无效请求、权限错误不应盲重试。

**Fallback** 换备选模型或模式；必须保证必要能力、数据使用范围和输出合同仍满足。检索挂了时，资料依赖型问答应该说明无法获取依据，而不是悄悄凭模型记忆编政策。结构化输出失败不能直接返回原始文本给只接受 JSON 的下游。

**幂等键**避免重试重复执行写操作。客户端断开时应取消不必要工作；重试产生的实际模型费用仍需记录。日志不能只记录最后一次成功。

## 六、可观测性的三个层次

**Log 日志**是单个事件；**Metric 指标**是聚合数，如错误率；**Trace 链路追踪**把一次请求的多个步骤串起来，每一步为 **span**。

例如总耗时 1000ms，检索 200ms、模型 700ms、校验100ms，就知道优化检索最多能省哪一部分。嵌套 span 或并行 span 不能简单全加，因为时间可能重叠。

P50 表示一半请求不超过该延迟，P99 表示99%不超过；均值可能掩盖很慢的尾部。还需缓存命中、拦截率、失败率、每请求费用、token和任务质量。原课 P50<2s、P99<10s 等是演示目标，实际 SLO（服务目标）取决于业务。

## 七、原文 Build It：完整服务骨架

### 步骤 1：配置、日志与费用结构

设置模型枚举、演示价格、请求日志与累计费用。这里保存的数据合同将贯穿后续流程。该步代码来自完整源码，避免网页嵌套围栏导致代码截断。

````python
# 本块中文说明：配置、日志与费用结构
import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator


class ModelName(Enum):
    CLAUDE_SONNET = "claude-sonnet-5"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


def resolve_primary_model() -> ModelName:
    override = (os.environ.get("LLM_MODEL") or "").strip()
    if not override:
        return ModelName.CLAUDE_SONNET
    for model in ModelName:
        if model.value == override:
            return model
    known = ", ".join(m.value for m in ModelName)
    raise ValueError(f"LLM_MODEL={override!r} is not in the pricing registry (known: {known})")


PRIMARY_MODEL = resolve_primary_model()


MODEL_PRICING = {
    ModelName.CLAUDE_SONNET: {"input": 3.00, "output": 15.00},
    ModelName.GPT_4O: {"input": 2.50, "output": 10.00},
    ModelName.GPT_4O_MINI: {"input": 0.15, "output": 0.60},
}

FALLBACK_CHAIN = [PRIMARY_MODEL] + [m for m in ModelName if m is not PRIMARY_MODEL]


@dataclass
class RequestLog:
    request_id: str
    user_id: str
    timestamp: str
    prompt_template: str
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool
    guardrail_input_pass: bool
    guardrail_output_pass: bool
    cost_usd: float
    error: str | None = None


@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_requests: int = 0
    total_cache_hits: int = 0
    cost_by_user: dict = field(default_factory=lambda: defaultdict(float))
    cost_by_model: dict = field(default_factory=lambda: defaultdict(float))

    # 登记本次调用的统计信息。
    def record(self, user_id, model, input_tokens, output_tokens, cost):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.total_requests += 1
        self.cost_by_user[user_id] += cost
        self.cost_by_model[model] += cost

    # 汇总已累计的运行数据。
    def summary(self):
        avg_cost = self.total_cost_usd / max(self.total_requests, 1)
        cache_rate = self.total_cache_hits / max(self.total_requests, 1) * 100
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_request": round(avg_cost, 6),
            "cache_hit_rate_pct": round(cache_rate, 2),
            "cost_by_model": dict(self.cost_by_model),
            "top_users_by_cost": dict(
                sorted(self.cost_by_user.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }


````

### 步骤 2：提示模板与A/B选择

模板保存name/version/model；按稳定hash分桶选版本，format填入变量。字典中缺变量会报错，模板版本也应进入缓存键。

````python
# 本块中文说明：提示模板与A/B选择
@dataclass
class PromptTemplate:
    name: str
    version: str
    template: str
    model: ModelName = ModelName.GPT_4O
    max_output_tokens: int = 1024


PROMPT_TEMPLATES = {
    "general_chat": {
        "v1": PromptTemplate(
            name="general_chat",
            version="v1",
            template=(
                "You are a helpful AI assistant. Answer the user's question clearly and concisely.\n\n"
                "User question: {query}"
            ),
        ),
        "v2": PromptTemplate(
            name="general_chat",
            version="v2",
            template=(
                "You are an AI assistant that gives precise, actionable answers. "
                "If you are unsure, say so. Never fabricate information.\n\n"
                "Question: {query}\n\nAnswer:"
            ),
        ),
    },
    "rag_answer": {
        "v1": PromptTemplate(
            name="rag_answer",
            version="v1",
            template=(
                "Answer the question using ONLY the provided context. "
                "If the context does not contain the answer, say 'I don't have enough information.'\n\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ),
            max_output_tokens=512,
        ),
    },
    "code_review": {
        "v1": PromptTemplate(
            name="code_review",
            version="v1",
            template=(
                "You are a senior software engineer performing a code review. "
                "Identify bugs, security issues, and performance problems. "
                "Be specific. Reference line numbers.\n\n"
                "Code:\n```\n{code}\n```\n\nReview:"
            ),
            model=PRIMARY_MODEL,
            max_output_tokens=2048,
        ),
    },
}


AB_EXPERIMENTS = {
    "general_chat_v2_test": {
        "template": "general_chat",
        "control": "v1",
        "variant": "v2",
        "traffic_pct": 10,
    },
}


def select_prompt(template_name, user_id, variables):
    versions = PROMPT_TEMPLATES.get(template_name)
    if not versions:
        raise ValueError(f"Unknown template: {template_name}")

    version = "v1"
    for exp_name, exp in AB_EXPERIMENTS.items():
        if exp["template"] == template_name:
            bucket = int(hashlib.md5(f"{user_id}:{exp_name}".encode()).hexdigest(), 16) % 100
            if bucket < exp["traffic_pct"]:
                version = exp["variant"]
            else:
                version = exp["control"]
            break

    template = versions.get(version, versions["v1"])
    rendered = template.template.format(**variables)
    return template, rendered


````

### 步骤 3：教学语义缓存

存向量、答案和时间，按相似度找命中。示例键范围不足，不可直接用于真实多租户缓存。

````python
# 本块中文说明：教学语义缓存
def simple_embedding(text, dim=64):
    h = hashlib.sha256(text.lower().strip().encode()).hexdigest()
    raw = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim * 2), 2)]
    while len(raw) < dim:
        ext = hashlib.sha256(f"{text}_{len(raw)}".encode()).hexdigest()
        raw.extend([int(ext[i:i+2], 16) / 255.0 for i in range(0, min(len(ext), (dim - len(raw)) * 2), 2)])
    raw = raw[:dim]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm if norm > 0 else 0.0 for x in raw]


# 点积除以两向量范数；零向量按代码约定处理。
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    # 初始化本实例保存的配置和状态；self 指当前对象。
    def __init__(self, similarity_threshold=0.92, max_entries=10000, ttl_seconds=3600):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.entries = []
        self.hits = 0
        self.misses = 0

    # 读取已保存结果并检查相应命中条件。
    def get(self, query):
        query_emb = simple_embedding(query)
        now = time.time()

        best_score = 0.0
        best_entry = None

        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            score = cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            self.hits += 1
            return {
                "response": best_entry["response"],
                "similarity": round(best_score, 4),
                "original_query": best_entry["query"],
                "cached_at": best_entry["timestamp"],
            }

        self.misses += 1
        return None

    # 保存当前输入和结果，供后续查询使用。
    def put(self, query, response):
        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries = self.entries[len(self.entries) // 4:]

        self.entries.append({
            "query": query,
            "embedding": simple_embedding(query),
            "response": response,
            "timestamp": time.time(),
        })

    # 返回当前统计数据。
    def stats(self):
        total = self.hits + self.misses
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / max(total, 1) * 100, 2),
        }


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"you\s+are\s+now\s+DAN",
    r"system\s*:\s*override",
    r"<\s*system\s*>",
    r"jailbreak",
    r"\bpretend\s+you\s+have\s+no\s+(restrictions|rules|guidelines)\b",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
}

BANNED_OUTPUT_PATTERNS = [
    r"(?i)(DROP|DELETE|TRUNCATE)\s+TABLE",
    r"(?i)rm\s+-rf\s+/",
    r"(?i)(sudo\s+)?(chmod|chown)\s+777",
    r"(?i)exec\s*\(",
    r"(?i)__import__\s*\(",
]


````

### 步骤 4：输入输出护栏

执行规则匹配与必要修改，保存通过状态和原因。它们是规则示范，不是已经部署真实内容审核模型。

````python
# 本块中文说明：输入输出护栏
@dataclass
class GuardrailResult:
    passed: bool
    blocked_reason: str | None = None
    pii_detected: list = field(default_factory=list)
    modified_text: str | None = None


def check_input_guardrails(text):
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                blocked_reason="Potential prompt injection detected",
            )

    pii_found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            pii_found.append(pii_type)

    if pii_found:
        redacted = text
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        return GuardrailResult(
            passed=True,
            pii_detected=pii_found,
            modified_text=redacted,
        )

    return GuardrailResult(passed=True)


def check_output_guardrails(text):
    for pattern in BANNED_OUTPUT_PATTERNS:
        if re.search(pattern, text):
            return GuardrailResult(
                passed=False,
                blocked_reason="Response contained potentially unsafe content",
            )
    return GuardrailResult(passed=True)


````

### 步骤 5：调用、重试、费用与模拟流式

使用模拟调用展示异步等待、退避和备用模型。stream_response把完整文本分词播放，不等于真实模型即时流式。

````python
# 本块中文说明：调用、重试、费用与模拟流式
def estimate_tokens(text):
    return max(1, len(text.split()) * 4 // 3)


# 按输入/输出 token 及演示单价计算费用，注意每千或每百万的单位。
def calculate_cost(model, input_tokens, output_tokens):
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[ModelName.GPT_4O])
    input_cost = input_tokens / 1_000_000 * pricing["input"]
    output_cost = output_tokens / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 8)


SIMULATED_RESPONSES = {
    "general": (
        "Based on the information available, here is a clear and concise answer to your question. "
        "The key points are: first, the fundamental concept involves understanding the relationship "
        "between the components. Second, practical implementation requires attention to error handling "
        "and edge cases. Third, performance optimization comes from measuring before optimizing. "
        "Let me know if you need more detail on any specific aspect."
    ),
    "rag": (
        "According to the provided context, the answer is as follows. The documentation states that "
        "the system processes requests through a pipeline of validation, transformation, and execution stages. "
        "Each stage can be configured independently. The context specifically mentions that caching reduces "
        "latency by 40-60% for repeated queries."
    ),
    "code_review": (
        "Code Review Findings:\n\n"
        "1. Line 12: SQL query uses string concatenation instead of parameterized queries. "
        "This is a SQL injection vulnerability. Use prepared statements.\n\n"
        "2. Line 28: The try/except block catches all exceptions silently. "
        "Log the exception and re-raise or handle specific exception types.\n\n"
        "3. Line 45: No input validation on user_id parameter. "
        "Validate that it matches the expected UUID format before database lookup.\n\n"
        "4. Performance: The loop on line 33-40 makes a database query per iteration. "
        "Batch the queries into a single SELECT with an IN clause."
    ),
}


async def call_llm_with_retry(prompt, model, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            failure_chance = 0.15 if attempt == 0 else 0.05
            if random.random() < failure_chance:
                raise ConnectionError(f"API error from {model.value}: 500 Internal Server Error")

            await asyncio.sleep(random.uniform(0.1, 0.3))

            if "code" in prompt.lower() or "review" in prompt.lower():
                response_text = SIMULATED_RESPONSES["code_review"]
            elif "context" in prompt.lower():
                response_text = SIMULATED_RESPONSES["rag"]
            else:
                response_text = SIMULATED_RESPONSES["general"]

            return {
                "text": response_text,
                "model": model.value,
                "input_tokens": estimate_tokens(prompt),
                "output_tokens": estimate_tokens(response_text),
            }

        except (ConnectionError, TimeoutError):
            if attempt < max_retries:
                backoff = min(2 ** attempt + random.uniform(0, 1), 10)
                await asyncio.sleep(backoff)
            else:
                raise

    raise ConnectionError(f"All {max_retries} retries exhausted for {model.value}")


async def call_with_fallback(prompt, preferred_model=None):
    chain = list(FALLBACK_CHAIN)
    if preferred_model and preferred_model in chain:
        chain.remove(preferred_model)
        chain.insert(0, preferred_model)

    last_error = None
    for model in chain:
        try:
            return await call_llm_with_retry(prompt, model)
        except ConnectionError as e:
            last_error = e
            continue

    return {
        "text": "I apologize, but I am temporarily unable to process your request. Please try again in a moment.",
        "model": "fallback",
        "input_tokens": estimate_tokens(prompt),
        "output_tokens": 20,
        "error": str(last_error),
    }


async def stream_response(text):
    words = text.split()
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield token
        await asyncio.sleep(random.uniform(0.02, 0.08))


````

### 步骤 6：请求服务组合

handle_request按流程调组件，记录日志和费用；cache hit、blocked、fallback等分支要分别理解。运行代码的缺失能力与风险见正文核对段。

````python
# 本块中文说明：请求服务组合
class ProductionLLMService:
    # 初始化本实例保存的配置和状态；self 指当前对象。
    def __init__(self):
        self.cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=3600)
        self.cost_tracker = CostTracker()
        self.request_logs = []
        self.eval_results = []

    # 把缓存、模型、校验、日志和费用串成一条请求路径。
    async def handle_request(self, user_id, query, template_name="general_chat", variables=None):
        request_id = str(uuid.uuid4())[:12]
        start_time = time.time()
        variables = variables or {}
        variables["query"] = query

        input_check = check_input_guardrails(query)
        if not input_check.passed:
            return self._blocked_response(request_id, user_id, template_name, input_check, start_time)

        effective_query = input_check.modified_text or query
        if input_check.modified_text:
            variables["query"] = effective_query

        cached = self.cache.get(effective_query)
        if cached:
            self.cost_tracker.total_cache_hits += 1
            log = RequestLog(
                request_id=request_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                prompt_template=template_name,
                prompt_version="cached",
                model="cache",
                input_tokens=0,
                output_tokens=0,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                cache_hit=True,
                guardrail_input_pass=True,
                guardrail_output_pass=True,
                cost_usd=0.0,
            )
            self.request_logs.append(log)
            self.cost_tracker.record(user_id, "cache", 0, 0, 0.0)
            return {
                "request_id": request_id,
                "response": cached["response"],
                "cache_hit": True,
                "similarity": cached["similarity"],
                "latency_ms": log.latency_ms,
                "cost_usd": 0.0,
            }

        template, rendered_prompt = select_prompt(template_name, user_id, variables)
        result = await call_with_fallback(rendered_prompt, template.model)

        output_check = check_output_guardrails(result["text"])
        if not output_check.passed:
            result["text"] = "I cannot provide that response as it was flagged by our safety system."
            result["output_tokens"] = estimate_tokens(result["text"])

        cost = calculate_cost(
            ModelName(result["model"]) if result["model"] != "fallback" else ModelName.GPT_4O_MINI,
            result["input_tokens"],
            result["output_tokens"],
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        log = RequestLog(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_template=template_name,
            prompt_version=template.version,
            model=result["model"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=latency_ms,
            cache_hit=False,
            guardrail_input_pass=True,
            guardrail_output_pass=output_check.passed,
            cost_usd=cost,
            error=result.get("error"),
        )
        self.request_logs.append(log)
        self.cost_tracker.record(user_id, result["model"], result["input_tokens"], result["output_tokens"], cost)

        self.cache.put(effective_query, result["text"])

        self._log_eval(request_id, template_name, template.version, result, latency_ms)

        return {
            "request_id": request_id,
            "response": result["text"],
            "model": result["model"],
            "cache_hit": False,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "pii_detected": input_check.pii_detected,
            "guardrail_output_pass": output_check.passed,
        }

    # 原演示先取得完整回答再分词播放，不是模型端真实流式。
    async def handle_streaming_request(self, user_id, query, template_name="general_chat"):
        result = await self.handle_request(user_id, query, template_name)
        if result.get("cache_hit"):
            return result

        tokens = []
        async for token in stream_response(result["response"]):
            tokens.append(token)
        result["streamed"] = True
        result["stream_tokens"] = len(tokens)
        return result

    def _blocked_response(self, request_id, user_id, template_name, guardrail_result, start_time):
        log = RequestLog(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_template=template_name,
            prompt_version="blocked",
            model="none",
            input_tokens=0,
            output_tokens=0,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            cache_hit=False,
            guardrail_input_pass=False,
            guardrail_output_pass=True,
            cost_usd=0.0,
            error=guardrail_result.blocked_reason,
        )
        self.request_logs.append(log)
        return {
            "request_id": request_id,
            "blocked": True,
            "reason": guardrail_result.blocked_reason,
            "latency_ms": log.latency_ms,
            "cost_usd": 0.0,
        }

    def _log_eval(self, request_id, template_name, version, result, latency_ms):
        self.eval_results.append({
            "request_id": request_id,
            "template": template_name,
            "version": version,
            "model": result["model"],
            "output_length": len(result["text"]),
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def health_check(self):
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache": self.cache.stats(),
            "cost": self.cost_tracker.summary(),
            "total_requests": len(self.request_logs),
            "eval_entries": len(self.eval_results),
        }


````

### 步骤 7：演示入口

创建服务并发送固定样例，输出汇总信息。完整演示数据在原仓库，不重复粘贴冗长样本；运行命令见正文。

````python
# 本块中文说明：演示入口
if __name__ == "__main__":
    asyncio.run(run_production_demo())
````


## 八、运行步骤和能看到什么

先在本课目录运行 `python code/production_app.py`，这是标准库模拟 demo，会打印请求结果、命中、拦截、成本、健康状态和日志等。它不会自动部署外网服务。

Use It 中 FastAPI 包装是单独的接入示意。若想实验 HTTP，可在自己的练习文件里定义 `app=FastAPI()`，把请求映射到服务对象，安装 `fastapi uvicorn`，再运行 `uvicorn app:app --host 127.0.0.1 --port 8000`；这里前一个 app 指文件 `app.py`，后一个指变量 app。若没有创建该文件，不能直接执行该命令。访问本地 `/docs` 可手动测试端点。

Docker 部分展示如何把环境和启动命令封装进镜像；增加 worker 不等于一定支持某个固定并发量，且每个进程的内存缓存/限流/指标不共享。规模扩大时需要共享存储、连接池、队列、水平扩容和压测。

## 九、源码核对：不要把演示当成已实现的能力

1. `call_llm_with_retry` 返回模拟文字和估算 token，不是已接真实提供商。
2. `handle_streaming_request` 先等待完整回答，再把现成字符串逐词播放并收集；不是端到端降低模型 TTFT 的真实流式。
3. 缓存主要按 query 匹配，未充分带上用户权限、模板、版本和资料，真实多用户场景可能错复用。
4. 缓存返回路径对 output pass 的处理不等于重新验证当前政策；应统一交付入口检查。
5. 生成后被护栏替换成拒答文字，不能按替换后的长度重算服务商已发生的输出账单，应保留原 usage。
6. `eval_results` 多为长度/时间等日志，并未真正评估答案正确性。
7. 原文把 RAG 和真实工具整合列为练习，说明骨架未完整实现它们；不能说“所有十二课能力已经全接好”。

## 十、原文部署检查项与练习

原文十五项检查覆盖：密钥环境变量、用户限流、输入输出检查、缓存、流式、退避、备用模型、请求日志、成本统计、健康检查、输入输出上限、超时、CORS、负载测试。应以实际实现证据验收，不能只在配置里写一个布尔值；示例请求/并发阈值不是普遍标准。

五项练习：

1. 接入20篇文档、检索top-3，比较有/无资料的质量，单独记录检索耗时与来源。
2. 接入第09课工具注册表，在返回中记录 tools_used，验证执行结果确实参与回答。
3. 模拟每用户每天0.5美元、全局100美元等门槛，测试降级/拒绝规则，避免切模型绕过质量要求。
4. 保存提示版本与时间、指标，模拟100请求内错误率翻倍触发回滚；验证回滚后缓存键也随版本变化。
5. 每个阶段加 OpenTelemetry span，展示单次 trace，并避免在 trace 属性里写秘密。

交付物可以是本笔记里的架构审查：组件图、数据合同、状态/错误、版本、预算、恢复与验证结果。学到能追踪“一次失败发生在哪一步、是否已产生费用/副作用”，比只让网页能聊天更重要。


## 本次验证记录（2026-11-01）

内嵌代码已语法检查；未运行完整服务、真实流式、模型API、HTTP部署或并发压测。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<12：护栏与内容过滤——把检测、权限和证据核对分开.md>) · [下一课](<14：MCP——Host、Client、Server与工具通信协议.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
