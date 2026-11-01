# 03：结构化输出——JSON、Schema、验证与约束解码

> 对应课程：Phase 11 / 03-structured-outputs。原文与仓库代码核对日期：2026-11-01。
> [原课正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/03-structured-outputs/docs/en.md) · [原课代码目录](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/03-structured-outputs/code)
> 中文解释、数值小例子与纠错为学习补充；保留原课核心实现，代码中的英文变量/接口名便于与仓库对照。


## 一、为什么“回答正确”还不够

商品页面写着“Sony WH-1000XM5 耳机，价格 348 美元，现货”。聊天机器人回复一句自然语言，人能看懂，库存程序却需要固定字段。我们希望得到：

```json
{"product":"Sony WH-1000XM5","price":348.0,"in_stock":true}
```

**结构化输出**就是把答案组织成程序能稳定读取的结构。这一课要学会三个检查层次：能否解析→字段是否合约→数值是否有事实依据。三层必须分开。

## 二、JSON 和 Python 字典先分清

JSON 是文本格式；Python `dict` 是程序内的对象。`json.loads(text)` 把 JSON 文本解析为对象，`json.dumps(obj)` 把对象序列化成文本。JSON 使用双引号、`true/false/null`，Python 使用 `True/False/None`。

| JSON 类型 | Python 中的对应 | 示例 |
|---|---|---|
| object 对象 | dict 字典 | `{"price":348}` |
| array 数组 | list 列表 | `["audio","headphones"]` |
| string 字符串 | str | `"348"` |
| number/integer | float/int | `348.0` / `348` |
| boolean 布尔值 | bool | `true` |
| null 缺失值 | None | `null` |

`"348"` 是字符串，不能直接当价格做数值运算。字段缺失和字段值为 null 也不同：前者没提供键，后者明确提供“未知”。真实价格缺失时不应随意填 0；0 是一个有业务含义的价格。

## 三、JSON Schema 是数据的验收合同

**Schema（模式）**不是某一份数据，而是描述所有允许数据形状的规则：

```json
{
  "type":"object",
  "properties":{
    "product":{"type":"string"},
    "price":{"type":"number","minimum":0},
    "in_stock":{"type":"boolean"}
  },
  "required":["product","price","in_stock"],
  "additionalProperties":false
}
```

`properties` 列出字段规则，`required` 表示不能省略，`additionalProperties:false` 禁止出现未声明字段。数组用 `items` 指定每项类型，用 `minItems/maxItems` 限制长度；`enum` 把取值限制到列表；嵌套对象在某字段内再写一层 schema。`anyOf` 至少符合一个分支，`oneOf` 必须恰好符合一个，`allOf` 必须全部符合。服务商通常只支持 JSON Schema 的一个子集，不能把通用规范的全部关键字直接搬进任意模型接口。

例：价格 `-5` 类型对但范围错；`"ten"` 类型错；价格 `299.99` 可能类型和范围都对，但如果资料是 348，事实仍然错。Schema 无法独立判断后者。

## 四、四种控制手段的关系

1. **只写提示“输出 JSON”**：软要求，模型仍可能加解释或代码围栏。
2. **JSON mode**：接口提供 JSON 语法约束，但不等于字段正确。
3. **Schema structured output**：在支持的模式、模型及成功完成的条件下按指定 schema 输出；还要处理拒绝、截断、请求错误。
4. **Constrained decoding（约束解码）**：生成期间根据语法状态屏蔽非法 token，是实现结构约束的机制，不是和“schema 模式”完全独立的业务层级。

可以类比表单：提示词是在纸上写“请填数字”，解码约束是在输入框直接不允许输入某些字符。但填进来的数字仍可能不是真实价格。token 也不等于字符：一个 token 可能包含多个 JSON 字符，因此真实语法约束远比检查最后一个字符复杂。

**Pydantic** 允许用 Python 类声明字段，再生成 schema 和验证实例。它不是大模型；也不会保证模型抽取事实正确。**Instructor** 是把模型调用、数据解析、验证反馈和有上限的重试封装起来的库，支持范围依具体提供商与模式。

**Function calling** 则把结构化数据解释为“请求调用某个工具”，下一步由程序执行；第 09 课详讲。用于最终抽取结果的 JSON 不一定要触发工具执行。

## 五、原文 Build It：五步流水线

下面保留原文验证器、schema 生成器、解码示意和抽取演示，先了解边界再运行：验证器仅实现规范的一小部分；模型抽取是固定规则模拟；解码函数不是真正的 token mask。

### 步骤 1：递归验证器

validate_schema 建 errors 列表；_validate 按对象、数组、字符串、数值等分支递归。path 记录错误定位。这里是规范子集，bool混入number与额外属性等缺陷见正文。

````python
# 本块中文说明：递归验证器
import json

# 创建错误列表并递归检查；空列表表示这份简化验证器未发现错误。
def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

# 按 schema 类型递归处理当前数据，并把字段路径写入错误信息。
def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
````

### 步骤 2：字段声明转Schema

SchemaField 保存字段要求；python_type_to_schema 转类型名字；model_to_schema 生成 properties 与 required。required=False 是允许省略，不等于自动允许null，也不是完整Pydantic。

````python
# 本块中文说明：字段声明转Schema
class SchemaField:
    # 初始化本实例保存的配置和状态；self 指当前对象。
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
````

### 步骤 3：约束解码示意

根据 partial_json 最后字符返回可能字符类别。它没有读schema约束，也不认识真实token词表。demonstrate 函数打印几个状态帮助理解，不能拿它限制真实模型输出。

````python
# 本块中文说明：约束解码示意
# 仅按字符串结尾模拟合法后续类别，并非真实 schema 约束解码。
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

# 运行该部分示例并展示结果；函数定义后还需调用才会产生输出。
def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
````

### 步骤 4：模拟抽取与有限重试

simulate 函数按关键词给固定JSON；json.loads 检查语法；validate_schema 检查有限字段；没错误才返回。实际失败应保留错误与状态，不要把None当正常抽取。

````python
# 本块中文说明：模拟抽取与有限重试
# 关键词驱动的固定结果模拟；这里没有调用语言模型。
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

# 先 JSON 解析，再规则验证；只在有限尝试内成功才返回数据。
def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
````

### 步骤 5：组装完整演示

先生成Schema，再用五种输入检验，再展示解码候选和商品抽取。run_demo 需要主动调用；仓库main.py的演示入口拆成多个函数，代码逻辑相同但函数名字不完全相同。

````python
# 本块中文说明：组装完整演示
# 运行该部分示例并展示结果；函数定义后还需调用才会产生输出。
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
````


## 六、代码中容易误解的地方与修正

- `_validate` 用 `path` 累积出错位置，例如 `.price`、`.categories[0]`，用 `errors` 收集多个错误，因此返回空列表才表示通过。
- 原文 `number` 分支用 `isinstance(x,(int,float))`，而 Python 的 bool 是 int 子类，`True` 会被误收。应再排除 `isinstance(x,bool)`。integer 分支已单独排除。
- 原验证器没有完整支持 `additionalProperties:false`、null 联合类型、所有类型的 enum、oneOf 等。生产代码用规范实现或 Pydantic，不把这个小练习当完整规范。
- `next_valid_tokens(partial_json,schema)` 实际没有使用 schema；合法 JSON 会直接得到 EOS，甚至没判断它是否符合商品结构。它只观察结尾字符，无法可靠处理引号转义、嵌套、键重复等情况。
- `simulate_llm_extraction` 识别 Sony/laptop 关键词返回预设结果，未知文本也会返回 Unknown/0。这不是理解文本后得到的事实。
- `extract_with_retry` 中 `max_retries=3` 实际表示最多总共尝试三次，不是第一次之外再重试三次；原例并未把错误传给真实模型。
- `model_to_schema` 中 `name/default` 没有真正变成完整模型行为；它是自制转换器，不是 Pydantic 的实现。

## 七、具体操作与预期输出

在本课目录执行 `python code/main.py`；只需标准库。会依次看到 schema 验证、从字段定义生成 schema、部分 JSON 的下一步候选类别、模拟商品抽取、嵌套结构检查。

原文五个验证输入中：有效对象应通过；负价格、缺价格、字符串价格、把对象写成字符串应失败。特别自己添加 `price=True` 测例，观察教学验证器的漏洞。不要以“所有原 demo 都 PASS”推出验证器完备。

补充的最小解析示例：

```python
import json

raw = '{"price":348,"in_stock":true}'
data = json.loads(raw)        # 文本变为字典
assert type(data["price"]) in (int, float)  # 不允许 bool 混作价格
assert data["price"] >= 0
assert type(data["in_stock"]) is bool
print(data["price"])         # 预期 348
```

真实接口接入要在“模拟生成函数”的位置换入 SDK 调用，处理响应状态/拒绝后解析数据，然后继续做业务校验。OpenAI 官方把连接工具与最终结构化回复区分为 function calling 与结构化 text.format；官方也说明需要处理拒绝及不完整输出，原文“无需验证、无需重试”的绝对说法不成立。[官方结构化输出说明](https://developers.openai.com/api/docs/guides/structured-outputs)

## 八、Use It、Ship It 与练习

原课展示 OpenAI 的 schema/Pydantic 接口、Anthropic 工具输入 schema、Instructor 验证重试包装。这些是接入路径，不是必须全部安装的三个依赖。保留的抽取模板应包括：字段定义、数据来源、缺失值规则、禁止猜测、输出 schema。策略指南应注明模型能力、嵌套复杂度、重试上限、失败返回方式。

1. **oneOf 扩展**：把每个分支分别验证，必须恰好一个通过。测试零个、一个、两个分支通过三种情况。
2. **Schema 版本差异**：比较增加必填字段、改变类型、移除字段、放宽范围；是否破坏兼容取决于生产者/消费者方向，不能机械地把“删必填字段”都归为同一类。
3. **解码模拟器**：用 100 个候选 token 展示逐步 mask。跟踪嵌套/字符串状态，说明仍未达到通用 JSON Schema 解码器。
4. **50 条商品抽取评估**：分别算格式通过率、字段准确率、整条完全匹配率。把缺失、否定、促销价/原价等情况加入数据。
5. **字段不确定性**：多次抽取的一致性是信号，不是校准概率。用人工标注评估哪些置信范围值得复核。

自测：一个 JSON 能 loads 成功，为什么还可能不合格？因为解析只检查语法，不检查字段合同和事实。若你能自己设计有效/无效/事实错误三类输入，并准确指出失败层次，本课就学会了。


## 本次验证记录（2026-11-01）

完整标准库demo运行成功，并实测发现price=True被原number分支接受。补充JSON例子输出348。

---

[返回学习导航](<00：Phase11学习导航与验证说明.md>) · [基础概念预备课](<00A：LLM应用入门预备课——API、JSON、向量与状态.md>)

[上一课](<02：少样本与推理策略——Few-shot、CoT、投票与思维树.md>) · [下一课](<04：嵌入与向量检索——让文本成为可比较的数字.md>)

> 来源及许可：原课代码与文档来自 Rohit Ghumare 的 ai-engineering-from-scratch，MIT License，Copyright (c) 2026 Rohit Ghumare。中文讲解及教学纠错为补充；本章导航附完整许可声明。
