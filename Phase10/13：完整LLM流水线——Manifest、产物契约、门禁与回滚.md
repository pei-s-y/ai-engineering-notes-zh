> 路径约定：本文的 `~/projects/ai-engineering-from-scratch` 是通用示例，不表示仓库已位于该处。执行命令前，请将它替换为实际克隆目录；以 `phases/` 开头的路径均相对于课程仓库根目录。

# 13：完整LLM流水线——Manifest、产物契约、门禁与回滚

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/13-building-complete-llm-pipeline) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/13-building-complete-llm-pipeline/docs/en.md)。原题：Building a Complete LLM Pipeline。

## 零、初学者先读：怎样让许多步骤成为能追溯的流程

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

有了分词、数据、训练、评价和部署，下一问题是知道每份结果从哪来。流水线要记录依赖、配置和产物，避免把不同模型的数据和结果接错。

### 0.2 本课术语先翻成白话

- artifact=一个步骤产出的文件或对象

- hash=按内容算的指纹

- manifest=记录步骤、版本与产物的清单

- DAG=有方向且无循环的依赖图

- gate=达标才继续的门槛

- rollback=恢复较早版本。

### 0.3 用小数字走一遍

假设A训练分词器，B用它生成数据，C训练模型。若A的词表编号改变，B的旧ID数据未必还能接新的C；记录A→B→C及每个产物指纹才能发现这种不匹配。文件名都叫latest并不能保证对应。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先读STAGES里的依赖列表，再看ArtifactStore.put怎样按内容生成hash，最后看run与gate。仓库用字节串假装模型产物、用固定数值假装评估结果；运行12阶段不是训练12个模型。

前置阅读：[第03课](<03：预训练数据流水线——清洗、MinHash去重、打包与批次.md>)；[第04课](<04：从零搭建Mini GPT——因果注意力、前向计算与真实训练.md>)；[第06课](<06：指令微调SFT——聊天格式、响应Mask与真实梯度.md>)；[第10课](<10：LLM评估——自定义测试、EM与F1、PPL和Elo.md>)；[第12课](<12：推理优化——KV缓存、连续批处理、分页与前缀复用.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** hash一样能证明模型质量一样好吗？

**参考答案：** 它证明比较口径下内容一致，不是质量评分。

**问题2：** 门槛通过是否保证可上线？

**参考答案：** 还要保证指标可信、步骤成功、产物匹配和实际部署验证。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

把前12课串成可追踪实验，不是在笔记本训练70B。重点是输入/输出契约、DAG、hash、预算、评价和恢复。原课前面有未完成训练器，所以这里只能把它们当待替换阶段；编排成功不等于真实训练/部署成功。

## 二、核心概念与推导

### 2.1 十二环节及依赖
词表/分词器→数据→预训练→规模配置→SFT→PPO或DPO→可选CAI/GRPO→评估→量化→推理。课程顺序不是唯一生产DAG：并行recipe常先于预训练，PPO/DPO可选支路，不一定两条都做后自动合并；两个policy checkpoint无法仅拼hash就融合权重。

### 2.2 Manifest与类型
记录代码commit、环境、seed、数据/tokenizer/config/checkpoint、step与optimizer、gate和成本。Tokenizer/Dataset/Checkpoint/Policy/Eval/Quant/ServerSpec需要不同schema和兼容性检查；Python字符串stage_type不会自动提供编译期类型安全。

### 2.3 Hash与复现
SHA256标识某份具体字节，取回需重算核验，文件名latest不是身份。多输入DAG应记录有序/规范化父hash集合，而非线性一根链。训练数值非确定时输出字节/hash会变，应区分“相同产物完整性”和“等效指标复现”，不能要求不同权重却同hash。

### 2.4 Gate与风险
指标门槛先定义，失败HOLD，可由具名审核记录override。数字不替代高风险人工审查。safety_refusal_rate<=.05只有定义为良性请求误拒率才可能合理，不能对危险请求拒绝率要求越低越好。量化/最终服务也要重新评，不只量化前模型。

### 2.5 成本/回滚
预训练近似6ND FLOPs，再除有效FLOPs/s和3600得GPU-hours；MFU/价格是题设。budget应事前预估、阶段前和执行中约束。依赖失效重跑后代，同时重算依赖最终产物的gate；训练重启还需optimizer、调度、RNG、数据位置，不只weights。

tracker与artifact store职责不同，可本地或托管；并非强制上云。原文费用/时间和2026统一recipe是来源描述，不当当前唯一方案。

### 2.9 补充算例与理解检查

若tokenizer hash改变，旧数据token ID和模型权重可能都不兼容，整条后代需重新验证；若只改评价rubric，模型字节不用重训，但发布结论要重算。7B×2T×6=8.4e22 FLOPs，按989TFLOPs、MFU=.4的题设再算小时，不把报价当实时价格。

### 原课概念部分的公式与结构示例

~~~text
pipeline_version: 1.2.3
seed: 42
git_commit: a1b2c3d4
stages:
  01_tokenizer:
    recipe: bpe_32k
    input_hash: sha256:...
    output_hash: sha256:...
    wall_clock_sec: 3600
    cost_usd: 12
~~~

~~~text
gates:
  mmlu:      >= baseline + 0.5   # no regression
  humaneval: >= baseline + 1.0
  truthfulqa: >= baseline         # no drop
  safety_refusal_rate: <= 0.05
  kl_from_reference: <= 25.0
  cost_total_usd: <= 50000
~~~

## 三、操作前准备

原仓库main.py已核对，是标准库占位编排、内存store、固定成本/指标。它不访问网络、不真实训练，也不会按文档所说写manifest.out.yaml；run打印JSON，gate重新模拟一次而不是读上次文件。下面内嵌其核心，不需要脚本附件。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 读懂原课编排核心

原文此节没有代码，只引用仓库。下面第五节保留仓库核心类和函数。注意实现并不具备文案声称的完整DAG校验、缓存跳过、CostTracker独立类或跨进程持久化；学习时逐项核对。



## 五、把代码运行起来：补充入口与检查

以下来自原仓库code/main.py的核心部分，保留以便只读本笔记学习；不是en.md里原有代码，也不是本次实际模型训练。

~~~python
"""End-to-end LLM pipeline orchestrator.

Twelve stages wired as a DAG. Each stage is a placeholder that emits a typed
artifact with a content-addressed hash. The orchestrator resolves dependencies,
runs stages, records a manifest, and applies eval gates before shipping.

No network, no GPUs, stdlib only. Replace each stage's `run` with the real
training script from the corresponding Phase 10 lesson.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict


STAGES = [
    ("01_tokenizer_vocab",         [],                     "tokenizer"),
    ("02_tokenizer_trained",       ["01_tokenizer_vocab"], "tokenizer"),
    ("03_dataset_sharded",         ["02_tokenizer_trained"], "dataset"),
    ("04_pretrained_base",         ["03_dataset_sharded"], "checkpoint"),
    ("05_scaled_recipe",           ["04_pretrained_base"], "checkpoint"),
    ("06_sft_checkpoint",          ["05_scaled_recipe"],   "sft_model"),
    ("07_reward_ppo_policy",       ["06_sft_checkpoint"],  "policy"),
    ("08_dpo_policy",              ["06_sft_checkpoint"],  "policy"),
    ("09_cai_grpo_policy",         ["07_reward_ppo_policy", "08_dpo_policy"], "policy"),
    ("10_eval_report",             ["09_cai_grpo_policy"], "eval_report"),
    ("11_quantized_weights",       ["09_cai_grpo_policy"], "quantized_model"),
    ("12_inference_server",        ["11_quantized_weights"], "server_spec"),
]


DEFAULT_GATES = {
    "mmlu":                 {"op": ">=", "value": 65.0},
    "humaneval":            {"op": ">=", "value": 40.0},
    "truthfulqa":           {"op": ">=", "value": 50.0},
    "safety_refusal_rate":  {"op": "<=", "value": 0.05},
    "kl_from_reference":    {"op": "<=", "value": 25.0},
    "cost_total_usd":       {"op": "<=", "value": 50000.0},
}


@dataclass
class StageRecord:
    name: str
    stage_type: str
    input_hashes: list[str] = field(default_factory=list)
    output_hash: str = ""
    wall_clock_sec: float = 0.0
    cost_usd: float = 0.0
    status: str = "pending"


@dataclass
class Manifest:
    pipeline_version: str = "1.0.0"
    seed: int = 42
    git_commit: str = "unknown"
    stages: list[StageRecord] = field(default_factory=list)
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES))
    eval_metrics: dict = field(default_factory=dict)
    budget_usd: float = 50000.0
    total_cost_usd: float = 0.0
    shippable: bool = False


class ArtifactStore:
    """In-memory stand-in for an S3 / R2 / GCS bucket addressed by SHA-256."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, blob: bytes) -> str:
        h = hashlib.sha256(blob).hexdigest()
        self._store[h] = blob
        return h

    def get(self, h: str) -> bytes:
        return self._store[h]

    def has(self, h: str) -> bool:
        return h in self._store

    def __len__(self) -> int:
        return len(self._store)


def simulate_stage(name: str, stage_type: str, inputs: list[str], seed: int) -> tuple[bytes, float, float]:
    """Placeholder: emits a deterministic blob, a wall-clock, and a cost.
    Swap this for the real Phase 10 lesson scripts."""

    payload = {
        "stage": name,
        "type": stage_type,
        "inputs": inputs,
        "seed": seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")

    cost_table = {
        "tokenizer":        (60,    5),
        "dataset":          (1800,  50),
        "checkpoint":       (7200,  400),
        "sft_model":        (3600,  150),
        "policy":           (5400,  300),
        "eval_report":      (600,   20),
        "quantized_model":  (300,   10),
        "server_spec":      (30,    1),
    }
    wall, cost = cost_table.get(stage_type, (100, 5))
    return blob, float(wall), float(cost)


def plan(manifest: Manifest) -> str:
    """Validate the manifest, print the DAG, compute the cost estimate."""

    lines = ["PLAN"]
    lines.append("=" * 60)
    lines.append(f"pipeline_version  : {manifest.pipeline_version}")
    lines.append(f"seed              : {manifest.seed}")
    lines.append(f"budget_usd        : ${manifest.budget_usd:,.0f}")
    lines.append("")
    lines.append("dag:")
    for name, deps, stage_type in STAGES:
        dep_str = ", ".join(deps) if deps else "-"
        lines.append(f"  {name:28s}  [{stage_type}]  <- {dep_str}")
    lines.append("")
    lines.append("gates:")
    for metric, gate in manifest.gates.items():
        lines.append(f"  {metric:22s}  {gate['op']}  {gate['value']}")
    return "\n".join(lines)


def run(manifest: Manifest, store: ArtifactStore, injected_eval: dict | None = None) -> Manifest:
    """Execute stages in DAG order. Halts on budget or hash failure."""

    name_to_hash: dict[str, str] = {}

    for name, deps, stage_type in STAGES:
        input_hashes = [name_to_hash[d] for d in deps]
        for h in input_hashes:
            if not store.has(h):
                raise RuntimeError(f"input hash missing for stage {name}: {h[:12]}...")

        blob, wall, cost = simulate_stage(name, stage_type, input_hashes, manifest.seed)
        output_hash = store.put(blob)

        manifest.total_cost_usd += cost
        if manifest.total_cost_usd > manifest.budget_usd:
            record = StageRecord(
                name=name, stage_type=stage_type,
                input_hashes=input_hashes, output_hash="",
                wall_clock_sec=wall, cost_usd=cost,
                status="halted_over_budget",
            )
            manifest.stages.append(record)
            return manifest

        record = StageRecord(
            name=name, stage_type=stage_type,
            input_hashes=input_hashes, output_hash=output_hash,
            wall_clock_sec=wall, cost_usd=cost, status="ok",
        )
        manifest.stages.append(record)
        name_to_hash[name] = output_hash

    manifest.eval_metrics = injected_eval if injected_eval is not None else {
        "mmlu": 68.4,
        "humaneval": 42.1,
        "truthfulqa": 53.7,
        "safety_refusal_rate": 0.03,
        "kl_from_reference": 18.5,
        "cost_total_usd": manifest.total_cost_usd,
    }
    return manifest


def gate(manifest: Manifest) -> tuple[bool, list[str]]:
    """Apply each gate. Return (ship?, reasons)."""

    reasons = []
    all_pass = True
    for metric, g in manifest.gates.items():
        value = manifest.eval_metrics.get(metric)
        if value is None:
            reasons.append(f"HOLD: missing metric {metric}")
            all_pass = False
            continue

        passed = (value >= g["value"]) if g["op"] == ">=" else (value <= g["value"])
        if not passed:
            reasons.append(
                f"HOLD: {metric}={value} fails gate {g['op']} {g['value']}"
            )
            all_pass = False
        else:
            reasons.append(f"PASS: {metric}={value} {g['op']} {g['value']}")

    manifest.shippable = all_pass
    return all_pass, reasons


def manifest_to_json(manifest: Manifest) -> str:
    d = asdict(manifest)
    return json.dumps(d, indent=2, sort_keys=True)



~~~

定义后可运行占位门禁：
~~~python
manifest = Manifest()
store = ArtifactStore()
run(manifest, store)
assert len(manifest.stages) == 12
ok, reasons = gate(manifest)
assert ok
manifest.eval_metrics["mmlu"] = 0.0
ok, reasons = gate(manifest)
assert not ok
print("占位阶段数", len(manifest.stages), "产物", len(store), "最终门禁", ok)
~~~
预期12阶段和12份内存产物，第一次PASS、改低指标后HOLD。全部eval数值是硬编码模拟；没有12项真实实验。

实现边界：plan不累计成本/检查完整DAG；run只验证hash存在，不对store字节重算、不查输出期望、不缓存跳过；预算在产物写入后才判；gate未知op按<=处理，没有完整状态成功校验。实际应用必须补足，不能只照名字当已实现。

## 六、Use It：使用现成工具

在WSL已有仓库运行原命令的正确位置：
~~~bash
cd ~/projects/ai-engineering-from-scratch/phases/10-llms-from-scratch/13-building-complete-llm-pipeline
python3 code/main.py plan
python3 code/main.py run
python3 code/main.py gate
~~~
plan打印计划，run打印manifest JSON，gate本进程重跑模拟后打印SHIP/HOLD，默认无参数还展示失败gate。文档说读写manifest.out.yaml不符实际，不要到磁盘找不存在的文件。本次只安全测试内存核心，不创建云资源或部署。

~~~text
python code/main.py plan    # validate manifest, compute cost estimate, print DAG
python code/main.py run     # execute stages, writing to manifest.out.yaml
python code/main.py gate    # read manifest.out.yaml, apply eval gates, ship-or-hold
~~~

## 七、失败模式、原文边界与工程注意

1. hash存在不等于字节未损坏，要重新算摘要。
2. stage_type字符串不自动验证输入schema/模型兼容。
3. 原run没有缓存/断点重启，文档中的rollback是待实现目标。
4. gate不能仅比较硬编码eval；失败阶段/空报告/NaN必须fail closed。
5. 数字门槛仍需人工业务审核，尤其拒绝率需明确分母和方向。
6. DAG双policy不是自动合并模型；变体可跳过某些对齐。
7. 量化/服务更改要回归最终部署产物，不能旧report给新权重放行。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
检查Manifest包含代码/环境/seed/数据/tokenizer/权重/optimizer与父hash；类型、内容重校验、预算preflight、失败持久化、缓存命中条件、gate和override均明确。分别验证量化前后及实际服务，列回滚后代、checkpoint恢复信息和责任人；占位跑通只叫plumbing通过。
~~~

## 九、Exercises：保留原课练习与验收方法

1. concurrent.futures并行07/08，规范父hash排序；先明确09到底选哪policy或怎样合并，不能用两个hash代替权重融合。
2. 污染gate查精确/13gram重叠，hash本身不能测文本重合；.1%为题设，注明样本/分母与误报。
3. 7B/2T、6ND、MFU40%、989TFLOPs、$2.5/GPU-hour算理论，附硬件利用率假设，不当当前报价。
4. 模拟09失败，仅重跑09—12且缓存01—08；需先自己补真实缓存，因为原run不会skip。
5. 加OpenTelemetry每阶段span，先本地/脱敏，不上传私有训练文本；记录trace_id、状态、参数量、token和成本。

## 十、术语与完成标准

Manifest：可追溯实验清单；artifact：阶段产物；content-addressed：按内容摘要寻址；gate：放行条件；DAG：有向无环依赖；MFU：有效模型FLOPs利用率；rollback：恢复有效版本。

完成检查：

- 能解释hash与等效复现区别。
- 能让模拟gate因错误指标HOLD。
- 能发现源码不写YAML/不支持resume。
- 能列最终部署重新评估的原因。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行12个内存占位阶段，12份产物；默认门禁PASS，把mmlu改0后HOLD；模拟成本1941（硬编码计价单位美元），没有实际费用和真实训练。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 1 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[12：推理优化——KV缓存、连续批处理、分页与前缀复用]] · 下一课：[[14：开源模型架构解析——RMSNorm、RoPE、GQA、MoE与配置预算]]
