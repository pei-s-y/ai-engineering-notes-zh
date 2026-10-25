# 06：指令微调SFT——聊天格式、响应Mask与真实梯度

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/06-instruction-tuning-sft) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/06-instruction-tuning-sft/docs/en.md)。原题：Instruction Tuning (SFT)。

## 一、这一课解决什么问题？

SFT用指令/回答样本继续训练模型，让它更倾向按任务与角色响应。不是把随机模型稍微跑几遍就变助手；原课sft_train虽算出dlogits，却未反传到参数，而是给FFN加随机噪声，因此不是正确SFT。本篇保留原课代码用于辨析，不能把输出当学习证据。

## 二、核心概念与推导

### 2.1 SFT和预训练的关系
两者都可用下一token交叉熵，数据组织与监督位置不同。SFT常调整行为/风格，也可能学习新信息或改变既有能力，原文“绝不教新知识”过于绝对。base模型也可能通过上下文回答，不能说它完全没有回答概念。历史Alpaca/Llama数据量和费用只作来源背景。

### 2.2 三种格式
Alpaca为instruction/input/output，ShareGPT为多轮conversations，ChatML为角色边界串。格式字段名字不等于模型实际特殊ID；训练/推理必须同一chat template，多轮要明确哪些assistant turn监督。特殊标记不是所有模型微调时才随手添加。

### 2.3 Mask怎样对齐
输入tokens[:-1]，目标tokens[1:]，loss mask也取mask[1:]。只将目标为回答的CE计入分子，以有效回答token数作分母，不以整条长度稀释。系统/user文字仍影响回答计算并接收经上下文路径传播的参数梯度；“不直接监督提示词”不是detach提示计算。

### 2.4 超参和遗忘
较低LR/短训练/通用数据混合是候选，不能预设2e-5或最多3epoch通吃所有参数规模。保留领域任务与通用能力的独立验证集；raw text loss变化只是探针，不等于全面能力评测。降低loss也不是遗忘，不能使用绝对变化把改善当损伤。

### 2.5 原代码的真实性
原Use It直接初始化随机MiniGPT，不是加载预训练权重。dlogits算完没用到更新，随机噪声无视数据方向，不能据此观察SFT效果、最佳LR或灾难遗忘。真正SFT需要先完成上一课真实反传链路。

### 2.9 补充算例与理解检查

tokens=[INST,题干,RESP,答案A,答案B]，完整mask=[0,0,0,1,1]；input=[INST,题干,RESP,答案A]，target=[题干,RESP,答案A,答案B]，监督mask=[0,0,1,1]。这样RESP位置的logits负责预测答案A。若误用mask[:-1]，就漏监督第一个答案词而错位。

### 原课概念部分的公式与结构示例

~~~json
{
  "system": "You are a helpful assistant.",
  "user": "What is the capital of France?",
  "assistant": "The capital of France is Paris."
}
~~~

~~~json
{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "The European Central Bank raised interest rates...",
  "output": "The ECB increased rates by 25 basis points..."
}
~~~

~~~json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What causes tides?"},
    {"from": "gpt", "value": "Tides are caused by the gravitational pull of the Moon..."},
    {"from": "human", "value": "How often do they occur?"},
    {"from": "gpt", "value": "Most coastal areas experience two high tides and two low tides per day..."}
  ]
}
~~~

~~~text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
~~~

~~~text
Tokens:    [SYS] You are helpful [USER] What is the capital? [ASST] Paris is the capital [EOS]
Loss mask:   0    0    0     0      0     0   0  0     0       1     1    1   1     1      1
~~~

## 三、操作前准备

源码需NumPy及第04课MiniGPT类，源码import路径依赖脚本__file__，在交互笔记直接复制会未定义。可以先在同一练习会话定义上一课类，跳过该路径导入；不要把本笔记文件路径当可执行脚本路径。真实训练额外需自动微分框架或正确解析反传，本次不安装/训练。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 指令样本

8组示例含事实、数学、摘要等，作为格式样本，不足以训练出通用助手。诗歌内容也需人工质量审查，不能把少量手写例子当高质量标准。

~~~python
import numpy as np

INSTRUCTION_DATA = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Explain gravity in one sentence.",
        "response": "Gravity is the force that attracts objects with mass toward each other."
    },
    {
        "instruction": "Write a haiku about the ocean.",
        "response": "Waves crash on the shore, salt and foam beneath the sun, endless blue expanse."
    },
    {
        "instruction": "What is 15 multiplied by 7?",
        "response": "15 multiplied by 7 is 105."
    },
    {
        "instruction": "Name three programming languages.",
        "response": "Three programming languages are Python, Rust, and TypeScript."
    },
    {
        "instruction": "Summarize photosynthesis.",
        "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."
    },
    {
        "instruction": "What year did World War II end?",
        "response": "World War II ended in 1945."
    },
    {
        "instruction": "Define machine learning.",
        "response": "Machine learning is a field where algorithms learn patterns from data to make predictions."
    },
]
~~~

### 4.2 角色标记与loss mask

INST_START/END/RESP_START固定253/254/255，字节被min压到vocab_size-4。默认有效UTF8不使用253～255，但更小vocab会丢信息且固定特殊ID越界；真实系统不要用clamp替代词表。没有EOS/响应终止标记，不能学完整停止协议。

~~~python
SPECIAL_TOKENS = {
    "INST_START": 253,
    "INST_END": 254,
    "RESP_START": 255,
}


# 角色ID固定且对普通字节clamp，只作教学格式。
def tokenize_instruction_pair(instruction, response, vocab_size=256):
    inst_tokens = list(instruction.encode("utf-8"))
    resp_tokens = list(response.encode("utf-8"))

    inst_tokens = [min(t, vocab_size - 4) for t in inst_tokens]
    resp_tokens = [min(t, vocab_size - 4) for t in resp_tokens]

    tokens = (
        [SPECIAL_TOKENS["INST_START"]]
        + inst_tokens
        + [SPECIAL_TOKENS["INST_END"]]
        + [SPECIAL_TOKENS["RESP_START"]]
        + resp_tokens
    )

    return tokens


# 标出响应token，训练时要与target一起右移。
def create_loss_mask(tokens):
    mask = np.zeros(len(tokens), dtype=np.float32)
    in_response = False

    for i, token in enumerate(tokens):
        if token == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            continue
        if in_response:
            mask[i] = 1.0

    return mask
~~~

### 4.3 只平均响应位置CE

用稳定log-softmax、按mask求和并除有效token数。无响应返回0是占位行为，应跳过并记录，而非把它算为零loss优秀样本。mask需要与labels同样右移并处理PAD。

~~~python
# 分母为有效响应token数；无响应应记录并跳过。
def masked_cross_entropy_loss(logits, targets, loss_mask):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    per_token_loss = -log_softmax[np.arange(len(targets_flat)), targets_flat]

    masked_loss = per_token_loss * mask_flat
    num_response_tokens = mask_flat.sum()
    if num_response_tokens == 0:
        return 0.0
    loss = masked_loss.sum() / num_response_tokens

    return loss
~~~

### 4.4 所谓训练：噪声更新

代码计算dlogits后没有链式反传，只给FFN矩阵/bias加随机数。即使参数变化也不是数据驱动梯度！截断前seq_len可能整段删掉响应，仍加噪声。把这个循环当真实SFT会使后续实验失去意义。

~~~python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, Embedding


# 严重边界：原版未用dlogits反传，而是随机扰动FFN，不是真SFT。
def sft_train(model, dataset, num_epochs=2, lr=2e-5, seq_len=64):
    formatted_data = []
    for example in dataset:
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        formatted_data.append((tokens, mask))

    print(f"SFT Training: {len(formatted_data)} examples, {num_epochs} epochs, lr={lr}")
    print(f"Total tokens: {sum(len(t) for t, _ in formatted_data):,}")
    print()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(formatted_data))

        for idx in indices:
            tokens, mask = formatted_data[idx]

            if len(tokens) < 3:
                continue
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
                mask = mask[:seq_len]

            input_ids = np.array(tokens[:-1]).reshape(1, -1)
            target_ids = np.array(tokens[1:]).reshape(1, -1)
            loss_mask = np.array(mask[1:]).reshape(1, -1)

            logits = model.forward(input_ids)
            loss = masked_cross_entropy_loss(logits, target_ids, loss_mask)

            batch_size, s_len, v_size = logits.shape
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            dlogits = probs.copy()
            dlogits[np.arange(batch_size)[:, None], np.arange(s_len), target_ids] -= 1.0

            mask_expanded = loss_mask[:, :, np.newaxis]
            num_resp = loss_mask.sum()
            if num_resp > 0:
                dlogits = dlogits * mask_expanded / num_resp

            for block in model.blocks:
                block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 -= lr * np.random.randn(*block.ffn.W2.shape) * 0.01
                block.ffn.b1 -= lr * np.random.randn(*block.ffn.b1.shape) * 0.01
                block.ffn.b2 -= lr * np.random.randn(*block.ffn.b2.shape) * 0.01

            epoch_loss += loss
            num_batches += 1
            losses.append(loss)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    return model, losses
~~~

### 4.5 生成并对比

generate_response用温度分布采样，无EOS终止。evaluate只保留<128的生成ID，中文/emoji字节被丢，不能用于多语言质量。打印几个回答不是held-out准确率。

~~~python
def generate_response(model, prompt_tokens, max_new_tokens=50, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / max(temperature, 1e-8)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))

    return tokens


def evaluate_instruction_following(model, instructions):
    print("Evaluating instruction following:")
    print("-" * 50)

    for instruction in instructions:
        tokens = (
            [SPECIAL_TOKENS["INST_START"]]
            + [min(t, 252) for t in list(instruction.encode("utf-8"))]
            + [SPECIAL_TOKENS["INST_END"]]
            + [SPECIAL_TOKENS["RESP_START"]]
        )

        output = generate_response(model, tokens, max_new_tokens=30, temperature=0.6)
        response_start = len(tokens)
        response_tokens = output[response_start:]
        response_bytes = bytes([t for t in response_tokens if t < 128])
        response_text = response_bytes.decode("utf-8", errors="replace")

        print(f"  Q: {instruction}")
        print(f"  A: {response_text[:80]}")
        print()
~~~

### 4.6 遗忘探针

measure_forgetting算窗口CE；短文本可能没有窗口而返回0，末窗口range也可能遗漏边界。应报告有效token/窗口数，无样本返回NA或报错。固定留出集比较前后，不用随机初始化模型证明预训练知识被遗忘。

~~~python
# 必须报告窗口数，原版短文本返回0不表示能力好。
def measure_forgetting(model, test_text, seq_len=64):
    tokens = np.array(list(test_text.encode("utf-8")[:512]))

    total_loss = 0.0
    num_windows = 0

    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        input_ids = tokens[start:start + seq_len].reshape(1, -1)
        target_ids = tokens[start + 1:start + seq_len + 1].reshape(1, -1)

        logits = model.forward(input_ids)

        batch, s_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        max_logits = logits_flat.max(axis=-1, keepdims=True)
        log_softmax = logits_flat - max_logits - np.log(
            np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
        )

        loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
        total_loss += loss
        num_windows += 1

    return total_loss / max(num_windows, 1)
~~~

## 五、把代码运行起来：补充入口与检查

无需NumPy先手工检查监督对齐，补充标准库示意：
~~~python
tokens = [253, 65, 254, 255, 66, 67]
mask = [0, 0, 0, 0, 1, 1]
inputs, targets, target_mask = tokens[:-1], tokens[1:], mask[1:]
assert [t for t, m in zip(targets, target_mask) if m] == [66, 67]
print(list(zip(inputs, targets, target_mask)))
~~~
若已用PyTorch实现可训练模型，真实masked loss更新骨架如下（补充，不是原NumPy类可直接调用）：
~~~python
# logits = torch_model(input_ids)        # (B,T,V)，真正torch.Tensor
# per_token = torch.nn.functional.cross_entropy(
#     logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none"
# ).reshape_as(target_ids)
# valid = loss_mask.sum()
# if valid > 0:
#     loss = (per_token * loss_mask).sum() / valid
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
~~~
这里要求torch_model与optimizer真连接同一参数；不能把注释取消后给NumPy数组直接backward。先检查真实梯度及前后参数，再用训练/验证集检验效果。本次只测标准库对齐，不宣称训练。

## 六、Use It：使用现成工具

原Full SFT Pipeline Demo会打印pre/post raw loss、噪声“训练”曲线和生成回答。请把它当格式与统计流程示例，不当SFT基线。每个LR对照从同一个预训练checkpoint/优化器初态开始，保留base副本，否则顺序训练把先前更新混入对照。

~~~python
if __name__ == "__main__":
    np.random.seed(42)

    test_text = """The transformer architecture processes sequences through self-attention.
Each layer applies multi-head attention followed by a feedforward network.
Residual connections and layer normalization stabilize deep networks.
The model learns to predict the next token given all previous tokens."""

    print("=" * 70)
    print("INSTRUCTION TUNING (SFT) DEMO")
    print("=" * 70)
    print()

    model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Config: 4 layers, 4 heads, 128 dims (mini GPT from Lesson 04)")
    print()

    print("PRE-SFT: Measuring base model loss on raw text")
    base_loss = measure_forgetting(model, test_text)
    print(f"  Base model loss: {base_loss:.4f}")
    print()

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)

    model, losses = sft_train(
        model, INSTRUCTION_DATA, num_epochs=3, lr=2e-5, seq_len=128
    )

    print()
    print("POST-SFT: Measuring fine-tuned model loss on raw text")
    sft_loss = measure_forgetting(model, test_text)
    print(f"  SFT model loss: {sft_loss:.4f}")
    print(f"  Change: {((sft_loss - base_loss) / base_loss * 100):+.1f}%")
    if abs(sft_loss - base_loss) / base_loss < 0.15:
        print("  Minimal forgetting (< 15% change)")
    else:
        print("  Significant forgetting detected")
    print()

    print("=" * 70)
    print("INSTRUCTION FOLLOWING EVALUATION")
    print("=" * 70)
    print()

    test_instructions = [
        "What is the capital of France?",
        "Name a programming language.",
        "Define gravity.",
    ]
    evaluate_instruction_following(model, test_instructions)

    print("=" * 70)
    print("DATA FORMAT EXAMPLES")
    print("=" * 70)
    print()

    for i, example in enumerate(INSTRUCTION_DATA[:3]):
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        resp_count = int(mask.sum())
        total_count = len(tokens)
        print(f"  Example {i + 1}: {total_count} tokens, {resp_count} response tokens ({resp_count/total_count:.0%} of sequence)")
        print(f"    Instruction: {example['instruction']}")
        print(f"    Response: {example['response']}")
        print()

    print("=" * 70)
    print("TRAINING LOSS CURVE")
    print("=" * 70)
    print()

    if losses:
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"  Steps {i:3d}-{i + len(chunk) - 1:3d}: avg loss = {avg:.4f}")
~~~

## 七、失败模式、原文边界与工程注意

1. 噪声更新不是梯度，不能通过调LR修好。
2. mask只控制直接目标loss，不切断提示上下文的学习路径。
3. 截断后有效响应为0要跳过，记录每类保留率。
4. 原输出滤非ASCII会隐藏错误、破坏多语言，不是真decoder。
5. raw loss前后无窗口或绝对改善不能判遗忘。
6. 角色协议、EOS、PAD、response-only/全序列监督需明确，并非所有SFT都只能采用唯一mask策略。
7. 原时间/费用/“2e-5最优”不可转为你4060性能保证。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
设计指令数据：目标能力、instruction/input/response、角色模板与EOS、来源许可、去重、质量与多样性。检验label右移、assistant监督范围、PAD和截断响应比例；要求真实梯度更新与预训练起点，评估held-out任务、通用保留及无样本状态，不凭打印曲线验收。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 加system prompt，5个不同角色例子，检查prompt格式与mask，不能只换显示字符串。
2. 5%原始文本全位置监督+95%SFT响应监督，对照同初始checkpoint和token预算；真实反传后比较遗忘。
3. 统计响应token数、指令/回答比例、词表多样性，试<10或<.3过滤；这些阈值是练习超参，短正确答案不能一律当垃圾。
4. 3轮user/assistant对话，mask覆盖每轮assistant，明确是否监督角色终止符，打印逐token对齐。
5. 比较1e-4/2e-5/1e-6，同初态同数据真实训练，画train/val；不预先保证哪条曲线最快/最好，原噪声循环无法完成该目标。

## 十、术语与完成标准

SFT：监督微调；response-only loss：仅响应目标监督；chat template：会话协议；loss mask：有效监督位置；catastrophic forgetting：先前能力退化；data mixing：混合训练来源。

完成检查：

- 能手写右移后的mask。
- 能说明噪声不是真训练。
- 能区分随机base与预训练checkpoint。
- 能设计响应截断和遗忘检查。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行响应监督对齐小例，shift后监督目标为[66,67]。未执行NumPy随机扰动训练或真实SFT，不能据此声称指令跟随能力提升。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

原课 Build It 共 6 个分段，相关代码和示例均嵌入本篇；Use It 的代码也保留。正文必要的补充入口会明确标为补充，不冒充原文代码。Python 代码做静态语法检查；含交互提示符、展示输出或占位的片段单独标识，不假装可以直接作为完整脚本运行。具体执行结果以第五节及章节导航的验证记录为准；未执行的训练、下载、外部模型/API 结果不写成实测。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[05：分布式训练——DP、TP、PP、FSDP与ZeRO预算]] · 下一课：[[07：RLHF——奖励模型、Bradley-Terry、PPO与KL约束]]
