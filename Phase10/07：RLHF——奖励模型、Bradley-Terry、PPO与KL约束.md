# 07：RLHF——奖励模型、Bradley-Terry、PPO与KL约束

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/10-llms-from-scratch/07-rlhf) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/10-llms-from-scratch/07-rlhf/docs/en.md)。原题：RLHF: Reward Model + PPO。

## 零、初学者先读：示范答案不够时，怎样利用哪个好一点的比较

> 2026-10-31 新增的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。需要补数组、概率、梯度或attention时，打开[初学者预备课](<00A：初学者预备课——从数组到语言模型.md>)；它包含本章反复使用的基础解释，不依赖英文原文。

### 0.1 先理解这一课在做什么

有些问题没有唯一标准文本，但人能比较两个回答哪个好。RLHF先让奖励模型学习这种比较，再用它的评分帮助生成模型改变回答倾向。评分器与生成器是不同角色。

### 0.2 本课术语先翻成白话

- 偏好对=同一问题下较好/较差两个回答

- reward=评分

- policy=生成回答的概率规则

- reference=冻结的参照模型

- rollout=按当前策略生成一批样本

- PPO=限制每批更新幅度的策略优化方法。

### 0.3 用小数字走一遍

奖励模型给好答案2分、差答案1分，差值1。胜出概率σ(1)=1/(1+exp(-1))≈.731；若两者均0分，概率.5。损失希望好答案比差答案更高，而不是要求好答案必须等于某个绝对分数。两个分数同时加100，比较概率不变。

以上是用于理解的算例，不是新增的模型训练或性能实测。读下面的公式时，把每个符号先对应回这里的对象。

### 0.4 怎样连接到正文代码

先读bradley_terry_loss，理解输入只是两个分数；再看正确线性评分头的小更新。之后读PPO：old-policy是本批样本产生时的策略，用于概率比；reference是较长期锚点，用于约束漂移。二者不要因为都是旧模型就混为一谈。

前置阅读：[第06课](<06：指令微调SFT——聊天格式、响应Mask与真实梯度.md>)。先确定每个函数接收什么、返回什么，再追内部计算；对下面的复杂表达式，使用预备课中的“中文含义→符号→小数字→shape”四步。

### 0.5 理解检查（含参考答案）

**问题1：** 奖励提高是否保证人更满意？

**参考答案：** 不保证，模型可能钻评分规则的漏洞。

**问题2：** KL约束做什么？

**参考答案：** 度量并约束策略分布相对参照的变化，不能保证堵住所有漏洞。

能解释这个小例后，继续阅读下文“核心概念与推导”。第一次遇到版本纠错和集群预算时，可以先标记问题，理解基础流程后再回来逐条核对；所有原课内容仍保留。

---

## 一、这一课解决什么问题？

偏好对表达同一prompt下哪个回答更符合准则，奖励模型学习比较，策略据此优化。原文把简洁一律当更好、KL说成自动防止reward hacking均过强；偏好有任务与人群背景。本课源码的RM和PPO更新不正确，不能用输出数字证明真实RLHF有效。

## 二、核心概念与推导

### 2.1 三阶段与模型角色
先SFT；再对prompt/chosen/rejected训练RM；然后policy采样并优化。policy可训练、reference冻结、RM通常冻结，value/critic可另头或另模型，不是必须四个同样大的独立副本。

### 2.2 BT成对损失
P(chosen胜)=σ(rw−rl)，L=−logσ(rw−rl)=softplus(−差值)。∂L/∂rw=σ(差)−1，∂L/∂rl=1−σ(差)。线性reward=h·w时，梯度=(σ(差)−1)(hw−hl)，两个分支缺一不可。所有reward加同常数不改变偏好，绝对分不是自然校准分数。

### 2.3 PPO两个参照
ratio=πθ(action|state)/πold(action|state)，old是采样策略，同批更新时固定；reference是KL锚点，不能混同。min(rA,clip(r)A)限制surrogate激励，优势常由回报/baseline/GAE估计。reward缩放随机噪声不是policy gradient。

### 2.4 KL与代理作弊
目标常为E[R]−βKL(πθ||πref)。全分布KL=Σp log(p/q)≥0；策略采样上的单点log比可以负，要与精确分布分开。β减轻偏移但不保证无作弊；长度、奉承、回避、格式都可能骗RM。评估需held-out、对抗与人工。

原文InstructGPT/Anthropic/Llama的样本量、胜率、成本与超参仅是源文历史背景，不作统一配方。SFT可通过筛选数据间接表达偏好，但没有同样显式的成对监督；“SFT完全无法区分好坏”并不严谨。

### 2.9 补充算例与理解检查

rw=rl=0时胜率.5、loss≈.6931；差值1时胜率≈.731、loss≈.3133。hw=[1,0]、hl=[0,1]、w=[0,0]时梯度[−.5,+.5]，下降让第一权重升、第二降；只看chosen特征漏了另一半。

### 原课概念部分的公式与结构示例

$$
\text{loss} = -\log\big(\mathop{\mathrm{sigmoid}}(\text{reward}(\text{preferred}) - \text{reward}(\text{rejected}))\big)
$$

$$
\text{maximize:} \quad \mathbb{E}[R(\text{prompt}, \text{response})] - \beta \cdot \mathrm{KL}(\text{policy} \,\|\, \text{reference})
$$

$$
\begin{aligned}
\text{ratio} &= \pi_{\text{new}}(\text{action} \mid \text{state}) / \pi_{\text{old}}(\text{action} \mid \text{state}) \\
\text{clipped\_ratio} &= \mathop{\mathrm{clip}}(\text{ratio}, 1 - \epsilon, 1 + \epsilon) \\
\text{loss} &= -\min(\text{ratio} \cdot \text{advantage}, \text{clipped\_ratio} \cdot \text{advantage})
\end{aligned}
$$

~~~text
advantage = reward(prompt, response) - baseline
~~~

## 三、操作前准备

原代码需要NumPy和第04课类；交互运行不要直接复制依赖__file__的路径导入，先定义类。原Use It新建随机“SFT”模型，未加载第06课已训练checkpoint。先跑下文标准库BT例，不下载模型。

先激活你已有的 WSL Python 虚拟环境；下文安装命令只供你按需执行，本次没有替你安装或下载模型。代码按定义顺序运行，含外部数据、预训练模型或服务调用的片段需先满足本节条件。

## 四、Build It：原课分步代码与中文讲解

### 4.1 合成偏好数据

6对手工将短回答标chosen，是一个特定评价标准，不能说任何人必选它。加入长度相近、不同事实质量的样本，避免只学长度。评价不要仍用同6训练对。

~~~python
import numpy as np

PREFERENCE_DATA = [
    {
        "prompt": "What is the capital of France?",
        "preferred": "The capital of France is Paris.",
        "rejected": "France is a country in Europe. It has many cities. The capital is Paris. Paris is known for the Eiffel Tower.",
    },
    {
        "prompt": "Explain gravity in one sentence.",
        "preferred": "Gravity is the force that attracts objects with mass toward each other.",
        "rejected": "Gravity is something that makes things fall down when you drop them.",
    },
    {
        "prompt": "What is 15 times 7?",
        "preferred": "15 times 7 is 105.",
        "rejected": "Let me think about this. 15 times 7. Well, 10 times 7 is 70, and 5 times 7 is 35, so the answer might be around 105.",
    },
    {
        "prompt": "Name three programming languages.",
        "preferred": "Python, Rust, and TypeScript.",
        "rejected": "There are many programming languages. Some popular ones include various languages like Python and others.",
    },
    {
        "prompt": "What year did World War II end?",
        "preferred": "World War II ended in 1945.",
        "rejected": "World War II was a major global conflict. It involved many countries. The war ended in the mid-1940s, specifically in 1945.",
    },
    {
        "prompt": "Define machine learning.",
        "preferred": "Machine learning is a field where algorithms learn patterns from data to make predictions without being explicitly programmed.",
        "rejected": "Machine learning is a type of AI. AI stands for artificial intelligence. Machine learning uses data to learn.",
    },
]
~~~

### 4.2 奖励模型架构

因果模型最后hidden可看前文，以scalar head打分。padding下取最后有效位；RM默认随机，不是自动复用SFT权重。奖励可学习偏好，但不是事实认证器。

~~~python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, Embedding, TransformerBlock


class RewardModel:
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4,
                 num_layers=4, max_seq_len=128, ff_dim=512):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.reward_head = np.random.randn(embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        last_hidden = x[:, -1, :]
        reward = last_hidden @ self.reward_head

        return reward
~~~

### 4.3 BT与原错误更新

loss形式表达差值，但train_reward_model只用preferred embedding+LN更新head，没有transformer同一路径hidden，也没减rejected hidden。既不是完整RM反传，也不是正确冻结特征BT梯度。np.where两边exp可都计算，仍有溢出风险。

~~~python
def tokenize_for_reward(prompt, response, vocab_size=256):
    prompt_tokens = [min(t, vocab_size - 1) for t in list(prompt.encode("utf-8"))]
    response_tokens = [min(t, vocab_size - 1) for t in list(response.encode("utf-8"))]
    return prompt_tokens + [0] + response_tokens


def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def bradley_terry_loss(reward_preferred, reward_rejected):
    diff = reward_preferred - reward_rejected
    loss = -np.log(sigmoid(diff) + 1e-8)
    return loss


# 原head梯度漏rejected且特征错配，不能当正确RM训练。
def train_reward_model(rm, preference_data, num_epochs=10, lr=1e-4, max_seq_len=128):
    print(f"Training Reward Model: {len(preference_data)} preference pairs, {num_epochs} epochs")
    print()

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        num_pairs = 0

        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            pair = preference_data[idx]

            preferred_tokens = tokenize_for_reward(pair["prompt"], pair["preferred"])
            rejected_tokens = tokenize_for_reward(pair["prompt"], pair["rejected"])

            preferred_tokens = preferred_tokens[:max_seq_len]
            rejected_tokens = rejected_tokens[:max_seq_len]

            preferred_ids = np.array(preferred_tokens).reshape(1, -1)
            rejected_ids = np.array(rejected_tokens).reshape(1, -1)

            r_preferred = rm.forward(preferred_ids)[0]
            r_rejected = rm.forward(rejected_ids)[0]

            loss = bradley_terry_loss(r_preferred, r_rejected)

            if r_preferred > r_rejected:
                epoch_correct += 1

            diff = r_preferred - r_rejected
            grad = sigmoid(diff) - 1.0

            rm.reward_head -= lr * grad * rm.ln_f.forward(
                rm.embedding.forward(preferred_ids)
            )[:, -1, :].flatten()

            epoch_loss += loss
            num_pairs += 1

        avg_loss = epoch_loss / max(num_pairs, 1)
        accuracy = epoch_correct / max(num_pairs, 1)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")

    return rm, losses, accuracies
~~~

### 4.4 简化PPO实际是随机扰动

函数没有old概率、clip surrogate、优势/value或真实反传，只用reward−βKL缩放FFN随机矩阵。KL数值计算不使这个更新变成PPO。截取prompt+生成前maxT可能把回答都删掉，RM便没看到回答。

~~~python
# 全词表分布KL，不等于单采样token的log比。
def compute_kl_divergence(policy_logits, reference_logits):
    policy_probs = np.exp(policy_logits - policy_logits.max(axis=-1, keepdims=True))
    policy_probs = policy_probs / policy_probs.sum(axis=-1, keepdims=True)
    policy_probs = np.clip(policy_probs, 1e-10, 1.0)

    ref_probs = np.exp(reference_logits - reference_logits.max(axis=-1, keepdims=True))
    ref_probs = ref_probs / ref_probs.sum(axis=-1, keepdims=True)
    ref_probs = np.clip(ref_probs, 1e-10, 1.0)

    kl = np.sum(policy_probs * np.log(policy_probs / ref_probs), axis=-1)
    return kl.mean()


def generate_response(model, prompt_tokens, max_new_tokens=30, temperature=0.8, max_seq_len=128):
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        context = np.array(tokens[-max_seq_len:]).reshape(1, -1)
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


# 复制数组，避免policy与冻结ref共享可变存储。
def copy_model_weights(source, target):
    target.embedding.token_embed = source.embedding.token_embed.copy()
    target.embedding.pos_embed = source.embedding.pos_embed.copy()
    target.ln_f.gamma = source.ln_f.gamma.copy()
    target.ln_f.beta = source.ln_f.beta.copy()
    for s_block, t_block in zip(source.blocks, target.blocks):
        t_block.attn.W_q = s_block.attn.W_q.copy()
        t_block.attn.W_k = s_block.attn.W_k.copy()
        t_block.attn.W_v = s_block.attn.W_v.copy()
        t_block.attn.W_out = s_block.attn.W_out.copy()
        t_block.ffn.W1 = s_block.ffn.W1.copy()
        t_block.ffn.W2 = s_block.ffn.W2.copy()
        t_block.ffn.b1 = s_block.ffn.b1.copy()
        t_block.ffn.b2 = s_block.ffn.b2.copy()
        t_block.ln1.gamma = s_block.ln1.gamma.copy()
        t_block.ln1.beta = s_block.ln1.beta.copy()
        t_block.ln2.gamma = s_block.ln2.gamma.copy()
        t_block.ln2.beta = s_block.ln2.beta.copy()


# 原版为奖励缩放随机噪声，无PPO真实梯度。
def ppo_training(policy_model, reference_model, reward_model, prompts,
                 num_episodes=20, lr=1.5e-5, kl_coeff=0.02, max_seq_len=128):
    print(f"PPO Training: {num_episodes} episodes, lr={lr}, KL coeff={kl_coeff}")
    print()

    rewards_history = []
    kl_history = []

    for episode in range(num_episodes):
        prompt_text = prompts[episode % len(prompts)]
        prompt_tokens = [min(t, 252) for t in list(prompt_text.encode("utf-8"))]

        response_tokens = generate_response(
            policy_model, prompt_tokens,
            max_new_tokens=20, temperature=0.8, max_seq_len=max_seq_len
        )

        response_ids = np.array(response_tokens[:max_seq_len]).reshape(1, -1)
        reward = reward_model.forward(response_ids)[0]

        policy_logits = policy_model.forward(response_ids)
        ref_logits = reference_model.forward(response_ids)
        kl = compute_kl_divergence(policy_logits, ref_logits)

        total_reward = reward - kl_coeff * kl

        rewards_history.append(float(reward))
        kl_history.append(float(kl))

        for block in policy_model.blocks:
            update_scale = lr * total_reward
            block.ffn.W1 += update_scale * np.random.randn(*block.ffn.W1.shape) * 0.01
            block.ffn.W2 += update_scale * np.random.randn(*block.ffn.W2.shape) * 0.01

        if episode % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:]) if rewards_history else 0
            avg_kl = np.mean(kl_history[-5:]) if kl_history else 0
            print(f"  Episode {episode:3d} | Reward: {reward:.4f} | KL: {kl:.4f} | "
                  f"Avg Reward: {avg_reward:.4f}")

    return policy_model, rewards_history, kl_history
~~~

### 4.5 对比代理reward

生成后用同RM评分只看代理，不证明人工质量/安全；需独立prompt、盲评与长度控制。空prompts会除零，随机seed也不代替足量统计。

~~~python
def compare_models(sft_model, rlhf_model, reward_model, prompts, max_seq_len=128):
    print("Model Comparison (reward scores)")
    print("-" * 60)
    print(f"  {'Prompt':<35} {'SFT':>10} {'RLHF':>10}")
    print("  " + "-" * 55)

    sft_total = 0.0
    rlhf_total = 0.0

    for prompt in prompts:
        prompt_tokens = [min(t, 252) for t in list(prompt.encode("utf-8"))]

        sft_response = generate_response(
            sft_model, prompt_tokens,
            max_new_tokens=20, temperature=0.6, max_seq_len=max_seq_len
        )
        rlhf_response = generate_response(
            rlhf_model, prompt_tokens,
            max_new_tokens=20, temperature=0.6, max_seq_len=max_seq_len
        )

        sft_ids = np.array(sft_response[:max_seq_len]).reshape(1, -1)
        rlhf_ids = np.array(rlhf_response[:max_seq_len]).reshape(1, -1)

        sft_reward = reward_model.forward(sft_ids)[0]
        rlhf_reward = reward_model.forward(rlhf_ids)[0]

        sft_total += sft_reward
        rlhf_total += rlhf_reward

        truncated_prompt = prompt[:33] + ".." if len(prompt) > 35 else prompt
        print(f"  {truncated_prompt:<35} {sft_reward:>10.4f} {rlhf_reward:>10.4f}")

    n = len(prompts)
    print("  " + "-" * 55)
    print(f"  {'Average':<35} {sft_total/n:>10.4f} {rlhf_total/n:>10.4f}")

    return sft_total / n, rlhf_total / n
~~~

## 五、把代码运行起来：补充入口与检查

补充冻结特征线性head的一次正确BT更新，纯标准库，不冒充完整RLHF：
~~~python
import math

def bt_linear_step(w, chosen_h, rejected_h, lr=0.1):
    delta_h = [a-b for a,b in zip(chosen_h, rejected_h)]
    margin = sum(a*b for a,b in zip(w, delta_h))
    probability = 1 / (1 + math.exp(-margin))
    gradient = [(probability-1)*x for x in delta_h]
    return [a-lr*g for a,g in zip(w, gradient)]

w = bt_linear_step([0.0,0.0], [1.0,0.0], [0.0,1.0])
assert w == [0.05,-0.05]
print(w)
~~~
完整RM需对选择训练的参数正确反传；PPO还需要rollout旧概率、响应mask、优势、KL与optimizer。参数变化不是充分验收，随机噪声也会使参数改变。

## 六、Use It：使用现成工具

Full RLHF Demo依次新建随机SFT/RM、错误head训练、噪声policy更新，最后打印比较和KL。可以理解模块接口与报表，不能当真实训练基线。若用成熟RL库，先在小可验证环境检查old/ref、reward范围、终止与真实梯度，再增加规模。

~~~python
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("RLHF PIPELINE: REWARD MODEL + PPO")
    print("=" * 70)
    print()

    print("STAGE 1: SFT Model (from Lesson 06)")
    print("-" * 40)
    sft_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    print()

    print("STAGE 2: Train Reward Model")
    print("-" * 40)
    rm = RewardModel(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )

    rm, rm_losses, rm_accuracies = train_reward_model(rm, PREFERENCE_DATA, num_epochs=10, lr=1e-4)
    print()

    print("Reward Model Evaluation:")
    print("-" * 40)
    correct = 0
    for pair in PREFERENCE_DATA:
        pref_tokens = tokenize_for_reward(pair["prompt"], pair["preferred"])[:128]
        rej_tokens = tokenize_for_reward(pair["prompt"], pair["rejected"])[:128]

        r_pref = rm.forward(np.array(pref_tokens).reshape(1, -1))[0]
        r_rej = rm.forward(np.array(rej_tokens).reshape(1, -1))[0]

        if r_pref > r_rej:
            correct += 1
        print(f"  Preferred: {r_pref:+.4f} | Rejected: {r_rej:+.4f} | {'Correct' if r_pref > r_rej else 'Wrong'}")

    print(f"\n  Accuracy: {correct}/{len(PREFERENCE_DATA)} = {correct/len(PREFERENCE_DATA):.1%}")
    print()

    print("STAGE 3: PPO Training")
    print("-" * 40)

    policy_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    reference_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )

    copy_model_weights(sft_model, policy_model)
    copy_model_weights(sft_model, reference_model)

    train_prompts = [pair["prompt"] for pair in PREFERENCE_DATA]

    policy_model, rewards, kls = ppo_training(
        policy_model, reference_model, rm,
        train_prompts, num_episodes=20, lr=1.5e-5, kl_coeff=0.02
    )
    print()

    print("=" * 70)
    print("COMPARISON: SFT vs RLHF")
    print("=" * 70)
    print()

    eval_prompts = [
        "What is the capital of France?",
        "Explain gravity.",
        "Name three programming languages.",
    ]

    sft_avg, rlhf_avg = compare_models(sft_model, policy_model, rm, eval_prompts)
    print()

    print("=" * 70)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 70)
    print()

    if kls:
        print(f"  Initial KL: {kls[0]:.4f}")
        print(f"  Final KL:   {kls[-1]:.4f}")
        print(f"  Max KL:     {max(kls):.4f}")
        kl_threshold = 0.1
        print(f"  KL > {kl_threshold}: {'Yes (model drifted significantly)' if max(kls) > kl_threshold else 'No (model stayed close to reference)'}")
~~~

## 七、失败模式、原文边界与工程注意

1. KL不保证消灭reward hacking，多RM也不保证都正确。
2. 原head更新漏rejected/错特征，policy更新是噪声。
3. reference必须冻结且不与policy共享可变数组。
4. reward绝对值可平移，均值上涨不等于偏好正确。
5. 同一训练对评价、截断丢响应与长度偏置都会误导。
6. 固定20token生成无法验证越训越长，需可变EOS与长度统计。
7. 不能用源码错误实验支持历史胜率/最佳β。

## 八、Ship It：本课可复用的中文检查模板

下面是原课 Ship It 的中文学习版，直接保存在本篇，不另外创建输出文件。原文中的固定样本数禁令、性能保证等，只能作为待验证建议，不能当作普遍定律。

~~~markdown
设计偏好准则、标注一致性与长度/领域覆盖，核查BT两分支梯度和held-out。PPO明确old/ref、rollout、mask、优势、clip/KL、optimizer与真实更新；报告人工质量、代理reward、长度和偏移。拒绝将奖励缩放随机扰动作为PPO结果。
~~~

## 九、Exercises：保留原课练习与验收方法

1. mean pooling与最后有效token比较，先修正RM梯度，同init同数据划分；6训练对不代表泛化。
2. chosen/rejected均值、margin与4新对比较，清晰margin不等于绝对概率校准。
3. 长度reward=len/100，先有真实策略梯度与EOS，再比较β=.1是否缓解，不预设必消除作弊。
4. .7帮助+.3简洁两RM组合，先校准尺度，查看多目标权衡，不保证同时都好。
5. β=.001/.02/.5同SFT起点真实训练，记录reward/KL/人工分，不预设.02最佳。

## 十、术语与完成标准

RM：偏好评分器；BT：成对概率模型；policy：生成策略；old policy：采样旧策略；reference：冻结锚点；advantage：相对回报；reward hacking：优化代理而非真目标。

完成检查：

- 能推导hw−hl的BT梯度。
- 能分清old和reference。
- 能识别噪声不是PPO。
- 能说明高RM分不等于真质量。

## 十一、验证范围与来源许可

### 本次核对记录（2026-10-25）

已执行补充冻结特征线性BT一次正确更新，w=[0.05,-0.05]。未运行原奖励Transformer/PPO，更没有人类偏好数据训练。

所有本文Python代码块已通过静态语法检查；原课可解析Python代码块逐段按语法树核对，保留原逻辑。语法通过不表示依赖齐全或教学片段已变成可训练系统。本次执行环境为Windows Python3.12.10，不是你的WSL环境；未安装NumPy、PyTorch、regex或transformers，未修改你的环境。

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

返回：[[00：Phase10学习导航与验证说明]] · 上一课：[[06：指令微调SFT——聊天格式、响应Mask与真实梯度]] · 下一课：[[08：DPO直接偏好优化——推导、响应对数概率与算法边界]]
