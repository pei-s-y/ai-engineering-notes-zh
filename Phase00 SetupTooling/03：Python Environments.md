> 路径约定：本文的 `~/projects/ai-engineering-from-scratch` 是通用示例，不表示仓库已位于该处。执行命令前，请将它替换为实际克隆目录；以 `phases/` 开头的路径均相对于课程仓库根目录。

本篇按原文“学习目标 → 问题 → 核心概念 → 三种方案 → 分阶段环境 → 项目配置 → 锁文件 → 常见错误 → 实践与练习”的顺序整理。命令示例使用 WSL/Linux 的 Bash 语法。

原文：[Python Environments](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/06-python-environments/docs/en.md)

## 一、学习目标

这节课要解决的核心问题是：不同项目需要不同的 Python 和依赖版本，怎样让它们互不干扰，并让环境可以重新建立？

具体需要掌握：

1. 用 uv、venv 或 Conda 建立隔离环境。
2. 用 pyproject.toml 声明项目依赖及可选依赖。
3. 用锁文件记录解析后的具体版本。
4. 理解全局安装、包管理工具混用、忘记激活、误提交环境目录和 CUDA 兼容性等问题。
5. 根据课程不同阶段的依赖需要安排环境。

## 二、原文提出的问题：依赖冲突

原文以两个项目为例：一个项目需要 PyTorch 2.4，另一个项目需要 PyTorch 2.1。如果它们共用一套环境，为第二个项目改变 PyTorch 版本，就可能影响第一个项目。

这种问题在 AI 项目中很常见：框架和模型库对版本有各自的要求，GPU 相关组件也需要兼容。安装一个包还可能连带改变其他包，最终出现“这个项目能运行，另一个项目却坏了”的情况。

原文给出的解决思路是：每个项目使用独立环境，各自维护自己的依赖。

## 三、核心概念：环境隔离

虚拟环境是为项目准备的一套独立 Python 环境，通常包含 Python 可执行文件的副本或链接，以及独立的第三方包安装位置。

```text
项目 A/.venv/ → 项目 A 需要的 Python 包版本
项目 B/.venv/ → 项目 B 需要的 Python 包版本
```

两个目录都叫 `.venv`，并不表示它们是同一个环境。它们属于不同项目，可以安装不同版本的包。

要区分三个步骤：

| 步骤 | 含义 |
|---|---|
| 创建环境 | 建立独立环境目录 |
| 激活环境 | 让当前终端优先使用该环境中的命令 |
| 安装依赖 | 将项目需要的包安装到目标环境 |

环境隔离主要针对 Python 及其依赖，不等于创建独立操作系统，也不会隔离显卡驱动。

## 四、方案一：uv，原文推荐的方案

原文看重 uv 的安装速度，以及它把 Python 版本、虚拟环境、依赖解析等功能集中在一个工具中的方式。

如果尚未安装 uv，原文给出的 Linux 安装命令是：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

这条命令下载官方安装脚本并执行。已经安装时不需要重复操作。

安装所需 Python，进入已有项目，再创建并激活环境：

```bash
uv python install 3.12
cd your-project
uv venv
source .venv/bin/activate
```

`your-project` 是示例路径，需要换成实际目录。`uv venv` 默认创建 `.venv`；如果需要明确使用 Python 3.12，可以写成 `uv venv --python 3.12`。

安装依赖：

```bash
uv pip install torch numpy
```

原文还介绍了另一种方式：直接初始化项目并记录依赖。

```bash
uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib
```

这里需要分清：

- `uv pip install`：重点是把包安装进环境。
- `uv add`：把包作为项目依赖记录下来，并更新锁文件及项目环境。

因此，“包已经装好了”和“项目已经记录这个依赖”是两件不同的事。

## 五、方案二：Python 自带的 venv

不能或不想使用 uv 时，可以用 Python 标准库中的 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy
```

第一条命令使用当前 `python3` 创建环境；第二条在当前 Shell 中激活它；第三条安装依赖。

也可以写成 `python -m pip install torch numpy`，明确使用当前 Python 对应的 pip。某些 Linux 安装需要先补充对应的 venv 系统包。

原文同时列出了 Windows 的激活方式。你使用 WSL，应采用 Linux 的 `source .venv/bin/activate`。

## 六、方案三：有需要时使用 Conda

Conda 除了管理 Python 包，还能管理一些非 Python 依赖，例如原生库和特定开发工具。

原文列出的典型使用场景是：

- 项目需要特定 CUDA Toolkit 等组件，希望放在环境中管理。
- 使用共享计算集群，不能随意安装系统软件。
- 项目文档明确要求使用 Conda。

原文建议安装较轻量的 Miniconda。以下假设它已安装并完成当前 Shell 的初始化：

```bash
conda create -n myproject python=3.12
conda activate myproject
```

`-n myproject` 指定环境名称。Conda 环境通常由 Conda 管理，不一定放在项目的 `.venv` 中。

原文还给出了安装 PyTorch 和 CUDA 组件的示例：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

这是原文中的特定版本和软件源示例，不代表当前所有平台、所有 PyTorch 版本的通用安装命令。实际安装应核对目标版本的发布说明。

原文强调尽量让同一个工具管理环境中的包，避免依赖状态混乱。它后文也给出了例外：确实需要 pip 专有包时，先安装 Conda 包，再安装 pip 包。

## 七、课程如何安排环境：按阶段区分依赖

原文不建议把整套课程的所有依赖都堆进一个环境，因为不同阶段可能需要相互冲突的版本。

它给出的思路是：

| 用途 | 环境安排 |
|---|---|
| 前期基础内容 | 仓库根目录的轻量环境 |
| 神经网络、视觉相关内容 | 包含 PyTorch 的环境，兼容时可共享 |
| Transformer 等特定项目 | 按版本要求使用独立环境 |
| API 调用类内容 | 包含相关 SDK 的环境 |

重点是按依赖需求隔离。原文的目录树是策略示意，其中部分阶段名称可能与当前仓库不同，应以实际目录为准。

课程提供的 `code/env_setup.sh` 用于建立基础环境。

## 八、pyproject.toml：声明项目和依赖

原文接下来介绍 pyproject.toml。它是 Python 项目的标准配置文件，可以集中记录项目元信息、Python 要求和依赖。

原文示例：

```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]
```

理解各字段：

- `name`：项目名称。
- `version`：项目自身的版本。
- `requires-python`：项目允许使用的 Python 版本范围。
- `dependencies`：基础依赖。
- `optional-dependencies`：按功能选择安装的额外依赖，称为 extras。

这里的 `torch` 和 `llm` 是自行命名的可选功能集合。`numpy>=1.26` 表示允许 1.26 及之后的兼容候选版本，不是固定安装某一个版本。

原文使用以下命令安装：

```bash
uv pip install -e ".[torch]"
uv pip install -e ".[llm]"
uv pip install -e ".[torch,llm]"
```

其中 `.` 代表当前项目，方括号选择 extra，`-e` 表示可编辑安装。实际可编辑安装需要项目具有适当的打包配置和源码结构；上面的依赖片段本身不保证任意目录都可以直接安装。

原文的重点是用配置文件表达依赖，而不是只在终端里安装、却不留下项目要求。pyproject.toml 可以承担许多传统配置文件的职责，但 requirements.txt 并未失效，也可以与它共存。

## 九、锁文件：记录具体解析出的版本

只写版本范围，不能确定下一次安装究竟会选中哪个版本。锁文件用于记录解析后的具体依赖，其中也包括间接依赖。

例如：

```text
你的项目 → 包 A → 包 B
```

包 A 是直接依赖；项目通过 A 使用到的 B 是间接依赖，也叫传递依赖。

在 uv 项目工作流中：

```bash
uv add numpy
```

会维护项目的 `uv.lock`。

原文还介绍了 requirements 格式的锁定方式：

```bash
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock
```

第一条解析依赖并输出具体版本；第二条根据文件安装。这里的 requirements.lock 与 uv.lock 是不同格式、不同工作流的文件，不是两个都必须创建。

| 对象 | 作用 | 通常是否提交 Git |
|---|---|---|
| pyproject.toml | 声明项目需要什么 | 是 |
| 项目采用的锁文件 | 记录具体解析结果 | 是 |
| .venv/ | 本机实际安装的环境 | 否 |

原文希望达到的效果是：别人 Clone 项目后，可以根据记录重建相近且可控的依赖环境。锁文件提高可复现性，但不保证跨平台安装的所有文件、硬件行为或训练结果完全相同。

## 十、原文列出的五个常见错误

### 1. 安装进全局或错误的环境

没有确认环境就执行 `pip install`，可能影响其他项目。问题的关键在于命令对应哪个 Python，而不只是有没有写 `pip install`。

原文建议检查：

```bash
which python
which pip
```

激活普通 venv 后，路径通常应指向 `.venv/bin/`。进一步可以查看：

```bash
python -c "import sys; print(sys.executable)"
```

补充：uv 创建的环境默认可能没有独立 pip。此时继续使用 `uv pip`，不必因为 `which pip` 未指向环境就断定环境损坏。

### 2. 随意混用 pip 和 Conda

两套工具对环境状态的掌握不同，交替安装可能产生难以追踪的冲突。原文的处理原则是优先用 Conda，确有必要时最后用 pip 补充。

### 3. 忘记激活

环境已经创建、包也已经安装，但执行 `python train.py` 时仍可能调用另一个 Python。

```bash
source .venv/bin/activate
python train.py
```

终端中的 `(.venv)` 是提示；实际解释器路径更可靠。激活作用于当前终端，新终端不一定继承它。

### 4. 把 .venv 提交到 Git

环境可能包含大量依赖文件，且与本机平台和路径有关。应在 `.gitignore` 中加入：

```gitignore
.venv/
```

共享项目配置和锁文件，让其他人重新建立环境。

### 5. 忽视 CUDA 兼容性

原文要求检查驱动信息和 PyTorch 的 CUDA 构建信息：

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

两者需要兼容，但不必显示相同版本号。

**原文勘误：**正文把“CUDA 11.8 构建不能在支持 CUDA 12.x 的驱动上运行”写得过于绝对；较新驱动通常具有向后兼容能力，同一大版本内也有条件性小版本兼容。`nvidia-smi` 的 CUDA 显示不是当前环境安装的 Toolkit 版本。应依据具体 PyTorch 构建的驱动要求判断，不能仅比较两个数字。

## 十一、把课程环境建立起来

在 WSL 中进入你的仓库：

```bash
cd ~/projects/ai-engineering-from-scratch
bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh
```

根据已读取的本地脚本，它会创建或尝试复用根目录 `.venv`，安装 NumPy、Matplotlib、Jupyter、scikit-learn、Pandas，并检查导入及矩阵运算。PyTorch 不在其基础必装列表中。

脚本完成后，要在当前终端使用这个环境，还需要：

```bash
source .venv/bin/activate
```

这是因为 `bash` 启动的脚本在子 Shell 中执行，内部激活不会自动保留到原来的终端。

## 十二、原文四个练习

1. **运行初始化脚本。**检查环境是否创建成功、包能否导入，以及各项验证结果。
2. **创建第二个虚拟环境。**分别安装不同版本的 NumPy，比较版本输出，证明环境互相隔离。
3. **编写 pyproject.toml。**为同时需要 PyTorch 和 Anthropic SDK 的项目声明依赖。
4. **观察全局安装位置。**原文要求暂时进行一次全局安装，观察位置后卸载，用于理解包装到了哪里。

第四项若遇到 WSL 系统 Python 的 `externally-managed-environment` 提示，说明系统阻止随意修改受管理的环境。可以记录这一现象，不必强行绕过；在可丢弃的环境中模拟装错位置，也能学习这个区别。

## 十三、术语速查

| 术语 | 理解 |
|---|---|
| Virtual environment | 独立管理 Python 包的环境 |
| Dependency | 项目需要的其他软件包 |
| Transitive dependency | 依赖包进一步需要的包 |
| pyproject.toml | 项目的标准配置和依赖声明文件 |
| Optional dependency / Extra | 按功能选择启用的额外依赖 |
| Lockfile | 记录具体依赖解析结果的文件 |
| Dependency resolution | 找出同时满足各项版本约束的依赖组合 |
| CUDA mismatch | GPU 相关软件与驱动不兼容的一类问题 |

学习后可以用自己的话解释：虚拟环境解决“不同项目怎样互不干扰”，依赖声明解决“项目需要什么”，锁文件解决“具体采用了哪些版本”。

## 补充说明的参考资料

- [Python：venv](https://docs.python.org/3/library/venv.html)
- [uv：项目管理](https://docs.astral.sh/uv/guides/projects/)
- [uv：虚拟环境](https://docs.astral.sh/uv/pip/environments/)
- [uv：依赖锁定](https://docs.astral.sh/uv/pip/compile/)
- [Conda：环境管理](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Python Packaging：pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [NVIDIA：CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/latest/)
