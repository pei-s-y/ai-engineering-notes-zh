> 路径约定：本文的 `~/projects/ai-engineering-from-scratch` 是通用示例，不表示仓库已位于该处。执行命令前，请将它替换为实际克隆目录；以 `phases/` 开头的路径均相对于课程仓库根目录。

## 一、这节课要解决什么问题？

写 AI 代码之前，先建立一个能够稳定运行、容易排错的开发环境。否则后面遇到 `import` 失败、版本冲突或 GPU 不可用时，很难判断是代码错了，还是环境错了。

课程希望你在这一课理解并验证：

1. 操作系统和基础工具是否可用；
2. Python、Node.js、Rust 等语言环境如何安装；
3. 如何用虚拟环境隔离项目依赖；
4. PyTorch 能否调用 GPU；
5. 如何运行项目自带的环境检查脚本。[课程目标与步骤](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/01-dev-environment/docs/en.md)

## 二、核心概念：开发环境有四层

|层级|包含什么|你现在的对应例子|
|---|---|---|
|1. 系统基础|操作系统、终端、Git、编辑器、显卡驱动|Windows、WSL2 中的 Ubuntu、Git、NVIDIA 驱动|
|2. 包管理工具|下载和管理软件、依赖的工具|Ubuntu 的 `apt`、Python 的 `uv`、Node 的 `pnpm`|
|3. 语言运行环境|运行代码所需的解释器或编译器|Python、Node.js、Rust、Julia|
|4. AI 库|建模、计算和训练所用的软件包|NumPy、PyTorch 等|

课程强调**从下往上检查**：如果系统或 Python 环境不对，直接排查 PyTorch 往往会浪费时间。

这里还有一个对你特别重要的区别：**Windows 中安装的 Python、Node.js，与 Ubuntu（WSL）中安装的版本是两套环境。**之前检查到 Windows 有 Python 3.12 和 Node.js，并不代表 Ubuntu 中也已安装它们。

## 三、步骤 1：系统基础

原文给 macOS、Ubuntu/Debian 和 Windows 三套入口。你选择的是 **Windows + WSL2 + Ubuntu 24.04**；WSL 与 Ubuntu 已经安装成功，因此不用再次执行 `wsl --install`。

在 **Ubuntu 终端**中，课程要求更新软件包列表并安装基础工具：

```
sudo apt update
sudo apt install -y build-essential git curl wget unzip
```

各部分的意思：

- `sudo`：临时取得 Ubuntu 管理员权限；
- `apt update`：刷新“有哪些软件版本可安装”的列表，不是升级所有软件；
- `apt install`：安装指定工具；
- `build-essential`：一组常用的编译工具；
- `git`：管理代码版本；
- `curl`、`wget`：从网络获取内容；
- `unzip`：解压 ZIP 文件；
- `-y`：对安装确认自动回答“是”。

装完可以逐项检查：

```
git --version
curl --version
unzip -v
```

只需确认能显示版本信息，不用背输出的全部内容。

## 四、步骤 2：Python 与 `uv`

课程选用 `uv` 管理 Python 版本、虚拟环境和依赖。它是工具，不是新的编程语言。原文给出的 Ubuntu 安装方式是：

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

这会从网络取得安装脚本并执行。执行前应确认网址来自 [`uv` 官方文档](https://docs.astral.sh/uv/getting-started/installation/)；如果你想先看脚本内容，官方也提供先查看、再决定是否执行的方式。安装后若出现 `uv: command not found`，先按安装器的提示更新当前终端环境，或关闭并重新打开 Ubuntu 终端，不要立刻重复安装。

然后检查并准备 Python 3.12：

```
uv --version
uv python install 3.12
```

`uv python install 3.12` 安装的是 **Ubuntu 这套环境中的 Python**，不会覆盖 Windows 里的 Python。

在项目目录中建立虚拟环境：

```
uv venv --python 3.12

cd ~/projects/ai-engineering-from-scratch
source .venv/bin/activate
```

`.venv` 是该项目独立的 Python 环境；激活后，终端提示符前通常会出现 `(.venv)`。此时再安装课程需要的包：

```
uv pip install numpy matplotlib jupyter
```

检查“当前到底在用哪个 Python”比单看版本更有用：

```
which python
python --version
```

`which python` 应指向当前虚拟环境内的路径。若以后出现“装了 NumPy 却仍然无法导入”，先检查这两条命令。

课程用一个小 NumPy 例子验证环境：构造向量 `[1, 2, 3]`，计算它与自身的点积，结果应为 **14**。在已激活虚拟环境的 Ubuntu 终端中可以运行：

```
python -c 'import numpy as np; a=np.array([1,2,3]); print(np.dot(a,a))'
```

### 如何定位课程仓库？

先在 WSL 中定位已经克隆的课程仓库。下面使用通用示例目录，请按实际克隆位置替换：

```
cd ~/projects/ai-engineering-from-scratch
```

课程脚本要从这个**仓库根目录**运行，即 `ls` 能看到 `README.md` 和 `phases` 的地方。当前先从这里学习和验证即可；如果以后运行大量 Python 文件或训练任务遇到跨文件系统性能问题，再考虑把工作副本放到 Ubuntu 文件系统中。微软对跨 Windows／Linux 文件系统的使用也有[说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)。

## 五、步骤 3：Node.js 与 `pnpm`

课程为后续的 TypeScript、Agent、MCP 和 Web 课程准备 Node.js，并使用 `fnm` 管理 Node 版本、`pnpm` 管理 Node 软件包。原文安装的是 Node 22；课程要求是 Node 20 或以上。[课程步骤 3](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/01-dev-environment/docs/en.md)

```
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22
npm install -g pnpm
node -e "console.log('Node', process.version)"
```

这组命令如果要跟着做，应在 **Ubuntu 终端**运行。你在 Windows 中已有 Node 24，但那不能直接算作 Ubuntu 的 Node 环境。安装器如果提示 `unzip` 缺失，回头确认步骤 1 的 `unzip` 是否安装成功。

## 六、步骤 4 和 5：Rust、Julia

课程随后介绍：

- **Rust**：用于部分更偏系统或性能的内容；检查命令是 `rustc --version`、`cargo --version`。
- **Julia**：课程标为可选，主要服务于某些数学示例；检查命令是 `julia -e 'println(VERSION)'`。

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh 
rustc --version 
cargo --version
```

For math-heavy lessons where Julia shines.
```
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

原文提供了二者的官方安装脚本。你可以按原课完成练习，但要记住它们是**各自独立的语言环境**，不是安装 PyTorch 的前置条件；Julia 在课文里也明确标为 optional。[课程步骤 4–5](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/01-dev-environment/docs/en.md)

## 七、步骤 6：确认 GPU 和 PyTorch

你有 NVIDIA RTX 4060 Laptop GPU。先在 **Ubuntu 终端**运行：

```
nvidia-smi
```

如果能显示显卡和驱动信息，说明 WSL 能看见显卡，这是第一层检查；**还不等于 PyTorch 已能使用 CUDA**。在 WSL 中使用 NVIDIA GPU 时，驱动由 Windows 侧提供；不要为了修复问题随意在 Ubuntu 中安装 Linux 显卡驱动。[NVIDIA 的 CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

课程原文给了一条固定 CUDA 版本的 PyTorch 安装命令。这个地址是课文编写时的示例，**不要把其中的 CUDA 版本当成永远适合你电脑的版本**。到实际安装时，应在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)按 Linux、包管理方式和 CUDA 支持选择当前命令，再装进已激活的 `.venv`。[课程步骤 6](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/01-dev-environment/docs/en.md)

装好后这样验证：

```
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU")'
```

理想输出是 CUDA 可用，并显示 RTX 4060。若显示 `False`，先保留完整输出，不要盲目重装驱动。课程也说明：没有 GPU 时很多入门练习仍能在 CPU 上运行。

## 八、步骤 7：运行课程自带的检查脚本

进入仓库根目录，再运行：

```
cd ~/projects/ai-engineering-from-scratch
python3 phases/00-setup-and-tooling/01-dev-environment/code/verify.py --route beginner
```

课程还提供 `ml-foundations`、`llm-engineering`、`agents`、`mcp` 等不同检查路线。**`beginner` 检查通过，不代表 Node、Rust、PyTorch、GPU 全部完成**；我核对了仓库中的 [`verify.py`](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/01-dev-environment/code/verify.py)，`beginner` 的必需检查重点是 Python 和 Git，其他很多工具是后续或可选项。

如果脚本提示下一条示例命令，按它在仓库根目录运行，并保存命令和输出。课程建议留下这些“学习证据”：**运行位置、执行的命令、退出状态、关键输出，以及你做出的文件或结果**。[仓库学习说明](https://github.com/rohitg00/ai-engineering-from-scratch)

## 九、原课练习与完成标准

原课末尾的三项练习是：

1. 运行环境验证脚本并解决必需项错误；
2. 为课程建立 Python 虚拟环境并安装 PyTorch；
3. 分别在课程涉及的四种语言中运行一个 Hello World。[原课练习](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/01-dev-environment/docs/en.md)

我建议你用下面四个问题检验自己是否真的学会了这一课：

- 我现在处于 PowerShell 还是 Ubuntu？怎样看出来？
- 我运行的 `python` 属于 Windows、Ubuntu 系统，还是当前 `.venv`？
- `nvidia-smi` 成功与 `torch.cuda.is_available()` 为 `True` 有什么区别？
- 为什么课程脚本必须从仓库根目录运行？

**你现在可以做的下一步**：如果 Ubuntu 的 `apt` 安装已经完成，就先在 Ubuntu 中运行 `python3 --version`、`git --version`、`nvidia-smi`，记录输出；然后我们接着做 `uv` 和虚拟环境，逐条核对，不必一次性执行整篇笔记中的所有安装命令。
