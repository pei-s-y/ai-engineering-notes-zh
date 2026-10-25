# Windows 11＋WSL 2＋RTX 4060


本文按原课的“问题 → 概念 → 七个实践步骤 → 使用方式 → 练习”展开，以中文解释原课内容，并补充你的设备适配。标注 **【原课】** 的内容来自课程；**【设备适配】** 是针对本机的建议；**【纠正／补充】** 用于避免照抄时产生误解。它是一份学习笔记，不是逐字翻译。

课程来源：[课程页面](https://aiengineeringfromscratch.com/lesson?path=phases%2F00-setup-and-tooling%2F07-docker-for-ai)、[英文正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/07-docker-for-ai/docs/en.md)、[课程 Dockerfile](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/07-docker-for-ai/code/Dockerfile)、[课程 Compose 文件](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/07-docker-for-ai/code/docker-compose.yml)。课程仓库采用 MIT 许可证，相关声明附于文末。

## 0. 先看你的设备与学习路线

### 已确认的情况

| 项目 | 本机检测结果 | 对本课的意义 |
|---|---|---|
| 系统 | Windows 11 家庭版中文版，64 位，构建 26200 | 使用 Linux 容器，不需要为本课升级专业版 |
| CPU | AMD Ryzen 7 8845H | 下载 Docker Desktop 时选 x86_64／AMD64 版本；AMD64 不代表必须用 AMD 显卡 |
| 内存 | 约 16 GB，系统报告约 15.3 GiB | 先做一个开发容器＋小型 Qdrant，避免同时启动重型训练 |
| 独立显卡 | NVIDIA GeForce RTX 4060 Laptop GPU | 本课 CUDA 路线使用这张显卡 |
| 显存 | 8188 MiB，约 8 GB | 足够做 CUDA 张量验证；不等于能装下任意大模型 |
| 核显 | AMD Radeon 780M Graphics | 本课的 NVIDIA CUDA 配置不使用它 |
| NVIDIA 驱动 | 555.97 | 是已安装版本，不代表已验证 Docker 内 GPU；仍需逐层测试 |
| 虚拟化 | 固件虚拟化启用，检测到 hypervisor | 已具备重要基础条件 |
| WSL | Ubuntu-24.04，运行模式为 WSL 2；检查时处于 Stopped | 无须重新安装一份 Ubuntu；Stopped 本身不是故障 |
| Windows Python | 在 D 盘发现 Python312 路径 | 容器中的 Python 与这份 Windows Python 分开管理 |
| Docker | 当前 Windows PATH 未找到 docker；两个常见 Desktop 安装位置均未找到程序 | 暂按“尚未确认安装”安排流程；未检查所有自定义路径或 WSL 内安装 |
| C 盘剩余 | 约 20.4 GiB | 完整 CUDA 开发镜像与构建缓存可能造成空间压力 |
| D 盘剩余 | 约 30.0 GiB | 也不宽裕，不应默认能同时放多个镜像和大模型 |

**推荐路线：Windows 上安装 Docker Desktop → 采用 WSL 2 后端 → 为 Ubuntu-24.04 开启集成 → 在 Ubuntu 终端运行原课 Bash 命令。** Windows 家庭版支持本课需要的 Linux 容器。[Docker Windows 安装说明](https://docs.docker.com/desktop/setup/install/windows-install/)

本文后续大多数命令采用 Ubuntu 的 Bash。标有 PowerShell 的代码才在 Windows PowerShell 执行。不要把 Bash 的反斜杠续行直接粘贴到 PowerShell。

本次只检查环境并整理笔记；没有安装 Docker、修改驱动、构建镜像或运行容器。以下“预期结果”均为之后实践时的验收标准。

### 磁盘与显存是两回事

- 磁盘放镜像、构建缓存、模型文件和数据集。
- 内存供 Windows、WSL、Python 和数据处理使用。
- 显存供 GPU 上的模型权重、激活值等使用。

7B 参数的 FP16 权重粗略需要 `7×10^9×2 ≈ 14 GB`，还没包括推理缓存和其他开销。因此，你的 8 GB 显存不能直接完整容纳这样的权重。量化和卸载属于后续内容，本课用小张量即可验证 GPU。

对于磁盘，我建议在存放 Docker 数据的盘上争取预留 **40–60 GB** 再做完整实验。这是给镜像、构建和缓存留余量的实践建议，不是官方最低要求，也不代表最终一定占用这么多。暂时无法腾出空间时，先完成概念与 hello-world。

## 1. 原课要解决什么问题？

【原课】同一份 AI 代码，在不同机器上可能遇到不同的 Python、PyTorch、CUDA 和系统库。代码本身没有改变，环境差异却让它运行失败。

Docker 的价值是把应用需要的用户态环境整理为镜像，使换机器、交接项目和重建环境更可控。AI 项目尤其受益于三点：

1. CUDA、PyTorch、编译工具及系统依赖关系复杂。
2. 模型和数据很大，应当独立保存，避免随开发环境反复下载。
3. 应用往往包含多个服务，例如 Python 服务、向量数据库和前端。

【纠正】Docker 不能保证“任何机器、任何硬件都完全一样”。CPU 架构、GPU 型号、宿主机驱动、内存和显存仍然影响运行。镜像固定了大量软件条件，但不把一张不存在的 NVIDIA GPU 变出来。

## 2. 核心概念：把这些东西分开理解

| 概念 | 在本课里的含义 | 例子 |
|---|---|---|
| Dockerfile | 构建环境的说明书 | 安装 Python、PyTorch、Jupyter |
| Image／镜像 | 构建后的只读环境模板 | `ai-dev:lesson07` |
| Container／容器 | 基于镜像创建的运行实例，可运行也可停止 | 正在运行 Jupyter 的实例 |
| Bind mount／绑定挂载 | 把明确的宿主机路径映射进容器 | 项目目录 → `/workspace` |
| Named volume／命名卷 | Docker 管理的持久存储 | `qdrant_data` |
| Compose | 用 YAML 描述并管理多个服务 | 一起启动 ai-dev 和 qdrant |
| Registry／镜像仓库 | 下载或推送镜像的服务 | Docker Hub |

基本关系：

```text
Dockerfile ── docker build ──> 镜像
镜像       ── docker run   ──> 容器
宿主机目录／数据卷 ── 挂载 ──> 容器内路径
Compose 文件 ── docker compose up ──> 多个容器＋网络＋挂载
```

### 在你的 Windows 上，“共享内核”是什么意思？

原课说容器共享宿主内核。对你这套环境，需要补上一层：Linux 容器使用 WSL 2 提供的 Linux 内核，并不是直接共享 Windows 的 NT 内核。

```text
Windows 11：桌面、浏览器、NVIDIA Windows 驱动
└─ WSL 2：提供 Linux 内核
   ├─ Ubuntu-24.04：你输入 Linux 命令、存放课程代码的环境
   └─ Docker Desktop 管理的环境：运行 Docker 引擎和 Linux 容器
      ├─ ai-dev：Python / CUDA 用户态库 / Jupyter
      └─ qdrant：向量数据库
```

Ubuntu 终端可以通过集成访问 Docker 引擎；这不等于 Docker 引擎必须由你在 Ubuntu 中另外安装。[Docker WSL 2 后端说明](https://docs.docker.com/desktop/features/wsl/)

### Docker 与上一课的 venv／uv 有什么区别？

`venv` 主要隔离 Python 包。Docker 还能固定 Linux 用户态环境和系统依赖。它们可以配合使用；本课先理解容器自身的 Python 环境，不必在 Windows、WSL、容器里各装一遍相同依赖。

【原课】常见容器用途包括：开发容器（Jupyter、调试工具齐全）、训练容器（围绕训练任务）、推理容器（围绕在线服务）。这节课实际搭建的是开发环境，Compose 里的 ai-dev 启动 Jupyter，并非已经实现了模型推理服务器。

## 3. Step 1：安装与验证 Docker

【原课】列出了 macOS 和原生 Ubuntu 的安装方式。你的情况应补用 Docker Desktop 的 Windows 路线，不要直接照搬原生 Linux 的 Docker 安装脚本。

### 3.1 Windows 侧准备

在 PowerShell 检查：

```powershell
wsl --version
wsl --list --verbose
nvidia-smi
```

`wsl --list --verbose` 中的 VERSION 2 指发行版运行模式；`wsl --version` 才用于查看 WSL 软件版本。此次确认了前者，未确认后者。

按 Docker 文档检查 WSL 版本；必要时运行：

```powershell
wsl --update
```

从 [Docker 官方 Windows 安装页](https://docs.docker.com/desktop/setup/install/windows-install/) 获取 x86_64 版本，安装并启动 Docker Desktop，采用 WSL 2 后端、Linux 容器模式。

在设置中核对：

- General：使用 WSL 2 based engine；某些版本默认启用后不显示开关。
- Resources → WSL Integration：启用 Ubuntu-24.04 的集成。
- 镜像／磁盘数据存放位置：本机 C 盘紧张，应在大规模下载前核对。

安装程序所在盘和 Docker 镜像数据所在盘不是同一个概念。把安装程序放 D 盘，不保证镜像也在 D 盘。[Docker WSL 设置](https://docs.docker.com/desktop/features/wsl/)

如果之后发现 Ubuntu 内已单独安装 Docker Engine，先判断当前连接的是哪套引擎；不要盲目叠加安装或直接卸载已有环境。

### 3.2 进入 Ubuntu 后验证

PowerShell：

```powershell
wsl -d Ubuntu-24.04
```

以下在 Ubuntu：
安装docker
```
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```


```bash
docker --version
docker version
docker compose version
docker run --rm hello-world
```

验收：`docker version` 能看到 Client 和 Server；hello-world 成功打印测试消息。只有版本号不代表引擎已启动或镜像下载可用。

`--rm` 表示这个测试容器退出后自动删除；下载的镜像仍可保留复用。

## 4. Step 2：让容器访问 NVIDIA GPU

### 4.1 按实际安装路线理解这一步

【原课】原文称 Windows（WSL 2）用户可以跳过 Linux 的 Toolkit 安装段落，这里的前提是使用 Docker Desktop。**不能仅因为系统是 Windows＋WSL 2，就认为一定可以跳过。**

【实际操作更新】你已在 Ubuntu 24.04 中通过 `get.docker.com` 脚本安装 Docker Engine，并将用户 pei 加入 docker 组。当前采用的是：

```text
Windows 11＋NVIDIA Windows 驱动
└─ WSL 2 / Ubuntu 24.04
   └─ Docker Engine（由 Ubuntu 中的 systemd 管理）
      └─ CUDA 容器
```

这与本文前面最初建议的 Docker Desktop 路线不同。**本节以你的实际安装为准：继续使用现有 Engine，需要手动安装并配置 NVIDIA Container Toolkit。**

Windows NVIDIA 驱动向 WSL 提供 GPU 能力，Container Toolkit 再为容器配置 GPU 访问。不要在 WSL 中另外安装 Linux NVIDIA 显示驱动。[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

当前进度按已看到的操作结果记录：

| 项目 | 已确认状态 |
|---|---|
| Docker Engine | 29.8.1，服务正常运行 |
| 用户权限 | 新会话中 pei 可以不加 sudo 使用 Docker |
| WSL 显卡识别 | nvidia-smi 能识别 RTX 4060 Laptop GPU |
| 镜像下载与普通容器运行 | 配置代理后，hello-world 已成功 |
| NVIDIA Container Toolkit | 最近检查返回 nvidia-ctk: command not found，尚待安装 |
| 容器内 GPU 测试 | 尚未提供成功输出，不能标记为通过 |

### 4.2 这次遇到的错误：下载失败，不是 GPU 报错

执行原课测试命令时，Docker 需要先下载 CUDA 镜像，再创建 GPU 容器。你遇到的：

```text
Unable to find image ... locally
failed to resolve reference ... registry-1.docker.io ... i/o timeout
```

第一行只是说明本地没有镜像；真正失败的是连接 Docker Hub 超时。此时容器尚未启动，还没进入 GPU 验证。

排查发现终端的 HTTP_PROXY／HTTPS_PROXY 指向 `http://127.0.0.1:7897`，而 Docker 服务的 Environment 为空。终端 curl 能收到仓库的 `401`，说明该请求能连通；这个未携带令牌的接口返回 401 是正常现象。Docker 服务需要单独配置代理，不能自动继承当前终端变量。

你已按以下方式配置，并通过 hello-world 验证。**下面是成功配置的记录，无需重复执行：**

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7897"
Environment="HTTPS_PROXY=http://127.0.0.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

检查命令为 `systemctl show docker --property=Environment`。HTTPS_PROXY 的值使用 `http://` 是正确的，它描述连接代理的协议。该地址适用于你当前已验证的网络环境；下载时需保持代理运行。[Docker 服务代理配置](https://docs.docker.com/engine/daemon/proxy/)

### 4.3 安装 NVIDIA Container Toolkit（下一步）

以下均在 **Ubuntu 终端**分段运行；某一段报错先处理该错误。使用当前官方稳定版软件源，不照搬原课按发行版拼接软件源地址的旧写法。[NVIDIA 官方安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

① 安装准备工具。保留终端代理变量供 apt 使用：

```bash
sudo --preserve-env=http_proxy,https_proxy apt-get update
sudo --preserve-env=http_proxy,https_proxy apt-get install -y ca-certificates curl gnupg
```

② 导入软件源签名密钥：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

③ 添加稳定版软件源：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

④ 安装并检查版本：

```bash
sudo --preserve-env=http_proxy,https_proxy apt-get update
sudo --preserve-env=http_proxy,https_proxy apt-get install -y nvidia-container-toolkit
nvidia-ctk --version
```

预期显示版本号，不再提示 command not found。这里安装的是容器 GPU 支持工具，不是完整 CUDA 开发工具链。

⑤ 配置 Docker 并重启：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

第一条更新 `/etc/docker/daemon.json` 中的 NVIDIA 运行时设置；第二条使配置生效，并会影响正在运行的容器。这不会替代你已经写入 systemd 配置的代理设置。

### 4.4 测试容器内 GPU 访问

先确认 WSL 自身仍能识别 GPU：

```bash
nvidia-smi
```

若提示找不到命令，可再检查 `/usr/lib/wsl/lib/nvidia-smi`，不要立即判断 GPU 不受支持。

再执行原课的测试命令：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

首次运行需要下载镜像。预期最终显示包含 **RTX 4060 Laptop GPU** 的显卡信息表，才说明容器能访问 GPU。只有 Pull complete 或镜像下载成功，还不能算 GPU 验证通过。

镜像标签中的 `ubuntu22.04` 指容器内部的 Ubuntu 用户态环境，不要求你的 WSL Ubuntu 也是 22.04。你目前的 Ubuntu 24.04 无需降级，原命令保持不变。

参数拆解：

| 部分 | 意义 |
|---|---|
| `docker run` | 基于镜像创建并启动新容器 |
| `--rm` | 退出后删除容器 |
| `--gpus all` | 向容器开放可用 NVIDIA GPU |
| `nvidia/cuda:12.4.1-base-ubuntu22.04` | 本次测试使用的 CUDA 基础镜像 |
| `nvidia-smi` | 要在容器中执行的命令 |

GPU 信息能显示，不代表 PyTorch 已能成功计算；后面 Step 4 还要做张量计算测试。若仍报仓库连接超时，应检查下载与代理；若镜像下载完成后报 NVIDIA 运行时或 GPU 设备错误，才进入 Toolkit／驱动链路排查。

【纠正】`nvidia-smi` 显示的 CUDA Version 不是“你已安装的 Python CUDA 包版本”。宿主机驱动、容器 CUDA 工具链、PyTorch 编译对应的 CUDA 版本需要分别理解；Windows 本机无需为了本课额外安装完整 CUDA Toolkit。
## 5. Step 3：理解基础镜像

【原课】给出四种选择：

| 镜像 | 包含什么 | 什么时候选 |
|---|---|---|
| `nvidia/cuda:12.4.1-devel-ubuntu22.04` | CUDA 开发工具，含编译工具链 | 跟随本课；需要编译 CUDA 扩展时 |
| `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | CUDA 运行库，不含完整开发工具链 | 运行已编译的代码，减少体积 |
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` | 预装 PyTorch 的运行环境 | 希望省去自行安装 PyTorch 的步骤 |
| `python:3.12-slim` | 精简 Python 环境，没有预装 CUDA | CPU 任务、轻量服务 |

本课以 devel 为主，是为了学习“从 CUDA 基础环境组装 AI 环境”。先保留课程版本，不要一边学习 Docker 一边随意升级整套依赖。

原课给出的镜像大小只是粗略示意。标签、平台、压缩下载量、解压后的本地大小和共享层都会影响实际占用。本机用 `docker images` 看镜像，用 `docker system df` 看整体使用情况。

WSL 的 Ubuntu 24.04 与镜像中的 Ubuntu 22.04 不冲突：容器自带其用户态文件系统，并不要求与终端所在发行版版本一致。

## 6. Step 4：读懂 Dockerfile，并构建 AI 开发环境

### 6.1 先确定命令所在目录

以下约定你已经有一份课程仓库，并在它的根目录操作。根目录应当包含 `README.md` 和 `phases/`。

```bash
pwd
ls README.md phases
```

若尚无仓库，可在自己选定的学习目录克隆；已有副本就复用，不必重新下载。本文未确认你的课程仓库实际位置。不要在本项目只读的 `sources/` 中修改课程文件。

### 6.2 按原课 Dockerfile 的顺序理解

| 原课设置／操作 | 作用与需要理解的点 |
|---|---|
| `FROM nvidia/cuda:12.4.1-devel-ubuntu22.04` | 选择构建起点；它不是空环境 |
| `DEBIAN_FRONTEND=noninteractive` | 让软件安装避免等待交互选择 |
| `PYTHONUNBUFFERED=1` | Python 输出及时进入日志，便于看进度 |
| 安装 git、curl、build-essential | 分别用于源码获取、下载和编译 |
| 添加 deadsnakes PPA，安装 Python 3.12 | 在该 Ubuntu 基础镜像中提供课程需要的 Python |
| `update-alternatives` | 让镜像内 `python` 命令指向 Python 3.12 |
| 下载固定提交的 get-pip.py，并校验 SHA256 | 安装 pip，同时核对下载内容 |
| 安装 PyTorch 的 cu124 版本 | 课程固定 torch 2.6.0、torchvision 0.21.0、torchaudio 2.6.0 |
| 安装 numpy、pandas 等 | 数据处理、绘图、Notebook、模型和数据集相关工具 |
| `WORKDIR /workspace` | 设置容器内默认工作目录 |
| `VOLUME ["/workspace", "/models"]` | 声明卷挂载点，不会自动绑定你的指定宿主目录 |
| `EXPOSE 8888` | 声明服务端口；真正发布到宿主机还需要 `-p` 或 Compose ports |
| `CMD ["python"]` | 默认启动 Python；运行时可以改为启动 Jupyter |

`RUN` 在构建时执行；`CMD` 在容器启动时作为默认命令。`ENV` 等指令可以修改镜像配置，不能把所有 Dockerfile 指令都理解为增加了一份文件层。[Dockerfile 官方参考](https://docs.docker.com/reference/dockerfile/)

原课文件可直接从仓库阅读，不必在这里再复制一份长安装脚本。下面构建命令使用原文件，并沿用原课环境。

### 6.3 构建前补上原课连接数据库所需的包

【纠正】原课后面调用 `qdrant_client`，但当前课程 Dockerfile 的依赖列表没有安装 `qdrant-client`。应在自己的可写课程副本中，把它加入安装 AI Python 包的那条 `RUN python -m pip install ...`。

例如列表末尾原来是 `safetensors`，改为：

```dockerfile
    safetensors \
    qdrant-client
```

这是原有 RUN 的最后两行，不是独立的 Dockerfile 指令。正式复现项目应在验证后固定客户端及其他依赖版本；本笔记未完成运行验证，因此不虚构一个已验证的锁定版本。

### 6.4 构建镜像

在仓库根目录、Ubuntu 终端：

```bash
docker build -t ai-dev:lesson07 -f phases/00-setup-and-tooling/07-docker-for-ai/code/Dockerfile .
```

- `-t ai-dev:lesson07`：给镜像命名并加标签，便于辨认本课版本。
- `-f .../Dockerfile`：指出构建说明书。
- 最后的 `.`：构建上下文是当前目录，不是标点。

这里将原课的 `ai-dev` 改成了明确标签，后文统一使用这个名称。第一次构建会下载基础镜像与依赖；后续未改变的步骤可以利用缓存。

依赖变更后需要重新构建。进入临时容器手动 `pip install` 并不会把安装步骤自动写回镜像；删除容器后这种修改也不会成为新容器的配置。

【补充】Dockerfile 有助于复现，但原课的部分 pip 包没有锁版本，基础镜像标签也可能变化。因此“同一 Dockerfile 在任何时候重建都完全相同”并不成立。

### 6.5 验证 PyTorch 与实际 GPU 计算

Ubuntu 终端：

```bash
docker run --rm --gpus all ai-dev:lesson07 python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('available:', torch.cuda.is_available()); assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0)); x=torch.ones(3, device='cuda'); print((x+x).cpu())"
```

预期：PyTorch 对应 2.6.0 的 CUDA 构建，available 为 True，GPU 名称为 RTX 4060 Laptop GPU，最终输出三个 2。若断言失败，先检查 GPU 链路，不必下载模型来“试试看”。

### 6.6 启动 Jupyter

仍在仓库根目录。先创建存储目录：

```bash
mkdir -p "$HOME/models" "$HOME/datasets"
```

然后运行：

```bash
docker run --rm -it --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --mount type=bind,source="$HOME/models",target=/models \
  --mount type=bind,source="$HOME/datasets",target=/data \
  -e HF_HOME=/models/huggingface \
  -p 127.0.0.1:8888:8888 \
  ai-dev:lesson07 \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

这里的 `$HOME` 是 Ubuntu 用户家目录，不是 Windows 用户目录。目录实际占用哪个 Windows 磁盘，取决于该 WSL 发行版的数据位置；本文没有确认该位置。

`-it` 让你在终端交互并看到输出。容器里的 Jupyter 监听 `0.0.0.0`，使容器网络可以访问它；Windows 侧只发布到 `127.0.0.1:8888`，适合本机学习。

在 Windows 浏览器打开日志给出的 localhost 地址并保留 token。不要把日志中容器内部主机名当成 Windows 浏览器必须使用的地址。结束时按 Ctrl+C，按提示确认；本命令使用 `--rm`，退出后容器会删除。

单独运行的 Jupyter 与下一节 Compose 都要使用 8888。进入 Compose 实验前先停止这个实例，避免端口占用。

## 7. Step 5：让代码、模型和数据持久保存

【原课】建议把代码、模型、数据集分别挂载到 `/workspace`、`/models`、`/data`。这三个路径都是容器看到的位置。

### 7.1 停止容器不等于删除数据

| 操作 | 容器可写层 | 挂载的数据 |
|---|---|---|
| `docker stop` | 保留，容器只是停止 | 保留 |
| `docker start` | 继续使用原容器的可写层 | 继续使用 |
| `docker rm` 或 `--rm` 退出 | 随容器删除 | 绑定目录仍在；命名卷通常独立保留 |
| 根据镜像新建容器 | 获得新的可写层 | 挂载同一目录／卷后可访问旧数据 |

【纠正】原课“容器停止，模型下载就消失”的表述不准确。真正要防的是删除、替换容器，或者模型被下载进未持久化的位置。绑定挂载和命名卷也应分开理解。[Docker 存储说明](https://docs.docker.com/engine/storage/)

### 7.2 挂载目录不会自动改变模型缓存位置

创建 `/models` 挂载后，程序仍可能默认把下载文件写到别处。因此上面的命令额外设置：

```text
HF_HOME=/models/huggingface
```

它把 Hugging Face 的缓存根目录放在持久化挂载下。这是对原课的补充。[Hugging Face 环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

原课使用 `/models/llama-7b` 演示“从本地目录加载模型”，该目录必须实际包含合适的配置与权重。挂载目录本身不会下载模型，也不会自动把缓存变成这个目录。

### 7.3 一个不下载模型的持久化实验

在 Ubuntu 运行：

```bash
mkdir -p "$HOME/models"
docker run --rm --mount type=bind,source="$HOME/models",target=/models ai-dev:lesson07 python -c "from pathlib import Path; Path('/models/docker-lesson-marker.txt').write_text('persisted', encoding='utf-8')"
cat "$HOME/models/docker-lesson-marker.txt"
```

预期读到 `persisted`。第一个容器已经删除，但宿主目录里的文件仍在。这比下载十几 GB 模型更适合验证本课原理。

## 8. Step 6：用 Compose 同时管理开发环境与 Qdrant

### 8.1 原课结构与两个实际问题

【原课】ai-dev 运行 Jupyter，qdrant 提供向量数据库；Compose 自动建立共享网络，服务之间可以用服务名访问。

【纠正】原课 Compose 位于：

```text
仓库根目录/phases/00-setup-and-tooling/07-docker-for-ai/code/docker-compose.yml
```

从 code 目录向上三级是 `phases/`，不是仓库根目录。因此，如果目的确实是挂载整个仓库，原课的 `../../../:/workspace` 应改为 `../../../../:/workspace`。这是根据当前目录层级得出的路径核对结果。

另一个问题是缺少 `qdrant-client`，已在 Step 4 中补充。

### 8.2 适配版 Compose 示例

在自己的可写课程副本中，将以下内容另存到 code 目录的 `compose.local.yaml`。这是供你之后操作的示例，本次没有写入你的课程仓库。

它复用前面已经构建的 `ai-dev:lesson07`，所以不再配置 build，避免首次学习时重复构建不同名称的镜像。

```yaml
services:
  ai-dev:
    image: ai-dev:lesson07
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      HF_HOME: /models/huggingface
    volumes:
      - ../../../../:/workspace
      - ${HOME}/models:/models
      - ${HOME}/datasets:/data
    ports:
      - "127.0.0.1:8888:8888"
    stdin_open: true
    tty: true
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

这里保留课程的 Qdrant 标签，方便对照，不声称它是当前最新版本。此文件的 `${HOME}` 以 **从 Ubuntu 终端启动 Compose** 为前提。

### 8.3 启动、查看和验证

从仓库根目录进入课程 code 目录：

```bash
cd phases/00-setup-and-tooling/07-docker-for-ai/code
docker compose -f compose.local.yaml config
docker compose -f compose.local.yaml up -d
docker compose -f compose.local.yaml ps
docker compose -f compose.local.yaml logs --tail=50 ai-dev
```

`config` 用来检查配置解析和实际路径；`-d` 表示后台运行；`ps` 查看服务状态；日志提供 Jupyter 登录地址。

Qdrant 完成启动后，在 ai-dev 容器内用 Python 标准库检查网络和 HTTP 接口：

```bash
docker compose -f compose.local.yaml exec ai-dev python -c "import urllib.request; print(urllib.request.urlopen('http://qdrant:6333/collections').read().decode())"
```

再做与原课对应的客户端验证：

```bash
docker compose -f compose.local.yaml exec ai-dev python -c "from qdrant_client import QdrantClient; print(QdrantClient(host='qdrant', port=6333).get_collections())"
```

预期成功返回集合列表，新数据库通常为空。如果 HTTP 成功而 Python 客户端提示版本兼容问题，网络已经通了，应处理客户端与服务端版本匹配，而不是反复修改端口。

### 8.4 为什么有时用 localhost，有时用 qdrant？

| 发起访问的位置 | Qdrant 地址 |
|---|---|
| Windows 浏览器 | `http://localhost:6333/collections` |
| ai-dev 容器内部 | `http://qdrant:6333/collections` |

在 ai-dev 里，localhost 指 ai-dev 自己，不是 qdrant。服务之间使用服务名与容器端口通信；发布端口主要用于宿主机访问。[Compose 网络说明](https://docs.docker.com/compose/how-tos/networking/)

### 8.5 停止与删除要区分

```bash
docker compose -f compose.local.yaml stop
docker compose -f compose.local.yaml start
```

这组命令停止／恢复现有容器。

```bash
docker compose -f compose.local.yaml down
```

这会删除该项目的容器和网络，默认保留命名卷。原课还介绍 `down -v`，它会进一步删除这里的 qdrant_data 数据卷；只在明确要清空练习数据库时使用，不作为日常退出命令。

每次都带上 `-f compose.local.yaml`，避免误用同目录下的原课 Compose 文件。

## 9. Step 7：常用命令与读法

| 命令                                                   | 解决的问题            |
| ---------------------------------------------------- | ---------------- |
| `docker ps`                                          | 哪些容器正在运行？        |
| `docker ps -a`                                       | 是否存在已退出的容器？      |
| `docker images`                                      | 本地有哪些镜像，显示大小是多少？ |
| `docker system df`                                   | 镜像、容器、缓存和卷占用了多少？ |
| `docker logs -f 容器名`                                 | 跟踪该容器的日志         |
| `docker exec -it 容器名 bash`                           | 进入已经运行的容器        |
| `docker exec 容器名 nvidia-smi`                         | 从正在运行的容器检查 GPU   |
| `docker cp 容器名:/workspace/results.csv ./results.csv` | 将容器中的文件复制出来      |

表中的“容器名”需要替换为 `docker ps` 显示的实际名字。

记住：`run` 创建新容器，`exec` 在已有运行中容器里执行命令，`start` 重新启动已停止的容器。

【纠正】原课把 `docker system prune -a` 简写为“删除不用的镜像”，实际清理范围还包括停止的容器、未使用网络和构建缓存等；它不是本课日常退出步骤。本机磁盘紧张时，先看占用，再有针对性地处理，不要直接把这条命令当万能修复。

## 10. 后续课程如何复用，以及没有 GPU 时怎么办

【原课】后续课程可继续复用开发镜像，用挂载保存代码和数据，用 Compose 管理多服务；新增依赖后更新 Dockerfile 再构建。

对本文适配版来说，修改 Dockerfile 后，在仓库根目录重新执行前面的 build，再到 code 目录执行：

```bash
docker compose -f compose.local.yaml up -d --force-recreate ai-dev
```

它让 ai-dev 采用刚构建的同标签镜像；工作成果应提前放在挂载目录里。

【原课】无 GPU 可以去掉 `--gpus all` 和 Compose 的 NVIDIA deploy 配置。

【纠正】程序不一定自动回退 CPU。若代码写死 `.cuda()`，仍然会失败。需要显式选择设备，例如：

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.ones(3, device=device)
print(x)
```

删除 GPU 参数也不会把 CUDA 镜像自动变成小型 CPU 镜像。你有 NVIDIA GPU，先走 GPU 验证路线即可，不必同时维护两套配置。

## 11. 原课四个练习：适合你的完成标准

### 练习 1：构建并打印 PyTorch 版本

完成 Step 4。既打印版本，也让一个小张量在 GPU 上计算成功。无需下载模型。

### 练习 2：从 AI 容器访问 Qdrant

完成 Step 6。必须从 ai-dev 容器访问服务名 `qdrant`。Windows 浏览器能打开接口，只能证明宿主机访问这一段正常，不能替代容器间测试。

### 练习 3：加入 Flask，运行 5000 端口 API

在 Dockerfile 依赖列表中加入 flask 并重新构建。在自己的课程仓库根目录建立 `app.py`：

```python
from flask import Flask

app = Flask(__name__)

@app.get("/")
def index():
    return {"message": "Docker lesson works"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

从仓库根目录、Ubuntu 终端启动：

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  -p 127.0.0.1:5000:5000 \
  ai-dev:lesson07 python app.py
```

Windows 浏览器打开 `http://localhost:5000`，看到 JSON 即完成。这个练习不需要 GPU。Flask 必须监听容器的 `0.0.0.0`，只写端口映射并不足够。

### 练习 4：比较 devel 与 runtime

空间允许后再做。保留原 Dockerfile，另建比较文件，只改变基础镜像并使用另一个镜像标签，避免覆盖前一份结果。

比较时记录：基础镜像标签、最终镜像大小、是否仍能通过 PyTorch 计算测试、是否需要编译 CUDA 扩展。runtime 缺少 nvcc，所以不能只凭体积更小就断定一定更适合所有项目。

本机当前磁盘空间有限，此练习排在最后；不要同时保存多套大型环境只为第一次练习。

## 12. 常见问题：按层定位

| 现象 | 优先检查 |
|---|---|
| PowerShell 或 Ubuntu 提示 docker 找不到 | Desktop 是否安装；Ubuntu 的 WSL Integration 是否开启；终端是否重开 |
| 客户端有版本，连接 Server 失败 | Docker Desktop 是否启动并就绪；是否连接了错误的 Docker context |
| hello-world 下载超时 | 网络、代理或镜像仓库访问；这不说明 Dockerfile 有语法错 |
| `--gpus all` 报错 | Windows nvidia-smi → WSL nvidia-smi → 容器 nvidia-smi 逐层检查 |
| torch CUDA 为 False | 本次启动是否传入 GPU；是否装了 CUDA 版 PyTorch；驱动与运行环境是否兼容 |
| Jupyter 打不开 | 服务是否存活；日志是否报错；端口是否冲突；是否使用 localhost 和正确 token |
| `No module named qdrant_client` | 是否把 qdrant-client 写入 Dockerfile、重建并替换了旧容器 |
| ai-dev 访问 localhost:6333 失败 | 改用服务名 qdrant，确认两个服务属于同一 Compose 网络 |
| `/workspace` 只有课程子目录 | 检查相对挂载路径；本课文件位置需要向上四级才到仓库根目录 |
| 每次都重新下载模型 | 检查实际缓存路径是否位于挂载目录；仅创建 /models 不会自动迁移缓存 |
| GPU out of memory | 显存不足，与镜像是否构建成功无关；先减小任务规模 |
| No space left on device | 检查 Docker 数据盘、WSL 数据位置与构建缓存；不要只看代码所在盘 |

## 13. 复习检查表

- [ ] 我能解释 Dockerfile、镜像、容器的区别。
- [ ] 我知道 Windows、WSL、容器各自的职责。
- [ ] 我没有混用 PowerShell 与 Bash 的路径和续行语法。
- [ ] hello-world 成功，说明最基础的链路打通。
- [ ] 容器内 nvidia-smi 成功，PyTorch 小张量 GPU 计算也成功。
- [ ] 我知道停止容器、删除容器、删除数据卷是不同操作。
- [ ] 我能指出模型实际缓存在哪个持久化目录。
- [ ] 我能从 ai-dev 用 qdrant 服务名访问数据库。
- [ ] 我知道新增依赖要改 Dockerfile，再构建并替换容器。

三个自测问题：为什么模型要放在挂载目录？容器使用谁的 GPU 驱动？为什么 ai-dev 访问数据库用 qdrant 而不是 localhost？

参考答案：让数据独立于容器生命周期；这条 WSL 路线最终依赖 Windows NVIDIA 驱动提供 GPU；localhost 指当前容器自身，而 Compose 服务名指向另一个服务。

## 附：课程许可声明

本笔记依照课程结构整理、解释并改编部分示例。设备检查、Windows 适配及纠错为新增内容，不代表课程作者意见。上游采用 [MIT License](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/LICENSE)，声明如下：

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
