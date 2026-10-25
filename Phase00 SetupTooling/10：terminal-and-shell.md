> 路径约定：本文的 `~/projects/ai-engineering-from-scratch` 是通用示例，不表示仓库已位于该处。执行命令前，请将它替换为实际克隆目录；以 `phases/` 开头的路径均相对于课程仓库根目录。

# Terminal & Shell：终端与 Shell 学习笔记（WSL 2 Ubuntu 版）

整理日期：2026-10-01。对应 Phase 00 / 10 Terminal & Shell，原课预计学习时间约 35 分钟。

本文按原课八个步骤展开，保留主要命令与四个练习，并用【设备适配】和【纠正／补充】区分新增说明。课程页面未能直接完整读取，正文及配套脚本已通过同项目公开源码核对。本次只整理笔记，没有修改你的 Ubuntu 配置、安装工具或运行训练。

来源：[课程页面](https://aiengineeringfromscratch.com/lesson?path=phases%2F00-setup-and-tooling%2F10-terminal-and-shell)、[英文正文](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/10-terminal-and-shell/docs/en.md)、[shell_aliases.sh](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh)。课程采用 MIT 许可证，声明见文末。

## 0. 先对应你的电脑：到底在哪个窗口执行？

你使用 Windows 11＋WSL 2 Ubuntu 24.04，Ubuntu 用户为 pei，之前的提示符是 `pei@lovelygirl:~$`。硬件为约 16 GB 内存、RTX 4060 Laptop GPU 约 8 GB 显存。以下除特别标明外，全部在 **Ubuntu 的 Bash** 执行。

| 名称 | 是什么 | 你的例子 |
|---|---|---|
| 终端 Terminal | 显示输入输出、承载命令行程序的界面 | Windows Terminal 的一个标签页 |
| Shell | 解释命令、启动程序、处理管道和变量 | Ubuntu 里的 Bash；Windows 里的 PowerShell |
| WSL 2 | 运行 Linux 环境的基础 | Ubuntu-24.04 发行版 |
| tmux | 在终端中管理可分离的会话和多个窗格 | 一边训练，一边看日志 |

Windows Terminal 并不等于 PowerShell：同一个终端程序里可以打开 PowerShell，也可以打开 Ubuntu。看见 `PS C:\...>` 时不要直接粘贴本课的 Bash 循环。

如果当前在 PowerShell，用下面命令进入 Ubuntu：

```powershell
wsl -d Ubuntu-24.04
```

本课不需要 Docker，也不要求真正训练模型。后面用 Python 标准库生成模拟日志即可。

### 进入已克隆仓库的课程目录

Ubuntu 中执行：

```bash
mkdir -p ~/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell/logs ~/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell/results
cd ~/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell
```

这里使用通用示例仓库位置；请先确认该目录确实是你的课程仓库，再执行后续命令。进入本课目录后，后面的相对路径均以本课目录为起点；只按练习需要创建 logs、results 等输出子目录，不另建独立学习文件夹。模拟脚本和日志放在本课目录，保留原有 code/、docs/ 文件。

## 1. 原课的目标与整体用法

原课聚焦四件事：过滤训练日志；用 tmux 同时管理多个任务；监控 CPU／内存／GPU；通过 SSH、scp、rsync 操作远程机器。

典型工作区是：

```text
tmux 会话 training
├─ 窗格 1：运行 Python，记录日志
├─ 窗格 2：观察 GPU 或系统资源
└─ 窗格 3：过滤 loss、错误和实验结果
```

先学会“命令在哪里运行、输入输出去哪儿、进程如何管理”，不需要一开始背完整 Bash 语法。

## 2. Step 1：认识你的 Shell

原课用以下命令查看 Shell：

```bash
echo "$SHELL"
```

【纠正】SHELL 通常记录用户的登录 Shell，不保证就是此刻正在执行命令的 Shell。Bash 中进一步检查当前进程：

```bash
ps -p $$ -o comm=
```

`$$` 在这里表示当前 Shell 的进程 ID。本文以 Bash 为准，不要求改装 zsh。

### 路径与目录操作

```bash
pwd
ls -la
cd ~/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell
cd ..
cd -
```

| 写法 | 含义 |
|---|---|
| `pwd` | 当前工作目录 |
| `ls -la` | 详细显示文件，包括以点开头的隐藏项 |
| `~` | 当前 Ubuntu 用户的家目录 |
| `.` / `..` | 当前目录／父目录 |
| `cd -` | 回到上一次工作目录 |
| `/mnt/c/Users/<用户名>` | Windows 用户目录在 WSL 中的通用路径示例 |
| `/mnt/d` | Windows D 盘挂载点 |

路径包含空格时加引号，例如 `cd "/mnt/d/My Projects"`。Linux 路径通常区分大小写。不要把 Windows 的 `D:\...` 原样当作 Bash 路径。

### 常用键盘操作

| 按键 | 作用 |
|---|---|
| Tab | 补全命令或路径 |
| Ctrl+R | 搜索历史命令，再按一次查找更早匹配 |
| Ctrl+L / clear | 清理当前显示，不删除日志文件或历史 |
| Ctrl+C | 向前台任务发送中断，通常用于停止它 |
| Ctrl+Z | 暂停前台任务；它还在，不等于退出 |
| `fg` | 把当前作业恢复到前台 |

Shell 中 Ctrl+C 不是普通的复制快捷键。复制／粘贴使用终端菜单或对应设置的快捷键。

### 为后续命令补充三个符号

- `"$var"`：展开变量并尽量保持为一个参数；单引号中的 `$var` 按原文字面保留。
- `$(命令)`：先执行内部命令，将其输出放进外层命令。
- `&&`：前一个命令成功后再执行后一个；`;` 则是不论成功失败都继续。

多行命令末尾的 `\` 表示续行，后面不要再加空格。终端出现 `>` 等待输入时，可能是引号没闭合或续行尚未结束；可用 Ctrl+C 取消当前未完成输入。

## 3. Step 2：管道与重定向

程序通常有三个标准通道：stdin（输入，0）、stdout（普通输出，1）、stderr（错误／诊断输出，2）。

| 符号 | 意义 |
|---|---|
| `>` | 将 stdout 写入文件，覆盖原内容 |
| `>>` | 将 stdout 追加到文件 |
| `2>` | 将 stderr 写入文件 |
| `2>&1` | 让 stderr 去当前 stdout 所指的位置 |
| `\|` | 将前一命令的 stdout 交给后一命令的 stdin |
| `tee 文件` | 同时在终端显示并写入文件；默认覆盖 |

### 先生成一份可用日志

在练习目录创建 fake_train.py：

```bash
nano fake_train.py
```

写入：

```python
import sys
import time

for epoch in range(1, 101):
    print(f"epoch {epoch} loss: {1 / epoch:.4f}", flush=True)
    if epoch == 30:
        print("ERROR demo-only: simulated diagnostic", file=sys.stderr, flush=True)
    time.sleep(0.1)
```

Ctrl+O、回车保存，Ctrl+X 退出。这里的 ERROR 是有意打印的模拟诊断，不是程序崩溃。脚本只使用 Python 标准库，直接运行 Ubuntu 的 python3 即可。

```bash
python3 -u fake_train.py > train.log 2> errors.log
```

大约十秒后结束。终端不显示训练输出，是因为它已经写入文件，不代表没有运行。`-u` 让 Python 输出不缓冲，便于实时观察。

### 对照原课处理日志

```bash
cat train.log | grep "loss" | wc -l
grep "loss:" train.log | awk '{print $NF}' > losses.txt
tail -n 5 losses.txt
```

第一条：读取日志 → 保留含 loss 的行 → 统计行数，预期 100。**原课称“loss 出现次数”，严格说这里统计的是匹配行数，同一行出现两次仍只算一行。** 这个例子也可以直接写 `grep -c "loss" train.log`。

第二条：awk 的 `$NF` 表示每行最后一个字段。本例最后一列刚好是 loss 数字；如果真实日志写成 `loss: 0.1 lr: 0.001`，照抄会取到学习率，必须根据实际格式修改。

原课的实时过滤：

```bash
tail -f errors.log | grep --line-buffered "ERROR"
```

该命令先显示现有文件尾部，再等待新内容，Ctrl+C 停止观察。`--line-buffered` 减少 grep 输出等待，但不能迫使训练程序及时写日志。日志文件可能被轮换／重建时，可考虑 `tail -F`。

### 合并输出和同时显示

```bash
python3 -u fake_train.py > train_full.log 2>&1
python3 -u fake_train.py 2>&1 | tee train_visible.log
```

重定向按从左到右处理，`> file 2>&1` 与 `2>&1 > file` 不相同。前者把两种输出都送到文件，后者通常仍让 stderr 留在原终端。[Bash 重定向参考](https://www.gnu.org/software/bash/manual/html_node/Redirections.html)

### 按最终准确率排序

先建立练习数据：

```bash
printf 'final_accuracy=0.82\n' > results/exp1.log
printf 'final_accuracy=0.91\n' > results/exp2.log
grep "final_accuracy" results/*.log | sort -t= -k2,2nr
```

`-t=` 指定分隔符，按第二字段数值倒序，预期 exp2 排在前面。这依赖日志使用 `final_accuracy=数值` 的格式。

## 4. Step 3：后台进程

```bash
python3 -u fake_train.py > background.log 2>&1 &
training_pid=$!
jobs -l
```

`&` 让 Shell 不必等待命令结束；`$!` 是刚启动的后台任务 PID。脚本只有约十秒，检查时已经完成也正常。

| 概念 | 含义 |
|---|---|
| PID | Linux 进程号 |
| `%1` | 当前 Shell 的第 1 个作业，不是 PID 1 |
| `jobs` | 当前 Shell 管理的作业列表，不是全系统进程 |
| `ps` | 查看进程状态 |

若 jobs 显示任务仍在运行，可用 `fg %1` 拉到前台；若用 Ctrl+Z 暂停了任务，可用 `bg %1` 恢复后台执行。

原课还介绍：

```bash
nohup python3 -u fake_train.py > nohup_train.log 2>&1 < /dev/null &
```

nohup 让程序忽略挂断信号；这里也断开了终端输入。它不能提供重新接入原交互界面的能力，要查看日志。仅有 `&` 不提供可靠的终端断开保护；具体存活行为还受 Shell 和会话设置影响。

### 结束任务：先看清楚再结束

原课用 `kill $(pgrep -f "train.py")`，可能匹配多个任务。学习时先查看：

```bash
pgrep -af '[f]ake_train.py'
```

确认仍是刚启动的任务，再对它的实际 PID 使用 kill。若上面的 training_pid 仍对应这次任务，可核对：

```bash
ps -p "$training_pid" -o pid,etime,args
```

确认后才执行 `kill "$training_pid"`。默认发送 TERM，请求退出；不必一开始使用 kill -9。不要在无关任务运行时试用按名称批量杀进程的别名。

## 5. Step 4：tmux 会话与分屏

【设备适配】安装工具只需在 Ubuntu 中做。若尚未安装：

```bash
sudo apt-get update
sudo apt-get install -y tmux htop bc
```

如果 apt 下载超时，可沿用之前终端代理，在确有 http_proxy／https_proxy 变量时，用 `sudo --preserve-env=http_proxy,https_proxy apt-get update` 等命令保留变量。Docker 服务代理不会自动替 apt 配置网络。

### 创建与连接

```bash
tmux new -s training
```

进入后通常会看到底部状态栏。tmux 的层次为“会话 → 窗口 → 窗格”。本课先用一个会话、一个窗口、三个窗格即可。

| 目的         | 命令 / 快捷键               | 含义                     |
| ---------- | ---------------------- | ---------------------- |
| 创建会话       | `tmux new -s train`    | 创建名为 `train` 的 session |
| 查看所有会话     | `tmux ls`              | 看当前有哪些 session         |
| 暂时离开       | `Ctrl+B → d`           | detach，任务继续运行          |
| 重新进入       | `tmux attach -t train` | 回到 `train`             |
| 简写进入       | `tmux a -t train`      | 同上                     |
| 左右分屏       | `Ctrl+B → %`           | 创建左右 pane              |
| 上下分屏       | `Ctrl+B → "`           | 创建上下 pane              |
| 切换 pane    | `Ctrl+B → 方向键`         | 在窗格之间移动                |
| 显示 pane 编号 | `Ctrl+B → q`           | 显示 0、1、2…              |
| 新建 window  | `Ctrl+B → c`           | 新建一个完整窗口               |
| 下一个 window | `Ctrl+B → n`           | 切到下一个窗口                |
| 上一个 window | `Ctrl+B → p`           | 切到上一个窗口                |
| 关闭当前 pane  | `exit`                 | 结束当前窗格里的 shell         |
如果你确定整个 `train` 会话都不要了，在 tmux 外面执行：

```
tmux kill-session -t train
```

这些是前缀后再按键，不是一直按住 Ctrl+B 同时输入 d。原课“水平／垂直分割”的说法容易混淆，记住最后是上下还是左右即可。[tmux 官方入门](https://github.com/tmux/tmux/wiki/Getting-Started)

### 在三个窗格中分工

训练窗格：

```bash
cd ~/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell
python3 -u fake_train.py > logs/experiment.log 2>&1
```

监控窗格：

```bash
watch -n 1 nvidia-smi
```

日志窗格：

```bash
cd ~/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell
tail -f logs/experiment.log | grep --line-buffered "loss"
```

先启动写日志的任务，再启动 tail，以免文件尚不存在。模拟脚本不使用 GPU，GPU 利用率接近 0 属正常现象。

每个窗格都是独立 Shell。进入真实项目时检查 pwd 与解释器，并在需要的窗格激活项目虚拟环境，不要假设一个窗格中的 cd／source 会同步改变其他窗格。

### WSL 上“保持运行”的边界

tmux 支持分离和重新连接，但它仍是 Ubuntu 内的进程。保持 Windows 与 WSL 运行时可继续工作；不能让任务在 Windows 关机、重启、WSL 被终止后继续。睡眠期间也不能期待计算持续推进。

PowerShell 中 `wsl --shutdown` 会关闭 WSL 环境；`wsl --terminate Ubuntu-24.04` 会终止该发行版。运行训练时不要把它们当普通的“关闭窗口”操作。[Microsoft WSL 命令说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)

远程训练则要在 **SSH 登录后的远程服务器里**启动 tmux。本地电脑休眠并不等于远程服务器休眠，但本地 WSL 的任务不会因此变成云端任务。

练习结束后，先在各窗格停止观察／任务，再输入 exit 关闭 Shell。`tmux kill-session -t training` 是终止整场会话，不是分离，不能用来保留训练。

## 6. Step 5：htop、nvtop 与 nvidia-smi

### htop：看 CPU、内存和进程

```bash
htop
```

原课主要按键：F6 排序、F5 树形显示、`/` 搜索、F9 向选中进程发信号，q 退出。笔记本功能键可能需配合 Fn，取决于键盘设置。

htop 主要观察 Linux 环境；Windows 应用的资源使用还要结合任务管理器。系统内存与显存不同：内存不足和 CUDA out of memory 不是同一个问题。

### NVIDIA 显卡监控

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

`watch` 每秒重复执行并刷新屏幕，Ctrl+C 退出。若 nvidia-smi 不在 PATH，可检查 `/usr/lib/wsl/lib/nvidia-smi`。

原课按 GPU 进程查询：

```bash
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

【设备适配】WSL 中部分 NVML／nvidia-smi 查询和进程统计存在限制。显示 N/A、进程表不完整或 nvtop 无法显示全部信息，不应直接当作 CUDA 计算失败；结合实际计算验证与 Windows 监控判断。[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

nvtop 可作为可选工具，`sudo apt-get install nvtop` 后尝试运行。先掌握 nvidia-smi 即可，不必为完成本课强求每个指标都可用。

### 怎么判断训练慢？

- GPU 利用率低：可能在读数据、CPU 预处理、等待网络，也可能程序本来就在 CPU 上。
- 显存使用接近上限：留意模型、batch 和缓存，但高占用本身不代表错误。
- Linux 内存压力大：检查数据是否一次装入大列表、worker 是否太多。
- 磁盘满：检查日志、模型检查点、HF 缓存和 Docker 数据，不能只看 GPU。

这些是排查方向，单次截图不能确定根因。

## 7. Step 6：SSH 与远程文件传输

暂时没有远程服务器也可以先学语法，不需要为了本课购买云 GPU。下面的 user、gpu-box-ip 和文件名都是占位符，不能原样当真实地址执行。

### 登录与密钥

```bash
ssh user@gpu-box-ip
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip
```

ssh 命令在本地 Ubuntu 执行，登录成功后命令运行在远程机器。用 hostname、pwd 辅助确认位置。私钥应是你实际持有且被目标服务器授权的密钥。

服务器若使用非默认端口，ssh 用小写 `-p`，scp 用大写 `-P`；两者不要混用。

### scp：复制文件

```bash
scp model.pt user@gpu-box-ip:~/models/
scp user@gpu-box-ip:~/results/metrics.json ./
```

方向由源、目标的位置决定。第一条本地到远程，第二条远程到本地；远程目录需要存在。命令应从能访问所需本地文件的 Ubuntu 目录运行。

### rsync：同步目录

```bash
rsync -avzn ./data/ user@gpu-box-ip:~/data/
```

先用 `-n` 预览，确认后再去掉 n 执行。`-a` 为归档模式，`-v` 显示详情，`-z` 压缩传输。双方通常都需要 rsync。

`./data/` 末尾斜杠表示复制目录内容；没有斜杠时复制目录本身，目标布局可能不同。已压缩权重不一定从再次压缩获益，也不要默认加 `--delete`。

### 端口转发：在本地看远程 Jupyter

```bash
ssh -N -L 127.0.0.1:8888:localhost:8888 user@gpu-box-ip
```

在本地 Ubuntu 执行并保持连接。远端 Jupyter 必须已在远程 8888 端口监听；隧道不负责启动它。`-N` 表示只建立转发，不执行远程命令。右侧 localhost 从远端看，左侧 8888 是本地监听端口。[OpenSSH 手册](https://man.openbsd.org/ssh)

Windows 浏览器通常可通过 WSL 的 localhost 转发打开 `http://localhost:8888`，仍需 Jupyter 的 token。若本地端口已占用，把左侧端口换成 8889，浏览器相应打开 8889；远端端口仍是 8888。

### SSH 配置文件

在 `~/.ssh/config` 中合并实际服务器配置，例如：

```sshconfig
Host gpu
    HostName YOUR_SERVER_IP
    User YOUR_REMOTE_USER
    IdentityFile ~/.ssh/my_gpu_key
    Port 22
```

替换地址、用户名和密钥后，可用 `ssh gpu`。此文件属于 Ubuntu 用户，和 Windows 用户目录里的 .ssh/config 是不同文件，不会自动同步。

## 8. Step 7：为常用命令设置别名

原课提供 shell_aliases.sh，包含 GPU、虚拟环境、日志、磁盘、tmux、传输与实验目录工具。应先阅读，再选择需要的部分。

对于 Bash，把自己的别名合并到 `~/.bashrc`，不要覆盖整个文件。可先用 nano 打开，在末尾添加：

```bash
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias ae='source .venv/bin/activate'
alias tls='tmux ls'
alias ta='tmux attach -t'
alias diskuse='df -h .'
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

应用到当前 Shell：

```bash
source ~/.bashrc
type gpu
type ae
```

这里 gpu 简化为完整 nvidia-smi 输出，减少 WSL 对个别查询字段的兼容性影响。`ae` 要在存在 .venv 的项目目录用，`watchloss` 要在存在 logs/*.log 的目录用。

若要加载原课整份脚本，应在 bashrc 中使用不依赖当前工作目录的路径，例如 `source "$HOME/projects/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh"`。这是通用示例：请按实际克隆位置调整 `$HOME/projects/ai-engineering-from-scratch`。原文的相对路径会依赖打开 Shell 时的当前目录，不适合直接作为通用启动配置。

【纠正／补充】整份原课脚本有几个需要先知道的行为：

- killtraining 和不带参数的 killtrain 会按模式批量结束 Python 训练进程，可能影响多个实验。
- bigfiles／bigmodels 原写法用 xargs 处理路径，带空格的文件名可能被拆开；下一节提供 Ubuntu 可用的改进版。
- checkgpu／checkenv 依赖当前 Python 已安装 torch，当前环境没有包并不代表显卡坏了。
- hfdownload 不带文件名时尝试克隆整个模型仓库，不是只下载小配置文件。

本课练习不必为了方便而启用所有别名。

## 9. Step 8：常见 AI 终端组合

### 记录日志，并正确判断成功

原课在训练后用邮件通知，但邮件命令需要额外配置；分号也会让失败后照样发出完成消息。这里改成本地、可检查退出状态的版本：

```bash
set -o pipefail
if python3 -u fake_train.py 2>&1 | tee logs/run.log; then
    echo "任务成功结束"
else
    echo "任务失败，请检查 logs/run.log"
fi
```

默认管道状态通常来自最后一个命令；pipefail 能避免 Python 失败而 tee 成功时误判整体成功。这里的成功指退出状态为 0，不能只凭日志中有没有 ERROR 判断；模拟脚本故意打印 ERROR 但仍正常结束。

### 比较两份实验结果

```bash
diff <(grep "accuracy" results/exp1.log) <(grep "accuracy" results/exp2.log)
```

`<(...)` 是 Bash 的进程替换，把命令输出交给 diff 当作可读输入。diff 无输出通常表示相同；返回状态 1 表示有差异，不一定是命令故障。

### 找最大的模型文件

Ubuntu 中可写成：

```bash
find . -type f \( -name '*.pt' -o -name '*.safetensors' \) -printf '%s\t%p\n' | sort -nr | head -n 20
```

第一列是文件逻辑大小（字节），后面是路径；此写法能保留带空格的文件名，极端的带换行文件名不在此练习范围。它与 du 的磁盘实际占用口径不同。这里只列出，不自动删除。

### 下载、解压、磁盘与环境

原课 wget 中的 model 地址是占位示例，不能当成一定存在的权重 URL。下载真实文件前确认仓库、文件名、权限和大小；上一课的 hf_hub_download 更适合学习模型缓存。

已有 dataset.tar.gz 时，先查看再解压：

```bash
tar tzf dataset.tar.gz | head
mkdir -p data
tar xzf dataset.tar.gz -C ./data/
```

查看占用：

```bash
df -h .
du -sh ./data ./logs
```

WSL 虚拟文件系统报告的剩余容量不能替代 Windows 实际分区余量。你此前磁盘空间较紧张，日志追加、模型检查点和缓存都要计算在内。

统计 Python 源文件行数，处理带空格路径：

```bash
find . -type f -name '*.py' -print0 | xargs -0 -r wc -l
```

应在小型项目目录执行，避免把 .venv 或其他依赖源码也算进去。

原课 `env | grep -i cuda`、`env | grep -i torch` 只查看名字或值中含对应文字的环境变量。没有输出不代表没装 CUDA／PyTorch。

## 10. 原课四个练习：操作与完成标准

### 练习一：三个 tmux 窗格，分离再连接

安装 tmux、htop 后，在练习目录执行：

```bash
tmux new -s terminal-practice
```

第一个窗格运行 htop。按 Ctrl+B 再按 `%` 建立第二个窗格，运行 `watch -n 1 date`。在第二个窗格按 Ctrl+B 再按 `"`，建立第三个窗格，进入练习目录运行：

```bash
python3 -u fake_train.py
```

按 Ctrl+B 再按 d 分离，在外面的 Shell 执行：

```bash
tmux ls
tmux attach -t terminal-practice
```

完成标准：能看到三个窗格，并能回到原会话。Python 只有约十秒，回来时已经完成是正常的，不能因此认为 tmux 没保住任务。

### 练习二：添加别名并重载

按 Step 7 在 bashrc 加入所选别名，source 后运行 `type gpu`、`gpu` 和 `tls`。能说明“别名就是常用命令的短名字”，并知道 ae 的当前目录要求，即达到目标。

如果使用课程原脚本，先阅读内容，再使用 Step 7 中已按你的仓库位置填写的绝对路径 source。

### 练习三：原课模拟日志与 loss 提取

安装 bc 后，运行原课同等逻辑的循环：

```bash
for i in $(seq 1 100); do
    echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"
    sleep 0.1
done > fake_train.log
```

seq 生成 1～100，bc 计算四位小数除法，每轮暂停 0.1 秒，整个循环的输出写到文件。bc 是截断小数，与前面 Python 格式化四舍五入的结果可能在末位不同。

```bash
grep -c 'loss:' fake_train.log
tail -n 5 fake_train.log
grep 'loss:' fake_train.log | awk '{print $NF}' > fake_losses.txt
wc -l fake_losses.txt
head -n 3 fake_losses.txt
```

完成标准：匹配行数和导出行数都为 100，导出文件每行只有一个 loss 数值。想实时看增长时，在另一窗格运行 tail -f；任务结束后 tail 仍会等待，需手动 Ctrl+C。

### 练习四：SSH 配置

已有服务器：按 Step 6 写入真实配置，用 `ssh -G gpu` 检查展开结果，再 `ssh gpu` 实际连接。

没有服务器：可在配置中添加一个仅练语法的条目：

```sshconfig
Host local-practice
    HostName localhost
    User pei
    Port 22
```

然后执行：

```bash
ssh -G local-practice | grep -E '^(hostname|user|port) '
```

该命令只输出解析后的配置，不建立网络连接。这样完成的是语法练习，不是实际 SSH 登录验证。`ssh localhost` 真正登录还需要本地 SSH 服务正在运行并允许认证，WSL 不会因为有 ssh 客户端就自动满足条件。

## 11. 容易混淆的点与复习顺序

| 问题 | 先检查 |
|---|---|
| Bash 命令在窗口中报奇怪语法错误 | 是否实际在 PowerShell，是否粘贴了提示符 |
| 运行后看不到输出 | 是否用了 >；Python 是否缓冲；程序是否已退出 |
| Ctrl+Z 后程序不动 | 它被暂停了，用 jobs 查看、fg 恢复 |
| 新终端看不到 jobs | jobs 属于原 Shell；用 ps／tmux 查看对应任务 |
| tmux 分离后怎么回来 | tmux ls，再 attach 指定会话 |
| 重启后 tmux 会话没了 | 正常，tmux 不是断点恢复或开机续训系统 |
| 新窗格找不到 Python 包 | 核对当前目录、虚拟环境与解释器 |
| alias 新终端不生效 | 是否编辑了 Bash 的 bashrc、source 路径是否正确 |
| localhost SSH 拒绝连接 | 是否有 SSH 服务监听，不能只检查客户端 |
| WSL GPU 监控显示 N/A | 先考虑 WSL 支持范围，再结合实际计算验证 |

建议先完成模拟日志与管道，再练后台作业和 tmux，然后看资源监控；远程服务器部分先理解方向与路径即可。能解释“进程在哪台机器上、输出写到哪里、关闭什么会结束它”，比背命令更重要。

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


