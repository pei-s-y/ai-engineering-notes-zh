
课程原文：[Git & Collaboration](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/02-git-and-collaboration/docs/en.md)

## 一、本课学习目标

学完这一课，需要掌握四件事：

1. 配置 Git 用户身份。
2. 掌握 `add → commit → push` 日常工作流。
3. 使用分支进行独立实验，并把分支合并回主分支。
4. 使用 `.gitignore` 排除模型权重、大文件和临时文件。
5. 使用 `git log` 查看项目的发展历史。
6. 理解 Git、GitHub、Fork、Clone、Branch、Commit、Merge 和 Remote 等概念。

本课的最终目标不是背命令，而是能够安全地保存代码、回看历史、进行实验并把工作备份到 GitHub。

---

## 二、Git 和 GitHub不是同一个东西

### 1. Git

Git 是安装在本地电脑上的版本控制工具，主要负责：

- 记录文件变化；
- 保存不同时间点的代码；
- 创建和切换分支；
- 合并代码；
- 恢复以前的版本；
- 支持多人协作。

即使电脑没有联网，Git 的大部分功能仍然可以使用，例如：

```
git status
git add
git commit
git branch
git log
```

### 2. GitHub

GitHub 是托管 Git 仓库的网络平台，主要负责：

- 保存远程仓库；
- 在线备份代码；
- 多人协作；
- Fork 别人的项目；
- 提交 Pull Request；
- 管理 Issue、讨论和代码审查。

可以简单理解为：

```
Git：本地管理版本
GitHub：在线托管和协作
```

GitHub 不是 Git 的必要组成部分。除了 GitHub，还可以使用 GitLab、Gitee 等平台。

---

## 三、什么是版本控制

版本控制就是持续记录项目中文件的变化历史。

例如，你正在修改一个机器学习项目：

```
版本1：模型能够运行
版本2：增加特征预处理
版本3：修改损失函数
版本4：实验失败
```

如果每一个重要阶段都进行了 Commit，即使版本4失败，也可以找到版本3时的代码。

没有版本控制时，人们可能会创建大量混乱的文件：

```
model.py
model_new.py
model_new2.py
model_final.py
model_final_really.py
model_final_really_new.py
```

使用 Git 后，不需要靠修改文件名保存版本。文件可以一直叫 `model.py`，不同版本由 Commit 历史管理。

---

## 四、Repository：仓库

Repository，简称 Repo，中文叫“仓库”。

一个 Git 仓库包括：

- 项目文件；
- 文件的修改记录；
- 分支信息；
- Commit 历史；
- 远程仓库配置；
- Git 的内部数据。

本地仓库的根目录一般包含一个隐藏的 `.git` 目录：

```
ai-engineering-from-scratch/
├── .git/
├── phases/
├── projects/
├── README.md
└── ...
```

`.git` 目录保存项目的版本历史。不要随意删除它，否则普通文件虽然还在，但这个目录将不再是原来的 Git 仓库。

查看当前目录是否位于 Git 仓库中：

```
git status
```

如果不在 Git 仓库中，会出现类似提示：

```
fatal: not a git repository
```

---

## 五、Git 的四个区域

这是本课最重要的知识点。

```
工作区             暂存区             本地仓库             远程仓库
Working Directory  Staging Area       Local Repository     GitHub
       │                 │                    │                 │
       └── git add ─────>│                    │                 │
                         └── git commit ─────>│                 │
                                              └── git push ───>│
```

### 1. 工作区 Working Directory

工作区就是电脑上实际看到和编辑的文件。

例如，你修改了：

```
model.py
README.md
```

这些修改首先只存在于工作区中。

保存编辑器中的文件，只是把内容写入硬盘，并没有创建 Git 历史。

### 2. 暂存区 Staging Area

暂存区表示“下一次 Commit 准备包含哪些变化”。

把一个文件放入暂存区：

```
git add model.py
```

添加多个指定文件：

```
git add model.py README.md
```

添加当前目录中的全部变化：

```
git add .
```

`git add` 并不是永久保存，也没有上传到 GitHub。它只是选择下一次 Commit 要包含的内容。

暂存区的价值在于：你可以修改很多文件，但只提交其中一部分。

### 3. 本地仓库 Local Repository

执行 Commit 后，暂存区中的内容会成为本地仓库的一次历史快照：

```
git commit -m "Add action encoder"
```

Commit 会记录：

- 哪些文件发生了变化；
- 具体改变了什么；
- 提交者是谁；
- 提交时间；
- Commit 说明；
- 上一个 Commit 是什么。

Commit 只保存在本地电脑，并不会自动上传到 GitHub。

### 4. 远程仓库 Remote Repository

执行 Push 后，本地 Commit 才会上传到远程仓库：

```
git push
```

也可以明确指定远程仓库和分支：

```
git push origin main
```

其中：

- `origin`：远程仓库的默认别名；
- `main`：准备推送的分支名称。

---

## 六、日常 Git 工作流

课程给出的核心顺序是：

```
git status
git add file.py
git commit -m "Add perceptron implementation"
git push origin main
```

可以记成：

```
查看 → 选择 → 保存 → 上传
status → add → commit → push
```

### 第一步：查看状态

```
git status
```

它会告诉你：

- 当前位于哪个分支；
- 哪些文件被修改；
- 哪些文件还未被跟踪；
- 哪些修改已经进入暂存区；
- 是否有内容可以提交。

`git status` 是最安全、最常使用的命令之一。不确定下一步做什么时，先运行它。

### 第二步：选择要提交的内容

```
git add file.py
```

这一步把指定变化加入暂存区。

执行后应再次查看：

```
git status
```

此时被暂存的文件通常会出现在 `Changes to be committed` 下。

### 第三步：创建本地快照

```
git commit -m "Add perceptron implementation"
```

好的 Commit 信息应当说明本次提交做了什么，例如：

```
Add GPU availability check
Implement linear regression training loop
Fix tensor shape mismatch
Update experiment documentation
Ignore model checkpoint files
```

不推荐：

```
update
change
test
123
修改
```

Commit 信息应该让未来的自己知道这次改动的目的。

### 第四步：上传到远程仓库

```
git push origin main
```

第一次推送新分支时通常使用：

```
git push -u origin my-progress
```

`-u` 会在本地分支和远程分支之间建立跟踪关系。以后通常直接运行：

```
git push
```

---

## 七、`git add`、`commit` 和 `push` 的区别

|命令|作用范围|是否创建历史|是否上传GitHub|
|---|---|---|---|
|保存文件|工作区|否|否|
|`git add`|工作区 → 暂存区|否|否|
|`git commit`|暂存区 → 本地仓库|是|否|
|`git push`|本地仓库 → 远程仓库|上传已有历史|是|

必须记住：

```
add 选择内容
commit 保存历史
push 上传历史
```

正确顺序是：

```
git add → git commit → git push
```

不能先 Push 尚未 Commit 的工作区修改，因为 Push 上传的是 Commit，不是编辑器中尚未提交的文件。

---

## 八、远程仓库与 `origin`

查看当前仓库配置的远程地址：

```
git remote -v
```

可能显示：

```
origin  https://github.com/username/project.git (fetch)
origin  https://github.com/username/project.git (push)
```

`origin` 只是一个常用别名，不是特殊的服务器名称。它通常代表 Clone 时使用的远程仓库。

这里出现两行，是因为 Git 分别显示：

- Fetch 使用的地址；
- Push 使用的地址。

查看更详细的信息：

```
git remote show origin
```

---

## 九、Fetch、Pull 和 Push

### `git fetch`

```
git fetch
```

作用是下载远程仓库中的新提交和分支信息，但通常不会立刻修改当前工作区文件。

可以理解为：

```
“检查并下载远程发生了什么，但先不把变化合并进当前工作。”
```

### `git pull`

```
git pull
```

作用是获取远程变化，并将其整合到当前分支。

初学阶段可以粗略理解为：

```
git pull ≈ 获取远程更新 + 整合到当前分支
```

### `git push`

```
git push
```

把本地已经创建的 Commit 上传到远程仓库。

方向可以这样记：

```
pull：从远程拉到本地
push：从本地推到远程
```

---

## 十、Branch：分支

### 1. 分支的意义

分支用于隔离不同的工作。

例如：

```
main
 ├── experiment/new-optimizer
 ├── feature/data-loader
 └── fix/tensor-shape
```

你可以在实验分支上修改优化器，即使实验失败，也不会直接破坏 `main`。

### 2. 分支不是完整复制项目

课程中的简化说法可能把 Branch 理解成“一份副本”，但从 Git 的内部机制看，分支本质上是一个指向某次 Commit 的可移动指针。

当你在该分支上继续 Commit 时，分支指针会向前移动。

### 3. 创建并切换分支

课程使用：

```
git checkout -b experiment/new-optimizer
```

它完成两件事：

1. 创建 `experiment/new-optimizer`；
2. 立即切换到该分支。

现代 Git 也可以写成：

```
git switch -c experiment/new-optimizer
```

两种方式都可以，本课主要记住课程使用的 `checkout -b`。

### 4. 查看当前分支

```
git branch
```

示例：

```
  main
* experiment/new-optimizer
```

星号代表当前分支。

也可以只显示当前分支：

```
git branch --show-current
```

### 5. 切换已有分支

课程写法：

```
git checkout main
```

现代写法：

```
git switch main
```

### 6. 合并分支

假设已经在实验分支完成修改并 Commit：

```
git checkout main
git merge experiment/new-optimizer
```

含义是：

1. 先切换到接收变化的 `main`；
2. 把 `experiment/new-optimizer` 的变化合并到当前 `main`。

记忆方法：

```
先站到目标分支，再把来源分支合进来。
```

如果两个分支修改了同一文件的同一部分，可能产生 Merge Conflict，也就是合并冲突。Git 会要求人工决定保留哪一部分。

---

## 十一、Fork 与 Clone

### Fork

Fork 是在 GitHub 上将别人的仓库复制到自己的账户中。

例如：

```
原作者：
rohitg00/ai-engineering-from-scratch

Fork之后：
你的用户名/ai-engineering-from-scratch
```

Fork 发生在 GitHub 服务器上。

### Clone

Clone 是把远程仓库完整下载到本地电脑：

```
git clone https://github.com/你的用户名/ai-engineering-from-scratch.git
```

Clone 不只下载当前文件，也会下载 Git 历史和远程配置。

### 二者区别

|操作|发生位置|结果|
|---|---|---|
|Fork|GitHub → GitHub|在自己的账户下创建项目副本|
|Clone|GitHub → 本地电脑|把仓库下载到电脑|
|Branch|同一个Git仓库内部|创建独立开发线|
|Commit|本地仓库|创建历史快照|
|Push|本地 → GitHub|上传本地Commit|

课程要求先 Fork，是因为普通用户没有权限直接向课程作者的仓库 Push。[课程说明](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/00-setup-and-tooling/02-git-and-collaboration/docs/en.md)

---

## 十二、针对课程仓库的操作流程

### 第一步：在 GitHub 上 Fork

进入课程仓库：

[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)

点击页面右上方的 **Fork**，把仓库复制到自己的 GitHub 账户。

### 第二步：Clone 自己的 Fork

```
git clone https://github.com/你的用户名/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch
```

### 第三步：创建学习进度分支

```
git checkout -b my-progress
```

### 第四步：学习并修改文件

例如添加自己的笔记、修改练习代码或记录学习进度。

### 第五步：提交

```
git status
git add 文件名
git commit -m "Add lesson 02 learning notes"
```

### 第六步：推送

```
git push -u origin my-progress
```

以后继续在这个分支工作时：

```
git push
```

---

## 十三、`.gitignore`

`.gitignore` 是仓库中的一个普通文本文件，用于告诉 Git 忽略哪些文件或目录。

AI 项目会产生很多不适合 Git 管理的内容，例如：

- Python 缓存；
- 虚拟环境；
- 模型权重；
- 数据集；
- 实验日志；
- API 密钥；
- 编辑器配置；
- 临时输出。

典型内容：

```
# Python缓存
__pycache__/
*.pyc

# 虚拟环境
.venv/
venv/

# 环境变量和密钥
.env

# Jupyter临时文件
.ipynb_checkpoints/

# 模型权重
*.pt
*.pth
*.safetensors

# 实验输出
checkpoints/
runs/
wandb/

# 编辑器配置
.vscode/
.idea/
```

### 通配符含义

```
*.pt
```

表示忽略所有以 `.pt` 结尾的文件。

```
checkpoints/
```

表示忽略整个 `checkpoints` 目录。

```
.env
```

表示忽略名为 `.env` 的文件。

### 为什么不提交模型权重

`.pt`、`.pth` 和 `.safetensors` 通常是模型权重或检查点，可能达到数百 MB 甚至数 GB。

普通 Git 不适合频繁管理大型二进制文件，因为：

- 仓库会迅速膨胀；
- 即使删除文件，历史中仍可能保存旧版本；
- Clone 和 Pull 会变得很慢；
- GitHub 对单个文件大小有限制。

模型文件一般放在：

- Hugging Face Hub；
- 云盘或对象存储；
- 实验服务器；
- Git LFS；
- 本地检查点目录。

### `.gitignore` 的重要限制

`.gitignore` 主要对尚未被 Git 跟踪的文件生效。

如果 `model.pt` 已经被 Commit，后来再添加：

```
*.pt
```

Git 仍可能继续跟踪它。`.gitignore` 不会自动删除以前的提交历史。

---

## 十四、查看 Commit 历史

课程要求使用：

```
git log --oneline
```

它会以简洁格式显示历史：

```
a1b2c3d Add Git collaboration lesson
d4e5f6g Add development environment verification
h7i8j9k Initial commit
```

每一行包括：

- Commit 哈希值的缩写；
- Commit 信息。

查看最近五条：

```
git log --oneline -5
```

查看分支结构：

```
git log --oneline --graph --decorate --all
```

这些参数的含义：

- `--oneline`：每个 Commit 显示一行；
- `--graph`：显示分支图；
- `--decorate`：显示分支和标签名称；
- `--all`：显示全部分支。

`git log` 不只是查看谁改了代码，还能帮助理解项目如何逐步发展。课程练习要求阅读仓库历史，观察各课程是如何陆续加入的。

---

## 十五、本课核心术语

|术语|准确定义|
|---|---|
|Version Control|记录并管理文件随时间发生的变化|
|Repository|项目文件及其完整版本历史|
|Working Directory|当前实际编辑文件的区域|
|Staging Area|准备进入下一次Commit的变化|
|Commit|项目在某一时刻的历史快照|
|Branch|指向Commit并随新提交移动的独立开发线|
|Merge|将一个分支的变化整合到另一个分支|
|Remote|位于其他位置的仓库副本，通常托管在GitHub|
|Origin|Clone后默认远程仓库的常用名称|
|Fork|在GitHub账户间复制仓库|
|Clone|把远程仓库下载到本地|
|Push|把本地Commit上传到远程|
|Fetch|下载远程历史，但通常不直接改变当前工作区|
|Pull|获取远程变化并整合到当前分支|
|`.gitignore`|声明不希望Git跟踪的文件或目录|

---

## 十六、常见错误

### 1. 在错误目录运行 Git

出现：

```
fatal: not a git repository
```

通常说明当前目录不在仓库中。

先检查：

```
Get-Location
Get-ChildItem
```

然后进入项目目录：

```
cd "仓库路径"
```

### 2. 修改文件后直接 Push

Push 只能上传 Commit。正确顺序：

```
git add 文件名
git commit -m "说明"
git push
```

### 3. `git add .` 前不检查

`git add .` 会添加当前目录中的大量变化，可能误加入：

- 数据集；
- 模型权重；
- `.env`；
- 临时文件；
- 虚拟环境。

因此推荐：

```
git status
git add 指定文件
git status
```

确认无误后再 Commit。

### 4. Clone 原作者仓库后尝试 Push

如果 `origin` 是：

```
https://github.com/rohitg00/ai-engineering-from-scratch.git
```

你通常没有写入权限。课程要求先 Fork，再使用自己账户中的仓库。

检查：

```
git remote -v
```

### 5. 在 `main` 上直接做高风险实验

小修改可以直接进行，但独立实验更适合创建分支：

```
git checkout -b experiment/test-model
```

### 6. 把密钥提交到 Git

不要提交：

```
.env
API Key
访问令牌
数据库密码
云服务凭据
```

即使后来删除，密钥也可能仍存在于 Git 历史里。发生这种情况后通常需要立即撤销并重新生成密钥。

### 7. 把大型模型文件提交到普通 Git

训练前就应配置 `.gitignore`：

```
*.pt
*.pth
*.safetensors
checkpoints/
```

---

## 十七、本课完整练习

课程要求完成三个练习。

### 练习一：完成 Fork—Branch—Commit—Push

目标：

1. Fork 课程仓库；
2. Clone 自己的 Fork；
3. 创建 `my-progress` 分支；
4. 创建或修改一个文件；
5. `git add`；
6. `git commit`；
7. `git push`。

完成标准：

- GitHub 上能看到 `my-progress` 分支；
- 分支中能看到自己修改的文件；
- GitHub 上能看到自己的 Commit 信息。

### 练习二：创建 `.gitignore`

至少添加：

```
*.pt
*.pth
*.safetensors
```

然后验证：

```
git status
```

创建一个符合忽略规则的测试文件后，它不应作为普通未跟踪文件出现。

### 练习三：阅读项目历史

运行：

```
git log --oneline
```

观察：

- 最近有哪些 Commit；
- Commit 信息通常如何描述变化；
- 课程文件是怎样逐步加入仓库的；
- 分支和合并历史是什么样的。

---

## 十八、需要记住的最小命令表

```
# 查看状态
git status

# 查看远程仓库
git remote -v

# 查看当前分支
git branch --show-current

# 创建并切换分支
git checkout -b my-progress

# 暂存指定文件
git add 文件名

# 创建本地提交
git commit -m "描述本次改动"

# 第一次推送新分支
git push -u origin my-progress

# 后续推送
git push

# 切换到主分支
git checkout main

# 合并实验分支
git merge my-progress

# 查看简洁历史
git log --oneline

# 查看图形化分支历史
git log --oneline --graph --decorate --all
```

## 十九、一句话总结

这节课可以浓缩为：

> Git 用 Commit 管理本地代码历史，用 Branch 隔离实验，用 Merge 整合成果，用 Push 将提交备份到 GitHub；日常工作的核心顺序是 `status → add → commit → push`。

最关键的记忆口诀是：

```
Fork 是复制到自己的 GitHub
Clone 是下载到自己的电脑
Branch 是隔离新的工作
Add 是选择下一次提交内容
Commit 是保存本地历史
Push 是上传本地提交
Merge 是合并不同分支
Log 是查看项目历史
```
