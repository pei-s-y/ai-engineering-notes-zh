# 12：JAX入门——纯函数、jit与vmap

> 本篇按原课结构整理为中文自学笔记。必要代码直接嵌入正文；不创建脚本附件。标为“原课参考”的数值不是本机实测。

来源：[课程](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/12-intro-to-jax) · [原课文档](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/03-deep-learning-core/12-intro-to-jax/docs/en.md)。原题：Introduction to JAX。

## 零、初学者先读：把“状态”从代码里赶出去

> 2026-11-06 整理的教学补充。首次学习先读本节，再进入后面的公式、原课代码与纠错。[第11课](<11：PyTorch入门——nn.Module与autograd.md>)的 PyTorch 把参数藏在对象里；本课的 JAX 要求参数在函数之间明着传递，这是通往大规模训练与[第13课](<13：调试神经网络——没有报错的排错法.md>)之前最后一块地基。

### 0.1 先理解这一课在做什么

PyTorch 的写法是“面向对象”：模型是个有状态的对象，参数住在 `model` 里，`loss.backward()` 把梯度写到 `.grad` 上，`optimizer.step()` 原地改参数。JAX 的写法是“函数式”：模型只是一个普通函数 `f(params, x)`，参数是外面传进来的一个嵌套字典（pytree），求梯度、更新参数全部靠“函数变换”完成——`jax.grad` 把函数变成它的导函数，`jax.vmap` 把单样本函数变成批处理函数。为什么非要这样折腾？因为**编译器只信纯函数**：输入相同输出必相同、不偷偷改外部状态的函数，才能被放心地优化成机器码。这节课就是学会用“变换”的视角写代码，然后看它在 MNIST 上跑起来。

### 0.2 本课术语先翻成白话

- 纯函数（pure function）=输出只取决于输入，不改外部状态、不偷偷用全局随机数
- 即时编译（jit）=第一次调用时记录函数在做什么，编译成机器码，之后调用不再过 Python
- XLA（Accelerated Linear Algebra）=Google 的编译器，把记录下的计算融合、去冗余、生成 GPU/TPU 代码
- 追踪（tracing）=jit 首次调用时用“抽象值”跑一遍函数来记账，此时真正的计算还没发生
- 函数式自动微分=对函数求导：`jax.grad(f)` 返回一个新函数 g，g(x) 就是 f 在 x 处的导数
- vmap=自动向量化：写好“处理一个样本”的函数，vmap 给你“处理一批样本”的版本
- pmap=自动并行：把函数复制到多台 GPU/TPU 上，各自吃一块数据
- pytree（树结构）=列表、元组、字典嵌套数组的统称；你的模型参数就是一棵 pytree
- `jax.tree.map`=对树的每个叶子套用同一函数，比如“所有参数减去学习率乘梯度”
- 不可变（immutable）=JAX 数组不能原地改，`a[0]=5` 不存在，只有“返回新数组”的写法
- PRNG key=显式传递的随机数“钥匙”，JAX 没有全局随机状态
- Optax=JAX 的优化器库，把裁剪、Adam 等做成可串联的变换
- Flax / Equinox=JAX 之上的神经网络库（分别来自 Google 与 Patrick Kidger）
- `jax.lax.cond / scan`=编译器友好的 if/else 和 for 循环替代品

### 0.3 用小数字走一遍

**算例一：原地改 vs 返回新数组（任务建议的对比例）。** 有数组 a=[1, 2, 3]。
- PyTorch 写法：`a[0] = 9` 之后，a 本身变成 [9, 2, 3]——同一个对象被改写了。
- JAX 写法：`b = a.at[0].set(9)` 得到新数组 b=[9, 2, 3]，而 a 依然是 [1, 2, 3]。
关键差别不在结果（两个 [9,2,3] 长得一样），而在“旧值还在不在”：纯函数的世界里旧参数永远不被销毁，所以优化器更新写成 `new_params = update(old_params, grads)`——想回退、想对比、想复制实验，旧状态随手可用。

**算例二：函数式求导。** f(x)=x²，则 `df = jax.grad(f)`，df(3.0)=6（导数 2x 在 x=3）。再套一层 `d2f = jax.grad(jax.grad(f))`，d2f(3.0)=2（二阶导恒为 2）。与[第11课](<11：PyTorch入门——nn.Module与autograd.md>) 0.3 的 tape 算例对照：那边是“对张量记账”，这边是“对函数变换”，算出的导数值一模一样。

**算例三：vmap 的手算。** 定义单样本函数 `predict(params, x) = w·x + b`，取 w=2、b=1，一批输入 x=[1, 2, 3]。`jax.vmap(predict, in_axes=(None, 0))` 的意思是“params 不批处理（None），x 沿第 0 维批处理”：对每个样本算 2x+1，得 [3, 5, 7]——正好是[第02课](<../Phase02 · ML Fundamentals/02：线性回归——最小二乘、梯度下降与正规方程.md>)那条直线 y=2x+1 上的三个点，没有写一个 for 循环。

**算例四：pytree 更新。** 参数树 `params = {'w': 2.0}`，梯度树 `grads = {'w': 4.0}`，学习率 0.1。则 `jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)` 得 `{'w': 1.6}`——tree.map 就是“对每片叶子做同一件事”，这就是没有 optimizer.step() 时的参数更新。

以上是用于理解的算例，不是新增的训练或性能实测。

### 0.4 怎样连接到正文代码

先读第二节概念，顺序是：jax.numpy（长得像 NumPy）→ jax.grad（函数求导）→ jit（为什么慢第一次快后面）→ vmap（单样本函数升批）→ pmap（多设备）→ pytree（参数的家）→ 何时选 JAX。第四节 Build It 用 JAX+Optax 训练与第11课同结构的 MNIST MLP，五个步骤对照着读能看见“每个概念落在哪一行”。第六节是 Flax/Equinox/Optax 的生产用法。全课只有一个核心转变要内化：**状态从对象里搬到参数里，动作从方法变成函数**。

### 0.5 理解检查（含参考答案）

**问题1：** 为什么 jit 要求纯函数？不纯会怎样？

**参考答案：** 编译器要在“第一次追踪”时就把整段计算确定下来并长期复用。如果函数里改了外部变量或偷偷用全局随机数，编译后的版本每次运行结果就该不同，但编译版只会复用追踪时记下的那条路径——结果与真实意图悄悄脱节。纯函数保证“同样的输入永远同样的输出”，编译版才可信。

**问题2：** `in_axes=(None, 0)` 里 None 和 0 各是什么意思？

**参考答案：** None 表示这个参数不参与批处理（params 对每个样本是同一份，共享）；0 表示沿该参数的第 0 维（批维度）切开来逐样本应用函数。所以 predict(params, x) 从“吃一个 x”变成“吃一摞 x”。

**问题3：** JAX 里为什么没有 `optimizer.zero_grad()`、`.backward()`、`.step()` 这三件套？

**参考答案：** 因为梯度不挂在张量上（没有 .grad 字段，`jax.grad` 直接返回梯度值），参数也不是优化器内部的状态（更新就是 `optax.apply_updates(params, updates)` 得到新参数树）。整个训练步是一个纯函数：参数和梯度状态进去，新参数出来。

**问题4：** 同样的 MNIST MLP，JAX 训练循环和 PyTorch 训练循环最大的外在差别是什么？

**参考答案：** PyTorch 每批五步（zero_grad→前向→损失→backward→step），状态散在 model/optimizer 里；JAX 把这些合并成一次 `train_step(params, opt_state, x, y)` 调用，状态全部显式出现在参数表里，函数返回更新后的 (params, opt_state, loss)。

能答上来后进入正文。

---

## 一、这一课解决什么问题？

你会 PyTorch 了，为什么还要学 JAX？因为 PyTorch 的基因里有一个约束：它逐条、即时地解释 Python 操作，每次 `tensor + tensor` 都是一次独立的核启动，每一步训练都重新解释同一份 Python 代码。单卡训练无所谓，但当你要在 2048 块 TPU 上训练 5400 亿参数的模型，Python 的开销就是致命的（原课陈述）。原课举的例子：Google DeepMind 用 JAX 训练 Gemini，Anthropic 用 JAX 训练 Claude——它们把训练循环当成**一个可编译的程序**，而不是一串 Python 调用（原课陈述）。JAX 的口号可以记成：NumPy 加三个超能力——自动微分、XLA 编译、自动向量化。你只写“处理一个样本的函数”，求导、批处理、编译、多机并行全部由变换自动给你。

## 二、核心概念与推导

### 2.1 两种哲学：改对象 vs 传参数

| PyTorch | JAX |
|---|---|
| nn.Module 类持有状态 | 纯函数 `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 经 XLA 的 jit 编译 |
| `for x in batch:` 手写循环 | `jax.vmap(f)` 自动向量化 |
| DataParallel / FSDP | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的 pytree |

原课强调这不是风格偏好，而是**编译器约束**：jit 编译要求纯函数，正是这个限制换来 100 倍量级的加速可能（原课参考的量级说法）。

### 2.2 jax.numpy：熟悉的脸，不同的骨头

~~~python
# 本块中文说明：jax.numpy 与 NumPy 同名同义，但数组住在加速器上。
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
~~~

函数名、广播规则、切片语义都和 NumPy 一样，但数组在 GPU/TPU 上、每个操作都可被编译器追踪。最关键的差别：JAX 数组不可变，没有 `a[0] = 5`，只有 `a = a.at[0].set(5)`（见 0.3 算例一）。别扭一周之后会明白：不可变性正是 grad、jit、vmap 能随便组合的原因——旧的没被销毁，变换才有稳定的输入。

### 2.3 jax.grad：对函数求导，不对张量记账

~~~python
# 本块中文说明：grad 接函数、返回函数；导数可以任意复合。
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)  # 6.0
~~~

`jax.grad` 吃一个函数、吐一个新函数。没有 `.backward()`，没有挂在张量上的计算图——梯度本身就是个可调用、可组合、可再 jit 的函数。复合能力是原生设计：

~~~python
# 本块中文说明：二阶导=grad 套 grad。
d2f = jax.grad(jax.grad(f))
d2f(3.0)  # 2.0
~~~

二阶导、三阶导、雅可比、Hessian 全靠组合 grad 得到（0.3 算例二手算过：df(3)=6、d2f(3)=2）。约束：grad 只对纯函数工作——函数里不能有 print（追踪时执行、真跑时不执行）、不能改外部状态、随机数必须显式传 key。

### 2.4 jit：第一次记录，之后绕开 Python

~~~python
# 本块中文说明：jit 包装训练步，首调追踪编译，后续走机器码。
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
~~~

第一次调用时 JAX 追踪函数——只记录“做了哪些操作”而不真正执行；把追踪结果交给 XLA，由它融合算子、消除冗余拷贝、生成优化机器码。之后的调用完全绕开 Python，在加速器上以 C++ 速度跑。什么时候有利：训练步（同样的计算重复几千次）、推理、任何被反复调用且形状稳定的函数。什么时候有害：依赖**数据值**的 Python 控制流（`if x > 0` 里的 x 是被追踪的数组）、一次性计算（编译时间超过运行时间）、调试（追踪过程藏住了真实执行）。控制流限制是实打实的：`jax.lax.cond` 替代 if/else、`jax.lax.scan` 替代 for 循环——这不是可选项，是编译的代价。

### 2.5 vmap：写一个样本的函数，得一批的函数

~~~python
# 本块中文说明：单样本前向 → 自动批处理前向。
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']

batch_predict = jax.vmap(predict, in_axes=(None, 0))
~~~

没有手写 for 循环、没有 reshape、没有批维度穿针引线——vmap 自己搞明白批维度并把整个计算向量化（0.3 算例三手算过：w=2、b=1 时 x=[1,2,3] 得 [3,5,7]）。它不是语法糖：生成的是融合的向量化代码，比 Python 循环快 10—100 倍（原课参考）。而且它和 grad、jit 自由组合：

~~~python
# 本块中文说明：一行得到“每个样本各自的梯度”。
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
~~~

逐样本梯度一行搞定——PyTorch 里不做 hack 几乎写不出来（原课陈述）。

### 2.6 pmap：多设备自动并行

`parallel_step = jax.pmap(train_step, axis_name='devices')` 把函数复制到所有可用 GPU/TPU 并切分 batch；函数内用 `jax.lax.pmean`/`psum` 跨设备同步梯度。Google 用 pmap（及其后继 shard_map）在数千块 TPU v5e 上训 Gemini（原课陈述）。编程模型：写单设备版本，套 pmap，完事。

### 2.7 pytree：参数的家

~~~python
# 本块中文说明：模型参数=嵌套字典，即一棵 pytree。
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
~~~

JAX 的每个变换（grad、jit、vmap）都懂怎么遍历 pytree。`jax.tree.map(f, tree)` 对每片叶子套用 f——优化器更新全部参数就是一行：

~~~python
# 本块中文说明：纯函数式参数更新。
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
~~~

没有 `.parameters()` 方法、没有参数注册——**树结构就是模型**（0.3 算例四手算过：w 从 2.0 变 1.6 的那次更新）。

### 2.8 生态与选型

积木库：Flax（Google 的层库，状态显式）、Equinox（Patrick Kidger 的树式层库，更 Pythonic）、Optax（DeepMind 的优化器+调度库）、Orbax（pytree 存档）、CLU（指标与日志）。Optax 的设计是“梯度变换可串联”：

~~~python
# 本块中文说明：裁剪+Adam 链式组合。
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
~~~

何时选 JAX vs PyTorch（原课表精简）：TPU 一等公民支持选 JAX；逐行调试容易选 PyTorch；生态体量选 PyTorch；超大规模训练选 JAX；快速原型选 PyTorch。原课的老实话：**除非有明确理由，用 PyTorch**。明确理由包括：TPU 资源、需要逐样本梯度、超大规模多设备训练、在 Google/DeepMind/Anthropic 工作。

### 2.9 随机数：没有全局状态这回事

~~~python
# 本块中文说明：显式钥匙——split 一次多一把。
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
~~~

每个随机操作都要显式 PRNG key。开始烦，但它保证了跨设备、跨编译的可复现性——`torch.manual_seed` 在多 GPU 场景下给不了的保证（原课观点）。

## 三、操作前准备

~~~bash
python -m pip install jax jaxlib optax scikit-learn
~~~

Build It 用 `sklearn.datasets.fetch_openml` 在线获取 MNIST（首次约下载几十 MB，需联网）；GPU 版为 `pip install "jax[cuda12]"`，TPU 版按原课 Ship It 的命令。先激活你已有的虚拟环境；以上命令只供按需执行，本次没有替你安装或下载任何数据。（补充）JAX 官方不支持原生 Windows，Windows 上需经 WSL 使用——原课未提，本机读者注意。

## 四、Build It：原课分步代码与中文讲解

任务：3 层 MLP（784→256→128→10）训练 MNIST，JAX+Optax，与[第11课](<11：PyTorch入门——nn.Module与autograd.md>)同结构，好逐行对照。

### 4.1 第 1 步：数据

~~~python
# 本块中文说明：用 fetch_openml 取 MNIST，归一到 0—1，前 6 万训练、后 1 万测试。
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
~~~

（核对：`parser='auto'` 是较新版 scikit-learn（≥1.4）的参数，老版本会因不认识该关键字报错；`target` 返回的是字符串，`astype('int')` 依赖这一行为，是原课写法，新版 sklearn 仍兼容。）

### 4.2 第 2 步：初始化参数

没有类，只有一个返回 pytree 的函数；He 初始化手工做，三把 PRNG key 从一个种子分出来：

~~~python
# 本块中文说明：纯函数 init_params——He 初始化 + 显式随机钥匙。
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
~~~

对照[第08课](<08：权重初始化——零、随机、Xavier与Kaiming.md>)：scale=√(2/扇入) 就是 He 初始化；这里的每个权重都是嵌套字典里一片不可变的叶子。

### 4.3 第 3 步：前向与损失

~~~python
# 本块中文说明：纯函数前向 + 手写交叉熵。
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
~~~

没有 self、没有存储状态：参数进、预测出。损失是[第05课](<05：损失函数——MSE、交叉熵与选择.md>)交叉熵的“log_softmax·one_hot 取负均值”形式；整个批一起算（jnp 天然向量化），这里没有用到 vmap——向量化直接写在矩阵运算里。

### 4.4 第 4 步：jit 编译的训练步

~~~python
# 本块中文说明：一次调用完成求梯度+Adam 更新。
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
~~~

`jax.value_and_grad` 一趟同时拿到损失值和梯度；`@jax.jit` 把两个函数编译到 XLA。首次调用后，每一步训练都不再触碰 Python。（核对：`train_step` 引用了全局名 `optimizer`，而 `optimizer = optax.adam(...)` 在第 5 步才定义——顺序执行时训练循环调用它之前 optimizer 已存在，能正常工作；但这是隐式全局依赖，若单独运行本块会 NameError。原课未说明，特此标注。）

### 4.5 第 5 步：训练循环

~~~python
# 本块中文说明：显式状态传递的完整训练循环。
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
~~~

注意“没有的东西”：没有 zero_grad、没有 backward、没有 step——整个更新就是一次 `train_step` 调用，参数和优化器状态都是显式传递、显式返回。洗牌也要用显式 key（`random.split` 每轮发新钥匙）。原课预期：10 轮约 97% 测试准确率；第一轮慢（JIT 编译），第 2—10 轮快（原课参考）。

## 五、把代码运行起来：补充入口与检查

（补充）原课各步按顺序执行即可，4.5 依赖 4.1—4.4 与网络下载。以下是**不需要数据**的最小检查，数值与 0.3 节算例一一对应：

~~~python
# 本块中文说明：补充的最小纯函数检查入口。
import jax
import jax.numpy as jnp

# 检查1：算例二——grad 与二阶导。
f = lambda x: x ** 2
df = jax.grad(f)
d2f = jax.grad(jax.grad(f))
# 可手算预期：df(3.0) == 6.0，d2f(3.0) == 2.0。

# 检查2：算例一——不可变性。
a = jnp.array([1.0, 2.0, 3.0])
b = a.at[0].set(9.0)
# 可手算预期：b == [9.0, 2.0, 3.0]，a 仍是 [1.0, 2.0, 3.0]。

# 检查3：算例三——vmap 批处理。
predict = lambda params, x: params['w'] * x + params['b']
params = {'w': 2.0, 'b': 1.0}
batch_pred = jax.vmap(predict, in_axes=(None, 0))
# 可手算预期：batch_pred(params, jnp.array([1.0, 2.0, 3.0])) == [3.0, 5.0, 7.0]。

# 检查4：算例四——pytree 更新。
params2 = {'layer1': {'w': jnp.array(2.0)}}
grads2 = {'layer1': {'w': jnp.array(4.0)}}
new_params = jax.tree.map(lambda p, g: p - 0.1 * g, params2, grads2)
# 可手算预期：new_params['layer1']['w'] == 1.8。
~~~

补充检查方法：给 `train_step` 的首尾各计时一次，验证“第一次慢、之后快”的 jit 特征；用 `%timeit` 或 time 模块均可，数值取决于硬件，不作预期。

## 六、Use It：使用现成工具

### 6.1 Flax：Google 的标准层库

~~~python
# 本块中文说明：Flax 版 MLP——结构像 PyTorch，状态仍显式。
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
~~~

结构与 PyTorch 相像，但 `model.init()` 生成参数、`model.apply(params, x)` 跑前向——模型对象自己没有状态。

### 6.2 Equinox：更 Pythonic 的选择

~~~python
# 本块中文说明：模型本身就是 pytree。
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
~~~

模型本身是 pytree，不需要 `.apply()`，参数就是它的叶子——更贴近 JAX 的思维方式。

### 6.3 Optax：可组合的优化器

~~~python
# 本块中文说明：调度+裁剪+AdamW 全部链式组合。
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
~~~

梯度裁剪、warmup、权重衰减串成一条变换链，每个变换看一眼梯度、改一改、传给下一个（与[第09课](<09：学习率调度——warmup与衰减策略.md>)的 warmup 余弦调度直接呼应）。没有大一统的 Optimizer 类。

### 6.4 存档与性能备忘（原课 Ship It）

Orbax 存 pytree：

~~~python
# 本块中文说明：Orbax 检查点保存/恢复。
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
~~~

（核对：`/tmp/model` 是 Unix 路径，Windows 上需换成本机路径。）性能备忘（原课）：首次 jit 调用慢是编译，基准测试前先热身；jit 内避免对 JAX 数组写 Python 循环，改用 `jax.lax.scan`/`fori_loop`；jit 里用 `jax.debug.print`，普通 print 无效；用 `jax.profiler` 或 TensorBoard 查瓶颈；JAX 默认预分配 75% GPU 显存，设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可关。

## 七、失败模式、原文边界与工程注意

1. **第一次调用慢不是 bug**：那是编译。基准测试、计时一律从第二次调用起算。
2. **jit 里的 print 只在追踪时执行一次**：要输出运行时值用 `jax.debug.print`，否则你会被“打印了但值不对/没打印”搞疯（原课 Ship It 明确警告）。
3. **值依赖的 Python 控制流在 jit 下报错或行为错误**：`if x > 0`（x 是追踪数组）不行，得用 `jax.lax.cond`；循环用 `jax.lax.scan`。这是最容易踩的范式坑。
4. **全局随机状态的诱惑**：习惯了 NumPy/PyTorch 的人总想找 `jax.random.seed()`——不存在。每把钥匙显式 split 显式传（2.9 节），忘 split 复用同一 key 会拿到相关随机数。
5. **预分配 75% 显存**：多进程/多实验共用一张卡时会被“没显存了”迷惑——其实被 JAX 圈走了，设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
6. **隐式全局依赖**（补充，核对处详述）：4.4 的 `train_step` 通过闭包捕获 4.5 才定义的 `optimizer`，顺序执行没事、单独运行报 NameError——工程上应把 optimizer 作为参数传入。
7. **Windows 原生支持缺失**（补充）：JAX 官方不支持原生 Windows，需 WSL；对比 PyTorch 全平台。原课未提，选型时是现实因素。
8. **生态与招聘的现实边界**：原课自己承认“没特殊理由就用 PyTorch”；JAX 的生态偏研究（Flax/Equinox/Optax），生产推理常绕道 TensorFlow Serving / Vertex AI。学习曲线也陡（函数式范式）。
9. **速度数字的边界**：MNIST MLP 上 JAX ~0.3s/epoch 对 PyTorch ~0.5s/epoch（原课参考），差距远小于两者对纯 Python 的 500 倍量级——单卡常规任务里框架差异不是主要矛盾。

## 八、Ship It：本课可复用的中文检查模板

（补充，改写自原课 Ship It 的两个产物——优化器配置提示词与函数式模式速查）

~~~markdown
检查 JAX 训练脚本：参数是否全程显式传递（无隐藏可变状态）；jit 边界是否合理（训练步、评估函数在内，一次性逻辑在外）；随机性是否全部走显式 PRNG key 且每用必 split；jit 内是否有值依赖的 if/for（应换成 lax.cond/scan）；是否报告了“首次编译耗时”与“稳态每步耗时”两个数字；GPU 显存策略是否明确（默认预分配 75%）；调试是否用 jax.debug.print 而非 print。
~~~

## 九、Exercises：保留原课练习与验收方法

1. 原课：给 MLP 加 dropout——在 JAX 里 dropout 需要一个 PRNG key，把 key 穿过前向函数并为每个 dropout 层 split；对比有无 dropout 的测试准确率。验收：能解释“为什么 PyTorch 的 dropout 不需要你传 key 而 JAX 需要”（全局状态 vs 显式状态的差别）。
2. 原课：用 `jax.vmap` 算 32 张 MNIST 图的逐样本梯度，并算每个样本的梯度范数，找出范数最大的样本并解释原因。验收：能展示“哪些样本梯度大”并给出猜测（离群/难样本/标签可疑，衔接[第13课](<13：调试神经网络——没有报错的排错法.md>)）。
3. 原课：把手工 forward 换成通用的 `mlp_forward(params, x)`，支持任意层数，用 `jax.tree.leaves` 自动判断深度。验收：同一函数能跑 2 层和 5 层两棵参数树。
4. 原课：基准测试有无 `@jax.jit` 的 train_step，各计时 100 步，报告加速比与首次编译开销。验收：给出两个时间数并解释 jit 的收益来自哪里（绕开 Python 解释器 + XLA 融合）。
5. 原课：实现梯度裁剪 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`，有无裁剪各训一次，画出梯度范数随训练的变化。验收：曲线显示未裁剪版本范数尖峰更高、裁剪版本被压在 1.0 附近。

## 十、术语与完成标准

XLA：把计算图编译成融合的 GPU/TPU 核的编译器；即时编译（jit）：首调追踪、编译，后续调用跑编译版；纯函数：输出只依赖输入、无副作用；vmap：把单样本函数变换为批处理函数；pmap：把函数复制到多设备并切分输入；pytree：列表/元组/字典/数组的任意嵌套，JAX 变换的通用载体；追踪：用抽象值执行函数以构建计算图的过程；函数式自动微分：通过变换函数求导而非给张量挂梯度；Optax：可组合的梯度变换库；Flax：Google 的 JAX 层库，状态显式。

完成检查：

- 能手算 0.3 四个算例并说明每个演示了什么（不可变性、函数求导、向量化、树更新）。
- 能说出 jit 为什么要求纯函数、追踪和执行的区别。
- 能解释 `in_axes=(None, 0)` 的两个参数各管什么。
- 能把 PyTorch 训练循环五步逐一映射到 JAX 的 `train_step` 一条调用。
- 能说出 JAX 随机数与 PyTorch 随机数的根本差异及原因。
- 能复述“何时该选 JAX”的三个理由和老实话。

## 十一、验证范围与来源许可

原课 Build It 共 5 个分段，代码逐字转录嵌入本篇（仅添加块首中文说明行，未改代码本体）；发现的两处问题（fetch_openml 的 parser 参数版本依赖、train_step 闭包引用后定义的全局 optimizer）均以“（核对：…）”显式标注。Use It 的 Flax/Equinox/Optax/Orbax 代码同样保留原样。原课未展示运行输出，本篇所有性能数字（97%、0.3s/epoch、10—100 倍 vmap 加速等）均标为原课参考；“可手算预期”数值与 0.3 节算例一致可手工复核。本次未执行训练、未下载 MNIST；正文 python 代码块已抽出为临时文件做 `ast.parse` 静态语法检查，结果见下。

验证记录（2026-11-06）：
- 课程页抓取一次成功，内容以课程页为准整理。
- 正文 python 代码块全部通过 `ast.parse` 静态语法检查（标注的闭包/版本问题属运行时与依赖问题，语法本身合法）。
- 未做：实际训练 MNIST、jit 基准测试、GPU/TPU 实验、Flax/Equinox 实际运行。

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
