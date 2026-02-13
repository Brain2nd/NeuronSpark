# SNN 隐神经元状态空间：设计思考

> 日期: 2025-02
> 状态: 概念设计阶段，尚未实现
> 核心思路: 利用SNN神经元的**原生物理机制**（spike/静默、泄漏、soft reset、阈值）构建选择性记忆的隐状态空间，不是对Mamba的SNN翻译
> **关键设计**: β、α和V_th由当前输入动态计算，不是训练后固定的常数——这是隐神经元空间区别于普通LIF层的根本机制
> **注**: 架构和训练方法均为当前方案，将在实验中持续迭代优化

---

## 1. 问题起源

### 1.1 SNN时间残差的本质

SNN中，LIF神经元天然保留膜电位残差：

$$V_i[t] = \beta_i \cdot V_i[t-1] + I_i[t]$$

其中 $\beta_i$ 是每个神经元独立的可训练参数（对角矩阵 $\text{diag}(\beta_1, \beta_2, ..., \beta_n)$）。

这个时间残差是LIF神经元的**原子物理属性**——不是设计出来的，而是神经元膜电位动力学天然具备的。

### 1.2 长程依赖的根本困难

token t 的信息在 t+k 步时的残留量为 $\beta^k$，指数衰减：

| β值 | 10步后 | 50步后 | 100步后 |
|-----|--------|--------|---------|
| 0.90 | 0.349 | 0.005 | ~0 |
| 0.95 | 0.599 | 0.077 | 0.006 |
| 0.99 | 0.904 | 0.605 | 0.366 |

混合时间 $t_{mix} \sim 1/|\ln\beta|$：β=0.90 → ~9.5步，β=0.99 → ~100步。超过 t_mix 后早期信息基本丢失。

### 1.3 核心矛盾

β操作横贯所有历史——$\beta \cdot V[t-1]$ 对 $V[t-1]$ 整体操作。$V[t-1]$ 里混着所有历史token的信息，无法选择性地"保留token 5的贡献，衰减token 3的贡献"。单个神经元膜电位作为标量的根本限制——所有历史信息已经混在一个数里，任何操作都是不可分的。

### 1.4 SNN的方向不是模仿Transformer

Attention将序列同时输入，计算全局Q·K^T相关性——这是空间/并行操作。SNN的时间残差是时序/串行操作。两者是不同范式。

SNN也不应该照搬Mamba的公式——把Mamba的 $\Delta_t$, $B_t$, $C_t$, $\exp(\Delta \cdot A)$ 用SNN门电路重新实现，本质上只是用更慢的硬件做同样的事，没有意义。

**SNN需要的是：利用自身原生物理机制（spike、泄漏、阈值、soft reset）构建选择性记忆能力。**

---

## 2. 隐神经元状态空间：核心概念

### 2.1 基本思想

构建一个**专门的神经元结构**担任隐状态空间的角色：

- **膜电位** = 隐状态的载体，记录累积的上下文信息
- **发放（spike）** = 稀疏选择——该神经元记录的信息**当前相关**，参与本时间步的输出计算
- **静默（不发放）** = 上下文积累——该神经元记录的信息**当前不需要表达**，继续在膜电位中积累
- **泄漏（β衰减）** = 自然遗忘——不重要的历史信息随时间自然衰减
- **soft reset** = 非线性状态压缩——发放后 $V -= V_{th}$，清除已表达的部分，保留残差

### 2.2 spike/静默的双重角色

这是和Mamba最根本的区别。Mamba的所有状态维度在每个时间步都通过 $C_t \cdot h[t]$ 线性读出——没有"哪些参与、哪些不参与"的区分。

SNN的隐神经元空间天然分成两组：

**发放的神经元（当前活跃）**：
- 膜电位超过阈值 → 产生spike
- spike信号参与当前时间步的下游计算
- soft reset清除一部分状态（已表达的信息"释放"了）
- 残差继续保留（部分上下文传递到未来）

**静默的神经元（背景积累）**：
- 膜电位未达阈值 → 不产生spike
- 不参与当前时间步的输出
- 膜电位继续积累：$V[t] = \beta \cdot V[t-1] + I[t]$
- 在背景中默默构建上下文，等待未来某个时刻被激活

**spike模式本身就是信息选择的结果**——哪些神经元发放、哪些静默，取决于当前输入和历史积累的膜电位之间的交互。不需要额外的门控信号，阈值判定本身就是门控。

### 2.3 与"一层LIF神经元"的区别

这不是简单地放一层LIF然后让它跑。关键区别在于**结构设计**和**功能分工**：

1. **有目的的时间尺度分布**：不同神经元有不同的β，覆盖从短程到长程的记忆需求
2. **有目的的阈值分布**：不同神经元有不同的V_th，控制发放的稀疏度和选择性
3. **输入依赖的调制**：β和/或V_th可以被当前输入调制，实现动态选择性
4. **功能性解读**：spike模式不是副产物，而是核心输出——它编码了"当前上下文中哪些信息被选中"

### 2.4 选择性的根本来源：输入依赖的β和V_th

**这是隐神经元空间区别于"一堆LIF神经元"的核心机制。**

如果β和V_th是训练后固定的常数，那么：
- β=0.99的神经元**永远**保留99%旧信息，不管当前输入是什么
- V_th=1.0的神经元**永远**在V>1.0时发放，不管当前上下文如何
- 没有"这个输入重要→多记"和"这个输入无关→快忘"的区分能力
- 这就是普通的一层LIF，没有选择性可言

**选择性意味着：β和V_th由当前输入 $x_t$ 和隐神经元膜电位 $V[t-1]$ 共同决定。**

$$\beta_{d,n}(t) = f_\beta(x_t, V[t-1]) \quad \quad V_{th_{d,n}}(t) = f_{th}(x_t, V[t-1])$$

其中 $V[t-1] \in \mathbb{R}^{D \times N}$ 是隐神经元的膜电位——所有历史输入的指数加权累积，携带完整上下文。

这样每个时间步、每个神经元的遗忘速率和发放阈值同时取决于"现在来了什么"和"之前全部历史积累了什么"：

- 重要输入 + 已有相关积累 → β大幅降低 → 清除旧信息为新信息腾空间
- 重要输入 + 无相关积累 → β适度降低 → 开始新的积累周期
- 不重要输入 + 已有重要积累 → β升高 → 保护已有信息不被冲刷
- 需要表达时（膜电位指示积累充足） → V_th降低 → 积累的信息被释放
- 需要继续积累时 → V_th升高 → 信息在背景中沉淀

**与Mamba和LSTM的对比**：

| 模型 | 门控/调制依赖 | 隐状态利用 |
|---|---|---|
| Mamba | $\Delta_t = f(x_t)$ 只看输入 | 状态 $h[t]$ 不参与Δ计算 |
| LSTM | $f_t = \sigma(W \cdot [h_{t-1}, x_t])$ 输入+隐状态 | $h_{t-1}$ 参与所有门控 |
| **SNN隐空间** | $\beta(t) = f(x_t, V[t-1])$ 输入+膜电位 | $V[t-1]$（完整历史）参与调制 |

我们的设计在门控机制上更接近LSTM（同时看输入和隐状态），但在状态更新上更像SSM（线性递归+非线性输出）。膜电位V[t-1]比LSTM的h[t-1]携带更原始、更完整的信息——h是经过门控筛选后的输出，V是未经门控过滤的原始积累。

**Voltage-gated channels类比**：$W^{(V)} \cdot V[t-1]$ 对应生物神经元中膜电位调制离子通道电导率的机制。不需要softplus/exp，通过膜电位直接线性调制+sigmoid非线性约束。

**遗忘的两个协同机制**：

1. **β衰减（被动/连续遗忘）**：输入依赖的β控制每步衰减多少。这是平滑的、渐进的遗忘。
2. **spike+soft reset（主动/离散遗忘）**：发放时 $V -= V_{th}$，一次性清除一部分状态。输入依赖的V_th控制清除的力度和频率。

两种遗忘协同工作——β是背景衰减，spike+reset是主动释放。这比Mamba只有单一Δ门控更丰富。

---

## 3. 数学框架

### 3.1 结构设定

```
D = 可见维度（和上下游交互）
N = 状态扩展因子（每个可见通道的状态神经元数）
总隐神经元数 = D × N
基底神经元 = PLIF（Parametric Leaky Integrate-and-Fire）
```

**基底选型：PLIF**（基于 neuron_comparison.ipynb 的系统评估）：
- PLIF 的动力学 $V[t] = \beta \cdot V[t-1] + (1-\beta) \cdot I[t]$，其中 $\beta = \sigma(w)$ 可学习，与设计公式 $V[t] = \beta(t) \cdot V[t-1] + I[t]$ 结构完全匹配——只需将标量参数 $w$ 替换为调制网络
- 输入接收实数，输出二值 spike（0/1），已实验验证
- 简单基底 + 外部调制网络的复杂性，便于归因和对照实验

**原版 PLIF 参数与改造**（基于 SpikingJelly 源码 `ParametricLIFNode` 分析）：

| 参数 | 原版 PLIF | 可训练 | 形状 | 随时间变化 |
|---|---|---|---|---|
| `w`（→β） | `nn.Parameter(scalar)` | 是 | **标量**（整层共享一个 β） | 否（训练后固定） |
| `v_threshold` | `float` | 否 | **标量**（整层共享一个阈值） | 否 |
| `v_reset` | `float` 或 `None` | 否 | **标量** | 否 |
| `v`（膜电位） | 运行时状态 | 否 | **逐神经元**（与输入同形） | **是**（每步更新） |

原版 PLIF 中唯一逐神经元、随时间变化的量只有膜电位 V。β 和 V_th 都是整层共享的标量常数。

**我们的改造**：β 和 V_th 不再是静态参数，而是每步由调制网络动态计算的值。真正的可训练 `nn.Parameter` 变为调制网络的权重矩阵：

| 可训练参数 | 形状 | 作用 |
|---|---|---|
| $W_\beta^{(x)}$ | D → D×N | 输入 → β 调制 |
| $W_\beta^{(V)}$ | N×N（D 通道共享） | 膜电位 → β 调制 |
| $b_\beta$ | D×N | β 偏置（按多时间尺度 0.80~0.99 初始化） |
| $W_\alpha^{(x)}$ | D → D×N | 输入 → α 调制 |
| $W_\alpha^{(V)}$ | N×N（D 通道共享） | 膜电位 → α 调制 |
| $b_\alpha$ | D×N | α 偏置（初始化使 softplus(b_α) ≈ 1.0） |
| $W_{th}^{(x)}$ | D → D×N | 输入 → V_th 调制 |
| $W_{th}^{(V)}$ | N×N（D 通道共享） | 膜电位 → V_th 调制 |
| $b_{th}$ | D×N | V_th 偏置（与 β 协同初始化） |

改造后 β(t) 和 V_th(t) 变为**逐神经元、逐时间步**的动态值——从"一个标量控制整层"变为"一组权重矩阵动态计算每个神经元每个时间步的值"。

隐神经元空间不直接和外界通信，通过输入投影和输出投影与可见维度交互。

### 3.2 输入投影

当前时间步输入 $x_t \in \mathbb{R}^D$，投影为隐神经元的输入电流：

$$I[t] = W_{in} \cdot x_t \in \mathbb{R}^{D \times N}$$

$W_{in}$ 是可训练的标准线性层（接收实数输入），将可见维度的信息分发到N个状态神经元。

### 3.3 输入+隐状态依赖的参数计算

**这是核心步骤——β和V_th由当前输入和隐神经元膜电位共同决定。**

仅依赖输入 $x_t$ 是不够的：同一个token在不同上下文中应触发不同的选择行为。调制必须同时看到"现在来了什么"和"之前积累了什么"。

**上下文信号的选择：膜电位 $V[t-1] \in \mathbb{R}^{D \times N}$**

膜电位是隐神经元的**完整累积状态**：

$$V[t-1] = \sum_{k=0}^{t-1} \beta^k \cdot I[t-1-k]$$

它携带了所有历史token的指数加权信息——不是上一步的快照，而是**全部上下文的压缩**。

为什么用 $V[t-1]$ 而不是 $s[t-1]$（上一步spike模式）：
- $s[t-1]$ 只是t-1时刻的二值快照，只知道"上一步谁发放了"，不携带之前的积累
- $V[t-1]$ 是从第0步到t-1步所有信息的指数加权和——完整的上下文记忆
- 用 $s[t-1]$ 做调制 = 只看上一个token的选择结果；用 $V[t-1]$ 做调制 = 看全部历史

**V是连续值，但这不破坏SNN约束**：膜电位是神经元的**内部状态**，不是神经元间传输的信号。V调制自身的β和V_th是神经元的固有属性——生物学上对应**voltage-gated ion channels**（电压门控离子通道）：通道电导率取决于膜电位，这是神经科学最基础的机制之一。

**调制公式**：

$$\beta(t) = \sigma\left(W_\beta^{(x)} \cdot x_t + W_\beta^{(V)} \cdot V[t-1] + b_\beta\right) \in (0, 1)^{D \times N}$$
$$\alpha(t) = \text{softplus}\left(W_\alpha^{(x)} \cdot x_t + W_\alpha^{(V)} \cdot V[t-1] + b_\alpha\right) \in \mathbb{R}_+^{D \times N}$$
$$V_{th}(t) = V_{th,min} + \left|W_{th}^{(x)} \cdot x_t + W_{th}^{(V)} \cdot V[t-1] + b_{th}\right|$$

其中：
- $W_\beta^{(x)}$：输入→β调制（D→D×N，SNN突触层，spike输入）——"当前来了什么"
- $W_\beta^{(V)}$：膜电位→β调制（N→N，所有D通道共享）——"之前积累了什么上下文"
- $W_\alpha^{(x)}$：输入→α调制（D→D×N，SNN突触层，spike输入）——"当前输入写入多少"
- $W_\alpha^{(V)}$：膜电位→α调制（N→N，所有D通道共享）——"已有上下文如何影响写入量"
- $W_{th}^{(x)}$：输入→V_th调制（D→D×N，SNN突触层，spike输入）——"当前输入如何影响发放门槛"
- $W_{th}^{(V)}$：膜电位→V_th调制（N→N，所有D通道共享）——"已有上下文如何影响发放门槛"

**衰减与写入解耦**：β 和 α 由独立的调制网络控制。Mamba 的 Δ 耦合了衰减（$\bar{A}$）和写入（$\bar{B}$），迫使"忘多↔写多"或"忘少↔写少"。我们解耦后支持"忘少+写多"（重要 token 既保留旧上下文又大量吸收新信息）。V 的增长由阈值+soft reset 天然约束——Mamba 没有此机制因此需要耦合防止状态增长。

**$W^{(V)}$ 的结构：通道独立 + 跨通道共享 N×N**
- 通道间独立：不同特征维度 d 之间无直接因果关系（与 Mamba 的对角 A 相同假设）
- 通道内 N×N 交互：同通道的 N 个时间尺度神经元通过共享的 N×N 矩阵协调——短程膜电位变化影响长程行为，反之亦然。这是 Mamba 对角结构未利用的多时间尺度协同
- 所有 D 个通道共享同一个 N×N：时间尺度间的交互模式（"短程检测到重要输入→长程准备接收"）是跨通道通用的
- 参数量：$W_\beta^{(V)}$ = N² = 64，$W_\alpha^{(V)}$ = 64，$W_{th}^{(V)}$ = 64（N=8），**总计 192 个参数**

**直觉**：
- $W^{(x)}$ 回答"现在来了什么？"
- $W^{(V)}$ 回答"之前的全部历史积累了什么？"
- 两者结合 = 真正的**上下文相关的选择性**

**初始化**：$V[-1] = \mathbf{0}$（初始无历史），此时β和V_th完全由输入和偏置决定。

### 3.4 膜电位动力学

每个隐神经元 $(d, n)$ 的更新使用**当前输入计算出的** β 和 α：

$$V_{d,n}[t] = \beta_{d,n}(t) \cdot V_{d,n}[t-1] + \alpha_{d,n}(t) \cdot I_{d,n}[t]$$

- β(t) 控制衰减：保留多少旧状态（独立于写入）
- α(t) 控制写入增益：写入多少新信息（独立于衰减）
- 两者都由输入 + 膜电位动态计算，但通过各自独立的调制网络
- 这是选择性的来源：同一神经元在不同时间步有不同的衰减率和写入率

### 3.5 阈值判定与状态分裂

使用**当前输入计算出的**V_th判定发放：

$$s_{d,n}[t] = \begin{cases} 1, & V_{d,n}[t] > V_{th_{d,n}}(t) \quad \text{（发放：参与输出，清除部分状态）} \\ 0, & V_{d,n}[t] \leq V_{th_{d,n}}(t) \quad \text{（静默：不参与输出，继续积累）} \end{cases}$$

发放后soft reset：

$$V_{d,n}[t] \leftarrow V_{d,n}[t] - V_{th_{d,n}}(t) \cdot s_{d,n}[t]$$

V_th(t)随输入变化——同一个膜电位在不同上下文下可能发放也可能不发放。

### 3.6 输出读出

只有发放的神经元贡献输出：

$$y_t = W_{out} \cdot \mathbf{s}[t] \in \mathbb{R}^D$$

其中 $\mathbf{s}[t] \in \{0, 1\}^{D \times N}$ 是spike向量。

**注意**：输出是spike（二值），不是膜电位（连续值）。这保持了SNN的脉冲域特性，也天然实现了稀疏性——只有发放的神经元有非零贡献。

### 3.7 数据流全景

```
输入 x_t (D维, 实数)            V[t-1] (D×N, 隐神经元膜电位=完整上下文)
  │                                    │
  │  ┌─────────────────────────────────┤ (神经元内部状态，连续值)
  │  │                                 │
  ├──┼──→ W_in · x_t ────────────────→ I[t] (D×N)  输入电流
  │  │
  ├──┼──→ W_β^(x)·x_t + W_β^(V)·V[t-1] ──→ σ ──→ β(t) (D×N)  动态衰减率
  │  │
  ├──┼──→ W_α^(x)·x_t + W_α^(V)·V[t-1] ──→ softplus ──→ α(t) (D×N)  动态写入增益
  │  │
  └──┴──→ W_th^(x)·x_t + W_th^(V)·V[t-1] ──→ |·| ──→ V_th(t) (D×N)  动态阈值
         │
  ╔══════╪═══════════════════════════════════════════════════════════╗
  ║      ↓                                                           ║
  ║  隐神经元状态空间 (D×N 个PLIF神经元)                               ║
  ║                                                                   ║
  ║  膜电位更新: V[t] = β(t)·V[t-1] + α(t)·I[t]                      ║
  ║              ↑         ↑         ↑                                ║
  ║         由(x_t,V[t-1])  携带全部   由(x_t,V[t-1])                  ║
  ║         动态决定衰减    历史上下文  动态决定写入增益                  ║
  ║                                                                   ║
  ║  ┌──────────────────────────┐  ┌─────────────────────────────┐   ║
  ║  │ 发放的神经元              │  │ 静默的神经元                │   ║
  ║  │ V[t] > V_th(t)           │  │ V[t] ≤ V_th(t)             │   ║
  ║  │                          │  │                             │   ║
  ║  │ → spike = 1              │  │ → spike = 0                 │   ║
  ║  │ → V[t] -= V_th(t)        │  │ → V[t]保持，继续积累        │   ║
  ║  │ → 参与输出               │  │ → 构建背景上下文             │   ║
  ║  └──────────────────────────┘  └─────────────────────────────┘   ║
  ║                                                                   ║
  ║  V[t] 更新完成 ──→ 存储为下一步的 V[t] (= 下一步的V[t-1])  ──┐  ║
  ╚══════════════════════════════════════════════════════════════╪═══╝
         │                                                      │
         ↓ spike模式 s[t] (D×N, 二值稀疏)                      │
         │                                                      │
         ↓ W_out (SNN线性层)                                    │
         │                                                      │
         ↓ 输出 y_t (D维, 脉冲序列)           V[t] 反馈 ───────┘
                                               (内部状态自回归)
```

**四条前馈通路 + 一条内部状态反馈**：
- $W_{in}$：信息写入（当前输入→隐空间电流）
- $W_\beta$：遗忘控制（输入+膜电位→每个神经元的衰减率）
- $W_\alpha$：写入增益控制（输入+膜电位→每个神经元的写入量，独立于衰减）
- $W_{th}$：发放控制（输入+膜电位→每个神经元的发放门槛）
- $V[t] \rightarrow V[t-1]$：膜电位作为完整上下文反馈给调制网络

这构成了一个**自回归循环**：膜电位（完整历史积累）影响下一步的β、α和V_th，β、α和V_th又决定膜电位如何更新。

**与voltage-gated ion channels的类比**：生物神经元的离子通道电导率取决于膜电位——膜电位高时某些通道打开/关闭，改变神经元的动态行为。我们的 $W^{(V)} \cdot V[t-1]$ 正是这种机制的计算模型。

---

## 4. 输入依赖的β和V_th计算网络

β和V_th由当前输入动态计算——这不是可选功能，而是隐神经元空间的核心机制。没有输入依赖性，就没有选择性，就只是普通LIF层。

### 4.1 调制网络的设计

调制网络接收两个信号源：当前输入 $x_t$（实数）和膜电位 $V[t-1]$（连续值，神经元内部状态）。

**β调制网络**：
$$\beta(t) = \sigma\left(W_\beta^{(x)} \cdot x_t + W_\beta^{(V)} \cdot V[t-1] + b_\beta\right) \in (0, 1)^{D \times N}$$

- $W_\beta^{(x)}$：输入投影（标准线性层）——"当前token是什么"
- $W_\beta^{(V)}$：膜电位投影（标准线性）——"历史积累了什么上下文"
- $\sigma$：sigmoid，确保输出在(0,1)范围

**V_th调制网络**：
$$V_{th}(t) = V_{th,min} + \left|W_{th}^{(x)} \cdot x_t + W_{th}^{(V)} \cdot V[t-1] + b_{th}\right|$$

- 同样双路输入
- 取绝对值 + 最小值下限确保V_th > 0

**关于 $W^{(V)}$ 使用标准线性而非SNN**：膜电位V是神经元内部的连续量，不是神经元间传递的脉冲信号。$W^{(V)} \cdot V$ 是神经元自身的属性调制（类比voltage-gated channels），属于神经元内部计算，不需要走脉冲通路。这和encoder/decoder的边界转换不同——这里根本不存在"传输"，只有内部状态的自反馈。

### 4.2 调制网络的初始化

虽然β和V_th是动态计算的，但调制网络的**权重初始化**仍然需要结构化设计，使得初始输出分布合理：

**β网络初始化目标**：初始输出的β分布覆盖多时间尺度
- 偏置 $b_\beta$ 按HiPPO思想初始化，使N个状态通道的初始β从~0.80到~0.99线性分布
- 权重 $W_\beta$ 小随机初始化，使训练初期β主要由偏置决定
- 训练过程中，$W_\beta$ 学会根据输入内容调制β

| 初始β分布 | t_mix (步) | 初始角色 |
|---|---|---|
| ~0.80 | ~4.5 | 即时上下文 |
| ~0.90 | ~9.5 | 短程模式 |
| ~0.95 | ~19.5 | 中程依赖 |
| ~0.99 | ~99.5 | 长程趋势 |

**α网络初始化目标**：初始写入增益为单位值
- 偏置 $b_\alpha$ 初始化使 $\text{softplus}(b_\alpha) \approx 1.0$
- $\text{softplus}^{-1}(1.0) = \ln(e^1 - 1) \approx 0.5413$
- 所有 D×N 个神经元初始 α 相同（≈1.0），由训练学习差异化
- 权重 $W_\alpha$ 小随机初始化（Kaiming × 0.1），使训练初期 α 主要由偏置决定（解耦原则，见 5.8.10.7）

**V_th网络初始化目标**：初始V_th与β协同
- 长程通道（初始β大）的初始V_th较高——积累更久才发放
- 短程通道（初始β小）的初始V_th较低——快速响应
- 偏置 $b_{th}$ 基于 σ\_V 校准：$b_{th,n} = \sigma_{V,n} \cdot \Phi^{-1}(1 - p_{fire,n}) - V_{th,min}$（详见 5.8.10）
- 目标发放率从短程 ~25% 到长程 ~8% 线性过渡
- $V_{th,min} = 0.1$（阈值下限超参数，确保 $V_{th}(t) > 0$）

### 4.3 三个层次的对比

| 场景 | 固定β/V_th | 仅输入依赖 | 输入+膜电位依赖 |
|---|---|---|---|
| 重要token到来 | β不变 | β降低→腾空间 | β降低幅度取决于V中已有积累量 |
| 噪声token到来 | β不变 | β升高→保护旧信息 | V中已积累重要信息→更强保护 |
| 相同token不同上下文 | 完全相同反应 | 完全相同反应 | V不同→调制不同→差异化响应 |
| 需要输出长程信息 | 全靠被动碰阈值 | V_th降低 | V_th基于V判断哪些该释放 |
| 长程依赖的保持 | 靠大β硬扛 | 无法根据已积累内容调整 | V高的通道自动获得保护 |

**本质区别**：
- 固定参数 = 无选择性，被动等待
- 仅输入依赖 = 有选择性，但无记忆（同一token永远触发相同选择）
- 输入+膜电位依赖 = 有选择性+有完整历史记忆（同一token在不同上下文触发不同选择）

第三层才是真正的**上下文相关的选择性状态空间**。膜电位V[t-1]携带从第0步到当前的全部累积信息，不是上一步的快照。

---

## 5. 完整网络架构

### 5.1 设计原则

1. **全网 SNN**：所有层间通信通过 spike（0/1），不存在实数值的层间传递
2. **只有隐状态空间使用特殊设计**：动态 β(t)/V_th(t) 仅用于负责记忆的隐状态神经元。其余神经元为普通 SNN（固定 β/V_th），负责信号转换和维度变换——就像 Mamba 中只有 SSM 负责隐状态，其余组件（Linear/Conv1D/SiLU）是标准计算
3. **受 Mamba 启发的双分支结构**：串行记忆路径（对应 SSM）+ 并行门控路径（对应 Gate 分支），门控路径只看当前输入、无时间状态

### 5.2 模型级架构

```
原始输入（实数）
  ↓
[编码层]  实数 → spike                ← 普通SNN: Linear + PLIF (固定参数)
  ↓
spike ∈ {0,1}^D
  ↓
[SNN Block 1]  spike → spike          ← 内部维护 V_1 隐状态空间
  ↓
[SNN Block 2]  spike → spike          ← 内部维护 V_2 隐状态空间
  ↓
  ...
  ↓
[SNN Block L]  spike → spike          ← 内部维护 V_L 隐状态空间
  ↓
spike ∈ {0,1}^D
  ↓
[解码层]  spike → 实数                ← 二进制解码（MSB-first）或发放率统计
  ↓
输出
```

每个 Block 内部的膜电位 V 和 spike 模式是该 Block **私有**的，不暴露给其他 Block。Block 间只通过 spike 通信。这与 Mamba 的 Block 间传连续值、SSM 状态 h 为 Block 私有完全类比。

### 5.3 SNN Block 详细结构

单个 SNN Block 在时间步 t 的完整计算：

**输入**：$spike_{in} \in \{0,1\}^D$（来自上一Block），$V[t{-}1] \in \mathbb{R}^{D \times N}$（本Block私有隐状态）

**阶段一：六条并行输入路径**（全部为 SNN 突触层：spike × W → 实数电流）

```
spike_in ∈ {0,1}^D
  │
  ├──→ [W_in · spike_in]                → I[t] ∈ R^{D×N}
  │     输入电流路径
  │
  ├──→ [W_β^(x) · spike_in] ──┐
  │     β 调制路径              ├→ + W_β^(V)·V[t-1] + b_β → σ → β(t) ∈ (0,1)^{D×N}
  │                             │   (voltage-gated: 内部V参与)
  │                             ┘
  │
  ├──→ [W_α^(x) · spike_in] ──┐
  │     α 调制路径              ├→ + W_α^(V)·V[t-1] + b_α → softplus → α(t) ∈ R_+^{D×N}
  │                             │   (voltage-gated: 内部V参与, 独立于β)
  │                             ┘
  │
  ├──→ [W_th^(x) · spike_in] ──┐
  │     V_th 调制路径            ├→ + W_th^(V)·V[t-1] + b_th → |·|+V_min → V_th(t) ∈ R_+^{D×N}
  │                              │   (voltage-gated: 内部V参与)
  │                              ┘
  │
  ├──→ [W_gate · spike_in] → sigmoid → gate ∈ (0,1)^D
  │     门控路径（只看当前输入，无状态，对应 Mamba Gate/SiLU 分支）
  │
  └──→ [W_skip · spike_in]              → I_skip ∈ R^D
        残差路径（输入直通）
```

六组 W 的输入全部是 spike（SNN 突触连接）。β、α 和 V_th 路径额外引入 V[t-1]（神经元内部 voltage-gated 反馈，不走 spike 通路）。

**阶段二：隐状态空间**（特殊 PLIF 神经元，D×N 个，负责记忆）

```
  I[t], β(t), α(t), V_th(t) 汇入
          ↓
  V[t] = β(t) · V[t-1] + α(t) · I[t]             膜电位更新
  s[t] = Θ(V[t] - V_th(t))        ∈ {0,1}^{D×N}  阈值判定
  V[t] -= V_th(t) · s[t]                           soft reset
          ↓
  s[t]（spike，二值稀疏）
  V[t] → 保存为下一步的 V[t-1]（Block内部状态自回归）
```

**阶段三：门控 + 残差 + 输出**（普通 SNN 神经元，D 个，固定参数）

```
  [W_out · s[t]]  → I_out ∈ R^D        SNN突触（spike → 电流）

  I_out × gate                           门控：当前输入决定放行哪些维度
    + I_skip                             残差：输入电流直通相加
    → I_total ∈ R^D
          ↓
  普通 PLIF（固定 β_out, V_th_out）:
    V_out[t] = β_out · V_out[t-1] + I_total
    spike_out = Θ(V_out - V_th_out)     ∈ {0,1}^D
```

**输出**：$spike_{out} \in \{0,1\}^D$（传给下一Block）

### 5.4 Block 内组件清单

| 组件 | 输入 | 输出 | 类型 | 对应 Mamba |
|---|---|---|---|---|
| $W_{in}$ | spike | 电流 $\mathbb{R}^{D \times N}$ | SNN 突触 | $\bar{B} \cdot x$ 写入 |
| $W_\beta^{(x)}$ + $W_\beta^{(V)}$ | spike + V[t-1] | β(t) ∈ (0,1)^{D×N} | SNN 突触 + voltage-gated | Δ(x) → Ā 选择性 |
| $W_\alpha^{(x)}$ + $W_\alpha^{(V)}$ | spike + V[t-1] | α(t) ∈ R_+^{D×N} | SNN 突触 + voltage-gated | $\bar{B}$ 写入增益（独立于衰减） |
| $W_{th}^{(x)}$ + $W_{th}^{(V)}$ | spike + V[t-1] | V_th(t) ∈ R_+^{D×N} | SNN 突触 + voltage-gated | C(x) 读出控制 |
| $W_{gate}$ | spike | gate ∈ $(0,1)^D$（sigmoid） | SNN 突触 + sigmoid（无状态） | **Gate 分支**（Mamba 用 SiLU） |
| $W_{skip}$ | spike | I_skip ∈ $\mathbb{R}^D$ | SNN 突触 | 残差连接 |
| **隐状态空间** | I, β, α, V_th | spike s[t] | **特殊 PLIF**（动态参数） | **SSM**（$h = \bar{A}h + \bar{B}x$） |
| $W_{out}$ | spike | 电流 $\mathbb{R}^D$ | SNN 突触 | $C \cdot h$ 读出 |
| 输出神经元 | 电流 | spike | 普通 PLIF（固定参数） | 非线性输出 |

### 5.5 两种 SNN 神经元的分工

| | 隐状态空间神经元 | 普通 SNN 神经元 |
|---|---|---|
| 位于 | Block 内核心（D×N 个） | 编码层、输出层、Block 输出（D 个） |
| β | **动态**：由 spike_in + V[t-1] 每步计算 | **固定**：训练后不变 |
| α | **动态**：由 spike_in + V[t-1] 每步计算 | 无（直通输入） |
| V_th | **动态**：由 spike_in + V[t-1] 每步计算 | **固定**：训练后不变 |
| 角色 | 选择性记忆（核心能力） | 信号转换（spike↔电流↔spike） |
| 对应 Mamba | SSM 模块 | SSM 以外的标准组件 |

### 5.6 与 Mamba Block 的结构对应

```
Mamba Block:                              Our SNN Block:
x → Linear (D→2E, 分两路)                spike_in → 六条并行SNN突触路径
  ┌─ Path A: Conv1D→SiLU→SSM ──┐          ┌─ W_in, W_β, W_α, W_th → 隐状态空间 ─┐
  └─ Path B: SiLU (gate) ──────┤          └─ W_gate (gate) ────────────────┤
                         × 相乘 ↓                                    × 门控 ↓
                    Linear (E→D)                                W_out + 残差
                    + residual                                 输出PLIF → spike
                         ↓                                          ↓
                    output (实数)                             spike_out (二值)
```

| 结构特征 | Mamba | 我们的 SNN |
|---|---|---|
| 串行记忆路径 | SSM（线性递归，维护 h） | 隐状态空间（非线性递归，维护 V） |
| 并行门控路径 | SiLU(Linear(x))，无状态 | sigmoid(W_gate · spike_in)，无状态 |
| 选择性来源 | Δ,B,C 由 x 计算 | β,α,V_th 由 spike_in + V[t-1] 计算 |
| 层间通信 | 实数值 | spike（0/1） |
| 维度扩展 | D→E（expand factor） | D→D×N（N 个状态神经元/通道） |
| 残差连接 | 加实数值 | 加电流（spike→W_skip→电流，在输出神经元前求和） |

### 5.7 时间步对齐：K 步流水线同步处理

#### 5.7.1 时间索引体系

定义三层时间索引：

- **外部序列索引** $n = 1, 2, \ldots, T$：对应 T 个输入 token
- **token 内步索引** $k = 1, 2, \ldots, K$：每个 token 对应 K 个 SNN 内部时间步
- **全局 SNN 时间步** $\tau = (n-1) \cdot K + k$，$\tau \in \{1, 2, \ldots, K \cdot T\}$

K 为全网固定超参数（如 K=8），由网络边界的编码精度需求决定（K-bit 二进制编码→ $2^K$ 级量化）。**K 同时也是每个 token 的 SNN 动力学步数——更大的 K 给隐状态更多演化时间。**

#### 5.7.2 全网同步处理的形式化

**编码层（网络输入边界）**：

Token $x_n \in \mathbb{R}^D$ 通过编码器产生 K 个 spike 帧：

$$e_d[n, k] = \text{bit}_k\!\left(\text{encode}(x_{n,d})\right) \in \{0,1\}, \quad d = 1, \ldots, D, \quad k = 1, \ldots, K$$

其中 $\text{bit}_k$ 提取 MSB-first 二进制编码的第 $k$ 位。记 $\mathbf{e}[n,k] \in \{0,1\}^D$。

**Block $l$ 在全局时间步 $\tau$ 的完整计算**：

**(a) 输入确定**：

$$\text{spike}_{in}^{(l)}[\tau] = \begin{cases} \mathbf{e}[n, k] & l = 1 \text{（来自编码器）} \\ \text{spike}_{out}^{(l-1)}[\tau] & l > 1 \text{（来自上一 Block 在同一步 τ 的输出）} \end{cases}$$

**(b) 六条并行路径**：

$$I^{(l)}[\tau] = W_{in}^{(l)} \cdot \text{spike}_{in}^{(l)}[\tau] \in \mathbb{R}^{D \times N}$$

$$\beta^{(l)}[\tau] = \sigma\!\left(W_{\beta}^{(x,l)} \cdot \text{spike}_{in}^{(l)}[\tau] + W_{\beta}^{(V,l)} \cdot V^{(l)}[\tau{-}1] + b_{\beta}^{(l)}\right) \in (0,1)^{D \times N}$$

$$\alpha^{(l)}[\tau] = \text{softplus}\!\left(W_{\alpha}^{(x,l)} \cdot \text{spike}_{in}^{(l)}[\tau] + W_{\alpha}^{(V,l)} \cdot V^{(l)}[\tau{-}1] + b_{\alpha}^{(l)}\right) \in \mathbb{R}_+^{D \times N}$$

$$V_{th}^{(l)}[\tau] = V_{min} + \left|W_{th}^{(x,l)} \cdot \text{spike}_{in}^{(l)}[\tau] + W_{th}^{(V,l)} \cdot V^{(l)}[\tau{-}1] + b_{th}^{(l)}\right| \in \mathbb{R}_+^{D \times N}$$

$$\text{gate}^{(l)}[\tau] = \sigma\!\left(W_{gate}^{(l)} \cdot \text{spike}_{in}^{(l)}[\tau]\right) \in (0,1)^D$$

$$I_{skip}^{(l)}[\tau] = W_{skip}^{(l)} \cdot \text{spike}_{in}^{(l)}[\tau] \in \mathbb{R}^D$$

**(c) 隐状态更新**：

$$V^{(l)}[\tau] = \beta^{(l)}[\tau] \cdot V^{(l)}[\tau{-}1] + \alpha^{(l)}[\tau] \cdot I^{(l)}[\tau]$$

$$s^{(l)}[\tau] = \Theta\!\left(V^{(l)}[\tau] - V_{th}^{(l)}[\tau]\right) \in \{0,1\}^{D \times N}$$

$$V^{(l)}[\tau] \leftarrow V^{(l)}[\tau] - V_{th}^{(l)}[\tau] \odot s^{(l)}[\tau] \quad \text{(soft reset)}$$

**(d) 输出**：

$$\text{gate}^{(l)}[\tau] = \sigma\!\left(W_{gate}^{(l)} \cdot \text{spike}_{in}^{(l)}[\tau]\right) \in (0,1)^D$$

$$I_{out}^{(l)}[\tau] = W_{out}^{(l)} \cdot s^{(l)}[\tau] \odot \text{gate}^{(l)}[\tau] + I_{skip}^{(l)}[\tau]$$

$$V_{out}^{(l)}[\tau] = \beta_{out} \cdot V_{out}^{(l)}[\tau{-}1] + I_{out}^{(l)}[\tau]$$

$$\text{spike}_{out}^{(l)}[\tau] = \Theta\!\left(V_{out}^{(l)}[\tau] - V_{th,out}\right) \in \{0,1\}^D$$

**解码层（网络输出边界）**：

收集 Block L 处理 token n 的 K 个输出 spike 后解码：

$$\hat{y}_{n,d} = \text{decode}\!\left(\text{spike}_{out,d}^{(L)}[(n{-}1)K{+}1], \ldots, \text{spike}_{out,d}^{(L)}[nK]\right)$$

二进制解码（MSB-first）：$\hat{y}_{n,d} = \sum_{k=1}^{K} \text{spike}_{out,d}^{(L)}[(n{-}1)K{+}k] \cdot 2^{-(k)}$

#### 5.7.3 隐状态跨 token 连续性证明

**命题 1**：Block $l$ 的隐状态 $V^{(l)}$ 在 token 边界处连续演化，不存在重置。

**证明**：Token $n$ 的最后一步对应全局时间步 $\tau_{\text{end}} = nK$。Token $n{+}1$ 的第一步对应 $\tau_{\text{start}} = nK + 1$。

由递推公式：

$$V^{(l)}[\tau_{\text{start}}] = \beta^{(l)}[\tau_{\text{start}}] \cdot V^{(l)}[\tau_{\text{start}} - 1] + \alpha^{(l)}[\tau_{\text{start}}] \cdot I^{(l)}[\tau_{\text{start}}]$$

其中 $V^{(l)}[\tau_{\text{start}} - 1] = V^{(l)}[\tau_{\text{end}}] = V^{(l)}[nK]$——恰好是上一 token 结束时的隐状态。

递推公式对全局时间步 $\tau$ 统一定义，不包含 token 边界的条件分支。因此 V 跨 token 边界连续演化。$\square$

**推论**：在全局时间步 $\tau$，Block $l$ 的隐状态携带从 $\tau=1$ 起所有输入的指数加权历史：

$$V^{(l)}[\tau] = \sum_{j=1}^{\tau} \left(\prod_{m=j+1}^{\tau} \beta^{(l)}[m]\right) \cdot \alpha^{(l)}[j] \cdot I^{(l)}[j] \; - \; \sum_{\{j \,:\, s^{(l)}[j]=1\}} \text{reset\_correction}^{(l)}[j, \tau]$$

其中第一项是全部历史输入的加权和（权重为累积衰减因子 $\prod \beta$），第二项为每次 spike 发放时 soft reset 的累积修正。膜电位 V 同时携带**全部 token 的全部内部步**的信息——不区分"同一 token 的 K 步"和"跨 token 的步"。

#### 5.7.4 块间同步对齐证明

**命题 2**：在流水线同步模式下，Block $l$ 在时间步 $\tau$ 的输入严格等于 Block $l{-}1$ 在同一时间步 $\tau$ 的输出。不存在延迟或错位。

**证明**：在每个全局时间步 $\tau$，计算按 $l = 1, 2, \ldots, L$ 的顺序串行执行：

1. Block 1：接收 $\mathbf{e}[n,k]$，完成 (a)-(d) 全部计算，产生 $\text{spike}_{out}^{(1)}[\tau]$
2. Block 2：接收 $\text{spike}_{out}^{(1)}[\tau]$（Block 1 的新鲜输出），完成 (a)-(d)，产生 $\text{spike}_{out}^{(2)}[\tau]$
3. ...
4. Block L：接收 $\text{spike}_{out}^{(L-1)}[\tau]$，完成 (a)-(d)，产生 $\text{spike}_{out}^{(L)}[\tau]$

$\text{spike}_{in}^{(l)}[\tau]$ 在被使用前已由 Block $l{-}1$ 在同一步 $\tau$ 计算完成。因此全部 L 个 Block 在同一全局步 $\tau$ 处理同一"时刻"的信号。$\square$

**注**：这是 SpikingJelly `step_mode='s'`（单步模式）的天然行为。不需要额外同步、缓冲或对齐机制。

#### 5.7.5 块间无需编解码的证明

**命题 3**：Block 间传递的 spike 直接作为突触输入驱动下游神经元膜电位，无需解码为实数再重新编码。

**证明**：

设 Block $l$ 在步 $\tau$ 的输出 $\text{spike}_{out}^{(l)}[\tau] \in \{0,1\}^D$。Block $l{+}1$ 的输入处理为：

$$I^{(l+1)}[\tau] = W_{in}^{(l+1)} \cdot \text{spike}_{out}^{(l)}[\tau]$$

这是标准 SNN 突触操作：spike $\times$ W = 突触电流。$W_{in}^{(l+1)}$ 的第 $j$ 列在 spike $j$ 发放时被选中注入下游神经元。

Block $l{+}1$ 的膜电位在 K 步内逐步积累来自 Block $l$ 的全部 K 个 spike 帧的信息：

$$V^{(l+1)}[nK] = \sum_{k=1}^{K} \left(\prod_{j=k+1}^{K} \beta^{(l+1)}[(n{-}1)K{+}j]\right) \cdot \alpha^{(l+1)}[(n{-}1)K{+}k] \cdot W_{in}^{(l+1)} \cdot \text{spike}_{out}^{(l)}[(n{-}1)K{+}k] + \text{(历史项 + reset修正)}$$

信息通过三个维度在 Block 间传递：

1. **空间模式**：每步 $\tau$ 有 D 维 spike 向量，D 位同时携带 D 维信息
2. **时间模式**：K 步内的 spike 时间序列，信息由"哪些神经元在哪些步发放"的时空结构承载
3. **膜电位积累**：下游 Block 的 V 在 K 步内逐步积累来自上游的突触电流，α 和 β 调制决定积累方式

**不需要将 K 个 spike 解释为 K-bit 二进制数再还原为实数**——Block 间的信息载体是 spike 驱动的膜电位积累过程本身。编解码仅在网络边界（输入编码层/输出解码层）发生。$\square$

#### 5.7.6 多时间尺度在 K 步内的动力学分析

**命题 4**：不同时间尺度的神经元在同一 token 的 K 步内自然分化为不同的功能角色。

**分析**：

考虑 Block $l$ 中通道 $d$ 的第 $j$ 个状态神经元，设其在 token $n$ 的 K 步内有效 $\beta$ 为近似常数 $\bar{\beta}_j$（忽略步内调制的小波动，用于渐近分析）。

Token $n$ 结束时（经过 K 步），该神经元的状态为（忽略 reset 修正）：

$$V_{d,j}[nK] = \underbrace{\bar{\beta}_j^K}_{\text{token级保留率}} \cdot V_{d,j}[(n{-}1)K] \;+\; \sum_{k=1}^{K} \bar{\beta}_j^{K-k} \cdot \alpha_{d,j}[(n{-}1)K{+}k] \cdot I_{d,j}[(n{-}1)K{+}k]$$

定义 **token 级有效保留率**：$\bar{A}_j \triangleq \bar{\beta}_j^K$

| 神经元 | $\bar{\beta}_j$ | $\bar{A}_j = \bar{\beta}_j^{K}$ (K=8) | token 级混合时间 $\approx \frac{1}{\lvert\ln \bar{A}_j\rvert}$ | 功能角色 |
|---|---|---|---|---|
| 短程 $j=1$ | 0.80 | 0.168 | ~0.56 token | token 内细节处理，每 token 几乎重置 |
| 中短 $j=3$ | 0.90 | 0.430 | ~1.18 token | 相邻 token 桥接 |
| 中程 $j=5$ | 0.95 | 0.663 | ~2.43 token | 短语/子句级别记忆 |
| 长程 $j=8$ | 0.99 | 0.923 | ~12.5 token | 段落/全局级别记忆 |

**K 步内的短程 vs 长程行为**：

设 token $n$ 的 K 步内输入电流均值为 $\bar{I}$，写入增益均值为 $\bar{\alpha}$。

- **短程神经元**（$\bar{\beta} = 0.80$）：K=8 步后旧状态仅剩 16.8%。K 步内的电流累积系数 $\sum_{k=0}^{K-1} \bar{\beta}^k = \frac{1-\bar{\beta}^K}{1-\bar{\beta}} = \frac{1-0.168}{0.20} = 4.16$。该神经元在一个 token 内几乎完全由当前 token 的输入主导——**专注于 token 内细粒度结构**。
- **长程神经元**（$\bar{\beta} = 0.99$）：K=8 步后旧状态保留 92.3%。K 步内的电流累积系数 $= \frac{1-0.923}{0.01} = 7.70$。虽然累积系数更大，但由于旧状态保留率高（0.923），当前 token 的写入仅占总状态的一小部分——**主要由跨 token 的历史主导**。

定量地，token $n$ 结束后，当前 token 信息占总状态的比例为：

$$\rho_j^{(\text{current})} = \frac{\bar{\alpha} \cdot \bar{I} \cdot \frac{1-\bar{\beta}_j^K}{1-\bar{\beta}_j}}{\bar{A}_j \cdot |V_{\text{old}}| + \bar{\alpha} \cdot \bar{I} \cdot \frac{1-\bar{\beta}_j^K}{1-\bar{\beta}_j}}$$

当 $V_{\text{old}}$ 和 $\bar{I}$ 数量级相当时：$\rho_1^{(\text{current})} \gg \rho_8^{(\text{current})}$——短程神经元以当前 token 为主，长程神经元以历史为主。**多时间尺度自然对齐，无需人工干预。**

**N×N 交互矩阵 $W^{(V)}$ 的步内协调**：

在每个时间步 $\tau$，$W^{(V)} \cdot V[\tau{-}1]$ 使不同时间尺度的神经元 V 相互影响 $\beta$、$\alpha$、$V_{th}$ 的计算：

- 短程神经元 $V$ 快速变化（检测到新特征）$\xrightarrow{W^{(V)}}$ 长程神经元 $\beta$ 被调低 → 长程开始接纳新信息
- 长程神经元 $V$ 持续高位（已有重要积累）$\xrightarrow{W^{(V)}}$ 短程神经元 $V_{th}$ 被调高 → 短程更谨慎发放，减少对下游的干扰

这种跨时间尺度协调在**每个 SNN 时间步**都发生（K 次/token），不需要等 token 处理完成。比 Mamba 的对角 A（N 个状态完全独立）具有更强的多尺度协调能力。

#### 5.7.7 调制参数的计算时机

**每个 SNN 时间步重新计算**（不是每 token 计算一次）：

- 步 $\tau_1 = (n{-}1)K + 1$：$\text{spike}_{in}[\tau_1]$ 到来，与当前 $V[\tau_1 - 1]$ 一起计算 $\beta[\tau_1]$, $\alpha[\tau_1]$, $V_{th}[\tau_1]$
- 步 $\tau_2 = (n{-}1)K + 2$：$\text{spike}_{in}[\tau_2]$ 到来，V 已更新过一次（包含 $\tau_1$ 的积累和可能的 reset），计算新的 $\beta[\tau_2]$, $\alpha[\tau_2]$, $V_{th}[\tau_2]$
- ...每步的调制都反映**最新的输入和最新的状态**

这比 Mamba 更细粒度：

| | Mamba | 我们的 SNN |
|---|---|---|
| 调制频率 | 1次/token | **K 次/token** |
| 调制输入 | $x_t$（无状态） | $\text{spike}_{in}[\tau] + V[\tau{-}1]$（有状态） |
| 步内状态变化 | 无（单步更新） | 有（K 步内 V 持续演化，每步触发新调制） |

#### 5.7.8 与 Mamba $\Delta$ 的等价性映射

**Mamba 的 per-token 状态更新**：

$$h[n] = \bar{A}[n] \cdot h[n{-}1] + \bar{B}[n] \cdot x_n$$

其中 $\bar{A}[n] = \exp(\Delta(x_n) \cdot A)$，$\bar{B}[n] = \Delta(x_n) \cdot B(x_n)$。一个 token，一次状态更新。

**我们的 SNN per-token 等效更新**：

将一个 token 的 K 步递推合并。忽略 spike reset 的非线性项，做线性近似：

$$V^{(l)}[nK] \approx \underbrace{\left(\prod_{k=1}^{K} \beta^{(l)}[(n{-}1)K{+}k]\right)}_{\triangleq\; \bar{A}_{eff}^{(l)}[n]} \cdot V^{(l)}[(n{-}1)K] \;+\; \underbrace{\sum_{k=1}^{K} \left(\prod_{j=k+1}^{K} \beta^{(l)}[(n{-}1)K{+}j]\right) \cdot \alpha^{(l)}[(n{-}1)K{+}k] \cdot I^{(l)}[(n{-}1)K{+}k]}_{\triangleq\; \bar{B}_{eff}^{(l)}[n]}$$

简记为：

$$\boxed{V[nK] \approx \bar{A}_{eff}[n] \cdot V[(n{-}1)K] + \bar{B}_{eff}[n]}$$

**与 Mamba 同构**：$h[n] = \bar{A}[n] \cdot h[n{-}1] + \bar{B}[n] \cdot x_n$。

但我们的版本更丰富：

| 特性 | Mamba | SNN（线性近似） | SNN（完整版） |
|---|---|---|---|
| $\bar{A}$ 的计算 | $\exp(\Delta(x_n) \cdot A)$，一次计算 | $\prod_{k=1}^{K} \beta[\tau_k]$，K 个因子之积 | 同左 + spike reset 引入非线性修正 |
| $\bar{B}$ 的计算 | $\Delta(x_n) \cdot B(x_n) \cdot x_n$，一次计算 | $\sum_{k=1}^{K}$ 加权累积，α 逐步调制 | 同左 + 步内 spike 可重分配电流 |
| 步内非线性 | 无 | 无（近似掉了） | **有**：阈值+reset 在 K 步内持续作用 |
| N×N 交互 | 无（A 对角） | K 次 $W^{(V)}$ 反馈 | K 次 $W^{(V)}$ 反馈 |
| 等效步长控制 | $\Delta$ 可变 → 步长可变 | K 固定 × $\beta$ 可变 → 有效速率可变 | 同左 + α 独立控制写入量 |

**本质等价**：

$$\text{Mamba: 可变步长} \times \text{固定动力学} = \text{可变有效步进}$$
$$\text{SNN: 固定 K 步} \times \text{可变动力学速率 (β, α)} = \text{可变有效步进}$$

两者在 token 级别产生等价的宏观效果，但 SNN 版本具有更细的粒度（K 次调制 vs 1 次）和更强的非线性（spike+reset）。

#### 5.7.9 K 的确定

K 由**网络边界的编码精度需求**决定，与中间层无关：

| K | 边界量化精度 | 每 token 动力学步数 | 适用场景 |
|---|---|---|---|
| 4 | 16 级 | 4 步 | 粗粒度分类任务 |
| 8 | 256 级 | 8 步 | 通用任务（推荐起步） |
| 16 | 65536 级 | 16 步 | 高精度回归/生成任务 |

K 更大 → 边界精度更高，且隐状态每 token 有更多演化时间（短程神经元可在 token 内多次发放/reset，产生更丰富的动力学）。代价是计算量线性增长（$\text{总步数} = K \times T$）。

### 5.8 实现规范：张量形状、信号类型与框架兼容性

> 本节记录所有将数学公式映射到代码所需的实现细节，确保设计不因遗忘而失真。

#### 5.8.1 张量形状约定

**批次维度**：所有张量在运行时都携带 `batch` 维度作为第 0 维。

**隐状态展平**：数学公式中 $V \in \mathbb{R}^{D \times N}$，代码中展平为 `(batch, D*N)`。D 个通道各 N 个神经元按行优先排列：索引 `[d*N + n]` 对应通道 d 的第 n 个状态神经元。

**完整张量形状表**：

| 张量 | 数学形状 | 代码形状 | 值域 | 信号类型 |
|---|---|---|---|---|
| `spike_in` | $\{0,1\}^D$ | `(batch, D)` | {0, 1} | **二值脉冲** |
| `I[t]` | $\mathbb{R}^{D \times N}$ | `(batch, D*N)` | R | 实数电流 |
| `β(t)` | $(0,1)^{D \times N}$ | `(batch, D*N)` | (0, 1) | 实数（sigmoid 输出） |
| `α(t)` | $\mathbb{R}_+^{D \times N}$ | `(batch, D*N)` | R+ | 实数（softplus 输出） |
| `V_th(t)` | $\mathbb{R}_+^{D \times N}$ | `(batch, D*N)` | [V_min, ∞) | 实数（abs + V_min） |
| `gate` | $(0,1)^D$ | `(batch, D)` | (0, 1) | 实数（sigmoid 输出） |
| `I_skip` | $\mathbb{R}^D$ | `(batch, D)` | R | 实数电流 |
| `V[t]` | $\mathbb{R}^{D \times N}$ | `(batch, D*N)` | R | 实数膜电位（内部状态） |
| `s[t]` | $\{0,1\}^{D \times N}$ | `(batch, D*N)` | {0, 1} | **二值脉冲** |
| `I_out` | $\mathbb{R}^D$ | `(batch, D)` | R | 实数电流 |
| `spike_out` | $\{0,1\}^D$ | `(batch, D)` | {0, 1} | **二值脉冲** |

**信号类型总结**：Block 间传递的只有 `spike_in`/`spike_out`（二值）。Block 内部的中间信号（I, β, α, V_th, gate, V）全部是实数。这不违反"全网 SNN"约束——所有实数值都是 SNN 突触电流或神经元内部状态，不是层间通信。

#### 5.8.2 每条路径的信号流类型

```
spike_in {0,1}^D
  │
  │  [layer.Linear(D, D*N)]    ← spike × W = 实数电流（SNN突触操作）
  │  spike 输入，实数输出
  │
  ├──→ W_in:     {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流
  ├──→ W_β^(x):  {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流（β调制的外部分量）
  ├──→ W_α^(x):  {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流（α调制的外部分量）
  ├──→ W_th^(x): {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流（V_th调制的外部分量）
  ├──→ W_gate:   {0,1}^D → R^D → sigmoid → (0,1)^D   脉冲 → 电流 → 门控值
  └──→ W_skip:   {0,1}^D → R^D             信号类型: 脉冲 → 电流（残差）

V[t-1] ∈ R^{D*N}（连续值，神经元内部状态，NOT 脉冲）
  │
  │  [nn.Linear(N, N)]   ← 标准线性层，NOT SNN突触
  │  连续值输入，连续值输出（voltage-gated 内部反馈）
  │
  ├──→ W_β^(V):  R^N → R^N (per channel d)   信号类型: 连续 → 连续
  ├──→ W_α^(V):  R^N → R^N (per channel d)   信号类型: 连续 → 连续
  └──→ W_th^(V): R^N → R^N (per channel d)   信号类型: 连续 → 连续

合并后:
  β(t) = sigmoid(W_β^(x)·spike + W_β^(V)·V + b_β)    → (0,1)^{D*N}
  α(t) = softplus(W_α^(x)·spike + W_α^(V)·V + b_α)   → R_+^{D*N}
  V_th(t) = V_min + |W_th^(x)·spike + W_th^(V)·V + b_th|  → R_+^{D*N}

隐状态空间:
  V[t] = β(t)·V[t-1] + α(t)·I[t]    实数 × 实数 + 实数 × 实数 → 实数
  s[t] = Θ(V[t] - V_th(t))           实数 → {0,1}（surrogate gradient）
  V[t] -= V_th(t)·s[t]               soft reset（实数运算）

输出:
  W_out: {0,1}^{D*N} → R^D            信号类型: 脉冲 → 电流（SNN突触）
  I_out × gate + I_skip → R^D          实数 × 实数 + 实数 → 实数
  输出 PLIF: R^D → {0,1}^D             信号类型: 电流 → 脉冲
```

#### 5.8.3 W^(V) 的高效计算

$W^{(V)}$ 是 N×N 矩阵，D 个通道共享。对每个通道 d，需计算 $W^{(V)} \cdot V[t-1]_{d,:}$（N 维向量乘 N×N 矩阵）。

**高效实现**：利用 reshape 将 D 个通道的 N 维向量合并为一次矩阵乘法。

```python
# V 形状: (batch, D*N)
V_reshaped = V.view(batch * D, N)           # (batch*D, N)
V_proj = F.linear(V_reshaped, W_V.weight)   # (batch*D, N) @ (N, N)^T → (batch*D, N)
V_proj = V_proj.view(batch, D * N)          # (batch, D*N)
```

计算量：$O(\text{batch} \times D \times N^2)$。与对每个 d 单独做 N×N 乘法等价，但利用了 GPU 的批量矩阵乘法效率。

#### 5.8.4 首时间步处理

SpikingJelly 的 `BaseNode` 将膜电位 `v` 初始化为 float `0.`（通过 `register_memory('v', 0.)`）。首次 `single_step_forward` 时，`v_float_to_tensor(x)` 将其扩展为与输入同形的全零张量。

**SNNBlock 的特殊处理**：在首时间步，`hidden_neuron.v` 仍是 float `0.`（尚未被 `v_float_to_tensor` 扩展）。但 W^(V) 需要 V[t-1] 的张量形式。因此 SNNBlock 需显式检查：

```python
if isinstance(self.hidden_neuron.v, float):
    V_prev = torch.zeros(batch, D*N, device=spike_in.device)
else:
    V_prev = self.hidden_neuron.v
```

这只在每次 `reset_net()` 后的首步发生。后续步 V 已是张量。

#### 5.8.5 SpikingJelly 兼容性

**继承关系**：

```
spikingjelly.activation_based.base.MemoryModule
  └── spikingjelly.activation_based.neuron.BaseNode
        └── SelectivePLIFNode（自定义隐状态神经元）

spikingjelly.activation_based.base.MemoryModule
  └── SNNBlock（自定义 Block，内含 SelectivePLIFNode + ParametricLIFNode）
```

**`functional.reset_net(net)` 兼容**：遍历 `net.modules()`，对所有有 `reset()` 方法的模块调用 `reset()`。我们的继承链保证：
- `SelectivePLIFNode` 继承 `BaseNode` → 有 `reset()` → `self.v` 被重置为 `0.`（float）
- `SNNBlock` 继承 `MemoryModule` → 有 `reset()` → 但无直接 memory（不需要）
- `SNNBlock` 内部的 `ParametricLIFNode`（输出神经元）→ 有 `reset()` → `self.v` 被重置
- 所有 `layer.Linear` → 无状态，无 `reset()` → 不受影响

**`step_mode='s'` 专用**：我们只使用单步模式。`MemoryModule.forward(*args, **kwargs)` 透传所有参数到 `single_step_forward`，支持我们的非标准签名 `(x, beta, alpha, v_th)`。**不兼容 `step_mode='m'`**（`multi_step_forward` 只传一个参数）。

**`surrogate_function`**：使用 `surrogate.Sigmoid(alpha=4.0)`。前向：Heaviside 阶跃函数（输出 0/1）；反向：sigmoid surrogate gradient。

#### 5.8.6 完整可训练参数清单

以 D=128, N=8 为例（单个 Block）：

| 参数 | PyTorch 类型 | 形状 | 参数量 | 初始化 |
|---|---|---|---|---|
| `W_in.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform + √(1−β²) 时间尺度缩放 |
| `W_beta_x.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform × 0.1（调制解耦） |
| `W_alpha_x.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform × 0.1（调制解耦） |
| `W_th_x.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform × 0.1（调制解耦） |
| `W_gate.weight` | `layer.Linear` | (D, D) = (128, 128) | 16,384 | Kaiming uniform |
| `W_skip.weight` | `layer.Linear` | (D, D) = (128, 128) | 16,384 | Kaiming uniform |
| `W_out.weight` | `layer.Linear` | (D, D\*N) = (128, 1024) | 131,072 | Kaiming uniform + 1/√p\_fire 发放率均衡 |
| `W_beta_V.weight` | `nn.Linear` | (N, N) = (8, 8) | 64 | 零矩阵 + ε对角线 (ε=0.05) |
| `W_alpha_V.weight` | `nn.Linear` | (N, N) = (8, 8) | 64 | 零矩阵 + ε对角线 (ε=0.05) |
| `W_th_V.weight` | `nn.Linear` | (N, N) = (8, 8) | 64 | 零矩阵 + ε对角线 (ε=0.05) |
| `b_beta` | `nn.Parameter` | (D\*N,) = (1024,) | 1,024 | logit-spaced: $\ln(\beta_n / (1{-}\beta_n))$, $\beta \in$ linspace(0.80, 0.99, N) |
| `b_alpha` | `nn.Parameter` | (D\*N,) = (1024,) | 1,024 | 全部 0.5413（使 softplus ≈ 1.0） |
| `b_th` | `nn.Parameter` | (D\*N,) = (1024,) | 1,024 | σ\_V 校准: $\sigma_V \cdot \Phi^{-1}(1 - p_{fire}) - V_{th,min}$（见 5.8.10） |
| `output_neuron.w` | `nn.Parameter` | scalar (1,) | 1 | $-\ln(\tau_{init} - 1)$, $\tau_{init}=2.0$ → $w = 0$, $\beta_{out}=0.5$; $V_{threshold}=0.3$ |
| **总计** | | | **691,393** | |

**注**：所有 `layer.Linear` 的 `bias=False`（无偏置）。偏置通过独立的 `nn.Parameter`（`b_beta`, `b_alpha`, `b_th`）实现，以便结构化初始化。

#### 5.8.7 非可训练超参数

| 超参数 | 值 | 含义 |
|---|---|---|
| `V_th_min` | 0.1 | 动态阈值下限，确保 $V_{th}(t) > 0$ |
| `V_th_base` | 0.5 | V_th 偏置初始化的基础值 |
| `v_reset` | `None` | Soft reset 模式：$V -= V_{th} \cdot s$ |
| `surrogate_alpha` | 4.0 | Surrogate gradient 的锐度（Sigmoid 默认值） |
| `detach_reset` | `False` | Reset 操作保留在计算图中 |
| `step_mode` | `'s'` | 单步模式（每次调用处理一个 SNN 时间步） |
| `K` | 8（推荐起步） | 每 token 的 SNN 时间步数 |
| `K_ref` | 8 | σ\_V 校准使用的参考时间步数（见 5.8.10） |
| `ε_V` | 0.05 | W^(V) 对角线初始化值 |
| `target_p_fire` | linspace(0.25, 0.08, N) | 各时间尺度的目标初始发放率 |
| `modulation_scale` | 0.1 | 调制路径权重缩放因子（解耦原则） |
| `v_threshold_out` | 0.3 | 输出 PLIF 的发放阈值（匹配 σ(I\_total)） |

#### 5.8.8 偏置初始化的精确公式

**b_beta（β 偏置）**：

```python
beta_values = torch.linspace(0.80, 0.99, N)  # N 个目标 β 值
b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))  # inverse sigmoid

# N=8 时的具体数值:
# β:     [0.800, 0.827, 0.854, 0.881, 0.909, 0.936, 0.963, 0.990]
# b_β:   [1.386, 1.563, 1.766, 2.003, 2.296, 2.676, 3.258, 4.595]

# 扩展到 D 个通道: 每个通道的 N 个神经元有相同的初始 β 分布
b_beta = b_beta_per_n.repeat(D)  # shape: (D*N,)
```

**b_alpha（α 偏置）**：

```python
b_alpha = torch.full((D * N,), 0.5413)  # softplus(0.5413) ≈ 1.0
```

**b_th（V_th 偏置）—— σ_V 校准**：

```python
K_ref = 8
sigma_I_base = math.sqrt(1.0 / 6.0)  # ≈ 0.408

# W_in 缩放后的 σ_V（推导见 5.8.10）:
# σ²_V,n = σ²_{I,base} · (1 - β_n^{2K_ref})
sigma_V_per_n = sigma_I_base * torch.sqrt(
    1.0 - beta_values ** (2 * K_ref)
)

# 目标发放率：短程 ~25% → 长程 ~8%
target_p_fire = torch.linspace(0.25, 0.08, N)

# V_th = σ_V · Φ^{-1}(1 - p_fire)
z_scores = math.sqrt(2.0) * torch.erfinv(
    2.0 * (1.0 - target_p_fire) - 1.0
)
target_V_th = sigma_V_per_n * z_scores

# b_th = target_V_th - V_th_min, 下限 0.05
b_th_per_n = torch.clamp(target_V_th - self.v_th_min, min=0.05)

# N=8 时的具体数值:
# β:         [0.800, 0.827, 0.854, 0.881, 0.909, 0.936, 0.963, 0.990]
# σ_V:       [0.403, 0.399, 0.394, 0.386, 0.373, 0.353, 0.321, 0.158]
# p_fire:    [0.250, 0.226, 0.201, 0.177, 0.153, 0.129, 0.104, 0.080]
# target_Vth:[0.272, 0.301, 0.331, 0.358, 0.382, 0.400, 0.404, 0.222]
# b_th:      [0.172, 0.201, 0.231, 0.258, 0.282, 0.300, 0.304, 0.122]

b_th = b_th_per_n.repeat(D)  # shape: (D*N,)
```

**对比旧公式**：旧 $b_{th} = V_{th,base} \cdot (1 + \gamma \cdot \beta)$ ≈ [0.70, 0.75]，远高于 σ_V ≈ [0.16, 0.40]，导致初始发放率接近 0。新公式直接从 σ_V 和目标发放率反推，确保神经元从第一步就有合理活动。

**直觉**：训练初期（W^(V) ≈ εI），β/α/V_th 几乎完全由偏置决定。随训练推进，W 学到输入/膜电位依赖的调制，偏置提供的多时间尺度结构逐渐被动态调制增强而非取代。

#### 5.8.9 代码文件结构

```
atomic_ops/
  __init__.py              # 公开 API: SelectivePLIFNode, SNNBlock
  selective_plif.py        # SelectivePLIFNode 类（继承 BaseNode）
  snn_block.py             # SNNBlock 类（继承 MemoryModule）
```

#### 5.8.10 功能引导初始化：信号传播分析与校准

> 本节推导初始化时的信号传播统计量，并基于此校准各参数，使不同时间尺度的神经元从第一步就按照预设职能工作。

##### 5.8.10.1 输入电流统计量

**Kaiming uniform 参数**：`nn.init.kaiming_uniform_(W, a=√5)` 产生 $W_{ij} \sim \text{Uniform}(-b, b)$，其中 $b = \sqrt{6 / ((1 + a^2) \cdot \text{fan\_in})} = \sqrt{6 / (6D)} = 1/\sqrt{D}$。

$$\text{Var}(W_{ij}) = \frac{(2/\sqrt{D})^2}{12} = \frac{1}{3D}$$

**输入电流方差**：spike 输入 $s_j \sim \text{Bernoulli}(p)$，$p \approx 0.5$。

$$I_i = \sum_{j=1}^{D} W_{ij} \cdot s_j, \quad E[I_i] = 0$$

$$\sigma_I^2 = \text{Var}(I_i) = \sum_{j=1}^{D} E[W_{ij}^2] \cdot E[s_j^2] = D \cdot \frac{1}{3D} \cdot p = \frac{p}{3} = \frac{1}{6}$$

$$\boxed{\sigma_{I,base} = \sqrt{1/6} \approx 0.408}$$

##### 5.8.10.2 W_in 时间尺度缩放

**问题**：不同 β 的神经元积累速度不同。稳态方差 $\sigma_V^2 = \sigma_I^2 / (1 - \beta^2)$，长程（β=0.99）比短程（β=0.80）高 16 倍。不缩放会导致长程神经元方差爆炸。

**方案**：W_in 的第 n 行乘以 $\sqrt{1 - \beta_n^2}$，使所有神经元的稳态 $\sigma_V$ 相等。

缩放后（有限 K 步）：

$$\sigma_{V,n}^2(K) = \sigma_{I,base}^2 \cdot (1 - \beta_n^2) \cdot \frac{1 - \beta_n^{2K}}{1 - \beta_n^2} = \sigma_{I,base}^2 \cdot (1 - \beta_n^{2K})$$

$$\boxed{\sigma_{V,n}(K) = \sigma_{I,base} \cdot \sqrt{1 - \beta_n^{2K}}}$$

| n | β_n | scale √(1−β²) | σ\_V(K=8) | 无缩放 σ\_V |
|---|---|---|---|---|
| 0 | 0.800 | 0.600 | 0.403 | 0.671 |
| 3 | 0.881 | 0.473 | 0.386 | 0.816 |
| 5 | 0.936 | 0.352 | 0.353 | 1.003 |
| 7 | 0.990 | 0.141 | 0.158 | 1.115 |

**效果**：缩放后 σ\_V 的极值比从 1.66× 降至 2.55×（K=8 下长程神经元因未达稳态仍有差异，K→∞ 时完全均等）。

##### 5.8.10.3 b_th 的 σ_V 校准

**旧公式的问题**：$b_{th} = V_{th,base} \cdot (1 + 0.5 \cdot \beta)$ 给出 $b_{th} \in [0.70, 0.75]$，而 W_in 缩放后 $\sigma_V \in [0.16, 0.40]$。阈值远超方差 → 初始发放率 ≈ 0%。

**新公式**：从目标发放率反推。假设 $V \sim \mathcal{N}(0, \sigma_V^2)$（中心极限定理近似），令 $P(V > V_{th}) = p_{fire}$：

$$V_{th} = \sigma_V \cdot \Phi^{-1}(1 - p_{fire})$$

$$b_{th} = \max\left(V_{th} - V_{th,min},\ 0.05\right)$$

**目标发放率设计**：
- 短程（β=0.80）：$p_{fire} = 25\%$——快速响应，频繁发放，提供即时上下文
- 长程（β=0.99）：$p_{fire} = 8\%$——慢速积累，谨慎发放，承载长程趋势
- 中间神经元：线性插值

**注意**：$V_{th}(t) = V_{th,min} + |W_{th}^{(x)} \cdot spike + b_{th}|$ 中 $W_{th}^{(x)} \cdot spike$ 项（σ ≈ 0.408）提供输入依赖的 V_th 波动。这是刻意设计：V_th 的选择性本身就是架构核心功能，初始噪声不需要被消除。

##### 5.8.10.4 W_out 发放率均衡缩放

**问题**：低发放率的长程神经元对输出的贡献 ∝ $p_{fire}$。如果不补偿，输出将被高发放率的短程神经元主导，长程信息被淹没。

**方案**：W_out 的第 n 列乘以 $1/\sqrt{p_{fire,n}}$（归一化到均值 1）。

$$\text{scale}_n = \frac{1/\sqrt{p_{fire,n}}}{\text{mean}_m(1/\sqrt{p_{fire,m}})}$$

| n | β_n | p_fire | 1/√p | 归一化 scale |
|---|---|---|---|---|
| 0 | 0.800 | 0.250 | 2.00 | 0.77 |
| 3 | 0.881 | 0.177 | 2.38 | 0.91 |
| 5 | 0.936 | 0.129 | 2.78 | 1.07 |
| 7 | 0.990 | 0.080 | 3.54 | 1.36 |

**效果**：长程神经元（β=0.99）的 W_out 列权重放大 1.36×，补偿其低发放率。

##### 5.8.10.5 W^(V) 结构化初始化

**旧方案**：`Uniform(±0.01)` — 随机且无结构。

**新方案**：零矩阵 + ε 对角线（ε=0.05）。

$$W^{(V)} = \varepsilon \cdot I_N = \begin{pmatrix} 0.05 & 0 & \cdots & 0 \\ 0 & 0.05 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & 0.05 \end{pmatrix}$$

**设计理由**：
- **对角线 +ε**：每个神经元的 V 仅反馈到自身的 β/α/V_th 调制（自监测），不干扰其他时间尺度
- **非对角线 = 0**：训练初期各时间尺度完全独立演化，避免随机跨尺度耦合干扰多时间尺度结构的建立
- **ε = 0.05（而非 0）**：给训练一个起点——纯零矩阵使 W^(V) 梯度方向完全依赖随机扰动，ε 提供有信息的初始梯度

**训练演化**：随训练推进，W^(V) 的非对角元素将从零增长，学习有意义的跨时间尺度交互（如：长程 V 高 → 短程 β 微调）。对角初始化仅提供起点，不限制最终结构。

##### 5.8.10.6 输出 PLIF 的校准

**ParametricLIFNode 方程**：$V[t] = \beta_{out} \cdot V[t-1] + (1-\beta_{out}) \cdot x[t]$

**关键洞察**：稳态 $V_{ss} = x$（输入本身）。因此 $V_{threshold}$ 直接决定"需要多大的输入才能触发输出 spike"。

**输入信号统计量**：$I_{total} = I_{out} \cdot gate + I_{skip}$

$$\sigma(I_{skip}) = \sqrt{p_{in}/3} \approx 0.41\ (p_{in} \approx 0.5)$$
$$\sigma(I_{out} \cdot gate) \approx \sigma(I_{out}) \cdot E[gate] \approx \sqrt{p_{fire}/3} \cdot 0.5 \approx 0.11$$
$$\sigma(I_{total}) \approx \sqrt{0.11^2 + 0.41^2} \approx 0.42$$

**v_threshold 校准**：以 $\sigma(I_{total}) \approx 0.42$，目标 ~20% 输出发放率：

$$V_{threshold} = \sigma(I_{total}) \cdot \Phi^{-1}(0.80) \approx 0.42 \times 0.84 = 0.35$$

取 $V_{threshold} = 0.3$，对应 ~24% 发放率。

**init_tau 选择**：$\tau = 2$（$\beta_{out} = 0.5$），因为：
- 输出神经元的职责是 **信号转换**（电流→脉冲），不是长期记忆（那是隐神经元的工作）
- 快 τ = 快响应 → 每帧的输出 spike 反映当前帧的状态，有利于 K-bit 二进制编码
- 慢 τ（如 10）会导致帧间膜电位高度相关，降低 K 帧的信息表达力

| init_tau | β\_out | V\_threshold | ~P(fire) at K=8 |
|---|---|---|---|
| **2.0** | **0.50** | **0.3** | **~20-25%** |
| 2.0 | 0.50 | 1.0 | ~3%（原始，太低） |
| 10.0 | 0.90 | 1.0 | ~0%（V 只达 0.57x，更差） |

**级联衰减**：多 Block 级联时，后续 Block 的输入 spike 率 << 50%。例如 Block 1 收到 ~8% 输入：$\sigma(I_{skip}) = \sqrt{0.08/3} \approx 0.16$，$P(fire) \approx P(z > 0.3/0.16) = P(z > 1.88) \approx 3\%$。这是深层 SNN 的固有特性，通过训练中 V_th 的自适应调节逐步改善。

##### 5.8.10.7 调制路径的解耦原则

**问题**：$V_{th}(t) = V_{min} + |W_{th}^{(x)} \cdot spike + b_{th}|$。若 $W_{th}^{(x)}$ 用 Kaiming init（σ ≈ 0.41），则 $|b_{th} + noise|$ 的均值被 $|noise|$ 主导（下限 $\sigma \sqrt{2/\pi} \approx 0.33$），b_th 的校准被噪声淹没。

**解决方案**：调制路径权重 $W_{\beta}^{(x)}, W_{\alpha}^{(x)}, W_{th}^{(x)}$ 乘以 0.1，使得：
- 训练初期：$\beta, \alpha, V_{th}$ 由偏置主导（结构化的多时间尺度设计生效）
- 训练推进：$W^{(x)}$ 逐渐学到输入依赖的调制（选择性增强）

| 权重类型 | 初始化 | 理由 |
|---|---|---|
| 信号路径（W\_in, W\_gate, W\_skip, W\_out） | Kaiming uniform | 需要合理的电流幅度 |
| 调制路径（W\_β^(x), W\_α^(x), W\_th^(x)） | Kaiming × 0.1 | 偏置主导，避免噪声淹没校准 |
| 反馈路径（W^(V)） | εI (ε=0.05) | 自监测起点，初期无跨尺度耦合 |

##### 5.8.10.8 完整初始化清单

```python
# === 功能引导初始化 ===
# 1. β偏置: logit-spaced [0.80, 0.99]       → 多时间尺度
# 2. α偏置: 0.5413 (softplus→1.0)           → 单位写入增益
# 3. 信号路径: Kaiming uniform               → 合理电流幅度
# 4. 调制路径: Kaiming × 0.1                 → 偏置主导（解耦原则）
# 5. W_in: × √(1-β²) per neuron             → σ_V 均衡
# 6. b_th: σ_V · Φ⁻¹(1-p_fire) - V_th_min  → 目标发放率
# 7. W_out: × 1/√p_fire (归一化)            → 输出贡献均衡
# 8. W^(V): εI (ε=0.05)                     → 自监测起点
# 9. Output PLIF: τ=2, v_threshold=0.3      → 快速响应 + 信号匹配
```

---

## 6. 与Mamba的关系：借鉴思想而非照搬公式

### 6.1 隐状态空间的构建对比

**Mamba 的 SSM 构建**：
- 源自连续时间 SSM：$dx/dt = Ax + Bu$，$y = Cx$，离散化后 $h[t] = \bar{A} h[t{-}1] + \bar{B} x[t]$，$y[t] = C \cdot h[t]$
- A 取对角矩阵：N 个状态维度各自独立演化，无维度间交互
- HiPPO 初始化：使 N 个维度覆盖不同时间尺度
- 选择性（S4→S6/Mamba）：$\Delta, B, C$ 变成 $x_t$ 的函数，每个 token 动态计算控制信号——这就是"Selective"的含义
- **$\Delta$ 只看 $x_t$，不看 $h[t{-}1]$**——选择性是无记忆的

**我们的 SNN 隐状态空间构建**：
- 状态载体 = D×N 个 PLIF 神经元的膜电位 $V[t] \in \mathbb{R}^{D \times N}$（累积膜电位，不是输出 spike）
- $W^{(V)}$ 为 N×N（D通道共享）：N 个时间尺度的神经元之间有交互（Mamba 的对角 A 没有）
- β 偏置初始化覆盖 0.80~0.99 多时间尺度（类比 HiPPO）
- **选择性**：β(t), α(t), V_th(t) 由 spike_in + $V[t{-}1]$ 共同计算——选择性是**有记忆的**
- β、α 和 V_th 不是训练后固定的参数，而是每步由输入动态决定的值（对应 Mamba 的选择性思想）。真正的可训练参数是调制网络的权重矩阵
- β 与 α 解耦：衰减和写入由独立网络控制，支持 Mamba 的 Δ 耦合无法实现的"高保留+高写入"组合

**逐机制对应**：

| 机制 | Mamba | SNN | 差异 |
|---|---|---|---|
| 衰减控制 | $\bar{A}[t] = \exp(\Delta(x_t) \cdot A)$ | $\beta(t) = \sigma(W \cdot spike_{in} + W \cdot V[t{-}1] + b)$ | Mamba 只看输入；我们看输入+历史 |
| 写入 | $\bar{B}[t] \cdot x_t$ | $\alpha(t) \cdot I[t] = \alpha(t) \cdot W_{in} \cdot spike_{in}$ | α 独立于 β；Mamba 的 $\bar{B}$ 经 Δ 与 $\bar{A}$ 耦合 |
| 遗忘机制 | 单一 $\bar{A}$（乘性衰减） | β衰减（乘性，每步）+ soft reset（减性，条件触发） | 我们将遗忘拆分为两个协同机制 |
| 读出 | $C(x_t) \cdot h[t]$，线性，全体参与 | $\Theta(V[t] - V_{th}(t))$，二值，稀疏选择 | 非线性门控 |
| 输出→状态反馈 | 无（读出不修改 h） | 有（soft reset 修改 V） | 双向耦合 |
| 状态维度间交互 | 无（A 对角） | 有（$W^{(V)}$ 是 N×N） | 时间尺度间可协调 |
| 门控分支 | SiLU(Linear(x))，并行无状态 | $\sigma(W_{gate} \cdot spike_{in})$，并行无状态 | Mamba 用 SiLU，我们用 sigmoid |

### 6.2 我们从Mamba借鉴的

| Mamba的思想 | SNN中的对应 | 如何借鉴 |
|---|---|---|
| 多时间尺度状态空间 | β的结构化分布 | 用HiPPO的思想初始化β，覆盖短程到长程 |
| 输入依赖的选择性 | 输入依赖的β/V_th调制 | 通过调制网络而非Mamba的Δ投影 |
| 状态维度N独立于模型维度D | 每个通道有N个状态神经元 | 直接采用，扩展状态容量 |
| 遗忘-写入耦合 | spike+reset天然耦合 | 发放=释放旧信息+腾出空间给新信息 |

### 6.3 我们不照搬的

| Mamba的组件 | SNN的对应方式 | 差异 |
|---|---|---|
| $\Delta_t = f(x_t)$ 只看输入 | $\beta(t) = f(x_t, V[t-1])$ 看输入+膜电位 | SNN的调制有完整上下文 |
| $\bar{A} = \exp(\Delta \cdot A)$ | 直接用β(t)作为衰减因子 | 不需要exp运算 |
| $B_t = \text{Linear}(x_t)$ 写入 | $W_{in} \cdot x_t$ SNN输入投影 | 类似 |
| $C_t = \text{Linear}(x_t)$ 读出 | spike模式本身就是选择性读出 | 不需要额外C_t |
| 纯线性状态方程 | 非线性（阈值+reset）+ 输入依赖参数 | 表达能力更强 |

**关键差异**：Mamba的Δ只看当前输入——同一token无论上下文如何都产生相同的Δ。我们的β(t)和V_th(t)同时看输入和膜电位——同一token在不同上下文（不同V[t-1]）中产生不同的调制。这是比Mamba更强的选择性。

### 6.4 SNN相比Mamba的潜在优势

**Mamba的状态更新是纯线性的**：$h[t] = \bar{A} h[t-1] + \bar{B} x[t]$，$y[t] = C \cdot h[t]$。没有非线性，状态可以无限增长，需要靠 $\bar{A} < 1$ 来控制。

**SNN天然具备三种Mamba没有的机制**：

1. **阈值判定 = 硬性非线性门控**
   - Mamba的所有状态都参与输出，没有"选不选"的问题
   - SNN只有超阈值的才发放，实现了真正的稀疏选择

2. **Soft reset = 主动状态压缩**
   - Mamba只靠 $\bar{A}$ 的被动衰减来控制状态
   - SNN发放后主动清除 $V -= V_{th}$，防止状态无限累积，同时保留残差

3. **二值spike = 天然信息瓶颈**
   - Mamba的输出是连续值，信息量不受限
   - SNN的spike是0/1，天然构成信息瓶颈，迫使网络学到更紧凑的表示

这些机制可能让SNN在某些方面超越Mamba的纯线性状态空间。

---

## 7. 理论定位

### 7.1 在序列建模谱系中的位置

```
Transformer (O(n²), 全序列并行)
  │
  ├── 不是SNN的方向
  │
Mamba/S6 (O(n), 选择性线性递归)
  │
  ├── 借鉴其思想（多时间尺度、选择性）
  │
SNN 隐神经元状态空间 (O(n), 非线性递归 + spike稀疏选择)
  │
  ├── 原生物理机制提供的：阈值门控、soft reset压缩、二值信息瓶颈
  │
  └── 在Mamba证明可行的范式（选择性递归）上，叠加SNN特有的非线性能力
```

### 7.2 关键理论支撑

**Linear Attention ↔ RNN 等价性**（Katharopoulos et al., ICML 2020）：

SNN时间残差 $V[t] = \beta V[t-1] + I[t]$ 在数学上属于Linear Attention的递归分支。但标准Linear Attention是纯线性的，表达能力弱于Softmax Attention。

SNN的阈值+reset在这个线性递归上叠加了非线性，理论上应该增强表达能力。这是SNN相对于纯线性递归模型（包括Mamba）的独特优势点。

**信息瓶颈理论**（Tishby et al.）：

spike阈值天然实现了信息瓶颈——只传递超阈值的信息。泛化界 $R - \hat{R} \leq C\sqrt{I(T;X)/n}$ 表明适度的信息压缩有利于泛化。V_th的设置直接控制瓶颈的松紧度。

**混合时间与遍历理论**：

$t_{mix} \sim 1/|\ln\beta|$ 给出了每个β值对应的有效记忆长度。多时间尺度的β分布 = 多个并行的信息信道，每个信道有不同的有效带宽。

---

## 8. 训练方法：信息论引导的零阶优化（IG-ZO）

> **当前方案**：以下训练框架是初步设计，将在实验中持续迭代优化。

### 8.1 大方向

**零阶优化，不反向传播。** spike 的不可微性、循环计算图的深度、soft reset 的条件分支——这些使反向传播在 SNN 中极其困难。零阶方法完全绕过梯度，只需要前向传播。

传统零阶优化（SPSA、ES）的问题：在全部参数空间中随机扰动，方差与参数维度成正比，高维空间中收敛极慢。

**我们的核心改进**：利用 Q5.md 的信息论数学工具，将任务损失的全局信号**分解**为参数组级别的**有方向性**的更新信号。不是随机扰动，而是数学引导的定向扰动。

### 8.2 设计目标驱动的诊断-处方体系

训练不是盲目降 loss。我们的架构每个组件有明确的设计意图——训练是引导每个组件学会它被设计要做的事。**L_task 是唯一的最终目标**，信息论度量负责在 L_task 不降时**定位瓶颈在哪个组件、什么问题**。

#### 设计目标 1：上下文相关的选择性

**设计意图**：同一输入在不同上下文（不同 V[t-1]）中应触发不同的 β(t)/α(t)/V_th(t)。这是隐状态空间区别于普通 LIF 层的根本能力。

**诊断度量——选择性方差**：

$$\text{Var}_{ctx}\!\left[\beta(t) \;\big|\; \text{spike}_{in} \text{ 固定}\right]$$

固定输入，只改变 V[t-1]（通过在不同序列位置提取相同输入的实例），测量 β(t) 的变化幅度。α(t)、V_th(t) 同理。

| 诊断结果 | 含义 | 病因 | 处方 |
|---|---|---|---|
| 方差 ≈ 0 | $W^{(V)}$ 没在起作用 | 选择性退化为仅输入依赖 | 增大 $W^{(V)}$ 扰动幅度和学习率 |
| 方差显著 | V[t-1] 在调制参数 | 上下文选择性在工作 | 保持当前方向 |

**关联参数**：$W_\beta^{(V)}$, $W_\alpha^{(V)}$, $W_{th}^{(V)}$（N×N 矩阵组）

#### 设计目标 2：多时间尺度分化

**设计意图**：N 个状态神经元应覆盖从短程（β≈0.80）到长程（β≈0.99），各司其职——不是全部坍缩到同一个 β。

**诊断度量**：

1. β 分布的方差/熵：训练中 N 个神经元的有效 β 是否仍然分散？
2. $MI_{retro}(k)$ 按 β 分组：不同组的 MI 衰减曲线是否分化？（短程组短、长程组长）

| 诊断结果 | 含义 | 病因 | 处方 |
|---|---|---|---|
| β 全部收敛到 ~0.93 | 时间尺度坍缩 | 训练冲刷了初始化结构 | 约束 $b_\beta$ 偏离初始化的幅度，降低 $W_\beta$ 学习率 |
| MI 曲线各组无差异 | 功能分化未建立 | 不同时间尺度没有学到不同角色 | 增大 β 偏置间距，强化初始化结构 |
| MI 按 β 分组清晰分化 | 多时间尺度在工作 | — | 保持当前方向 |

**关联参数**：$b_\beta$（偏置），$W_\beta^{(x)}$

#### 设计目标 3：静默积累的有效性

**设计意图**：静默神经元的膜电位在构建对未来有价值的上下文，不是无意义的数值叠加。

**诊断度量**（来自 Q5.md）：

1. KSG 互信息：$I(V_{silent}[t]; \; y[t+k]) > 0$
2. 线性探针准确率：$\hat{y}[t+k] = W_{probe} \cdot V_{silent}[t]$
3. Ablation：归零静默 V → 损失增大幅度

| 诊断结果 | 含义 | 病因 | 处方 |
|---|---|---|---|
| MI ≈ 0 | 积累无效 | $W_{in}$ 没写入有用信息，或 β 太低衰减光了 | 增大 $W_{in}$ 扰动；检查 β 下限 |
| MI > 0 但仅小 k | 短程有效，长程失败 | 长程 β 不够大，或 V_th 太低导致频繁发放打断积累 | 对长程组增大 β；增大长程 V_th |
| MI > 0 且大 k 仍显著 | 静默积累有效 | — | 保持当前方向 |

**关联参数**：$W_{in}$, $W_\beta$, $W_{th}$

#### 设计目标 4：发放的信息效率（信息瓶颈）

**设计意图**：V_th 控制信息瓶颈的松紧——发放应稀疏但每次携带高价值信息。

**诊断度量**：

1. 发放效率：$\eta_{fire} = \Delta I(Y; \text{downstream} \,|\, s=1) \;/\; T_{silent}$
2. 压缩率：$\rho = I(s[t]; Y) \;/\; I(V[t]; X)$
3. IB 理论最优压缩率 $\rho^*$ 作为基准

| 诊断结果 | 含义 | 病因 | 处方 |
|---|---|---|---|
| $\eta_{fire}$ 低 | 发放不携带有用信息 | V_th 太低，频繁发放稀释了信息密度 | 沿增大 V_th 方向扰动 $W_{th}$ |
| $\eta_{fire}$ 高但发放极稀疏 | 信息积压释放不出来 | V_th 太高 | 沿减小 V_th 方向扰动 $W_{th}$ |
| $\eta_{fire}$ 与 $T_{silent}$ 正相关 | 越久的积累越有价值的发放 | 设计按预期工作 | 保持当前方向 |

**关联参数**：$W_{th}^{(x)}$, $W_{th}^{(V)}$, $b_{th}$

#### 设计目标 5：衰减-写入解耦

**设计意图**：β 和 α 独立控制，支持 Mamba 的 Δ 耦合无法实现的"高保留+高写入"等组合。

**诊断度量**：

$$r = \text{Corr}\!\left[\alpha(t), \; \beta(t)\right] \quad \text{跨全部时间步}$$

| 诊断结果 | 含义 | 病因 | 处方 |
|---|---|---|---|
| $|r|$ 高（>0.7） | α 和 β 在做同一件事 | 解耦设计未被利用，退化为单因子 | 差异化扰动：推动 $W_\alpha^{(V)}$ 和 $W_\beta^{(V)}$ 学到不同的 V 响应 |
| $|r|$ 低（<0.3） | 网络在利用四个象限 | 解耦在工作 | 保持当前方向 |

**关联参数**：$W_\alpha^{(x)}$, $W_\alpha^{(V)}$, $W_\beta^{(x)}$, $W_\beta^{(V)}$

#### 设计目标 6：跨时间尺度协调

**设计意图**：短程神经元检测到新输入 → 长程应调整策略；长程积累充分 → 短程应配合。N×N 矩阵 $W^{(V)}$ 是唯一的跨尺度通道。

**诊断度量**：

$$I\!\left(\Delta V_{short}[t]; \; \Delta\beta_{long}[t]\right)$$

短程膜电位变化是否影响了长程衰减率的变化。

| 诊断结果 | 含义 | 病因 | 处方 |
|---|---|---|---|
| MI ≈ 0 | N 个时间尺度各自独立 | $W^{(V)}$ 形同虚设 | 增大 $W^{(V)}$ 的离对角元素扰动 |
| MI > 0 | 跨尺度信号在传递 | 协调在工作 | 保持当前方向 |

**关联参数**：$W_\beta^{(V)}$, $W_\alpha^{(V)}$, $W_{th}^{(V)}$（共享的 N×N 矩阵）

### 8.3 Fisher 信息引导的扰动策略

**传统 SPSA**：各参数维度等权随机扰动 $\delta_i \sim \text{Uniform}\{-1, +1\}$。

**Fisher 加权扰动**：$\delta_i \sim \mathcal{N}(0, \sigma_i^2)$，其中 $\sigma_i \propto 1/\sqrt{F_i}$。

$F_i$ 是第 $i$ 个参数组的 Fisher 信息，通过前向扰动估计（无需反向传播）：

$$F_i \approx \mathbb{E}\!\left[\left(\frac{L(\theta + \epsilon \cdot \mathbf{e}_i) - L(\theta)}{\epsilon}\right)^2\right]$$

**效果**：在 Fisher 度量下的各向同性扰动，等价于零阶版自然梯度下降。敏感参数小扰动（精细调整），不敏感参数大扰动（探索新区域）。

### 8.4 多时间尺度学习率

混合时间 $t_{mix,j} \sim 1/|\ln \bar{\beta}_j|$ 直接决定第 $j$ 个时间尺度的参数学习率：

$$\eta_j = \eta_{base} \cdot \frac{t_{mix,\min}}{t_{mix,j}}$$

| 时间尺度 | $\bar{\beta}_j$ | $t_{mix}$ | 相对学习率 | 理由 |
|---|---|---|---|---|
| 短程 | 0.80 | ~4.5 步 | 1.0× | 效果即时可见，快速调整 |
| 中程 | 0.95 | ~19.5 步 | ~0.23× | 效果延迟观测，中速调整 |
| 长程 | 0.99 | ~99.5 步 | ~0.045× | 效果需要很多 token 才体现，缓慢调整 |

理由：长程参数的效果要跨越多个 token 才能在 L_task 中体现。过快调整会在信号到达前就被反复修改，造成振荡。

### 8.5 完整算法流程

```
IG-ZO 算法

输入: 训练数据, 初始参数 θ（按设计初始化）
超参: 诊断间隔 M, 基础学习率 η_base, 扰动缩放 c

初始化:
  b_β 按多时间尺度 [0.80, 0.99] 初始化
  b_α 初始化使 α ≈ 1.0
  b_th 与 β 协同初始化
  所有 W 小随机初始化
  诊断报告 D ← 空

训练循环:

  ┌─ 每 M 步: 信息论诊断 ───────────────────────────────┐
  │                                                       │
  │  收集最近 M 步的 (V, s, spike_in, y) 数据             │
  │                                                       │
  │  目标1: Var_ctx[β | same input]       → 选择性方差     │
  │  目标2: β 分布熵 + MI_retro(k) 分组   → 多尺度分化    │
  │  目标3: I(V_silent; y_future)          → 静默积累值    │
  │  目标4: η_fire + ρ                    → 发放效率      │
  │  目标5: Corr(α, β)                    → 解耦程度      │
  │  目标6: I(ΔV_short; Δβ_long)          → 跨尺度协调    │
  │  Fisher: 各参数组前向扰动敏感度         → 扰动幅度      │
  │                                                       │
  │  D ← 诊断报告（每组参数的处方方向 + 幅度 + 学习率）   │
  └───────────────────────────────────────────────────────┘

  每步训练:
    1. 前向传播: 计算 L_task
    2. 生成定向扰动 δ:
       - 有处方的参数组: 沿诊断指示的方向，Fisher 加权幅度
       - 无处方/无异常的参数组: Fisher 加权随机扰动
    3. 双向前向: L+ = L(θ + c·δ), L- = L(θ - c·δ)
    4. 梯度估计: ĝ = (L+ - L-) / (2c) · δ^{-1}
    5. 分组更新: θ_j ← θ_j - η_j · ĝ_j
       （η_j 按时间尺度缩放）
```

### 8.6 与纯 SPSA 的对比

| | 纯 SPSA | IG-ZO（本方案） |
|---|---|---|
| 扰动方向 | 随机 | 诊断引导 + Fisher 加权 |
| 扰动幅度 | 统一 | 按参数组 Fisher 加权 |
| 学习率 | 统一 | 按 $t_{mix}$ 分组 |
| 有效维度 | 全参数空间 | 压缩到 6 个设计目标的诊断维度 |
| L_task 不降时 | 继续随机搜索 | 诊断定位瓶颈 → 针对性调整 |
| 训练信号 | 仅 L_task | L_task + 6 个信息论诊断 |
| 收敛效率 | 低（高维高方差） | 较高（定向低方差） |

### 8.7 安全机制

**处方方向最终由 L_task 验证**：诊断只提供扰动的偏置方向。如果处方错了，$L^+ > L^-$ 不会成立，更新方向自动由 L_task 主导。诊断是建议，L_task 是仲裁。

**β 分布保护**：若诊断发现时间尺度坍缩，可对 $b_\beta$ 施加硬约束：$b_{\beta,j} \in [b_{\beta,j}^{(init)} - \epsilon, \; b_{\beta,j}^{(init)} + \epsilon]$，防止训练完全冲刷多时间尺度结构。

**诊断开销控制**：KSG 互信息和 Fisher 估计只在每 M 步执行，不增加逐步训练的计算代价。M 步内的训练使用上一次诊断的缓存结果。

---

## 9. 实现路径与开放问题

### 9.1 最小可行实验

**目标**：验证"输入依赖的β/α/V_th + spike/静默分化"是否能提供有效的上下文选择性。

**设计**：
1. 隐神经元空间：D=128, N=8（1024个隐神经元），K=8（每 token 8个SNN时间步）
2. β调制网络：$W_\beta^{(x)}$（128→1024, SNN）+ $W_\beta^{(V)}$（N×N 共享），偏置按多时间尺度初始化 [0.80, 0.99]
3. α调制网络：$W_\alpha^{(x)}$（128→1024, SNN）+ $W_\alpha^{(V)}$（N×N 共享），偏置初始化使 α≈1.0
4. V_th调制网络：$W_{th}^{(x)}$（128→1024, SNN）+ $W_{th}^{(V)}$（N×N 共享），偏置与β协同初始化
5. 全网 SNN：所有 W 层为 SNN 突触层（spike 输入），隐状态空间为特殊 PLIF（动态 β/α/V_th），输出层为普通 PLIF（固定参数），含门控路径和残差路径（见第 5 节架构），时间步对齐按 5.7 节的 K 步流水线同步处理
6. 训练采用 IG-ZO（信息论引导的零阶优化，见第 8 节）：六个设计目标的诊断驱动定向扰动，Fisher 加权，多时间尺度学习率
7. 在Sequential MNIST或简单序列任务上测试

**观测指标**：
- β(t)是否随不同(输入, 膜电位)组合产生不同分布
- **关键验证**：同一token在不同上下文（不同V[t-1]）中是否产生不同的β(t)和V_th(t)
- V_th(t)是否在膜电位积累充足时自动降低
- 不同通道的发放频率分布是否分化
- spike模式的稀疏度随训练的变化
- $W^{(V)}$ 的权重模式：是否学到了有意义的膜电位→调制映射

**对照组**：
- A组（完整版）：β(t), V_th(t) 依赖输入 $x_t$ + 膜电位 $V[t-1]$
- B组（仅输入依赖）：β(t), V_th(t) 只依赖 $x_t$，$W^{(V)}=0$
- C组（固定参数）：β和V_th是可训练常数
- D组（无结构）：全部β=0.95, V_th=1.0

A vs B 验证膜电位反馈的必要性。B vs C 验证输入依赖的必要性。C vs D 验证结构化初始化的价值。

### 9.2 开放问题

**Q1：输出应该用spike还是膜电位？** → 已确定

输出为 spike（0/1），通过二进制编码（MSB-first）解读为实数值。K bit 精度由模型需求决定。

**Q2：W_in和W_out是否应该有时间状态？** → 属于整体架构设计问题，非本模块内部问题，随架构确定。

**Q3：N的选择？** → 实验超参数，非设计阶段问题，由实验确定。

**Q4：β调制网络的复杂度？** → 已确定

$W^{(V)}$ 系列：通道独立 + 跨通道共享 N×N。$W_\beta^{(V)}$ = N² = 64，$W_\alpha^{(V)}$ = 64，$W_{th}^{(V)}$ = 64，**共 192 个参数**（N=8）。
$W^{(x)}$ 系列：标准线性层 D→D×N。$W_\beta^{(x)}$ + $W_\alpha^{(x)}$ + $W_{th}^{(x)}$ 参数量 = $3 \times D \times D \times N$。D=128, N=8 时 ≈ 393K。
$W_{in}$ = D×D×N = 131K，$W_{out}$ = D×N×D = 131K，$W_{gate}$ = D×D = 16K，$W_{skip}$ = D×D = 16K。
偏置：$b_\beta$ + $b_\alpha$ + $b_{th}$ = 3 × D×N = 3072。
输出 PLIF 神经元：1 个可训练参数（$w$，控制 $\beta_{out}$）。
**单个 Block 总参数量**（D=128, N=8）：≈ 688K + 192 + 3072 + 1 ≈ **691K**。
如需降低 $W^{(x)}$ 参数量，可考虑对角映射（D→D 广播到 N），参数量降至 $3 \times D$ ≈ 384。起步时可先验证。

**Q5：如何验证"静默积累"确实在构建有用的上下文？** → 已解决

完整验证方法论见 Q5.md。三阶段验证：存在性验证（KSG 互信息 + 线性探针 + Ablation）、多时间尺度验证、动态调制验证。

---

## 10. 参考来源

### 思想来源
- **Mamba/S6**（Gu & Dao, 2023）：多时间尺度状态空间、选择性机制的思想启发
- **HiPPO**（Gu et al., 2020）：多时间尺度结构化初始化的概念
- **Linear Attention ↔ RNN**（Katharopoulos et al., ICML 2020）：SNN时间残差的理论定位

### 理论基础
- 混合时间 $t_{mix} \sim 1/|\ln\beta|$ 给出有效记忆长度
- 生物物理β约束（0.8-0.99）
- Linear Attention ↔ RNN 等价性，SNN时间残差的理论定位
- 信息瓶颈与泛化界，spike阈值的信息论意义

---

*本文档记录了SNN隐神经元状态空间的设计思考。核心机制是：β、α和V_th由当前输入和隐神经元膜电位 $V[t-1]$ 共同动态计算——膜电位携带全部历史上下文，使得同一token在不同上下文中触发不同的选择行为。训练采用信息论引导的零阶优化（IG-ZO），将任务损失分解为六个设计目标的诊断信号，实现参数组级别的定向更新。*

*状态: 概念设计阶段，架构和训练方法均为当前方案，将在实验中持续迭代优化。*
