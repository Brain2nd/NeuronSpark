# SNN 隐神经元状态空间：设计思考

> 日期: 2025-02（初版），2026-02（v7.5 更新）
> 状态: **已实现并训练中**（v7.5: Parallel Scan + 连续残差流 + Surrogate Gradient Backprop）
> 核心思路: 利用SNN神经元的**原生物理机制**（spike/静默、泄漏、soft reset、阈值）构建选择性记忆的隐状态空间，不是对Mamba的SNN翻译
> **关键设计**: β、α和V_th由当前输入动态计算，不是训练后固定的常数——这是隐神经元空间区别于普通LIF层的根本机制
> **架构演进**: v6（串行）→ v7（Parallel Scan，移除 W^(V)）→ v7.1（Surrogate Gradient 反向传播替代 SPSA）→ v7.5（连续残差流解决深层梯度消失）
> **注**: 第 1-6 节为原始概念设计（含 W^(V)），第 7 节记录 v7 并行化（W^(V) 移除），第 7.13 节记录 v7.5 连续残差流

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

> **v7 变更**：W^(V) 系列参数（膜电位反馈路径）已在 v7 中移除，以满足 parallel scan 的前提条件。详见第 7.2 节。以下保留原始设计供参考，v7 实际实现仅使用 W^(x) 路径。

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

1. **SNN 核心计算**：所有信号处理通过 SNN 神经元（spike 发放 + 膜电位动力学）。v7.5 引入连续残差流（层间传递连续值 h），但 SNN 子层内部严格使用 spike 通信——残差流是梯度传播的工程解决方案，不改变 SNN 子层本身的计算语义（详见 7.13 节）
   > **注**：原设计（v6/v7）要求"所有层间通信通过 spike（0/1）"。v7.5 为解决 20 层梯度消失而引入 PLIFNode → SNN子层 → out_proj → 残差连接的结构，层间传递连续值。PLIFNode 直接接收原始 h（未归一化），其 D 维可学习 V_th 作为 SNN 原生的幅度调节机制。SNN 核心计算（SelectivePLIFNode 隐状态空间、门控、AND 门等）保持不变。LateralInhibition 仅用于模型输出层（decode_proj 之后）。
2. **只有隐状态空间使用特殊设计**：动态 β(t)/V_th(t) 仅用于负责记忆的隐状态神经元（SelectivePLIFNode）。其余神经元为 PLIFNode（D 维可学习 β/V_th），负责信号转换和维度变换——就像 Mamba 中只有 SSM 负责隐状态，其余组件（Linear/Conv1D/SiLU）是标准计算
3. **受 Mamba 启发的双分支结构**：串行记忆路径（对应 SSM）+ 并行门控路径（对应 Gate 分支），门控路径只看当前输入、无时间状态

### 5.2 模型级架构

> **v7.5 更新**：引入连续残差流。层间传递连续值 h，SNN 子层内部仍为 spike 通信。

```
原始输入（实数）
  ↓
[Embedding + encode_proj + sigmoid]  实数 → [0,1]^D
  ↓
[K-bit 二进制编码 (STE)]  [0,1]^D → {0,1}^{D×K}  ← K 帧 spike（也是合法连续值）
  ↓
h ∈ R^{TK×batch×D}（连续残差流，初始值为 spike 帧）
  ↓
[SNNDecoderLayer 1]  h → PLIFNode → spike → SNNBlock → spike → out_proj → +h
                     h → PLIFNode → spike → SNNFFN  → spike → out_proj → +h
  ↓
[SNNDecoderLayer 2]  同上
  ↓
  ...（L 层，每层内部独立维护隐状态空间 V_l）
  ↓
[SNNDecoderLayer L]  同上
  ↓
h ∈ R^{TK×batch×D}（连续值）
  ↓
[K 帧加权求和]  h → [0,1]^D（二进制位权 2^{-k}）
  ↓
[decode_proj → LateralInhibition → Embedding^T]  → logits
  ↓
输出
```

每个 DecoderLayer 内部的膜电位 V 和 spike 模式是该层**私有**的，不暴露给其他层。层间通过连续残差流 h 通信（v7.5），SNN 子层内部通过 spike 通信。这与 Transformer 的 Pre-norm 残差结构完全类比（h = h + sublayer(norm(h))）。

### 5.3 SNN Block 详细结构

> **v7 变更**：W^(V) 系列（β/α/V_th 的膜电位反馈路径）已移除，调制仅依赖 spike_in。详见 7.2 节。

单个 SNN Block 在时间步 t 的完整计算：

**输入**：$spike_{in} \in \{0,1\}^D$（来自输入 PLIFNode），$V[t{-}1] \in \mathbb{R}^{D \times N}$（本Block私有隐状态）

**阶段一：六条并行输入路径**（全部为 SNN 突触层：spike × W → 实数电流）

```
spike_in ∈ {0,1}^D
  │
  ├──→ [W_in · spike_in]                → I[t] ∈ R^{D×N}
  │     输入电流路径
  │
  ├──→ [W_β^(x) · spike_in + b_β] → σ → β(t) ∈ (0,1)^{D×N}
  │     β 调制路径（v7: 仅依赖 spike_in）
  │
  ├──→ [W_α^(x) · spike_in + b_α] → softplus → α(t) ∈ R_+^{D×N}
  │     α 调制路径（v7: 仅依赖 spike_in, 独立于β）
  │
  ├──→ [W_th^(x) · spike_in + b_th] → |·|+V_min → V_th(t) ∈ R_+^{D×N}
  │     V_th 调制路径（v7: 仅依赖 spike_in）
  │
  ├──→ [W_gate · spike_in] → sigmoid → gate ∈ (0,1)^D
  │     门控路径（只看当前输入，无状态，对应 Mamba Gate/SiLU 分支）
  │
  └──→ [W_skip · spike_in]              → I_skip ∈ R^D
        残差路径（输入直通）
```

六组 W 的输入全部是 spike（SNN 突触连接）。v7 中调制参数仅依赖当前输入 spike_in，使 parallel scan 成为可能。

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
  PLIFNode（D 维可学习 β_out, V_th_out）:
    V_out[t] = β_out · V_out[t-1] + (1 - β_out) · I_total
    spike_out = Θ(V_out - V_th_out)     ∈ {0,1}^D
```

**输出**：$spike_{out} \in \{0,1\}^D$（传给下一Block）

### 5.4 Block 内组件清单

> **v7 更新**：W^(V) 已移除，调制仅依赖 spike_in。输出神经元从 ParametricLIFNode（标量 β）更新为 PLIFNode（D 维可学习 β + V_th）。

| 组件 | 输入 | 输出 | 类型 | 对应 Mamba |
|---|---|---|---|---|
| $W_{in}$ | spike | 电流 $\mathbb{R}^{D \times N}$ | SNN 突触 | $\bar{B} \cdot x$ 写入 |
| $W_\beta^{(x)}$ | spike | β(t) ∈ (0,1)^{D×N} | SNN 突触 + sigmoid | Δ(x) → Ā 选择性 |
| $W_\alpha^{(x)}$ | spike | α(t) ∈ R_+^{D×N} | SNN 突触 + softplus | $\bar{B}$ 写入增益（独立于衰减） |
| $W_{th}^{(x)}$ | spike | V_th(t) ∈ R_+^{D×N} | SNN 突触 + abs | C(x) 读出控制 |
| $W_{gate}$ | spike | gate ∈ $(0,1)^D$（sigmoid） | SNN 突触 + sigmoid（无状态） | **Gate 分支**（Mamba 用 SiLU） |
| $W_{skip}$ | spike | I_skip ∈ $\mathbb{R}^D$ | SNN 突触 | 残差连接 |
| **隐状态空间** | I, β, α, V_th | spike s[t] | **SelectivePLIFNode**（动态参数） | **SSM**（$h = \bar{A}h + \bar{B}x$） |
| $W_{out}$ | spike | 电流 $\mathbb{R}^D$ | SNN 突触 | $C \cdot h$ 读出 |
| 输出神经元 | 电流 | spike | **PLIFNode**（D 维可学习 β, V_th） | 非线性输出 |

### 5.5 两种 SNN 神经元的分工

> **v7 更新**：隐状态神经元的调制仅依赖 spike_in（W^(V) 移除）。普通神经元从 ParametricLIFNode（标量 w）更新为自定义 PLIFNode（D 维可学习 w + v_th）。

| | 隐状态空间神经元（SelectivePLIFNode） | 信号转换神经元（PLIFNode） |
|---|---|---|
| 位于 | Block 内核心（D×N 个） | 层输入（D）、Block 输出（D）、FFN 内部（D\_ff×2 + D） |
| β | **动态**：由 spike_in 每步计算 | **可学习**：D 维向量，训练中更新 |
| α | **动态**：由 spike_in 每步计算 | 隐含在 (1-β) 系数中 |
| V_th | **动态**：由 spike_in 每步计算 | **可学习**：D 维向量，训练中更新 |
| 方程 | $V = \beta(t) \cdot V + \alpha(t) \cdot I$ | $V = \beta \cdot V + (1-\beta) \cdot x$ |
| 角色 | 选择性记忆（核心能力） | 信号转换（连续值→spike，电流→spike） |
| 对应 Mamba | SSM 模块 | SSM 以外的标准组件 |
| 代码 | `selective_plif.py` | `plif_node.py` |

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

> **v7 更新**：W^(V) 路径已移除。调制公式简化为仅依赖 spike_in。

```
spike_in {0,1}^D
  │
  │  [layer.Linear(D, D*N)]    ← spike × W = 实数电流（SNN突触操作）
  │  spike 输入，实数输出
  │
  ├──→ W_in:     {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流
  ├──→ W_β^(x):  {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流（β调制）
  ├──→ W_α^(x):  {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流（α调制）
  ├──→ W_th^(x): {0,1}^D → R^{D*N}         信号类型: 脉冲 → 电流（V_th调制）
  ├──→ W_gate:   {0,1}^D → R^D → sigmoid → (0,1)^D   脉冲 → 电流 → 门控值
  └──→ W_skip:   {0,1}^D → R^D             信号类型: 脉冲 → 电流（残差）

调制计算（v7: 仅依赖 spike_in）:
  β(t) = sigmoid(W_β^(x)·spike + b_β)                  → (0,1)^{D*N}
  α(t) = softplus(W_α^(x)·spike + b_α)                 → R_+^{D*N}
  V_th(t) = V_min + |W_th^(x)·spike + b_th|            → R_+^{D*N}

隐状态空间（SelectivePLIFNode）:
  V[t] = β(t)·V[t-1] + α(t)·I[t]    实数 × 实数 + 实数 × 实数 → 实数
  s[t] = Θ(V[t] - V_th(t))           实数 → {0,1}（surrogate gradient）
  V[t] -= V_th(t)·s[t]               soft reset（实数运算）

输出:
  W_out: {0,1}^{D*N} → R^D            信号类型: 脉冲 → 电流（SNN突触）
  I_out × gate + I_skip → R^D          实数 × 实数 + 实数 → 实数
  输出 PLIFNode: R^D → {0,1}^D         信号类型: 电流 → 脉冲
    V_out = β_out · V_out + (1 - β_out) · I_total（β_out, V_th_out 为 D 维可学习向量）
```

#### 5.8.3 W^(V) 的高效计算

> **v7 变更**：W^(V) 已移除。本节保留原始设计供参考。

~~$W^{(V)}$ 是 N×N 矩阵，D 个通道共享。对每个通道 d，需计算 $W^{(V)} \cdot V[t-1]_{d,:}$（N 维向量乘 N×N 矩阵）。~~

~~高效实现：利用 reshape 将 D 个通道的 N 维向量合并为一次矩阵乘法。~~

#### 5.8.4 首时间步处理

SpikingJelly 的 `BaseNode` 将膜电位 `v` 初始化为 float `0.`（通过 `register_memory('v', 0.)`）。首次 `single_step_forward` 时，`v_float_to_tensor(x)` 将其扩展为与输入同形的全零张量。

**SNNBlock 的特殊处理**：在首时间步，`hidden_neuron.v` 仍是 float `0.`（尚未被 `v_float_to_tensor` 扩展）。SNNBlock 需显式检查并初始化为全零张量。

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
        └── SelectivePLIFNode（自定义隐状态神经元，动态 β/α/V_th）

spikingjelly.activation_based.base.MemoryModule
  └── PLIFNode（自定义信号转换神经元，D 维可学习 β/V_th）

spikingjelly.activation_based.base.MemoryModule
  └── SNNBlock（自定义 Block，内含 SelectivePLIFNode + PLIFNode）

spikingjelly.activation_based.base.MemoryModule
  └── SNNFFN（自定义 FFN，内含 3 个 PLIFNode + AND 门）

spikingjelly.activation_based.base.MemoryModule
  └── SNNDecoderLayer（v7.5b: PLIFNode×2 + SNNBlock + SNNFFN + out_proj×2 + 残差连接）
```

**`functional.reset_net(net)` 兼容**：遍历 `net.modules()`，对所有有 `reset()` 方法的模块调用 `reset()`。我们的继承链保证：
- `SelectivePLIFNode` 继承 `BaseNode` → 有 `reset()` → `self.v` 被重置为 `0.`（float）
- `PLIFNode` 继承 `MemoryModule` → 有 `reset()` → `self.v` 被重置为 `0.`（float）
- `SNNBlock` 继承 `MemoryModule` → 有 `reset()` → 内部 PLIFNode 递归重置
- `SNNFFN` 继承 `MemoryModule` → 有 `reset()` → 内部 3 个 PLIFNode 递归重置
- `SNNDecoderLayer` 继承 `MemoryModule` → 有 `reset()` → 内部所有子模块递归重置
- 所有 `nn.Linear`、`LateralInhibition`（仅输出层） → 无状态，无 `reset()` → 不受影响

**`step_mode='s'` 专用**：我们只使用单步模式。`MemoryModule.forward(*args, **kwargs)` 透传所有参数到 `single_step_forward`，支持我们的非标准签名 `(x, beta, alpha, v_th)`。**不兼容 `step_mode='m'`**（`multi_step_forward` 只传一个参数）。

**`surrogate_function`**：使用 `surrogate.Sigmoid(alpha=4.0)`。前向：Heaviside 阶跃函数（输出 0/1）；反向：sigmoid surrogate gradient。

#### 5.8.6 完整可训练参数清单

> **v7 更新**：W^(V) 已移除。output_neuron 从标量 ParametricLIFNode 更新为 D 维 PLIFNode。

以 D=128, N=8 为例（单个 SNNBlock）：

| 参数 | PyTorch 类型 | 形状 | 参数量 | 初始化 |
|---|---|---|---|---|
| `W_in.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform + √(1−β²) 时间尺度缩放 |
| `W_beta_x.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform × 0.1（调制解耦） |
| `W_alpha_x.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform × 0.1（调制解耦） |
| `W_th_x.weight` | `layer.Linear` | (D\*N, D) = (1024, 128) | 131,072 | Kaiming uniform × 0.1（调制解耦） |
| `W_gate.weight` | `layer.Linear` | (D, D) = (128, 128) | 16,384 | Kaiming uniform |
| `W_skip.weight` | `layer.Linear` | (D, D) = (128, 128) | 16,384 | Kaiming uniform |
| `W_out.weight` | `layer.Linear` | (D, D\*N) = (128, 1024) | 131,072 | Kaiming uniform + 1/√p\_fire 发放率均衡 |
| `b_beta` | `nn.Parameter` | (D\*N,) = (1024,) | 1,024 | logit-spaced: $\ln(\beta_n / (1{-}\beta_n))$, $\beta \in$ linspace(0.80, 0.99, N) |
| `b_alpha` | `nn.Parameter` | (D\*N,) = (1024,) | 1,024 | 全部 0.5413（使 softplus ≈ 1.0） |
| `b_th` | `nn.Parameter` | (D\*N,) = (1024,) | 1,024 | σ\_V 校准: $\sigma_V \cdot \Phi^{-1}(1 - p_{fire}) - V_{th,min}$（见 5.8.10） |
| `output_neuron.w` | `nn.Parameter` | (D,) = (128,) | 128 | $-\ln(\tau_{init} - 1)$, $\tau_{init}=2.0$; D 维可学习 β |
| `output_neuron.v_th` | `nn.Parameter` | (D,) = (128,) | 128 | 初始值 0.3（第 0 层）或 0.05（深层） |
| **SNNBlock 总计** | | | **~691,457** | |

**v7.5 SNNDecoderLayer 额外参数**（每层）：

| 参数 | PyTorch 类型 | 形状 | 参数量 | 初始化 |
|---|---|---|---|---|
| `block_out_proj.weight` | `nn.Linear` | (D, D) | 16,384 | $\mathcal{N}(0, 0.02/\sqrt{2L})$（GPT-2 style） |
| `ffn_out_proj.weight` | `nn.Linear` | (D, D) | 16,384 | $\mathcal{N}(0, 0.02/\sqrt{2L})$ |
| `input_neuron1.w` | `nn.Parameter` | (D,) | 128 | PLIFNode, init_tau=2.0 |
| `input_neuron1.v_th` | `nn.Parameter` | (D,) | 128 | 0.5 |
| `input_neuron2.w` | `nn.Parameter` | (D,) | 128 | PLIFNode, init_tau=2.0 |
| `input_neuron2.v_th` | `nn.Parameter` | (D,) | 128 | 0.5 |

**注**：所有 `layer.Linear` 的 `bias=False`（无偏置）。偏置通过独立的 `nn.Parameter`（`b_beta`, `b_alpha`, `b_th`）实现，以便结构化初始化。

#### 5.8.7 非可训练超参数

> **v7 更新**：K 升级为 16，K_ref 同步更新，ε_V 已随 W^(V) 移除。

| 超参数 | 值 | 含义 |
|---|---|---|
| `V_th_min` | 0.1 | 动态阈值下限，确保 $V_{th}(t) > 0$ |
| `V_th_base` | 0.5 | V_th 偏置初始化的基础值 |
| `v_reset` | `None` | Soft reset 模式：$V -= V_{th} \cdot s$ |
| `surrogate_alpha` | 4.0 | Surrogate gradient 的锐度（Sigmoid 默认值） |
| `detach_reset` | `False` | Reset 操作保留在计算图中 |
| `step_mode` | `'s'` | 单步模式（每次调用处理一个 SNN 时间步） |
| `K` | **16**（v7 升级） | 每 token 的 SNN 时间步数（16-bit 精度） |
| `K_ref` | **16** | σ\_V 校准使用的参考时间步数（见 5.8.10） |
| `target_p_fire` | linspace(0.25, 0.08, N) | 各时间尺度的目标初始发放率 |
| `modulation_scale` | 0.1 | 调制路径权重缩放因子（解耦原则） |
| `v_threshold_out` | 0.3（第 0 层）/ 0.05（深层） | 输出 PLIFNode 的初始发放阈值（D 维可学习） |

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
K_ref = 16  # v7 升级: 与 K=16 同步
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

**直觉**：训练初期，β/α/V_th 几乎完全由偏置决定（调制路径 W^(x) 乘以 0.1 很小）。随训练推进，W^(x) 学到输入依赖的调制，偏置提供的多时间尺度结构逐渐被动态调制增强而非取代。

#### 5.8.9 代码文件结构

> **v7.5 更新**：新增 PLIFNode、SNNFFN、SNNDecoderLayer、LateralInhibition、parallel_scan。

```
atomic_ops/
  __init__.py              # 公开 API: SelectivePLIFNode, PLIFNode, LateralInhibition,
                           #           SNNBlock, SNNFFN, SNNDecoderLayer, parallel_scan
  selective_plif.py        # SelectivePLIFNode 类（继承 BaseNode，动态 β/α/V_th）
  plif_node.py             # PLIFNode 类（继承 MemoryModule，D 维可学习 β/V_th）
  lateral_inhibition.py    # LateralInhibition 类（侧抑制，Triton fused kernel）
  snn_block.py             # SNNBlock 类（6 路突触 + SelectivePLIF + PLIFNode 输出）
  snn_ffn.py               # SNNFFN 类（gate/up/down 投影 + 3 个 PLIFNode + AND 门）
  snn_decoder_layer.py     # SNNDecoderLayer 类（v7.5: 侧抑制 + PLIFNode + 子层 + 残差）
  parallel_scan.py         # Triton fused PLIF parallel scan kernel（v7.2-v7.4）
model.py                   # SNNLanguageModel（编码/解码 + L 层 SNNDecoderLayer）
train.py                   # 训练循环（surrogate gradient backprop, bf16 混合精度）
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

##### 5.8.10.5 ~~W^(V) 结构化初始化~~（v7 已移除）

> **v7 变更**：W^(V) 已移除以支持 parallel scan。本节保留原始设计供参考。

~~**旧方案**：`Uniform(±0.01)` — 随机且无结构。~~

~~**新方案**：零矩阵 + ε 对角线（ε=0.05）。~~

~~$$W^{(V)} = \varepsilon \cdot I_N$$~~

~~**设计理由**：对角线自监测，非对角线零（训练初期各时间尺度独立），ε=0.05 提供有信息的初始梯度。~~

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
| ~~反馈路径（W^(V)）~~ | ~~εI (ε=0.05)~~ | ~~v7 已移除~~ |

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
# 8. [v7 已移除] W^(V): εI (ε=0.05)         → 原自监测起点
# 9. Output PLIFNode: τ=2, v_threshold=0.3   → 快速响应 + 信号匹配
# 10. [v7.5] LateralInhibition.gain: 全 1    → 仅输出层侧抑制增益
# 11. [v7.5] out_proj: N(0, 0.02/√(2L))     → 残差输出缩放
# 12. [v7.5] input_neuron PLIFNode: τ=2, v_th=0.5 → 输入 spike 转换
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
- ~~$W^{(V)}$ 为 N×N（D通道共享）：N 个时间尺度的神经元之间有交互~~（**v7 已移除**，跨时间尺度协调改由层间 spike 传播实现）
- β 偏置初始化覆盖 0.80~0.99 多时间尺度（类比 HiPPO）
- **选择性**：β(t), α(t), V_th(t) 由 spike_in 动态计算（v7 移除 $V[t{-}1]$ 依赖以支持 parallel scan）
- β、α 和 V_th 不是训练后固定的参数，而是每步由输入动态决定的值（对应 Mamba 的选择性思想）。真正的可训练参数是调制网络的权重矩阵
- β 与 α 解耦：衰减和写入由独立网络控制，支持 Mamba 的 Δ 耦合无法实现的"高保留+高写入"组合

**逐机制对应**：

| 机制 | Mamba | SNN | 差异 |
|---|---|---|---|
| 衰减控制 | $\bar{A}[t] = \exp(\Delta(x_t) \cdot A)$ | $\beta(t) = \sigma(W^{(x)} \cdot spike_{in} + b)$（v7: 仅看输入） | v7 与 Mamba 对齐（仅看输入）；spike/reset 提供隐式状态反馈 |
| 写入 | $\bar{B}[t] \cdot x_t$ | $\alpha(t) \cdot I[t] = \alpha(t) \cdot W_{in} \cdot spike_{in}$ | α 独立于 β；Mamba 的 $\bar{B}$ 经 Δ 与 $\bar{A}$ 耦合 |
| 遗忘机制 | 单一 $\bar{A}$（乘性衰减） | β衰减（乘性，每步）+ soft reset（减性，条件触发） | 我们将遗忘拆分为两个协同机制 |
| 读出 | $C(x_t) \cdot h[t]$，线性，全体参与 | $\Theta(V[t] - V_{th}(t))$，二值，稀疏选择 | 非线性门控 |
| 输出→状态反馈 | 无（读出不修改 h） | 有（soft reset 修改 V） | 双向耦合 |
| 状态维度间交互 | 无（A 对角） | ~~有（$W^{(V)}$ 是 N×N）~~（v7 移除） | v7: 各维度独立，层间 spike 传播替代 |
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

**关键差异**：v7 移除 $W^{(V)}$ 后，调制公式仅依赖 spike_in，与 Mamba 的 $\Delta(x_t)$ 对齐。但 SNN 通过 spike/reset 的非线性机制提供**隐式**状态反馈——发放后 $V -= V_{th}$ 直接修改状态，相当于物理级别的"自监测"。这比 $W^{(V)}$ 的参数化反馈更原始但也更强：不需要学习就天然具备。

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

## 7. 并行计算理论：从串行递推到 Parallel Scan

> **架构演进 v7**：移除 $W^{(V)}$ 电压反馈矩阵，使 SNN 递推满足 parallel scan 的前提条件。K 从 8 增加到 16（16-bit 精度）。
> 本节包含完整的数学推导，证明并行化的可行性和正确性。所有分析基于 soft reset（$V \leftarrow V - V_{th} \cdot s$，不归零）。
> **核心思想**：将 SNN 递推分解为"线性轨迹 + spike 修正"，线性部分用 parallel scan 在 $O(\log(TK))$ 深度内求解，spike 修正用不动点迭代（2-3 次）收敛。

### 7.1 串行瓶颈的本质

当前实现的计算结构为三重串行循环：

```
for n = 1 to T:          # T 个 token（串行，因 V 跨 token 连续）
  for k = 1 to K:        # 每 token K 个 SNN 帧（串行，因 V[k] 依赖 V[k-1]）
    for l = 1 to L:      # L 个 DecoderLayer（串行，因 layer l 输入来自 layer l-1 输出）
      layer_l.forward(spike)
```

**总串行深度**：

$$N_{serial} = T \times K \times L = 512 \times 16 \times 20 = 163{,}840$$

每步的工作量为 6 条输入路径的矩阵乘法 + 隐状态更新 + 输出投影，矩阵维度为 (batch, D) $\times$ (D, D$\cdot$N)。这是**小矩阵乘法**，GPU 利用率极低。

**速度瓶颈的本质**：不是每步计算量大，而是串行步数太多。$163{,}840$ 个小操作排成链，GPU 的数千个 CUDA core 只有一小部分在工作。

**对比 Mamba**：Mamba 将 $T$ 个 token 的线性递推用 parallel scan 压缩到 $O(\log T)$ 深度，每步是大矩阵运算（充分利用 GPU）。层间仍串行，总深度 $O(L \cdot \log T)$。

**目标**：将三重串行 $T \times K \times L$ 转化为 $L \times O(\log(TK))$，即**层间串行、层内并行**。

### 7.2 并行化的前提：W^(V) 依赖分析与移除

#### 7.2.1 W^(V) 造成的串行依赖

当前架构中，$\beta$、$\alpha$、$V_{th}$ 的计算引用了 $V[k{-}1]$：

$$\beta[k] = \sigma\!\left(W_\beta^{(x)} \cdot \text{spike}_{in}[k] + W_\beta^{(V)} \cdot V[k{-}1] + b_\beta\right)$$

$$\alpha[k] = \text{softplus}\!\left(W_\alpha^{(x)} \cdot \text{spike}_{in}[k] + W_\alpha^{(V)} \cdot V[k{-}1] + b_\alpha\right)$$

$$V_{th}[k] = V_{min} + \left|W_{th}^{(x)} \cdot \text{spike}_{in}[k] + W_{th}^{(V)} \cdot V[k{-}1] + b_{th}\right|$$

**依赖链**：

$$V[0] \xrightarrow{\text{计算}} \beta[1], \alpha[1], V_{th}[1] \xrightarrow{\text{计算}} V[1] \xrightarrow{\text{计算}} \beta[2], \alpha[2], V_{th}[2] \xrightarrow{\text{计算}} V[2] \xrightarrow{} \cdots$$

核心问题：**转移参数（$\beta$，即"矩阵 A"）依赖于状态 $V$**。

在 parallel scan 框架中，线性递推 $V[k] = A[k] \cdot V[k{-}1] + B[k]$ 要求 $A[k]$ 和 $B[k]$ 在扫描前全部已知。若 $A[k]$ 依赖 $V[k{-}1]$，则 $A[k]$ 本身需要串行计算——parallel scan 的前提被破坏。

#### 7.2.2 与 Mamba 的关键对齐

Mamba 的状态方程：

$$h[t] = \bar{A}(x_t) \cdot h[t{-}1] + \bar{B}(x_t) \cdot x_t$$

**Mamba 的设计选择**：$\bar{A}$ 和 $\bar{B}$ **仅依赖输入 $x_t$，不依赖状态 $h[t{-}1]$**。这是刻意为之——Gu & Dao (2023) 明确指出这是 parallel scan 的必要条件。

给定输入序列 $(x_1, \ldots, x_T)$，Mamba 可以：
1. **并行计算**所有 $\bar{A}[t]$, $\bar{B}[t]$（$O(1)$ 深度）
2. **Parallel scan** 求解所有 $h[t]$（$O(\log T)$ 深度）

我们的 $W^{(V)}$ 打破了第 1 步：$\beta[k]$ 依赖 $V[k{-}1]$，无法与 $V$ 的计算解耦。

#### 7.2.3 移除 W^(V) 的决策

**移除后的方程**：

$$\beta[k] = \sigma\!\left(W_\beta^{(x)} \cdot \text{spike}_{in}[k] + b_\beta\right)$$

$$\alpha[k] = \text{softplus}\!\left(W_\alpha^{(x)} \cdot \text{spike}_{in}[k] + b_\alpha\right)$$

$$V_{th}[k] = V_{min} + \left|W_{th}^{(x)} \cdot \text{spike}_{in}[k] + b_{th}\right|$$

现在 $\beta[k]$、$\alpha[k]$、$V_{th}[k]$ **仅依赖 $\text{spike}_{in}[k]$**（来自编码器或上一层的输出）。给定所有输入 spike，全部调制参数可**一次性并行计算**。

#### 7.2.4 保留与失去的能力对照

| 能力 | 移除前 | 移除后 | 状态 |
|---|---|---|---|
| 输入依赖的选择性 | $W^{(x)} \cdot \text{spike}_{in}$ + $W^{(V)} \cdot V$ | $W^{(x)} \cdot \text{spike}_{in}$ | **保留**（核心选择性机制） |
| 多时间尺度 $\beta$ 结构 | $b_\beta$ logit-spaced [0.80, 0.99] | 同左 | **保留** |
| Spike/reset 非线性 | $\Theta(V - V_{th})$, $V -= V_{th} \cdot s$ | 同左 | **保留** |
| 跨层信息流 | spike 传递 | 同左 | **保留** |
| 自监测（V→自身β/α/V_th） | $W^{(V)} \cdot V$ 直接反馈 | spike/reset 隐式反馈 | **隐式保留** |
| N×N 跨时间尺度协调 | $W^{(V)}$ 非对角元素 | 跨层 spike 传播 | **间接保留** |

**自监测的隐式替代**：Spike/reset 机制本身就是 V-dependent 的状态调节——当 $V$ 超过 $V_{th}$ 时发放 spike 并减去 $V_{th}$，膜电位被动态压缩。这是比 $W^{(V)}$ 更原始的"自监测"：不通过参数调制，而通过物理机制直接作用于状态。

**跨时间尺度协调的替代路径**：$L = 20$ 层的深度网络中，每层的 spike 输出是该层所有时间尺度神经元的综合表达。下一层通过 $W^{(x)} \cdot \text{spike}_{in}$ 接收这些信息，实现层间的时间尺度协调（而非层内的 $W^{(V)}$ 直接反馈）。

**Mamba 的实证支持**：Mamba 在完全没有状态反馈的情况下（$\bar{A}(x_t)$ 不依赖 $h$），在多项 NLP 基准上取得与 Transformer 可比的性能。这证明输入依赖的选择性本身已经是足够强大的建模机制。

#### 7.2.5 参数变更明细

**移除的参数**（每个 SNNBlock）：
- $W_\beta^{(V)}$: `nn.Linear(N, N, bias=False)` — $N^2 = 64$ 个参数
- $W_\alpha^{(V)}$: `nn.Linear(N, N, bias=False)` — $N^2 = 64$ 个参数
- $W_{th}^{(V)}$: `nn.Linear(N, N, bias=False)` — $N^2 = 64$ 个参数
- **总计**：每 Block $3N^2 = 192$ 个参数，$L = 20$ 层共 $3{,}840$ 个参数

相对于总参数量 $\sim 1.1B$，移除的参数占比 $< 0.0004\%$——参数量几乎不变，但计算结构发生质变。

**保留的参数**（不变）：
- $W_\beta^{(x)}$, $W_\alpha^{(x)}$, $W_{th}^{(x)}$: 输入依赖的调制网络
- $b_\beta$, $b_\alpha$, $b_{th}$: 结构化偏置
- $W_{in}$, $W_{gate}$, $W_{skip}$, $W_{out}$: 信号路径
- SelectivePLIFNode, PLIFNode: 神经元动力学

### 7.3 Soft Reset 递推的精确分解

以下推导针对单个神经元（省略通道下标 $d$ 和时间尺度下标 $n$），适用于全局时间步 $\tau = 1, 2, \ldots, T \cdot K$。

#### 7.3.1 基本递推方程

设初始状态 $V_{post}[0]$ 来自前一次 `reset_net()` 后的值（通常为 0），或来自上一个序列的末态（若不重置）。

**Phase 1: Charge（充电）**：

$$V_{pre}[\tau] = \beta[\tau] \cdot V_{post}[\tau{-}1] + \alpha[\tau] \cdot I[\tau]$$

**Phase 2: Fire（发放）**：

$$s[\tau] = \Theta\!\left(V_{pre}[\tau] - V_{th}[\tau]\right) \in \{0, 1\}$$

**Phase 3: Soft Reset**：

$$V_{post}[\tau] = V_{pre}[\tau] - V_{th}[\tau] \cdot s[\tau]$$

合并为**单一递推方程**：

$$\boxed{V_{post}[\tau] = \beta[\tau] \cdot V_{post}[\tau{-}1] + \alpha[\tau] \cdot I[\tau] - V_{th}[\tau] \cdot s[\tau]}$$

其中 $\beta[\tau], \alpha[\tau], V_{th}[\tau], I[\tau]$ 在移除 $W^{(V)}$ 后仅依赖 $\text{spike}_{in}[\tau]$——全部可预先计算。

唯一的自引用：$s[\tau]$ 通过 $V_{pre}[\tau]$ 依赖 $V_{post}[\tau{-}1]$。

#### 7.3.2 线性-非线性分解

**定义线性轨迹** $V_L[\tau]$（假设从不发放的膜电位演化）：

$$V_L[\tau] = \beta[\tau] \cdot V_L[\tau{-}1] + \alpha[\tau] \cdot I[\tau], \quad V_L[0] = V_{post}[0]$$

**定义 spike 修正量** $\Delta S[\tau]$：

$$\Delta S[\tau] \triangleq V_L[\tau] - V_{post}[\tau]$$

**$\Delta S$ 的递推推导**：

$$\Delta S[\tau] = V_L[\tau] - V_{post}[\tau]$$

$$= \bigl(\beta[\tau] \cdot V_L[\tau{-}1] + \alpha[\tau] \cdot I[\tau]\bigr) - \bigl(\beta[\tau] \cdot V_{post}[\tau{-}1] + \alpha[\tau] \cdot I[\tau] - V_{th}[\tau] \cdot s[\tau]\bigr)$$

$$= \beta[\tau] \cdot \bigl(V_L[\tau{-}1] - V_{post}[\tau{-}1]\bigr) + V_{th}[\tau] \cdot s[\tau]$$

$$\boxed{\Delta S[\tau] = \beta[\tau] \cdot \Delta S[\tau{-}1] + V_{th}[\tau] \cdot s[\tau], \quad \Delta S[0] = 0}$$

**关键性质**：$\Delta S$ 的递推与 $V_L$ 具有**相同的 $\beta$ 系数**——它也是一个线性递推（给定 spike 模式 $s$）。

#### 7.3.3 发放条件的等价表述

将 $V_{pre}[\tau]$ 用 $V_L$ 和 $\Delta S$ 表示：

$$V_{pre}[\tau] = \beta[\tau] \cdot V_{post}[\tau{-}1] + \alpha[\tau] \cdot I[\tau]$$

$$= \beta[\tau] \cdot \bigl(V_L[\tau{-}1] - \Delta S[\tau{-}1]\bigr) + \alpha[\tau] \cdot I[\tau]$$

$$= V_L[\tau] - \beta[\tau] \cdot \Delta S[\tau{-}1]$$

因此发放条件变为：

$$\boxed{s[\tau] = \Theta\!\left(V_L[\tau] - \beta[\tau] \cdot \Delta S[\tau{-}1] - V_{th}[\tau]\right)}$$

**物理含义**：神经元是否发放取决于"无 spike 轨迹"$V_L[\tau]$ 减去"历史 spike 的累积修正"$\beta[\tau] \cdot \Delta S[\tau{-}1]$ 是否超过阈值。

#### 7.3.4 显式展开与传播系数

定义**传播系数**（从步 $j$ 到步 $k$ 的累积衰减）：

$$P(j, k) \triangleq \prod_{m=j+1}^{k} \beta[m], \quad P(k, k) = 1, \quad P(j, k) = 0\ \text{if}\ j > k$$

**$V_L$ 的显式解**：

$$V_L[k] = P(0, k) \cdot V_{post}[0] + \sum_{j=1}^{k} P(j, k) \cdot \alpha[j] \cdot I[j]$$

**$\Delta S$ 的显式解**：

$$\Delta S[k] = \sum_{j=1}^{k} P(j, k) \cdot V_{th}[j] \cdot s[j]$$

**完整膜电位**：

$$V_{post}[k] = V_L[k] - \Delta S[k] = P(0,k) \cdot V_{post}[0] + \sum_{j=1}^{k} P(j,k) \cdot \bigl[\alpha[j] \cdot I[j] - V_{th}[j] \cdot s[j]\bigr]$$

**物理解读**：$V_{post}[k]$ 是初始状态 $V_{post}[0]$ 经累积衰减后的残余，加上每个历史步 $j$ 的"净输入"$\alpha[j] \cdot I[j] - V_{th}[j] \cdot s[j]$（输入电流减去 spike reset 扣除）经衰减后的贡献。传播系数 $P(j,k) = \prod_{m=j+1}^{k} \beta[m]$ 是信号从 $j$ 传播到 $k$ 的保留率。

### 7.4 线性递推的 Parallel Scan

#### 7.4.1 仿射映射表示

$V_L$ 的递推 $V_L[\tau] = \beta[\tau] \cdot V_L[\tau{-}1] + u[\tau]$（其中 $u[\tau] = \alpha[\tau] \cdot I[\tau]$）是**仿射映射**（affine map）的复合：

$$f_\tau(x) = \beta[\tau] \cdot x + u[\tau]$$

前缀复合：

$$F_k = f_k \circ f_{k-1} \circ \cdots \circ f_1$$

$$V_L[k] = F_k(V_{post}[0])$$

两个仿射映射的复合满足结合律：

$$(a_2, b_2) \circ (a_1, b_1) = (a_2 \cdot a_1,\ a_2 \cdot b_1 + b_2)$$

验证：$f_2(f_1(x)) = a_2(a_1 x + b_1) + b_2 = a_2 a_1 x + a_2 b_1 + b_2$。 $\checkmark$

等价地，使用增广向量的 2$\times$2 矩阵表示：

$$\begin{pmatrix} V_L[k] \\ 1 \end{pmatrix} = \underbrace{\begin{pmatrix} \beta[k] & u[k] \\ 0 & 1 \end{pmatrix}}_{\mathbf{M}[k]} \begin{pmatrix} V_L[k{-}1] \\ 1 \end{pmatrix}$$

前缀积 $\mathbf{P}[k] = \mathbf{M}[k] \cdot \mathbf{M}[k{-}1] \cdots \mathbf{M}[1]$ 给出：

$$\begin{pmatrix} V_L[k] \\ 1 \end{pmatrix} = \mathbf{P}[k] \cdot \begin{pmatrix} V_{post}[0] \\ 1 \end{pmatrix}$$

记 $\mathbf{P}[k] = \begin{pmatrix} A_k & B_k \\ 0 & 1 \end{pmatrix}$，则 $V_L[k] = A_k \cdot V_{post}[0] + B_k$。

#### 7.4.2 Hillis-Steele 并行扫描算法

计算全部前缀复合 $(A_1, B_1), (A_2, B_2), \ldots, (A_{TK}, B_{TK})$ 的经典 **Hillis-Steele scan**（也称 Kogge-Stone scan）：

```
输入: 仿射映射序列 (a[1], b[1]), ..., (a[K], b[K])
      其中 a[τ] = β[τ], b[τ] = α[τ] · I[τ]

初始化:
  A[τ] ← a[τ],  B[τ] ← b[τ]    ∀τ

for d = 0, 1, ..., ⌈log₂ K⌉ - 1:
  offset = 2^d
  for τ = offset+1, ..., K  (全部并行):
    A[τ] ← A[τ] · A[τ - offset]        // 旧值
    B[τ] ← A[τ] · B[τ - offset] + B[τ]  // 旧值

输出: A[τ], B[τ] 为步 1 到 τ 的前缀复合
      V_L[τ] = A[τ] · V_{post}[0] + B[τ]
```

**注意**：每轮迭代中，右侧的 $A[\tau]$, $B[\tau]$, $A[\tau - \text{offset}]$, $B[\tau - \text{offset}]$ 使用**上一轮的旧值**（Hillis-Steele 特性）。

**正确性证明**（归纳法）：

记 $F_k^{(d)}$ 为第 $d$ 轮后 $(A[k], B[k])$ 所表示的仿射映射。

- **初始**（$d = 0$ 之前）：$F_k^{(-1)} = f_k$，即原始的单步映射。

- **第 $d$ 轮后**：$F_k^{(d)}$ 表示从 $\max(1, k - 2^{d+1} + 1)$ 到 $k$ 的连续复合。

  - 当 $k \geq 2^d + 1$（执行合并）：
    $$F_k^{(d)} = F_k^{(d-1)} \circ F_{k-2^d}^{(d-1)}$$
    $F_k^{(d-1)}$ 覆盖长度 $2^d$ 的区间 $[k-2^d+1, k]$，$F_{k-2^d}^{(d-1)}$ 覆盖 $[k-2^{d+1}+1, k-2^d]$。
    复合后覆盖长度 $2^{d+1}$ 的区间 $[k-2^{d+1}+1, k]$。 $\checkmark$

- **$d = \lceil \log_2 K \rceil - 1$ 后**：$F_k^{(d)}$ 覆盖 $[1, k]$，即完整前缀复合。 $\square$

**复杂度**：
- 并行深度：$\lceil \log_2(TK) \rceil$ 轮。$T = 512, K = 16$ 时为 $\lceil \log_2 8192 \rceil = 13$ 轮。
- 每轮工作量：$O(TK)$ 次仿射映射复合（每次 2 次乘法 + 1 次加法）。
- 总工作量：$O(TK \cdot \log(TK))$，比串行的 $O(TK)$ 多 $\log$ 倍（用额外工作换取并行深度的指数缩减）。

#### 7.4.3 GPU 上的向量化

在实际计算中，$\beta[\tau]$ 和 $u[\tau]$ 是形状为 $(\text{batch}, D \times N)$ 的张量。$D \times N$ 个神经元各自独立进行仿射递推。

parallel scan 的每轮迭代是纯**逐元素张量运算**（element-wise multiply, add, slice），形状为 $(TK, \text{batch}, D \times N)$。

- $TK = 8192$, $\text{batch} = 16$, $D \times N = 8192$
- 每轮迭代处理 $8192 \times 16 \times 8192 \approx 10^9$ 个元素
- 这是**大规模并行张量运算**，GPU 利用率极高

对比原始串行实现：每步处理 $16 \times 8192 \approx 10^5$ 个元素，GPU 严重空闲。

#### 7.4.4 $\Delta S$ 的 Parallel Scan

给定 spike 模式 $s[\tau]$（见 7.5 节），$\Delta S$ 的递推：

$$\Delta S[\tau] = \beta[\tau] \cdot \Delta S[\tau{-}1] + V_{th}[\tau] \cdot s[\tau], \quad \Delta S[0] = 0$$

这与 $V_L$ 的递推**结构相同**：仿射映射 $\Delta S[\tau] = \beta[\tau] \cdot \Delta S[\tau{-}1] + c[\tau]$，其中 $c[\tau] = V_{th}[\tau] \cdot s[\tau]$。

因此用相同的 parallel scan 算法求解，$O(\log(TK))$ 深度。

### 7.5 Spike 模式的确定：不动点迭代

#### 7.5.1 问题形式化

已知所有 $V_L[\tau]$（由 7.4 节的 parallel scan 计算），需求解**联立方程**：

$$s[\tau] = \Theta\!\left(V_L[\tau] - \beta[\tau] \cdot \Delta S[\tau{-}1] - V_{th}[\tau]\right), \quad \tau = 1, \ldots, TK$$

$$\Delta S[\tau] = \beta[\tau] \cdot \Delta S[\tau{-}1] + V_{th}[\tau] \cdot s[\tau], \quad \Delta S[0] = 0$$

$s$ 和 $\Delta S$ 相互依赖：$s$ 决定 $\Delta S$（哪些步有 reset 修正），$\Delta S$ 决定 $s$（累积修正后哪些步仍超阈值）。

**定义映射** $T: \{0,1\}^{TK} \to \{0,1\}^{TK}$：

给定 spike 模式 $\mathbf{s} = (s[1], \ldots, s[TK])$：
1. 由 $\mathbf{s}$ 计算 $\Delta S[\tau]$（线性递推，可用 parallel scan）
2. 返回 $\mathbf{s}' = T(\mathbf{s})$，其中 $s'[\tau] = \Theta(V_L[\tau] - \beta[\tau] \cdot \Delta S[\tau{-}1] - V_{th}[\tau])$

**不动点**：$\mathbf{s}^* = T(\mathbf{s}^*)$ 即为物理上正确的 spike 模式。

#### 7.5.2 反单调性定理

**定理 1**（反单调性）：映射 $T$ 是**反单调的**（anti-monotone）：

$$\mathbf{s} \geq \mathbf{s}' \quad \Longrightarrow \quad T(\mathbf{s}) \leq T(\mathbf{s}')$$

其中 $\geq$ 为逐分量偏序。

**证明**：

设 $\mathbf{s} \geq \mathbf{s}'$，即 $s[\tau] \geq s'[\tau]$ 对所有 $\tau$。

**Step 1**：证明 $\Delta S[\tau] \geq \Delta S'[\tau]$ 对所有 $\tau$。

归纳法。基例：$\Delta S[0] = \Delta S'[0] = 0$。

归纳步：设 $\Delta S[\tau{-}1] \geq \Delta S'[\tau{-}1]$。

$$\Delta S[\tau] = \underbrace{\beta[\tau]}_{> 0} \cdot \underbrace{\Delta S[\tau{-}1]}_{\geq \Delta S'[\tau{-}1]} + \underbrace{V_{th}[\tau]}_{> 0} \cdot \underbrace{s[\tau]}_{\geq s'[\tau]}$$

$$\geq \beta[\tau] \cdot \Delta S'[\tau{-}1] + V_{th}[\tau] \cdot s'[\tau] = \Delta S'[\tau] \quad \checkmark$$

**Step 2**：证明 $T(\mathbf{s})[\tau] \leq T(\mathbf{s}')[\tau]$。

$$T(\mathbf{s})[\tau] = \Theta\!\left(V_L[\tau] - \beta[\tau] \cdot \underbrace{\Delta S[\tau{-}1]}_{\geq \Delta S'[\tau{-}1]} - V_{th}[\tau]\right)$$

$$\leq \Theta\!\left(V_L[\tau] - \beta[\tau] \cdot \Delta S'[\tau{-}1] - V_{th}[\tau]\right) = T(\mathbf{s}')[\tau] \quad \checkmark$$

（$\Theta$ 是单调非递减函数，参数减小则输出不增。）$\square$

**推论 1**：$T^2 = T \circ T$ 是**单调的**：$\mathbf{s} \geq \mathbf{s}' \Rightarrow T^2(\mathbf{s}) \geq T^2(\mathbf{s}')$。

（反单调两次 = 单调。）

#### 7.5.3 收敛性定理

**定理 2**（交替有界收敛）：从 $\mathbf{s}^{(0)} = \mathbf{0}$ 出发的迭代序列 $\mathbf{s}^{(i+1)} = T(\mathbf{s}^{(i)})$ 满足：

$$\mathbf{s}^{(0)} \leq \mathbf{s}^{(2)} \leq \mathbf{s}^{(4)} \leq \cdots \leq \cdots \leq \mathbf{s}^{(5)} \leq \mathbf{s}^{(3)} \leq \mathbf{s}^{(1)}$$

即**偶数次迭代单调递增，奇数次迭代单调递减，偶数列 $\leq$ 奇数列**。

在有限格 $\{0,1\}^{TK}$ 上，两个单调有界序列均在有限步内收敛。

**证明**：

**(a)** $\mathbf{s}^{(0)} \leq \mathbf{s}^{(1)}$：$\mathbf{s}^{(0)} = \mathbf{0}$，$\mathbf{s}^{(1)} = T(\mathbf{0}) \in \{0,1\}^{TK} \geq \mathbf{0}$。 $\checkmark$

**(b)** $\mathbf{s}^{(1)} \geq \mathbf{s}^{(2)}$：由反单调性，$\mathbf{s}^{(0)} \leq \mathbf{s}^{(1)} \Rightarrow T(\mathbf{s}^{(0)}) \geq T(\mathbf{s}^{(1)})$，即 $\mathbf{s}^{(1)} \geq \mathbf{s}^{(2)}$。 $\checkmark$

**(c)** $\mathbf{s}^{(2)} \leq \mathbf{s}^{(3)}$：由反单调性，$\mathbf{s}^{(1)} \geq \mathbf{s}^{(2)} \Rightarrow T(\mathbf{s}^{(1)}) \leq T(\mathbf{s}^{(2)})$，即 $\mathbf{s}^{(2)} \leq \mathbf{s}^{(3)}$。 $\checkmark$

**(d)** $\mathbf{s}^{(3)} \leq \mathbf{s}^{(1)}$：由反单调性，$\mathbf{s}^{(2)} \geq \mathbf{s}^{(0)} \Rightarrow T(\mathbf{s}^{(2)}) \leq T(\mathbf{s}^{(0)})$，即 $\mathbf{s}^{(3)} \leq \mathbf{s}^{(1)}$。 $\checkmark$

由 $T^2$ 单调性，归纳可得完整不等式链。在 $\{0,1\}^{TK}$（有限集）上，单调有界序列必在有限步收敛。 $\square$

**定理 3**（不动点存在性）：若偶数序列极限 $\underline{\mathbf{s}}$ 等于奇数序列极限 $\overline{\mathbf{s}}$，则 $\underline{\mathbf{s}} = \overline{\mathbf{s}}$ 是 $T$ 的不动点，即唯一物理正确的 spike 模式。

#### 7.5.4 实用快速收敛

**定理 4**（稀疏 spike 的快速收敛）：当 spike 发放率 $p_{fire} \ll 1$ 时，迭代通常在 $I = 2$ 次内收敛。

**直觉论证**：

- **第 0 次**：$\mathbf{s}^{(0)} = \mathbf{0}$（无 spike）
- **第 1 次**：$\mathbf{s}^{(1)} = \Theta(V_L - V_{th})$——不考虑 reset 修正的"最大 spike 模式"
- **第 2 次**：$\mathbf{s}^{(2)} = T(\mathbf{s}^{(1)})$——考虑了 $\mathbf{s}^{(1)}$ 的 reset 修正后的模式

$\mathbf{s}^{(1)}$ 和 $\mathbf{s}^{(2)}$ 的差异仅发生在"边界区域"——$V_L[\tau]$ 接近 $V_{th}[\tau]$ 的位置。对于大多数 $\tau$：
- 若 $V_L[\tau] \gg V_{th}[\tau]$：即使有 reset 修正，仍然超阈值。$s^{(1)}[\tau] = s^{(2)}[\tau] = 1$。
- 若 $V_L[\tau] \ll V_{th}[\tau]$：即使无 reset 修正，仍然亚阈值。$s^{(1)}[\tau] = s^{(2)}[\tau] = 0$。

只有 $|V_L[\tau] - V_{th}[\tau]| < \beta[\tau] \cdot \Delta S[\tau{-}1]$ 的位置可能发生变化。当 $p_{fire}$ 低时，$\Delta S$ 的幅度小（很少有 spike 贡献 reset），边界区域极窄。

**实验观测预期**：初始发放率 $p_{fire} \in [8\%, 25\%]$，边界区域宽度约 $p_{fire}^2$（两步内先发后修正的概率），$\mathbf{s}^{(2)} \approx \mathbf{s}^{(1)}$ 的概率 $> 95\%$。

**实用算法**：

```
输入: V_L[1..TK], β[1..TK], V_th[1..TK]
参数: max_iter = 3

s ← Θ(V_L - V_th)                           // 迭代 1: 无修正的初始 spike 模式

for i = 2, ..., max_iter:
  ΔS ← parallel_scan(β, V_th · s, 0)        // O(log TK) 深度
  ΔS_prev ← [0, ΔS[1:TK-1]]                 // 位移一步
  s_new ← Θ(V_L - β · ΔS_prev - V_th)       // O(1) 深度
  if s_new == s:
    break                                     // 收敛
  s ← s_new

输出: s（确定的 spike 模式）
```

**每次迭代的并行深度**：$O(\log(TK))$（来自 parallel scan）+ $O(1)$（逐元素判定）。
**总并行深度**：$O(I \cdot \log(TK))$，其中 $I \leq 3$。

### 7.6 Token 级折叠与有效 SSM

#### 7.6.1 K 帧折叠为 token 级仿射映射

一旦 spike 模式 $s[\tau]$ 确定（7.5 节），实际的 $V_{post}$ 递推变为：

$$V_{post}[\tau] = \beta[\tau] \cdot V_{post}[\tau{-}1] + u_{eff}[\tau]$$

其中**有效输入**：

$$u_{eff}[\tau] = \alpha[\tau] \cdot I[\tau] - V_{th}[\tau] \cdot s[\tau]$$

这是标准的仿射递推。将 token $n$ 的 $K$ 步折叠（$\tau$ 从 $(n{-}1)K{+}1$ 到 $nK$）：

$$V_{post}[nK] = \underbrace{\left(\prod_{k=1}^{K} \beta[(n{-}1)K{+}k]\right)}_{\bar{A}_{eff}[n]} \cdot V_{post}[(n{-}1)K] + \underbrace{\sum_{k=1}^{K} \left(\prod_{j=k+1}^{K} \beta[(n{-}1)K{+}j]\right) \cdot u_{eff}[(n{-}1)K{+}k]}_{\bar{B}_{eff}[n]}$$

简记：

$$\boxed{V_{post}[nK] = \bar{A}_{eff}[n] \cdot V_{post}[(n{-}1)K] + \bar{B}_{eff}[n]}$$

**注意**：与 5.7.8 节的线性近似不同，这里的折叠是**精确的**——因为 spike 模式已经确定，$u_{eff}[\tau]$ 是已知量，不含未知变量。

#### 7.6.2 与 Mamba 的精确对应

$$\text{Mamba}: \quad h[n] = \bar{A}(x_n) \cdot h[n{-}1] + \bar{B}(x_n) \cdot x_n$$

$$\text{SNN}: \quad V[nK] = \bar{A}_{eff}[n] \cdot V[(n{-}1)K] + \bar{B}_{eff}[n]$$

两者在 token 级别具有**完全相同的数学结构**——仿射递推。区别在于 $\bar{A}_{eff}$ 和 $\bar{B}_{eff}$ 的生成方式：

| | Mamba | SNN (v7) |
|---|---|---|
| $\bar{A}$ 来源 | $\exp(\Delta(x_n) \cdot A)$ | $\prod_{k=1}^{K} \beta[\text{spike}_{in}[n,k]]$（K 个衰减因子之积） |
| $\bar{B}$ 来源 | $\Delta(x_n) \cdot B(x_n) \cdot x_n$ | $\sum_{k=1}^{K} P(k,K) \cdot (\alpha \cdot I - V_{th} \cdot s)$（K 步加权和） |
| 非线性 | 无 | spike + soft reset 在 K 步内作用（$s$ 出现在 $\bar{B}_{eff}$ 中） |
| 计算粒度 | 1次/token | K 次/token（更精细的时间步控制） |

#### 7.6.3 Token 级 Parallel Scan

Token 级递推 $V[nK] = \bar{A}_{eff}[n] \cdot V[(n{-}1)K] + \bar{B}_{eff}[n]$ 是仿射递推——再次使用 7.4 节的 parallel scan 算法。

**并行深度**：$\lceil \log_2 T \rceil = \lceil \log_2 512 \rceil = 9$ 轮。

**但实际上不需要单独的 token 级 scan**：7.4 节的 parallel scan 已经直接在全局时间步 $\tau = 1, \ldots, TK$ 上进行，自然包含了 token 间和 token 内的状态传播。Token 级折叠在**概念上**有助于理解与 Mamba 的等价性，但在**实现上**，直接对全序列做一次 parallel scan 更简洁。

### 7.7 单层完整并行算法

以下是 SNNBlock 处理 $TK$ 帧的完整并行化流程。设输入为 $\text{spike}_{in}[\tau] \in \{0,1\}^D$，$\tau = 1, \ldots, TK$。

#### Phase 1: 批量投影 [O(1) 并行深度]

所有 $TK$ 帧的 6 条输入路径**同时**计算：

$$I[\tau] = W_{in} \cdot \text{spike}_{in}[\tau] \in \mathbb{R}^{D \times N} \quad \forall \tau$$

$$\beta[\tau] = \sigma\!\left(W_\beta^{(x)} \cdot \text{spike}_{in}[\tau] + b_\beta\right) \in (0,1)^{D \times N} \quad \forall \tau$$

$$\alpha[\tau] = \text{softplus}\!\left(W_\alpha^{(x)} \cdot \text{spike}_{in}[\tau] + b_\alpha\right) \in \mathbb{R}_+^{D \times N} \quad \forall \tau$$

$$V_{th}[\tau] = V_{min} + \left|W_{th}^{(x)} \cdot \text{spike}_{in}[\tau] + b_{th}\right| \in \mathbb{R}_+^{D \times N} \quad \forall \tau$$

$$\text{gate}[\tau] = \sigma\!\left(W_{gate} \cdot \text{spike}_{in}[\tau]\right) \in (0,1)^D \quad \forall \tau$$

$$I_{skip}[\tau] = W_{skip} \cdot \text{spike}_{in}[\tau] \in \mathbb{R}^D \quad \forall \tau$$

**实现**：将 $TK$ 帧堆叠为 $(TK \times \text{batch}, D)$ 的矩阵，一次矩阵乘法完成每条路径。6 条路径互相独立，可进一步并行（或顺序计算 6 次大矩阵乘法）。

#### Phase 2: 线性轨迹 Parallel Scan [$O(\log(TK))$ 并行深度]

计算 $V_L[\tau] = \beta[\tau] \cdot V_L[\tau{-}1] + \alpha[\tau] \cdot I[\tau]$：

$$u[\tau] = \alpha[\tau] \cdot I[\tau] \quad \forall \tau$$

$$(A[\tau], B[\tau]) = \text{HillisSteeleScan}\!\left((\beta[\tau], u[\tau])_{\tau=1}^{TK}\right)$$

$$V_L[\tau] = A[\tau] \cdot V_{init} + B[\tau] \quad \forall \tau$$

#### Phase 3: Spike 不动点迭代 [$O(I \cdot \log(TK))$ 并行深度]

```
s ← Θ(V_L - V_th)                                              // O(1)
for i = 1, 2 (typically):
  (A_s, B_s) = HillisSteeleScan((β, V_th · s))                 // O(log TK)
  ΔS = A_s · 0 + B_s = B_s                                     // ΔS[0] = 0
  ΔS_prev ← shift_right(ΔS, fill=0)                            // O(1)
  V_pre ← V_L - β · ΔS_prev                                    // O(1)
  s ← Θ(V_pre - V_th)                                          // O(1)
```

#### Phase 4: 隐 spike 输出投影 [O(1) 并行深度]

$$I_{out}[\tau] = W_{out} \cdot s[\tau] \quad \forall \tau$$

$$I_{total}[\tau] = I_{out}[\tau] \odot \text{gate}[\tau] + I_{skip}[\tau] \quad \forall \tau$$

#### Phase 5: 输出神经元 Parallel Scan [$O(\log(TK))$ 并行深度]

输出 PLIF 神经元（固定 $\beta_{out}$, $V_{th,out}$）：

$$V_{out}[\tau] = \beta_{out} \cdot V_{out}[\tau{-}1] + (1 - \beta_{out}) \cdot I_{total}[\tau]$$

使用 parallel scan + spike 不动点迭代（与 Phase 2-3 相同结构，但 $\beta$ 为标量常数 $\beta_{out}$，$V_{th}$ 为标量常数 $V_{th,out}$——更简单）。

$$\text{spike}_{out}[\tau] \in \{0,1\}^D \quad \forall \tau$$

#### Phase 6: 状态更新 [O(1)]

保存末时间步的膜电位供下一次调用：

$$V_{hidden} \leftarrow V_{post}[TK], \quad V_{output} \leftarrow V_{out}[TK]$$

### 7.8 SNNFFN 的并行化

SNNFFN 结构（3 个 PLIF 神经元 + AND 门 + skip）同样可并行化。

**输入**：$\text{spike}_{mid}[\tau] \in \{0,1\}^D$（来自 SNNBlock 的输出）

#### Phase 1: 批量投影 [O(1)]

$$I_{gate}[\tau] = W_{gate\_proj} \cdot \text{spike}_{mid}[\tau] \in \mathbb{R}^{D_{ff}} \quad \forall \tau$$

$$I_{up}[\tau] = W_{up\_proj} \cdot \text{spike}_{mid}[\tau] \in \mathbb{R}^{D_{ff}} \quad \forall \tau$$

$$I_{skip\_ffn}[\tau] = W_{skip\_proj} \cdot \text{spike}_{mid}[\tau] \in \mathbb{R}^D \quad \forall \tau$$

#### Phase 2: Gate/Up 神经元 Parallel Scan [$O(\log(TK))$]

两个独立的 PLIF parallel scan（可并行）：

$$\text{gate\_spike}[\tau] = \text{PLIF\_scan}(\beta_{gate}, I_{gate})[\tau] \quad \forall \tau$$

$$\text{up\_spike}[\tau] = \text{PLIF\_scan}(\beta_{up}, I_{up})[\tau] \quad \forall \tau$$

#### Phase 3: AND 门 + 降维 [O(1)]

$$\text{gated}[\tau] = \text{gate\_spike}[\tau] \odot \text{up\_spike}[\tau] \quad \forall \tau$$

$$I_{ffn\_out}[\tau] = W_{down\_proj} \cdot \text{gated}[\tau] + I_{skip\_ffn}[\tau] \quad \forall \tau$$

#### Phase 4: 输出神经元 Parallel Scan [$O(\log(TK))$]

$$\text{spike}_{ffn\_out}[\tau] = \text{PLIF\_scan}(\beta_{ffn\_out}, I_{ffn\_out})[\tau] \quad \forall \tau$$

### 7.9 完整网络的并行化流程

#### 7.9.1 编码并行化

对所有 $T$ 个 token 的 embedding 同时编码为 $K$ 帧 spike：

$$\text{spike\_frames}[n, k] = \text{bit}_k\!\left(\sigma(\text{encode\_proj}(x_n))\right) \in \{0,1\}^D$$

这是纯逐元素操作 + 一次矩阵乘法，$O(1)$ 深度。将 $T \times K$ 帧展平为序列长度 $TK$。

#### 7.9.2 逐层并行处理

> **v7.5 更新**：引入连续残差流。层间传递连续值 `h`，不再是二值 spike。

```
h ← encode(all_tokens)                          // (TK, batch, D), 值域 {0,1}
                                                  // K-bit 二进制编码，也是合法连续值

for l = 1 to L:
  h ← SNNDecoderLayer_l.forward_parallel(h)
    // 内部: PLIFNode → SNNBlock → out_proj → 残差
    //      → PLIFNode → SNNFFN  → out_proj → 残差
    // PLIFNode 直接接收原始 h — V_th 作为 SNN 原生归一化机制
    // 残差连接: h = h + out_proj(spike_out)
    // 并行深度: O(log(TK))

// L 层串行, 每层 O(log(TK)) 深度
// 总深度: L × O(log(TK))
// 梯度: ∂h_out/∂h_in = I + ∂(out_proj·snn(plif(h)))/∂h（恒等项保证梯度不消失）
```

#### 7.9.3 解码与 Loss 计算

从 $TK$ 帧的输出 spike 重构 $T$ 个 token 的 logits：

$$\hat{y}_n = \sum_{k=1}^{K} \text{spike}_{out}[n, k] \cdot 2^{-k} \in [0, 1)^D \quad \forall n$$

$$\text{logits}_n = \text{Embedding}^T \cdot \text{LateralInhibition}\!\left(\text{decode\_proj}(\hat{y}_n)\right) \quad \forall n$$

$$\mathcal{L} = \frac{1}{T} \sum_{n=1}^{T} \text{CrossEntropy}(\text{logits}_n,\ \text{target}_n)$$

所有 $T$ 个 token 的解码和 loss 同时计算（$O(1)$ 深度 + 一次矩阵乘法）。

### 7.10 复杂度分析

#### 7.10.1 并行深度对比

| 阶段 | 当前（串行） | v7（并行） |
|---|---|---|
| 编码 | $O(T)$ | $O(1)$ |
| 每层 SNNBlock Phase 1 (投影) | $O(TK)$（逐帧计算） | $O(1)$（批量矩阵乘法） |
| 每层 SNNBlock Phase 2 (V_L scan) | $O(TK)$ | $O(\log(TK))$ |
| 每层 SNNBlock Phase 3 (spike 迭代) | — (隐含在串行中) | $O(I \cdot \log(TK))$, $I \leq 3$ |
| 每层 SNNBlock Phase 4-5 (输出) | $O(TK)$ | $O(1) + O(\log(TK))$ |
| 每层 SNNFFN | $O(TK)$ | $O(1) + O(\log(TK))$ |
| 解码 + Loss | $O(T)$ | $O(1)$ |
| **每层合计** | $O(TK) = O(8192)$ | $O(I \cdot \log(TK)) \approx O(40)$ |
| **全网合计** | $T \times K \times L = 163{,}840$ | $L \times O(I \cdot \log(TK)) \approx 20 \times 40 = 800$ |

**串行深度加速比**：$163{,}840 / 800 \approx \mathbf{205 \times}$

#### 7.10.2 工作量对比

| | 当前 | v7 |
|---|---|---|
| 矩阵乘法次数/层 | $TK \times 6 = 49{,}152$ 次小矩阵乘法 | $6$ 次大批量矩阵乘法 |
| 单次矩阵乘法规模 | $(\text{batch}, D) \times (D, DN)$ | $(TK \cdot \text{batch}, D) \times (D, DN)$ |
| parallel scan 工作量 | 0 | $O(TK \cdot \log(TK))$ 逐元素运算 |
| **总 FLOPs** | 相当 | **略多**（scan 额外 $\log$ 倍） |
| **GPU 利用率** | 极低（小矩阵） | **极高**（大矩阵 + 大张量） |

关键：总 FLOPs 不减少（甚至略增），但 GPU 利用率从 $< 5\%$ 跃升至 $> 80\%$。wall-clock 时间的加速来自于**计算密度**的提升，而非计算量的减少。

#### 7.10.3 显存分析

**当前**：逐帧计算，每帧的中间结果在下一帧前释放。峰值显存 $\approx O(\text{batch} \times D \times N)$。

**v7（并行）**：所有 $TK$ 帧的中间结果同时存在。峰值显存 $\approx O(TK \times \text{batch} \times D \times N)$。

$$\text{v7 峰值显存} \approx TK \times \text{batch} \times DN \times 4\text{B(bf16=2B)} \times \text{中间张量数}$$

$$\approx 8192 \times 16 \times 8192 \times 2 \times 10 \approx 21\text{GB}$$

在 128.5 GB VRAM 的 GB10 上完全可承受（约 16% 利用率）。

### 7.11 K = 16 的升级

#### 7.11.1 编码精度

| K | 量化级数 | 最小精度 | 适用场景 |
|---|---|---|---|
| 8 | $2^8 = 256$ | $1/256 \approx 0.004$ | 分类任务、粗粒度生成 |
| **16** | $2^{16} = 65{,}536$ | $1/65536 \approx 0.000015$ | **语言模型**（embedding 空间连续） |

语言模型的 embedding 维度 $D = 1024$，每个维度的有效信息需要至少 16-bit 精度才能保持 embedding 空间的区分能力。$K = 8$ 的 256 级量化会导致不可逆的信息损失。

#### 7.11.2 并行化后 K 增大的代价

**无并行化**：$K: 8 \to 16$，计算量翻倍（$TK$ 翻倍）。

**有并行化**：$\log(TK)$ 仅增加 $\log(2) = 1$ 轮扫描步骤。$K = 8$ 时 $\log_2(4096) = 12$；$K = 16$ 时 $\log_2(8192) = 13$。**增加不到 8%**。

$$\text{K 翻倍的额外代价} = \frac{\log_2(T \cdot 16)}{\log_2(T \cdot 8)} - 1 = \frac{13}{12} - 1 = 8.3\%$$

这是 parallel scan 的核心优势：序列长度翻倍，计算深度只增加 $O(1)$。

#### 7.11.3 隐状态动力学的丰富化

$K = 16$ 给每个 token 提供 16 步的动力学演化时间：

- **短程神经元**（$\beta = 0.80$）：$\bar{A}_{eff} = 0.80^{16} = 0.028$。每 token 几乎完全刷新，可以在 16 步内多次发放/reset，产生丰富的时间模式。
- **长程神经元**（$\beta = 0.99$）：$\bar{A}_{eff} = 0.99^{16} = 0.851$。跨 token 保持 85% 的状态，16 步的精细写入让信息更精确地累积。

### 7.12 架构变更总结

#### v6 → v7 的核心差异

| 方面 | v6 | v7 |
|---|---|---|
| $W^{(V)}$ | $W_\beta^{(V)}, W_\alpha^{(V)}, W_{th}^{(V)}$（N×N，3组） | **移除** |
| 调制依赖 | $\text{spike}_{in} + V[t{-}1]$ | 仅 $\text{spike}_{in}$ |
| K | 8 | **16** |
| 计算模式 | 逐帧串行 | **Parallel scan（层内全并行）** |
| 串行深度 | $T \times K \times L$ | $L \times O(\log(TK))$ |
| 矩阵乘法规模 | 小矩阵（batch × D） | 大矩阵（TK·batch × D） |
| GPU 利用率 | < 5% | > 80% |
| 理论加速比 | 1× | ~200× |

#### 与 Mamba 的完整对齐

| 特性 | Mamba | SNN v7 |
|---|---|---|
| 状态方程 | $h = \bar{A}(x) h + \bar{B}(x) x$ | $V = \bar{A}_{eff}(\text{spike}) V + \bar{B}_{eff}(\text{spike})$ |
| A 依赖 | 仅输入 $x$ | 仅输入 spike |
| 并行化 | Parallel scan, $O(\log T)$ | Parallel scan, $O(\log(TK))$ |
| 非线性 | 无（纯线性 SSM） | **有**（spike + soft reset，在 K 步内作用） |
| 读出 | $y = C \cdot h$（线性） | $s = \Theta(V - V_{th})$（非线性，稀疏） |
| 门控 | SiLU gate | sigmoid gate + AND gate (FFN) |
| 精度 | 连续值 | K-bit 二进制（K=16，精度 $2^{-16}$） |

**SNN v7 在保持 Mamba 的并行化优势的同时，通过 spike + soft reset 提供了 Mamba 所没有的非线性表达能力。**

#### v7 → v7.5 的核心差异

| 方面 | v7 | v7.5 |
|---|---|---|
| 层间信号 | 二值 spike {0,1} | **连续值** h ∈ R（连续残差流） |
| 残差连接 | 无（spike 直传） | **有**：h = h + out_proj(spike)（恒等梯度路径） |
| 归一化 | 无 | **PLIFNode V_th**（SNN 原生幅度调节），输出层 LateralInhibition |
| 输入 spike 转换 | STE 阈值 | **PLIFNode**（D 维可学习 β/V_th，完整膜电位动力学） |
| 输出投影 | 无 | **nn.Linear(D,D)**：spike→连续空间 |
| 深层梯度 | 20 层梯度消失（L0: 3.16e-22） | **全层有效梯度**（L0/L19 ratio 0.22~2.9x） |
| 训练方法 | Surrogate gradient backprop | 同（v7.1 引入） |
| 额外参数/层 | 0 | ~1.18M（PLIFNode×2 + Linear(D,D)×2） |

**梯度分析**：

$$\frac{\partial h_{out}}{\partial h_{in}} = I + \frac{\partial}{\partial h}\left[\text{out\_proj}\!\left(\text{SNN}\!\left(\text{PLIFNode}\!\left(\text{LI}(h)\right)\right)\right)\right]$$

恒等项 $I$ 保证梯度不因层数衰减——彻底解决 v7 的 20 层梯度消失问题。

### 7.13 v7.5 连续残差流（Continuous Residual Stream）

#### 7.13.1 问题根因

v7.1 训练发现 loss 在 ~7.5 处 plateau（step 1200 后 3000+ 步无下降）。梯度诊断：

| Layer | Avg Grad Norm |
|-------|---------------|
| Layer 19（最后） | 4.39e-02 |
| Layer 18 | 9.73e-04 |
| Layer 17 | 2.37e-05 |
| Layer 0（第一） | 3.16e-22 |

**根因**：v7 的 `SNNDecoderLayer` 无层间残差连接。每层 spike 直接传递，5 个 surrogate gradient 运算（乘以 <1 的系数）× 20 层 = 100 次衰减 → 梯度归零。

#### 7.13.2 解决方案：连续残差流

借鉴 Transformer 的 Pre-Norm + 残差连接模式，但保持全网 SNN 约束：

**Transformer（对照）**：
```
h = x + Attention(RMSNorm(x))    // 残差 1
h = h + FFN(RMSNorm(h))           // 残差 2
```

**SNN v7.5**：
```
h = h + out_proj(SNNBlock(PLIFNode(h)))    // 残差 1
h = h + out_proj(SNNFFN(PLIFNode(h)))      // 残差 2
```

**关键设计**：
1. **PLIFNode 直接接收原始 h**：D 维可学习 β 和 V_th 的完整 PLIF 动力学，V_th 作为 SNN 原生的幅度归一化机制。不使用层内归一化——归一化会压制振幅信息，抑制神经元参数和调制参数的学习。
2. **PLIFNode 膜电位动力学**：连续值 → 二值 spike 的转换经过真正的膜电位积累和发放过程，不是简单的 STE 阈值。
3. **out_proj**：nn.Linear(D,D) 突触，将 spike {0,1} 映射回连续空间。初始化使用 GPT-2 style 缩放 $\mathcal{N}(0, 0.02/\sqrt{2L})$ 防止深层梯度爆炸。
4. **残差加法**：h = h + out_proj(spike) 创建恒等梯度路径。

**连续值从何而来**：spike × W（突触操作）产生实数值，残差加法 h + (spike × W) 在连续空间中累积。这完全符合 SNN 生物学——突触后电位本身就是连续值，层间传递的是突触后电流/电位的叠加。

#### 7.13.3 归一化策略

**v7.5b 设计决策**：层内不使用归一化（LateralInhibition 已从层内移除）。

**原因**：层内归一化（无论 Pre-norm 还是 Post-norm）会压制输入的振幅信息，导致 PLIFNode 的 V_th 动力学和调制参数（β/α/V_th 路径）无法有效学习。PLIFNode 的 D 维可学习 V_th 本身就是 SNN 原生的幅度调节机制——它能根据各维度的信号统计特性自适应调整发放阈值。

**残差增长有界性分析**：spike ∈ {0,1}，out_proj 初始化 $\sigma = 0.02/\sqrt{2L}$，20 层后残差流幅度约从 0.5 增长到 ~2.1（约 4×），完全在 PLIFNode 可学习 V_th 的调节范围内。

**LateralInhibition（侧抑制）仅用于模型输出层**：decode_proj 之后、Embedding^T 之前。数学等价于 RMSNorm，有 Triton fused kernel。

#### 7.13.4 验证结果

**梯度覆盖率**：606/606 可训练参数全部有有效梯度（ALL PASSED）。

**层间梯度衰减**（v7.5 vs v7.1）：

| 探针参数 | L0 Grad | L19 Grad | Ratio L0/L19 |
|----------|---------|----------|--------------|
| block_out_proj | ~1e-2 | ~1e-2 | ~1.0 |
| ffn_out_proj | ~1e-2 | ~1e-2 | ~1.0 |
| block.W_in | ~1e-3 | ~1e-3 | ~0.5 |
| input_neuron.w | ~1e-3 | ~5e-3 | ~0.2 |

所有比值在 0.2~3.0 范围内——完全正常的梯度传播，无衰减无爆炸。

---

## 8. 理论定位

### 8.1 在序列建模谱系中的位置

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

### 8.2 关键理论支撑

**Linear Attention ↔ RNN 等价性**（Katharopoulos et al., ICML 2020）：

SNN时间残差 $V[t] = \beta V[t-1] + I[t]$ 在数学上属于Linear Attention的递归分支。但标准Linear Attention是纯线性的，表达能力弱于Softmax Attention。

SNN的阈值+reset在这个线性递归上叠加了非线性，理论上应该增强表达能力。这是SNN相对于纯线性递归模型（包括Mamba）的独特优势点。

**信息瓶颈理论**（Tishby et al.）：

spike阈值天然实现了信息瓶颈——只传递超阈值的信息。泛化界 $R - \hat{R} \leq C\sqrt{I(T;X)/n}$ 表明适度的信息压缩有利于泛化。V_th的设置直接控制瓶颈的松紧度。

**混合时间与遍历理论**：

$t_{mix} \sim 1/|\ln\beta|$ 给出了每个β值对应的有效记忆长度。多时间尺度的β分布 = 多个并行的信息信道，每个信道有不同的有效带宽。

---

## 9. 训练方法

> **v7.1 更新**：从零阶优化（IG-ZO/SPSA）切换到 **Surrogate Gradient Backpropagation**。实验证明 surrogate gradient 在 SNN 中完全可行且效率远超零阶方法（24× TPS 提升）。

### 9.1 大方向

**Surrogate gradient backpropagation**——spike 的阶跃函数 $\Theta(x)$ 在前向保持离散（输出 {0,1}），反向传播时用光滑函数的梯度替代：

$$\frac{\partial s}{\partial V_{pre}} \approx \frac{\alpha}{2(1 + \alpha|V_{pre} - V_{th}|)^2} \quad \text{(Sigmoid surrogate, } \alpha=4.0\text{)}$$

**为什么放弃零阶优化**：
- SPSA 在 619M 参数空间中收敛极慢（~5 TPS）
- Surrogate gradient 是 SNN 训练的主流方法（Neftci et al., 2019），已在大规模 SNN 中证明有效
- v7 的 parallel scan 将递推展开为显式计算图，PyTorch autograd 可直接反向传播
- 实测：surrogate gradient 120 TPS vs SPSA 5 TPS = **24× 加速**

### 9.2 Surrogate Gradient 在 SNN 中的数学基础

**前向**（精确）：
$$s[\tau] = \Theta(V_{pre}[\tau] - V_{th}[\tau]) = \begin{cases} 1 & V_{pre} > V_{th} \\ 0 & V_{pre} \leq V_{th} \end{cases}$$

**反向**（surrogate）：
$$\frac{\partial \mathcal{L}}{\partial V_{pre}[\tau]} = \frac{\partial \mathcal{L}}{\partial s[\tau]} \cdot \underbrace{\sigma'_\alpha(V_{pre}[\tau] - V_{th}[\tau])}_{\text{surrogate gradient}}$$

其中 $\sigma'_\alpha(x) = \frac{\alpha}{2(1+\alpha|x|)^2}$ 是 Sigmoid surrogate 的导数。$\alpha=4.0$ 控制锐度（值越大越接近真实阶跃但梯度越稀疏）。

**Triton 实现**：v7.3 的 fused PLIF kernel 将 forward（scan + spike + reset）和 backward（surrogate gradient + chain rule）融合在单个 Triton kernel 中。Backward 中 $-\alpha \cdot x$ 被 clamp 到 88 防止 exp overflow。

### 9.3 训练配置

```python
# v7.5 训练配置
optimizer = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)

# Cosine decay with warmup
warmup_iters = 1000
max_iters = 100000

# 混合精度
dtype = torch.bfloat16
grad_scaler = None  # bf16 不需要 grad scaler

# 梯度检查点（按层重计算，省显存）
for layer in model.layers:
    h = checkpoint(_layer_forward, layer, h, use_reentrant=False)

# 梯度累积
accumulation_steps = 32  # effective batch = 2 × 32 = 64
```

**参数分组与权重衰减**：
| 参数组 | 权重衰减 | 理由 |
|--------|----------|------|
| embedding, encode/decode proj | 0.1 | 标准正则化 |
| W_in, W_gate, W_skip, W_out | 0.1 | 信号路径突触 |
| W_beta, W_alpha, W_th | 0.1 | 调制路径突触 |
| b_beta, b_alpha, b_th | 0.0 | 偏置不做衰减 |
| PLIFNode.w, PLIFNode.v_th | 0.0 | 神经元动力学参数 |
| LateralInhibition.gain（仅输出层） | 0.0 | 归一化增益 |
| out_proj | 0.1 | 残差投影 |

### 9.4 训练进展

| 版本 | 方法 | 参数量 | TPS | 最佳 Loss | 备注 |
|------|------|--------|-----|-----------|------|
| v7.0 | SPSA | 619M | ~5 | 7.78 (760 steps) | 零阶优化，极慢 |
| v7.1 | Surrogate gradient | 619M | ~120 | 7.40 (1010 steps) | 24× 加速，突破 7.78 |
| v7.5 | Surrogate gradient | ~643M | TBD | TBD | 连续残差流，待训练 |

### 9.5 早期设计：信息论诊断体系（参考）

> 以下为 v7.0 IG-ZO 阶段设计的诊断框架。切换到 surrogate gradient 后不再用于训练，但诊断度量可用于模型分析。

**六个设计目标的诊断**（参见 Q5.md 完整方法论）：
1. 上下文选择性方差：$\text{Var}_{ctx}[\beta(t) | \text{spike}_{in}]$
2. 多时间尺度分化：β 分布熵 + MI 按 β 分组
3. 静默积累有效性：$I(V_{silent}; y_{future})$
4. 发放信息效率：$\eta_{fire}$
5. 衰减-写入解耦：$\text{Corr}[\alpha(t), \beta(t)]$
6. 跨时间尺度协调：$I(\Delta V_{short}; \Delta\beta_{long})$（v7 移除 W^(V) 后此项需改为层间分析）

---

## 10. 实现路径与开放问题

### 10.1 实现进展

| 版本 | 状态 | 核心内容 |
|------|------|----------|
| v6 | 完成 | 串行 SNN 递推，SPSA 训练 |
| v7 | 完成 | Parallel scan，移除 W^(V)，K=16 |
| v7.1 | 完成 | Surrogate gradient backprop 替代 SPSA，24× TPS 提升 |
| v7.2 | 完成 | Triton linear_recurrence kernel，33× vs Hillis-Steele |
| v7.3 | 完成 | Fused PLIF kernel（scan + spike + reset + surrogate），4-20× |
| v7.4 | 完成 | Row-param PLIF kernel + torch.compile fused modulation，1.74× 整层 |
| v7.5 | **当前** | 连续残差流 + PLIFNode 输入（V_th 原生归一化），解决 20 层梯度消失 |

### 10.2 当前训练配置（v7.5）

```
硬件: NVIDIA DGX Spark (GB10, 128.5 GB VRAM, sm_121a)
模型: D=768, N=8, K=16, Layers=20, D_ff=2304, vocab=6144
参数: ~643M (619M base + ~24M residual stream)
数据: seq_monkey_datawhale.jsonl (29M samples, seq_len=512)
训练: AdamW, lr=2e-4, warmup=1000, bf16, gradient checkpoint
有效批量: batch_size=2 × accumulation_steps=32 = 64
TPS: ~120 tokens/sec
```

### 10.3 已解决的设计问题

| 问题 | 结论 |
|------|------|
| Q1: 输出用 spike 还是膜电位？ | spike（K-bit 二进制编码） |
| Q2: W_in/W_out 是否有时间状态？ | 无（标准突触） |
| Q3: N 的选择？ | N=8（实验超参） |
| Q4: β 调制网络复杂度？ | $W^{(x)}$ 系列（D→DN），v7 移除 $W^{(V)}$ |
| Q5: 静默积累验证？ | Q5.md 三阶段验证方法论 |
| Q6: 训练方法？ | Surrogate gradient（v7.1+），弃用 IG-ZO/SPSA |
| Q7: 深层梯度消失？ | 连续残差流（v7.5），恒等梯度路径 |
| Q8: 归一化层？ | 层内无归一化（PLIFNode V_th 原生调节），输出层 LateralInhibition |

### 10.4 开放问题

1. **长序列外推**：当前 seq_len=512，K=16 → TK=8192。更长序列的 parallel scan 显存 $O(TK)$ 是否需要分段处理？
2. **β 分布动态**：训练中多时间尺度结构是否保持？需监控 β 分布的方差/熵。
3. **输出 PLIFNode v_th 自适应**：深层级联衰减导致 firing rate 降低，动态 V_th 是否需要显式补偿？
4. **模型规模扩展**：D=1024, L=22 的配置尚未实验。

---

## 11. 参考来源

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

*本文档记录了SNN隐神经元状态空间的完整设计与实现。核心机制：β(t)、α(t)、V_th(t) 由当前输入 spike 动态计算（v7: 移除膜电位反馈 W^(V) 以支持 parallel scan），spike/reset 提供隐式状态反馈。v7.5 引入连续残差流（PLIFNode V_th 原生归一化 + 残差连接）解决 20 层梯度消失问题。训练采用 surrogate gradient backpropagation（v7.1+），在 DGX Spark (GB10) 上以 ~95 TPS 运行。*

*状态: v7.5 已实现，训练中。*
