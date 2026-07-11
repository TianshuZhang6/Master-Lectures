import json
from pathlib import Path


TITLE = "L15-L15 Neutral network representation and the memorize and optimization （Chinese）"
OUT = Path("output")
OUT.mkdir(exist_ok=True)
NOTEBOOK = OUT / f"{TITLE}.ipynb"


sections = [
r"""# 神经网络：表示、记忆与优化

> **L15–L16 整合讲义（中文）**  
> 课程：CIT413048 Mathematical Foundations of Machine Learning  
> 基于 Suvrit Sra 的 L15/L16，并结合 Francis Bach 的 *Learning Theory from First Principles* 与 Shalev-Shwartz、Ben-David 的 *Understanding Machine Learning*。

本讲义围绕四条主线展开：

1. 神经网络如何从固定特征走向可学习特征；
2. ReLU 网络能表示什么，以及深度为何有用；
3. 有限样本记忆能力与 VC 维的区别；
4. SGD、反向传播、初始化、Dropout 与 Batch Normalization 如何共同决定训练行为。

---

### 统一符号

- 输入：$x\in\mathbb R^d$，标签：$y$；
- 第 $l$ 层权重和偏置：$W^l\in\mathbb R^{m_l\times m_{l-1}}$、$b^l\in\mathbb R^{m_l}$；
- 预激活与激活：$z^l,a^l$，其中 $a^0=x$；
- 激活函数：$\sigma$，ReLU 为 $[u]_+=\max\{u,0\}$；
- 网络参数集合：$\theta=\{W^l,b^l\}_{l=1}^L$；
- 总体风险与经验风险：$R(\theta)$、$\widehat R_N(\theta)$。
""",

r"""## 1. 从固定特征到可学习特征

### 1.1 核方法视角

核方法先选定核函数 $k$，再由训练数据产生隐式特征。以核 SVM 为例，表示定理给出

$$
w=\sum_{i=1}^n\alpha_i y_i\phi(x_i),
$$

因此对新样本 $x$ 的预测为

$$
\langle w,\phi(x)\rangle
=\sum_{i=1}^n\alpha_i y_i k(x_i,x)
=\langle\alpha,\Phi(x)\rangle,
$$

其中

$$
\Phi(x)=\big[y_1k(x_1,x),\ldots,y_nk(x_n,x)\big]^\top.
$$

分类器对 $\Phi(x)$ 是线性的，但 $\Phi$ 的形式由核预先固定。神经网络的关键变化是将特征映射参数化，并与最后的线性分类器联合学习：

$$
x\longmapsto \Phi_{\theta_1}(x)
\longmapsto w^\top\Phi_{\theta_1}(x)+b,
\qquad \theta=(\theta_1,w,b).
$$

### 1.2 神经网络的四种视角

1. **可学习特征：** 隐藏层从数据中学习非线性表示；
2. **嵌套分类器：** 一个分类器的输出成为下一层分类器的输入；
3. **由简单到复杂：** 简单局部决策逐层组成复杂决策；
4. **函数逼近器：** 网络通过基函数叠加与函数复合逼近目标函数。

### 1.3 前馈网络的统一形式

令 $a^0=x$。对于 $l=1,\ldots,L$，

$$
z^l=W^l a^{l-1}+b^l,
\qquad
a^l=\sigma^l(z^l).
$$

若最后一层为线性输出，则 $F(x;\theta)=z^L$。展开后，

$$
F(x;\theta)
=W^L\sigma^{L-1}\!\left(
W^{L-1}\cdots\sigma^1(W^1x+b^1)\cdots+b^{L-1}
\right)+b^L.
$$

单隐层、标量输出网络为

$$
f(x)=\sum_{j=1}^m\eta_j\sigma(w_j^\top x+b_j).
$$

固定 $(w_j,b_j)$ 时，这是以 $\sigma(w_j^\top x+b_j)$ 为固定特征的线性模型；同时优化这些参数时，网络开始学习特征。

**来源：** L15 pp. 5–18；Bach §9.1–9.2；UML §20.1。
""",

r"""## 2. 激活函数、输出层与损失

### 2.1 常见激活函数

| 函数 | 定义 | 导数或性质 | 典型用途 |
|---|---|---|---|
| Sigmoid | $\sigma(u)=1/(1+e^{-u})$ | $\sigma'(u)=\sigma(u)(1-\sigma(u))$；大 $|u|$ 时饱和 | 二分类概率输出 |
| $\tanh$ | $\tanh(u)$ | $1-\tanh^2(u)$；零中心但会饱和 | 传统隐藏层 |
| ReLU | $[u]_+=\max(0,u)$ | $\mathbf 1\{u>0\}$，$u=0$ 处使用次梯度 | 现代隐藏层 |
| Softmax | $p_k=e^{z_k}/\sum_j e^{z_j}$ | $J=\operatorname{diag}(p)-pp^\top$ | 多分类输出 |

### 2.2 二分类交叉熵与逻辑损失

令 $y\in\{-1,+1\}$，$g(x)=\operatorname{sigmoid}(f(x))$。负对数似然为

$$
-\mathbf 1\{y=1\}\log g(x)
-\mathbf 1\{y=-1\}\log(1-g(x))
=\log\bigl(1+e^{-yf(x)}\bigr).
$$

所以 sigmoid 与 binary cross-entropy 的组合等价于直接对 logit 使用 logistic loss。

### 2.3 Softmax 交叉熵梯度

令 $p=\operatorname{softmax}(z)$，真实类别的 one-hot 向量为 $y$，

$$
\ell(z,y)=-\sum_k y_k\log p_k.
$$

因为

$$
\frac{\partial\log p_r}{\partial z_k}=\delta_{rk}-p_k,
$$

所以

$$
\boxed{\frac{\partial\ell}{\partial z}=p-y.}
$$

这是多分类输出层反向传播最常用的起始公式。
""",

r"""## 3. 非线性特征与线性可分性

两个 ReLU 单元定义两个半空间响应：

$$
z_1=w_1^\top x+b_1,
\qquad
z_2=w_2^\top x+b_2,
$$

$$
\Phi(x)=\big([z_1]_+,[z_2]_+\big).
$$

每个 ReLU 由超平面 $w_j^\top x+b_j=0$ 把空间分成两部分：

- 一侧被截断为零；
- 另一侧保留到超平面的线性距离信息。

多个 ReLU 共同把输入空间切分成多面体区域，并在每个区域上实现不同的仿射映射。因此，原空间中非线性可分的数据，在特征空间 $\Phi(x)$ 中可能变得线性可分。

> **表示存在性不等于训练可达性。** 即使存在一组超平面能把样本变得线性可分，梯度法仍需要从随机初始化中找到这些方向。法向量符号翻转可能产生完全不同的隐藏表示。

**来源：** L15 pp. 19–25。
""",

r"""## 4. 表达能力、万能逼近与深度优势

### 4.1 万能逼近定理

对紧集 $K\subset\mathbb R^d$、连续函数 $g$ 和任意 $\varepsilon>0$，在适当的非多项式激活函数下，存在单隐层网络使

$$
\sup_{x\in K}
\left|
\sum_{j=1}^m\eta_j\sigma(w_j^\top x+b_j)-g(x)
\right|<\varepsilon.
$$

ReLU 是非多项式函数，因此满足相应的稠密性结论。

万能逼近定理**不保证**：

1. 所需宽度 $m$ 很小；
2. SGD 能找到所需参数；
3. 从有限数据得到的逼近具有小测试误差。

### 4.2 一维 ReLU 表示的直接推导

设 $f$ 在 $[-R,R]$ 上连续分段仿射，折点为

$$
a_1<\cdots<a_k,
$$

各区间斜率为 $v_0,\ldots,v_k$。则

$$
\boxed{
f(x)=f(-R)+v_0(x+R)
+\sum_{j=1}^k(v_j-v_{j-1})(x-a_j)_+.
}
$$

验证：在越过折点 $a_j$ 时，导数增加 $v_j-v_{j-1}$，因此新斜率恰好变为 $v_j$。连续函数可由细网格上的分段线性插值一致逼近，从而得到一维万能逼近。

### 4.3 光滑函数的积分表示

若 $f$ 二次连续可微，Taylor 公式的积分余项给出

$$
f(x)=f(-R)+f'(-R)(x+R)
+\int_{-R}^R f''(b)(x-b)_+\,db.
$$

因此连续的 ReLU 混合可以精确表示光滑函数；输出权重的总变差由 $\int|f''(b)|db$ 控制。

### 4.4 三层 ReLU 的构造思路

1. 把 $[0,1]^d$ 划分为足够小的超矩形；
2. 在每个小块内用常数近似连续函数 $g$；
3. 用两层 ReLU 构造超矩形的连续软指示器；
4. 用线性输出层叠加这些软指示器。

### 4.5 深度为什么可能节省宽度

以单位球指示函数为例，深层构造可写为

$$
x
\longmapsto (x_1^2,\ldots,x_d^2)
\longmapsto \sum_{i=1}^d x_i^2=\|x\|_2^2
\longmapsto \mathbf 1\{\|x\|_2\le1\}.
$$

深度允许复用中间计算并显式编码函数复合结构。某些目标可由三层网络以多项式宽度逼近，但两层网络达到同等精度可能需要指数宽度。深度优势依赖目标函数是否具有与网络匹配的组合结构，并非所有函数都受益于加深网络。

**来源：** L15 pp. 26–39；Bach §9.3；UML §20.3。
""",

r"""## 5. 近似误差、估计误差与优化误差

令 $\mathcal F$ 为模型类，$f^*$ 为总体最优函数，

$$
f_{\mathcal F}^*\in\arg\min_{f\in\mathcal F}R(f),
$$

$\widehat f_{\mathrm{ERM}}$ 为经验风险最小化解，$\widehat f$ 为实际算法输出。概念上，

$$
R(\widehat f)-R(f^*)
=\underbrace{R(f_{\mathcal F}^*)-R(f^*)}_{\text{近似误差}}
+\underbrace{R(\widehat f_{\mathrm{ERM}})-R(f_{\mathcal F}^*)}_{\text{估计误差}}
+\underbrace{R(\widehat f)-R(\widehat f_{\mathrm{ERM}})}_{\text{优化误差}}.
$$

严格证明通常通过经验风险与总体风险之间的偏差控制估计项。

### 5.1 单隐层 ReLU 网络的 Rademacher 界

考虑

$$
f(x)=\sum_{j=1}^m\eta_j(w_j^\top x+b_j)_+,
$$

假设

$$
\|x\|_2\le R,
\qquad
\|w_j\|_2^2+b_j^2/R^2=1,
\qquad
\|\eta\|_1\le D.
$$

若损失对预测值是 $G$-Lipschitz，则

$$
\mathbb E\left[
R(\widehat f)-\inf_{f\in\mathcal F}R(f)
\right]
\le \frac{16GDR}{\sqrt n}.
$$

推导主链：先用收缩不等式去掉损失，再利用 $\ell_1/\ell_\infty$ 对偶把所有神经元的优化化为单神经元上确界：

$$
\mathcal R_n(\ell\circ\mathcal F)
\le 2GD\,
\mathbb E\sup_{\|w\|^2+b^2/R^2=1}
\left|
\frac1n\sum_{i=1}^n\varepsilon_i(w^\top x_i+b)
\right|.
$$

由 Cauchy–Schwarz、Jensen 不等式以及 Rademacher 变量的独立零均值性质，

$$
\mathcal R_n(\ell\circ\mathcal F)
\le \frac{4GDR}{\sqrt n}.
$$

该界没有显式依赖隐藏单元数 $m$；控制估计误差的是权重范数，而不是裸参数数量。

### 5.2 ReLU 正齐次性与隐含 $\ell_1$ 正则

ReLU 满足 $(\alpha u)_+=\alpha(u)_+$，$\alpha>0$。缩放

$$
\eta\mapsto\alpha\eta,
\qquad
(w,b)\mapsto(w,b)/\alpha
$$

不改变神经元输出。平方 $\ell_2$ 权重惩罚可写为

$$
\alpha^2\eta^2+rac{\|w\|^2+b^2/R^2}{\alpha^2}.
$$

令 $c^2=\|w\|^2+b^2/R^2$，对 $\alpha$ 优化得到

$$
\boxed{
\min_{\alpha>0}
\left(\alpha^2\eta^2+\frac{c^2}{\alpha^2}\right)
=2|\eta|c.
}
$$

当 $c=1$ 时，它等价于输出权重的 $\ell_1$ 惩罚。

**来源：** Bach §9.2.2–9.2.3。
""",

r"""## 6. 有限样本表达能力、记忆容量与 VC 维

### 6.1 有限样本表达能力

给定 $N$ 个互异输入 $x_i\in\mathbb R^d$ 和任意目标 $y_i\in\mathbb R^p$，若存在参数 $\theta$ 使

$$
f_\theta(x_i)=y_i,
\qquad i=1,\ldots,N,
$$

则网络可以插值该数据集。若对任意数据集都成立，则网络具有 $N$ 点的普适有限样本表达能力。

记忆容量定义为

$$
\operatorname{MemCap}(\mathcal F)
=\max\left\{
N:\ \forall\{(x_i,y_i)\}_{i=1}^N,\ \exists\theta,
f_\theta(x_i)=y_i
\right\}.
$$

### 6.2 与 VC 维的区别

| 概念 | 输入点的量词 | 标签 | 含义 |
|---|---|---|---|
| 记忆容量 | 对所有互异输入点集 | 通常为实数或向量 | 最坏数据也能精确插值 |
| VC 维 | 存在一个输入点集 | 二元标签 | 该点集可实现所有二元标记 |

在标量二分类情形，

$$
\operatorname{MemCap}(\mathcal F)
\le \operatorname{VCdim}(\mathcal F).
$$

### 6.3 深度–宽度权衡

L16 汇总的结果包括：

- 单隐层 ReLU 网络用 $\Theta(N)$ 个隐藏单元可以记忆 $N$ 个任意样本；
- 三隐层充分条件之一为 $d_1d_2\ge4N$ 且 $d_3\ge4p$；
- 两隐层在 $d_1d_2\ge4Np$ 时可以记忆 $N$ 点、$p$ 维输出；
- 单隐层若 $d_1+2<N$，则存在无法记忆的 $N$ 点数据集；
- 两隐层标量输出若 $2d_1d_2+d_2+2<N$，也存在无法记忆的数据集；
- 增加深度可使多类问题的节点需求从 $\Theta(Np)$ 改善到 $\Theta(N+p)$。

### 6.4 VC 维上界

对 sign 激活的固定有向图，

$$
\operatorname{VCdim}(H_{V,E,\mathrm{sign}})
=O(|E|\log|E|).
$$

证明轮廓：每个神经元是半空间，网络是各层和各单元函数类的复合与乘积，因此增长函数满足

$$
\tau_H(m)\le(em)^{|E|}.
$$

若 $m$ 点被打散，则 $2^m\le(em)^{|E|}$，从而 $m=O(|E|\log|E|)$。

> **记忆不等于学习失败。** 记忆随机标签说明模型容量很大，也说明零训练误差不足以解释泛化；但真实任务中，数据结构、正则化以及优化算法的隐式偏置可能在众多插值解中选择更简单的解。

**来源：** L16 pp. 3–14；UML §20.4；Yun、Sra、Jadbabaie (2019)。
""",

r"""## 7. 经验风险与随机梯度下降

训练目标通常是正则化经验风险：

$$
\widehat R_N(\theta)
=\frac1N\sum_{i=1}^N\ell(y_i,F(x_i;\theta))
+\frac\lambda2\|\theta\|_2^2.
$$

随机选择样本 $i_t$ 或 mini-batch $B_t$，构造梯度估计 $g_t$。SGD 更新为

$$
\theta_{t+1}=\theta_t-\eta_tg_t,
\qquad
\mathbb E[g_t\mid\theta_t]=\nabla\widehat R_N(\theta_t).
$$

### 7.1 凸情形的基本收敛推导

假设 $F$ 凸、随机梯度无偏且 $\mathbb E\|g_t\|^2\le B^2$。平方距离递推为

$$
\begin{aligned}
\mathbb E\|\theta_{t+1}-\theta^*\|^2
&=\mathbb E\|\theta_t-\theta^*\|^2
-2\eta_t\mathbb E\langle\nabla F(\theta_t),\theta_t-\theta^*\rangle\\
&\quad+\eta_t^2\mathbb E\|g_t\|^2.
\end{aligned}
$$

凸性给出

$$
F(\theta_t)-F(\theta^*)
\le\langle\nabla F(\theta_t),\theta_t-\theta^*\rangle.
$$

若 $\|\theta^* -\theta_0\|\le D$，预先知道总步数 $T$ 并取常数步长

$$
\eta=\frac{D}{B\sqrt T},
$$

则平均迭代点满足

$$
\boxed{
\mathbb E[F(\bar\theta_T)-F(\theta^*)]
\le\frac{D^2}{2\eta T}+\frac{\eta B^2}{2}
=\frac{DB}{\sqrt T}.
}
$$

### 7.2 神经网络中的限制

神经网络目标高度非凸，上述凸收敛保证不能直接套用。一般光滑非凸分析通常只能保证某个迭代点的梯度范数较小，而不能保证到达全局最优。过参数化、特殊初始化和网络结构等附加条件可能带来更强结论。

实践中还常使用 momentum、Adam、梯度裁剪、学习率衰减和 early stopping。

**来源：** L16 pp. 15–22；Bach §5.4；UML §20.6。
""",

r"""## 8. 初始化：为什么 ReLU 使用 He 方差 $2/n$

零初始化会使同层单元保持完全对称；在某些 ReLU 结构中还会使激活和梯度持续为零。随机初始化的目的之一是打破对称性，另一个目的是保持信号尺度。

设某层有 $n$ 个独立输入 $a_j$，权重 $w_j$ 相互独立、零均值，并忽略偏置：

$$
z=\sum_{j=1}^n w_ja_j.
$$

在独立近似下，

$$
\operatorname{Var}(z)
=n\operatorname{Var}(w)\,\mathbb E[a^2].
$$

若 $z$ 的分布近似关于零对称，ReLU 保留约一半二阶矩：

$$
\mathbb E[\operatorname{ReLU}(z)^2]
\approx\frac12\mathbb E[z^2].
$$

希望下一层的二阶矩与当前层近似相同：

$$
\frac12n\operatorname{Var}(w)\mathbb E[a^2]
\approx\mathbb E[a^2].
$$

因此

$$
\boxed{\operatorname{Var}(w)=\frac2n.}
$$

常见选择是 $w_j\sim\mathcal N(0,2/n)$，或使用具有相同方差的均匀分布。对线性或 tanh 类型激活，类似推导给出 Xavier 尺度约 $1/n$；更细致的版本会同时考虑 fan-in 与 fan-out。

**来源：** L16 pp. 20–21；He et al. (2015)。
""",

r"""## 9. 反向传播：完整矩阵推导

### 9.1 前向传播

采用列向量约定：

$$
z^l=W^la^{l-1}+b^l,
\qquad
a^l=\sigma^l(z^l).
$$

定义误差信号

$$
\delta^l:=\frac{\partial\ell}{\partial z^l}.
$$

输出层误差由损失决定：

- 线性输出加平方损失：$\delta^L=a^L-y$；
- softmax 加交叉熵：$\delta^L=p-y$。

### 9.2 误差信号递推

因为

$$
z^{l+1}=W^{l+1}a^l+b^{l+1},
$$

所以

$$
\frac{\partial\ell}{\partial a^l}
=(W^{l+1})^\top\delta^{l+1}.
$$

再由 $a^l=\sigma(z^l)$，

$$
\boxed{
\delta^l
=\left[(W^{l+1})^\top\delta^{l+1}\right]
\odot\sigma'(z^l),
\qquad l=L-1,\ldots,1.
}
$$

等价的矩阵形式为

$$
\delta^l
=\operatorname{Diag}(\sigma'(z^l))
(W^{l+1})^\top\delta^{l+1}.
$$

### 9.3 参数梯度

对单个元素，

$$
z_i^l=\sum_jW_{ij}^la_j^{l-1}+b_i^l,
$$

因此

$$
\boxed{
\frac{\partial\ell}{\partial W^l}
=\delta^l(a^{l-1})^\top,
\qquad
\frac{\partial\ell}{\partial b^l}=\delta^l.
}
$$

### 9.4 单隐层 hinge-loss 示例

设

$$
z_i=w_i^\top x+b_i,
\qquad
h_i=[z_i]_+,
\qquad
s=\sum_i v_ih_i+c,
$$

$$
\ell(y,s)=\max(0,1-ys).
$$

则

$$
\frac{\partial\ell}{\partial s}
=-y\mathbf 1\{1-ys>0\},
$$

$$
\boxed{
\frac{\partial\ell}{\partial w_i}
=\big[-y\mathbf 1\{1-ys>0\}\big]
v_i\mathbf 1\{z_i>0\}x.
}
$$

四个乘子分别对应：损失边际门、输出权重、ReLU 门和输入。

### 9.5 伪代码

1. 设置 $a^0=x$；
2. 对 $l=1,\ldots,L$ 做前向传播；
3. 计算输出层 $\delta^L$；
4. 对 $l=L-1,\ldots,1$ 递推 $\delta^l$；
5. 计算 $G_W^l=\delta^l(a^{l-1})^\top$、$G_b^l=\delta^l$；
6. 在 mini-batch 上平均梯度并更新参数。

**来源：** L16 pp. 23–33；UML §20.6。
""",

r"""## 10. 梯度消失与梯度爆炸

把反向递推展开：

$$
\delta^l
=D^l(W^{l+1})^\top
D^{l+1}(W^{l+2})^\top
\cdots D^L\delta_{\mathrm{out}},
$$

其中

$$
D^r=\operatorname{Diag}(\sigma'(z^r)).
$$

取算子范数得到

$$
\|\delta^l\|
\le
\left(\prod_{r=l}^L\|D^r\|\right)
\left(\prod_{r=l+1}^L\|W^r\|\right)
\|\delta_{\mathrm{out}}\|.
$$

- 如果多数因子的范数小于 $1$，梯度随深度指数衰减；
- 如果多数因子的范数大于 $1$，梯度可能指数增长；
- sigmoid 饱和区的导数接近零，会加重梯度消失；
- ReLU 激活侧的导数为 $1$，但不能单独解决权重矩阵连乘问题。

| 方法 | 主要机制 |
|---|---|
| He/Xavier 初始化 | 保持初始激活与梯度尺度 |
| 梯度裁剪 | 直接限制爆炸梯度的范数 |
| 正交或规范化参数化 | 控制权重矩阵奇异值 |
| 残差连接 | 引入恒等路径，使 Jacobian 含 $I$ 项 |
| Batch/Layer Normalization | 稳定中间激活尺度并改变优化几何 |

**来源：** L16 pp. 36–37。
""",

r"""## 11. 显式正则化与 Dropout

### 11.1 权重衰减

加入平方 $\ell_2$ 惩罚：

$$
\widehat R_N(\theta)+\frac\lambda2\|\theta\|_2^2.
$$

普通 SGD 更新为

$$
\theta
\leftarrow
\theta-\eta\left(\nabla\widehat R_N(\theta)+\lambda\theta\right).
$$

这等价于每步把权重乘以 $1-\eta\lambda$，再减去数据梯度。在自适应优化器中，$\ell_2$ 正则化与解耦 weight decay 不一定等价。

### 11.2 Dropout

训练时对激活 $a$ 采样独立掩码

$$
r_j\sim\operatorname{Bernoulli}(q).
$$

现代 inverted dropout 写作

$$
\widetilde a=\frac{r\odot a}{q}.
$$

其条件期望保持不变：

$$
\mathbb E[\widetilde a\mid a]=a.
$$

因此测试时直接使用 $a$，无需再次缩放。反向传播为

$$
\frac{\partial\ell}{\partial a}
=\frac rq\odot
\frac{\partial\ell}{\partial\widetilde a}.
$$

被丢弃单元的局部梯度为零。Dropout 可被理解为训练大量共享参数的随机子网络，抑制脆弱的单元共适应，并近似执行模型平均。

原始 dropout 约定是在训练时不除以 $q$，测试时把相应贡献乘以 $q$；两种约定等价。

**来源：** L16 pp. 38–42；Srivastava et al. (2014)。
""",

r"""## 12. Batch Normalization 与完整梯度

### 12.1 前向变换

对 mini-batch 中某个标量特征 $x_1,\ldots,x_m$，定义

$$
\mu_B=\frac1m\sum_{i=1}^m x_i,
$$

$$
\sigma_B^2=\frac1m\sum_{i=1}^m(x_i-\mu_B)^2,
$$

$$
\widehat x_i
=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
y_i=\gamma\widehat x_i+\beta.
$$

$\gamma$ 与 $\beta$ 允许网络重新学习尺度和平移。训练时使用 batch 统计量；推理时通常使用移动平均得到的总体统计量。

### 12.2 $\gamma$ 与 $\beta$ 的梯度

令

$$
d_i:=\frac{\partial\ell}{\partial y_i},
\qquad
s:=\sqrt{\sigma_B^2+\varepsilon}.
$$

则

$$
\boxed{
\frac{\partial\ell}{\partial\beta}=\sum_i d_i,
\qquad
\frac{\partial\ell}{\partial\gamma}=\sum_i d_i\widehat x_i.
}
$$

### 12.3 输入梯度

必须同时考虑 $x_i$ 对自身、均值和方差的影响。展开链式法则并合并，得到

$$
\boxed{
\frac{\partial\ell}{\partial x_i}
=\frac{\gamma}{ms}
\left[
md_i-\sum_jd_j
-\widehat x_i\sum_jd_j\widehat x_j
\right].
}
$$

该公式表明 BN 的梯度在 batch 内耦合：一个样本的梯度依赖整批的 $d_j$ 与 $\widehat x_j$。

### 12.4 尺度不变性

忽略 $\varepsilon$ 且 $\alpha>0$ 时，

$$
\operatorname{BN}(\alpha x)=\operatorname{BN}(x),
$$

因此

$$
\operatorname{BN}(Wx)=\operatorname{BN}(\alpha Wx).
$$

这减弱了权重尺度对前向输出的影响，通常允许更大的学习率。

> **解释边界：** “减少 internal covariate shift”是 BN 原论文的动机，但不是其效果的唯一或最终解释。更稳妥的说法是：BN 进行了重参数化、改变了优化几何，并通过 batch 统计噪声带来一定正则化效应。

**来源：** L16 pp. 43–47；Ioffe、Szegedy (2015)。
""",

r"""## 13. 综合理解：表达、记忆、优化与泛化

| 主线 | 核心结论 | 不能过度推断 |
|---|---|---|
| 表达 | ReLU 网络能逼近广泛函数；深度可利用组合结构节省宽度 | 万能逼近不保证网络小或可训练 |
| 记忆 | 较小的深网也能插值任意有限数据，甚至随机标签 | 零训练误差不等于零测试误差 |
| 优化 | 反向传播高效计算梯度；SGD 在非凸地形中提供可行搜索 | 凸 SGD 收敛率不能无条件套到深网 |
| 泛化 | 范数、数据结构和算法隐式偏置比裸参数计数更关键 | 容量大不自动意味着泛化好或坏 |
| 稳定性 | 初始化、残差、归一化与裁剪控制信号及梯度尺度 | 单一技巧不是深度学习成功的完整解释 |

### 复习检查表

- [ ] 能从核 SVM 的预测式解释固定特征与学习特征的区别；
- [ ] 能写出任意深度前馈网络的 $z^l,a^l$ 递推并检查矩阵维度；
- [ ] 能从斜率变化推导一维分段仿射函数的 ReLU 表示；
- [ ] 能说明万能逼近、有限样本记忆与 VC 维分别使用什么量词；
- [ ] 能完整推导 $\delta^l$、$\partial\ell/\partial W^l$ 和 $\partial\ell/\partial b^l$；
- [ ] 能用方差守恒推导 He 初始化 $\operatorname{Var}(w)=2/n$；
- [ ] 能从矩阵乘积解释梯度消失与梯度爆炸；
- [ ] 能写出 inverted dropout 与 BatchNorm 的前向、反向公式；
- [ ] 能区分训练误差、经验风险、总体风险以及近似、估计、优化误差。
""",

r"""## 参考文献与材料对应

### 课程材料

- **Suvrit Sra, L15:** 四种神经网络视角、网络结构、ReLU、非线性分离、万能逼近、深度分离、VC 维备注。
- **Suvrit Sra, L16:** 有限样本记忆、SGD、初始化、反向传播、梯度不稳定、Dropout、Batch Normalization。

### 教材

1. Bach, F. *Learning Theory from First Principles*. MIT Press draft; 本讲义主要使用 Chapter 5 与 Chapter 9。
2. Shalev-Shwartz, S. & Ben-David, S. *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press, 2014; 本讲义主要使用 Chapter 20。

### 延伸文献

3. Cybenko, G. Approximation by superpositions of a sigmoidal function. 1989.
4. Leshno, M. et al. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. 1993.
5. He, K. et al. Delving Deep into Rectifiers. ICCV, 2015.
6. Srivastava, N. et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 2014.
7. Ioffe, S. & Szegedy, C. Batch Normalization. ICML, 2015.
8. Yun, C., Sra, S. & Jadbabaie, A. Small ReLU Networks Are Powerful Memorizers. NeurIPS, 2019.
9. Bartlett, P. et al. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. JMLR, 2019.
""",
]


def markdown_cell(text):
    # Keep line endings explicit and portable in the notebook JSON.
    lines = text.strip().splitlines()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


notebook = {
    "cells": [markdown_cell(section) for section in sections],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
        "title": TITLE,
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK.resolve())
print(f"markdown_cells={len(notebook['cells'])}")
