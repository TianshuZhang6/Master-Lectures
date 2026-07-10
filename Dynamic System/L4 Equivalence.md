# L4 Equivalence: Hartman-Grobman and Local Topological Conjugacy

本讲义基于 Section 4 **Equivalence**，并结合前面笔记中关于 $h(Ax)=g(h(x))$、time-one map、范数估计、Neumann 级数、$\mathcal L^{-1}$ 估计等问题的讨论。

## 0. L4 的核心主题

L4 的主题是动力系统的 **等价性** 和 **局部拓扑线性化**。

核心问题是：

$$
\boxed{\text{在双曲不动点附近，非线性系统是否与其线性化系统具有相同的轨道结构？}}
$$

对于离散映射，写成

$$
g(x)=Ax+R(x),
$$

其中 $A$ 是线性部分，$R(x)$ 是非线性扰动。我们希望找到一个 homeomorphism

$$
h(x)=x+H(x)
$$

使得

$$
\boxed{h\circ A=g\circ h.}
$$

等价地，

$$
\boxed{h(Ax)=g(h(x)).}
$$

这正是 Hartman-Grobman 型定理的局部版本：双曲性保证非线性系统在拓扑意义下像它的线性化系统。

## 1. Topological equivalence and conjugacy

Definition 4.1 讨论两个动力系统

$$
(X,T,\phi_t)
\qquad\text{and}\qquad
(Y,T,\psi_t)
$$

之间的关系。

如果存在 homeomorphism

$$
h:X\to Y
$$

把第一个系统的轨道映到第二个系统的轨道，并保持时间方向，则称两个系统 **topologically equivalent**。

如果还保持时间参数，也就是说

$$
h(\phi_t(x))=\psi_t(h(x))
$$

对所有允许的 $t$ 成立，则称两个系统 **conjugate**。

如果 $h$ 和 $h^{-1}$ 都是 $C^k$，则称为 $C^k$-equivalence 或 $C^k$-conjugacy。

直观上：

$$
\boxed{\text{equivalence 保持轨道形状；conjugacy 还保持时间节奏。}}
$$

## 2. Discrete maps: conjugacy equation

Proposition 4.2 说，对于两个离散映射

$$
x\mapsto f(x),
\qquad
 y\mapsto g(y),
$$

它们 topologically conjugate 当且仅当存在 homeomorphism

$$
h:\mathbb R^d\to\mathbb R^d
$$

使得

$$
f=h^{-1}\circ g\circ h.
$$

等价地，

$$
h\circ f=g\circ h.
$$

如果第一个系统先走一步 $f$，再用 $h$ 映过去，等于先用 $h$ 映过去，再在第二个系统中走一步 $g$。

这就是交换图：

$$
\begin{array}{ccc}
x & \xrightarrow{f} & f(x) \\
h\downarrow & & \downarrow h \\
h(x) & \xrightarrow{g} & g(h(x))
\end{array}
$$

对于 L4 的证明，$f=A$，所以共轭方程变成

$$
\boxed{h(Ax)=g(h(x)).}
$$

## 3. Continuous flows: conjugacy and vector fields

Proposition 4.3 讨论连续时间系统

$$
x'=f(x),
\qquad
 y'=g(y).
$$

如果

$$
y=h(x)
$$

是 $C^1$-diffeomorphism，则沿轨道求导：

$$
y'=Dh(x)x'=Dh(x)f(x).
$$

另一方面，第二个系统要求

$$
y'=g(y)=g(h(x)).
$$

所以必须有

$$
g(h(x))=Dh(x)f(x).
$$

等价地，

$$
\boxed{f(x)=(Dh)^{-1}(x)g(h(x)).}
$$

这说明，连续系统的光滑共轭不仅要匹配轨道，还要匹配向量场方向和速度。

## 4. Local topological equivalence

Definition 4.4 把等价性局部化。

如果系统在平衡点 $x_*$ 附近与另一个系统在平衡点 $y_*$ 附近 topologically equivalent，则称它们在该平衡点附近 **locally topologically equivalent**。

也就是说，只需要在某些邻域

$$
U(x_*),\qquad V(y_*)
$$

中存在 homeomorphism

$$
h:U(x_*)\to V(y_*),
$$

把局部轨道结构对应起来。

Hartman-Grobman 定理正是这种局部结论。

## 5. Hartman-Grobman theorem

Theorem 4.5 说：考虑 ODE

$$
x'=f(x),
\qquad
x\in\mathbb R^d,
\qquad
f\in C^1.
$$

设 $x_*$ 是双曲平衡点，即

$$
A=Df(x_*)
$$

没有特征值落在虚轴上。则非线性 flow $\phi_t$ 在 $x_*$ 附近与线性化系统

$$
x'=Ax
$$

的 flow

$$
e^{tA}
$$

局部拓扑共轭。

核心意思是：

$$
\boxed{\text{双曲平衡点附近，非线性系统的轨道结构由线性化决定。}}
$$

注意：这是拓扑共轭，一般不是 $C^1$ 共轭。

## 6. Map version: Theorem 4.6

PDF 中主要证明 map 版本。设

$$
g(x)=Ax+R(x),
\qquad
x\in\mathbb R^d,
$$

其中 $g\in C^1$ 且可逆，$0$ 是双曲不动点。

Theorem 4.6 说，存在唯一的 homeomorphism

$$
h(x)=x+H(x),
\qquad
H(0)=0,
\qquad
H\text{ bounded},
$$

使得在 $0$ 的某个邻域 $U$ 中

$$
\boxed{h\circ A=g\circ h.}
$$

也就是说，$g$ 在局部拓扑共轭于它的线性部分 $A$。

## 7. Why $h(Ax)=g(h(x))$?

共轭方程

$$
h(Ax)=g(h(x))
$$

表示：

先按线性系统走一步，再用 $h$ 送到非线性系统；等于先用 $h$ 送到非线性系统，再按非线性映射走一步。

因此 $h$ 不只是普通换坐标，而是把线性系统的轨道对应到非线性系统的轨道。

如果写成复合映射形式，就是

$$
h\circ A=g\circ h.
$$

也可等价写作

$$
A=h^{-1}\circ g\circ h.
$$

证明中选择 $h\circ A=g\circ h$ 是因为代入

$$
h(x)=x+H(x)
$$

后能得到适合 fixed point 方法的方程。

## 8. Deriving the equation for $H$

由

$$
h(x)=x+H(x)
$$

和

$$
g(x)=Ax+R(x),
$$

代入共轭方程

$$
h(Ax)=g(h(x)).
$$

左边为

$$
h(Ax)=Ax+H(Ax).
$$

右边为

$$
g(h(x))=g(x+H(x))=A(x+H(x))+R(x+H(x)).
$$

所以

$$
Ax+H(Ax)=Ax+AH(x)+R(x+H(x)).
$$

消去 $Ax$，得到

$$
\boxed{H(Ax)-AH(x)=R(x+H(x)).}
$$

定义线性算子

$$
\mathcal L H:=H(Ax)-AH(x),
$$

则方程变为

$$
\mathcal L H=R(x+H(x)).
$$

## 9. Fixed point formulation

如果 $\mathcal L^{-1}$ 存在，则

$$
H=\mathcal L^{-1}R(x+H(x)).
$$

定义

$$
\mathcal T(H):=\mathcal L^{-1}R(x+H(x)).
$$

于是要求解

$$
\boxed{H=\mathcal T(H).}
$$

这里 fixed point 不是说 $x$ 固定，也不是说 $H(x)$ 是固定点。

而是在函数空间中找一个函数 $H$，使得

$$
\mathcal T(H)=H.
$$

找到这个修正函数 $H$ 后，

$$
h=\mathrm{Id}+H
$$

就给出拓扑共轭。

## 10. Hyperbolic splitting and the key norm choice

由于 $0$ 是双曲不动点，$A$ 的谱分为 stable 和 unstable 两部分。对应分解为

$$
\mathbb R^d=E^s\oplus E^u.
$$

记

$$
A_s=A|_{E^s},
\qquad
A_u=A|_{E^u}.
$$

证明中选择合适的矩阵范数，使得

$$
\alpha:=\max\left(\|A_u^{-1}\|,\|A_s\|\right)<1.
$$

这正是双曲性的核心作用：

$$
\boxed{\text{stable 方向正向收缩，unstable 方向反向收缩。}}
$$

同时通过 cut-off 把 $R$ 局部化，使得全局 Lipschitz 常数足够小：

$$
|R(x)-R(y)|\le \varepsilon |x-y|,
\qquad
\varepsilon<\frac{1-\alpha}{2}.
$$

## 11. Why $A_u$ seems to disappear

在 unstable 方向上，正向迭代会膨胀，所以不能用 $A_u$ 的正向幂做收缩估计。

因为 unstable 特征值满足

$$
|\lambda|>1,
$$

所以 $A_u$ 本身可能放大向量。但

$$
A_u^{-1}
$$

在反向时间中是收缩的。

因此估计中真正出现的是

$$
\|A_u^{-1}\|<1,
$$

而不是

$$
\|A_u\|<1.
$$

所以 $A_u$ 的范数并不是消失了，而是通过“反向解 unstable 方向”转化成了 $A_u^{-1}$ 的范数。

## 12. The operator $\mathcal L$ and its inverse

考虑 Banach 空间

$$
C(\mathbb R^d,\mathbb R^d)
$$

配备 supremum norm。

定义两个线性算子

$$
(KH)(x):=H(Ax),
\qquad
(\mathcal AH)(x):=AH(x).
$$

则

$$
\mathcal L=K-\mathcal A.
$$

沿 stable/unstable 分解，

$$
C(\mathbb R^d,\mathbb R^d)
=
C(\mathbb R^d,E^s)\oplus C(\mathbb R^d,E^u),
$$

并有

$$
\mathcal L=\mathcal L_s\oplus\mathcal L_u.
$$

在 unstable 部分，

$$
\mathcal L_u=K_u-A_u=-A_u(I-A_u^{-1}K_u).
$$

若

$$
\|A_u^{-1}K_u\|<1,
$$

则可以用 Neumann 级数构造逆。

## 13. Neumann series

Neumann 级数的标准形式是

$$
(I-B)^{-1}=I+B+B^2+\cdots,
$$

只要

$$
\|B\|<1.
$$

在这里，取

$$
B=A_u^{-1}K_u.
$$

因为

$$
\|A_u^{-1}K_u\|\le \alpha<1,
$$

所以

$$
(I-A_u^{-1}K_u)^{-1}
=
\sum_{k=0}^\infty (A_u^{-1}K_u)^k.
$$

于是

$$
\mathcal L_u^{-1}
=-(I-A_u^{-1}K_u)^{-1}A_u^{-1}.
$$

stable 部分类似处理，最终得到估计

$$
\boxed{\|\mathcal L^{-1}\|\le \frac{2}{1-\alpha}.}
$$

这就是为什么 L4 中反复要证明某些算子范数小于 $1$：这样才能用 Neumann 级数构造逆算子。

## 14. Contraction estimate

有了

$$
H=\mathcal L^{-1}R(x+H(x)),
$$

定义

$$
\mathcal T(H)=\mathcal L^{-1}R(x+H(x)).
$$

对两个函数 $H_1,H_2$，有

$$
\begin{aligned}
\|\mathcal T(H_1)-\mathcal T(H_2)\|
&\le \|\mathcal L^{-1}\|\,
\|R(\mathrm{Id}+H_1)-R(\mathrm{Id}+H_2)\|\\
&\le \frac{2}{1-\alpha}\,\varepsilon\,
\|H_1-H_2\|.
\end{aligned}
$$

由于

$$
\varepsilon<\frac{1-\alpha}{2},
$$

所以

$$
\frac{2\varepsilon}{1-\alpha}<1.
$$

因此 $\mathcal T$ 是压缩映射。

由 Banach fixed point theorem，存在唯一的连续有界 $H$，从而得到

$$
h=\mathrm{Id}+H.
$$

## 15. Why $h$ is invertible

证明还需要说明

$$
h=\mathrm{Id}+H
$$

是 homeomorphism。

PDF 中的思路是构造反向共轭映射

$$
\widetilde h(x)=x+\widetilde H(x)
$$

满足

$$
A\circ\widetilde h=\widetilde h\circ g.
$$

这相当于交换 $A$ 和 $g$ 的角色后重复同样的论证。

由

$$
h\circ A=g\circ h
$$

和

$$
A\circ\widetilde h=\widetilde h\circ g
$$

可得

$$
A\circ\widetilde h\circ h
=
\widetilde h\circ g\circ h
=
\widetilde h\circ h\circ A.
$$

利用线性情形下共轭的唯一性，可以推出

$$
\widetilde h\circ h=\mathrm{Id}
\qquad\text{and}\qquad
h\circ\widetilde h=\mathrm{Id}.
$$

所以

$$
\widetilde h=h^{-1}.
$$

## 16. Why $H(0)=0$

证明还需要检查 $H(0)=0$。

由反向共轭关系可以得到

$$
Ah^{-1}(0)=h^{-1}(0).
$$

也就是

$$
(A-I)h^{-1}(0)=0.
$$

这意味着 $h^{-1}(0)$ 是 $A$ 对应 multiplier $1$ 的特征向量。

但 $0$ 是双曲不动点，所以 $A$ 没有模长为 $1$ 的特征值。

因此只能有

$$
h^{-1}(0)=0.
$$

从而

$$
h(0)=0,
\qquad
H(0)=0.
$$

## 17. Flow case and the time-one map

对于连续系统

$$
x'=f(x),
$$

其 flow 记作

$$
\phi_t(x).
$$

为了证明 Hartman-Grobman 定理，PDF 中使用 **time-one map**：

$$
g(x)=\phi_1(x).
$$

如果线性化系统是

$$
x'=Ax,
$$

则它的 time-one map 是

$$
x\mapsto e^Ax.
$$

因此连续时间问题可以先转化成离散映射问题：

$$
\phi_1(x)=e^Ax+R(x).
$$

应用 Theorem 4.6 得到

$$
\boxed{h\circ e^A=\phi_1\circ h.}
$$

也就是

$$
h(e^Ax)=\phi_1(h(x)).
$$

## 18. Why $e^{sA}e^{-sA}=I$

矩阵指数满足

$$
e^{sA}e^{tA}=e^{(s+t)A},
$$

因为这里两个矩阵都是 $A$ 的倍数，所以它们交换。

取 $t=-s$，得到

$$
e^{sA}e^{-sA}=e^{0A}=I.
$$

从动力系统角度看：先沿线性 flow 走时间 $s$，再走时间 $-s$，就回到原点。

## 19. Extending from time-one conjugacy to flow conjugacy

PDF 中定义

$$
h_s:=\phi_s\circ h\circ e^{-sA}.
$$

利用

$$
h\circ e^A=\phi_1\circ h,
$$

可以检查 $h_s$ 也满足同一个 time-one 共轭方程。

由 Theorem 4.6 的唯一性，得到

$$
h_s\equiv h.
$$

因此

$$
\phi_s\circ h\circ e^{-sA}=h.
$$

右乘 $e^{sA}$，得到

$$
\boxed{h\circ e^{sA}=\phi_s\circ h.}
$$

这就从 time-one map 的共轭推广到了整个 flow 的共轭。

## 20. Overall proof logic

L4 的证明逻辑可以概括为：

$$
\text{双曲性}
\Longrightarrow
\text{stable/unstable 方向上的收缩估计}
\Longrightarrow
\mathcal L^{-1}\text{ 有界}
\Longrightarrow
\text{fixed point 方程}
\Longrightarrow
\text{压缩映射定理得到 }H
\Longrightarrow
h=\mathrm{Id}+H\text{ 给出拓扑共轭}.
$$

更具体地：

1. 分离线性和非线性部分：

$$
g(x)=Ax+R(x).
$$

2. 寻找共轭映射：

$$
h(x)=x+H(x).
$$

3. 代入共轭方程：

$$
h(Ax)=g(h(x)).
$$

4. 得到修正项方程：

$$
H(Ax)-AH(x)=R(x+H(x)).
$$

5. 定义线性算子：

$$
\mathcal L H=H(Ax)-AH(x).
$$

6. 写成 fixed point：

$$
H=\mathcal L^{-1}R(x+H(x)).
$$

7. 用 Neumann 级数证明 $\mathcal L^{-1}$ 存在并有界。

8. 用范数估计证明 fixed point operator 是压缩映射。

9. 得到唯一 $H$，于是得到局部拓扑共轭 $h=\mathrm{Id}+H$。

## 21. One-sentence summary

L4 的核心不是单纯讲 Neumann 级数，而是用 Neumann 级数和范数估计证明关键线性算子可逆，并控制非线性扰动，从而构造出

$$
h=\mathrm{Id}+H
$$

证明双曲不动点附近的非线性系统与线性化系统拓扑共轭。

也就是：

$$
\boxed{
\text{双曲性提供收缩估计，Neumann 级数提供逆算子，fixed point 方法构造共轭。}
}
$$
