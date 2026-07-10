# Section 5 笔记：Poincaré Map, Multipliers, Monodromy Matrix, Suspension Flow

## 1. 从周期轨道到 Poincaré 截面

考虑自治系统

$$
x'=f(x), \qquad x\in \mathbb R^d,\quad f\in C^1.
$$

假设它有一条周期轨道

$$
\Gamma(t+T)=\Gamma(t),
$$

其中 $T>0$ 是周期。取周期轨道上的一点

$$
y\in \Gamma.
$$

因为 $y$ 在周期轨道上，所以它不是平衡点，因此

$$
f(y)\neq 0.
$$

这里 $f(y)$ 是轨道经过 $y$ 时的速度方向，也就是周期轨道在 $y$ 点的切向方向。

为了研究周期轨道附近的稳定性，Poincaré map 的思想是：不直接看整条连续轨道，而是选一个横截面，只记录轨道每次回到这个截面的位置。

讲义中取截面

$$
\Sigma:=\{x\in\mathbb R^d:(x-y)\cdot f(y)=0\}.
$$

这个式子的意思是：$x-y$ 与 $f(y)$ 垂直。因此 $\Sigma$ 是经过 $y$，并且垂直于周期轨道切向方向 $f(y)$ 的 $(d-1)$ 维超平面。

所以 $\Sigma$ 是一个 Poincaré section。它和周期轨道在 $y$ 点横截相交，也就是 transversal intersection。这里的 transversally 表示轨道方向 $f(y)$ 不在截面的切空间里，而是穿过截面。

---

## 2. 返回时间函数 $\tau(x)$

Poincaré map 定义为

$$
P(x):=\phi_{\tau(x)}(x).
$$

这里 $\phi_t(x)$ 表示从初始点 $x$ 出发，沿着 flow 走时间 $t$ 后到达的位置。

而 $\tau(x)$ 是返回时间函数。它表示：从点 $x$ 出发，沿着 flow 走，第一次在 $T$ 附近重新落到截面 $\Sigma$ 所需要的时间。

所以 $\tau(x)$ 不是常数，而是依赖于初始点 $x$ 的函数：

$$
\tau:B(y;\delta)\to \mathbb R.
$$

特别地，当 $x=y$ 时，因为 $y$ 在周期轨道上，并且

$$
\phi_T(y)=y,
$$

所以从 $y$ 出发绕一圈后回到 $y$，也重新回到截面 $\Sigma$。因此

$$
\tau(y)=T.
$$

虽然 $\phi_0(y)=y\in\Sigma$，所以 $t=0$ 也让轨道位于截面上，但 Poincaré map 研究的是“下一次返回”，所以选的是 $T$ 附近的返回时间，而不是 $0$。

---

## 3. 为什么 $\tau(x)$ 存在：隐函数定理

定义

$$
F(t,x):=(\phi_t(x)-y)\cdot f(y).
$$

注意：

$$
\phi_t(x)\in \Sigma
$$

当且仅当

$$
(\phi_t(x)-y)\cdot f(y)=0.
$$

也就是

$$
F(t,x)=0.
$$

所以求返回时间 $\tau(x)$，本质上就是解方程 $F(t,x)=0$ 中关于 $t$ 的解。

因为

$$
\phi_T(y)=y,
$$

所以

$$
F(T,y)=(\phi_T(y)-y)\cdot f(y)=0.
$$

接着对 $t$ 求偏导：

$$
\partial_tF(T,y)
=
\partial_t\phi_t(y)|_{t=T}\cdot f(y).
$$

由于

$$
\partial_t\phi_t(y)=f(\phi_t(y)),
$$

所以

$$
\partial_tF(T,y)
=
f(\phi_T(y))\cdot f(y).
$$

而

$$
\phi_T(y)=y,
$$

因此

$$
\partial_tF(T,y)=f(y)\cdot f(y)=\|f(y)\|^2>0.
$$

这个非零条件正是隐函数定理需要的条件。因此可以在 $y$ 附近解出一个 $C^1$ 函数 $\tau(x)$，使得

$$
F(\tau(x),x)=0.
$$

也就是说

$$
\phi_{\tau(x)}(x)\in\Sigma.
$$

于是 Poincaré map

$$
P(x)=\phi_{\tau(x)}(x)
$$

在局部是良定义的，而且是 $C^1$ 的。

---

## 4. 为什么可以写 $P(0)=0$

原来周期轨道上的点是

$$
y\in\Gamma.
$$

因为从 $y$ 出发经过一个周期回到自己，所以

$$
P(y)=y.
$$

但是研究局部稳定性时，我们会在截面 $\Sigma$ 上重新选局部坐标

$$
x=(x_1,\dots,x_{d-1}).
$$

这个 $x$ 不是原来 $\mathbb R^d$ 中的全局坐标，而是截面 $\Sigma$ 上的局部坐标。选坐标时，可以把 $y$ 作为坐标原点：

$$
y\longleftrightarrow 0.
$$

所以原来的 $P(y)=y$ 在新的局部坐标中就写成

$$
P(0)=0.
$$

因此这里的 $0$ 不是原系统的平衡点，而是周期轨道上的点 $y$ 在截面坐标中的表示。

严格写应该是：选一个局部坐标图

$$
\psi:\Sigma\supset U\to\mathbb R^{d-1},
$$

使得

$$
\psi(y)=0.
$$

然后新的 Poincaré map 是

$$
\widetilde P=\psi\circ P\circ\psi^{-1}.
$$

于是

$$
\widetilde P(0)=0.
$$

讲义为了简化符号，仍然把 $\widetilde P$ 写成 $P$。

---

## 5. Multipliers：周期轨道的横向线性稳定性

在截面坐标中，$P$ 有一个不动点

$$
P(0)=0.
$$

对它线性化：

$$
A=DP(0).
$$

矩阵 $A$ 的特征值

$$
\mu_1,\dots,\mu_{d-1}
$$

称为周期轨道的 multipliers。

因为 Poincaré map 定义在 $(d-1)$ 维截面上，所以这里只得到 $d-1$ 个 multipliers。它们描述的是周期轨道附近的横向扰动每绕一圈之后如何变化。

如果某个方向的 multiplier 满足

$$
|\mu|<1,
$$

则该横向扰动每次返回截面时都会缩小。

如果

$$
|\mu|>1,
$$

则该横向扰动会放大。

所以分析周期轨道的局部稳定性，可以转化为分析 Poincaré map 的不动点稳定性。

---

## 6. 为什么 multipliers 不依赖截面、点和局部坐标

假设选了两组不同的数据：

$$
y,\Sigma,x
$$

和

$$
\widetilde y,\widetilde\Sigma,\widetilde x.
$$

它们给出两个 Poincaré maps：

$$
P,\qquad \widetilde P.
$$

虽然这两个 map 的表达式可能不同，但它们描述的是同一条周期轨道附近的返回结构。因此它们局部共轭：

$$
P=h^{-1}\circ \widetilde P\circ h.
$$

这里 $h(x)=\widetilde x$ 表示把第一组截面坐标转换到第二组截面坐标的局部微分同胚。

对上式在 $x=0$ 处求导，用链式法则：

$$
DP(0)
=
D(h^{-1})(\widetilde P(h(0)))D\widetilde P(h(0))Dh(0).
$$

由于两组坐标都把各自周期轨道上的交点放在原点，所以

$$
h(0)=0,
$$

并且

$$
\widetilde P(0)=0.
$$

因此

$$
DP(0)
=
D(h^{-1})(0)D\widetilde P(0)Dh(0).
$$

令

$$
A=DP(0),
\qquad
\widetilde A=D\widetilde P(0),
\qquad
B=Dh(0).
$$

因为 $h$ 是微分同胚，

$$
D(h^{-1})(0)=B^{-1}.
$$

所以

$$
A=B^{-1}\widetilde A B.
$$

这说明 $A$ 和 $\widetilde A$ 是相似矩阵，因此有相同特征值。于是 multipliers 不依赖于 $y$、$\Sigma$ 以及局部坐标。

这里两个 $0$ 实际上不一定是原相空间里的同一个点。第一个 $0$ 表示第一组截面坐标中的 $y$，第二个 $0$ 表示第二组截面坐标中的 $\widetilde y$。只是两组局部坐标都把各自的周期轨道交点记作原点。

---

## 7. 变分方程：沿周期轨道线性化扰动

为了研究周期轨道附近的稳定性，令

$$
x(t)=\Gamma(t)+u(t),
$$

其中 $u(t)$ 是小扰动。

原系统为

$$
x'=f(x).
$$

因为 $\Gamma(t)$ 是原系统的一条解，所以

$$
\Gamma'(t)=f(\Gamma(t)).
$$

于是

$$
u'
=
x'-\Gamma'
=
f(\Gamma(t)+u)-f(\Gamma(t)).
$$

对 $f$ 在 $\Gamma(t)$ 附近做 Taylor 展开：

$$
f(\Gamma(t)+u)
=
f(\Gamma(t))+Df(\Gamma(t))u+o(|u|).
$$

因此

$$
u'=Df(\Gamma(t))u+o(|u|).
$$

忽略高阶项，得到线性变分方程：

$$
u'=A(t)u,
$$

其中

$$
A(t)=Df(\Gamma(t)).
$$

这里 $A(t)$ 一般依赖于 $t$，因为它是沿着周期轨道 $\Gamma(t)$ 取 Jacobian。

---

## 8. Fundamental solution 和 monodromy matrix

对线性系统

$$
u'=A(t)u,
$$

一个 fundamental solution $M(t)$ 满足

$$
M(0)=I,
$$

$$
M'(t)=A(t)M(t).
$$

这表示 $M(t)$ 是把初始扰动推进到时间 $t$ 的线性算子。

如果初始扰动是 $u(0)$，那么解为

$$
u(t)=M(t)u(0).
$$

这里必须注意：$u(0)$ 是常向量，不随 $t$ 变化。所以代入时

$$
\frac{d}{dt}\bigl(M(t)u(0)\bigr)=M'(t)u(0).
$$

因为

$$
M'(t)=A(t)M(t),
$$

所以

$$
\frac{d}{dt}\bigl(M(t)u(0)\bigr)
=
A(t)M(t)u(0)
=
A(t)u(t).
$$

因此

$$
u(t)=M(t)u(0)
$$

确实是变分方程的解。

如果误写成 $M(t)u(t)$，那就会出现乘积法则的第二项：

$$
M'(t)u(t)+M(t)u'(t).
$$

这不是这里的表达式。正确表达始终是

$$
u(t)=M(t)u(0).
$$

当 $t=T$ 时，

$$
u(T)=M(T)u(0).
$$

矩阵 $M(T)$ 称为 monodromy matrix。它表示扰动绕周期轨道一圈后如何变化。

---

## 9. 为什么 $D_x\phi(T,y)=M(T)$

流写作

$$
\phi(t,x).
$$

这里有两个变量：

$$
t=\text{时间},
\qquad
x=\text{初始位置}.
$$

所以

$$
D_x\phi(T,y)
$$

表示固定时间 $T$，对初始位置 $x$ 在 $y$ 处求导。

它不是对时间求导，而是对位置参数求导。它描述的是：如果初始点 $y$ 发生一个小扰动 $v$，那么时间 $T$ 后这个扰动变成什么。

令

$$
u(t)=D_x\phi(t,y)v.
$$

因为

$$
\frac{d}{dt}\phi(t,x)=f(\phi(t,x)),
$$

对初始位置 $x$ 求导得到

$$
\frac{d}{dt}D_x\phi(t,y)
=
Df(\phi(t,y))D_x\phi(t,y).
$$

沿周期轨道有

$$
\phi(t,y)=\Gamma(t),
$$

所以

$$
Df(\phi(t,y))=Df(\Gamma(t))=A(t).
$$

因此

$$
\frac{d}{dt}D_x\phi(t,y)
=
A(t)D_x\phi(t,y).
$$

同时

$$
D_x\phi(0,y)=I.
$$

这和 $M(t)$ 的定义完全相同：

$$
M'(t)=A(t)M(t),
\qquad
M(0)=I.
$$

由矩阵 ODE 解的唯一性，

$$
D_x\phi(t,y)=M(t).
$$

所以

$$
D_x\phi(T,y)=M(T).
$$

这说明 monodromy matrix 本质上就是 flow 对初始位置微分后，在一个周期 $T$ 处得到的矩阵。

---

## 10. $DP(y)$ 与 $M(T)$ 的关系

Poincaré map 是

$$
P(x)=\phi(\tau(x),x).
$$

这里 $P$ 同时依赖于初始位置 $x$ 和返回时间 $\tau(x)$。

所以对 $x$ 求导时，要用链式法则：

$$
DP(y)
=
\partial_t\phi(T,y)(\nabla\tau(y))^T+D_x\phi(T,y).
$$

由于

$$
\partial_t\phi(T,y)=f(\phi(T,y))=f(y),
$$

并且

$$
D_x\phi(T,y)=M(T),
$$

得到

$$
DP(y)=f(y)(\nabla\tau(y))^T+M(T).
$$

所以

$$
M(T)=DP(y)-f(y)(\nabla\tau(y))^T.
$$

关键区别是：

- $D_x\phi(T,y)$ 是固定时间 $T$ 时 flow 对初始位置的导数；
- $DP(y)$ 是 Poincaré map 的导数，它还包含返回时间 $\tau(x)$ 对 $x$ 的变化。

因此二者一般不完全相同，中间多了

$$
f(y)(\nabla\tau(y))^T.
$$

---

## 11. Rectification theorem 的作用

Rectification theorem，也叫向量场拉直定理。

它说：如果

$$
f(y)\neq 0,
$$

那么在 $y$ 附近可以选局部坐标，使得向量场变成

$$
f(x)=(0,\dots,0,1)^T.
$$

这表示局部上轨道沿最后一个坐标方向匀速前进。

于是可以选坐标

$$
(x_1,\dots,x_{d-1},z),
$$

其中 $(x_1,\dots,x_{d-1})$ 是截面方向，$z$ 是沿 flow 的方向。

在这个坐标下，Poincaré map 只作用在前 $d-1$ 个坐标上，因为返回截面时我们只关心横向坐标的位置。

---

## 12. 为什么 $M(T)$ 可以写成分块矩阵

在坐标

$$
(x_1,\dots,x_{d-1},z)
$$

下，整个相空间是 $d$ 维的，所以 $M(T)$ 是一个 $d\times d$ 矩阵。

把前 $d-1$ 个截面方向和最后一个流方向分开，就可以写成分块形式：

$$
M(T)=
\begin{pmatrix}
D_xP(y)&v_1\\
v_2^T&m
\end{pmatrix}.
$$

这里 $D_xP(y)$ 是 Poincaré map 对截面坐标的导数，是一个 $(d-1)\times(d-1)$ 矩阵。

而 $v_1$、$v_2$、$m$ 表示截面方向和流方向之间的其他耦合项。

这里的微分仍然是对位置参数的微分，不是对时间微分。区别只是：

- $D_x\phi(T,y)$ 是对整个 $d$ 维初始位置求导；
- $D_xP(y)$ 是对截面上的 $(d-1)$ 维初始位置求导。

因此 $D_xP(y)$ 自然出现在 $M(T)$ 的左上角。

---

## 13. 为什么 $M(T)$ 总有特征值 $1$

周期轨道本身的切向方向是一个特殊扰动。

因为 $\Gamma(t)$ 是原系统

$$
x'=f(x)
$$

的一条解，所以

$$
\Gamma'(t)=f(\Gamma(t)).
$$

这里不能简单写成 $f(x)$，因为 $x$ 是抽象状态变量；沿具体轨道时，当前位置是

$$
x=\Gamma(t).
$$

所以必须写成

$$
\Gamma'(t)=f(\Gamma(t)).
$$

继续对 $t$ 求导：

$$
\Gamma''(t)
=
\frac{d}{dt}f(\Gamma(t))
=
Df(\Gamma(t))\Gamma'(t).
$$

令

$$
A(t)=Df(\Gamma(t)),
$$

得到

$$
\Gamma''(t)=A(t)\Gamma'(t).
$$

这一步不能化成 $A(t)^2$。因为 $A(t)$ 是矩阵，而 $\Gamma'(t)$ 是向量。表达式 $A(t)\Gamma'(t)$ 是矩阵乘向量；而 $A(t)^2$ 是矩阵乘矩阵，类型不对。

上式说明 $\Gamma'(t)$ 本身满足变分方程

$$
u'=A(t)u.
$$

因此根据 fundamental solution，

$$
\Gamma'(T)=M(T)\Gamma'(0).
$$

另一方面，因为 $\Gamma$ 是周期轨道，

$$
\Gamma(t+T)=\Gamma(t).
$$

对 $t$ 求导：

$$
\Gamma'(t+T)=\Gamma'(t).
$$

取 $t=0$，得到

$$
\Gamma'(T)=\Gamma'(0).
$$

所以

$$
M(T)\Gamma'(0)=\Gamma'(0).
$$

又因为

$$
y=\Gamma(0),
$$

所以

$$
\Gamma'(0)=f(\Gamma(0))=f(y).
$$

因此

$$
M(T)f(y)=f(y).
$$

这说明 $f(y)$ 是 $M(T)$ 的特征向量，对应特征值

$$
1.
$$

这个特征值 $1$ 来自沿周期轨道切向方向的扰动。沿切向方向稍微移动一点，相当于在同一条周期轨道上改变相位；绕一圈之后仍然是同样大小的切向扰动。因此这个 $1$ 是平凡 multiplier，不反映横向稳定性。

---

## 14. 为什么 $v_1=0,\ m=1$

在 rectification 坐标中，

$$
f(y)=e_d=(0,\dots,0,1)^T.
$$

而

$$
M(T)=
\begin{pmatrix}
D_xP(y)&v_1\\
v_2^T&m
\end{pmatrix}.
$$

计算

$$
M(T)f(y)=M(T)e_d.
$$

因为 $e_d$ 的前 $d-1$ 个分量是 $0$，最后一个分量是 $1$，所以矩阵乘法给出

$$
M(T)e_d=
\begin{pmatrix}
v_1\\
m
\end{pmatrix}.
$$

但是上一节已经证明

$$
M(T)f(y)=f(y).
$$

也就是

$$
M(T)e_d=e_d.
$$

所以

$$
\begin{pmatrix}
v_1\\
m
\end{pmatrix}
=
\begin{pmatrix}
0\\
1
\end{pmatrix}.
$$

因此

$$
v_1=0,
\qquad
m=1.
$$

这里不是先假设 $v_1=0$，而是由 $M(T)f(y)=f(y)$ 推出 $v_1=0$ 和 $m=1$。

于是

$$
M(T)=
\begin{pmatrix}
D_xP(y)&0\\
v_2^T&1
\end{pmatrix}.
$$

这是块下三角矩阵。因此它的特征值由左上角块和右下角块共同给出：

$$
\operatorname{eig}(M(T))
=
\operatorname{eig}(D_xP(y))\cup\{1\}.
$$

所以 $M(T)$ 的特征值是

$$
1,\mu_1,\dots,\mu_{d-1}.
$$

这就是 Theorem 5.6 的结论：monodromy matrix 的非平凡特征值就是 Poincaré map 的 multipliers。

---

## 15. 一维 map 为什么不一定来自 flow

讲义中的例子是

$$
g(x)=-\frac12 x.
$$

它不能实现为某个一维 ODE

$$
x'=f(x),\qquad x\in\mathbb R
$$

的 time-$T$ map。

原因是：一维 flow 的 time-$T$ map 必须保持顺序。

如果

$$
x_1<x_2,
$$

那么两条解轨道不能相交。否则如果某个时刻

$$
\phi_t(x_1)=\phi_t(x_2),
$$

由 ODE 解的唯一性，两条解从那一刻开始完全相同，反推回去会得到

$$
x_1=x_2,
$$

矛盾。

所以必须有

$$
\phi_T(x_1)<\phi_T(x_2).
$$

因此一维 flow 的 time-$T$ map 是严格递增的。

但是

$$
g(x)=-\frac12 x
$$

是严格递减的。例如

$$
1<2,
$$

但

$$
g(1)=-\frac12>-1=g(2).
$$

所以它反转顺序，不可能是一维 flow 的 time-$T$ map。

从导数角度也可以看：一维 flow 的 time-$T$ map 满足

$$
D\phi_T(x)
=
\exp\left(\int_0^T f'(\phi_t(x))\,dt\right)>0.
$$

但

$$
g'(x)=-\frac12<0.
$$

因此不可能。

---

## 16. Suspension flow：把离散映射变成连续流

给定一个 diffeomorphism

$$
g:\mathbb R^d\to\mathbb R^d,
$$

可以构造一个连续时间系统，使得 $g$ 成为它的 time-$T$ map，但需要把空间升高一维。

定义

$$
\mathcal M
:=
\{(x,t)\in\mathbb R^d\times\mathbb R:t\in[0,T]\}
/((g(x),0)\sim(x,T)).
$$

这叫 suspension space，也可以看作 mapping torus。

先看

$$
\mathbb R^d\times[0,T].
$$

它像一个柱体：

- 底部是 $t=0$；
- 顶部是 $t=T$；
- $x\in\mathbb R^d$ 是横向坐标。

粘合关系

$$
(g(x),0)\sim(x,T)
$$

表示：顶部的点 $(x,T)$ 和底部的点 $(g(x),0)$ 被看成同一个点。

这不表示

$$
(x,0)\sim(g(x),0).
$$

底部的 $x$ 和底部的 $g(x)$ 一般仍然是不同点。它只表示

$$
(x,T)\sim(g(x),0).
$$

---

## 17. Suspension flow 如何连续运动

在 $\mathcal M$ 上定义 flow：

$$
(x',t')=(0,1).
$$

也就是说，$x$ 不变，$t$ 匀速增加。

从

$$
(x_0,0)
$$

出发，先沿着竖直方向走：

$$
(x_0,0)\to(x_0,T).
$$

由于粘合关系

$$
(x_0,T)\sim(g(x_0),0),
$$

所以时间 $T$ 后它到达截面 $\{t=0\}$ 上的点

$$
(g(x_0),0).
$$

继续第二圈：

$$
(g(x_0),0)\to(g(x_0),T).
$$

再利用粘合关系：

$$
(g(x_0),T)\sim(g(g(x_0)),0)=(g^2(x_0),0).
$$

因此时间 $2T$ 后到达

$$
(g^2(x_0),0).
$$

继续下去：

$$
(x_0,0)
\mapsto
(g(x_0),0)
\mapsto
(g^2(x_0),0)
\mapsto
(g^3(x_0),0)
\mapsto\cdots.
$$

所以 time-$T$ map 在截面 $\{t=0\}$ 上诱导出的映射正是

$$
x\mapsto g(x).
$$

这就是 suspension 的意义：把离散动力系统

$$
x_{n+1}=g(x_n)
$$

嵌入到一个连续时间流中，使得每经过时间 $T$，截面上的点就迭代一次 $g$。

---

## 18. 为什么 suspension flow 看起来跳跃但实际上连续

如果只看普通柱体

$$
\mathbb R^d\times[0,T],
$$

轨道从

$$
(x,T)
$$

到

$$
(g(x),0)
$$

似乎发生了跳跃。

但在商空间 $\mathcal M$ 里，这两个点被定义为同一个点：

$$
(x,T)=(g(x),0)\quad \text{in }\mathcal M.
$$

所以在真正的空间 $\mathcal M$ 中，轨道没有跳跃，而是连续穿过粘合边界。

这和圆周的构造类似：

$$
S^1=[0,T]/(0\sim T).
$$

在区间 $[0,T]$ 中，从 $T$ 回到 $0$ 看起来像跳跃；但在圆周上，$T$ 和 $0$ 是同一个点，所以运动是连续的。

Suspension flow 只是把普通粘合

$$
(x,T)\sim(x,0)
$$

换成了带有 $g$ 的粘合：

$$
(x,T)\sim(g(x),0).
$$

因此它不是普通圆柱，而是一个被 $g$ 扭转后粘起来的空间。

---

## 19. 整体结构

这一节的核心逻辑是：

$$
\text{连续时间周期轨道}
\longrightarrow
\text{截面上的离散返回映射}.
$$

周期轨道 $\Gamma$ 对应 Poincaré map 的不动点：

$$
P(0)=0.
$$

周期轨道的横向稳定性对应不动点的线性稳定性：

$$
A=DP(0).
$$

矩阵 $A$ 的特征值

$$
\mu_1,\dots,\mu_{d-1}
$$

就是 Poincaré multipliers。

另一方面，沿周期轨道线性化得到变分方程

$$
u'=A(t)u.
$$

其 fundamental solution $M(t)$ 在一个周期后的矩阵

$$
M(T)
$$

称为 monodromy matrix。

它的特征值是

$$
1,\mu_1,\dots,\mu_{d-1}.
$$

其中 $1$ 是沿周期轨道切向方向的平凡特征值，而 $\mu_1,\dots,\mu_{d-1}$ 是横向方向上的 multipliers，真正决定周期轨道局部稳定性。

最后，suspension flow 表明：离散映射也可以通过升高一维变成连续流的 time-$T$ map。它把

$$
x\mapsto g(x)
$$

解释为连续流每走一圈之后在截面上的返回映射。
