# L11–L14：优化、随机方法、稀疏性与低秩学习

本笔记严格按照 slides L11–L14 的 section 顺序组织。两本参考书中的材料只补入与其对应的 slide section，不另设书本专题。

**参考书简称**

- **LTfFP：** Francis Bach，*Learning Theory from First Principles*，2025 年 5 月 26 日版本。
- **UML：** Shai Shalev-Shwartz 与 Shai Ben-David，*Understanding Machine Learning: From Theory to Algorithms*，2014 年电子版。
- LTfFP 的 PDF 在正文前有 16 页前置内容，因此引用同时给出书内页码和 PDF 页码；UML 使用本电子版中显示的电子版/PDF 页码。

# L11：优化与学习 - 1

## 1. 将学习表述为优化问题

> **页码：** Slides：L11，第 3-13 页。书本补充：LTfFP 第 2.2.2-2.3.2 节，书内第 27-35 页（PDF 第 43-51 页），以及第 5.1 节，书内第 109-110 页（PDF 第 125-126 页）；UML 第 2.2-2.3 节，电子版第 35-40 页，第 5.2 节，电子版第 64-65 页，以及第 13.2-13.3 节，电子版第 171-179 页。

设训练样本独立同分布：

$$
(x_i,y_i)\overset{\mathrm{iid}}{\sim}P_{X,Y},
\qquad i=1,\ldots,n.
$$

对于预测器 $f_\theta$ 和损失函数 $\ell$，总体风险为

$$
R(\theta)=\mathbb E[\ell(Y,f_\theta(X))],
$$

而经验风险为

$$
\widehat R_n(\theta)
=\frac1n\sum_{i=1}^n\ell(y_i,f_\theta(x_i)).
$$

训练通常求解一个正则化经验风险问题：

$$
\widehat\theta\approx
\operatorname*{argmin}_\theta
\left\{
\widehat R_n(\theta)+\Omega(\theta)
\right\}.
$$

这里最重要的提醒是：机器学习并不只是对训练目标进行数值最小化。真正的目标是在未见数据上获得较低的风险。

令

$$
\theta^\star\in\operatorname*{argmin}_\theta R(\theta).
$$

通过加上再减去经验风险，可得

$$
\begin{aligned}
R(\widehat\theta)-R(\theta^\star)
={}&[R(\widehat\theta)-\widehat R_n(\widehat\theta)]\\
&+[\widehat R_n(\widehat\theta)-\widehat R_n(\theta^\star)]\\
&+[\widehat R_n(\theta^\star)-R(\theta^\star)].
\end{aligned}
$$

由于 $\inf_\theta\widehat R_n(\theta)\le\widehat R_n(\theta^\star)$，

$$
\widehat R_n(\widehat\theta)-\widehat R_n(\theta^\star)
\le
\widehat R_n(\widehat\theta)-\inf_\theta\widehat R_n(\theta).
$$

因此，超额风险由两个统计偏差项和一个优化误差项控制。如果统计误差的量级为 $\delta_n$，那么把优化误差降到远低于 $\delta_n$，通常并不会改善测试性能。

**与书本对应的补充。** 两本书还分离出了一个额外的建模误差项。若 $\mathcal F$ 是选定的预测器族，$R^\star$ 是不受模型类限制的 Bayes 风险，则

$$
R(\widehat f)-R^\star
=
\underbrace{R(\widehat f)-\inf_{f\in\mathcal F}R(f)}_{\text{估计与优化}}
+
\underbrace{\inf_{f\in\mathcal F}R(f)-R^\star}_{\text{近似误差}}.
$$

第一项受到有限数据和不完全优化的影响；第二项源于模型类的归纳偏置。UML 将 ERM 描述为一种归纳偏置机制，并通过过拟合解释了为什么在不控制模型类时，仅仅最小化训练损失可能会失败。LTfFP 随后把优化误差加入通常的经验风险与总体风险分解中。这使实际停止准则变得明确：除非出于其他原因需要更精确的经验解，否则只需优化到优化误差与估计误差处于相近量级。

UML 还把 slides 中提出的稳定性问题进行了严格形式化。若 $S^{(i)}$ 是通过替换 $S$ 中的一个样本得到的数据集，则期望泛化间隙等于平均的单样本替换敏感度：

$$
\mathbb E[L_D(A(S))-L_S(A(S))]
=
\mathbb E[\ell(A(S^{(i)}),z_i)-\ell(A(S),z_i)].
$$

对于正则化损失最小化

$$
A(S)=\operatorname*{argmin}_w
\{L_S(w)+\lambda\|w\|^2\},
$$

当损失函数为凸且 $\rho$-Lipschitz 时，UML 得到

$$
\mathbb E L_D(A(S))
\le
L_D(w^\star)+\lambda\|w^\star\|^2
+\frac{2\rho^2}{\lambda n}.
$$

正则化偏置项随 $\lambda$ 增大，而稳定性项随 $\lambda$ 减小。这给出了精确的拟合与泛化之间的权衡。这里使用的稳定性概念是平均单样本替换稳定性，而不是更强的均匀稳定性。

## 2. 下降方向与梯度下降的几何

> **页码：** Slides：L11，第 14-27 页。书本补充：LTfFP 第 5.2.2-5.2.4 节，书内第 116-126 页（PDF 第 132-142 页）；UML 第 12.1 节和第 14.1 节，电子版第 156-163 页和第 185-188 页。

一般的一阶迭代可写为

$$
\theta^{k+1}=\theta^k+\eta_kd^k,
\qquad \eta_k>0.
$$

在 $\theta$ 处作 Taylor 展开，得到

$$
F(\theta+\eta d)
=F(\theta)+\eta\langle\nabla F(\theta),d\rangle+o(\eta).
$$

因此，任何满足

$$
\langle\nabla F(\theta),d\rangle<0
$$

的方向，在正步长充分小时都是下降方向。负梯度是欧氏几何下的最陡下降方向，并且垂直于局部等高线：

$$
\theta^{k+1}=\theta^k-\eta_k\nabla F(\theta^k).
$$

仅有正确的方向还不够：过大的步长可能越过低函数值区域，反而使目标函数增大。

对于 $L$-smooth 函数，

$$
F(\theta-\eta\nabla F(\theta))
\le
F(\theta)
-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla F(\theta)\|^2.
$$

因此，在非驻点处，每个 $0<\eta<2/L$ 都能保证下降。

**与书本对应的补充。** UML 还把梯度下降推导为对局部线性模型进行正则化最小化：

$$
\theta_{t+1}
=\operatorname*{argmin}_\theta
\left\{
\langle\nabla F(\theta_t),\theta-\theta_t\rangle
+\frac1{2\eta}\|\theta-\theta_t\|^2
\right\}.
$$

这一视角解释了为什么改变二次项会改变几何结构或预条件器。对于次梯度 $v_t$ 的范数由 $G$ 控制的凸函数，望远镜求和恒等式

$$
\sum_{t=1}^T
\langle\theta_t-\theta^\star,v_t\rangle
\le
\frac{\|\theta_1-\theta^\star\|^2}{2\eta}
+\frac\eta2\sum_{t=1}^T\|v_t\|^2
$$

可推出平均迭代点 $\bar\theta_T$ 满足

$$
F(\bar\theta_T)-F(\theta^\star)
\le
\frac{\|\theta_1-\theta^\star\|^2}{2\eta T}
+\frac{\eta G^2}{2}.
$$

平衡右侧两项可得到一般非光滑凸问题的 $O(T^{-1/2})$ 收敛率。Smoothness 将确定性凸问题的收敛率提高到 $O(1/T)$，而强凸性则允许几何收敛。

## 3. 最小二乘上的梯度下降

> **页码：** Slides：L11，第 28-32 页。书本补充：LTfFP 第 3.2-3.6 节，书内第 46-59 页（PDF 第 62-75 页），第 5.2.1 节，书内第 112-116 页（PDF 第 128-132 页），以及第 12.1.1 节，书内第 344-346 页（PDF 第 360-362 页）；UML 第 9.2 节，电子版第 123-125 页，以及附录 C，电子版第 430-434 页。

考虑

$$
F(\theta)=\frac1{2n}\|\Phi\theta-y\|_2^2.
$$

其梯度和 Hessian 分别为

$$
\nabla F(\theta)=\frac1n\Phi^\top(\Phi\theta-y),
\qquad
H=\frac1n\Phi^\top\Phi\succeq0.
$$

每个最优解都满足正规方程

$$
H\theta^\star=\frac1n\Phi^\top y.
$$

设 $H$ 的特征值位于 $[\mu,L]$ 中。对于固定步长 GD，

$$
e_t:=\theta_t-\theta^\star
=(I-\gamma H)^te_0.
$$

当 $\mu>0$ 时，最优常数步长为

$$
\gamma^\star=\frac2{L+\mu},
$$

其收缩因子为

$$
q^\star=\frac{\kappa-1}{\kappa+1},
\qquad
\kappa=\frac L\mu.
$$

更简单的选择 $\gamma=1/L$ 给出

$$
\|e_t\|^2
\le
\left(1-\frac1\kappa\right)^{2t}\|e_0\|^2
\le
e^{-2t/\kappa}\|e_0\|^2.
$$

因此，迭代复杂度为

$$
O\!\left(\kappa\log\frac1\varepsilon\right).
$$

若 $H$ 奇异，到任意指定最优解的参数距离不一定趋于零，但

$$
F(\theta)-F^\star
=\frac12(\theta-\theta^\star)^\top H(\theta-\theta^\star)
$$

会忽略零空间方向。当 $0<\gamma\le1/L$ 时，

$$
F(\theta_t)-F^\star
\le
\frac{\|e_0\|^2}{8\gamma t}
=O(1/t).
$$

加入 ridge 正则化后，Hessian 变为 $H+\lambda I$，因此最小特征值至少为 $\lambda$。

**与书本对应的补充。** 最小二乘相关章节给出了一个几何解释：$\Phi\theta^\star$ 是 $y$ 到 $\operatorname{im}(\Phi)$ 上的正交投影。这证明了即使 $H$ 奇异，最优解仍然存在。过参数化章节还补充了隐式偏置结论。如果 $\theta_0=0$ 且步长处于稳定范围内，GD 始终位于 $\operatorname{im}(\Phi^\top)$ 中，并收敛到

$$
\theta_\infty=\Phi^\dagger y,
$$

即欧氏范数最小的最小二乘解。对于一般初始化，$\ker(\Phi)$ 中的分量会被保留。因此，尽管所有最优解都给出相同的预测，优化算法仍会从无穷多个解中选择一个特定解。

# L12：优化与学习 - 2

## 1. 为什么需要随机梯度下降

> **页码：** Slides：L12，第 2–10 页。书本补充：LTfFP 第 5.1 节和第 5.4 节，书内第 109–110 页和第 134–139 页（PDF 第 125–126 页和第 150–155 页）；UML 第 14.3 节和第 14.5.1 节，电子版第 191–198 页。

大规模学习问题通常具有有限和形式

$$
F(\theta)=\frac1n\sum_{i=1}^nf_i(\theta).
$$

完整梯度下降在每次迭代中都要计算全部 $n$ 个分量的梯度。SGD 抽取 $i_k$，并采用更新

$$
\theta_{k+1}
=\theta_k-\gamma_k\nabla f_{i_k}(\theta_k).
$$

在有放回均匀抽样下，

$$
\mathbb E[\nabla f_{i_k}(\theta_k)\mid\theta_k]
=\nabla F(\theta_k).
$$

同一思想也可以直接应用于总体风险

$$
F(\theta)=\mathbb E_\xi[\phi(\theta,\xi)]
$$

的优化：每次迭代抽取一个新的 $\xi_k$。一次随机更新的成本很低，但其轨迹带有噪声，而且通常不具有单调性。

**与书本对应的补充。** UML 强调，在线 SGD 并不是先构造一个经验目标，再精确地将其最小化。相反，

$$
g_t\in\partial_\theta\phi(\theta_t,\xi_t)
$$

是总体风险的一个无偏次梯度。因此，迭代复杂度界同时也是样本复杂度界：每次更新消耗一个新样本。对于比较点范数至多为 $B$ 的凸 $G$-Lipschitz 问题，$T=O(B^2G^2/\varepsilon^2)$ 个新样本足以使期望超额风险不超过 $\varepsilon$。LTfFP 强调了同样的优化—统计匹配关系：仅进行一次数据遍历，就已经能够达到正则化 ERM 的统计精度。

## 2. 标量最小二乘的直观解释

> **页码：** Slides：L12，第 11–18 页。书本补充：LTfFP 第 5.4.3 节，书内第 143–146 页（PDF 第 159–162 页）；UML 第 9.2 节和第 14.3 节，电子版第 123–125 页和第 191–193 页。

对于

$$
F(\theta)=\frac12\sum_{i=1}^n(x_i\theta-b_i)^2,
$$

全局最小点为

$$
\theta^\star
=\frac{\sum_i x_ib_i}{\sum_i x_i^2}
=\sum_i
\frac{x_i^2}{\sum_jx_j^2}\theta_i^\star,
\qquad
\theta_i^\star=\frac{b_i}{x_i}.
$$

因此，

$$
\theta^\star\in
[\min_i\theta_i^\star,\max_i\theta_i^\star].
$$

由于

$$
\nabla f_i(\theta)=x_i^2(\theta-\theta_i^\star),
$$

在该区间之外，所有分量梯度的符号都相同；在区间内部，它们则可能相互冲突。这解释了为什么 SGD 会发生振荡，以及为什么需要递减步长和迭代平均。

**与书本对应的补充。** LTfFP 分析了多维带噪声的最小均方递推。对于

$$
y_t=\phi(x_t)^\top\theta^\star+\varepsilon_t,
$$

固定步长 SGD 满足

$$
\theta_t-\theta^\star
=
(I-\gamma\phi_t\phi_t^\top)(\theta_{t-1}-\theta^\star)
+\gamma\varepsilon_t\phi_t.
$$

第一项收缩由初始条件造成的偏差，第二项则持续注入方差。在特征有界的条件下，平均迭代满足一个代表性上界

$$
\mathbb E[F(\bar\theta_t)-F(\theta^\star)]
\le
\frac{\|\theta_0-\theta^\star\|^2}{2\gamma t}
+\frac{\gamma\sigma^2R^2}{2}.
$$

该公式明确展示了步长的权衡：较小的 $\gamma$ 会降低噪声平台，但也会减慢初始偏差的消除速度。

## 3. 抽样、mini-batch、迭代平均及其与 GD 的比较

> **页码：** Slides：L12，第 19–28 页。书本补充：LTfFP 第 5.4 节和第 5.4.4 节，书内第 134–141 页和第 146–151 页（PDF 第 150–157 页和第 162–167 页）；UML 第 14.3–14.4 节，电子版第 191–196 页。

有放回抽样可以得到简洁的条件无偏性证明。随机重排在每个 epoch 中恰好访问每个样本一次，但会失去逐步独立性。一个 mini-batch $I_k$ 给出

$$
\theta_{k+1}
=\theta_k-
\frac{\gamma_k}{|I_k|}
\sum_{j\in I_k}\nabla f_j(\theta_k).
$$

较大的 batch 可以减小方差并支持并行计算，但其边际收益最终会下降。

Polyak–Ruppert 平均定义为

$$
\bar\theta_k=\frac1{k+1}\sum_{j=0}^k\theta_j.
$$

典型的凸与强凸收敛率分别为

$$
\mathbb E[F(\bar\theta_k)-F^\star]
=O(k^{-1/2})
$$

和

$$
\mathbb E[F(\bar\theta_k)-F^\star]
=O\!\left(\frac{\kappa}{k}\right).
$$

完整 GD 的轨迹更稳定；对于光滑强凸目标，其迭代收敛率为 $e^{-k/\kappa}$，但每一步需要使用全部 $n$ 个分量。第 28 页 slide 中并列出现的 $O(\kappa/k)$ 和 $O(1/\varepsilon^2)$ 对应不同的误差机制：$1/k$ 收敛率对应 $O(1/\varepsilon)$ 次迭代，而 $1/\sqrt{k}$ 收敛率对应 $O(1/\varepsilon^2)$ 次迭代。

**与书本对应的补充。** 两本书都推导了通用的平均 SGD 不等式

$$
\mathbb E[F(\bar\theta_T)-F(\theta^\star)]
\le
\frac{\|\theta_1-\theta^\star\|^2}{2\gamma T}
+\frac{\gamma G^2}{2}.
$$

它将有限时间偏差与梯度噪声分离开来。UML 还列出了输出随机迭代、后缀平均和加权平均等方案；这些变体能够改善强凸问题中的表现。LTfFP 的方差缩减章节则针对有限和问题解释了另一条路径：偶尔计算完整梯度，或维护完整梯度的相关信息，使随机方向在保持无偏的同时，其方差在最优点附近逐渐消失。

## 4. 光滑非凸目标上的 SGD

> **页码：** Slides：L12，第 29–37 页。书本补充：LTfFP 第 5.2.6 节，书内第 129–130 页（PDF 第 145–146 页），作为确定性非凸问题的基准。UML 第 14 章，电子版第 184–201 页，讨论凸 SGD，但没有给出 slide 中的非凸随机定理。

假设各分量函数都是 $L$-smooth，随机梯度条件无偏，并且

$$
\mathbb E\|g_k\|^2\le G^2.
$$

光滑性不等式给出

$$
\mathbb E F(\theta_{k+1})
\le
\mathbb E F(\theta_k)
-\gamma_k\mathbb E\|\nabla F(\theta_k)\|^2
+\frac{L\gamma_k^2G^2}{2}.
$$

当 $\gamma_k=c/\sqrt T$ 时，

$$
\boxed{
\frac1T\sum_{k=1}^T
\mathbb E\|\nabla F(\theta_k)\|^2
\le
\frac1{\sqrt T}
\left[
\frac{F(\theta_1)-F_\inf}{c}
+\frac{LcG^2}{2}
\right].
}
$$

结论是近似驻点性，而不是全局最优性。若要求期望梯度平方范数不超过 $\varepsilon$，则需要 $T=O(\varepsilon^{-2})$。

**与书本对应的补充。** LTfFP 给出了确定性情形的比较基准。当步长为 $1/L$ 时，

$$
F(\theta_{t+1})
\le
F(\theta_t)-\frac1{2L}\|\nabla F(\theta_t)\|^2,
$$

因而

$$
\min_{0\le s<T}\|\nabla F(\theta_s)\|^2
\le
\frac{2L(F(\theta_0)-F_\inf)}{T}.
$$

slide 中的定理收敛较慢，是因为随机噪声引入了额外的 $\gamma^2G^2$ 项。两本书都没有利用这一结论声称算法会收敛到局部或全局最小点。

## 5. 凸与强凸目标上的投影 SGD

> **页码：** Slides：L12，第 38–50 页。书本补充：LTfFP 第 5.4.1 节，书内第 139–141 页（PDF 第 155–157 页）；UML 第 14.4.1 节和第 14.4.4 节，电子版第 193–196 页。

投影 SGD 为

$$
\theta_{k+1}=P_{\mathcal X}(\theta_k-\gamma_kg_k).
$$

投影具有非扩张性，因此令 $R_k=\|\theta_k-\theta^\star\|^2$，有

$$
R_{k+1}
\le
R_k+\gamma_k^2\|g_k\|^2
-2\gamma_k\langle g_k,\theta_k-\theta^\star\rangle.
$$

对于 $\mu$-强凸目标和 $\|g_k\|\le G$，

$$
\mathbb E[F(\theta_k)-F^\star]
\le
\frac{\gamma_kG^2}{2}
+\frac{\gamma_k^{-1}-\mu}{2}r_k
-\frac{r_{k+1}}{2\gamma_k}.
$$

取 $\gamma_k=1/(\mu k)$ 并进行平均，得到

$$
\mathbb E[F(\bar\theta_T)-F^\star]
\le
\frac{G^2(1+\log T)}{2\mu T}.
$$

使用线性递增的权重

$$
w_i=\frac{2(i+1)}{(T+1)(T+2)},
$$

可以去掉对数因子：

$$
\mathbb E[F(\bar\theta_T)-F^\star]
\le
\frac{2G^2}{\mu(T+1)}.
$$

**与书本对应的补充。** UML 使用到可行集的投影证明了相同的带对数收敛率，并指出两种改进：只对最后一半迭代取平均可以去掉对数因子，而更精细的分析也可以控制最后一个迭代点。LTfFP 进一步强调，对于这一问题类，$O(1/(\mu T))$ 在常数因子意义下达到了随机梯度 oracle 的最优收敛率；但步长 $1/(\mu t)$ 会失去自适应性，因为它需要事先知道强凸参数。

## 6. 梯度裁剪与归一化梯度下降

> **页码：** Slides：L12，第 51–56 页。书本补充：LTfFP 练习 5.20，书内第 132 页（PDF 第 148 页），对应归一化次梯度下降。UML 没有直接讨论梯度裁剪或归一化梯度。

裁剪梯度下降为

$$
x_{k+1}
=x_k-
\min\left(\eta,\frac{a\eta}{\|g_k\|}\right)g_k.
$$

等价地，

$$
x_{k+1}=
\begin{cases}
x_k-\eta g_k,&\|g_k\|\le a,\\
x_k-a\eta\dfrac{g_k}{\|g_k\|},&\|g_k\|>a.
\end{cases}
$$

因此，更新范数至多为 $a\eta$。

归一化梯度下降采用

$$
x_{k+1}
=x_k-
\frac{\eta}{\|g_k\|+b}g_k.
$$

令 $b=a$ 并匹配分子中的尺度后，由于

$$
\max(a,g)\le a+g\le2\max(a,g).
$$

两种方法的有效乘子至多相差两倍。其目的是在梯度较小时保留普通梯度行为，同时避免极端梯度决定更新长度。

**与书本对应的补充。** LTfFP 分析了与之相关的归一化次梯度更新

$$
\theta_t
=\theta_{t-1}
-\frac{D}{\sqrt t}\frac{g_t}{\|g_t\|}.
$$

对于凸 $B$-Lipschitz 目标，它给出

$$
\min_{0\le s\le t-1}F(\theta_s)-F^\star
\le
DB\frac{2+\log t}{2\sqrt t}.
$$

这一更新、梯度裁剪以及 slides 中的平滑归一化都会使步长与很大的原始梯度范数解耦，但它们是不同的算法；这里展示的定理并不能直接证明 slide 中的裁剪规则。

## 7. 动量法

> **页码：** Slides：L12，第 57–63 页。书本补充：LTfFP 第 5.2.5 节，书内第 126–129 页（PDF 第 142–145 页），用于相关的加速方法比较。UML 没有直接发展 heavy-ball 动量法。

slide 使用的参数化为

$$
m_t=\beta m_{t-1}+\nabla f(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta m_t.
$$

它等价于 heavy-ball 递推

$$
\theta_{t+1}
=\theta_t-\eta\nabla f(\theta_t)
+\beta(\theta_t-\theta_{t-1}).
$$

当 $m_0=0$ 时，

$$
\theta_{t+1}
=\theta_0-
\eta\sum_{i=1}^t
\frac{1-\beta^{t+1-i}}{1-\beta}
\nabla f(\theta_i).
$$

因此，较早的梯度会持续影响之后的许多更新。在狭长谷底中，高曲率方向上来回振荡的分量往往会相互抵消，而低曲率方向上一致的分量则会不断累积。

**与书本对应的补充。** LTfFP 展示的是 Nesterov 加速，而不是完全相同的 heavy-ball 递推。对于光滑强凸目标，它把条件数依赖从

$$
O\!\left(\kappa\log\frac1\varepsilon\right)
$$

改善为

$$
O\!\left(\sqrt\kappa\log\frac1\varepsilon\right).
$$

这为 slide 中关于加速的论述提供了支持，同时仍须明确区分两种算法：heavy-ball 动量法与 Nesterov 的双序列构造彼此相关，但并不相同。

# L13：优化收尾、自适应方法与稀疏变量选择

## 1. 缩放梯度与 AdaGrad

> **页码：** Slides：L13，第 2–7 页。书本补充：LTfFP 第 5.4.2 节，书内第 141–143 页（PDF 第 157–159 页）。UML 没有直接讨论 AdaGrad 的 section。

缩放梯度方法采用

$$
\theta_{t+1}=\theta_t-G_t^{-1/2}g_t.
$$

若 $G_t=\operatorname{Diag}(s_{t,1},\ldots,s_{t,d})$ 为对角矩阵，则

$$
\theta_{t+1,j}
=\theta_{t,j}-\frac{g_{t,j}}{\sqrt{s_{t,j}}}.
$$

AdaGrad 选择

$$
G_t=\sum_{i=1}^tg_ig_i^\top,
$$

而实际实现通常只保留它的对角线。因此

$$
s_{t,j}=\sum_{i=1}^tg_{i,j}^2.
$$

经常出现大梯度的坐标会得到越来越小的有效学习率，而很少活跃的坐标仍可保留较大步长。这对稀疏数据很有帮助，但永久累积也可能使学习率衰减得过快。

一种指数加权替代方案是

$$
G_t
=(1-\beta)\sum_{i=1}^t\beta^{t-i}g_ig_i^\top
=\beta G_{t-1}+(1-\beta)g_tg_t^\top.
$$

**与书本对应的补充。** LTfFP 把自适应性解释为随机预条件。对角度量 $M=\operatorname{Diag}(m_1,\ldots,m_d)$ 对应的一类代表性上界为

$$
\frac1{2T}\sum_{j=1}^d m_j(\theta_j^\star)^2
+\frac{G^2}{2}\sum_{j=1}^d\frac{\Sigma_{jj}}{m_j},
$$

其中 $\Sigma$ 是梯度或特征的二阶矩矩阵。对各个 $m_j$ 优化可以看出：当不同坐标的尺度差异很大时，逐坐标缩放可能明显优于统一的全局步长。AdaGrad 不需要预先知道这些对角统计量，而是在运行中在线估计它们。

## 2. Adam 与偏差修正

> **页码：** Slides：L13，第 8–17 页。书本补充：LTfFP 第 5.4.2 节，书内第 143 页（PDF 第 159 页）；该页把 Adam 归类为在线自适应预条件方法，但没有推导其完整递推。UML 没有直接讨论 Adam。

Adam 维护梯度一阶原始矩和二阶原始矩的指数移动平均：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t).
$$

从零初始化时，

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}
$$

消除了早期朝零方向的偏差。标准更新为

$$
\boxed{
\theta_{t+1}
=\theta_t-
\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}
}
$$

其中开方和除法都逐坐标进行。$v_t$ 是二阶原始矩，并不是减去均值后的统计方差。

若把某个梯度坐标乘以 $c$，则 $m_{t,j}$ 近似乘以 $c$，$v_{t,j}$ 近似乘以 $c^2$，所以比值 $m_{t,j}/\sqrt{v_{t,j}}$ 近似具有尺度不变性。

Slides 的一处公式把 $\varepsilon$ 放在平方根内，另一处矩阵公式把它放在平方根外且省略了 $\eta$。上面的常用写法把 $\varepsilon$ 放在平方根外。参数更新应使用经过偏差修正的量，但不能用它们覆盖未修正的 EMA 状态。

**与书本对应的补充。** LTfFP 把 AdaGrad 和 Adam 都联系到梯度协方差的在线估计。书中的作用是给出理论定位，而不是再次推导 Adam：自适应方法在优化运行过程中学习一个预条件器。因此，本节详细的一阶矩、二阶矩和偏差修正式仍来自 slides。

## 3. L2 正则化、权重衰减与 AdamW

> **页码：** Slides：L13，第 18–20 页。直接书本补充：LTfFP 与 UML 都没有讨论 AdamW 或 decoupled weight decay。

若目标函数为

$$
\mathcal L(\theta)+\frac\lambda2\|\theta\|^2,
$$

则梯度为 $g_t+\lambda\theta_t$。若把整个梯度送入 Adam，正则项会同时进入 $m_t$ 与 $v_t$，从而使收缩作用与逐坐标自适应缩放纠缠在一起。

AdamW 将两项操作解耦：

$$
\theta_{t+1}
=\theta_t-
\eta(d_t+\lambda\theta_t)
=(1-\eta\lambda)\theta_t-\eta d_t,
$$

其中

$$
d_t=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}.
$$

当参数随时间变化时，

$$
\theta_{t+1}
=(1-\eta_t\lambda_t)\theta_t-\eta_td_t.
$$

如果没有梯度项，累积衰减为

$$
\theta_T
=\left[\prod_{t=0}^{T-1}(1-\eta_t\lambda_t)\right]\theta_0
\approx
\exp\left(-\sum_t\eta_t\lambda_t\right)\theta_0.
$$

Slides 中更宽泛的形式 $\theta_{t+1}=\alpha_t\theta_t+\beta_tm_t$ 只有在把 $m_t$ 重新解释为已经完成全部预条件的方向时才能表示 Adam；单个标量 $\beta_t$ 无法表达对原始一阶矩进行逐坐标除法。

## 4. 稀疏变量选择

> **页码：** Slides：L13，第 21–23 页。书本补充：LTfFP 第 8.1–8.2.1 节，书内第 221–228 页（PDF 第 237–244 页）；UML 第 25.1 与 25.1.3 节，电子版第 358–365 页。

考虑固定设计线性模型

$$
y=\Phi\theta^\star+\varepsilon,
\qquad
\mathbb E\varepsilon=0,
\qquad
\operatorname{Cov}(\varepsilon)=\sigma^2I.
$$

目标量是预测误差

$$
\frac1n\|\Phi(\widehat\theta-\theta^\star)\|^2,
$$

而不一定是参数的欧氏误差。普通最小二乘的风险量级为 $\sigma^2d/n$。若

$$
\|\theta^\star\|_0\le k\ll d,
$$

就希望用依赖稀疏度的复杂度替代 $d$。

**与书本对应的补充。** LTfFP 把已知支持集的情形称为 oracle case。若 $A=\operatorname{supp}(\theta^\star)$ 已知，则只在 $\Phi_A$ 上做最小二乘，预测风险的量级为

$$
\frac{\sigma^2k}{n}.
$$

支持集未知时，估计量必须在大约 $\binom dk$ 个模型中搜索，因此额外支付 $\log(d/k)$ 因子。UML 强调计算方面：穷举所有大小为 $k$ 的子集通常不可行，这推动了过滤方法、贪心选择和 $\ell_1$ 松弛。特征选择还可以降低存储、预测、特征测量与统计估计成本。

## 5. 约束最小二乘的基本不等式

> **页码：** Slides：L13，第 23–25 页。书本补充：LTfFP 第 8.1.1–8.1.2 节，书内第 223–226 页（PDF 第 239–242 页）。UML 没有以同样形式给出该固定设计投影证明。

设 $\Omega$ 是包含 $\theta^\star$ 的约束集，并令

$$
\widehat\theta\in
\operatorname*{argmin}_{\theta\in\Omega}
\|y-\Phi\theta\|^2.
$$

记 $\Delta=\widehat\theta-\theta^\star$。由最优性，

$$
\|\varepsilon-\Phi\Delta\|^2\le\|\varepsilon\|^2,
$$

所以

$$
\boxed{
\|\Phi\Delta\|^2
\le2\varepsilon^\top\Phi\Delta.
}
$$

将可行误差方向归一化可得

$$
\|\Phi\Delta\|^2
\le
4\sup_{\theta\in\Omega}
\left[
\varepsilon^\top
\frac{\Phi(\theta-\theta^\star)}
{\|\Phi(\theta-\theta^\star)\|}
\right]^2.
$$

当 $\Omega=\mathbb R^d$ 时，归一化后的预测方向充满 $\operatorname{im}(\Phi)$ 中的单位球面。因此

$$
\sup_{\substack{z\in\operatorname{im}(\Phi)\\\|z\|=1}}
(\varepsilon^\top z)^2
=\|\Pi_\Phi\varepsilon\|^2,
$$

从而

$$
\frac1n\mathbb E\|\Phi\Delta\|^2
\le
\frac{4\sigma^2\operatorname{rank}(\Phi)}n.
$$

**与书本对应的补充。** LTfFP 把这一推导作为可复用的证明模板。它不要求估计量有闭式解，只需要 $\theta^\star$ 可行以及 $\widehat\theta$ 近似最优。若经验目标只求解到加性误差 $\delta$，基本不等式会多出一个 $n\delta$ 量级的项。随后相同的几何化简会把优化误差与可行方向的 Gaussian 或 sub-Gaussian 复杂度分开。

## 6. 最佳子集选择的风险界

> **页码：** Slides：L13，第 26–31 页。书本补充：LTfFP 第 8.2.1–8.2.2 节，书内第 226–231 页（PDF 第 242–247 页）；UML 第 23.3 与 25.1.3 节，电子版第 330–338、363–365 页，用于说明 $\ell_0$ 与 $\ell_1$ 的计算差异。

令

$$
\Omega=\{\theta:\|\theta\|_0\le k\},
\qquad
\varepsilon\sim N(0,\sigma^2I_n).
$$

则约束最小二乘估计量满足

$$
\boxed{
\mathbb E\left[
\frac1n\|\Phi(\widehat\theta-\theta^\star)\|^2
\right]
\le
32\sigma^2\frac{k}{n}
\left(\log\frac dk+1\right).
}
$$

证明分为四步：

1. 两个 $k$-稀疏向量之差最多是 $2k$-稀疏的。
2. 对每个支持集 $B$，可行预测误差位于 $\operatorname{im}(\Phi_B)$。
3. 在该子空间上的上确界等于 $\|\Pi_{\Phi_B}\varepsilon\|^2$。
4. 候选支持集不超过

   $$
   \binom d{2k}\le\left(\frac{ed}{2k}\right)^{2k}
   $$

   再用 Gaussian 矩母函数界控制这些投影的最大值。

其中的对数项正是支持集未知所支付的代价。这个定理控制的是预测误差，不是参数恢复误差，也不保证精确恢复支持集。

**与书本对应的补充。** LTfFP 再给出三个重要结论。第一，即使允许指数级计算，这个速率在常数因子意义下仍是 minimax 最优的。第二，若 $k$ 未知，可以使用

$$
\min_\theta
\left\{
\frac1n\|y-\Phi\theta\|^2
+\lambda\|\theta\|_0
\right\},
$$

并选择依赖噪声的惩罚参数；这可以自适应稀疏度，但仍是组合优化。第三，实用的贪心法与硬阈值法需要额外条件才能继承相近的快速统计率。UML 在压缩感知中给出同样的计算区分：$\ell_0$ 描述理想的稀疏模型，而 $\ell_1$ 可以把恢复问题转化为可处理的凸规划。

# L14：稀疏性与低秩——第二部分

## 1. 硬稀疏约束算法

> **页码：** Slides：L14，第 2 页。书本补充：LTfFP 第 8.2.1 节，书内第 228 页（PDF 第 244 页），以及第 10.3.3 节，书内第 302–304 页（PDF 第 318–320 页）；UML 第 25.1.2 节，电子版第 360–363 页。

投影到 $\ell_0$ 球上求解的是

$$
\min_x\frac12\|x-y\|^2
\quad\text{subject to}\quad
\|x\|_0\le k.
$$

其解 $H_k(y)$ 保留绝对值最大的 $k$ 个坐标，并将其余坐标置零。因此，投影梯度法就成为迭代硬阈值法：

$$
x_{t+1}
=H_k\left(x_t-\gamma\nabla F(x_t)\right).
$$

OMP 则每次把一个变量加入支持集，并在已选支持集上重新拟合。

**与书本对应的补充。** UML 明确给出了最小二乘 OMP 的选择准则。若 $I_t$ 是已选集合，且 $V_t$ 的列构成 $X_{I_t}$ 所张成空间的一组标准正交基，则把每一个候选列分解为

$$
X_j=V_tV_t^\top X_j+u_j.
$$

下一项特征选择为

$$
j_t\in\operatorname*{argmax}_{j\notin I_t}
\frac{\langle u_j,y\rangle^2}{\|u_j\|^2},
$$

然后重新计算最小二乘系数。LTfFP 说明，精确的最佳子集搜索具有 $d^k$ 量级的成本；如果希望获得快速统计保证，OMP 和迭代硬阈值法都需要附加的设计条件。

## 2. Lasso、软阈值与近端梯度

> **页码：** Slides：L14，第 3–5 页。书本补充：LTfFP 第 5.2.5、5.3 与 8.3.1–8.3.5 节，书内第 126–134、231–240 页（PDF 第 142–150、247–256 页）；UML 第 25.1.3 节，电子版第 363–365 页。

Lasso 为

$$
\min_\theta
\frac1{2n}\|y-\Phi\theta\|^2
+\lambda\|\theta\|_1.
$$

在一维情形下，ridge 给出连续收缩：

$$
\operatorname*{argmin}_w
(y-w)^2+\lambda w^2
=\frac{y}{1+\lambda},
$$

而 $\ell_1$ 给出软阈值：

$$
S_\tau(y)=\operatorname{sign}(y)(|y|-\tau)_+.
$$

近端算子定义为

$$
\operatorname{prox}_{\eta f}(y)
=\operatorname*{argmin}_x
\left\{
\frac12\|x-y\|^2+\eta f(x)
\right\}.
$$

当 $f=\lambda\|\cdot\|_1$ 时，近端梯度法给出 ISTA：

$$
\boxed{
\theta_t
=S_{\lambda/L}
\left(
\theta_{t-1}
-\frac1L\nabla h(\theta_{t-1})
\right),
}
$$

其中

$$
h(\theta)=\frac1{2n}\|y-\Phi\theta\|^2,
\qquad
\nabla h(\theta)=\frac1n\Phi^\top(\Phi\theta-y).
$$

若 $\|\Phi^\top\varepsilon\|_\infty\le n\lambda/2$，slides 中的慢速率引理给出

$$
\|\widehat\theta\|_1\le3\|\theta^\star\|_1,
$$

以及

$$
\frac1n\|\Phi(\widehat\theta-\theta^\star)\|^2
\le3\lambda\|\theta^\star\|_1.
$$

**与书本对应的补充。** LTfFP 解释了其中的几何结构：在 $[-1,1]^d$ 上，$\ell_1$ 范数是 $\ell_0$ 计数函数的凸包络，而 $\ell_1$ 球的角点偏好坐标轴。书中还区分了两种统计机制。慢速率不要求设计矩阵满足任何条件，但其量级通常为 $n^{-1/2}$。要得到

$$
\frac{\sigma^2k\log d}{n}
$$

量级的快速速率，则要求在锥 $\|\Delta_{A^c}\|_1\le3\|\Delta_A\|_1$ 上满足 restricted-eigenvalue 类型的不等式。由于 $\ell_1$ 惩罚是可分离的，坐标下降法也是一种实用的求解器。UML 独立推导出相同的软阈值算子，并解释了 ridge 为什么几乎从不产生严格为零的坐标。

## 3. RIP 与迭代硬阈值

> **页码：** Slides：L14，第 6 页。书本补充：UML 第 23.3 节，电子版第 330–338 页；LTfFP 第 8.2.1 与 8.3.5 节，书内第 228、239–241 页（PDF 第 244、255–257 页）。

如果对每一个 $s$-稀疏向量 $x$ 都有

$$
(1-\delta_s)\|x\|^2
\le
\|\Phi x\|^2
\le
(1+\delta_s)\|x\|^2
$$

则称矩阵 $\Phi$ 的 $s$-阶 restricted isometry constant 为 $\delta_s$。

稀疏最小二乘的 IHT 为

$$
x_{t+1}
=H_s\left[x_t-\gamma\Phi^\top(\Phi x_t-y)\right].
$$

RIP 表明测量算子在稀疏子空间的并集上近似保持等距，因此可以防止不同的稀疏信号坍缩成相同的观测。

**与书本对应的补充。** UML 精确说明了 RIP 对恢复的含义。若 $W$ 是满足 $\delta<1$ 的 $(\delta,2s)$-RIP 矩阵，则一个 $s$-稀疏向量是所有与 $y=Wx$ 相容的向量中唯一最稀疏的向量。在更强的数值 RIP 条件下，同一个向量也可以由下列凸规划恢复：

$$
\min_v\|v\|_1
\quad\text{subject to}\quad
Wv=y.
$$

对于非稀疏的 $x$，重构误差由其绝对值最大的 $s$ 个坐标之外的 $\ell_1$ 尾部控制。一个高斯测量矩阵大约需要

$$
s\log(d/\delta)/\varepsilon^2
$$

行，才能以高概率满足 $(\varepsilon,s)$-RIP 保证。LTfFP 将 RIP 与用于获得 Lasso 快速速率的 restricted eigenvalue 和 incoherence 条件联系起来。

## 4. 从稀疏到低秩、聚类、列选择与 Nystrom

> **页码：** Slides：L14，第 7–14 页。书本补充：LTfFP 第 1.1.4 节，书内第 6–7 页（PDF 第 22–23 页），以及第 7.4.2 节，书内第 197–198 页（PDF 第 213–214 页）；UML 第 22 章，电子版第 307–320 页，以及第 25.3.1 节，电子版第 369–370 页。

核心类比是

$$
\operatorname{rank}(X)=\|\sigma(X)\|_0.
$$

因此，低秩就是奇异值向量的稀疏性。

对于数据 $X\in\mathbb R^{d\times n}$，k-means 可以写成

$$
\min_{M,C}\frac12\|X-MC\|_F^2,
$$

其中 $M$ 的列是质心，$C$ 的列是 one-hot 分配向量。协同聚类同时对行和列聚类，使用

$$
X\approx RMC
$$

列子集选择保留真实列，可用于可解释性、sketching、特征或传感器选择以及核近似。对于正半定核矩阵，Nystrom 近似为

$$
\widetilde K
=K_{:,C}K_{C,C}^\dagger K_{C,:}.
$$

基于行列式的分布

$$
\Pr(C)\propto\det(K_{C,C})
$$

偏好具有多样性的列。

**与书本对应的补充。** UML 将 k-means 表示为一种稀疏编码器—解码器：编码器输出一个 one-hot 向量，以标明最近的质心；解码器则返回该质心。这把 k-means 与 PCA 放到同一个重构框架中，同时区分了离散稀疏编码和线性低维编码。LTfFP 的核方法章节把 Nystrom 构造解释为：把预测器限制在所选核截面的张成空间中。若选取 $q=|C|$ 列，它会用一个秩为 $q$ 的特征表示取代稠密核线性代数，并把工作量降至大约 $O(nq^2)$。基于行列式的采样仍然只是 slides 中的材料，而不是两本参考书中推导出的结果。

## 5. 矩阵补全与低秩形式

> **页码：** Slides：L14，第 15–24 页。书本补充：LTfFP 练习 8.14–8.15，书内第 244 页（PDF 第 260 页），用于核范数阈值与因子化。LTfFP 与 UML 都没有展开矩阵补全采样理论或用户—物品补全保证。

设 $A\in\mathbb R^{n\times m}$ 只在 $\Omega$ 上被观测。一个合理的估计器应当对行、列重新编号保持等变性。逐元素独立的 ridge 回归不能完成矩阵补全，因为它没有在不同元素之间建立耦合。

直接的低秩形式为

$$
\min_{\widehat A}
\sum_{(i,j)\in\Omega}
(Y_{ij}-\widehat A_{ij})^2
\quad\text{subject to}\quad
\operatorname{rank}(\widehat A)\le k.
$$

凸松弛使用核范数

$$
\|A\|_*=\sum_i\sigma_i(A):
$$

$$
\min_A
\frac12\|P_\Omega(A-Y)\|_F^2
+\lambda\|A\|_*.
$$

其近端算子对奇异值进行软阈值处理。实用的非凸替代方法把矩阵分解为

$$
A=UV^\top
$$

并分别在 $U$ 和 $V$ 上最小化观测元素的损失。精确恢复要求低秩、足够随机的观测、奇异向量的 incoherence 等假设；slides 把这些作为理论问题提出，而没有证明样本复杂度定理。

**与书本对应的补充。** 对一个完全观测矩阵

$$
Y=U\operatorname{Diag}(\sigma_i)V^\top,
$$

核范数近端问题

$$
\min_\Theta
\left\{
\frac12\|Y-\Theta\|_F^2+\tau\|\Theta\|_*
\right\}
$$

的解为

$$
\Theta^\star
=
U\operatorname{Diag}((\sigma_i-\tau)_+)V^\top.
$$

因此，奇异值软阈值正是逐坐标软阈值的矩阵类比。LTfFP 还给出因子化恒等式

$$
\boxed{
\|M\|_*
=
\min_{M=UV^\top}
\frac12(\|U\|_F^2+\|V\|_F^2).
}
$$

它解释了为什么因子的 Frobenius 正则化与乘积的核范数正则化相联系。这些事实为 slides 中的形式提供了依据，但并没有给出矩阵补全恢复定理。

## 6. 截断 SVD 与交替最小化

> **页码：** Slides：L14，第 25–27 页。书本补充：LTfFP 第 1.1.4 与 3.9 节，书内第 6–7、66–68 页（PDF 第 22–23、82–84 页）；UML 第 23.1 节与附录 C，电子版第 324–328、430–434 页。书中的 AltMin 补充限于完整观测的 PCA 特例，不包含 slides 所讨论的一般低秩或缺失数据景观。

对于

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^\top,
\qquad
\sigma_1\ge\cdots\ge\sigma_r,
$$

定义

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^\top.
$$

Eckart-Young-Mirsky 定理指出，对于每一个酉不变范数，$A_k$ 都是最佳秩-$k$ 近似。特别地，

$$
\|A-A_k\|_2=\sigma_{k+1},
$$

$$
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2,
$$

以及

$$
\|A-A_k\|_*=\sum_{i>k}\sigma_i.
$$

因子化 $X=UV^\top$ 把

$$
\min_{\operatorname{rank}(X)\le k}f(X)
$$

转化为关于 $U,V$ 的无约束但非凸的问题。交替最小化固定一个因子并优化另一个因子。全局保证需要针对具体问题的假设；仅有低秩性并不足以推出这些保证。

**与书本对应的补充。** UML 通过 PCA 重构问题推导截断 SVD，并说明存储因子只需大约 $(n+m)k$ 个数，而不是 $nm$ 个数。LTfFP 和 UML 都把奇异值尾部的平方和与被舍弃的方差联系起来。对于完全观测的因子化问题

$$
\min_{A,D}\|\Phi-AD\|_F^2,
$$

交替最小二乘更新为

$$
A\leftarrow \Phi D^\top(DD^\top)^\dagger,
\qquad
D\leftarrow(A^\top A)^\dagger A^\top\Phi.
$$

LTfFP Exercise 3.9 指出，在这一完全观测的 PCA 情形中，对于几乎所有初始化，该方法都会收敛到全局最佳主子空间。这一特殊结论不能自动转移到缺失数据矩阵补全；在后一问题中，损失面与收敛结论需要额外的采样和 incoherence 假设。

## 7. 矩阵感知

> **页码：** Slides：L14，第 28 页。书本补充：LTfFP 第 12.3.3–12.3.4 节，书内第 373–375 页（PDF 第 389–391 页），给出一个正半定因子化特例；UML 第 23.3 节，电子版第 330–338 页，给出向量压缩感知类比。

矩阵感知观测

$$
\mathcal A(M)
=
[\langle A_1,M\rangle,\ldots,\langle A_m,M\rangle]^\top
$$

并试图恢复一个低秩 $M$。秩约束梯度法通过截断 SVD 完成投影：

$$
M_{t+1}
=P_{\operatorname{rank}\le r}
\left[
M_t-\eta\mathcal A^*(\mathcal A(M_t)-b)
\right].
$$

若 $M\succeq0$，则 $\|M\|_*=\operatorname{tr}(M)$，从而可以使用半定松弛。

**与书本对应的补充。** LTfFP 研究了如下正半定特例：

$$
F(W)
=
\frac1n\sum_{i=1}^n
(\langle WW^\top,X_i\rangle-y_i)^2
=G(WW^\top),
$$

其中 $G(M)$ 关于 $M\succeq0$ 是凸的。因子化 $M=WW^\top$ 把一个凸矩阵问题转化成非凸参数化；在书中的附加假设下，梯度流结果仍然能够到达全局最优值，并可能表现出最小核范数的隐式偏置。这是一个 PSD 特例，而不是适用于 slides 中每一个感知算子的定理。

UML 的压缩感知章节给出向量类比：

$$
\text{sparse vector}
\leftrightarrow
\text{low-rank matrix},
\qquad
\ell_1\text{ norm}
\leftrightarrow
\text{nuclear norm},
$$

以及

$$
\text{hard coordinate thresholding}
\leftrightarrow
\text{truncated SVD}.
$$

向量 RIP 启发了作用于低秩差分上的 matrix-RIP 条件，但两本参考书都没有展开这一矩阵特定条件。

## 8. 张量估计

> **页码：** Slides：L14，第 29–33 页。直接书本补充：LTfFP 与 UML 都没有展开张量分解或张量 deflation。

对于向量 $u,v,w,x$，

$$
T=u\otimes v\otimes w\otimes x,
\qquad
T_{ijkl}=u_iv_jw_kx_l.
$$

对于一个正交可分解的四阶张量

$$
T=\sum_{i=1}^ru_i^{\otimes4},
$$

其收缩包括

$$
T(v,v,v,v)=\sum_i(u_i^\top v)^4,
$$

$$
T(I,v,v,v)=\sum_i(u_i^\top v)^3u_i,
$$

以及

$$
T(I,I,v,v)=\sum_i(u_i^\top v)^2u_iu_i^\top.
$$

可以通过

$$
\max_{\|u\|=1}T(u,u,u,u).
$$

恢复一个分量。全局最大点是 $\pm u_i$。找到一个分量后，使用

$$
T\leftarrow T-u_i^{\otimes4}.
$$

进行 deflation。对于所展示的最大化问题，slides 中的 “all local minima are global” 应理解为 “all local maxima are global”；或者，也可以把它理解为对负目标进行最小化时的陈述。一般张量并不存在类似矩阵的通用 SVD 理论。

## 9. LoRA

> **页码：** Slides：L14，第 34–35 页。书本补充：LTfFP 练习 8.15，书内第 244 页（PDF 第 260 页），给出一般核范数因子化恒等式。两本书都没有讨论 LoRA 本身。

LoRA 冻结预训练矩阵 $W_0$，并学习低秩更新

$$
\Delta W=BA.
$$

因此

$$
h=W_0x+BAx.
$$

对于

$$
W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
$$

选择

$$
A\in\mathbb R^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

可训练参数数量从 $d_{\mathrm{out}}d_{\mathrm{in}}$ 降为

$$
r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

随机初始化 $A$ 并令 $B=0$，会得到 $BA=0$，因此初始模型与预训练模型完全一致。

**与书本对应的补充。** 把 LTfFP 的一般因子化恒等式应用于 LoRA 更新，可得

$$
\|\Delta W\|_*
=
\min_{\Delta W=BA}
\frac12(\|B\|_F^2+\|A\|_F^2).
$$

对于任意一个具体因子化，

$$
\|BA\|_*
\le
\frac12(\|B\|_F^2+\|A\|_F^2).
$$

因此，对两个可训练因子施加 Frobenius 惩罚，会间接控制更新矩阵的核范数。这是对一般低秩恒等式的一项应用，而不是两本书中陈述的 LoRA 结论。

## 10. 主成分分析

> **页码：** Slides：L14，第 36–56 页。书本补充：LTfFP 第 3.9 节，书内第 66–68 页（PDF 第 82–84 页）；UML 第 23.1 节，电子版第 324–328 页，以及第 23.6 节，电子版第 339 页。

对于数据 $x_1,\ldots,x_n\in\mathbb R^d$，定义

$$
\bar x=\frac1n\sum_i x_i,
\qquad
S=\frac1n\sum_i(x_i-\bar x)(x_i-\bar x)^\top.
$$

最大方差意义下的第一主方向求解

$$
\max_{\|u\|=1}u^\top Su,
$$

因此它是 $S$ 的最大特征值所对应的特征向量。对于 $k$ 个方向，求解

$$
\max_{U^\top U=I_k}\operatorname{tr}(U^\top SU),
$$

其解由前 $k$ 个特征向量组成。

最小重构误差视角选择一个 $k$ 维仿射子空间。它的最优偏移是样本均值，最小误差为

$$
\sum_{j=k+1}^d\lambda_j(S).
$$

由于总方差固定，

$$
\sum_{j=1}^k\lambda_j(S)
+\sum_{j=k+1}^d\lambda_j(S)
=\operatorname{tr}(S),
$$

所以最大化保留方差等价于最小化重构误差。

中心化后的编码器和解码器为

$$
z=U^\top(x-\bar x),
$$

$$
\widehat x
=\bar x+Uz
=\bar x+UU^\top(x-\bar x).
$$

若

$$
X_c=[x_1-\bar x,\ldots,x_n-\bar x]
=U\Sigma V^\top,
$$

则

$$
S=\frac1nX_cX_c^\top,
\qquad
\lambda_i(S)=\frac{\sigma_i^2}{n}.
$$

因此，PCA、顶部奇异向量与最佳秩-$k$ 近似是同一个构造。

**与书本对应的补充。** UML 证明，在所有线性编码器 $W\in\mathbb R^{k\times d}$ 和线性解码器 $U\in\mathbb R^{d\times k}$ 中，可以选取一个最优解，使解码器各列标准正交且 $W=U^\top$。若 $d\gg n$，可以对更小的 Gram 矩阵 $X_c^\top X_c$ 进行对角化，再通过 $X_c$ 把其特征向量映射回去，从而降低特征分解成本。同一个观察还会导出 kernel PCA，因为计算中只需要内积。

LTfFP 还通过 PCA 后接最小二乘给出了一个监督学习方面的提醒。若 $V$ 包含前 $k$ 个样本主方向，且 $\widehat\Sigma=\Phi^\top\Phi/n$，则期望预测误差分解为

$$
\frac1n\mathbb E_\varepsilon
\|\Phi V\widehat\eta-\Phi\theta^\star\|^2
=
\frac{\sigma^2k}{n}
+
(\theta^\star)^\top
(I-VV^\top)\widehat\Sigma(I-VV^\top)\theta^\star.
$$

第一项是拟合 $k$ 个坐标产生的方差；第二项是舍弃其余方向产生的偏差。因此，PCA 对无监督平方重构是最优的，但它不一定保留下游预测任务最需要的方向。
