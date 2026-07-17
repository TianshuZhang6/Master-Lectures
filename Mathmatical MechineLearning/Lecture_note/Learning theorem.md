# Learning Theorem：统计学习理论完整笔记

> 本笔记把“学习理论概述”的总结性回答与两份教材资料重新组织为一条可独立学习的主线。重点是二分类中的 ERM、PAC/agnostic PAC、一致收敛、增长函数、VC 维、Sauer 引理、统计学习基本定理、Rademacher 复杂度与算法稳定性，并补充 Bayes 决策、优化误差、结构风险最小化和 double descent。

## 来源与记号

- **原总结**：用户提供的“学习理论概述”回答，负责课程主线与 slides 中的定理结构。
- **Bach**：Francis Bach, *Learning Theory from First Principles*。重点参考第 2 章（书页 28–37；PDF 第 44–53 页）、第 4 章（书页 84–102；PDF 第 100–118 页）和第 12.2 节（书页 355–360；PDF 第 371–376 页）。
- **UML-Sol**：Alon Gonen / Dana Rubinstein, *Understanding Machine Learning — Solution Manual*。重点参考第 3–6 章解答（PDF 第 3–20 页）和第 13 章“Regularization and Stability”（PDF 第 34–37 页）。

本笔记统一写

$$
R(h)=L_{\mathbb P}(h),\qquad \widehat R_S(h)=L_S(h),
$$

二者分别表示总体风险与经验风险。样本量统一记为 $n$，精度和置信参数分别记为 $\epsilon,\delta$。对数若无特别说明均为自然对数。

---

## 如何使用这份笔记

建议按下面顺序学习：

1. 先掌握风险、Bayes 预测器、ERM 与三类误差；
2. 再学习有限假设类的 PAC 证明，理解“固定坏假设 + union bound”；
3. 然后进入一致收敛、增长函数、VC 维与 Sauer 引理；
4. 最后学习两条更现代的泛化路线：Rademacher 复杂度与算法稳定性；
5. 用最后的“常见误区”和“自测题”检查是否真正掌握。

整条逻辑链为

$$
\boxed{
\text{Bayes 决策}
\to \text{ERM}
\to \text{泛化问题}
\to \text{PAC}
\to \text{一致收敛}
\to \text{VC/Rademacher}
\to \text{稳定性与隐式偏置}
}
$$

# 1. 监督学习的概率模型

令输入空间、标签空间和样本空间分别为

$$
\mathcal X,\qquad \mathcal Y,\qquad \mathcal Z=\mathcal X\times\mathcal Y.
$$

数据点 $Z=(X,Y)$ 服从未知分布 $\mathbb P$。训练集为

$$
S=(Z_1,\ldots,Z_n),\qquad Z_i=(X_i,Y_i)\overset{\mathrm{i.i.d.}}\sim\mathbb P.
$$

一个学习算法是从数据集到预测器的映射

$$
A:\mathcal Z^n\to\mathcal H,\qquad h_S=A(S).
$$

给定损失函数 $\ell$，预测器 $h$ 的总体风险为

$$
R(h)=\mathbb E_{Z\sim\mathbb P}[\ell(h,Z)],
$$

经验风险为

$$
\widehat R_S(h)=\frac1n\sum_{i=1}^n\ell(h,Z_i).
$$

二分类的 0-1 损失是

$$
\ell_{0/1}(h,(x,y))=\mathbf 1\{h(x)\neq y\},
$$

所以 $R(h)$ 是未来样本上的错误概率，而 $\widehat R_S(h)$ 是训练错误率。

对一个**预先固定且不依赖于训练集**的 $h$，有

$$
\mathbb E_S[\widehat R_S(h)]=R(h).
$$

但对数据依赖的 $h_S=A(S)$，通常不能写

$$
\mathbb E_S[\widehat R_S(h_S)]=\mathbb E_S[R(h_S)].
$$

学习算法恰恰会利用 $S$ 选择经验风险特别小的函数，这个选择效应就是泛化分析的难点。

# 2. Bayes 决策：如果知道分布，最优预测是什么

对给定输入 $x$ 和候选预测 $a\in\mathcal Y$，定义条件风险

$$
r(a\mid x)=\mathbb E[\ell(Y,a)\mid X=x].
$$

由全期望公式，

$$
R(h)=\mathbb E_X[r(h(X)\mid X)].
$$

如果允许在所有可测函数中选择，那么每个 $x$ 可单独最小化条件风险：

$$
h^*(x)\in\arg\min_{a\in\mathcal Y}r(a\mid x).
$$

这就是 Bayes 预测器。其风险

$$
R^*=R(h^*)
=\mathbb E_X\left[\inf_{a\in\mathcal Y}\mathbb E[\ell(Y,a)\mid X]\right]
$$

称为 Bayes 风险。任意预测器都满足 $R(h)\ge R^*$，而

$$
R(h)-R^*
$$

称为 excess risk（超额风险）。Bayes 预测器可能不唯一，但所有 Bayes 预测器具有相同风险。

一个重要提醒是：只有在不受限制的全体函数中，风险才能逐点最小化。一旦限制到某个假设类 $\mathcal H$，不同 $x$ 处的预测往往由共同参数耦合，不能再逐点选择。

## 2.1 常见损失下的 Bayes 预测器

### 二分类 0-1 损失

令 $\mathcal Y=\{-1,+1\}$，并记

$$
\eta(x)=\mathbb P(Y=+1\mid X=x).
$$

预测 $+1$ 的条件错误率是 $1-\eta(x)$，预测 $-1$ 的条件错误率是 $\eta(x)$，因此

$$
h^*(x)=
\begin{cases}
+1,&\eta(x)\ge \frac12,\\
-1,&\eta(x)<\frac12.
\end{cases}
$$

Bayes 风险为

$$
R^*=\mathbb E[\min\{\eta(X),1-\eta(X)\}].
$$

它一般不为零；只有标签在给定 $X$ 后几乎确定时，Bayes 风险才为零。

### 非对称分类代价

若假阳性的代价为 $c_{\mathrm{FP}}$，假阴性的代价为 $c_{\mathrm{FN}}$，则预测 $+1$ 的条件代价为

$$
c_{\mathrm{FP}}(1-\eta(x)),
$$

预测 $-1$ 的条件代价为

$$
c_{\mathrm{FN}}\eta(x).
$$

所以阈值不再是 $1/2$，而是

$$
h^*(x)=+1
\iff
\eta(x)\ge
\frac{c_{\mathrm{FP}}}{c_{\mathrm{FP}}+c_{\mathrm{FN}}}.
$$

### 回归

- 平方损失 $(Y-a)^2$：Bayes 预测器是条件均值 $\mathbb E[Y\mid X=x]$，Bayes 风险是期望条件方差。
- 绝对损失 $|Y-a|$：Bayes 预测器是条件中位数。
- pinball 损失 $\alpha(Y-a)_+ +(1-\alpha)(a-Y)_+$：Bayes 预测器是条件 $\alpha$-分位数。

因此“最佳预测”由损失决定；同一条件分布在不同任务代价下会产生不同的 Bayes 决策。

# 3. ERM、归纳偏置与过拟合

给定假设类 $\mathcal H$，经验风险最小化（ERM）定义为

$$
\widehat h_S\in\arg\min_{h\in\mathcal H}\widehat R_S(h).
$$

ERM 用可观测的经验风险替代不可观测的总体风险。核心问题是

$$
\boxed{\widehat R_S(\widehat h_S)\text{ 小，是否能推出 }R(\widehat h_S)\text{ 小？}}
$$

若 $\mathcal H$ 包含所有从 $\mathcal X$ 到 $\mathcal Y$ 的函数，算法可以记住训练样本，在训练集外任意预测，从而使训练误差为零但测试误差很大。这说明学习离不开归纳偏置。

归纳偏置可以来自：

- 显式限制假设类，如线性分类器、固定深度树、范数受限网络；
- 显式正则化，如 $\lambda\|w\|_2^2$ 或 $\lambda\|w\|_1$；
- 优化算法的隐式偏置，如梯度下降在多个插值解中选择最小范数解；
- 数据表示、初始化、早停、数据增强等。

PAC 理论首先研究假设类带来的归纳偏置；稳定性和现代过参数化理论则进一步研究算法带来的归纳偏置。

### 0-1 损失与凸替代损失

0-1 损失直接对应分类错误，却不连续、通常难以优化。实际分类常令模型输出分数 $f(x)\in\mathbb R$，再最小化 margin loss $\phi(Yf(X))$，例如

$$
\phi_{\mathrm{hinge}}(u)=(1-u)_+,
\qquad
\phi_{\mathrm{logistic}}(u)=\log(1+e^{-u}).
$$

替代风险小不自动等于分类风险小；需要 classification calibration。校准性质保证：当替代损失的超额风险趋于零时，经符号阈值化后的 0-1 超额风险也趋于零。因而理论分析常分成“优化/泛化替代风险”和“用校准函数转回分类风险”两步。

# 4. 风险分解：近似、估计与优化误差

令

$$
h_{\mathcal H}^*\in\arg\min_{h\in\mathcal H}R(h).
$$

正确的 Bayes 基准分解是

$$
\boxed{
R(\widehat h_S)-R^*
=
\underbrace{R(\widehat h_S)-R(h_{\mathcal H}^*)}_{\text{估计误差}}
+
\underbrace{R(h_{\mathcal H}^*)-R^*}_{\text{近似误差}}.
}
$$

原总结把 $R(\widehat h_S)$ 直接写成近似误差与估计误差之和，只有在 $R^*=0$ 或把风险本身相对于零定义时才完全准确。通常应写“超额风险”的分解。

若实际算法只得到 $\rho$-近似 ERM：

$$
\widehat R_S(\widetilde h_S)
\le
\inf_{h\in\mathcal H}\widehat R_S(h)+\rho,
$$

则

$$
\begin{aligned}
R(\widetilde h_S)-R(h_{\mathcal H}^*)
&=(R-\widehat R_S)(\widetilde h_S)
+[\widehat R_S(\widetilde h_S)-\widehat R_S(h_{\mathcal H}^*)]
+(\widehat R_S-R)(h_{\mathcal H}^*)\\
&\le 2\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|+\rho.
\end{aligned}
$$

因此更完整的结构是

$$
\boxed{
\text{总超额风险}
\le
\text{近似误差}
+\text{统计估计误差}
+\text{优化误差}.
}
$$

扩大 $\mathcal H$ 通常降低近似误差，却使统一控制整个类更困难；增加样本量通常降低估计误差；改进优化器或增加迭代主要降低优化误差。

## 4.1 欠拟合、过拟合与 double descent

经典复杂度曲线中：

- 模型太小：近似误差大，训练和测试风险都高；
- 模型适中：近似误差和估计误差平衡；
- 模型太大：训练风险继续下降，但测试风险因过拟合而上升。

现代过参数化模型中常出现 double descent：测试风险在插值阈值附近升高，进入更强的过参数化区后又下降。关键不是“参数多必然泛化差”，而是：

1. 参数数量不等于有效复杂度；
2. 在许多零训练误差解中，优化算法只选择其中一个；
3. 被选解的范数、间隔、稳定性与数据几何会影响泛化；
4. 显式正则化可能减弱甚至消除 double descent。

Bach 对高斯线性回归给出可计算的例子。设 $X\sim\mathcal N(0,I_d)$，$Y=X^\top\theta^*+\varepsilon$，噪声方差为 $\sigma^2$，梯度下降选择最小 $\ell_2$ 范数 ERM。则在 $d\le n-2$ 时

$$
\mathbb E[R(\widehat\theta)]
=\sigma^2\frac{d}{n-d-1},
$$

在 $d\ge n+2$ 时

$$
\mathbb E[R(\widehat\theta)]
=\sigma^2\frac{n}{d-n-1}
+\frac{d-n}{d}\|\theta^*\|_2^2.
$$

当 $d\approx n$ 时逆 Wishart 矩阵的期望发散，形成插值峰；在 $d>n$ 后方差下降但偏差变化，于是可能出现第二次下降。这个现象依赖最小范数插值器；换一个 ERM 解不一定出现相同曲线。

# 5. 可实现情形与有限假设类

可实现性假设是

$$
\exists h^*\in\mathcal H,\qquad R(h^*)=0.
$$

于是 $h^*$ 在 i.i.d. 训练集上的经验风险几乎必然为零。任意 ERM 也满足

$$
\widehat R_S(\widehat h_S)=0.
$$

但这不意味着 $\widehat h_S=h^*$，更不意味着 $R(\widehat h_S)=0$：训练集上可能有很多一致假设，其中一些在训练集外很差。

若 $|\mathcal H|<\infty$，则有最基本的样本复杂度定理：当

$$
n\ge \frac{\log|\mathcal H|+\log(1/\delta)}{\epsilon},
$$

任意 ERM 解以至少 $1-\delta$ 的概率满足

$$
R(\widehat h_S)\le\epsilon.
$$

这个结论只保证统计可学习性，并不保证计算上能高效求出 ERM。

## 5.1 有限类定理的完整证明

定义坏假设集合

$$
\mathcal H_B=\{h\in\mathcal H:R(h)>\epsilon\}.
$$

若 ERM 失败，则它是一个坏假设且仍与训练集一致。因此

$$
\{R(\widehat h_S)>\epsilon\}
\subseteq
\bigcup_{h\in\mathcal H_B}\{\widehat R_S(h)=0\}.
$$

固定 $h\in\mathcal H_B$。一个样本点被它正确分类的概率小于 $1-\epsilon$。由独立性，

$$
\mathbb P(\widehat R_S(h)=0)
=(1-R(h))^n
\le(1-\epsilon)^n
\le e^{-n\epsilon}.
$$

对所有坏假设使用 union bound：

$$
\begin{aligned}
\mathbb P(R(\widehat h_S)>\epsilon)
&\le \sum_{h\in\mathcal H_B}\mathbb P(\widehat R_S(h)=0)\\
&\le |\mathcal H|e^{-n\epsilon}.
\end{aligned}
$$

令右侧不超过 $\delta$，得到

$$
n\ge\frac{\log(|\mathcal H|/\delta)}{\epsilon}.
$$

这个证明模板非常重要：先固定一个坏假设求它“骗过样本”的概率，再用 union bound 同时排除所有坏假设。

# 6. PAC 与 agnostic PAC

## 6.1 PAC 学习

在可实现情形下，若存在算法 $A$ 和样本复杂度函数 $m_{\mathcal H}(\epsilon,\delta)$，使得对任意满足可实现性的分布，当

$$
n\ge m_{\mathcal H}(\epsilon,\delta)
$$

时都有

$$
\mathbb P_S(R(A(S))\le\epsilon)\ge1-\delta,
$$

则称 $\mathcal H$ PAC 可学习。

- Probably：随机训练集可能不具代表性，因此允许失败概率 $\delta$。
- Approximately：有限样本只要求误差不超过 $\epsilon$。

## 6.2 Agnostic PAC 学习

现实中可能有标签噪声、模型错设或非零 Bayes 风险。此时目标应改为接近类内最优：

$$
\mathbb P_S\left(
R(A(S))
\le
\inf_{h\in\mathcal H}R(h)+\epsilon
\right)
\ge1-\delta.
$$

Agnostic 的含义是：不假设数据由 $\mathcal H$ 中某个零风险函数生成。

PAC 定义中的量词顺序十分重要：算法必须在不知道 $\mathbb P$ 的情况下，对规定问题族中的每个分布都满足保证。对每个分布分别存在一个专用算法，不等于存在一个统一学习算法。

### Proper、improper 与一致性

- **Proper learner**：输出必须属于 $\mathcal H$。
- **Improper learner**：允许输出 $\mathcal H$ 外的预测器，只要风险达到类内最优基准。
- **Universal consistency**：对每个固定分布 $\mathbb P$，当 $n\to\infty$ 时超额风险趋于零，但收敛速度可以依赖 $\mathbb P$。
- **Uniform consistency over $\mathcal P$**：要求 $\sup_{\mathbb P\in\mathcal P}\mathbb E[R(A(S))-R_{\mathbb P}^*]\to0$。

NFL 排除的是在毫无结构的巨大问题族上存在统一速率，并不排除每个固定分布上的 universal consistency。若只有期望界，可用 Markov 不等式转成高概率界，但会产生较差的 $1/\delta$ 依赖；也可通过独立重复训练、再用验证集选择来做 confidence boosting。

## 6.3 有限类的 agnostic 样本复杂度

对固定 $h$ 和 0-1 损失，Hoeffding 不等式给出

$$
\mathbb P\left(|R(h)-\widehat R_S(h)|>t\right)
\le2e^{-2nt^2}.
$$

对有限 $\mathcal H$ 使用 union bound，

$$
\mathbb P\left(
\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|>t
\right)
\le2|\mathcal H|e^{-2nt^2}.
$$

令 $t=\epsilon/2$。只要

$$
n\ge \frac{2}{\epsilon^2}
\log\frac{2|\mathcal H|}{\delta},
$$

就以至少 $1-\delta$ 的概率有

$$
\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|\le\frac\epsilon2.
$$

在该事件上，若 $\widehat h_S$ 是 ERM、$h_{\mathcal H}^*$ 是总体风险最优函数，则

$$
R(\widehat h_S)
\le\widehat R_S(\widehat h_S)+\frac\epsilon2
\le\widehat R_S(h_{\mathcal H}^*)+\frac\epsilon2
\le R(h_{\mathcal H}^*)+\epsilon.
$$

可实现界是 $1/\epsilon$，agnostic 界是 $1/\epsilon^2$：前者只排除“真实误差至少 $\epsilon$ 却零训练误差”的假设，概率按 $e^{-n\epsilon}$ 衰减；后者要估计一般 Bernoulli 均值，偏差尾界按 $e^{-n\epsilon^2}$ 衰减。

# 7. No-Free-Lunch：为什么归纳偏置不可避免

No-Free-Lunch 的核心不是“机器学习不可能”，而是：若允许任意目标标记且不给结构假设，就不存在对所有问题都好的统一算法。

一种典型构造是：取一个比训练集大得多的有限集合，在训练集没有覆盖的点上随机选择目标标签。算法在未见点上没有信息，因此平均只能猜测。

UML-Sol 给出的常数版本说明：若假设类能打散大小 $d=2n$ 的集合，则对任意算法，都存在一个可实现分布，使其期望错误至少为 $1/4$；进一步可推出以至少 $1/7$ 的概率，错误至少为 $1/8$。常数不是重点，重点是存在与样本量无关的正失败概率和正误差。

NFL 告诉我们：

- 必须限制目标函数、数据分布、假设类或学习算法中的至少一项；
- “另一个算法能成功”通常只是存在性陈述，它可能事先编码了目标函数；
- 学习理论的复杂度概念是在刻画我们究竟加入了多少结构。

# 8. 一致收敛

对固定 $h$，经验风险会集中到总体风险。但 ERM 输出依赖样本，所以需要同一个训练集同时代表整个假设类。

称 $S$ 是 $\epsilon$-representative，如果

$$
\sup_{h\in\mathcal H}
|\widehat R_S(h)-R(h)|
\le\epsilon.
$$

若存在 $m_{\mathcal H}^{\mathrm{UC}}(\epsilon,\delta)$，使得对所有分布，当 $n$ 足够大时

$$
\mathbb P_S\left(
\sup_{h\in\mathcal H}|\widehat R_S(h)-R(h)|\le\epsilon
\right)\ge1-\delta,
$$

则称 $\mathcal H$ 具有一致收敛性质。

在 $\epsilon/2$-代表性事件上，ERM 满足

$$
R(\widehat h_S)
\le\inf_{h\in\mathcal H}R(h)+\epsilon.
$$

因此

$$
\boxed{\text{一致收敛}\Longrightarrow\text{ERM 是 agnostic PAC 学习器}.}
$$

逻辑上的关键差别是

$$
\forall h:\ \mathbb P(|\widehat R(h)-R(h)|\le\epsilon)\ge1-\delta
$$

并不能自动推出

$$
\mathbb P\left(\forall h:\ |\hat R(h)-R(h)|\le\epsilon\right)\ge1-\delta.
$$

后者才足以处理数据依赖的 ERM。

# 9. 无限假设类的有效规模：限制与增长函数

无限假设类不一定复杂。例如一维阈值函数有无限多个阈值，却只有 VC 维 1。关键不是 $|\mathcal H|$，而是 $\mathcal H$ 在有限样本上能产生多少种不同预测。

给定点集

$$
C=(x_1,\ldots,x_m),
$$

定义限制集

$$
\mathcal H|_C
=\{(h(x_1),\ldots,h(x_m)):h\in\mathcal H\}.
$$

增长函数（也常记为 shattering coefficient）为

$$
\Pi_{\mathcal H}(m)
=\max_{x_1,\ldots,x_m\in\mathcal X}
|\mathcal H|_{(x_1,\ldots,x_m)}|.
$$

二分类中总有

$$
1\le\Pi_{\mathcal H}(m)\le2^m.
$$

增长函数把可能无限的全局函数类，压缩成有限样本上的“有效标签数”。在泛化证明中，union bound 最终只需要覆盖这些不同标签向量，而不需要覆盖参数空间中的每个点。

# 10. 打散与 VC 维

若对某个大小为 $m$ 的集合 $C$，有

$$
|\mathcal H|_C|=2^m,
$$

则称 $\mathcal H$ 打散（shatter）了 $C$。等价地，对每个标签向量 $y\in\{0,1\}^m$，都存在一个依赖于 $y$ 的 $h_y\in\mathcal H$，使得

$$
h_y(x_i)=y_i,\qquad i=1,\ldots,m.
$$

注意量词是“对每种标记，存在一个对应的函数”，不是“存在一个函数同时实现所有标记”。

VC 维定义为最大可打散点集的大小：

$$
\operatorname{VCdim}(\mathcal H)
=\sup\{|C|:\mathcal H\text{ 打散 }C\}.
$$

要证明 $\operatorname{VCdim}(\mathcal H)=d$，必须完成两部分：

1. **下界**：构造一个大小为 $d$ 的可打散集合；
2. **上界**：证明任意大小为 $d+1$ 的集合都不能被打散。

只证明某一个 $d+1$ 点集合不能打散，通常不足以得到上界。

若 $\mathcal H$ 有限且能打散 $d$ 个点，则至少需要 $2^d$ 个不同假设，所以

$$
\operatorname{VCdim}(\mathcal H)\le\log_2|\mathcal H|.
$$

反过来不成立：$|\mathcal H|$ 可以很大甚至无限，而 VC 维仍很小。

集合系统写法与此完全等价。令

$$
\mathcal F=\{A_h:A_h=\{x:h(x)=1\},\ h\in\mathcal H\}.
$$

在点集 $C$ 上的 trace 是 $\{A\cap C:A\in\mathcal F\}$。打散 $C$ 就是该 trace 等于幂集 $2^C$：每种标签向量恰好对应“被标为 1 的点组成的子集”。

## 10.1 典型 VC 维结果

| 假设类 | VC 维 | 核心理由 |
|---|---:|---|
| 一维单向阈值 $\mathbf 1\{x\ge t\}$ | $1$ | 一个点可打散；两个有序点不能形成反向标签 |
| 一维仿射符号 $\operatorname{sign}(wx+b)$ | $2$ | 可选择两种方向；有序标签最多改变一次 |
| 一维区间 $\mathbf 1\{a\le x\le b\}$ | $2$ | 两点可任意选择；三点不能只选两端而排除中间 |
| $\mathbb R^d$ 中齐次半空间 | $d$ | 标准基给下界；线性相关性给上界 |
| $\mathbb R^d$ 中仿射半空间 | $d+1$ | 仿射独立单纯形给下界；Radon 定理给上界 |
| $\mathbb R^d$ 中轴对齐矩形 | $2d$ | 每个坐标有独立的上下边界 |
| $\mathbb R^d$ 中欧氏球 | $d+1$ | 可提升为一个 $(d+1)$ 维线性阈值问题 |
| $d$ 个布尔变量的 parity 类 | $d$ | 类大小为 $2^d$，标准基达到上界 |
| $\mathbb R^d$ 中次数不超过 $k$ 的全多项式阈值类 | $\binom{d+k}{k}$ | 提升到全部单项式组成的特征空间 |

这些结果说明“参数数量”和 VC 维常常同阶，但并非普遍相等。

## 10.2 例一：阈值函数

考虑

$$
\mathcal H=\{h_t(x)=\mathbf 1\{x\ge t\}:t\in\mathbb R\}.
$$

一个点 $x_1$ 可被标为 0 或 1，因此 VC 维至少为 1。

对任意 $x_1<x_2$，若 $h_t(x_1)=1$，则必有 $h_t(x_2)=1$，所以标签 $(1,0)$ 不可能。因此没有两个点可被打散，

$$
\operatorname{VCdim}(\mathcal H)=1.
$$

在 $m$ 个有序点上，阈值只能在 $m+1$ 个位置切开，故

$$
\Pi_{\mathcal H}(m)=m+1.
$$

这是一类“无限但很简单”的标准例子。

若允许 $w$ 正负两种方向，得到一维仿射符号类 $\operatorname{sign}(wx+b)$。它在两个点上可实现全部四种标签，所以 VC 维为 2；在 $m$ 个有序点上的标签最多改变一次。递增阈值和递减阈值各给 $m+1$ 种模式，两个常量模式重复计算，因此

$$
\Pi_{\mathcal H}(m)=2(m+1)-2=2m.
$$

这解释了为什么“单向阈值 VC 维 1”和“完整一维仿射分类器 VC 维 2”并不矛盾。

## 10.3 例二：轴对齐矩形的 VC 维为 $2d$

考虑 $\mathbb R^d$ 中的轴对齐矩形分类器：矩形内为 1，外部为 0。

**下界。** 取 $2d$ 个点

$$
C=\{e_1,\ldots,e_d,-e_1,\ldots,-e_d\}.
$$

每个坐标的上边界可以决定是否包含 $+e_j$，下边界可以独立决定是否包含 $-e_j$，而其他点在该坐标上都为 0。因此任意标签都能实现，VC 维至少为 $2d$。

**上界。** 对任意 $2d+1$ 个点，最多只有 $2d$ 个点能分别成为某个坐标方向上的唯一最小点或唯一最大点。因此至少有一个点 $x$，在每个坐标上都夹在其余点的坐标范围内。若把 $x$ 标为 0、其余点标为 1，则任何包含其余所有点的轴对齐矩形也必包含 $x$，所以该标签不可实现。

因此

$$
\operatorname{VCdim}(\mathcal H_{\mathrm{rect}}^d)=2d.
$$

这个证明展示了 VC 维上界的常用技巧：找出由几何结构强迫的一个不可实现标签。

## 10.4 例三：仿射半空间的 VC 维为 $d+1$

仿射半空间类为

$$
h_{w,b}(x)=\mathbf 1\{w^\top x+b\ge0\}.
$$

**下界。** 取 $d+1$ 个仿射独立点，即一个非退化单纯形的顶点。增广向量 $(x_i,1)\in\mathbb R^{d+1}$ 线性独立。对任意符号 $y_i\in\{-1,+1\}$，可解线性方程

$$
w^\top x_i+b=y_i,
$$

所以所有二分都可实现。

**上界。** Radon 定理说明任意 $d+2$ 个点都可分成两个子集，使两者凸包相交。把两个子集标成相反类别，则若存在严格分离超平面，两凸包也会被分离，与交点矛盾。

因此

$$
\operatorname{VCdim}(\mathcal H_{\mathrm{halfspace}}^d)=d+1.
$$

二维情形就是 $3$：三个不共线点能打散，四点总有 XOR 或“内部点对外部点”的不可分标签。

## 10.5 一个参数也可能有无限 VC 维

考虑高频振荡函数族，例如由频率参数控制的

$$
h_m(x)=\mathbf 1\{\sin(2^m\pi x)>0\}.
$$

可以精心选择 $n$ 个实数，使它们二进制展开的不同位依次列出全部 $2^n$ 种标签向量。改变 $m$ 相当于读取不同二进制位，于是任意 $n$ 点都能构造出一个被打散的例子。

所以某些只有一个数值参数的函数族仍可满足

$$
\operatorname{VCdim}(\mathcal H)=\infty.
$$

结论是：参数个数只有在附加结构条件下才是合理复杂度代理。参数的编码方式、振荡能力、范数和算法选择都可能改变有效复杂度。

# 11. Sauer–Shelah–Perles 引理

若

$$
d=\operatorname{VCdim}(\mathcal H)<\infty,
$$

则对 $m>d$，

$$
\boxed{
\Pi_{\mathcal H}(m)
\le\sum_{i=0}^d\binom mi
\le\left(\frac{em}{d}\right)^d.
}
$$

于是

$$
\log\Pi_{\mathcal H}(m)
\le d\log\frac{em}{d}.
$$

这产生增长函数的二分现象：

- 若 VC 维无限，则对每个 $m$，$\Pi_{\mathcal H}(m)=2^m$；
- 若 VC 维有限，则增长函数最终至多按 $m^d$ 多项式增长。

长期来看，不存在稳定的中间指数增长率。

## 11.1 Sauer 引理的递推证明

令 $g_d(m)$ 表示 VC 维不超过 $d$ 的集合系统在 $m$ 个点上最多能产生多少种不同子集。

观察最后一个点 $x_m$。前 $m-1$ 个点上的模式分两类：

1. 只能以一种方式扩展到 $x_m$；
2. 既能把 $x_m$ 标为 0，也能标为 1。

第二类模式在前 $m-1$ 个点上形成的系统 VC 维至多为 $d-1$。否则它若能打散 $d$ 个点，再加入 $x_m$ 就能打散 $d+1$ 个点。

因此

$$
g_d(m)\le g_d(m-1)+g_{d-1}(m-1).
$$

配合边界条件

$$
g_0(m)=1,\qquad g_d(0)=1,
$$

以及 Pascal 恒等式

$$
\binom mi=\binom{m-1}{i}+\binom{m-1}{i-1},
$$

归纳得到

$$
g_d(m)\le\sum_{i=0}^d\binom mi.
$$

Sauer 引理的作用是把“有限 VC 维”转换成可进入概率界的“有限有效标签数”。

# 12. 统计学习基本定理

对二分类和 0-1 损失，以下条件等价：

1. $\mathcal H$ 具有一致收敛性质；
2. 任意 ERM 都是 agnostic PAC 学习器；
3. $\mathcal H$ 是 agnostic PAC 可学习的；
4. $\mathcal H$ 是 PAC 可学习的；
5. 任意 ERM 都是 PAC 学习器；
6. $\operatorname{VCdim}(\mathcal H)<\infty$。

因此

$$
\boxed{
\operatorname{VCdim}(\mathcal H)<\infty
\iff
\mathcal H\text{ 在二分类 0-1 损失下可学习}.
}
$$

主要蕴含关系为：

- 一致收敛 $\Rightarrow$ ERM agnostic PAC：用两次 $\epsilon/2$ 偏差；
- agnostic PAC $\Rightarrow$ PAC：可实现问题是特例；
- PAC $\Rightarrow$ VC 维有限：若 VC 维无限，在任意样本量上都可嵌入 NFL 构造；
- VC 维有限 $\Rightarrow$ 一致收敛：由 Sauer 引理和对称化/集中不等式得到。

最后一个方向是技术核心。

## 12.1 样本复杂度：分清“基础上界”和“最优阶”

设 $d=\operatorname{VCdim}(\mathcal H)$。

| 情形 | 代表性样本复杂度 |
|---|---|
| 有限 $\mathcal H$，可实现 | $\displaystyle n\ge\frac{\log|\mathcal H|+\log(1/\delta)}{\epsilon}$ |
| 有限 $\mathcal H$，agnostic | $\displaystyle n\ge\frac{2}{\epsilon^2}\log\frac{2|\mathcal H|}{\delta}$ |
| VC 维 $d$，可实现，经典 ERM 上界 | $\displaystyle O\!\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right)$ |
| VC 维 $d$，可实现，下界 | $\displaystyle \Omega\!\left(\frac{d+\log(1/\delta)}{\epsilon}\right)$ |
| VC 维 $d$，agnostic 的标准最优阶 | $\displaystyle \Theta\!\left(\frac{d+\log(1/\delta)}{\epsilon^2}\right)$ |

通过最基础的“增长函数 + union bound”推导时，常会多出 $\log n$ 或 $\log(1/\epsilon)$。更精细的 VC/Rademacher 技术可以去掉部分非本质对数。阅读定理时应区分：

- 这是一个方便证明的充分上界，还是匹配下界的最优阶；
- 保证针对任意 ERM，还是存在某个专门设计的学习器；
- 是可实现情形，还是 agnostic 情形。

# 13. 有限 VC 维为何推出一致收敛

定义

$$
Z(S)=\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|.
$$

证明主线如下。

### 第一步：ghost sample

引入独立同分布的

$$
S'=(Z_1',\ldots,Z_n')\sim\mathbb P^n.
$$

由于 $R(h)=\mathbb E_{S'}[\widehat R_{S'}(h)]$，Jensen 不等式给出

$$
\mathbb E_S Z(S)
\le
\mathbb E_{S,S'}
\sup_{h\in\mathcal H}
|\widehat R_{S'}(h)-\widehat R_S(h)|.
$$

### 第二步：对称化

引入独立 Rademacher 符号 $\sigma_i\in\{-1,+1\}$。因为交换 $Z_i,Z_i'$ 不改变联合分布，

$$
\widehat R_{S'}(h)-\widehat R_S(h)
$$

可转化为带随机符号的和。随机符号使每一项条件均值为零，可使用 Hoeffding/Massart 型工具。

### 第三步：限制到 $2n$ 个点

固定 $S,S'$ 后，表达式只依赖 $h$ 在 $S\cup S'$ 上的标签。不同标签向量至多有

$$
\Pi_{\mathcal H}(2n)
$$

个。因此无限 $\mathcal H$ 被替换成有限有效类。

### 第四步：集中与 union bound

对固定有效标签向量，随机符号和的尾概率按 $e^{-c n t^2}$ 衰减；对至多 $\Pi_{\mathcal H}(2n)$ 个向量使用 union bound。

## 13.1 增长函数泛化界

忽略不重要的绝对常数，上述证明得到

$$
\mathbb E_S\left[
\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|
\right]
\lesssim
\sqrt{\frac{\log\Pi_{\mathcal H}(2n)}{n}}.
$$

使用 McDiarmid 等集中不等式，可得到高概率版本

$$
\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|
\lesssim
\sqrt{
\frac{\log\Pi_{\mathcal H}(2n)+\log(1/\delta)}{n}
}
$$

以至少 $1-\delta$ 的概率成立。不同教材对常数和单边/双边版本的写法不同，稳定结构都是

$$
\text{偏差}
\sim
\sqrt{\frac{\text{有效复杂度}+\text{置信项}}{n}}.
$$

若 VC 维为 $d$，Sauer 引理给出

$$
\log\Pi_{\mathcal H}(2n)
\le d\log\frac{2en}{d},
$$

故

$$
\mathbb E Z(S)
\lesssim
\sqrt{\frac{d\log(2en/d)}{n}}
\to0.
$$

这完成“有限 VC 维 $\Rightarrow$ 一致收敛”的基本证明。

若最后只用 Markov 不等式从期望转成高概率界，会出现较松的 $1/\delta$ 依赖；使用有界差分集中可得到更合理的 $\sqrt{\log(1/\delta)/n}$。

# 14. Rademacher 复杂度：比计数更一般的容量工具

对实值函数类 $\mathcal G\subseteq\{g:\mathcal Z\to\mathbb R\}$，定义总体 Rademacher 复杂度

$$
\mathfrak R_n(\mathcal G)
=
\mathbb E_{S,\sigma}
\left[
\sup_{g\in\mathcal G}
\frac1n\sum_{i=1}^n\sigma_i g(Z_i)
\right],
$$

其中 $\sigma_i$ 独立且以相同概率取 $\pm1$。经验版本为

$$
\widehat{\mathfrak R}_S(\mathcal G)
=
\mathbb E_{\sigma}
\left[
\sup_{g\in\mathcal G}
\frac1n\sum_{i=1}^n\sigma_i g(Z_i)
\right].
$$

直观上，它衡量函数类与随机符号相关的能力：若函数类连随机噪声也能很好拟合，其容量就大。

在监督学习中应先构造**损失函数类**

$$
\ell\circ\mathcal H
=\{(x,y)\mapsto\ell(y,h(x)):h\in\mathcal H\},
$$

然后用它控制风险偏差。直接计算预测函数类的复杂度，必须再通过 contraction principle 转移到损失类。

## 14.1 对称化定理

令 $\mathbb Pg=\mathbb E[g(Z)]$，$\mathbb P_ng=n^{-1}\sum_i g(Z_i)$。ghost sample 与随机符号给出

$$
\mathbb E\sup_{g\in\mathcal G}
(\mathbb Pg-\mathbb P_ng)
\le2\mathfrak R_n(\mathcal G),
$$

以及反方向的同类界。证明骨架是

$$
\begin{aligned}
\mathbb E\sup_g(\mathbb Pg-\mathbb P_ng)
&\le
\mathbb E\sup_g\frac1n\sum_i[g(Z_i')-g(Z_i)]\\
&=
\mathbb E\sup_g\frac1n\sum_i\sigma_i[g(Z_i')-g(Z_i)]\\
&\le2\mathbb E\sup_g\frac1n\sum_i\sigma_i g(Z_i).
\end{aligned}
$$

若所有 $g(Z)\in[0,B]$，结合 McDiarmid 不等式可得：以至少 $1-\delta$ 的概率，对所有 $g\in\mathcal G$，

$$
\mathbb Pg
\le
\mathbb P_ng
+2\mathfrak R_n(\mathcal G)
+B\sqrt{\frac{\log(1/\delta)}{2n}}.
$$

一个可由数据计算的版本是

$$
\mathbb Pg
\le
\mathbb P_ng
+2\widehat{\mathfrak R}_S(\mathcal G)
+3B\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

因此 Rademacher 复杂度直接提供“经验风险 + 复杂度惩罚 + 置信项”的泛化证书。

## 14.2 Lipschitz 收缩与线性预测器

若对所有 $y$，函数 $u\mapsto\ell(y,u)$ 是 $G$-Lipschitz，则 contraction principle 给出

$$
\mathfrak R_n(\ell\circ\mathcal F)
\le G\mathfrak R_n(\mathcal F).
$$

于是只需研究预测函数类。

考虑线性类

$$
\mathcal F
=\{x\mapsto w^\top\phi(x):\|w\|_2\le D\},
\qquad \|\phi(x)\|_2\le R.
$$

由 Cauchy–Schwarz，

$$
\begin{aligned}
\widehat{\mathfrak R}_S(\mathcal F)
&=\mathbb E_\sigma\sup_{\|w\|_2\le D}
w^\top\left(\frac1n\sum_i\sigma_i\phi(x_i)\right)\\
&\le\frac{DR}{\sqrt n}.
\end{aligned}
$$

所以 Lipschitz 损失下的估计误差为 $O(GDR/\sqrt n)$，这个界不显式依赖参数维数。若改用 $\ell_1$ 约束和 $\ell_\infty$ 有界特征，通常得到 $O(DR\sqrt{\log d/n})$，体现稀疏结构只付出对数维数代价。

这比单纯数参数更准确：范数约束、特征几何和损失光滑性共同决定容量。

# 15. 其他复杂度工具

## Covering number 与 metric entropy

在距离 $d(\cdot,\cdot)$ 下，覆盖数

$$
N(\varepsilon,\mathcal F,d)
$$

是用半径 $\varepsilon$ 的球覆盖 $\mathcal F$ 所需的最少球数；其对数称为 metric entropy。可用有限 $\varepsilon$-net、union bound 和 Lipschitz 性把无限类化为有限类。基础证明常产生额外 $\log n$，chaining 可进一步改善。

## Pseudo-dimension

Pseudo-dimension 把 VC 维推广到实值函数类。它允许每个点有不同阈值 $r_i$，研究 $\mathbf 1\{f(x_i)\ge r_i\}$ 能否实现全部标记，常用于回归。

## Gaussian complexity

把 Rademacher 符号替换为标准高斯变量。它与 Rademacher 复杂度在许多问题中等价到常数或对数因子。

## Local Rademacher complexity

只研究低经验风险或靠近最优解的局部函数集，可给出比全局最坏情况更快的速率。

## PAC-Bayes

在假设类上设置先验 $P$，学习得到后验 $Q$，泛化界由经验风险、$\mathrm{KL}(Q\|P)$ 和置信项控制。它提供算法/数据依赖的复杂度描述。

这些工具不是相互排斥的“不同真理”，而是从组合、几何、概率或算法角度度量同一个问题：学习器有多强的拟合随机波动能力。

# 16. 模型选择与结构风险最小化

只给一个固定 $\mathcal H$ 的泛化界，还没有解决“如何选择复杂度”。设有嵌套模型

$$
\mathcal H_1\subseteq\mathcal H_2\subseteq\cdots.
$$

结构风险最小化（SRM）选择

$$
\widehat h
\in
\arg\min_{k,h\in\mathcal H_k}
\left\{
\widehat R_S(h)+\operatorname{pen}(k,n,\delta)
\right\},
$$

其中惩罚项随类复杂度增加。若给模型索引分配权重 $\pi_k>0$、$\sum_k\pi_k=1$，可把每类置信度设为 $\delta\pi_k$，再用 union bound 同时保证所有模型的界。

典型惩罚形状为

$$
\operatorname{pen}(k,n,\delta)
\asymp
\sqrt{\frac{\operatorname{complexity}(\mathcal H_k)
+\log(1/\pi_k)+\log(1/\delta)}{n}}.
$$

另一种方法是留出验证集：先在训练集上生成有限个候选模型，再在独立验证集上做 ERM。若候选数为 $M$，选择代价通常只按 $\sqrt{\log M/n_{\mathrm{val}}}$ 增长。验证集参与了模型选择后就不能再当作无偏测试集。

# 17. 算法稳定性：从假设类转向具体算法

VC 与一致收敛控制整个 $\mathcal H$，可能对实际算法过于保守。稳定性研究：替换一个训练样本时，算法输出的损失变化多大。

取独立样本

$$
S=(Z_1,\ldots,Z_n),\qquad S'=(Z_1',\ldots,Z_n'),
$$

并定义替换第 $i$ 个点的数据集

$$
S^{(i)}=(Z_1,\ldots,Z_{i-1},Z_i',Z_{i+1},\ldots,Z_n).
$$

平均 replace-one 稳定性定义为

$$
\Delta(A)
=\mathbb E_{S,S'}\left[
\frac1n\sum_{i=1}^n
\bigl(
\ell(A(S),Z_i')-
\ell(A(S^{(i)}),Z_i')
\bigr)
\right].
$$

对 $A(S)$，$Z_i'$ 是未见样本；对 $A(S^{(i)})$，它是训练样本。差值衡量“把这个点加入训练集”能让该点的损失降低多少。

## 17.1 平均稳定性等于期望泛化间隙

期望泛化间隙是

$$
\mathbb E_S[R(A(S))-\widehat R_S(A(S))].
$$

由于 $Z_i'$ 独立于 $S$，

$$
\mathbb E[R(A(S))]
=\mathbb E\left[
\frac1n\sum_i\ell(A(S),Z_i')
\right].
$$

又因为 $(S,Z_i)$ 与 $(S^{(i)},Z_i')$ 在交换对应样本后同分布，

$$
\mathbb E[\ell(A(S),Z_i)]
=
\mathbb E[\ell(A(S^{(i)}),Z_i')].
$$

相减并对 $i$ 平均，得到精确恒等式

$$
\boxed{
\mathbb E_S[R(A(S))-\widehat R_S(A(S))]
=\Delta(A).
}
$$

这里控制的是 generalization gap，不应与

$$
R(A(S))-\inf_{h\in\mathcal H}R(h)
$$

这种 excess risk 混淆。要从泛化间隙得到超额风险，还需要算法近似最小化经验风险。

## 17.2 一致稳定性与正则化 ERM

若任意只相差一个样本的数据集 $S\simeq S'$ 都满足

$$
\sup_z
|\ell(A(S),z)-\ell(A(S'),z)|
\le\beta_n,
$$

则称算法具有 $\beta_n$-一致稳定性。显然

$$
|\mathbb E[R(A(S))-\widehat R_S(A(S))]|
\le\beta_n.
$$

考虑正则化 ERM

$$
\widehat w_S
\in\arg\min_w
\left\{
\frac1n\sum_{i=1}^n\ell(Y_i,w^\top\phi(X_i))
+\frac\lambda2\|w\|_2^2
\right\}.
$$

若损失关于预测值凸且 $G$-Lipschitz，$\|\phi(x)\|_2\le R$，平方范数使目标 $\lambda$-强凸，则 Bach 的练习给出

$$
\boxed{
\beta_n\le\frac{2G^2R^2}{\lambda n}.
}
$$

正则化越强，算法越稳定；但过大的 $\lambda$ 会增加偏差/近似误差。这是另一种 bias–stability 权衡。

## 17.3 稳定性 + 渐近 ERM 推出可学习性

假设算法满足两个期望条件：

$$
\mathbb E[R(A(S))-\widehat R_S(A(S))]
\le\epsilon_1(n),
$$

以及渐近 ERM 条件

$$
\mathbb E\left[
\widehat R_S(A(S))-inf_{h\in\mathcal H}\widehat R_S(h)
\right]
\le\epsilon_2(n).
$$

令 $h^*\in\arg\min_{h\in\mathcal H}R(h)$。由于固定 $h^*$ 满足 $\mathbb E\widehat R_S(h^*)=R(h^*)$，

$$
\begin{aligned}
\mathbb E[R(A(S))-R(h^*)]
&=\mathbb E[R(A(S))-\widehat R_S(A(S))]\\
&\quad+\mathbb E[\widehat R_S(A(S))-\widehat R_S(h^*)]\\
&\le\epsilon_1(n)+\epsilon_2(n).
\end{aligned}
$$

若两项都趋于零，算法在期望中可学习。这个结果解释了为什么仅有稳定性还不够：一个始终输出常数的算法很稳定，却未必接近经验最优。

## 17.4 可学习但不一致收敛：一般损失下的反例

统计学习基本定理中的“可学习 $\Leftrightarrow$ 一致收敛”要限定在二分类 0-1 损失的假设类框架。UML-Sol 第 13 章给出一般凸损失下的反例。

令假设空间为 $\mathbb R^d$ 的单位球 $B$，样本为 $(x,\alpha)\in B\times\{0,1\}^d$，损失为

$$
\ell(w,(x,\alpha))
=\sum_{j=1}^d\alpha_j(x_j-w_j)^2.
$$

该损失关于 $w$ 凸、非负且光滑，正则化损失最小化可获得不依赖 $d$ 的学习保证。

但若训练集中某个坐标 $j$ 从未被激活，即所有样本都有 $\alpha_j=0$，则经验风险对 $w_j$ 完全不敏感。于是某个 ERM 可以在该坐标取很大值，仍保持零经验风险，而总体风险可达 $1/2$。当 $d\gg2^n$ 时，高概率存在这样的漏看坐标，因此一致收敛所需样本量至少随 $\log d$ 增长。

取无限维极限，可得到“问题可学习，但整个假设类不一致收敛”。这不与二分类基本定理矛盾，因为损失与输出结构已经超出二分类 0-1 框架。

# 18. 现代过参数化：隐式偏置、稳定性与 double descent 的连接

在过参数化区，假设类可能包含大量插值解：

$$
\widehat R_S(h)=0.
$$

泛化表现取决于算法选了哪个解。梯度下降在线性最小二乘中从零初始化会选择最小 $\ell_2$ 范数插值器；不同参数化甚至可产生类似 $\ell_1$ 的隐式正则化。

隐式偏置可能依赖：

- 模型参数化与网络结构；
- 初始化；
- 梯度方法类型、步长和训练时长；
- SGD 的随机性；
- momentum、Adam/AdamW 中的归一化与动量；
- batch normalization、dropout 和数据处理。

它并不自动有益：若真实结构是稀疏的，最小 $\ell_2$ 范数偏置可能不如显式 $\ell_1$ 正则化。因而现代泛化不能只看“是否插值”，还要问：

1. 算法选出的插值解具有什么几何性质？
2. 它是否与任务的先验结构匹配？
3. 替换一个样本时，这个解变化多大？
4. 对应的局部复杂度、间隔或范数有多大？

VC/Rademacher 研究整个可选集合，稳定性研究选择映射 $S\mapsto A(S)$，隐式偏置研究这个映射偏向哪些解；三者互补。

# 19. 全部理论的证明地图

## 路线 A：有限可实现类

$$
R(h)>\epsilon
\Rightarrow
\mathbb P(\widehat R_S(h)=0)\le e^{-n\epsilon}
$$

$$
\Rightarrow
\mathbb P(\exists\text{ 坏的一致假设})
\le|\mathcal H|e^{-n\epsilon}
$$

$$
\Rightarrow
n\gtrsim\frac{\log|\mathcal H|+\log(1/\delta)}{\epsilon}.
$$

## 路线 B：VC 维与一致收敛

$$
\operatorname{VCdim}(\mathcal H)=d
\overset{\text{Sauer}}{\Longrightarrow}
\Pi_{\mathcal H}(m)\lesssim m^d
$$

$$
\overset{\text{ghost + symmetrization}}{\Longrightarrow}
\sup_h|R(h)-\widehat R(h)|\to0
$$

$$
\Longrightarrow
\text{ERM agnostic PAC}.
$$

## 路线 C：Rademacher 复杂度

$$
\mathbb E\sup_g(\mathbb Pg-\mathbb P_ng)
\le2\mathfrak R_n(\mathcal G)
$$

$$
\overset{\text{contraction}}{\Longrightarrow}
\mathfrak R_n(\ell\circ\mathcal F)
\le G\mathfrak R_n(\mathcal F)
$$

$$
\Longrightarrow
\text{范数、特征与数据依赖的泛化界}.
$$

## 路线 D：稳定性

$$
S\text{ 替换一个点}
\Rightarrow
\ell(A(S),z)\text{ 变化小}
$$

$$
\Rightarrow
\mathbb E[R(A(S))-\widehat R_S(A(S))]\text{ 小}
$$

$$
+\ \text{渐近 ERM}
\Rightarrow
\text{期望超额风险小}.
$$

# 20. 常见误区与纠正

1. **误区：训练误差的期望总等于测试误差。** 只对预先固定的 $h$ 成立；对 $h_S$ 会有选择偏差。
2. **误区：可实现性意味着学到真实函数。** 它只保证类中有零风险函数；有限样本上可能有多个一致解。
3. **误区：一个假设一个假设地集中就足够。** ERM 依赖样本，需要同时对全部 $h$ 控制。
4. **误区：某个 $d+1$ 点集合不能打散就证明 VC 维至多为 $d$。** 上界必须排除任意 $d+1$ 点集合。
5. **误区：参数个数就是复杂度。** 高频一参数族可有无限 VC 维；过参数化模型也可能因范数或算法偏置而有效复杂度较低。
6. **误区：PAC 可学习等于可高效学习。** PAC 是统计保证；ERM 可能是 NP-hard，甚至存在可学习但不可计算的构造。
7. **误区：Bayes 风险一定为零。** 噪声标签下通常非零；应分析超额风险 $R(h)-R^*$。
8. **误区：泛化间隙就是超额风险。** 前者是 $R(A(S))-\widehat R_S(A(S))$，后者还要和类内最优或 Bayes 最优比较。
9. **误区：Rademacher 复杂度直接套在预测类上即可。** 风险偏差对应损失类，需用 Lipschitz contraction 把预测类界转过去。
10. **误区：可学习与一致收敛在所有损失下都等价。** 该等价是二分类 0-1 框架的基本定理；一般损失存在反例。
11. **误区：double descent 推翻了经典理论。** 它说明“模型大小”的代理和“任意 ERM”分析不够细，需要考虑具体解、算法与正则化。
12. **误区：稳定算法一定学得好。** 稳定只控制泛化间隙；还必须近似最小化经验风险。

# 21. 自测题（附简答）

## 题 1

为什么不能对 ERM 输出直接使用“固定 $h$”的 Hoeffding 不等式？

**答：** 因为 $\widehat h_S$ 与 $S$ 相关，它会偏向经验风险偶然较低的函数。必须使用一致收敛、样本分割、稳定性或其他能处理数据依赖的方法。

## 题 2

若 $S$ 是 $\epsilon/2$-representative，证明 ERM 的类内超额风险不超过 $\epsilon$。

**答：**

$$
R(\widehat h)
\le\widehat R(\widehat h)+\epsilon/2
\le\widehat R(h^*)+\epsilon/2
\le R(h^*)+\epsilon.
$$

## 题 3

一维区间分类器为何 VC 维为 2？

**答：** 两个点的四种标签都可由空区间、全含区间或只含一个点的窄区间实现；三个有序点的标签 $(1,0,1)$ 无法由单一区间实现。

## 题 4

为什么有限 VC 维会导致增长函数从指数降为多项式？

**答：** Sauer 引理给出 $\Pi_{\mathcal H}(m)\le\sum_{i=0}^d\binom mi\le(em/d)^d$。

## 题 5

稳定性恒等式为什么需要替换样本 $S^{(i)}$？

**答：** 它把训练损失项 $\ell(A(S),Z_i)$ 通过交换性改写为 $\ell(A(S^{(i)}),Z_i')$，从而与新样本损失在同一个 $Z_i'$ 上比较。

## 题 6

正则化越强是否一定越好？

**答：** 不一定。较大 $\lambda$ 使稳定性界 $2G^2R^2/(\lambda n)$ 更小，但可能增加近似误差或偏差。

# 22. 核心公式速查

### Bayes 决策

$$
h^*(x)\in\arg\min_a\mathbb E[\ell(Y,a)\mid X=x].
$$

### ERM

$$
\widehat h_S\in\arg\min_{h\in\mathcal H}\widehat R_S(h).
$$

### 超额风险分解

$$
R(\widehat h)-R^*
=
[R(\widehat h)-\inf_{h\in\mathcal H}R(h)]
+[\inf_{h\in\mathcal H}R(h)-R^*].
$$

### 一致收敛控制 ERM

$$
R(\widehat h)-\inf_{h\in\mathcal H}R(h)
\le2\sup_{h\in\mathcal H}|R(h)-\widehat R_S(h)|.
$$

### 增长函数与 VC 维

$$
\Pi_{\mathcal H}(m)
=\max_{|C|=m}|\mathcal H|_C|,
\qquad
\operatorname{VCdim}(\mathcal H)
=\max\{m:\Pi_{\mathcal H}(m)=2^m\}.
$$

### Sauer 引理

$$
\Pi_{\mathcal H}(m)
\le\sum_{i=0}^d\binom mi
\le\left(\frac{em}{d}\right)^d.
$$

### Rademacher 复杂度

$$
\mathfrak R_n(\mathcal G)
=\mathbb E_{S,\sigma}
\sup_{g\in\mathcal G}
\frac1n\sum_i\sigma_i g(Z_i).
$$

### 平均稳定性恒等式

$$
\mathbb E[R(A(S))-\widehat R_S(A(S))]=\Delta(A).
$$

### 统计学习基本定理

$$
\boxed{
\operatorname{VCdim}(\mathcal H)<\infty
\iff
\mathcal H\text{ 在二分类 0-1 损失下 PAC 可学习}.
}
$$

# 23. 阅读定位与延伸

## 本地资料

- [Learning Theory from First Principles](../slides/ltfp_book.pdf)
  - Bayes 风险与 Bayes 预测器：书页 28–30；PDF 第 44–46 页。
  - ERM、风险分解与一致性：书页 32–37；PDF 第 48–53 页。
  - 风险最小化、Rademacher 复杂度、覆盖数和稳定性：书页 84–102；PDF 第 100–118 页。
  - 隐式偏置与 double descent：书页 355–360；PDF 第 371–376 页。
- [Understanding Machine Learning — Solution Manual](../slides/MLbookSol.pdf)
  - PAC、有限类与 Bayes 分类器练习：PDF 第 3–6 页。
  - 一致收敛与 VC 维练习：PDF 第 9–20 页。
  - 正则化、稳定性、一般损失下的反例：PDF 第 34–37 页。

## 建议继续学习的主题

1. margin bound 与 SVM：把线性可分性细化为间隔依赖的复杂度；
2. PAC-Bayes：对随机化预测器与神经网络后验给出数据依赖界；
3. local Rademacher complexity：解释低噪声和局部曲率下的快收敛率；
4. SGD stability：把步长、迭代次数和光滑性连接到泛化；
5. benign overfitting：研究插值器在何种数据谱结构下仍达到最优风险；
6. minimax lower bounds：判断上界中的 $d,n,\epsilon$ 依赖是否可改进。

---

**一句话总结：** 学习理论研究的不是“怎样把训练误差降到最低”，而是“在什么结构、多少数据和哪种算法选择下，训练表现能够可靠地转化为未知分布上的表现”。
