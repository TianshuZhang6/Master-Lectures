# L11-L14: Optimization, Stochastic Methods, Sparsity, and Low-Rank Learning

This note follows the section order of slides L11-L14. Material from the two reference books is inserted only inside the matching slide section.

**Reference abbreviations**

- **LTfFP:** Francis Bach, *Learning Theory from First Principles*, May 26, 2025 version.
- **UML:** Shai Shalev-Shwartz and Shai Ben-David, *Understanding Machine Learning: From Theory to Algorithms*, digital 2014 version.
- LTfFP citations give both the printed book page and the PDF page because the PDF has 16 pages of front matter. UML citations use the digital/PDF page shown in the file.

# L11: Optimization and Learning - 1

## 1. Learning as an optimization problem

> **Pages:** Slides: L11, pp. 3-13. Book supplements: LTfFP Sections 2.2.2-2.3.2, book pp. 27-35 (PDF pp. 43-51), and Section 5.1, book pp. 109-110 (PDF pp. 125-126); UML Sections 2.2-2.3, digital pp. 35-40, Section 5.2, digital pp. 64-65, and Sections 13.2-13.3, digital pp. 171-179.

Let the training examples be independent and identically distributed:

$$
(x_i,y_i)\overset{\mathrm{iid}}{\sim}P_{X,Y},
\qquad i=1,\ldots,n.
$$

For a predictor $f_\theta$ and a loss $\ell$, the population risk is

$$
R(\theta)=\mathbb E[\ell(Y,f_\theta(X))],
$$

whereas the empirical risk is

$$
\widehat R_n(\theta)
=\frac1n\sum_{i=1}^n\ell(y_i,f_\theta(x_i)).
$$

Training normally solves a regularized empirical problem,

$$
\widehat\theta\approx
\operatorname*{argmin}_\theta
\left\{
\widehat R_n(\theta)+\Omega(\theta)
\right\}.
$$

The central warning is that machine learning is not merely the numerical minimization of the training objective. The real target is low risk on unseen data.

Let

$$
\theta^\star\in\operatorname*{argmin}_\theta R(\theta).
$$

Adding and subtracting empirical risks gives

$$
\begin{aligned}
R(\widehat\theta)-R(\theta^\star)
={}&[R(\widehat\theta)-\widehat R_n(\widehat\theta)]\\
&+[\widehat R_n(\widehat\theta)-\widehat R_n(\theta^\star)]\\
&+[\widehat R_n(\theta^\star)-R(\theta^\star)].
\end{aligned}
$$

Since $\inf_\theta\widehat R_n(\theta)\le\widehat R_n(\theta^\star)$,

$$
\widehat R_n(\widehat\theta)-\widehat R_n(\theta^\star)
\le
\widehat R_n(\widehat\theta)-\inf_\theta\widehat R_n(\theta).
$$

Thus the excess risk is controlled by two statistical deviation terms and one optimization error. If the statistical error is of order $\delta_n$, reducing the optimization error far below $\delta_n$ usually does not improve test performance.

**Book-aligned supplement.** The books separate one additional modeling term. If $\mathcal F$ is the chosen predictor family and $R^\star$ is the unrestricted Bayes risk, then

$$
R(\widehat f)-R^\star
=
\underbrace{R(\widehat f)-\inf_{f\in\mathcal F}R(f)}_{\text{estimation and optimization}}
+
\underbrace{\inf_{f\in\mathcal F}R(f)-R^\star}_{\text{approximation error}}.
$$

The first term is affected by finite data and imperfect optimization; the second is caused by the inductive bias of the model class. UML describes ERM as an inductive-bias mechanism and uses overfitting to explain why minimizing training loss without controlling the class can fail. LTfFP then inserts the optimization error into the usual empirical-versus-population risk decomposition. This makes the practical stopping rule precise: optimize until optimization error is comparable to estimation error, unless a more accurate empirical solution is needed for another reason.

UML also makes the slide's stability question precise. If $S^{(i)}$ is obtained by replacing one example in $S$, the expected generalization gap equals an average replace-one sensitivity:

$$
\mathbb E[L_D(A(S))-L_S(A(S))]
=
\mathbb E[\ell(A(S^{(i)}),z_i)-\ell(A(S),z_i)].
$$

For regularized loss minimization

$$
A(S)=\operatorname*{argmin}_w
\{L_S(w)+\lambda\|w\|^2\},
$$

with a convex $\rho$-Lipschitz loss, UML obtains

$$
\mathbb E L_D(A(S))
\le
L_D(w^\star)+\lambda\|w^\star\|^2
+\frac{2\rho^2}{\lambda n}.
$$

The regularization-bias term grows with $\lambda$, whereas the stability term shrinks with $\lambda$. This is a precise fitting-generalization tradeoff. The stability notion here is on-average replace-one stability, not the stronger uniform-stability property.

## 2. Descent directions and gradient descent geometry

> **Pages:** Slides: L11, pp. 14-27. Book supplements: LTfFP Sections 5.2.2-5.2.4, book pp. 116-126 (PDF pp. 132-142); UML Sections 12.1 and 14.1, digital pp. 156-163 and 185-188.

A general first-order iteration is

$$
\theta^{k+1}=\theta^k+\eta_kd^k,
\qquad \eta_k>0.
$$

Taylor expansion at $\theta$ gives

$$
F(\theta+\eta d)
=F(\theta)+\eta\langle\nabla F(\theta),d\rangle+o(\eta).
$$

Therefore, any direction satisfying

$$
\langle\nabla F(\theta),d\rangle<0
$$

is a descent direction for a sufficiently small positive step. The negative gradient is the steepest Euclidean descent direction and is perpendicular to the local level set:

$$
\theta^{k+1}=\theta^k-\eta_k\nabla F(\theta^k).
$$

A correct direction is not enough: a step that is too large can cross the low-value region and increase the objective.

For an $L$-smooth function,

$$
F(\theta-\eta\nabla F(\theta))
\le
F(\theta)
-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla F(\theta)\|^2.
$$

Hence every $0<\eta<2/L$ gives descent at nonstationary points.

**Book-aligned supplement.** UML also derives gradient descent as a regularized minimization of the local linear model:

$$
\theta_{t+1}
=\operatorname*{argmin}_\theta
\left\{
\langle\nabla F(\theta_t),\theta-\theta_t\rangle
+\frac1{2\eta}\|\theta-\theta_t\|^2
\right\}.
$$

This view explains why changing the quadratic term changes the geometry or preconditioner. For convex functions with subgradients $v_t$ bounded by $G$, the telescoping identity

$$
\sum_{t=1}^T
\langle\theta_t-\theta^\star,v_t\rangle
\le
\frac{\|\theta_1-\theta^\star\|^2}{2\eta}
+\frac\eta2\sum_{t=1}^T\|v_t\|^2
$$

implies, for the averaged iterate $\bar\theta_T$, the bound

$$
F(\bar\theta_T)-F(\theta^\star)
\le
\frac{\|\theta_1-\theta^\star\|^2}{2\eta T}
+\frac{\eta G^2}{2}.
$$

Balancing the two terms yields the general nonsmooth convex rate $O(T^{-1/2})$. Smoothness improves the deterministic convex rate to $O(1/T)$, and strong convexity allows geometric convergence.

## 3. Gradient descent on least squares

> **Pages:** Slides: L11, pp. 28-32. Book supplements: LTfFP Sections 3.2-3.6, book pp. 46-59 (PDF pp. 62-75), Section 5.2.1, book pp. 112-116 (PDF pp. 128-132), and Section 12.1.1, book pp. 344-346 (PDF pp. 360-362); UML Section 9.2, digital pp. 123-125, and Appendix C, digital pp. 430-434.

Consider

$$
F(\theta)=\frac1{2n}\|\Phi\theta-y\|_2^2.
$$

Its gradient and Hessian are

$$
\nabla F(\theta)=\frac1n\Phi^\top(\Phi\theta-y),
\qquad
H=\frac1n\Phi^\top\Phi\succeq0.
$$

Every minimizer satisfies the normal equation

$$
H\theta^\star=\frac1n\Phi^\top y.
$$

Let the eigenvalues of $H$ lie in $[\mu,L]$. For fixed-step GD,

$$
e_t:=\theta_t-\theta^\star
=(I-\gamma H)^te_0.
$$

When $\mu>0$, the optimal constant step is

$$
\gamma^\star=\frac2{L+\mu},
$$

with contraction factor

$$
q^\star=\frac{\kappa-1}{\kappa+1},
\qquad
\kappa=\frac L\mu.
$$

The simpler choice $\gamma=1/L$ gives

$$
\|e_t\|^2
\le
\left(1-\frac1\kappa\right)^{2t}\|e_0\|^2
\le
e^{-2t/\kappa}\|e_0\|^2.
$$

Thus the iteration complexity is

$$
O\!\left(\kappa\log\frac1\varepsilon\right).
$$

If $H$ is singular, parameter distance to an arbitrary minimizer need not vanish, but

$$
F(\theta)-F^\star
=\frac12(\theta-\theta^\star)^\top H(\theta-\theta^\star)
$$

ignores null-space directions. For $0<\gamma\le1/L$,

$$
F(\theta_t)-F^\star
\le
\frac{\|e_0\|^2}{8\gamma t}
=O(1/t).
$$

Adding ridge regularization changes the Hessian to $H+\lambda I$, so the smallest eigenvalue is at least $\lambda$.

**Book-aligned supplement.** The least-squares chapters give a geometric interpretation: $\Phi\theta^\star$ is the orthogonal projection of $y$ onto $\operatorname{im}(\Phi)$. This proves existence even if $H$ is singular. The overparameterization chapter adds the implicit-bias result. If $\theta_0=0$ and the step is stable, GD stays in $\operatorname{im}(\Phi^\top)$ and converges to

$$
\theta_\infty=\Phi^\dagger y,
$$

the minimum-Euclidean-norm least-squares solution. With a general initialization, the component in $\ker(\Phi)$ is preserved. Thus optimization selects one solution among infinitely many even though all minimizers have the same predictions.

# L12: Optimization and Learning - 2

## 1. Why stochastic gradient descent is needed

> **Pages:** Slides: L12, pp. 2-10. Book supplements: LTfFP Sections 5.1 and 5.4, book pp. 109-110 and 134-139 (PDF pp. 125-126 and 150-155); UML Sections 14.3 and 14.5.1, digital pp. 191-198.

Large-scale learning often has the finite-sum form

$$
F(\theta)=\frac1n\sum_{i=1}^nf_i(\theta).
$$

Full GD evaluates all $n$ component gradients at every iteration. SGD samples $i_k$ and uses

$$
\theta_{k+1}
=\theta_k-\gamma_k\nabla f_{i_k}(\theta_k).
$$

For uniform sampling with replacement,

$$
\mathbb E[\nabla f_{i_k}(\theta_k)\mid\theta_k]
=\nabla F(\theta_k).
$$

The same idea directly applies to population risk

$$
F(\theta)=\mathbb E_\xi[\phi(\theta,\xi)]
$$

by drawing a fresh $\xi_k$ at every iteration. One stochastic step is cheap, but the trajectory is noisy and generally nonmonotone.

**Book-aligned supplement.** UML emphasizes that online SGD does not first construct and exactly minimize an empirical objective. Instead,

$$
g_t\in\partial_\theta\phi(\theta_t,\xi_t)
$$

is an unbiased subgradient of the population risk. Consequently, the iteration bound is also a sample-complexity bound: one fresh example is consumed per update. For a convex $G$-Lipschitz problem with comparator norm at most $B$, $T=O(B^2G^2/\varepsilon^2)$ fresh samples suffice for expected excess risk at most $\varepsilon$. LTfFP stresses the same optimization-statistics match: a single pass can already reach the statistical accuracy of regularized ERM.

## 2. Scalar least-squares intuition

> **Pages:** Slides: L12, pp. 11-18. Book supplements: LTfFP Section 5.4.3, book pp. 143-146 (PDF pp. 159-162); UML Sections 9.2 and 14.3, digital pp. 123-125 and 191-193.

For

$$
F(\theta)=\frac12\sum_{i=1}^n(x_i\theta-b_i)^2,
$$

the global minimizer is

$$
\theta^\star
=\frac{\sum_i x_ib_i}{\sum_i x_i^2}
=\sum_i
\frac{x_i^2}{\sum_jx_j^2}\theta_i^\star,
\qquad
\theta_i^\star=\frac{b_i}{x_i}.
$$

Therefore,

$$
\theta^\star\in
[\min_i\theta_i^\star,\max_i\theta_i^\star].
$$

Since

$$
\nabla f_i(\theta)=x_i^2(\theta-\theta_i^\star),
$$

all component gradients agree in sign outside this interval. Inside it, they may conflict, explaining oscillation, decreasing steps, and iterate averaging.

**Book-aligned supplement.** LTfFP analyzes the multidimensional noisy least-mean-squares recursion. For

$$
y_t=\phi(x_t)^\top\theta^\star+\varepsilon_t,
$$

constant-step SGD satisfies

$$
\theta_t-\theta^\star
=
(I-\gamma\phi_t\phi_t^\top)(\theta_{t-1}-\theta^\star)
+\gamma\varepsilon_t\phi_t.
$$

The first term contracts the initial-condition bias; the second continually injects variance. Under bounded features, the averaged iterate obeys a representative bound

$$
\mathbb E[F(\bar\theta_t)-F(\theta^\star)]
\le
\frac{\|\theta_0-\theta^\star\|^2}{2\gamma t}
+\frac{\gamma\sigma^2R^2}{2}.
$$

This formula makes the step-size tradeoff explicit: small $\gamma$ reduces the noise floor but slows removal of the initial bias.

## 3. Sampling, mini-batches, averaging, and comparison with GD

> **Pages:** Slides: L12, pp. 19-28. Book supplements: LTfFP Sections 5.4 and 5.4.4, book pp. 134-141 and 146-151 (PDF pp. 150-157 and 162-167); UML Sections 14.3-14.4, digital pp. 191-196.

Sampling with replacement gives a clean conditional-unbiasedness argument. Random reshuffling visits every example exactly once per epoch but loses that stepwise independence. A mini-batch $I_k$ gives

$$
\theta_{k+1}
=\theta_k-
\frac{\gamma_k}{|I_k|}
\sum_{j\in I_k}\nabla f_j(\theta_k).
$$

Larger batches reduce variance and enable parallel computation, but their marginal benefit eventually decreases.

The Polyak-Ruppert average is

$$
\bar\theta_k=\frac1{k+1}\sum_{j=0}^k\theta_j.
$$

Typical convex and strongly convex rates are

$$
\mathbb E[F(\bar\theta_k)-F^\star]
=O(k^{-1/2})
$$

and

$$
\mathbb E[F(\bar\theta_k)-F^\star]
=O\!\left(\frac{\kappa}{k}\right).
$$

Full GD has a more stable trajectory and, for strongly convex smooth objectives, an iteration rate $e^{-k/\kappa}$, but each step uses all $n$ components. The $O(\kappa/k)$ and $O(1/\varepsilon^2)$ statements shown together on slide 28 correspond to different error regimes: $1/k$ yields $O(1/\varepsilon)$ iterations, whereas $1/\sqrt{k}$ yields $O(1/\varepsilon^2)$.

**Book-aligned supplement.** Both books derive the generic averaged-SGD inequality

$$
\mathbb E[F(\bar\theta_T)-F(\theta^\star)]
\le
\frac{\|\theta_1-\theta^\star\|^2}{2\gamma T}
+\frac{\gamma G^2}{2}.
$$

It separates finite-time bias from gradient noise. UML also lists random-iterate, suffix-average, and weighted-average outputs; these variants can improve strongly convex performance. LTfFP's variance-reduction section explains a different route for finite sums: occasionally compute or maintain information about the full gradient so that the stochastic direction remains unbiased but its variance vanishes near the optimum.

## 4. SGD for smooth nonconvex objectives

> **Pages:** Slides: L12, pp. 29-37. Book supplements: LTfFP Section 5.2.6, book pp. 129-130 (PDF pp. 145-146), as the deterministic nonconvex baseline. UML Chapter 14, digital pp. 184-201, treats convex SGD and does not give the slide's nonconvex stochastic theorem.

Assume the component functions are $L$-smooth, the stochastic gradient is conditionally unbiased, and

$$
\mathbb E\|g_k\|^2\le G^2.
$$

The smoothness inequality gives

$$
\mathbb E F(\theta_{k+1})
\le
\mathbb E F(\theta_k)
-\gamma_k\mathbb E\|\nabla F(\theta_k)\|^2
+\frac{L\gamma_k^2G^2}{2}.
$$

With $\gamma_k=c/\sqrt T$,

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

The conclusion is approximate stationarity, not global optimality. Requiring expected squared gradient at most $\varepsilon$ needs $T=O(\varepsilon^{-2})$.

**Book-aligned supplement.** LTfFP gives the deterministic comparison. With step $1/L$,

$$
F(\theta_{t+1})
\le
F(\theta_t)-\frac1{2L}\|\nabla F(\theta_t)\|^2,
$$

and hence

$$
\min_{0\le s<T}\|\nabla F(\theta_s)\|^2
\le
\frac{2L(F(\theta_0)-F_\inf)}{T}.
$$

The slide theorem is slower because stochastic noise contributes the additional $\gamma^2G^2$ term. Neither book uses this result to claim convergence to a local or global minimum.

## 5. Projected SGD for convex and strongly convex objectives

> **Pages:** Slides: L12, pp. 38-50. Book supplements: LTfFP Section 5.4.1, book pp. 139-141 (PDF pp. 155-157); UML Sections 14.4.1 and 14.4.4, digital pp. 193-196.

Projected SGD is

$$
\theta_{k+1}=P_{\mathcal X}(\theta_k-\gamma_kg_k).
$$

Projection is nonexpansive, so with $R_k=\|\theta_k-\theta^\star\|^2$,

$$
R_{k+1}
\le
R_k+\gamma_k^2\|g_k\|^2
-2\gamma_k\langle g_k,\theta_k-\theta^\star\rangle.
$$

For a $\mu$-strongly convex objective and $\|g_k\|\le G$,

$$
\mathbb E[F(\theta_k)-F^\star]
\le
\frac{\gamma_kG^2}{2}
+\frac{\gamma_k^{-1}-\mu}{2}r_k
-\frac{r_{k+1}}{2\gamma_k}.
$$

Taking $\gamma_k=1/(\mu k)$ and averaging gives

$$
\mathbb E[F(\bar\theta_T)-F^\star]
\le
\frac{G^2(1+\log T)}{2\mu T}.
$$

With linearly increasing weights

$$
w_i=\frac{2(i+1)}{(T+1)(T+2)},
$$

the logarithm can be removed:

$$
\mathbb E[F(\bar\theta_T)-F^\star]
\le
\frac{2G^2}{\mu(T+1)}.
$$

**Book-aligned supplement.** UML proves the same logarithmic rate using projection onto the feasible set and notes two refinements: averaging only the last half of the iterates can remove the logarithmic factor, and a refined analysis can also control the last iterate. LTfFP further emphasizes that $O(1/(\mu T))$ is oracle-optimal for this class up to constants, but the step $1/(\mu t)$ loses adaptivity because it requires knowledge of the strong-convexity parameter.

## 6. Gradient clipping and normalized gradient descent

> **Pages:** Slides: L12, pp. 51-56. Book supplement: LTfFP Exercise 5.20, book p. 132 (PDF p. 148), for normalized subgradient descent. UML has no direct treatment of gradient clipping or normalized gradients.

Clipped GD is

$$
x_{k+1}
=x_k-
\min\left(\eta,\frac{a\eta}{\|g_k\|}\right)g_k.
$$

Equivalently,

$$
x_{k+1}=
\begin{cases}
x_k-\eta g_k,&\|g_k\|\le a,\\
x_k-a\eta\dfrac{g_k}{\|g_k\|},&\|g_k\|>a.
\end{cases}
$$

The update norm is therefore at most $a\eta$.

Normalized GD uses

$$
x_{k+1}
=x_k-
\frac{\eta}{\|g_k\|+b}g_k.
$$

After matching $b=a$ and the numerator scale, the two effective multipliers differ by at most a factor of two because

$$
\max(a,g)\le a+g\le2\max(a,g).
$$

The purpose is to retain ordinary gradient behavior for small gradients while preventing extreme gradients from determining the step length.

**Book-aligned supplement.** LTfFP analyzes the related normalized subgradient update

$$
\theta_t
=\theta_{t-1}
-\frac{D}{\sqrt t}\frac{g_t}{\|g_t\|}.
$$

For a convex $B$-Lipschitz objective, it gives

$$
\min_{0\le s\le t-1}F(\theta_s)-F^\star
\le
DB\frac{2+\log t}{2\sqrt t}.
$$

This update, clipping, and the smoothed normalization in the slides all decouple the step length from very large raw gradient norms, but they are different algorithms; the displayed theorem does not directly prove the slide's clipping rule.

## 7. Momentum

> **Pages:** Slides: L12, pp. 57-63. Book supplement: LTfFP Section 5.2.5, book pp. 126-129 (PDF pp. 142-145), for the related acceleration comparison. UML does not directly develop heavy-ball momentum.

The slide parameterization is

$$
m_t=\beta m_{t-1}+\nabla f(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta m_t.
$$

It is equivalent to the heavy-ball recurrence

$$
\theta_{t+1}
=\theta_t-\eta\nabla f(\theta_t)
+\beta(\theta_t-\theta_{t-1}).
$$

With $m_0=0$,

$$
\theta_{t+1}
=\theta_0-
\eta\sum_{i=1}^t
\frac{1-\beta^{t+1-i}}{1-\beta}
\nabla f(\theta_i).
$$

Old gradients therefore influence many later steps. In a narrow valley, oscillating high-curvature components tend to cancel while consistent low-curvature components accumulate.

**Book-aligned supplement.** LTfFP presents Nesterov acceleration rather than the exact heavy-ball recursion. For smooth strongly convex objectives it changes the condition-number dependence from

$$
O\!\left(\kappa\log\frac1\varepsilon\right)
$$

to

$$
O\!\left(\sqrt\kappa\log\frac1\varepsilon\right).
$$

This supports the slide's acceleration message while keeping the algorithms distinct: heavy-ball momentum and Nesterov's two-sequence construction are related but not identical.

# L13: Optimization Wrap-Up, Adaptive Methods, and Sparse Variable Selection

## 1. Scaled gradients and AdaGrad

> **Pages:** Slides: L13, pp. 2-7. Book supplement: LTfFP Section 5.4.2, book pp. 141-143 (PDF pp. 157-159). UML has no direct AdaGrad section.

A scaled-gradient method uses

$$
\theta_{t+1}=\theta_t-G_t^{-1/2}g_t.
$$

For a diagonal matrix $G_t=\operatorname{Diag}(s_{t,1},\ldots,s_{t,d})$,

$$
\theta_{t+1,j}
=\theta_{t,j}-\frac{g_{t,j}}{\sqrt{s_{t,j}}}.
$$

AdaGrad chooses

$$
G_t=\sum_{i=1}^tg_ig_i^\top,
$$

usually retaining only its diagonal. Hence

$$
s_{t,j}=\sum_{i=1}^tg_{i,j}^2.
$$

Frequently active coordinates receive progressively smaller effective learning rates, while rare coordinates retain larger steps. This is useful for sparse data, but permanent accumulation can make the learning rate decay too quickly.

An exponentially weighted alternative is

$$
G_t
=(1-\beta)\sum_{i=1}^t\beta^{t-i}g_ig_i^\top
=\beta G_{t-1}+(1-\beta)g_tg_t^\top.
$$

**Book-aligned supplement.** LTfFP derives adaptivity as stochastic preconditioning. For a diagonal metric $M=\operatorname{Diag}(m_1,\ldots,m_d)$, a representative bound has the form

$$
\frac1{2T}\sum_{j=1}^d m_j(\theta_j^\star)^2
+\frac{G^2}{2}\sum_{j=1}^d\frac{\Sigma_{jj}}{m_j},
$$

where $\Sigma$ is a gradient or feature second-moment matrix. Optimizing the $m_j$ shows why coordinatewise scaling can be much better than one global step when the coordinates have heterogeneous magnitudes. AdaGrad estimates the required diagonal statistics online instead of knowing them beforehand.

## 2. Adam and bias correction

> **Pages:** Slides: L13, pp. 8-17. Book supplement: LTfFP Section 5.4.2, book p. 143 (PDF p. 159), which identifies Adam as an online adaptive-preconditioning method but does not derive its full recursion. UML has no direct Adam coverage.

Adam maintains exponential moving averages of the first and second raw moments:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t).
$$

With zero initialization,

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}
$$

remove the early bias toward zero. The standard update is

$$
\boxed{
\theta_{t+1}
=\theta_t-
\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}
}
$$

with all square roots and divisions taken coordinatewise. $v_t$ is a second raw moment, not a centered statistical variance.

Scaling one gradient coordinate by $c$ approximately scales $m_{t,j}$ by $c$ and $v_{t,j}$ by $c^2$, leaving $m_{t,j}/\sqrt{v_{t,j}}$ approximately scale invariant.

The slides place $\varepsilon$ inside the square root in one formula but outside it in another matrix formula, and the matrix formula omits $\eta$. The conventional form above places $\varepsilon$ outside the root. Bias-corrected quantities should be used for the parameter update without overwriting the uncorrected EMA states.

**Book-aligned supplement.** LTfFP connects both AdaGrad and Adam to online estimates of gradient covariance. Its point is conceptual rather than a second Adam derivation: adaptive methods learn a preconditioner while optimization is running. Therefore the detailed first-moment, second-moment, and bias-correction formulas in this section come from the slides.

## 3. L2 regularization, weight decay, and AdamW

> **Pages:** Slides: L13, pp. 18-20. Direct book supplement: neither LTfFP nor UML treats AdamW or decoupled weight decay.

If the objective is

$$
\mathcal L(\theta)+\frac\lambda2\|\theta\|^2,
$$

then its gradient is $g_t+\lambda\theta_t$. Sending this entire gradient through Adam puts the regularizer into both $m_t$ and $v_t$, so the shrinkage becomes entangled with coordinatewise adaptive scaling.

AdamW decouples the operations:

$$
\theta_{t+1}
=\theta_t-
\eta(d_t+\lambda\theta_t)
=(1-\eta\lambda)\theta_t-\eta d_t,
$$

where

$$
d_t=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}.
$$

With time-varying parameters,

$$
\theta_{t+1}
=(1-\eta_t\lambda_t)\theta_t-\eta_td_t.
$$

Without the gradient term, the accumulated decay is

$$
\theta_T
=\left[\prod_{t=0}^{T-1}(1-\eta_t\lambda_t)\right]\theta_0
\approx
\exp\left(-\sum_t\eta_t\lambda_t\right)\theta_0.
$$

The slide's broader form $\theta_{t+1}=\alpha_t\theta_t+\beta_tm_t$ represents Adam only if $m_t$ is reinterpreted as the fully preconditioned direction; a scalar $\beta_t$ cannot by itself express coordinatewise division of the raw first moment.

## 4. Sparse variable selection

> **Pages:** Slides: L13, pp. 21-23. Book supplements: LTfFP Sections 8.1-8.2.1, book pp. 221-228 (PDF pp. 237-244); UML Sections 25.1 and 25.1.3, digital pp. 358-365.

Consider the fixed-design linear model

$$
y=\Phi\theta^\star+\varepsilon,
\qquad
\mathbb E\varepsilon=0,
\qquad
\operatorname{Cov}(\varepsilon)=\sigma^2I.
$$

The target quantity is prediction error,

$$
\frac1n\|\Phi(\widehat\theta-\theta^\star)\|^2,
$$

not necessarily Euclidean parameter error. Ordinary least squares has risk of order $\sigma^2d/n$. If

$$
\|\theta^\star\|_0\le k\ll d,
$$

one hopes to replace $d$ by a sparsity-dependent complexity.

**Book-aligned supplement.** LTfFP calls the known support the oracle case. If $A=\operatorname{supp}(\theta^\star)$ were known, least squares on $\Phi_A$ would have prediction risk of order

$$
\frac{\sigma^2k}{n}.
$$

When the support is unknown, the estimator must search among approximately $\binom dk$ models, producing the additional $\log(d/k)$ price. UML emphasizes the computational side: exhaustive search over all $k$-subsets is normally infeasible, motivating filters, greedy selection, and $\ell_1$ relaxation. It also notes that feature selection can reduce memory, prediction cost, measurement cost, and estimation error.

## 5. The constrained least-squares basic inequality

> **Pages:** Slides: L13, pp. 23-25. Book supplement: LTfFP Sections 8.1.1-8.1.2, book pp. 223-226 (PDF pp. 239-242). UML does not present this fixed-design projection proof in the same form.

Let $\Omega$ be a constraint set containing $\theta^\star$, and let

$$
\widehat\theta\in
\operatorname*{argmin}_{\theta\in\Omega}
\|y-\Phi\theta\|^2.
$$

With $\Delta=\widehat\theta-\theta^\star$, optimality gives

$$
\|\varepsilon-\Phi\Delta\|^2\le\|\varepsilon\|^2,
$$

and therefore

$$
\boxed{
\|\Phi\Delta\|^2
\le2\varepsilon^\top\Phi\Delta.
}
$$

Normalizing the feasible error direction yields

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

For $\Omega=\mathbb R^d$, the normalized prediction directions fill the unit sphere in $\operatorname{im}(\Phi)$. Hence

$$
\sup_{\substack{z\in\operatorname{im}(\Phi)\\\|z\|=1}}
(\varepsilon^\top z)^2
=\|\Pi_\Phi\varepsilon\|^2,
$$

and

$$
\frac1n\mathbb E\|\Phi\Delta\|^2
\le
\frac{4\sigma^2\operatorname{rank}(\Phi)}n.
$$

**Book-aligned supplement.** LTfFP presents this as a reusable proof template. It does not require a closed-form estimator: only feasibility of $\theta^\star$ and approximate optimality of $\widehat\theta$. If the empirical objective is solved only up to additive error $\delta$, the basic inequality acquires an additional $n\delta$-scale term. The same geometric reduction then separates optimization error from the Gaussian or sub-Gaussian complexity of the feasible directions.

## 6. Risk bound for best-subset selection

> **Pages:** Slides: L13, pp. 26-31. Book supplements: LTfFP Sections 8.2.1-8.2.2, book pp. 226-231 (PDF pp. 242-247); UML Sections 23.3 and 25.1.3, digital pp. 330-338 and 363-365, for the computational contrast between $\ell_0$ and $\ell_1$.

Let

$$
\Omega=\{\theta:\|\theta\|_0\le k\},
\qquad
\varepsilon\sim N(0,\sigma^2I_n).
$$

Then the constrained least-squares estimator satisfies

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

The proof uses four steps:

1. The difference of two $k$-sparse vectors is at most $2k$-sparse.
2. For each support $B$, feasible prediction errors lie in $\operatorname{im}(\Phi_B)$.
3. The supremum over that subspace is $\|\Pi_{\Phi_B}\varepsilon\|^2$.
4. There are at most
   $$
   \binom d{2k}\le\left(\frac{ed}{2k}\right)^{2k}
   $$
   candidate supports, and a Gaussian moment-generating-function bound controls their maximum.

The logarithm is the price of not knowing the support. The result controls prediction, not parameter recovery or exact support recovery.

**Book-aligned supplement.** LTfFP adds three important conclusions. First, the rate is minimax optimal up to constants even if exponential computation is allowed. Second, if $k$ is unknown, one may use

$$
\min_\theta
\left\{
\frac1n\|y-\Phi\theta\|^2
+\lambda\|\theta\|_0
\right\},
$$

with a noise-dependent penalty; this adapts to sparsity but remains combinatorial. Third, practical greedy and hard-thresholding algorithms require additional conditions to inherit comparable fast rates. UML makes the same computational distinction in compressed sensing: $\ell_0$ expresses the ideal sparse model, while $\ell_1$ can turn recovery into a tractable convex program.

# L14: Sparsity and Low Rank - Part 2

## 1. Algorithms for hard sparsity constraints

> **Pages:** Slides: L14, p. 2. Book supplements: LTfFP Section 8.2.1, book p. 228 (PDF p. 244), and Section 10.3.3, book pp. 302-304 (PDF pp. 318-320); UML Section 25.1.2, digital pp. 360-363.

Projection onto the $\ell_0$ ball solves

$$
\min_x\frac12\|x-y\|^2
\quad\text{subject to}\quad
\|x\|_0\le k.
$$

The solution $H_k(y)$ retains the $k$ largest coordinates in absolute value and zeros the rest. Projected gradient therefore becomes iterative hard thresholding:

$$
x_{t+1}
=H_k\left(x_t-\gamma\nabla F(x_t)\right).
$$

OMP instead grows the support one variable at a time and refits on the selected support.

**Book-aligned supplement.** UML gives the least-squares OMP criterion explicitly. If $I_t$ is the selected set and $V_t$ spans $X_{I_t}$ orthonormally, decompose each candidate column as

$$
X_j=V_tV_t^\top X_j+u_j.
$$

The next feature maximizes

$$
j_t\in\operatorname*{argmax}_{j\notin I_t}
\frac{\langle u_j,y\rangle^2}{\|u_j\|^2},
$$

after which the least-squares coefficients are recomputed. LTfFP explains that exact best-subset search costs on the order of $d^k$, while OMP and iterative hard thresholding require additional design conditions if one wants the fast statistical guarantee.

## 2. Lasso, soft thresholding, and proximal gradient

> **Pages:** Slides: L14, pp. 3-5. Book supplements: LTfFP Sections 5.2.5, 5.3, and 8.3.1-8.3.5, book pp. 126-134 and 231-240 (PDF pp. 142-150 and 247-256); UML Section 25.1.3, digital pp. 363-365.

Lasso is

$$
\min_\theta
\frac1{2n}\|y-\Phi\theta\|^2
+\lambda\|\theta\|_1.
$$

In one dimension, ridge gives continuous shrinkage,

$$
\operatorname*{argmin}_w
(y-w)^2+\lambda w^2
=\frac{y}{1+\lambda},
$$

whereas $\ell_1$ gives soft thresholding,

$$
S_\tau(y)=\operatorname{sign}(y)(|y|-\tau)_+.
$$

The proximal operator is

$$
\operatorname{prox}_{\eta f}(y)
=\operatorname*{argmin}_x
\left\{
\frac12\|x-y\|^2+\eta f(x)
\right\}.
$$

For $f=\lambda\|\cdot\|_1$, proximal gradient gives ISTA:

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

where

$$
h(\theta)=\frac1{2n}\|y-\Phi\theta\|^2,
\qquad
\nabla h(\theta)=\frac1n\Phi^\top(\Phi\theta-y).
$$

If $\|\Phi^\top\varepsilon\|_\infty\le n\lambda/2$, the slide's slow-rate lemma gives

$$
\|\widehat\theta\|_1\le3\|\theta^\star\|_1,
$$

and

$$
\frac1n\|\Phi(\widehat\theta-\theta^\star)\|^2
\le3\lambda\|\theta^\star\|_1.
$$

**Book-aligned supplement.** LTfFP explains the geometry: on $[-1,1]^d$, the $\ell_1$ norm is the convex envelope of the $\ell_0$ count, and the corners of the $\ell_1$ ball favor coordinate axes. It also distinguishes two statistical regimes. The slow rate requires no design-matrix condition but is generally of order $n^{-1/2}$. A fast rate of order

$$
\frac{\sigma^2k\log d}{n}
$$

requires a restricted-eigenvalue-type inequality on the cone $\|\Delta_{A^c}\|_1\le3\|\Delta_A\|_1$. Coordinate descent is another practical solver because the $\ell_1$ penalty is separable. UML independently derives the same soft-thresholding operator and explains why ridge almost never produces exact zeros.

## 3. RIP and iterative hard thresholding

> **Pages:** Slides: L14, p. 6. Book supplements: UML Section 23.3, digital pp. 330-338; LTfFP Sections 8.2.1 and 8.3.5, book pp. 228 and 239-241 (PDF pp. 244 and 255-257).

A matrix $\Phi$ has $s$-restricted isometry constant $\delta_s$ if

$$
(1-\delta_s)\|x\|^2
\le
\|\Phi x\|^2
\le
(1+\delta_s)\|x\|^2
$$

for every $s$-sparse $x$.

IHT for sparse least squares is

$$
x_{t+1}
=H_s\left[x_t-\gamma\Phi^\top(\Phi x_t-y)\right].
$$

RIP says that the measurement operator is approximately an isometry on the union of sparse subspaces, which prevents distinct sparse signals from collapsing to the same observations.

**Book-aligned supplement.** UML makes the recovery consequence precise. If $W$ is $(\delta,2s)$-RIP with $\delta<1$, an $s$-sparse vector is the unique sparsest vector consistent with $y=Wx$. Under a stronger numerical RIP condition, the same vector is recovered by the convex program

$$
\min_v\|v\|_1
\quad\text{subject to}\quad
Wv=y.
$$

For nonsparse $x$, the reconstruction error is controlled by the $\ell_1$ tail outside its largest $s$ coordinates. A Gaussian measurement matrix needs on the order of

$$
s\log(d/\delta)/\varepsilon^2
$$

rows to satisfy an $(\varepsilon,s)$-RIP guarantee with high probability. LTfFP relates RIP to restricted eigenvalue and incoherence conditions used for Lasso fast rates.

## 4. From sparsity to low rank, clustering, column selection, and Nystrom

> **Pages:** Slides: L14, pp. 7-14. Book supplements: LTfFP Section 1.1.4, book pp. 6-7 (PDF pp. 22-23), and Section 7.4.2, book pp. 197-198 (PDF pp. 213-214); UML Chapter 22, digital pp. 307-320, and Section 25.3.1, digital pp. 369-370.

The central analogy is

$$
\operatorname{rank}(X)=\|\sigma(X)\|_0.
$$

Low rank is therefore sparsity of the singular-value vector.

For data $X\in\mathbb R^{d\times n}$, k-means can be written as

$$
\min_{M,C}\frac12\|X-MC\|_F^2,
$$

where columns of $M$ are centroids and columns of $C$ are one-hot assignments. Co-clustering uses

$$
X\approx RMC
$$

to cluster rows and columns simultaneously.

Column subset selection retains actual columns for interpretability, sketching, feature or sensor selection, and kernel approximation. For a positive-semidefinite kernel matrix, the Nystrom approximation is

$$
\widetilde K
=K_{:,C}K_{C,C}^\dagger K_{C,:}.
$$

A determinant-based distribution

$$
\Pr(C)\propto\det(K_{C,C})
$$

favors diverse columns.

**Book-aligned supplement.** UML expresses k-means as a sparse encoder-decoder: the encoder outputs a one-hot vector identifying the closest centroid, and the decoder returns that centroid. This places k-means and PCA in the same reconstruction framework while distinguishing a discrete sparse code from a linear low-dimensional code. LTfFP's kernel chapter explains the Nystrom construction as restricting the predictor to the span of selected kernel sections. With $q=|C|$ selected columns, it replaces dense kernel linear algebra by a rank-$q$ feature representation and roughly $O(nq^2)$ work. Determinant-based sampling remains slide material rather than a result derived in either reference.

## 5. Matrix completion and low-rank formulations

> **Pages:** Slides: L14, pp. 15-24. Book supplement: LTfFP Exercises 8.14-8.15, book p. 244 (PDF p. 260), for nuclear-norm thresholding and factorization. Neither LTfFP nor UML develops matrix-completion sampling theory or user-item completion guarantees.

Let $A\in\mathbb R^{n\times m}$ be observed only on $\Omega$. A valid estimator should be equivariant to row and column relabeling. Independent entrywise ridge regression cannot complete the matrix because it does not couple different entries.

A direct low-rank formulation is

$$
\min_{\widehat A}
\sum_{(i,j)\in\Omega}
(Y_{ij}-\widehat A_{ij})^2
\quad\text{subject to}\quad
\operatorname{rank}(\widehat A)\le k.
$$

The convex relaxation uses the nuclear norm

$$
\|A\|_*=\sum_i\sigma_i(A):
$$

$$
\min_A
\frac12\|P_\Omega(A-Y)\|_F^2
+\lambda\|A\|_*.
$$

Its proximal operator soft-thresholds singular values. The practical nonconvex alternative factors

$$
A=UV^\top
$$

and minimizes observed-entry loss over $U$ and $V$. Exact recovery requires assumptions such as low rank, sufficiently random observations, and incoherent singular vectors; the slides pose these as theory questions rather than proving a sample-complexity theorem.

**Book-aligned supplement.** For a fully observed matrix

$$
Y=U\operatorname{Diag}(\sigma_i)V^\top,
$$

the nuclear-norm proximal problem

$$
\min_\Theta
\left\{
\frac12\|Y-\Theta\|_F^2+\tau\|\Theta\|_*
\right\}
$$

has solution

$$
\Theta^\star
=
U\operatorname{Diag}((\sigma_i-\tau)_+)V^\top.
$$

Thus singular-value soft thresholding is exactly the matrix analogue of coordinatewise soft thresholding. LTfFP also gives the factorization identity

$$
\boxed{
\|M\|_*
=
\min_{M=UV^\top}
\frac12(\|U\|_F^2+\|V\|_F^2).
}
$$

It explains why Frobenius regularization of the factors is linked to nuclear-norm regularization of the product. These facts support the formulations in the slides but do not supply a matrix-completion recovery theorem.

## 6. Truncated SVD and alternating minimization

> **Pages:** Slides: L14, pp. 25-27. Book supplements: LTfFP Sections 1.1.4 and 3.9, book pp. 6-7 and 66-68 (PDF pp. 22-23 and 82-84); UML Section 23.1 and Appendix C, digital pp. 324-328 and 430-434. Neither book directly analyzes the slide's matrix-factorization AltMin landscape.

For

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^\top,
\qquad
\sigma_1\ge\cdots\ge\sigma_r,
$$

define

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^\top.
$$

The Eckart-Young-Mirsky theorem says that $A_k$ is a best rank-$k$ approximation for every unitarily invariant norm. In particular,

$$
\|A-A_k\|_2=\sigma_{k+1},
$$

$$
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2,
$$

and

$$
\|A-A_k\|_*=\sum_{i>k}\sigma_i.
$$

Factoring $X=UV^\top$ turns

$$
\min_{\operatorname{rank}(X)\le k}f(X)
$$

into an unconstrained but nonconvex problem in $U,V$. Alternating minimization fixes one factor and optimizes the other. Global guarantees require problem-specific assumptions; they do not follow from low rank alone.

**Book-aligned supplement.** UML derives truncated SVD through the PCA reconstruction problem and shows that storing the factors costs about $(n+m)k$ numbers rather than $nm$. LTfFP and UML both connect the squared tail singular values to discarded variance. For the fully observed factorization

$$
\min_{A,D}\|\Phi-AD\|_F^2,
$$

the alternating least-squares updates are

$$
A\leftarrow \Phi D^\top(DD^\top)^\dagger,
\qquad
D\leftarrow(A^\top A)^\dagger A^\top\Phi.
$$

LTfFP Exercise 3.9 states convergence to the globally best principal subspace for almost every initialization in this complete-observation PCA setting. That special result cannot be transferred automatically to missing-data matrix completion; there the landscape and convergence claims require additional sampling and incoherence assumptions.

## 7. Matrix sensing

> **Pages:** Slides: L14, p. 28. Book supplements: LTfFP Sections 12.3.3-12.3.4, book pp. 373-375 (PDF pp. 389-391), for a positive-semidefinite factorized special case; UML Section 23.3, digital pp. 330-338, for the vector compressed-sensing analogue.

Matrix sensing observes

$$
\mathcal A(M)
=
[\langle A_1,M\rangle,\ldots,\langle A_m,M\rangle]^\top
$$

and seeks a low-rank $M$. A rank-constrained gradient method projects by truncated SVD:

$$
M_{t+1}
=P_{\operatorname{rank}\le r}
\left[
M_t-\eta\mathcal A^*(\mathcal A(M_t)-b)
\right].
$$

If $M\succeq0$, then $\|M\|_*=\operatorname{tr}(M)$, allowing a semidefinite relaxation.

**Book-aligned supplement.** LTfFP studies the positive-semidefinite special case

$$
F(W)
=
\frac1n\sum_{i=1}^n
(\langle WW^\top,X_i\rangle-y_i)^2
=G(WW^\top),
$$

where $G(M)$ is convex in $M\succeq0$. The factorization $M=WW^\top$ turns a convex matrix problem into a nonconvex parameterization; under the book's additional assumptions, gradient-flow results can still reach the global value and may exhibit a minimum-nuclear-norm implicit bias. This is a PSD special case, not a theorem for every sensing operator in the slides.

UML's compressed-sensing chapter provides the vector analogy:

$$
\text{sparse vector}
\leftrightarrow
\text{low-rank matrix},
\qquad
\ell_1\text{ norm}
\leftrightarrow
\text{nuclear norm},
$$

and

$$
\text{hard coordinate thresholding}
\leftrightarrow
\text{truncated SVD}.
$$

Vector RIP motivates a matrix-RIP condition on low-rank differences, but that matrix-specific condition is not developed in either reference book.

## 8. Tensor estimation

> **Pages:** Slides: L14, pp. 29-33. Direct book supplement: neither LTfFP nor UML develops tensor decomposition or tensor deflation.

For vectors $u,v,w,x$,

$$
T=u\otimes v\otimes w\otimes x,
\qquad
T_{ijkl}=u_iv_jw_kx_l.
$$

For an orthogonally decomposable fourth-order tensor

$$
T=\sum_{i=1}^ru_i^{\otimes4},
$$

the contractions include

$$
T(v,v,v,v)=\sum_i(u_i^\top v)^4,
$$

$$
T(I,v,v,v)=\sum_i(u_i^\top v)^3u_i,
$$

and

$$
T(I,I,v,v)=\sum_i(u_i^\top v)^2u_iu_i^\top.
$$

A component can be recovered by

$$
\max_{\|u\|=1}T(u,u,u,u).
$$

The global maximizers are $\pm u_i$. After finding one component, deflation uses

$$
T\leftarrow T-u_i^{\otimes4}.
$$

The slide phrase "all local minima are global" should be read as "all local maxima are global" for the displayed maximization problem, or as a statement about minimizing the negative objective. General tensors do not possess a matrix-like universal SVD theory.

## 9. LoRA

> **Pages:** Slides: L14, pp. 34-35. Book supplement: LTfFP Exercise 8.15, book p. 244 (PDF p. 260), for the general nuclear-norm factorization identity. Neither book covers LoRA itself.

LoRA freezes a pretrained matrix $W_0$ and learns a low-rank update

$$
\Delta W=BA.
$$

Thus

$$
h=W_0x+BAx.
$$

For

$$
W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
$$

choose

$$
A\in\mathbb R^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

The number of trainable parameters drops from $d_{\mathrm{out}}d_{\mathrm{in}}$ to

$$
r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

Initializing $A$ randomly and $B=0$ gives $BA=0$, so the initial model exactly matches the pretrained model.

**Book-aligned supplement.** Applying LTfFP's general factorization identity to a LoRA update gives

$$
\|\Delta W\|_*
=
\min_{\Delta W=BA}
\frac12(\|B\|_F^2+\|A\|_F^2).
$$

For any particular factorization,

$$
\|BA\|_*
\le
\frac12(\|B\|_F^2+\|A\|_F^2).
$$

Consequently, Frobenius penalties on the two trainable factors indirectly control the nuclear norm of the update. This is an application of a general low-rank identity, not a LoRA result stated in either book.

## 10. Principal component analysis

> **Pages:** Slides: L14, pp. 36-56. Book supplements: LTfFP Section 3.9, book pp. 66-68 (PDF pp. 82-84); UML Section 23.1, digital pp. 324-328, and Section 23.6, digital p. 339.

For data $x_1,\ldots,x_n\in\mathbb R^d$, define

$$
\bar x=\frac1n\sum_i x_i,
\qquad
S=\frac1n\sum_i(x_i-\bar x)(x_i-\bar x)^\top.
$$

The maximum-variance first principal direction solves

$$
\max_{\|u\|=1}u^\top Su,
$$

so it is a top eigenvector of $S$. For $k$ directions,

$$
\max_{U^\top U=I_k}\operatorname{tr}(U^\top SU),
$$

whose solution consists of the top $k$ eigenvectors.

The minimum-reconstruction-error view chooses a $k$-dimensional affine subspace. Its optimal offset is the sample mean, and its minimum error is

$$
\sum_{j=k+1}^d\lambda_j(S).
$$

Since total variance is fixed,

$$
\sum_{j=1}^k\lambda_j(S)
+\sum_{j=k+1}^d\lambda_j(S)
=\operatorname{tr}(S),
$$

maximizing retained variance is equivalent to minimizing reconstruction error.

The centered encoder and decoder are

$$
z=U^\top(x-\bar x),
$$

$$
\widehat x
=\bar x+Uz
=\bar x+UU^\top(x-\bar x).
$$

If

$$
X_c=[x_1-\bar x,\ldots,x_n-\bar x]
=U\Sigma V^\top,
$$

then

$$
S=\frac1nX_cX_c^\top,
\qquad
\lambda_i(S)=\frac{\sigma_i^2}{n}.
$$

Thus PCA, top singular vectors, and best rank-$k$ approximation are the same construction.

**Book-aligned supplement.** UML proves that, among all linear encoders $W\in\mathbb R^{k\times d}$ and linear decoders $U\in\mathbb R^{d\times k}$, an optimum can be chosen with orthonormal decoder columns and $W=U^\top$. If $d\gg n$, one can diagonalize the smaller Gram matrix $X_c^\top X_c$ and map its eigenvectors back through $X_c$, reducing the eigendecomposition cost. The same observation leads to kernel PCA because only inner products are required.

LTfFP also gives a supervised warning through PCA followed by least squares. If $V$ contains the top $k$ sample principal directions and $\widehat\Sigma=\Phi^\top\Phi/n$, then the expected prediction error decomposes as

$$
\frac1n\mathbb E_\varepsilon
\|\Phi V\widehat\eta-\Phi\theta^\star\|^2
=
\frac{\sigma^2k}{n}
+
(\theta^\star)^\top
(I-VV^\top)\widehat\Sigma(I-VV^\top)\theta^\star.
$$

The first term is variance from fitting $k$ coordinates; the second is bias from discarding the remaining directions. Therefore PCA is optimal for unsupervised squared reconstruction, but it does not necessarily preserve the directions most useful for a downstream prediction task.
