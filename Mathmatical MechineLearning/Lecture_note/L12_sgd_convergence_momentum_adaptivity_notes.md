# Lecture Note: SGD, Convergence, Momentum, and Adaptivity

Based on **L12: Optimization and Learning - 2** by Suvrit Sra. This note follows the whole slide deck: large-scale empirical risk minimization, finite-sum SGD, expected risk, stochastic gradients, mini-batches, convergence theory, convex and strongly convex analysis, clipping, normalized gradients, momentum, and adaptivity.

## 1. Large-scale learning and empirical risk minimization

In machine learning, we often minimize a regularized empirical risk of the form

$$
\min_{\theta\in\mathbb R^d}
\left\{
F(\theta):=\frac1n\sum_{i=1}^n f_i(\theta)+\lambda r(\theta)
\right\}.
$$

Here:

- $n$ is the number of training samples.
- $d$ is the number of parameters or features.
- $f_i(\theta)$ is the loss contributed by the $i$-th data point.
- $r(\theta)$ is a regularizer.
- $\lambda\ge0$ controls the strength of regularization.

In large-scale ML, both $n$ and $d$ can be huge. Full gradient descent uses

$$
\nabla F(\theta)=\frac1n\sum_{i=1}^n \nabla f_i(\theta)+\lambda\nabla r(\theta),
$$

so one iteration may require all $n$ component gradients. When $n$ is large, this is expensive.

The main motivation for SGD is to replace expensive full gradients by cheap noisy gradients.

## 2. Finite-sum SGD

For the finite-sum problem

$$
F(\theta)=\frac1n\sum_{i=1}^n f_i(\theta),
$$

full gradient descent uses

$$
\theta_{k+1}=\theta_k-\gamma_k\nabla F(\theta_k).
$$

SGD instead chooses a random index $i(k)$ and uses only one component gradient:

$$
\theta_{k+1}=\theta_k-\gamma_k\nabla f_{i(k)}(\theta_k).
$$

If

$$
i(k)\sim\operatorname{Unif}\{1,\ldots,n\},
$$

then

$$
\mathbb E[\nabla f_{i(k)}(\theta_k)\mid\theta_k]
=
\frac1n\sum_{i=1}^n\nabla f_i(\theta_k)
=
\nabla F(\theta_k).
$$

This is the key property: the stochastic gradient is noisy but unbiased.

## 3. Sampling variants

The slides distinguish two common ways to select data points.

**Sampling with replacement:** at each iteration, choose

$$
i(k)\sim\operatorname{Unif}\{1,\ldots,n\}.
$$

The same data point can be selected many times before another point is selected.

**Sampling without replacement:** shuffle the dataset and pass through it, often one epoch at a time. This is common in practical ML training.

Theoretical analysis is usually cleaner for sampling with replacement because the random index at step $k$ is independent of the past, conditional on $\theta_k$.

## 4. Expected risk versus empirical risk

The empirical risk is based on a finite sample:

$$
F_n(\theta)=\frac1n\sum_{i=1}^n\ell(y_i,f_\theta(x_i)).
$$

The expected risk, or population risk, is

$$
F(\theta)=\mathbb E[\ell(Y,f_\theta(X))].
$$

In expected risk minimization, SGD can be viewed as receiving a fresh sample $(x_k,y_k)$ at iteration $k$ and using the stochastic gradient of

$$
\theta\mapsto \ell(y_k,f_\theta(x_k)).
$$

This means SGD is directly optimizing the population objective in expectation.

In simple convex cases, the excess risk often scales like

$$
O\left(\frac1{\sqrt n}\right).
$$

The $1/\sqrt n$ behavior comes from balancing optimization progress and stochastic noise.

## 5. Scalar least-squares example

The slides use a scalar least-squares problem:

$$
F(\theta)=\frac12\sum_{i=1}^n(x_i\theta-b_i)^2.
$$

Here

$$
x_i,b_i,\theta\in\mathbb R,
$$

so $x_i\theta-b_i$ is ordinary scalar multiplication.

For one component,

$$
f_i(\theta)=\frac12(x_i\theta-b_i)^2,
$$

we have

$$
f_i'(\theta)=x_i(x_i\theta-b_i).
$$

The factor $1/2$ makes the derivative cleaner.

The component minimizer satisfies

$$
x_i\theta-b_i=0,
$$

so

$$
\theta_i^*=\frac{b_i}{x_i}.
$$

The global minimizer satisfies

$$
\sum_{i=1}^n x_i(x_i\theta-b_i)=0,
$$

hence

$$
\boxed{
\theta^*=\frac{\sum_{i=1}^n x_i b_i}{\sum_{i=1}^n x_i^2}.
}
$$

This minimizer lies in the interval

$$
R=\left[\min_i\theta_i^*,\max_i\theta_i^*\right].
$$

Outside $R$, every component gradient points in a useful direction. Inside $R$, a randomly chosen component gradient may point away from the global optimum. This is why SGD can make progress in expectation without improving at every single step.

## 6. Mini-batch SGD

Mini-batch SGD uses a random subset

$$
I_k\subseteq\{1,\ldots,n\}
$$

at iteration $k$. The update is

$$
\theta_{k+1}
=
\theta_k-
\frac{\gamma_k}{|I_k|}
\sum_{j\in I_k}\nabla f_j(\theta_k).
$$

The mini-batch gradient is

$$
g_k=\frac1{|I_k|}\sum_{j\in I_k}\nabla f_j(\theta_k).
$$

The division by $|I_k|$ makes this an average rather than a sum.

Mini-batches are useful because they reduce gradient noise and improve hardware efficiency through parallelism. The tradeoff is that larger batches cost more computation per update. The slides also note that very large mini-batches are not always favorable for deep networks.

## 7. Randomness and conditional unbiasedness

Let $\xi_k$ denote the randomness at step $k$. For example,

$$
\xi_k=i(k)
$$

for single-sample SGD, or

$$
\xi_k=I_k
$$

for mini-batch SGD.

The stochastic gradient is written as

$$
g_k=g(\theta_k,\xi_k).
$$

A crucial point is

$$
\theta_k \text{ depends on } \xi_1,\ldots,\xi_{k-1},
$$

but

$$
\theta_k \text{ does not depend on } \xi_k.
$$

That is because $\theta_k$ is already fixed before the fresh randomness for step $k$ is drawn.

The precise unbiasedness condition is therefore conditional:

$$
\boxed{
\mathbb E[g_k\mid\theta_k]=\nabla F(\theta_k).
}
$$

This is the formal version of saying that the stochastic gradient points in the correct direction on average.

## 8. GD versus SGD in the strongly convex case

For strongly convex finite-sum problems, batch GD and SGD have different cost profiles.

Batch GD has fast linear convergence in iterations, but every iteration uses all $n$ gradients. A typical complexity in gradient evaluations is of the form

$$
O\left(n\log\frac1\varepsilon\right).
$$

SGD uses only one or a small batch of component gradients per iteration. Its convergence is slower in iteration count, often sublinear, but each iteration is much cheaper.

The slide summary is:

$$
\text{GD: faster per iteration, expensive iterations.}
$$

$$
\text{SGD: noisier per iteration, cheap iterations.}
$$

This is why SGD is preferred for large-scale ML.

## 9. Smoothness and the Taylor upper bound

A differentiable function $f$ is $L$-smooth if

$$
f(y)\le f(x)+\langle\nabla f(x),y-x\rangle+
\frac L2\|y-x\|^2.
$$

For SGD,

$$
\theta_{k+1}=\theta_k-\gamma_k g_k.
$$

Thus

$$
y-x=\theta_{k+1}-\theta_k=-\gamma_k g_k.
$$

Substituting into the smoothness inequality gives

$$
F(\theta_{k+1})
\le
F(\theta_k)
-
\gamma_k\langle\nabla F(\theta_k),g_k\rangle
+
\frac L2\gamma_k^2\|g_k\|^2.
$$

Taking conditional expectation and using unbiasedness,

$$
\mathbb E[\langle\nabla F(\theta_k),g_k\rangle\mid\theta_k]
=
\|\nabla F(\theta_k)\|^2.
$$

So the expected progress inequality becomes

$$
\mathbb E[F(\theta_{k+1})]
\le
\mathbb E[F(\theta_k)]
-
\gamma_k\mathbb E[\|\nabla F(\theta_k)\|^2]
+
\frac L2\gamma_k^2G^2,
$$

assuming $\|g_k\|\le G$.

The negative term is true gradient progress. The positive quadratic term is the cost of noisy stochastic gradients.

## 10. Nonconvex smooth SGD theorem

For nonconvex smooth objectives, we usually cannot guarantee convergence to a global minimizer. Instead, we prove convergence toward stationarity.

Assume:

- each component $f_i$ is $L$-smooth;
- stochastic gradients are unbiased;
- stochastic gradient noise is controlled;
- stochastic gradients are bounded, for example $\|\nabla f_i(\theta)\|\le G$.

With a constant step size

$$
\gamma_k=\frac{c}{\sqrt T},
$$

the slides state a bound of the form

$$
\frac1T\sum_{k=1}^T
\mathbb E[\|\nabla F(\theta_k)\|^2]
\le
\frac1{\sqrt T}
\left(
\frac{F(\theta_1)-F(\theta^*)}{c}
+
\frac{Lc}{2}G^2
\right).
$$

Therefore,

$$
\min_{1\le k\le T}\mathbb E[\|\nabla F(\theta_k)\|^2]
=O\left(\frac1{\sqrt T}\right).
$$

This means SGD finds an approximate stationary point on average.

The step size $c/\sqrt T$ balances two competing terms:

$$
\frac1{\gamma T}
\quad\text{and}\quad
\gamma.
$$

Choosing $\gamma\asymp1/\sqrt T$ makes both terms order $1/\sqrt T$.

## 11. Convex projected SGD setup

For constrained stochastic optimization,

$$
\min_{\theta\in\mathcal X}F(\theta),
$$

projected SGD is

$$
\theta_{k+1}=P_{\mathcal X}(\theta_k-\gamma_k g_k),
$$

where $P_{\mathcal X}$ is Euclidean projection onto the feasible set $\mathcal X$.

Let $\theta^*$ be an optimal point and define

$$
R_k=\|\theta_k-\theta^*\|^2,
\qquad
r_k=\mathbb E[R_k].
$$

Projection is nonexpansive:

$$
\|P_{\mathcal X}(a)-P_{\mathcal X}(b)\|
\le
\|a-b\|.
$$

Since $\theta^*=P_{\mathcal X}(\theta^*)$, we have

$$
\begin{aligned}
R_{k+1}
&=\|P_{\mathcal X}(\theta_k-\gamma_k g_k)-P_{\mathcal X}(\theta^*)\|^2\\
&\le\|\theta_k-\theta^*-\gamma_k g_k\|^2.
\end{aligned}
$$

Expanding the square gives

$$
\boxed{
R_{k+1}
\le
R_k+
\gamma_k^2\|g_k\|^2
-
2\gamma_k\langle g_k,\theta_k-\theta^*\rangle.
}
$$

This is the core recursion in convex SGD analysis.

## 12. Strongly convex SGD analysis

Assume

$$
\|g_k\|\le G
$$

and

$$
\mathbb E[g_k\mid\theta_k]=\nabla F(\theta_k).
$$

Taking expectations in the distance recursion gives

$$
r_{k+1}
\le
r_k+\gamma_k^2G^2
-
2\gamma_k\mathbb E[\langle\nabla F(\theta_k),\theta_k-\theta^*\rangle].
$$

If $F$ is $\mu$-strongly convex, then

$$
\langle\nabla F(\theta_k),\theta_k-\theta^*\rangle
\ge
F(\theta_k)-F(\theta^*)+
\frac\mu2\|\theta_k-\theta^*\|^2.
$$

Substituting gives

$$
r_{k+1}
\le
r_k+\gamma_k^2G^2
-
2\gamma_k\mathbb E[F(\theta_k)-F(\theta^*)]
-
\mu\gamma_k r_k.
$$

Rearranging,

$$
\mathbb E[F(\theta_k)-F(\theta^*)]
\le
\frac{\gamma_kG^2}{2}
+
\frac{1-\mu\gamma_k}{2\gamma_k}r_k
-
\frac1{2\gamma_k}r_{k+1}.
$$

With

$$
\gamma_k=\frac1{\mu k},
$$

we get

$$
1-\mu\gamma_k=1-\frac1k=\frac{k-1}{k},
$$

which makes the recursion telescope after summing over $k$.

## 13. Averaging and the strongly convex rate

Using the telescoping argument, the slides obtain

$$
\frac1T\sum_{k=1}^T
\mathbb E[F(\theta_k)-F(\theta^*)]
\le
\frac{G^2}{2\mu T}
\sum_{k=1}^T\frac1k.
$$

Since

$$
\sum_{k=1}^T\frac1k\le1+\\log T,
$$

we get

$$
\boxed{
\frac1T\sum_{k=1}^T
\mathbb E[F(\theta_k)-F(\theta^*)]
\le
\frac{G^2}{2\mu T}(1+\log T).
}
$$

Now define the averaged iterate

$$
\bar\theta_T=\frac1T\sum_{k=1}^T\theta_k.
$$

By convexity,

$$
F(\bar\theta_T)
\le
\frac1T\sum_{k=1}^T F(\theta_k).
$$

Therefore the same average bound gives a guarantee for the single output point $\bar\theta_T$.

The slide exercise also states a weighted averaging result. With suitable weights, one can obtain a cleaner rate of the form

$$
\mathbb E[F(\bar\theta_k)-F(\theta^*)]
\le
\frac{2G^2}{\mu(k+1)}.
$$

The point is that averaging is not merely cosmetic: it converts bounds on many iterates into a bound on one final returned point.

## 14. Convexity and strong convexity reminders

A function $F$ is convex if

$$
F(\lambda x+(1-\lambda)y)
\le
\lambda F(x)+(1-\lambda)F(y),
\qquad
\lambda\in[0,1].
$$

For differentiable $F$, convexity is equivalent to the first-order inequality

$$
F(y)
\ge
F(x)+\langle\nabla F(x),y-x\rangle.
$$

Strong convexity adds a quadratic term:

$$
F(y)
\ge
F(x)+\langle\nabla F(x),y-x\rangle
+
\frac\mu2\|y-x\|^2.
$$

The parameter $\mu>0$ measures curvature. If $\mu=0$, we recover ordinary convexity.

For a quadratic

$$
F(\theta)=\frac12\theta^\top A\theta,
$$

with $A\succeq0$, the strong convexity constant is

$$
\mu=\lambda_{\min}(A).
$$

## 15. Gradient clipping

Gradient clipping is introduced as a practical method for dealing with gradient explosion. A clipped gradient step is

$$
x_{k+1}
=
x_k-
\min\left(\eta,\frac{a\eta}{\|\nabla f(x_k)\|}\right)
\nabla f(x_k).
$$

If $\|\nabla f(x_k)\|$ is small, the method behaves like standard gradient descent with step size $\eta$.

If $\|\nabla f(x_k)\|$ is large, the effective step length is capped:

$$
\left\|
\min\left(\eta,\frac{a\eta}{\|\nabla f(x_k)\|}\right)
\nabla f(x_k)
\right\|
\le
 a\eta.
$$

This prevents very large gradients from causing unstable updates.

The slides emphasize that clipping is widely used in noisy or nonsmooth-like regimes, such as language modeling.

## 16. Normalized gradient descent

Normalized gradient descent uses

$$
x_{k+1}
=
x_k-
\frac{\eta}{\|\nabla f(x_k)\|+b}
\nabla f(x_k),
$$

where $b>0$ prevents division by zero.

The effective step length is

$$
\frac{\eta\|\nabla f(x_k)\|}{\|\nabla f(x_k)\|+b}.
$$

When the gradient is large, this step length is close to $\eta$. When the gradient is small, the step length becomes smaller.

Clipped GD and normalized GD are related because both rescale the update based on the gradient norm. The slide exercise asks to show that clipped GD is comparable to normalized GD up to a constant factor in step size.

## 17. Momentum

Gradient descent with momentum introduces a memory variable $m_t$. A standard form is

$$
m_t=\beta m_{t-1}+\nabla f(\theta_t),
\qquad
0\le\beta<1,
$$

and

$$
\theta_{t+1}=\theta_t-\eta m_t.
$$

Here $m_t$ is an exponentially weighted history of past gradients.

Unrolling the momentum variable gives

$$
m_t
=
\nabla f(\theta_t)
+
\beta\nabla f(\theta_{t-1})
+
\beta^2\nabla f(\theta_{t-2})+
\cdots.
$$

So recent gradients receive larger weight, while older gradients are geometrically discounted.

If $\beta=0$, then

$$
m_t=\nabla f(\theta_t),
$$

and the method reduces to ordinary gradient descent.

If $\beta$ is close to $1$, the method strongly remembers past directions.

Momentum is useful for ill-conditioned problems, including convex quadratics, because it can damp oscillation in high-curvature directions while accumulating movement in consistent low-curvature directions.

## 18. Unrolling GD with and without momentum

Ordinary gradient descent satisfies

$$
\theta_{t+1}=\theta_0-
\eta\sum_{i=0}^{t}\nabla f(\theta_i).
$$

With momentum,

$$
\theta_{t+1}
=
\theta_0-
\eta\sum_{i=0}^{t}
\left(
\sum_{j=i}^{t}\beta^{j-i}
\right)
\nabla f(\theta_i).
$$

The inner geometric sum is

$$
\sum_{j=i}^{t}\beta^{j-i}
=
1+\beta+\cdots+\beta^{t-i}
=
\frac{1-\beta^{t+1-i}}{1-\beta}.
$$

Thus

$$
\theta_{t+1}
=
\theta_0-
\eta\sum_{i=0}^{t}
\frac{1-\beta^{t+1-i}}{1-\beta}
\nabla f(\theta_i).
$$

This formula shows that momentum gives larger cumulative weight to directions that persist over many iterations.

## 19. Adaptivity and coordinate-wise scaling

Adaptivity means the optimizer changes the effective step size based on observed gradients.

Plain SGD uses one global step size:

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

Adaptive methods use coordinate-wise scaling. A generic form is

$$
\theta_{t+1,j}
=
\theta_{t,j}
-
\frac{\eta}{\sqrt{v_{t,j}}+\epsilon}g_{t,j}.
$$

The effective coordinate-wise step size is

$$
\eta_{t,j}^{\mathrm{eff}}
=
\frac{\eta}{\sqrt{v_{t,j}}+\epsilon}.
$$

If past gradients in coordinate $j$ are large, then $v_{t,j}$ is large and the step becomes smaller. If past gradients are small, the step becomes larger.

Adam and Adam-like methods combine momentum-type averaging of gradients with adaptive coordinate-wise scaling.

## 20. Big-picture summary

The main story of the lecture is:

$$
\text{Full GD is accurate but expensive.}
$$

$$
\text{SGD is cheap but noisy.}
$$

$$
\text{Mini-batches reduce noise and improve hardware use.}
$$

$$
\text{Smoothness converts stochastic updates into expected progress inequalities.}
$$

$$
\text{Convexity and strong convexity convert distance recursions into risk bounds.}
$$

$$
\text{Averaging turns many-iterate guarantees into a guarantee for one output point.}
$$

$$
\text{Momentum uses past gradients to smooth and accelerate motion.}
$$

$$
\text{Adaptive methods rescale steps using gradient size or gradient history.}
$$

The conceptual formula behind the lecture is

$$
\boxed{
\text{SGD progress}
=
\text{true gradient signal}
-
\text{stochastic noise penalty}.
}
$$

The analysis is about balancing step size, unbiasedness, smoothness, convexity, strong convexity, and variance.

## 21. Key formulas to remember

Finite-sum objective:

$$
F(\theta)=\frac1n\sum_{i=1}^n f_i(\theta).
$$

SGD update:

$$
\theta_{k+1}=\theta_k-\gamma_k\nabla f_{i(k)}(\theta_k).
$$

Mini-batch update:

$$
\theta_{k+1}
=
\theta_k-
\frac{\gamma_k}{|I_k|}
\sum_{j\in I_k}\nabla f_j(\theta_k).
$$

Conditional unbiasedness:

$$
\mathbb E[g_k\mid\theta_k]=\nabla F(\theta_k).
$$

Smoothness inequality:

$$
F(y)\le F(x)+\langle\nabla F(x),y-x\rangle+
\frac L2\|y-x\|^2.
$$

Projected SGD distance recursion:

$$
R_{k+1}
\le
R_k+\gamma_k^2\|g_k\|^2
-2\gamma_k\langle g_k,\theta_k-\theta^*\rangle.
$$

Strongly convex averaged SGD rate:

$$
\frac1T\sum_{k=1}^T
\mathbb E[F(\theta_k)-F(\theta^*)]
\le
\frac{G^2}{2\mu T}(1+\log T).
$$

Momentum:

$$
m_t=\beta m_{t-1}+\nabla f(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta m_t.
$$

Clipped GD:

$$
x_{k+1}
=
x_k-
\min\left(\eta,\frac{a\eta}{\|\nabla f(x_k)\|}\right)
\nabla f(x_k).
$$

Normalized GD:

$$
x_{k+1}
=
x_k-
\frac{\eta}{\|\nabla f(x_k)\|+b}
\nabla f(x_k).
$$
