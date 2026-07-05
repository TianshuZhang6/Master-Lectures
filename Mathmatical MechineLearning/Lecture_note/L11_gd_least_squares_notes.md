# Lecture Note: Gradient Descent on Least Squares

These notes summarize the **GD on least-squares: a closer look** section from L11, mainly pages 29-31. The formatting is designed for VS Code Jupyter notebooks: inline math uses `$...$`, and display equations use `$$...$$`.

## 1. Least-squares objective

We study the least-squares loss

$$
F(\theta)=\frac{1}{2n}\|\Phi\theta-y\|_2^2,
$$

where

$$
\Phi\in\mathbb R^{n\times d}
$$

is the data/design matrix, and

$$
y\in\mathbb R^n
$$

is the vector of target values.

The gradient is

$$
\nabla F(\theta)=\frac1n\Phi^\top(\Phi\theta-y).
$$

The Hessian is

$$
H=\nabla^2F(\theta)=\frac1n\Phi^\top\Phi.
$$

So $H$ is symmetric and positive semidefinite.

## 2. Where does $\theta^*$ come from?

$\theta^*$ means the **optimal parameter vector**, meaning a minimizer of $F$:

$$
\theta^*\in\operatorname*{argmin}_\theta F(\theta).
$$

For a differentiable convex function, an optimum satisfies

$$
\nabla F(\theta^*)=0.
$$

For least squares,

$$
\nabla F(\theta^*)=\frac1n\Phi^\top(\Phi\theta^*-y)=0.
$$

Expanding gives

$$
\frac1n\Phi^\top\Phi\theta^*-
\frac1n\Phi^\top y=0.
$$

Therefore,

$$
\frac1n\Phi^\top\Phi\theta^*=
\frac1n\Phi^\top y.
$$

Since

$$
H=\frac1n\Phi^\top\Phi,
$$

we get the optimality equation

$$
\boxed{
H\theta^*=\frac1n\Phi^\top y.
}
$$

That is where $\theta^*$ comes from.

## 3. Why is there always a solution $\theta^*$?

The least-squares objective is

$$
F(\theta)=\frac{1}{2n}\|\Phi\theta-y\|^2.
$$

It is convex and nonnegative. More specifically, minimizing $F$ is the same as finding the closest point to $y$ in the column space of $\Phi$:

$$
\min_\theta \|\Phi\theta-y\|^2.
$$

The column space of $\Phi$ is a finite-dimensional closed subspace of $\mathbb R^n$, so the orthogonal projection of $y$ onto that subspace exists. Therefore at least one minimizer exists.

If $H$ is invertible, the minimizer is unique:

$$
\theta^*=H^{-1}\frac1n\Phi^\top y.
$$

If $H$ is singular, there may be many minimizers, but at least one minimizer still exists.

This matters especially when $d>n$. Then $\Phi^\top\Phi$ cannot have full rank, so $H$ has zero eigenvalues.

## 4. Gradient descent update

Gradient descent with fixed step size $\gamma$ is

$$
\theta_t=\theta_{t-1}-\gamma\nabla F(\theta_{t-1}).
$$

Using

$$
\nabla F(\theta)=H\theta-\frac1n\Phi^\top y,
$$

we get

$$
\theta_t
=
\theta_{t-1}
-
\gamma\left(H\theta_{t-1}-\frac1n\Phi^\top y\right).
$$

From the optimality condition,

$$
H\theta^*=\frac1n\Phi^\top y.
$$

So

$$
\nabla F(\theta_{t-1})
=
H\theta_{t-1}-H\theta^*
=
H(\theta_{t-1}-\theta^*).
$$

Therefore,

$$
\theta_t
=
\theta_{t-1}-\gamma H(\theta_{t-1}-\theta^*).
$$

Subtract $\theta^*$ from both sides:

$$
\theta_t-\theta^*
=
\theta_{t-1}-\theta^*
-
\gamma H(\theta_{t-1}-\theta^*).
$$

Factoring gives

$$
\boxed{
\theta_t-\theta^*
=
(I-\gamma H)(\theta_{t-1}-\theta^*).
}
$$

Unrolling the recursion gives

$$
\boxed{
\theta_t-\theta^*
=
(I-\gamma H)^t(\theta_0-\theta^*).
}
$$

This is the key formula.

## 5. Should the formula use $\theta_0-\theta^*$ or $\theta_t-\theta^*$?

In the standard derivation, the unrolled recursion uses the **initial error**:

$$
\theta_0-\theta^*.
$$

Since

$$
\theta_t-\theta^*
=
(I-\gamma H)^t(\theta_0-\theta^*),
$$

we get

$$
\|\theta_t-\theta^*\|^2
=
(\theta_0-\theta^*)^\top
(I-\gamma H)^{2t}
(\theta_0-\theta^*),
$$

because $H$ is symmetric, so $I-\gamma H$ is also symmetric.

So the clean formula is

$$
\boxed{
\|\theta_t-\theta^*\|^2
=
(\theta_0-\theta^*)^\top
(I-\gamma H)^{2t}
(\theta_0-\theta^*).
}
$$

If one writes

$$
(\theta_t-\theta^*)^\top(I-\gamma H)^{2t}(\theta_t-\theta^*),
$$

then that is not the usual unrolled parameter-error formula. It likely reflects a typo or misleading notation.

## 6. Why look at $\theta_t-\theta^*$?

There are two natural ways to judge convergence:

$$
\|\theta_t-\theta^*\|^2
$$

and

$$
F(\theta_t)-F(\theta^*).
$$

The parameter error is easier to analyze first because it follows the clean linear recursion

$$
\theta_t-\theta^*
=
(I-\gamma H)^t(\theta_0-\theta^*).
$$

This lets us study convergence using eigenvalues of $H$.

For this quadratic problem, the function gap is

$$
F(\theta)-F(\theta^*)
=
\frac12(\theta-\theta^*)^\top H(\theta-\theta^*).
$$

When $\mu>0$, the two quantities are closely related:

$$
\frac{\mu}{2}\|\theta_t-\theta^*\|^2
\le
F(\theta_t)-F(\theta^*)
\le
\frac{L}{2}\|\theta_t-\theta^*\|^2.
$$

So when $H$ is positive definite, controlling parameter error also controls function error.

When $\mu=0$, the parameter-error bound becomes weaker, so the lecture switches to studying function values.

## 7. Parameter error versus function gap

For deriving the convergence rate of gradient descent, parameter error is usually easier:

$$
\|\theta_t-\theta^*\|^2.
$$

For judging optimization performance, function gap is usually more meaningful:

$$
F(\theta_t)-F(\theta^*).
$$

So:

$$
\boxed{
\|\theta_t-\theta^*\|^2
\text{ is better for clean derivations.}
}
$$

and

$$
\boxed{
F(\theta_t)-F(\theta^*)
\text{ is better for judging objective performance.}
}
$$

If $H$ is positive definite, both are essentially equivalent. If $H$ is singular, function values are often the better quantity to study.

## 8. Eigenvalues and the shrinkage factor

Let the eigenvalues of $H$ lie in

$$
[\mu,L],
$$

where

$$
\mu=\lambda_{\min}(H),
\qquad
L=\lambda_{\max}(H).
$$

The condition number is

$$
\boxed{
\kappa=\frac{L}{\mu}.
}
$$

Because $H$ is symmetric, it has eigenvectors $v_i$ and eigenvalues $\lambda_i$:

$$
Hv_i=\lambda_i v_i.
$$

Now apply $I-\gamma H$:

$$
\begin{aligned}
(I-\gamma H)v_i
&=v_i-\gamma Hv_i\\
&=v_i-\gamma\lambda_i v_i\\
&=(1-\gamma\lambda_i)v_i.
\end{aligned}
$$

So the eigenvalues of $I-\gamma H$ are

$$
1-\gamma\lambda_i.
$$

The eigenvalues of $(I-\gamma H)^{2t}$ are

$$
(1-\gamma\lambda_i)^{2t}.
$$

The worst-case eigendirection is controlled by

$$
\boxed{
q(\gamma)=\max_{\lambda\in[\mu,L]}|1-\gamma\lambda|.
}
$$

Then

$$
\|\theta_t-\theta^*\|^2
\le
q(\gamma)^{2t}\|\theta_0-\theta^*\|^2.
$$

## 9. Why does smaller $q(\gamma)$ mean faster error decrease?

The bound is

$$
\|\theta_t-\theta^*\|^2
\le
q(\gamma)^{2t}\|\theta_0-\theta^*\|^2.
$$

So $q(\gamma)^{2t}$ is the shrinkage factor.

If $q(\gamma)<1$, then

$$
q(\gamma)^{2t}\to0.
$$

Smaller $q(\gamma)$ means this goes to zero faster.

For example,

$$
0.9^{20}\approx0.12,
$$

but

$$
0.5^{20}\approx 9.54\times10^{-7}.
$$

So $q=0.5$ gives much faster convergence than $q=0.9$.

## 10. Why minimize $q(\gamma)$?

Since

$$
q(\gamma)=\max_{\lambda\in[\mu,L]}|1-\gamma\lambda|
$$

controls the worst possible eigendirection, minimizing $q(\gamma)$ gives the fastest worst-case convergence.

So the step-size problem becomes

$$
\boxed{
\min_\gamma \max_{\lambda\in[\mu,L]}|1-\gamma\lambda|.
}
$$

The max occurs at the endpoints:

$$
q(\gamma)=\max\{|1-\gamma\mu|,\ |1-\gamma L|\}.
$$

To make the worst case as small as possible, balance the two endpoint errors:

$$
|1-\gamma\mu|=|1-\gamma L|.
$$

At the optimum, one endpoint is positive and the other is negative:

$$
1-\gamma\mu=-(1-\gamma L).
$$

Then

$$
1-\gamma\mu=-1+\gamma L.
$$

So

$$
2=\gamma(L+\mu).
$$

Therefore,

$$
\boxed{
\gamma^*=\frac{2}{L+\mu}.
}
$$

This is the optimal fixed step size for the worst-case parameter-error contraction.

## 11. Where does $\frac{\kappa-1}{\kappa+1}$ come from?

Plug

$$
\gamma^*=\frac{2}{L+\mu}
$$

into

$$
q(\gamma)=\max\{|1-\gamma\mu|,\ |1-\gamma L|\}.
$$

Use the $\mu$ endpoint:

$$
\begin{aligned}
1-\gamma^*\mu
&=1-\frac{2\mu}{L+\mu}\\
&=\frac{L+\mu-2\mu}{L+\mu}\\
&=\frac{L-\mu}{L+\mu}.
\end{aligned}
$$

So

$$
q(\gamma^*)=\frac{L-\mu}{L+\mu}.
$$

Using

$$
\kappa=\frac{L}{\mu},
$$

and dividing numerator and denominator by $\mu$ gives

$$
\frac{L-\mu}{L+\mu}
=
\frac{L/\mu-1}{L/\mu+1}
=
\frac{\kappa-1}{\kappa+1}.
$$

Thus

$$
\boxed{
q(\gamma^*)=
\frac{\kappa-1}{\kappa+1}.
}
$$

Also,

$$
\frac{\kappa-1}{\kappa+1}
=
\frac{\kappa+1-2}{\kappa+1}
=
1-\frac{2}{\kappa+1}.
$$

So

$$
\boxed{
q(\gamma^*)=
1-\frac{2}{\kappa+1}.
}
$$

## 12. Where does $1-\frac1\kappa$ come from?

Another common step size is

$$
\gamma=\frac1L.
$$

Then

$$
q(\gamma)=
\max\left\{
\left|1-\frac{\mu}{L}\right|,
\left|1-\frac{L}{L}\right|
\right\}.
$$

The second term is

$$
\left|1-\frac{L}{L}\right|=0.
$$

The first term is

$$
1-\frac{\mu}{L}.
$$

So

$$
q\left(\frac1L\right)
=
1-\frac{\mu}{L}.
$$

Since

$$
\kappa=\frac{L}{\mu},
$$

we have

$$
\frac{\mu}{L}=\frac1\kappa.
$$

Therefore,

$$
\boxed{
q\left(\frac1L\right)=1-\frac1\kappa.
}
$$

## 13. Do both step sizes lead to the optimal value?

No. This is an important distinction.

The step size

$$
\boxed{
\gamma=\frac{2}{L+\mu}
}
$$

is the **optimal step size** for minimizing the worst-case contraction factor. It gives

$$
q^*=rac{\kappa-1}{\kappa+1}=1-\frac{2}{\kappa+1}.
$$

The step size

$$
\boxed{
\gamma=\frac1L
}
$$

is not usually optimal. It is a common safe step size. It gives

$$
q=1-\frac1\kappa.
$$

Both can lead to convergence to $\theta^*$ when $\mu>0$, because both satisfy the convergence condition

$$
0<\gamma<\frac2L.
$$

But they do not converge at the same speed. Usually,

$$
\frac{\kappa-1}{\kappa+1}<1-\frac1\kappa.
$$

So $\gamma=2/(L+\mu)$ is faster in the worst-case parameter-error sense.

## 14. The bound with $\gamma=1/L$

The lecture then uses the safe step size

$$
\gamma=\frac1L.
$$

For this choice,

$$
q(\gamma)=1-\frac1\kappa.
$$

Therefore,

$$
\boxed{
\|\theta_t-\theta^*\|^2
\le
\left(1-\frac1\kappa\right)^{2t}
\|\theta_0-\theta^*\|^2.
}
$$

Using

$$
(1-a)^m\le e^{-am},
$$

we get

$$
\left(1-\frac1\kappa\right)^{2t}
\le
 e^{-2t/\kappa}.
$$

So

$$
\boxed{
\|\theta_t-\theta^*\|^2
\le
e^{-2t/\kappa}
\|\theta_0-\theta^*\|^2.
}
$$

This shows exponential convergence when $\mu>0$.

To make the relative squared error less than $\varepsilon$, we want

$$
e^{-2t/\kappa}\le\varepsilon.
$$

Taking logs gives

$$
-\frac{2t}{\kappa}\le \log\varepsilon.
$$

So

$$
t\ge \frac{\kappa}{2}\log\frac1\varepsilon.
$$

Thus the iteration complexity is

$$
\boxed{
t=O\left(\kappa\log\frac1\varepsilon\right).
}
$$

## 15. What if $\mu=0$?

If $\mu=0$, then

$$
\kappa=\frac{L}{\mu}=+\infty.
$$

Then the parameter-error bound becomes useless.

This can happen when $d>n$, because then

$$
H=\frac1n\Phi^\top\Phi
$$

is singular.

So the lecture switches to studying function values:

$$
F(\theta_t)-F(\theta^*).
$$

For least squares,

$$
F(\theta)-F(\theta^*)
=
\frac12(\theta-\theta^*)^\top H(\theta-\theta^*).
$$

Using

$$
\theta_t-\theta^*
=
(I-\gamma H)^t(\theta_0-\theta^*),
$$

we get

$$
F(\theta_t)-F(\theta^*)
=
\frac12(\theta_0-\theta^*)^\top
(I-\gamma H)^{2t}
H
(\theta_0-\theta^*).
$$

So instead of bounding eigenvalues of

$$
(I-\gamma H)^{2t},
$$

we bound eigenvalues of

$$
(I-\gamma H)^{2t}H.
$$

This gives a slower but still meaningful bound:

$$
\boxed{
F(\theta_t)-F(\theta^*)
\le
\frac{1}{8t\gamma}
\|\theta_0-\theta^*\|^2.
}
$$

So even when $\mu=0$, gradient descent can still decrease the function value at rate

$$
O\left(\frac1t\right).
$$

## 16. Big-picture summary

The section is about understanding why gradient descent converges quickly or slowly on least squares.

The central equation is

$$
\theta_t-\theta^*
=
(I-\gamma H)^t(\theta_0-\theta^*).
$$

This shows that convergence is controlled by the eigenvalues of

$$
I-\gamma H.
$$

The worst shrinkage factor is

$$
q(\gamma)=\max_{\lambda\in[\mu,L]}|1-\gamma\lambda|.
$$

Choosing $\gamma$ means choosing how fast the error shrinks.

The optimal fixed step size is

$$
\boxed{
\gamma^*=\frac{2}{L+\mu}.
}
$$

It gives contraction

$$
\boxed{
q^*=\frac{\kappa-1}{\kappa+1}
=1-\frac{2}{\kappa+1}.
}
$$

The common safe step size is

$$
\boxed{
\gamma=\frac1L.
}
$$

It gives contraction

$$
\boxed{
q=1-\frac1\kappa.
}
$$

Both lead to convergence when $\mu>0$, but only

$$
\gamma=\frac{2}{L+\mu}
$$

is optimal for the worst-case parameter-error contraction.

When $\mu=0$, the condition number is infinite, so parameter-error convergence may fail as a useful measure. Then we study the function gap instead, obtaining an $O(1/t)$ convergence bound.
