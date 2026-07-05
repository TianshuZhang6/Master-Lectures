
# L4 Lecture Notes: Linear Regression, OLS, and Ridge Regression

These notes summarize and expand the lecture slides **L4: Linear regression, OLS, ridge regression**.  The formulas are written in VS Code Jupyter-friendly Markdown style: display equations use `$ ... $`, and theorem/proof blocks are written as ordinary Markdown headings.

## 0. Notation

We observe training data

$$
S = \{(x_1,y_1),\ldots,(x_N,y_N)\}, \qquad x_i \in \mathbb{R}^d,\quad y_i\in\mathbb{R}.
$$

For multivariate linear regression, define the design matrix

$$
X = \begin{bmatrix}
- & x_1^\top & -\\
- & x_2^\top & -\\
& \vdots & \\
- & x_N^\top & -
\end{bmatrix}\in\mathbb{R}^{N\times d},
\qquad
 y = \begin{bmatrix}y_1\\ \vdots\\ y_N\end{bmatrix}\in\mathbb{R}^N,
\qquad
 w\in\mathbb{R}^d.
$$

A linear predictor without an explicit intercept is

$$
\widehat{y}_i = w^\top x_i.
$$

If an intercept is needed, augment the features by

$$
\tilde{x}_i = \begin{bmatrix}1\\x_i\end{bmatrix},
\qquad
\tilde{w}=\begin{bmatrix}w_0\\w\end{bmatrix},
\qquad
\widehat{y}_i = \tilde{w}^\top \tilde{x}_i = w_0+w^\top x_i.
$$

---

## 1. Classification versus regression

In classification, the target is categorical, for example

$$
y\in\{0,1\}.
$$

In regression, the target is numerical:

$$
y\in\mathbb{R}.
$$

The goal in regression is to learn a rule

$$
h:\mathbb{R}^d\to\mathbb{R}
$$

that predicts a real-valued response from the feature vector.

---

## 2. Empirical risk minimization for regression

The empirical risk minimization (ERM) framework chooses a hypothesis from a class \(\mathcal{H}\) by minimizing the average loss on the training data.

For least-squares regression, the loss is the squared error:

$$
\min_{h\in\mathcal{H}} \frac{1}{N}\sum_{i=1}^N \big(h(x_i)-y_i\big)^2.
$$

For linear least-squares regression,

$$
h(x)=w_0+w^\top x.
$$

Then the ERM problem becomes

$$
\min_{w_0,w}\frac{1}{N}\sum_{i=1}^N \big(w_0+w^\top x_i-y_i\big)^2.
$$

### 2.1 Why squared loss?

Squared loss is popular because:

1. It is differentiable and convex in the prediction.
2. It leads to closed-form normal equations for linear models.
3. It heavily penalizes large errors.
4. Under Gaussian observation noise, least squares coincides with maximum likelihood estimation.
5. At the population level, the best squared-loss predictor is the conditional mean \(\mathbb{E}[Y\mid X=x]\).

### 2.2 Other possible losses

The lecture also mentions other choices.

**Absolute loss:**

$$
\min_{h\in\mathcal{H}}\frac{1}{N}\sum_{i=1}^N |h(x_i)-y_i|.
$$

This is more robust to outliers than squared loss, but is not differentiable at zero. At the population level, it estimates a conditional median rather than a conditional mean.

**Asymmetric / pinball-style loss:**

For \(\tau\in(0,1)\), define

$$
\rho_\tau(u)=u\big(\tau-\mathbf{1}\{u<0\}\big).
$$

Then one can solve

$$
\min_{h\in\mathcal{H}}\frac{1}{N}\sum_{i=1}^N \rho_\tau\big(y_i-h(x_i)\big).
$$

This is used in quantile regression. It penalizes underprediction and overprediction differently.

---

## 3. Simple linear regression

Consider the one-dimensional model

$$
y = w_0 + wx.
$$

The least-squares objective is

$$
L(w_0,w)=\sum_{i=1}^N (y_i-w_0-wx_i)^2.
$$

### 3.1 Normal equations for simple linear regression

Take partial derivatives:

$$
\frac{\partial L}{\partial w_0}
= -2\sum_{i=1}^N (y_i-w_0-wx_i),
$$

and

$$
\frac{\partial L}{\partial w}
= -2\sum_{i=1}^N x_i(y_i-w_0-wx_i).
$$

At the minimizer, both derivatives are zero.

From \(\partial L/\partial w_0=0\),

$$
\sum_{i=1}^N y_i - Nw_0 - w\sum_{i=1}^N x_i = 0.
$$

Therefore

$$
w_0 = \bar{y}-w\bar{x},
$$

where

$$
\bar{x}=\frac{1}{N}\sum_{i=1}^N x_i,
\qquad
\bar{y}=\frac{1}{N}\sum_{i=1}^N y_i.
$$

From \(\partial L/\partial w=0\),

$$
\sum_{i=1}^N x_i y_i - w_0\sum_{i=1}^N x_i - w\sum_{i=1}^N x_i^2 = 0.
$$

Substitute \(w_0=\bar{y}-w\bar{x}\):

$$
\sum_{i=1}^N x_i y_i - (\bar{y}-w\bar{x})\sum_{i=1}^N x_i - w\sum_{i=1}^N x_i^2 = 0.
$$

Divide by \(N\). Let

$$
\overline{x^2}=\frac{1}{N}\sum_{i=1}^N x_i^2,
\qquad
\overline{xy}=\frac{1}{N}\sum_{i=1}^N x_i y_i.
$$

Then

$$
\overline{xy}-\bar{x}\bar{y}+w\bar{x}^2-w\overline{x^2}=0.
$$

Thus

$$
w(\overline{x^2}-\bar{x}^2)=\overline{xy}-\bar{x}\bar{y},
$$

so

$$
w = \frac{\overline{xy}-\bar{x}\bar{y}}{\overline{x^2}-\bar{x}^2}.
$$

This is the empirical version of

$$
w=\frac{\operatorname{cov}(X,Y)}{\operatorname{var}(X)}.
$$

Since the Pearson correlation is

$$
r(X,Y)=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y},
$$

we can also write

$$
w = r(X,Y)\frac{\sigma_Y}{\sigma_X}.
$$

The intercept is

$$
w_0=\mathbb{E}[Y]-w\mathbb{E}[X].
$$

:::{admonition} Key interpretation
The fitted line always passes through \((\bar{x},\bar{y})\). The slope is correlation times the ratio of standard deviations.
:::

### 3.2 Pearson father-son height example

The lecture uses approximate values:

$$
\bar{x}\approx 68,\qquad \sigma_X\approx 2.7,
$$

for fathers' heights, and

$$
\bar{y}\approx 69,\qquad \sigma_Y\approx 2.7,
$$

for sons' heights, with

$$
r\approx 0.5.
$$

Thus

$$
w = r\frac{\sigma_Y}{\sigma_X}\approx 0.5\cdot\frac{2.7}{2.7}=0.5.
$$

The intercept is

$$
w_0 = \bar{y}-w\bar{x}\approx 69-0.5\cdot 68 = 35.
$$

So the regression line is approximately

$$
\widehat{y}=35+0.5x.
$$

If a father's height is \(x=64\), then

$$
\widehat{y}=35+0.5\cdot 64=67.
$$

Equivalently,

$$
\widehat{y}-\bar{y}=r\frac{\sigma_Y}{\sigma_X}(x-\bar{x})
=0.5(64-68)=-2,
$$

so

$$
\widehat{y}=69-2=67.
$$

The answer is not \(65\). This is the classical **regression toward the mean** effect: because \(|r|<1\), being below average in \(x\) predicts being below average in \(y\), but by fewer standard deviations.

### 3.3 RMSE formula for simple regression

The lecture asks to show that the root mean square error is

$$
\operatorname{RMSE}=\sigma_Y\sqrt{1-r^2}.
$$

:::{admonition} Proof
Assume variables have been centered, so \(\mathbb{E}[X]=\mathbb{E}[Y]=0\). Then the best linear predictor without intercept is

$$
\widehat{Y}=wX,
\qquad
w=\frac{\operatorname{cov}(X,Y)}{\operatorname{var}(X)}
=r\frac{\sigma_Y}{\sigma_X}.
$$

The mean squared error is

$$
\mathbb{E}\big[(Y-wX)^2\big]
=\mathbb{E}[Y^2]-2w\mathbb{E}[XY]+w^2\mathbb{E}[X^2].
$$

Now use

$$
\mathbb{E}[Y^2]=\sigma_Y^2,
\qquad
\mathbb{E}[X^2]=\sigma_X^2,
\qquad
\mathbb{E}[XY]=r\sigma_X\sigma_Y.
$$

Therefore

$$
\begin{aligned}
\mathbb{E}\big[(Y-wX)^2\big]
&=\sigma_Y^2-2\left(r\frac{\sigma_Y}{\sigma_X}\right)(r\sigma_X\sigma_Y)
+\left(r\frac{\sigma_Y}{\sigma_X}\right)^2\sigma_X^2\\
&=\sigma_Y^2-2r^2\sigma_Y^2+r^2\sigma_Y^2\\
&=\sigma_Y^2(1-r^2).
\end{aligned}
$$

Taking the square root gives

$$
\operatorname{RMSE}=\sigma_Y\sqrt{1-r^2}.
$$
:::

---

## 4. Warning: correlation and summary statistics

A high linear correlation does not imply causation. Spurious correlations can occur when two unrelated quantities share a trend or are measured over a small range.

The slides also emphasize that identical summary statistics can hide very different data geometry. For example, Anscombe's quartet and related datasets can have similar means, variances, correlations, and regression lines, yet very different scatter plots. Always visualize the data when possible.

---

## 5. Population regression: the best squared-loss estimator

The population least-squares problem is

$$
\min_h \; \mathbb{E}_{(X,Y)}\big[(h(X)-Y)^2\big].
$$

If we impose no parametric restriction on \(h\), the optimal predictor is the conditional expectation

$$
\eta(x)=\mathbb{E}[Y\mid X=x].
$$

:::{admonition} Theorem: conditional mean is the best squared-loss predictor
Let

$$
\eta(x)=\mathbb{E}[Y\mid X=x].
$$

Then, for any measurable predictor \(h\),

$$
\mathbb{E}\big[(\eta(X)-Y)^2\big]
\le
\mathbb{E}\big[(h(X)-Y)^2\big].
$$
:::

:::{admonition} Proof
Condition on \(X=x\). We compare the conditional risks:

$$
\mathbb{E}\big[(h(X)-Y)^2\mid X=x\big].
$$

Add and subtract \(\eta(X)\):

$$
\begin{aligned}
\mathbb{E}\big[(h(X)-Y)^2\mid X=x\big]
&=\mathbb{E}\big[(h(X)-\eta(X)+\eta(X)-Y)^2\mid X=x\big]\\
&=(h(x)-\eta(x))^2
+2(h(x)-\eta(x))\mathbb{E}[\eta(X)-Y\mid X=x]\\
&\quad+\mathbb{E}\big[(\eta(X)-Y)^2\mid X=x\big].
\end{aligned}
$$

The middle term vanishes because

$$
\mathbb{E}[\eta(X)-Y\mid X=x]
=\eta(x)-\mathbb{E}[Y\mid X=x]
=0.
$$

Therefore

$$
\mathbb{E}\big[(h(X)-Y)^2\mid X=x\big]
=
(h(x)-\eta(x))^2
+
\mathbb{E}\big[(\eta(X)-Y)^2\mid X=x\big].
$$

Since \((h(x)-\eta(x))^2\ge 0\),

$$
\mathbb{E}\big[(h(X)-Y)^2\mid X=x\big]
\ge
\mathbb{E}\big[(\eta(X)-Y)^2\mid X=x\big].
$$

Integrating over \(X\) gives

$$
\mathbb{E}\big[(h(X)-Y)^2\big]
\ge
\mathbb{E}\big[(\eta(X)-Y)^2\big].
$$
:::

:::{admonition} Interpretation
Squared-loss regression estimates a conditional mean. Since the distribution \(\mathbb{P}(X,Y)\) is unknown, we cannot directly compute \(\mathbb{E}[Y\mid X=x]\). The practical solution is to assume a function class, such as linear functions or nonlinear features, and learn from data.
:::

---

## 6. Discriminative regression formulation

The discriminative idea is to model \(\mathbb{E}[Y\mid X=x]\) directly.

A linear model assumes

$$
\mathbb{E}[Y\mid X=x]\approx w^\top x+w_0.
$$

A nonlinear feature model assumes

$$
\mathbb{E}[Y\mid X=x]\approx w^\top \phi(x)+w_0,
$$

where

$$
\phi:\mathbb{R}^d\to\mathbb{R}^m
$$

is a feature map.

At the population level, the aim is to minimize

$$
\mathbb{E}\big[\ell(w^\top X+w_0,Y)\big].
$$

ERM replaces this expectation by the sample average:

$$
\frac{1}{N}\sum_{i=1}^N \ell(w^\top x_i+w_0,y_i).
$$

For least squares,

$$
\ell(\widehat{y},y)=\frac{1}{2}(\widehat{y}-y)^2.
$$

---

## 7. Multivariate ordinary least squares

The multivariate linear least-squares problem is

$$
\min_{w\in\mathbb{R}^d} L(w),
\qquad
L(w)=\sum_{i=1}^N (y_i-w^\top x_i)^2
=\|Xw-y\|_2^2.
$$

Expanding the objective:

$$
\begin{aligned}
L(w)
&=(Xw-y)^\top(Xw-y)\\
&=w^\top X^\top Xw-2w^\top X^\top y+y^\top y.
\end{aligned}
$$

The gradient is

$$
\nabla L(w)=2X^\top Xw-2X^\top y.
$$

Setting the gradient to zero gives the normal equations:

$$
X^\top Xw=X^\top y.
$$

If \(X^\top X\) is invertible, then

$$
\widehat{w}_{\mathrm{LS}}=(X^\top X)^{-1}X^\top y.
$$

:::{admonition} Geometric meaning
The fitted vector \(X\widehat{w}\) is the orthogonal projection of \(y\) onto the column space of \(X\). The residual

$$
r=y-X\widehat{w}
$$

is orthogonal to every column of \(X\), because the normal equations imply

$$
X^\top r=0.
$$
:::

### 7.1 Nonlinear features

If we replace \(x_i\) by \(\phi(x_i)\), define

$$
\Phi = \begin{bmatrix}
- & \phi(x_1)^\top & -\\
& \vdots &\\
- & \phi(x_N)^\top & -
\end{bmatrix}.
$$

The feature-space least-squares solution is

$$
\widehat{w}=(\Phi^\top \Phi)^{-1}\Phi^\top y,
$$

assuming \(\Phi^\top \Phi\) is invertible.

---

## 8. Observation model and prediction error

Assume the true model is linear and observations are noisy:

$$
y_i=w^\top x_i+\eta_i,
\qquad
\eta_i\sim\mathcal{N}(0,\sigma^2).
$$

In vector form,

$$
y=Xw+\eta,
\qquad
\eta\sim\mathcal{N}(0,\sigma^2 I_N).
$$

OLS estimates

$$
\widehat{w}=(X^\top X)^{-1}X^\top y.
$$

Substitute \(y=Xw+\eta\):

$$
\begin{aligned}
\widehat{w}
&=(X^\top X)^{-1}X^\top(Xw+\eta)\\
&=w+(X^\top X)^{-1}X^\top\eta.
\end{aligned}
$$

Therefore

$$
\widehat{w}-w=(X^\top X)^{-1}X^\top\eta.
$$

The prediction error on the training design points is

$$
\frac{1}{N}\|X\widehat{w}-Xw\|_2^2.
$$

Since

$$
X\widehat{w}-Xw = X(X^\top X)^{-1}X^\top\eta,
$$

define the projection matrix

$$
P_X = X(X^\top X)^{-1}X^\top.
$$

Then

$$
X\widehat{w}-Xw=P_X\eta.
$$

### 8.1 Expected prediction error

Because \(P_X\) is symmetric and idempotent when \(X\) has full column rank,

$$
P_X^\top=P_X,
\qquad
P_X^2=P_X.
$$

Thus

$$
\begin{aligned}
\mathbb{E}\big[\|X\widehat{w}-Xw\|_2^2\big]
&=\mathbb{E}\big[\|P_X\eta\|_2^2\big]\\
&=\mathbb{E}[\eta^\top P_X^\top P_X\eta]\\
&=\mathbb{E}[\eta^\top P_X\eta]\\
&=\mathbb{E}[\operatorname{trace}(P_X\eta\eta^\top)]\\
&=\operatorname{trace}(P_X\mathbb{E}[\eta\eta^\top])\\
&=\operatorname{trace}(P_X\sigma^2 I)\\
&=\sigma^2\operatorname{trace}(P_X).
\end{aligned}
$$

If \(X\in\mathbb{R}^{N\times d}\) has full column rank \(d\), then

$$
\operatorname{trace}(P_X)=d.
$$

Therefore

$$
\frac{1}{N}\mathbb{E}\big[\|X\widehat{w}-Xw\|_2^2\big]
=\frac{\sigma^2 d}{N}.
$$

This tends to zero as \(N\to\infty\) when \(d\) is fixed.

:::{admonition} Relation to the lecture proof sketch
The slides derive a looser but useful bound of the form

$$
\frac{1}{N}\mathbb{E}\big[\|X\widehat{w}-Xw\|_2^2\big]
\le
\frac{2\sigma^2}{N}\operatorname{trace}\big(X(X^\top X)^{-1}X^\top\big).
$$

For full column rank \(X\), this is

$$
\frac{1}{N}\mathbb{E}\big[\|X\widehat{w}-Xw\|_2^2\big]
\le
\frac{2\sigma^2 d}{N}.
$$

The exact calculation above removes the factor \(2\), but the asymptotic message is the same: with fixed dimension and increasing data, the expected prediction error goes to zero.
:::

### 8.2 Pseudoinverse case

If \(X^\top X\) is not invertible, use the Moore-Penrose pseudoinverse \(X^+\). The minimum-norm least-squares solution is

$$
\widehat{w}=X^+y.
$$

The fitted values are

$$
X\widehat{w}=XX^+y.
$$

The matrix

$$
P_X=XX^+
$$

is the orthogonal projector onto the column space of \(X\). Therefore

$$
X\widehat{w}-Xw=P_X\eta.
$$

The same computation gives

$$
\frac{1}{N}\mathbb{E}\big[\|X\widehat{w}-Xw\|_2^2\big]
=\frac{\sigma^2}{N}\operatorname{rank}(X).
$$

Since

$$
\operatorname{rank}(X)\le \min\{N,d\},
$$

we have

$$
\frac{1}{N}\mathbb{E}\big[\|X\widehat{w}-Xw\|_2^2\big]
\le
\frac{\sigma^2\min\{N,d\}}{N}.
$$

---

## 9. Regularization and ridge regression

OLS can become problematic when:

1. \(d>N\), so \(X^\top X\) is not invertible.
2. \(X^\top X\) is nearly singular, so the problem is ill-conditioned.
3. The model has enough flexibility to fit noise, producing overfitting.

Ridge regression modifies the least-squares objective by penalizing large weights.

Using the convention

$$
\min_w \frac{1}{2}\|Xw-y\|_2^2+\frac{\lambda}{2}\|w\|_2^2,
\qquad \lambda>0,
$$

the gradient is

$$
\nabla L(w)=X^\top(Xw-y)+\lambda w.
$$

Setting it to zero:

$$
X^\top Xw-X^\top y+\lambda w=0.
$$

Thus

$$
(X^\top X+\lambda I)w=X^\top y,
$$

and the ridge solution is

$$
\widehat{w}_{\lambda}=(X^\top X+\lambda I)^{-1}X^\top y.
$$

:::{admonition} Scaling conventions
Some slides write the objective as

$$
\frac{1}{N}\|Xw-y\|_2^2+\lambda\|w\|_2^2.
$$

For that convention,

$$
\widehat{w}_{\lambda}=(X^\top X+N\lambda I)^{-1}X^\top y.
$$

The difference is only a rescaling of \(\lambda\).
:::

### 9.1 Why ridge makes the problem invertible

The matrix \(X^\top X\) is positive semidefinite:

$$
z^\top X^\top Xz=\|Xz\|_2^2\ge 0.
$$

For \(\lambda>0\),

$$
z^\top(X^\top X+\lambda I)z
=\|Xz\|_2^2+\lambda\|z\|_2^2>0
$$

for every nonzero \(z\). Hence \(X^\top X+\lambda I\) is positive definite and invertible.

### 9.2 Ridge as shrinkage in singular-value directions

Let the compact singular value decomposition be

$$
X=U\Sigma V^\top,
$$

where

$$
\Sigma=\operatorname{diag}(s_1,\ldots,s_r),
\qquad
s_j>0.
$$

Then

$$
X^\top X=V\Sigma^2V^\top.
$$

The ridge estimator is

$$
\widehat{w}_\lambda
=V(\Sigma^2+\lambda I)^{-1}\Sigma U^\top y.
$$

Thus each singular direction is weighted by

$$
\frac{s_j}{s_j^2+\lambda}.
$$

For predictions,

$$
X\widehat{w}_\lambda
=U\Sigma(\Sigma^2+\lambda I)^{-1}\Sigma U^\top y
=U\operatorname{diag}\left(\frac{s_j^2}{s_j^2+\lambda}\right)U^\top y.
$$

The factor

$$
\frac{s_j^2}{s_j^2+\lambda}\in(0,1)
$$

shrinks unstable directions. Directions with small singular values are shrunk most strongly.

---

## 10. Bias-variance decomposition

Assume a noisy observation model

$$
y=f(x)+\eta,
\qquad
\mathbb{E}[\eta]=0.
$$

A learning algorithm trained on data produces a random predictor \(\widehat{f}\). We want to understand

$$
\mathbb{E}\big[\|y-\widehat{f}\|^2\big].
$$

First write

$$
y-\widehat{f}=y-f+f-\widehat{f}=\eta+f-\widehat{f}.
$$

Then

$$
\begin{aligned}
\mathbb{E}\big[\|y-\widehat{f}\|^2\big]
&=\mathbb{E}\big[\|\eta+f-\widehat{f}\|^2\big]\\
&=\mathbb{E}[\|\eta\|^2]
+\mathbb{E}[\|f-\widehat{f}\|^2]
+2\mathbb{E}[\langle \eta,f-\widehat{f}\rangle].
\end{aligned}
$$

Under the usual independence / zero-mean noise assumptions, the cross term is zero, so

$$
\mathbb{E}\big[\|y-\widehat{f}\|^2\big]
=
\mathbb{E}[\|\eta\|^2]
+\mathbb{E}[\|f-\widehat{f}\|^2].
$$

Now add and subtract \(\mathbb{E}[\widehat{f}]\):

$$
f-\widehat{f}=f-\mathbb{E}[\widehat{f}]+\mathbb{E}[\widehat{f}]-\widehat{f}.
$$

Therefore

$$
\begin{aligned}
\mathbb{E}[\|f-\widehat{f}\|^2]
&=\mathbb{E}\big[\|f-\mathbb{E}[\widehat{f}]
+\mathbb{E}[\widehat{f}]-\widehat{f}\|^2\big]\\
&=\|f-\mathbb{E}[\widehat{f}]\|^2
+\mathbb{E}[\|\widehat{f}-\mathbb{E}[\widehat{f}]\|^2],
\end{aligned}
$$

because the cross term has expectation zero.

Thus

$$
\boxed{
\mathbb{E}\big[\|y-\widehat{f}\|^2\big]
=
\underbrace{\mathbb{E}[\|\eta\|^2]}_{\text{noise}}
+
\underbrace{\|f-\mathbb{E}[\widehat{f}]\|^2}_{\text{bias}^2}
+
\underbrace{\mathbb{E}[\|\widehat{f}-\mathbb{E}[\widehat{f}]\|^2]}_{\text{variance}}.
}
$$

### 10.1 How regularization uses bias-variance tradeoff

OLS is unbiased under the standard linear model, and the Gauss-Markov theorem says that it is the best linear unbiased estimator. But the lowest-variance unbiased estimator may still have larger test error than a biased estimator.

Ridge regression deliberately introduces bias by shrinking \(w\), but can reduce variance enough to lower total prediction error.

---

## 11. OLS unbiasedness and ridge bias

Assume

$$
y=Xw_{\mathrm{true}}+\eta,
\qquad
\mathbb{E}[\eta]=0,
\qquad
\operatorname{Var}(\eta)=\sigma^2 I.
$$

### 11.1 OLS is unbiased

OLS gives

$$
\widehat{w}_{\mathrm{LS}}
=(X^\top X)^{-1}X^\top y.
$$

Substitute the observation model:

$$
\begin{aligned}
\widehat{w}_{\mathrm{LS}}
&=(X^\top X)^{-1}X^\top(Xw_{\mathrm{true}}+\eta)\\
&=w_{\mathrm{true}}+(X^\top X)^{-1}X^\top\eta.
\end{aligned}
$$

Taking expectation over the noise,

$$
\mathbb{E}[\widehat{w}_{\mathrm{LS}}]
=w_{\mathrm{true}}+(X^\top X)^{-1}X^\top\mathbb{E}[\eta]
=w_{\mathrm{true}}.
$$

Hence OLS is unbiased.

### 11.2 Variance of OLS

Using \(\operatorname{Var}(A\eta)=A\operatorname{Var}(\eta)A^\top\),

$$
\begin{aligned}
\operatorname{Var}(\widehat{w}_{\mathrm{LS}})
&=\operatorname{Var}\big((X^\top X)^{-1}X^\top\eta\big)\\
&=(X^\top X)^{-1}X^\top(\sigma^2 I)X(X^\top X)^{-1}\\
&=\sigma^2(X^\top X)^{-1}.
\end{aligned}
$$

This becomes large when \(X^\top X\) has small eigenvalues.

### 11.3 Ridge is biased

Use the convention

$$
\widehat{w}_\lambda
=(X^\top X+N\lambda I)^{-1}X^\top y.
$$

Let

$$
M=X^\top X.
$$

Then

$$
\widehat{w}_\lambda=(M+N\lambda I)^{-1}X^\top y.
$$

Substitute \(y=Xw_{\mathrm{true}}+\eta\):

$$
\begin{aligned}
\widehat{w}_\lambda
&=(M+N\lambda I)^{-1}X^\top(Xw_{\mathrm{true}}+\eta)\\
&=(M+N\lambda I)^{-1}Mw_{\mathrm{true}}
+(M+N\lambda I)^{-1}X^\top\eta.
\end{aligned}
$$

Taking expectation,

$$
\mathbb{E}[\widehat{w}_\lambda]
=(M+N\lambda I)^{-1}Mw_{\mathrm{true}}.
$$

Unless \(\lambda=0\) or \(w_{\mathrm{true}}\) lies in a special subspace, this differs from \(w_{\mathrm{true}}\). Hence ridge is biased.

The bias is

$$
\operatorname{Bias}(\widehat{w}_\lambda)
=\mathbb{E}[\widehat{w}_\lambda]-w_{\mathrm{true}}
=
\left[(M+N\lambda I)^{-1}M-I\right]w_{\mathrm{true}}.
$$

Since

$$
(M+N\lambda I)^{-1}M-I
=-(M+N\lambda I)^{-1}N\lambda I,
$$

we can write

$$
\operatorname{Bias}(\widehat{w}_\lambda)
=-N\lambda(M+N\lambda I)^{-1}w_{\mathrm{true}}.
$$

### 11.4 Variance of ridge

The random part of ridge is

$$
(M+N\lambda I)^{-1}X^\top\eta.
$$

Therefore

$$
\begin{aligned}
\operatorname{Var}(\widehat{w}_\lambda)
&=(M+N\lambda I)^{-1}X^\top(\sigma^2 I)X(M+N\lambda I)^{-1}\\
&=\sigma^2(M+N\lambda I)^{-1}M(M+N\lambda I)^{-1}.
\end{aligned}
$$

If \(M=V\operatorname{diag}(\mu_1,\ldots,
\mu_d)V^\top\), then OLS has coordinate variances

$$
\frac{\sigma^2}{\mu_j},
$$

whereas ridge has coordinate variances

$$
\sigma^2\frac{\mu_j}{(\mu_j+N\lambda)^2}.
$$

For small \(\mu_j\), ridge dramatically reduces variance.

---

## 12. General regularization

The general regularized ERM form is

$$
\min_w \frac{1}{N}\sum_{i=1}^N (y_i-w^\top x_i)^2 + \lambda \Omega(w).
$$

Here:

- the first term is the empirical loss;
- \(\Omega(w)\) is a regularizer;
- \(\lambda\ge 0\) controls the strength of regularization.

A common family is

$$
\min_w \frac{1}{N}\sum_{i=1}^N (y_i-w^\top x_i)^2 + \lambda\|w\|_p^p.
$$

Important cases:

$$
p=2 \quad \Rightarrow \quad \text{ridge regression},
$$

and

$$
p=1 \quad \Rightarrow \quad \text{LASSO}.
$$

Other regularizers include nuclear norms, atomic norms, group norms, and many problem-specific penalties.

:::{admonition} Main message
Regularization helps because it reduces effective model capacity and encodes a preferred structure. This can increase bias, but it can reduce variance and improve test error.
:::

---

# Additional solved exercises

## Exercise A: derivative of \(A(t)=\det(C+tD)\)

The lecture begins with the puzzle

$$
A(t)=\det(C+tD),
\qquad
\frac{d}{dt}A(t)=?
$$

Let

$$
M(t)=C+tD.
$$

If \(M(t)\) is invertible, Jacobi's formula gives

$$
\frac{d}{dt}\det M(t)
=\det M(t)\operatorname{trace}\left(M(t)^{-1}M'(t)\right).
$$

Since \(M'(t)=D\),

$$
\boxed{
A'(t)=\det(C+tD)\operatorname{trace}\left((C+tD)^{-1}D\right).
}
$$

More generally, even when \(M(t)\) is singular, one can write

$$
A'(t)=\operatorname{trace}\left(\operatorname{adj}(C+tD)D\right),
$$

where \(\operatorname{adj}(M)\) is the adjugate matrix.

:::{admonition} Proof of Jacobi's formula
For invertible \(M\), use

$$
\det(M+\Delta)
=\det(M)\det(I+M^{-1}\Delta).
$$

For small \(\Delta\),

$$
\det(I+E)=1+\operatorname{trace}(E)+o(\|E\|).
$$

Set \(\Delta=hM'(t)\). Then

$$
\begin{aligned}
\det(M(t+h))-
\det(M(t))
&=\det(M(t))
\left[\det(I+hM(t)^{-1}M'(t)+o(h))-1\right]\\
&=h\det(M(t))\operatorname{trace}(M(t)^{-1}M'(t))+o(h).
\end{aligned}
$$

Divide by \(h\) and let \(h\to 0\).
:::

## Exercise B: OLS solution with nonlinear features

Suppose the model is

$$
h(x)=w^\top \phi(x).
$$

Let \(\Phi\in\mathbb{R}^{N\times m}\) have rows \(\phi(x_i)^\top\). The objective is

$$
L(w)=\|\Phi w-y\|_2^2.
$$

Expanding,

$$
L(w)=w^\top\Phi^\top\Phi w-2w^\top\Phi^\top y+y^\top y.
$$

So

$$
\nabla L(w)=2\Phi^\top\Phi w-2\Phi^\top y.
$$

Setting the gradient to zero gives

$$
\Phi^\top\Phi w=\Phi^\top y.
$$

If \(\Phi^\top\Phi\) is invertible,

$$
\boxed{
\widehat{w}=(\Phi^\top\Phi)^{-1}\Phi^\top y.
}
$$

## Exercise C: prove ridge variance from the slide hint

The slides suggest writing ridge as a transformed OLS estimator. Let

$$
M=X^\top X.
$$

For the ridge convention

$$
\widehat{w}_\lambda=(M+N\lambda I)^{-1}X^\top y,
$$

and OLS

$$
\widehat{w}_{\mathrm{LS}}=M^{-1}X^\top y,
$$

we can write

$$
\widehat{w}_\lambda
=(M+N\lambda I)^{-1}MM^{-1}X^\top y
=(M+N\lambda I)^{-1}M\widehat{w}_{\mathrm{LS}}.
$$

So let

$$
A=(M+N\lambda I)^{-1}M.
$$

Then

$$
\widehat{w}_\lambda=A\widehat{w}_{\mathrm{LS}}.
$$

Using

$$
\operatorname{Var}(A Z)=A\operatorname{Var}(Z)A^\top
$$

and

$$
\operatorname{Var}(\widehat{w}_{\mathrm{LS}})=\sigma^2M^{-1},
$$

we obtain

$$
\begin{aligned}
\operatorname{Var}(\widehat{w}_\lambda)
&=A\operatorname{Var}(\widehat{w}_{\mathrm{LS}})A^\top\\
&=(M+N\lambda I)^{-1}M(\sigma^2M^{-1})M(M+N\lambda I)^{-1}\\
&=\sigma^2(M+N\lambda I)^{-1}M(M+N\lambda I)^{-1}.
\end{aligned}
$$

This matches the direct derivation.

---

# Study checklist

After studying this lecture, you should be able to:

1. State the ERM formulation for regression.
2. Explain why squared loss leads to conditional means.
3. Derive the simple linear regression slope and intercept.
4. Explain the relationship between regression slope and Pearson correlation.
5. Compute the father-son height prediction and explain regression to the mean.
6. Prove the RMSE formula \(\sigma_Y\sqrt{1-r^2}\).
7. Prove that \(\mathbb{E}[Y\mid X=x]\) is the optimal squared-loss predictor.
8. Derive the multivariate OLS solution.
9. Explain the geometric projection view of OLS.
10. Analyze OLS under the noisy linear observation model.
11. Use trace identities to compute expected prediction error.
12. Explain why ridge makes the normal equations invertible.
13. Derive the ridge solution.
14. Prove OLS is unbiased and ridge is biased.
15. Derive the variance of OLS and ridge.
16. Explain the bias-variance tradeoff and how regularization affects it.
17. Distinguish ridge, LASSO, and general regularization.

