# L5 Lecture Notes: Regression Regularization Wrap-Up and Kernels I

These notes summarize the L5 slides on regression regularization, bias-variance tradeoff, nonlinear features, and the first introduction to kernel methods. The formatting is designed for VS Code Jupyter notebooks: inline math uses `$...$`, and display equations use `$$...$$`.

## 1. Recall: least-squares regression

Given training data

$$
S=\{(x_1,y_1),\ldots,(x_N,y_N)\},
\qquad
x_i\in\mathbb R^d,
\quad
 y_i\in\mathbb R,
$$

define the design matrix $X\in\mathbb R^{N\times d}$ whose $i$-th row is $x_i^\top$, and let $y\in\mathbb R^N$ be the vector of labels.

The least-squares objective is

$$
L(w)=\sum_{i=1}^N (y_i-w^\top x_i)^2
=\|Xw-y\|^2.
$$

The empirical risk minimization problem is

$$
\min_{w\in\mathbb R^d}\|Xw-y\|^2.
$$

If $X^\top X$ is invertible, the normal equations give

$$
X^\top Xw=X^\top y,
$$

and therefore

$$
\widehat w_{\mathrm{LS}}=(X^\top X)^{-1}X^\top y.
$$

A key question is what happens when $X^\top X$ is singular or nearly singular. This can occur when $d>N$, when features are redundant, or when some features are highly correlated.

## 2. Ridge regression: the regularization hack

A simple fix is to replace $X^\top X$ by

$$
X^\top X+\lambda I,
\qquad
\lambda>0.
$$

Since $X^\top X$ is positive semidefinite, adding $\lambda I$ makes the matrix positive definite and therefore invertible.

The ridge estimator is

$$
\widehat w_\lambda
=
(X^\top X+\lambda I)^{-1}X^\top y.
$$

Depending on convention, the regularized objective may also be written with $N\lambda$:

$$
\widehat w_\lambda
=
(X^\top X+N\lambda I)^{-1}X^\top y.
$$

The important idea is not the exact scaling of $\lambda$, but the penalty on large weights.

## 3. Ridge regression as regularized least squares

Ridge regression is the solution of the optimization problem

$$
\min_{w\in\mathbb R^d}
\left\{
\|Xw-y\|^2+\lambda\|w\|^2
\right\}.
$$

If using the averaged empirical risk convention, one writes

$$
\min_{w\in\mathbb R^d}
\left\{
\frac1N\sum_{i=1}^N (y_i-w^\top x_i)^2
+\lambda\|w\|^2
\right\}.
$$

The gradient of the unaveraged objective is

$$
\nabla_w\left(\|Xw-y\|^2+\lambda\|w\|^2\right)
=2X^\top(Xw-y)+2\lambda w.
$$

Setting this gradient to zero gives

$$
X^\top Xw-X^\top y+\lambda w=0,
$$

so

$$
(X^\top X+
\lambda I)w=X^\top y.
$$

Thus

$$
\widehat w_\lambda=(X^\top X+\lambda I)^{-1}X^\top y.
$$

In words, ridge regression biases the estimator toward small norm. This reduces instability caused by ill-conditioned design matrices.

## 4. Why regularization helps

Regularization helps in several connected ways.

First, it makes the optimization problem better conditioned. If $X^\top X$ has very small eigenvalues, then the least-squares solution can have very large coefficients. Adding $\lambda I$ shifts every eigenvalue upward:

$$
\mu_j(X^\top X+\lambda I)=\mu_j(X^\top X)+\lambda.
$$

Second, it reduces effective model complexity. Large coefficients often allow the model to fit noise in the training data. Penalizing $\|w\|$ discourages this behavior.

Third, regularization introduces bias deliberately in order to reduce variance. The resulting estimator may no longer be unbiased, but its prediction error can be lower.

The general regularized ERM form is

$$
\min_w
\left\{
\frac1N\sum_{i=1}^N \ell(w;x_i,y_i)+\Omega(w)
\right\},
$$

where $\Omega(w)$ is the regularizer. For ridge regression,

$$
\Omega(w)=\lambda\|w\|_2^2.
$$

## 5. Bias-variance decomposition

The slides revisit the noisy observation model

$$
y=f+\eta,
$$

where $f$ is the true signal and $\eta$ is noise. Suppose our learning algorithm produces a predictor $\widehat f$.

The expected squared prediction error can be decomposed as

$$
\mathbb E\left[\|y-\widehat f\|^2\right]
=
\mathbb E\left[\|f+\eta-\widehat f\|^2\right].
$$

Expanding gives

$$
\begin{aligned}
\mathbb E\left[\|y-\widehat f\|^2\right]
&=
\mathbb E\left[\|\eta\|^2\right]
+
\mathbb E\left[\|f-\widehat f\|^2\right]
+2\mathbb E\left[\langle \eta,f-\widehat f\rangle\right].
\end{aligned}
$$

Under the usual zero-mean independence assumptions, the cross term is zero, so

$$
\mathbb E\left[\|y-\widehat f\|^2\right]
=
\mathbb E\left[\|\eta\|^2\right]
+
\mathbb E\left[\|f-\widehat f\|^2\right].
$$

Now add and subtract $\mathbb E[\widehat f]$:

$$
f-\widehat f
=
\left(f-\mathbb E[\widehat f]\right)
+
\left(\mathbb E[\widehat f]-\widehat f\right).
$$

This yields the bias-variance decomposition

$$
\boxed{
\mathbb E\left[\|y-\widehat f\|^2\right]
=
\underbrace{\mathbb E\left[\|\eta\|^2\right]}_{\text{noise}}
+
\underbrace{\left\|f-\mathbb E[\widehat f]\right\|^2}_{\text{bias}^2}
+
\underbrace{\mathbb E\left[\left\|\widehat f-\mathbb E[\widehat f]\right\|^2\right]}_{\text{variance}}
}.
$$

Noise is irreducible. Bias and variance can be traded against each other by choosing the model class, the algorithm, and the amount of regularization.

## 6. Gauss-Markov and the bias-variance tradeoff

The Gauss-Markov theorem says that, under the classical linear model assumptions, the ordinary least-squares estimator is the best linear unbiased estimator: among all linear unbiased estimators, it has the smallest variance.

The phrase **unbiased** is crucial. If we allow biased estimators, we may be able to reduce variance enough to lower total prediction error.

Ridge regression does exactly this. It generally introduces bias, but it can substantially reduce variance. This is why ridge can outperform ordinary least squares on unseen data even though OLS is optimal within the restricted class of unbiased linear estimators.

The practical message is

$$
\text{small bias increase} + \text{large variance decrease}
\quad\Rightarrow\quad
\text{better test error}.
$$

## 7. Exercise: ridge regression is biased

Consider the linear observation model

$$
y=Xw_{\mathrm{true}}+\eta,
\qquad
\mathbb E[\eta]=0.
$$

Using the ridge convention

$$
\widehat w_\lambda=(X^\top X+\lambda I)^{-1}X^\top y,
$$

we get

$$
\begin{aligned}
\mathbb E[\widehat w_\lambda]
&=(X^\top X+\lambda I)^{-1}X^\top\mathbb E[y]\\
&=(X^\top X+\lambda I)^{-1}X^\top Xw_{\mathrm{true}}.
\end{aligned}
$$

Therefore the bias is

$$
\operatorname{Bias}(\widehat w_\lambda)
=
\mathbb E[\widehat w_\lambda]-w_{\mathrm{true}}
=
\left[(X^\top X+\lambda I)^{-1}X^\top X-I\right]w_{\mathrm{true}}.
$$

Unless $\lambda=0$ or special degeneracies occur, this is not zero. Hence ridge regression is generally biased.

## 8. Variance reduction in ridge regression

Let

$$
M=X^\top X.
$$

OLS has estimator

$$
\widehat w_{\mathrm{LS}}=M^{-1}X^\top y,
$$

and under homoscedastic noise with variance $\sigma^2$,

$$
\operatorname{Var}(\widehat w_{\mathrm{LS}})=\sigma^2M^{-1}.
$$

For ridge,

$$
\widehat w_\lambda=(M+
\lambda I)^{-1}X^\top y.
$$

The variance is

$$
\operatorname{Var}(\widehat w_\lambda)
=
\sigma^2(M+\lambda I)^{-1}M(M+\lambda I)^{-1}.
$$

In the eigenbasis of $M$, if $M=Q\operatorname{diag}(\mu_j)Q^\top$, the shrinkage factor in direction $j$ is roughly

$$
\frac{\mu_j}{\mu_j+\lambda}.
$$

Directions with small $\mu_j$ are shrunk more strongly. These are exactly the unstable directions in ordinary least squares.

## 9. Regularization beyond ridge

Regularization is not limited to explicit $\ell_2$ penalties. A broad form is

$$
\min_w
\left\{
\frac1N\sum_{i=1}^N \ell(w;x_i,y_i)+\Omega(w)
\right\}.
$$

Different choices of $\Omega$ encode different inductive biases:

- Ridge: $\Omega(w)=\lambda\|w\|_2^2$, which encourages small dense weights.
- LASSO: $\Omega(w)=\lambda\|w\|_1$, which encourages sparsity.
- General $\ell_p$ penalties: $\Omega(w)=\lambda\|w\|_p^p$.
- Constraints can also act as regularization, for example $\|w\|\le R$.

Regularization can also be implicit. It may come from the algorithm, the model architecture, data augmentation, early stopping, feature design, or optimization dynamics.

## 10. Nonlinear classification: two strategies

Linear classifiers are powerful but limited. Some datasets cannot be separated well by a single hyperplane in the original input space.

There are two broad strategies:

1. Use a nonlinear classifier directly in the original feature space.
2. Map the data into nonlinear features and then use a linear classifier there.

The second idea is central to kernels. We introduce a feature map

$$
\phi:\mathcal X\to\mathcal F,
$$

and classify using a linear rule in feature space:

$$
h(x)=\operatorname{sign}(w^\top\phi(x)+w_0).
$$

The classifier is linear in $\phi(x)$, but nonlinear as a function of the original input $x$.

## 11. Hand-coding nonlinear features

A simple one-dimensional example is

$$
\phi(x)=(x,|x|).
$$

A classifier that is linear in $(x,|x|)$ may be nonlinear in the original scalar input $x$.

For two-dimensional data, one can use quadratic features such as

$$
\phi(x_1,x_2)=(x_1,x_2,x_1^2,x_1x_2,x_2^2).
$$

A linear separator in this feature space corresponds to a quadratic decision boundary in the original space:

$$
w_1x_1+w_2x_2+w_3x_1^2+w_4x_1x_2+w_5x_2^2+w_0=0.
$$

This can represent ellipses, parabolas, hyperbolas, and other quadratic boundaries.

## 12. How long are nonlinear feature vectors?

For $p$ original variables, the linear feature vector has length

$$
p.
$$

If we include all monomials of degree exactly $m$, the number of features is

$$
\binom{p+m-1}{m}.
$$

If we include all monomials of degree at most $m$, including the constant feature, the number is

$$
\binom{p+m}{m}.
$$

For quadratic features up to degree $2$, including the constant, the length is

$$
\binom{p+2}{2}=\frac{(p+1)(p+2)}{2}.
$$

Without the constant term, the length is

$$
\binom{p+2}{2}-1.
$$

This combinatorial growth is one reason why explicitly constructing nonlinear features can become expensive.

## 13. Feature extraction in practice

In practice, data often need an initial feature extraction step before a classifier can be trained.

Examples include:

- text $\to$ bag-of-words, TF-IDF, or embeddings;
- images $\to$ pixels, handcrafted descriptors, or learned representations;
- categorical variables $\to$ one-hot encodings;
- time series $\to$ lag features, Fourier features, or summary statistics.

The pipeline is often

$$
\text{data}
\longrightarrow
\text{feature extraction / vectors}
\longrightarrow
\text{classifier}.
$$

The lecture emphasizes that feature design is itself an inductive bias. Choosing features changes what patterns the model can easily represent.

## 14. Implicit and learned nonlinear features

Instead of hand-coding $\phi$, we can choose nonlinear features implicitly or learn them from data.

Implicit feature methods use algorithms whose computations depend on inner products in feature space, without explicitly writing the feature vectors. Kernel methods are the main example.

Learned feature methods estimate the representation from data. Neural networks are the canonical example:

$$
\text{data}
\longrightarrow
\text{learned feature map}
\longrightarrow
\text{classifier}.
$$

This is the machine learning mindset: rather than manually designing every useful feature, let the algorithm learn useful representations when enough data and appropriate structure are available.

## 15. SVMs with nonlinear features

The hard-margin SVM in feature space is

$$
\min_{w,w_0}\frac12\|w\|^2
$$

subject to

$$
y_i\left(\langle w,\phi(x_i)\rangle+w_0\right)\ge1,
\qquad i=1,\ldots,N.
$$

The soft-margin SVM in feature space is

$$
\min_{w,w_0,\xi}\frac12\|w\|^2+C\sum_{i=1}^N\xi_i
$$

subject to

$$
y_i\left(\langle w,\phi(x_i)\rangle+w_0\right)\ge1-\xi_i,
\qquad
\xi_i\ge0.
$$

The corresponding decision function is

$$
h(x)=\operatorname{sign}\left(\langle w,\phi(x)\rangle+w_0\right).
$$

The issue is that $\phi(x)$ may be very high-dimensional, or even infinite-dimensional.

## 16. The key SVM representation

For SVMs, the optimal weight vector lies in the span of the training feature vectors:

$$
w=\sum_{i=1}^N \alpha_i y_i\phi(x_i).
$$

Therefore,

$$
\begin{aligned}
\langle w,\phi(x)\rangle
&=
\left\langle
\sum_{i=1}^N \alpha_i y_i\phi(x_i),
\phi(x)
\right\rangle\\
&=
\sum_{i=1}^N \alpha_i y_i
\langle \phi(x_i),\phi(x)\rangle.
\end{aligned}
$$

Thus the decision function depends only on inner products between feature vectors:

$$
h(x)=\operatorname{sign}\left(
\sum_{i=1}^N \alpha_i y_i
\langle \phi(x_i),\phi(x)\rangle
+w_0
\right).
$$

This observation is the bridge to kernels.

## 17. The kernel trick

Suppose we have a function

$$
k:\mathcal X\times\mathcal X\to\mathbb R
$$

such that

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle
$$

for some feature map $\phi$.

Then we can compute inner products in feature space without explicitly constructing $\phi(x)$ or $\phi(x')$.

The SVM decision function becomes

$$
h(x)=\operatorname{sign}\left(
\sum_{i=1}^N \alpha_i y_i k(x_i,x)+w_0
\right).
$$

This is the kernel trick: replace feature-space inner products by kernel evaluations.

A kernel function must be symmetric:

$$
k(x,x')=k(x',x),
$$

and it must correspond to an inner product in some feature space. Equivalently, for any data points $x_1,\ldots,x_N$, the Gram matrix

$$
K_{ij}=k(x_i,x_j)
$$

must be positive semidefinite.

## 18. Common kernel examples

The linear kernel is

$$
k(x,x')=x^\top x'.
$$

It corresponds to the identity feature map $\phi(x)=x$.

The polynomial kernel is

$$
k(x,x')=(x^\top x'+c)^m,
$$

where $m$ is the degree and $c\ge0$ is an offset parameter. It corresponds to a feature map containing polynomial interactions.

The Gaussian radial basis function kernel is

$$
k(x,x')=\exp\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right).
$$

It corresponds to an infinite-dimensional feature map. The kernel trick lets us use this feature space without explicitly constructing it.

## 19. Kernel SVM summary

The kernel SVM uses the decision rule

$$
h(x)=\operatorname{sign}\left(
\sum_{i=1}^N \alpha_i y_i k(x_i,x)+w_0
\right).
$$

Only training points with nonzero $\alpha_i$ influence the prediction. These points are the support vectors.

The crucial practical point is that we do not need to construct the nonlinear feature vectors explicitly. We only need to compute kernel values $k(x_i,x_j)$.

This idea applies to any method whose solution can be expressed as a linear combination of feature vectors:

$$
w=\sum_i a_i\phi(x_i).
$$

Such methods can often be kernelized by replacing

$$
\langle\phi(x_i),\phi(x_j)\rangle
$$

with

$$
k(x_i,x_j).
$$

## 20. Key takeaways

The main messages of the lecture are:

- Ridge regression replaces $X^\top X$ by $X^\top X+\lambda I$, improving invertibility and stability.
- Regularization deliberately introduces bias to reduce variance.
- The bias-variance decomposition explains why a biased estimator can generalize better than an unbiased one.
- Nonlinear classification can be achieved either by nonlinear classifiers or by nonlinear feature maps.
- Explicit nonlinear feature maps can become very large.
- SVMs in feature space depend only on inner products of feature vectors.
- Kernels compute these inner products implicitly.
- Kernel SVMs allow nonlinear decision boundaries without explicitly constructing nonlinear features.

## 21. Suggested exercises

1. Show that $X^\top X+\lambda I$ is positive definite for every $\lambda>0$ when $X^\top X$ is positive semidefinite.
2. Derive the ridge solution from the gradient of the regularized least-squares objective.
3. Prove that ridge regression is biased under the model $y=Xw_{\mathrm{true}}+\eta$ with $\mathbb E[\eta]=0$.
4. Derive the variance formula

$$
\operatorname{Var}(\widehat w_\lambda)
=
\sigma^2(X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}.
$$

5. Count the number of polynomial features of degree at most $m$ for $p$ variables.
6. For $\phi(x)=(x,x^2)$, write the most general linear classifier in feature space and interpret the decision boundary in the original input.
7. Verify that the polynomial kernel $k(x,x')=(x^\top x'+c)^m$ is symmetric.
8. Explain why the SVM decision function can be computed using only $k(x_i,x)$.
