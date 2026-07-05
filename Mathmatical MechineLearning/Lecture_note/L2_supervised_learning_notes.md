# L2 Lecture Notes: Introduction to Supervised Learning

## 1. Supervised learning: basic setup

In supervised learning, we are given labeled examples and want to learn a prediction rule that performs well on unseen data.

A training set is

$$
S=\{(x_1,y_1),\ldots,(x_n,y_n)\}\subseteq \mathcal X\times \mathcal Y.
$$

Here:

- $\mathcal X$ is the **data domain**, often $\mathbb R^d$.
- $\mathcal Y$ is the **label domain**.
- A **classifier** or **hypothesis** is a function

$$
h:\mathcal X\to \mathcal Y.
$$

For binary classification, typical label sets are

$$
\mathcal Y=\{0,1\}
\quad\text{or}\quad
\mathcal Y=\{-1,+1\}.
$$

For regression, the output is numerical, for example

$$
\mathcal Y=\mathbb R
\quad\text{or}\quad
\mathcal Y=\mathbb R^m.
$$

So the difference is

$$
\text{classification: discrete labels},
\qquad
\text{regression: continuous/numerical labels}.
$$

---

## 2. Probabilistic model for classification

The lecture assumes that data are generated from an unknown but fixed joint distribution

$$
P \quad \text{on} \quad \mathcal X\times \mathcal Y.
$$

A random data point is written as

$$
(X,Y)\sim P.
$$

For binary labels $\mathcal Y=\{0,1\}$, define the class-conditional probability

$$
\eta(x)=P(Y=1\mid X=x)=\mathbb E[Y\mid X=x].
$$

The **risk**, also called **generalization error**, of a classifier $h$ is

$$
L(h)=P(h(X)\neq Y).
$$

Equivalently,

$$
L(h)=\mathbb E\left[\mathbf 1_{\{h(X)\neq Y\}}\right].
$$

This is the true population error, not merely the training error.

---

## 3. Bayes classifier

The goal is to find the classifier with the smallest possible risk:

$$
\inf_h L(h).
$$

For binary classification, the Bayes classifier is

$$
h^*(x)=
\begin{cases}
1, & \eta(x)>\frac12,\\
0, & \eta(x)\le \frac12.
\end{cases}
$$

The intuition is simple: if, given $X=x$, class $1$ is more likely than class $0$, predict $1$; otherwise predict $0$.

If $\eta(x)=1/2$, both labels have the same conditional error, so the tie-breaking rule does not matter for optimality.

---

## 4. Proof: Bayes classifier is optimal

We prove that for every classifier $h:\mathcal X\to\{0,1\}$,

$$
L(h^*)\le L(h).
$$

Fix $x\in\mathcal X$. The conditional error of predicting $1$ is

$$
P(1\neq Y\mid X=x)=P(Y=0\mid X=x)=1-\eta(x).
$$

The conditional error of predicting $0$ is

$$
P(0\neq Y\mid X=x)=P(Y=1\mid X=x)=\eta(x).
$$

Therefore, at each fixed $x$, the best possible prediction is the one with smaller conditional error:

$$
\min\{\eta(x),1-\eta(x)\}.
$$

The Bayes classifier does exactly this:

- if $\eta(x)>1/2$, then $1-\eta(x)<\eta(x)$, so predict $1$;
- if $\eta(x)\le 1/2$, then $\eta(x)\le 1-\eta(x)$, so predict $0$.

Thus for every $x$,

$$
P(h^*(X)\neq Y\mid X=x)
\le
P(h(X)\neq Y\mid X=x).
$$

Taking expectation with respect to $X$ and using the tower property gives

$$
P(h^*(X)\neq Y)
\le
P(h(X)\neq Y).
$$

Therefore,

$$
L(h^*)\le L(h).
$$

So $h^*$ is an optimal classifier.

---

## 5. Bayes error

The **Bayes error** is the smallest possible classification error:

$$
L^*=\inf_{h:\mathcal X\to\{0,1\}} P(h(X)\neq Y).
$$

Since the Bayes classifier is optimal,

$$
L^*=L(h^*).
$$

At each $x$, the Bayes classifier makes error

$$
\min\{\eta(x),1-\eta(x)\}.
$$

Therefore,

$$
L^*=\mathbb E\left[\min\{\eta(X),1-\eta(X)\}\right].
$$

Now prove the alternative formula:

$$
L^*=\frac12-\frac12\mathbb E\left[|2\eta(X)-1|\right].
$$

For any $a\in[0,1]$,

$$
\min\{a,1-a\}=\frac12-\left|a-\frac12\right|.
$$

Since

$$
\left|a-\frac12\right|=\frac12|2a-1|,
$$

we get

$$
\min\{a,1-a\}=\frac12-\frac12|2a-1|.
$$

Set $a=\eta(X)$ and take expectation to obtain

$$
L^*=\frac12-\frac12\mathbb E\left[|2\eta(X)-1|\right].
$$

---

## 6. Why is the Bayes classifier idealized?

The Bayes classifier requires knowing

$$
\eta(x)=P(Y=1\mid X=x).
$$

But in real supervised learning, the distribution $P$ is fixed but unknown. We only observe a finite training set

$$
S=\{(x_i,y_i)\}_{i=1}^n.
$$

Therefore, the Bayes classifier is usually not directly computable. It is an ideal benchmark: it tells us the best possible risk any classifier could achieve if the full data-generating distribution were known.

---

## 7. Multiclass Bayes classifier

For $K$ classes,

$$
\mathcal Y=\{1,2,\ldots,K\}.
$$

Define

$$
\eta_i(x)=P(Y=i\mid X=x).
$$

The Bayes classifier is

$$
h^*(x)\in \operatorname*{argmax}_{1\le i\le K} P(Y=i\mid X=x).
$$

That is, it predicts the most likely class given $X=x$.

### Proof of multiclass optimality

For any classifier $h$,

$$
P(h(X)\neq Y\mid X=x)=1-P(Y=h(x)\mid X=x).
$$

For the Bayes classifier,

$$
P(h^*(X)\neq Y\mid X=x)=1-P(Y=h^*(x)\mid X=x).
$$

Since $h^*(x)$ maximizes the conditional probability,

$$
P(Y=h^*(x)\mid X=x)
\ge
P(Y=h(x)\mid X=x).
$$

Therefore,

$$
P(h^*(X)\neq Y\mid X=x)
\le
P(h(X)\neq Y\mid X=x).
$$

Taking expectation over $X$ gives

$$
L(h^*)\le L(h).
$$

For $K=2$, this reduces to the binary classifier because

$$
P(Y=1\mid X=x)>\frac12
\iff
P(Y=1\mid X=x)>P(Y=0\mid X=x).
$$

---

## 8. Nearest-neighbor classification

The $k$-nearest-neighbor classifier is a simple nonparametric method.

Training phase: store the training data.

Prediction phase: for a test point $x$,

1. find the $k$ training points closest to $x$;
2. predict the majority label among them.

For $k=1$, the classifier predicts the label of the nearest training point.

Nearest-neighbor classifiers can create complex nonlinear decision boundaries because the prediction depends directly on the geometry of the training sample.

---

## 9. Asymptotic 1-NN error and Bayes error

The slides state that, asymptotically, the error of the 1-nearest-neighbor classifier is

$$
L_{\mathrm{NN}}=\mathbb E\left[2\eta(X)(1-\eta(X))\right].
$$

Intuition: as $n\to\infty$, the nearest neighbor of $X=x$ is very close to $x$. Under suitable regularity assumptions, its label behaves like an independent draw from the conditional distribution $P(Y\mid X=x)$.

Let $Y$ be the true label of the test point and $Y'$ be the nearest neighbor's label. Given $X=x$,

$$
P(Y=1\mid X=x)=\eta(x),
$$

and approximately also

$$
P(Y'=1\mid X=x)=\eta(x).
$$

The 1-NN classifier makes an error when $Y'\neq Y$:

$$
\begin{aligned}
P(Y'\neq Y\mid X=x)
&=P(Y'=1,Y=0\mid X=x)+P(Y'=0,Y=1\mid X=x)\\
&=\eta(x)(1-\eta(x))+(1-\eta(x))\eta(x)\\
&=2\eta(x)(1-\eta(x)).
\end{aligned}
$$

Taking expectation over $X$ gives

$$
L_{\mathrm{NN}}=\mathbb E\left[2\eta(X)(1-\eta(X))\right].
$$

---

## 10. Proof: $L_{\mathrm{NN}}\le 2L^*$

Let

$$
A(X)=\min\{\eta(X),1-\eta(X)\}.
$$

Then

$$
L^*=\mathbb E[A(X)].
$$

Also,

$$
\eta(X)(1-\eta(X))=A(X)(1-A(X)).
$$

So

$$
L_{\mathrm{NN}}=2\mathbb E[A(X)(1-A(X))].
$$

Now use

$$
\mathbb E[A(1-A)]=\mathbb E[A]-\mathbb E[A^2].
$$

Since

$$
\mathbb E[A^2]\ge (\mathbb E[A])^2,
$$

we have

$$
\mathbb E[A(1-A)]
\le
\mathbb E[A](1-\mathbb E[A]).
$$

Hence,

$$
L_{\mathrm{NN}}
\le
2L^*(1-L^*)
\le
2L^*.
$$

Thus asymptotically, 1-nearest neighbor has error at most twice the Bayes error:

$$
L_{\mathrm{NN}}\le 2L^*.
$$

---

## 11. No-Free-Lunch idea

The No-Free-Lunch theorem says there is no universal learner that succeeds on every possible data distribution.

More precisely: for every learning algorithm, there exists some distribution on which that learner performs badly, even though another learner could perform well on that same distribution.

This does **not** mean learning is impossible. It means learning requires assumptions, such as

$$
\text{smoothness},\quad
\text{low-dimensional structure},\quad
\text{a useful hypothesis class},\quad
\text{a meaningful metric},\quad
\text{or prior knowledge}.
$$

So the theorem explains why inductive bias is necessary.

It also answers the slide question: "Does this show that we cannot always match the Bayes classifier?"

The answer is: not for every distribution with one universal finite-sample learner. For a particular distribution and enough suitable assumptions, we may approximate Bayes risk. But without assumptions, no algorithm can guarantee Bayes-level performance on all possible tasks.

---

## 12. General loss functions

Instead of only using misclassification error, we can define a general nonnegative loss

$$
\ell:\mathcal H\times \mathcal X\times \mathcal Y\to \mathbb R_+.
$$

The population risk is

$$
L(h)=\mathbb E[\ell(h,X,Y)].
$$

The empirical risk is

$$
L_S(h)=\frac1n\sum_{i=1}^n \ell(h,x_i,y_i).
$$

For classification, the 0/1 loss is

$$
\ell_{0/1}(h,(x,y))=
\begin{cases}
1, & h(x)\neq y,\\
0, & h(x)=y.
\end{cases}
$$

Then

$$
\mathbb E[\ell_{0/1}(h,(X,Y))]
=\mathbb E\left[\mathbf 1_{\{h(X)\neq Y\}}\right]
=P(h(X)\neq Y).
$$

So the 0/1 loss recovers the earlier definition of classification risk.

---

## 13. Margin losses

For binary classification with labels $y\in\{-1,+1\}$ and score function $f_\theta(x)$, define the margin

$$
z=yf_\theta(x).
$$

A correct confident prediction has large positive $z$.

Common losses include

$$
\ell_{0/1}(z)=\mathbf 1_{\{z\le 0\}},
$$

$$
\ell_{\mathrm{hinge}}(z)=\max\{0,1-z\},
$$

and

$$
\ell_{\log}(z)=\log(1+e^{-z}).
$$

The hinge loss is a convex upper bound on the 0/1 loss. Logistic loss is a smooth convex surrogate; with suitable scaling, it can also be viewed as an upper bound. These losses are easier to optimize than 0/1 loss.

---

## 14. Empirical Risk Minimization

Because the true distribution $P$ is unknown, the true risk

$$
L(h)=\mathbb E[\ell(h,X,Y)]
$$

cannot usually be computed.

Instead, ERM minimizes the empirical risk:

$$
\operatorname{ERM}(S)
\in
\operatorname*{argmin}_{h\in\mathcal H} L_S(h).
$$

For 0/1 loss,

$$
L_S(h)=\frac1n\left|\{i\in[n]:h(x_i)\neq y_i\}\right|.
$$

The ERM principle is

$$
\text{choose a hypothesis that minimizes training error.}
$$

The main pitfall is overfitting: a classifier may perform perfectly on the training data but badly on unseen data.

---

## 15. Overfitting example: memorization

Define the classifier

$$
h(x)=
\begin{cases}
y_i, & x=x_i \text{ for some training point } x_i,\\
0, & \text{otherwise}.
\end{cases}
$$

This classifier has zero empirical risk:

$$
L_S(h)=0.
$$

But suppose $X$ is continuously distributed, for example uniformly over the unit square, and $Y$ is an independent fair coin. Then a new test point almost surely does not equal any training point:

$$
P(X\in\{x_1,\ldots,x_n\})=0.
$$

So the classifier almost always predicts $0$. Since

$$
P(Y=1)=\frac12,
$$

the test error is

$$
L(h)=\frac12.
$$

Thus the classifier has perfect training accuracy but random-guess test performance.

Memorization is not always bad, though. It can be useful when exact repeated inputs occur, or when combined with meaningful geometric assumptions, as in nearest-neighbor methods.

---

## 16. Inductive bias and restricted hypothesis classes

To make ERM meaningful, we restrict the search space to a hypothesis class $\mathcal H$ chosen before seeing the training data.

ERM over $\mathcal H$ is

$$
\operatorname{ERM}_{\mathcal H}(S)
\in
\operatorname*{argmin}_{h\in\mathcal H} L_S(h).
$$

Examples of hypothesis classes include

$$
\text{linear classifiers},\quad
\text{decision trees},\quad
\text{random forests},\quad
\text{neural networks}.
$$

A good hypothesis class balances

$$
\text{too large } \mathcal H \Rightarrow \text{overfitting},
$$

$$
\text{too small } \mathcal H \Rightarrow \text{underfitting}.
$$

This is the role of inductive bias.

---

## 17. Why 0/1-loss ERM is hard

The 0/1 empirical risk minimization problem is

$$
\min_{h\in\mathcal H}
\frac1n\sum_{i=1}^n \mathbf 1_{\{h(x_i)\neq y_i\}}.
$$

This is typically hard because the objective is discontinuous and nonconvex. For many rich hypothesis classes, exact minimization is NP-hard.

However, it is not always hard. Easy cases include:

1. **Small finite hypothesis class**. If $\mathcal H$ is finite and small, evaluate every $h\in\mathcal H$ and choose the best one.
2. **One-dimensional threshold classifiers**. Let $h_t(x)=\mathbf 1_{\{x\ge t\}}$. Sort the data points and scan possible thresholds.
3. **Decision stumps**. For depth-1 decision trees, scan each feature and each possible threshold.
4. **Linearly separable feasibility**. If the goal is only to find a separator with zero training error and the data are linearly separable, one can use the perceptron or linear programming. Exact minimization of the number of mistakes for linear classifiers is hard in general.

---

## 18. ERM as finite-sum optimization

Many machine learning training problems have the form

$$
f(\theta)=\frac1n\sum_{i=1}^n f_i(\theta),
$$

where $\theta$ denotes the parameters of the model.

Usually,

$$
f_i(\theta)=\ell(h_\theta,x_i,y_i).
$$

So empirical risk minimization becomes a finite-sum optimization problem:

$$
\min_\theta
\frac1n\sum_{i=1}^n \ell(h_\theta,x_i,y_i).
$$

Examples include

$$
\text{logistic regression},
\quad
\text{support vector machines},
\quad
\text{deep neural networks},
\quad
\text{maximum likelihood estimation}.
$$

---

## 19. Example: least squares as finite-sum optimization

For data matrix $X\in\mathbb R^{n\times d}$ and labels $y\in\mathbb R^n$, least squares solves

$$
\min_\theta \frac12\|X\theta-y\|^2.
$$

Writing $x_i^\top$ for the $i$-th row of $X$,

$$
\|X\theta-y\|^2=\sum_{i=1}^n (x_i^\top\theta-y_i)^2.
$$

Therefore,

$$
\frac12\|X\theta-y\|^2
=
\sum_{i=1}^n \frac12(x_i^\top\theta-y_i)^2.
$$

So the individual losses are

$$
f_i(\theta)=\frac12(x_i^\top\theta-y_i)^2.
$$

If using the averaged convention,

$$
f(\theta)=\frac1n\sum_{i=1}^n f_i(\theta).
$$

---

## 20. Example: neural network classification

Let $\operatorname{net}(\theta,x_i)$ be the neural network output on input $x_i$.

Training a classifier can be written as

$$
\min_\theta
\frac1n\sum_{i=1}^n
\ell(y_i,\operatorname{net}(\theta,x_i)).
$$

The individual loss is

$$
f_i(\theta)=\ell(y_i,\operatorname{net}(\theta,x_i)).
$$

For multiclass classification, a common choice is cross-entropy loss.

---

## 21. Example: PCA as finite-sum optimization

The usual first principal component problem is

$$
\max_{\|x\|=1}
x^\top
\left(
\sum_{i=1}^n z_i z_i^\top
\right)
x.
$$

Equivalently, it can be written as the minimization problem

$$
\min_{\|x\|=1}
-x^\top
\left(
\sum_{i=1}^n z_i z_i^\top
\right)
x.
$$

Since

$$
x^\top z_i z_i^\top x=(z_i^\top x)^2,
$$

we can rewrite the minimization objective as

$$
\sum_{i=1}^n -(z_i^\top x)^2.
$$

Therefore, the individual losses are

$$
f_i(x)=-(z_i^\top x)^2.
$$

Let

$$
C=\sum_{i=1}^n z_i z_i^\top.
$$

Then PCA solves

$$
\max_{\|x\|=1} x^\top Cx.
$$

By the Rayleigh quotient theorem, the solution is the top eigenvector of $C$.

---

## 22. Summary of the lecture

The main ideas are:

- The true risk is

$$
L(h)=P(h(X)\neq Y),
$$

but $P$ is unknown, so we cannot directly minimize $L(h)$.

- The Bayes classifier is the ideal optimal classifier:

$$
h^*(x)\in \operatorname*{argmax}_y P(Y=y\mid X=x).
$$

- Its risk is the Bayes error:

$$
L^*=\mathbb E\left[\min\{\eta(X),1-\eta(X)\}\right].
$$

- Nearest-neighbor methods are practical nonparametric classifiers. Asymptotically, 1-NN satisfies

$$
L_{\mathrm{NN}}\le 2L^*.
$$

- ERM replaces unknown population risk with training risk:

$$
L_S(h)=\frac1n\sum_{i=1}^n \ell(h,x_i,y_i).
$$

- Unrestricted ERM can overfit, so we need inductive bias through a hypothesis class $\mathcal H$.

- Many ML methods reduce to finite-sum optimization:

$$
\min_\theta \frac1n\sum_{i=1}^n f_i(\theta).
$$
