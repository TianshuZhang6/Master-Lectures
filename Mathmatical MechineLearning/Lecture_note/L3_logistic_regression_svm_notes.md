# L3 Lecture Notes: Logistic Regression and Support Vector Machines

## 1. Overfitting, ERM, and inductive bias

In empirical risk minimization, we choose a predictor by minimizing training loss:

$$
\operatorname{ERM}_{\mathcal H}(S)
\in
\operatorname*{argmin}_{h \in \mathcal H} L_S(h),
$$

where $S$ is the training set and $\mathcal H$ is a chosen hypothesis class.

The key problem is overfitting. If $\mathcal H$ is too large, the model can fit training data very well but generalize poorly. The lecture's main idea is therefore:

$$
\text{ERM can work if we restrict the search space } \mathcal H.
$$

This restriction is called **inductive bias**. In this lecture, the inductive bias is the use of **linear models**.

---

## 2. Linear hypothesis class

A linear classifier is based on a hyperplane:

$$
w^\top x+w_0=0,
$$

where:

- $w\in\mathbb R^d$ is the weight vector.
- $w_0\in\mathbb R$ is the offset or bias term.
- $x\in\mathbb R^d$ is the input.

The linear hypothesis class can be written as

$$
\mathcal H=
\left\{
 x\mapsto w^\top x+w_0
 \;:\;
 w\in\mathbb R^d,\; w_0\in\mathbb R
\right\}.
$$

For binary classification with labels $y_i\in\{-1,+1\}$, the classifier is

$$
h(x;w,w_0)=\operatorname{sign}(w^\top x+w_0).
$$

A point $(x_i,y_i)$ is correctly classified when

$$
y_i(w^\top x_i+w_0)>0.
$$

The quantity

$$
z_i=y_i(w^\top x_i+w_0)
$$

is called the **signed margin score**. Positive $z_i$ means correct classification; negative $z_i$ means incorrect classification.

---

## 3. Loss functions for classification

The ideal classification loss is the $0/1$ loss:

$$
\ell_{0/1}(z)=\mathbf 1_{\{z\le 0\}}.
$$

However, this loss is non-convex and difficult to optimize. Therefore, we often replace it with convex upper bounds.

Two important examples are the **logistic loss**

$$
\ell_{\log}(z)=\log(1+e^{-z}),
$$

and the **hinge loss**

$$
\ell_h(z)=\max\{0,1-z\}.
$$

They satisfy

$$
\ell_{0/1}(z)\le \ell_{\log}(z)
\quad\text{and}\quad
\ell_{0/1}(z)\le \ell_h(z),
$$

up to the usual normalization or convention. This means logistic loss and hinge loss penalize classification mistakes, but are easier to optimize.

---

## 4. Proof: $0/1$ loss equals misclassification probability

For a classifier $h$, define

$$
\ell_{0/1}(h(x),y)=
\begin{cases}
1, & h(x)\neq y,\\
0, & h(x)=y.
\end{cases}
$$

Then the population risk is

$$
L(h)=\mathbb E[\ell_{0/1}(h(X),Y)].
$$

Since $\ell_{0/1}(h(X),Y)$ is an indicator random variable,

$$
\ell_{0/1}(h(X),Y)=\mathbf 1_{\{h(X)\neq Y\}}.
$$

Therefore,

$$
L(h)=\mathbb E\left[\mathbf 1_{\{h(X)\neq Y\}}\right]
=\mathbb P(h(X)\neq Y).
$$

So minimizing the expected $0/1$ loss is exactly the same as minimizing the misclassification probability.

---

# Part I: Logistic Regression

## 5. Bayes classifier

For binary classification with labels $Y\in\{0,1\}$, define

$$
\eta(x)=\mathbb P(Y=1\mid X=x).
$$

The Bayes classifier is

$$
h^*(x)=
\begin{cases}
1, & \eta(x)>\frac12,\\
0, & \text{otherwise}.
\end{cases}
$$

The intuition is simple: if class $1$ is more likely than class $0$ given $x$, predict class $1$.

---

## 6. Proof: Bayes classifier is optimal

We prove that for every classifier $h:\mathbb R^d\to\{0,1\}$,

$$
\mathbb P(h^*(X)\neq Y)
\le
\mathbb P(h(X)\neq Y).
$$

Condition on $X=x$. If we predict $1$, the conditional probability of error is

$$
\mathbb P(Y=0\mid X=x)=1-\eta(x).
$$

If we predict $0$, the conditional probability of error is

$$
\mathbb P(Y=1\mid X=x)=\eta(x).
$$

So the best choice at $x$ is

$$
\min\{\eta(x),1-\eta(x)\}.
$$

If $\eta(x)>\frac12$, then $1-\eta(x)<\eta(x)$, so predicting $1$ is better. If $\eta(x)\le\frac12$, then $\eta(x)\le1-\eta(x)$, so predicting $0$ is better.

Thus $h^*$ minimizes the conditional error for every fixed $x$. Taking expectation over $X$ gives

$$
\mathbb P(h^*(X)\neq Y)
\le
\mathbb P(h(X)\neq Y).
$$

Hence the Bayes classifier is optimal.

---

## 7. Why not use $\hat\eta(x)=w^\top x+w_0$ directly?

The Bayes classifier uses $\eta(x)$ as a probability:

$$
\eta(x)=\mathbb P(Y=1\mid X=x).
$$

Therefore we must have

$$
0\le \eta(x)\le 1.
$$

But a linear function

$$
w^\top x+w_0
$$

can take any real value from $-\infty$ to $+\infty$. So it cannot directly represent a probability.

We need a function that maps real numbers into $[0,1]$. This is why logistic regression uses the sigmoid function.

---

## 8. Sigmoid function

The sigmoid, or logistic, function is

$$
\sigma(z)=\frac{1}{1+e^{-z}}=\frac{e^z}{1+e^z}.
$$

It satisfies

$$
\sigma:\mathbb R\to(0,1).
$$

So logistic regression models

$$
\hat\eta(x)=\mathbb P(Y=1\mid X=x)\approx \sigma(w^\top x+w_0).
$$

Usually we absorb the bias $w_0$ into the feature vector by defining

$$
\tilde x=
\begin{bmatrix}
x\\
1
\end{bmatrix},
\qquad
\tilde w=
\begin{bmatrix}
w\\
w_0
\end{bmatrix}.
$$

Then

$$
w^\top x+w_0=\tilde w^\top \tilde x.
$$

So we often write simply $\sigma(w^\top x)$.

---

## 9. Deriving logistic regression by maximum likelihood

Assume labels $y_i\in\{0,1\}$ and data

$$
S=\{(x_i,y_i)\}_{i=1}^N.
$$

Logistic regression models

$$
p(y_i=1\mid x_i)=\sigma(w^\top x_i),
$$

and

$$
p(y_i=0\mid x_i)=1-\sigma(w^\top x_i).
$$

Therefore, for one observation,

$$
p(y_i\mid x_i)=
\sigma(w^\top x_i)^{y_i}
\left(1-\sigma(w^\top x_i)\right)^{1-y_i}.
$$

Assuming the samples are iid, the likelihood is

$$
\mathcal L(w)=
\prod_{i=1}^N
\sigma(w^\top x_i)^{y_i}
\left(1-\sigma(w^\top x_i)\right)^{1-y_i}.
$$

Maximizing likelihood is equivalent to minimizing negative log-likelihood:

$$
L(w)=-\log\mathcal L(w).
$$

Thus

$$
L(w)=
-\sum_{i=1}^N
\log
\left[
\sigma(w^\top x_i)^{y_i}
\left(1-\sigma(w^\top x_i)\right)^{1-y_i}
\right].
$$

Using log rules,

$$
L(w)=
-\sum_{i=1}^N
\left[
y_i\log\sigma(w^\top x_i)
+
(1-y_i)\log(1-\sigma(w^\top x_i))
\right].
$$

This is the **cross-entropy loss**.

---

## 10. Logistic loss with labels $\{-1,+1\}$

If labels are $y_i\in\{-1,+1\}$, then logistic regression can be written as

$$
p(y_i\mid x_i)=\sigma(y_iw^\top x_i).
$$

The negative log-likelihood becomes

$$
L(w)=\sum_{i=1}^N -\log\sigma(y_iw^\top x_i).
$$

Since

$$
\sigma(z)=\frac{1}{1+e^{-z}},
$$

we have

$$
-\log\sigma(z)
=-\log\left(\frac{1}{1+e^{-z}}\right)
=\log(1+e^{-z}).
$$

Therefore

$$
L(w)=
\sum_{i=1}^N
\log\left(1+\exp(-y_iw^\top x_i)\right).
$$

So the logistic loss is

$$
\ell_{\log}(z)=\log(1+e^{-z}),
$$

where

$$
z_i=y_iw^\top x_i.
$$

The ERM formulation is

$$
\min_w
\frac1N
\sum_{i=1}^N
\log\left(1+\exp(-y_iw^\top x_i)\right).
$$

---

## 11. Gradient of logistic regression

Let

$$
L(w)=\sum_{i=1}^N \log(1+\exp(-y_iw^\top x_i)).
$$

Define

$$
z_i=y_iw^\top x_i.
$$

Then

$$
\ell_{\log}(z_i)=\log(1+e^{-z_i}).
$$

Differentiate:

$$
\frac{d}{dz}\log(1+e^{-z})
=
\frac{-e^{-z}}{1+e^{-z}}
=
-\frac{1}{1+e^z}.
$$

Since

$$
\nabla_w z_i=y_i x_i,
$$

we get

$$
\nabla_w L(w)=
-\sum_{i=1}^N
\frac{y_i x_i}{1+\exp(y_iw^\top x_i)}.
$$

Equivalently, for the $y_i\in\{0,1\}$ formulation,

$$
\nabla_w L(w)=
\sum_{i=1}^N
\left(\sigma(w^\top x_i)-y_i\right)x_i.
$$

---

## 12. What happens for linearly separable data?

Suppose the data are linearly separable. Then there exists a vector $u$ such that

$$
y_i u^\top x_i>0
\quad
\text{for all } i.
$$

Now consider scaling $u$ by a positive scalar $c$:

$$
w=cu.
$$

Then

$$
y_iw^\top x_i=c y_i u^\top x_i.
$$

As $c\to\infty$,

$$
y_iw^\top x_i\to\infty.
$$

Therefore each logistic loss term satisfies

$$
\log(1+\exp(-y_iw^\top x_i))\to\log(1+0)=0.
$$

So

$$
L(cu)\to 0
\quad\text{as}\quad
c\to\infty.
$$

But for finite $c$, each term is still positive:

$$
\log(1+\exp(-y_iw^\top x_i))>0.
$$

Therefore logistic regression has no finite minimizer on strictly linearly separable data. The loss approaches its infimum $0$, but only as

$$
\|w\|\to\infty.
$$

This is the “infinity problem” mentioned in the slides.

---

## 13. Regularized logistic regression

To fix the infinite norm problem, add a regularizer:

$$
\min_w
\sum_{i=1}^N
\log\left(1+\exp(-y_iw^\top x_i)\right)
+
\lambda\|w\|_q^q.
$$

Common choices:

$$
q=2
\quad\Rightarrow\quad
\lambda\|w\|_2^2,
$$

called **L2-regularized logistic regression**.

$$
q=1
\quad\Rightarrow\quad
\lambda\|w\|_1,
$$

called **L1-regularized sparse logistic regression**.

The L2 version is especially nice because

$$
\lambda\|w\|_2^2
$$

makes the objective strongly convex in $w$ when $\lambda>0$. Strong convexity improves optimization behavior and often gives more stable solutions.

---

# Part II: Multi-class Logistic Regression

For $K$ classes, $Y\in\{0,1,\dots,K-1\}$, binary logistic regression generalizes to **softmax regression**.

Each class $k$ has a weight vector $w_k$. The model is

$$
p(Y=k\mid X=x)=
\frac{\exp(w_k^\top x)}
{\sum_{j=0}^{K-1}\exp(w_j^\top x)}.
$$

The classifier predicts the most likely class:

$$
h(x)=\operatorname*{argmax}_{k} p(Y=k\mid X=x).
$$

Since the denominator is common to all classes,

$$
h(x)=\operatorname*{argmax}_{k} w_k^\top x.
$$

For one-hot encoded labels $y_{ik}$, the cross-entropy loss is

$$
L(W)=
-\sum_{i=1}^N
\sum_{k=0}^{K-1}
y_{ik}
\log p(Y=k\mid X=x_i).
$$

---

# Part III: Support Vector Machines

## 14. From hinge loss to SVM

SVMs use the hinge loss

$$
\ell_h(z)=\max\{0,1-z\},
$$

where

$$
z_i=y_i(w^\top x_i+w_0).
$$

The SVM objective is not just hinge loss. The important missing part is the regularizer:

$$
\frac12\|w\|^2.
$$

So the soft-margin SVM objective is

$$
\min_{w,w_0}
\frac12\|w\|^2
+
C\sum_{i=1}^N
\max\{0,1-y_i(w^\top x_i+w_0)\}.
$$

The regularizer controls the margin, while the hinge loss controls classification and margin violations.

---

## 15. Convex hull intuition

For two classes, consider the convex hull of positive points and the convex hull of negative points.

If the two convex hulls do not intersect, then there exists a separating hyperplane between them. Geometrically, SVM chooses a separating hyperplane that maximizes the distance to the closest points from each class.

The closest points to the separating hyperplane become the **support vectors**.

---

## 16. Canonical normalization

Assume the data are strictly linearly separable. Then there exists $(w,w_0)$ such that

$$
y_i(w^\top x_i+w_0)>0
\quad
\text{for all } i.
$$

If $(w,w_0)$ separates the data, then for any $\delta>0$,

$$
(\delta w,\delta w_0)
$$

also separates the data, because

$$
y_i(\delta w^\top x_i+\delta w_0)
=
\delta y_i(w^\top x_i+w_0)>0.
$$

So the scale of $(w,w_0)$ is arbitrary. To remove this ambiguity, SVM uses the **canonical normalization**:

$$
\min_{1\le i\le N}|w^\top x_i+w_0|=1.
$$

Equivalently, for labels $y_i\in\{-1,+1\}$,

$$
\min_{1\le i\le N}y_i(w^\top x_i+w_0)=1.
$$

Thus all training points satisfy

$$
y_i(w^\top x_i+w_0)\ge 1.
$$

---

## 17. Proof: distance from a point to a hyperplane

Let the hyperplane be

$$
H=\{x:w^\top x+w_0=0\}.
$$

For any point $x$, its distance to $H$ is

$$
\operatorname{dist}(x,H)=\frac{|w^\top x+w_0|}{\|w\|}.
$$

Proof:

Let $x_H$ be the projection of $x$ onto the hyperplane. Since $x-x_H$ is perpendicular to the hyperplane, it must be parallel to $w$. So

$$
x_H=x-\alpha w
$$

for some scalar $\alpha$.

Because $x_H$ lies on the hyperplane,

$$
w^\top x_H+w_0=0.
$$

Substitute $x_H=x-\alpha w$:

$$
w^\top(x-\alpha w)+w_0=0.
$$

Thus

$$
w^\top x-\alpha\|w\|^2+w_0=0.
$$

Solving for $\alpha$,

$$
\alpha=\frac{w^\top x+w_0}{\|w\|^2}.
$$

Therefore

$$
\|x-x_H\|=\|\alpha w\|=|\alpha|\|w\|=\frac{|w^\top x+w_0|}{\|w\|}.
$$

This proves the formula.

Under canonical normalization, the closest training points satisfy

$$
|w^\top x_i+w_0|=1.
$$

Therefore their distance to the separating hyperplane is

$$
\frac{1}{\|w\|}.
$$

So the margin is

$$
\gamma=\frac{1}{\|w\|}.
$$

The distance between the two supporting hyperplanes

$$
w^\top x+w_0=1
$$

and

$$
w^\top x+w_0=-1
$$

is

$$
\frac{2}{\|w\|}.
$$

---

## 18. Why large margins are preferred

Suppose a test point is generated by perturbing a training point:

$$
(x,y)\mapsto(x+\delta x,y),
$$

with bounded noise

$$
\|\delta x\|\le r.
$$

If the classifier has margin $\gamma>r$, then the perturbation cannot cross the decision boundary. Therefore, the perturbed point remains correctly classified.

Proof:

For a correctly classified training point,

$$
y(w^\top x+w_0)\ge \gamma\|w\|.
$$

For the perturbed point,

$$
y(w^\top(x+\delta x)+w_0)
=
y(w^\top x+w_0)+y w^\top\delta x.
$$

Using Cauchy-Schwarz,

$$
|w^\top\delta x|
\le
\|w\|\|\delta x\|
\le
r\|w\|.
$$

Therefore

$$
y(w^\top(x+\delta x)+w_0)
\ge
\gamma\|w\|-r\|w\|
=
(\gamma-r)\|w\|.
$$

If $\gamma>r$, then

$$
(\gamma-r)\|w\|>0.
$$

So the perturbed point is still correctly classified.

This explains the robustness intuition behind large-margin classifiers.

---

## 19. Hard-margin SVM

For linearly separable data, we want to maximize the margin:

$$
\gamma=\frac{1}{\|w\|}.
$$

Maximizing $1/\|w\|$ is equivalent to minimizing $\|w\|$. For mathematical convenience, SVM minimizes

$$
\frac12\|w\|^2.
$$

The hard-margin SVM is therefore

$$
\min_{w,w_0}
\frac12\|w\|^2
$$

subject to

$$
y_i(w^\top x_i+w_0)\ge 1,
\qquad
 i=1,\dots,N.
$$

This is a convex optimization problem because $\frac12\|w\|^2$ is convex and each constraint is affine.

---

## 20. Soft-margin SVM

If the data are not linearly separable, the hard-margin constraints may be impossible to satisfy. We introduce slack variables $\xi_i\ge0$:

$$
y_i(w^\top x_i+w_0)\ge 1-\xi_i.
$$

The soft-margin SVM is

$$
\min_{w,w_0,\xi}
\frac12\|w\|^2
+
C\sum_{i=1}^N \xi_i
$$

subject to

$$
y_i(w^\top x_i+w_0)\ge 1-\xi_i,
\qquad
 i=1,\dots,N,
$$

and

$$
\xi_i\ge0.
$$

Interpretation:

- $\xi_i=0$: point satisfies the margin constraint.
- $0<\xi_i<1$: point is correctly classified but inside the margin.
- $\xi_i\ge1$: point is misclassified or exactly beyond the wrong side.

The parameter $C>0$ controls the tradeoff:

- large $C$: heavily penalize violations, more like hard-margin SVM;
- small $C$: allow more violations, usually larger margin.

---

## 21. Proof: soft-margin SVM equals hinge-loss formulation

Start with the slack formulation:

$$
\min_{w,w_0,\xi}
\frac12\|w\|^2
+
C\sum_{i=1}^N \xi_i
$$

subject to

$$
y_i(w^\top x_i+w_0)\ge 1-\xi_i,
$$

and

$$
\xi_i\ge0.
$$

Rearrange the margin constraint:

$$
\xi_i\ge 1-y_i(w^\top x_i+w_0).
$$

Together with $\xi_i\ge0$, we get

$$
\xi_i\ge\max\{0,1-y_i(w^\top x_i+w_0)\}.
$$

For fixed $w,w_0$, the objective is increasing in each $\xi_i$, so the optimal slack is the smallest feasible value:

$$
\xi_i^*=\max\{0,1-y_i(w^\top x_i+w_0)\}.
$$

Substitute this back into the objective:

$$
\min_{w,w_0}
\frac12\|w\|^2
+
C\sum_{i=1}^N
\max\{0,1-y_i(w^\top x_i+w_0)\}.
$$

This is exactly the hinge-loss SVM formulation.

---

## 22. Logistic regression vs SVM

Both methods use linear scores

$$
w^\top x+w_0.
$$

But they use different losses.

Logistic regression:

$$
\min_w
\sum_{i=1}^N
\log(1+\exp(-y_iw^\top x_i))
+
\lambda\|w\|^2.
$$

SVM:

$$
\min_w
\frac12\|w\|^2
+
C\sum_{i=1}^N
\max\{0,1-y_iw^\top x_i\}.
$$

Main difference:

- Logistic regression gives probabilistic outputs through $\sigma(w^\top x)$.
- SVM focuses on maximizing the margin.
- Logistic loss is smooth.
- Hinge loss is non-smooth but creates sparse dependence on support vectors.
- Both are convex for linear models with convex regularization.

---

## 23. Key formulas to remember

Linear classifier:

$$
h(x)=\operatorname{sign}(w^\top x+w_0).
$$

Correct classification:

$$
y_i(w^\top x_i+w_0)>0.
$$

Logistic sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Cross-entropy loss:

$$
L(w)=
-\sum_{i=1}^N
\left[
y_i\log\sigma(w^\top x_i)
+
(1-y_i)\log(1-\sigma(w^\top x_i))
\right].
$$

Logistic loss for $y_i\in\{-1,+1\}$:

$$
L(w)=
\sum_{i=1}^N
\log(1+\exp(-y_iw^\top x_i)).
$$

Hinge loss:

$$
\ell_h(z)=\max\{0,1-z\}.
$$

Distance to hyperplane:

$$
\operatorname{dist}(x,H)=\frac{|w^\top x+w_0|}{\|w\|}.
$$

Hard-margin SVM:

$$
\min_{w,w_0}
\frac12\|w\|^2
\quad
\text{s.t.}
\quad
y_i(w^\top x_i+w_0)\ge1.
$$

Soft-margin SVM:

$$
\min_{w,w_0}
\frac12\|w\|^2
+
C\sum_{i=1}^N
\max\{0,1-y_i(w^\top x_i+w_0)\}.
$$

---

## 24. Answers to the main slide questions

**Question: Why does directly writing $\hat\eta(x)=w^\top x+w_0$ not work?**

Because $\eta(x)$ is a probability and must lie in $[0,1]$, but $w^\top x+w_0$ can be any real number. Logistic regression fixes this using the sigmoid:

$$
\hat\eta(x)=\sigma(w^\top x+w_0).
$$

**Question: What happens to logistic regression if data are linearly separable?**

The loss can be made arbitrarily close to $0$ by scaling $w$ to infinity:

$$
w=cu,
\qquad
c\to\infty.
$$

Therefore the unregularized problem has no finite minimizer.

**Question: Why add regularization?**

Regularization controls model complexity and prevents $\|w\|$ from going to infinity:

$$
L_{\text{reg}}(w)=L(w)+\lambda\|w\|_q^q.
$$

**Question: Why is the closest point to the separating hyperplane at distance $1/\|w\|$?**

Under canonical normalization,

$$
|w^\top x_i+w_0|=1
$$

for the closest training points. Since

$$
\operatorname{dist}(x_i,H)=\frac{|w^\top x_i+w_0|}{\|w\|},
$$

the closest distance is

$$
\frac1{\|w\|}.
$$

**Question: How does hinge loss appear from slack variables?**

The optimal slack is

$$
\xi_i^*=\max\{0,1-y_i(w^\top x_i+w_0)\}.
$$

Substituting into the slack-variable SVM gives the hinge-loss SVM.
