# L6 Stability Structure: Structural Stability

Based on Section 6 and the questions about structural stability, the torus example, nonwandering sets, Peixoto's theorem, Morse-Smale systems, transversality, and genericity.

## 0. Global view

This section asks:

$$
\text{When does the qualitative behavior of a dynamical system survive small perturbations?}
$$

For a vector field

$$
\dot{x}=f(x),
$$

we perturb it slightly to

$$
\dot{x}=g(x).
$$

Then we ask whether the phase portrait stays essentially the same. If yes, then $f$ is called **structurally stable**.

The logical structure is

$$
\text{closeness of vector fields}
\Longrightarrow
\text{structural stability}
\Longrightarrow
\text{examples}
\Longrightarrow
\text{Peixoto's theorem}
\Longrightarrow
\text{Morse--Smale systems}
\Longrightarrow
\text{genericity}.
$$

The deepest message is

$$
\boxed{
\text{Hyperbolicity controls local stability, and transversality controls global stability.}
}
$$

## 1. Why Definition 6.1 is needed

Before saying a vector field is structurally stable, we need to define what it means for two vector fields to be **close**.

That is why the section introduces the function space

$$
C^k(K,\mathbb R^d)
$$

and the norm

$$
\|f\|_{C^k}.
$$

This is a measuring tool. It does not only measure function values; it also measures derivatives.

So the purpose of Definition 6.1 is

$$
\boxed{\text{to define a precise meaning of small perturbation.}}
$$

## 2. The space $C^k(K,\mathbb R^d)$

The notation

$$
C^k(K,\mathbb R^d)
$$

means the space of functions

$$
f:K\to\mathbb R^d
$$

whose derivatives up to order $k$ exist and are continuous.

If

$$
f=(f_1,f_2,\ldots,f_d)^\top,
$$

then $f$ is vector-valued. In dimension $2$, for example,

$$
f(x,y)=
\begin{pmatrix}
f_1(x,y)\\
f_2(x,y)
\end{pmatrix}.
$$

A vector field can be written as

$$
\dot{x}=f(x),
$$

so $f$ gives the arrows of the dynamical system.

## 3. Meaning of the $C^k$ norm

The definition gives a norm of the form

$$
\|f\|_{C^k}
=
\max_l
\sum_{|\alpha|=0}^{k}
\sup_{x\in K}|D^\alpha f_l(x)|.
$$

This means:

$$
\boxed{
\text{measure every component of }f\text{ and every derivative up to order }k\text{ on }K.
}
$$

The index $l$ chooses the component $f_l$. The symbol $\alpha$ is a multi-index, telling us which partial derivative to take.

For example, if

$$
\alpha=(2,1),
$$

then

$$
D^\alpha f
=
\frac{\partial^3 f}{\partial x_1^2\partial x_2}.
$$

The expression

$$
\sup_{x\in K}|D^\alpha f_l(x)|
$$

means the largest size of that derivative on $K$.

Thus the $C^k$ norm controls

$$
f,\quad Df,\quad D^2f,\quad\ldots,\quad D^k f.
$$

## 4. Why derivatives are included

Closeness of function values is not enough for dynamics.

For example, let

$$
f(x)=0,
$$

and

$$
g(x)=0.001\sin(100000x).
$$

The values of $g$ are small, so $g$ is close to $f$ in $C^0$. But

$$
g'(x)=100\cos(100000x),
$$

which can be large. Thus $g$ is not close to $f$ in $C^1$.

In dynamical systems, derivatives control local behavior near equilibria and periodic orbits. So structural stability is naturally formulated using $C^1$ closeness:

$$
\|f-g\|_{C^1}<\varepsilon.
$$

This means both the vector fields and their derivatives are close.

## 5. Why non-compact spaces use compact windows $K_j$

If $K$ is compact, one can use a global supremum such as

$$
\sup_{x\in K}|f(x)-g(x)|.
$$

If $K$ is non-compact, this supremum may be infinite. For example, on $K=\mathbb R$,

$$
f(x)=x,\qquad g(x)=0
$$

gives

$$
\sup_{x\in\mathbb R}|f(x)-g(x)|
=
\sup_{x\in\mathbb R}|x|
=
\infty.
$$

Instead, choose compact pieces

$$
K_1,K_2,K_3,\ldots
$$

that grow larger and larger. For $K=\mathbb R$, a natural choice is

$$
K_j=[-j,j].
$$

The metric

$$
d_0(f,g)
=
\sum_{j=1}^{\infty}2^{-j}
\frac{\sup_{x\in K_j}|f(x)-g(x)|}
{\sup_{x\in K_j}|f(x)-g(x)|+1}
$$

compares $f$ and $g$ on every compact window and combines the comparisons.

The factor $2^{-j}$ makes the infinite sum converge. The fraction $A_j/(A_j+1)$ keeps each term bounded by $1$.

So this metric measures **local uniform closeness on all compact pieces**.

## 6. Structural stability

A vector field $f$ is structurally stable if there exists $\varepsilon>0$ such that whenever

$$
\|f-g\|_{C^1}<\varepsilon,
$$

then $g$ is topologically equivalent to $f$.

In simple words:

$$
\boxed{
 f\text{ is structurally stable if every sufficiently small }C^1\text{ perturbation has the same qualitative dynamics.}
}
$$

The exact trajectories do not need to be numerically identical. Equilibria and periodic orbits may move slightly. What must remain the same is the orbit structure of the phase portrait.

## 7. Example 6.3: flow on the torus

The torus is

$$
T^2=\mathbb R^2/\mathbb Z^2.
$$

Consider the constant vector field

$$
x'=\omega_1,
\qquad
 y'=\omega_2.
$$

The solution is

$$
x(t)=x_0+\omega_1t,
\qquad
 y(t)=y_0+\omega_2t.
$$

On the torus, coordinates are taken modulo $1$. The motion is straight-line motion on a square whose opposite sides are glued together.

## 8. Rational ratio gives closed orbits

An orbit is closed if there exists $T>0$ such that the point returns to its starting point on the torus. This means

$$
x_0+\omega_1T=x_0+m,
\qquad
 y_0+\omega_2T=y_0+n,
$$

for some integers $m,n\in\mathbb Z$.

Thus

$$
\omega_1T=m,
\qquad
\omega_2T=n.
$$

Dividing gives

$$
\frac{\omega_1}{\omega_2}=\frac{m}{n}.
$$

Therefore, if

$$
\frac{\omega_1}{\omega_2}\in\mathbb Q,
$$

then the orbit closes.

## 9. Irrational ratio gives non-closed orbits

If

$$
\frac{\omega_1}{\omega_2}\notin\mathbb Q,
$$

then no integers $m,n$ can satisfy

$$
\frac{\omega_1}{\omega_2}=\frac{m}{n}.
$$

Therefore, there is no $T>0$ such that

$$
\omega_1T=m,
\qquad
\omega_2T=n.
$$

So the orbit never returns exactly to its starting point. It can come arbitrarily close, but it never closes.

Thus

$$
\boxed{
\frac{\omega_1}{\omega_2}\notin\mathbb Q
\Longrightarrow
\text{the orbit is aperiodic.}
}
$$

## 10. Why the torus example is structurally unstable

Rational and irrational numbers are arbitrarily close. A tiny perturbation can change

$$
\frac{\omega_1}{\omega_2}
$$

from rational to irrational, or from irrational to rational.

The dynamics then changes from

$$
\text{all orbits are periodic}
$$

to

$$
\text{all orbits are aperiodic and dense.}
$$

That is a major qualitative change caused by a tiny perturbation. Therefore this torus flow is structurally unstable.

## 11. Nonwandering points

A point $x\in X$ is **nonwandering** if for every neighborhood $U$ of $x$ and every time threshold $t_0$, there exists $t>t_0$ such that

$$
\phi_t(U)\cap U\neq\emptyset.
$$

This means:

$$
\boxed{\text{some part of }U\text{ eventually returns and intersects }U.}
$$

It does not require $x$ itself to return. It only requires that some point near $x$ returns near $x$.

## 12. The nonwandering set

The nonwandering set is denoted

$$
NW(\phi).
$$

It is defined by

$$
NW(\phi)=\{x\in X:x\text{ is nonwandering}\}.
$$

In a figure, a shaded region such as

$$
\phi_t(U)\cap U
$$

is not the whole nonwandering set. It is only one overlap showing that the neighborhood has returned.

## 13. Geometric and topological meaning of nonwandering

Geometrically, imagine $U$ as a small cloud of points around $x$. The flow moves the cloud:

$$
U\mapsto\phi_t(U).
$$

If the moved cloud overlaps the original cloud, then

$$
\phi_t(U)\cap U\neq\emptyset.
$$

Thus $x$ is nonwandering if every small cloud around $x$ eventually overlaps itself again, arbitrarily far in the future.

Topologically, $x$ is wandering if there exists a neighborhood $U$ such that eventually

$$
\phi_t(U)\cap U=\emptyset
$$

for all large $t$.

Equilibria, periodic orbits, and dense irrational torus orbits are nonwandering. Transient points that pass through a region and never return are wandering.

## 14. Peixoto's theorem

Peixoto's theorem gives a complete characterization of structural stability for compact two-dimensional manifolds.

A $C^1$ vector field on a compact two-dimensional differentiable manifold $M$ is structurally stable if and only if the following conditions hold.

### T1. Finitely many hyperbolic equilibria and periodic orbits

There are only finitely many equilibria and periodic orbits, and each is hyperbolic.

Hyperbolic means there are no neutral directions. A sink remains a sink under small perturbation, a source remains a source, and a saddle remains a saddle.

### T2. No saddle-to-saddle connections

A saddle-to-saddle connection occurs when an unstable manifold of one saddle connects exactly to the stable manifold of another saddle.

Such a connection is fragile because a tiny perturbation can break it. Peixoto's theorem forbids these delicate saddle connections.

### T3. The nonwandering set is simple

The theorem requires

$$
NW(\phi)
$$

to consist only of equilibria and periodic orbits. Thus the long-term recurrent behavior must be simple.

## 15. Meaning of Peixoto's theorem

In dimension $2$,

$$
\boxed{\text{structural stability}}
$$

is equivalent to

$$
\boxed{
\text{finite hyperbolic critical elements}
+
\text{no saddle connections}
+
\text{simple nonwandering set}.
}
$$

On compact orientable two-dimensional manifolds, structurally stable vector fields are open and dense in $C^1(M)$. Thus they are typical in the topological sense.

## 16. Morse-Smale systems

In higher dimensions, Peixoto's exact characterization no longer holds. The section therefore introduces Morse-Smale flows.

A flow $\phi_t$ is Morse-Smale if:

1. The number of equilibria and periodic orbits is finite, and each is hyperbolic.
2. Stable and unstable manifolds intersect only transversally.
3. The nonwandering set consists only of equilibria and periodic orbits.

The theorem says

$$
\boxed{\text{Morse--Smale systems are structurally stable.}}
$$

However, for dimensions $d\ge3$, Morse-Smale systems are not generic.

## 17. Stable and unstable manifolds

For an equilibrium or periodic orbit $p$, the stable manifold is

$$
W^s(p)=\{x:\phi_t(x)\to p\text{ as }t\to+\infty\}.
$$

It is the set of points that approach $p$ in forward time.

The unstable manifold is

$$
W^u(p)=\{x:\phi_t(x)\to p\text{ as }t\to-\infty\}.
$$

Equivalently, in forward time, points in $W^u(p)$ move away from $p$.

So

$$
W^s(p)=\text{incoming set},
\qquad
W^u(p)=\text{outgoing set}.
$$

## 18. Transversality

Suppose

$$
W^s(p)
$$

and

$$
W^u(q)
$$

intersect at a point $z$.

They intersect transversally if

$$
T_zW^s(p)+T_zW^u(q)=T_zM.
$$

In simple language:

$$
\boxed{\text{they cross cleanly, not tangentially.}}
$$

In two dimensions, a transversal intersection looks like two curves crossing with a nonzero angle. A non-transversal intersection means they only touch or are tangent.

## 19. Why Morse-Smale systems need transversality

Hyperbolicity controls local behavior near equilibria and periodic orbits. But structural stability also needs the global connections between these objects to be robust.

Transversality gives that robustness.

If stable and unstable manifolds cross transversally, the intersection survives small perturbations. If they touch tangentially, a tiny perturbation may destroy the intersection.

Thus

$$
\boxed{\text{transversal intersection is robust, while tangential intersection is fragile.}}
$$

## 20. Genericity

A property $P$ of elements of a topological space $Z$ is called **generic** if it holds on a countable intersection of open dense sets.

The word is **generic**, not genetic.

Here $Z$ is the whole space of objects being studied. For example,

$$
Z=C^1(M,TM)
$$

could be the space of all $C^1$ vector fields, or

$$
Z=\mathbb R^{d\times d}
$$

could be the space of all $d\times d$ matrices.

A property $P$ is something an element of $Z$ may or may not satisfy, such as structural stability or hyperbolicity.

The set of elements satisfying $P$ is

$$
A_P=\{z\in Z:z\text{ satisfies }P\}.
$$

## 21. Intersections, open dense sets, and genericity

If

$$
U_1,U_2,U_3,\ldots\subset Z,
$$

then

$$
\bigcap_{j=1}^{\infty}U_j
=
U_1\cap U_2\cap U_3\cap\cdots.
$$

An element belongs to the intersection if it belongs to every $U_j$.

Important distinction:

$$
z\in Z
$$

means $z$ is one object, while

$$
U_j\subset Z
$$

means $U_j$ is a set of many objects.

An open set means small perturbations stay inside the set, so openness means robustness.

A dense set means every element of $Z$ can be approximated by elements of this set, so density means the property is everywhere nearby.

A property is generic if it holds on a very large topological subset of $Z$:

$$
\boxed{
P\text{ is generic if it holds on a countable intersection of open dense subsets of }Z.
}
$$

Generic does not mean every element has the property. It means the property is typical in the topological sense.

## 22. Summary table

| Concept | Meaning | Why it matters |
|---|---|---|
| $C^k$ norm | Measures a function and derivatives up to order $k$ | Defines precise closeness |
| $C^1$ closeness | $f$ and $g$ close, and $Df$ and $Dg$ close | Needed for structural stability |
| Structural stability | Small perturbations preserve qualitative dynamics | Main theme of the section |
| Torus example | Rational slope gives closed orbits, irrational slope gives non-closed/dense orbits | Shows structural instability |
| Nonwandering point | Every neighborhood returns to itself eventually | Describes recurrent dynamics |
| Peixoto's theorem | Complete 2D characterization of structural stability | Clean result in dimension 2 |
| Transversality | Stable and unstable manifolds cross cleanly | Makes intersections robust |
| Morse-Smale system | Hyperbolic, transversal, simple nonwandering set | Gives structurally stable systems |
| Generic | Holds on a countable intersection of open dense sets | Means topologically typical |

## 23. Final core message

The whole section can be remembered as

$$
\boxed{
\text{Structural stability means qualitative dynamics survive small }C^1\text{ perturbations.}
}
$$

The three main mechanisms are

$$
\boxed{\text{Hyperbolicity gives local robustness.}}
$$

$$
\boxed{\text{Transversality gives global robustness.}}
$$

$$
\boxed{\text{A simple nonwandering set prevents complicated recurrence.}}
$$

In dimension $2$, Peixoto's theorem gives a complete picture. In higher dimensions, Morse-Smale systems are structurally stable, but they are no longer generic.

Thus:

$$
\boxed{\text{Dimension 2 is clean; higher dimensions are much harder.}}
$$
