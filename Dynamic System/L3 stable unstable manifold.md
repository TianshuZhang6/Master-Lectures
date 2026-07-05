# L3 Stable/Unstable Manifolds

Based on the lecture section on stable and unstable manifolds, including Definitions 3.1-3.7, Theorems 3.8-3.9, the Lyapunov-Perron method, the Hadamard-Perron graph transform, and Arnold's cat map.

## 1. Manifolds and atlases

A **$d$-dimensional topological manifold** $\mathcal M$ is a space that locally looks like $\mathbb R^d$.

That means: for every point $x\in\mathcal M$, there is a neighborhood $\mathcal U_\alpha$ and a coordinate map

$$
h_\alpha:\mathcal U_\alpha\to V_\alpha\subset\mathbb R^d.
$$

The pair

$$
(\mathcal U_\alpha,h_\alpha)
$$

is called a **chart**.

An **atlas** is a collection of charts covering the whole manifold:

$$
\mathcal M=\bigcup_\alpha\mathcal U_\alpha.
$$

If two charts overlap, then we can change coordinates from one chart to another using

$$
h_\alpha\circ h_\beta^{-1}.
$$

For a $C^k$ manifold, this transition map must be $C^k$:

$$
h_\alpha\circ h_\beta^{-1}\in C^k.
$$

This guarantees that calculus on the manifold is independent of the chart we choose.

## 2. Examples of manifolds

Important examples include

$$
\mathbb R^d,
\qquad
(0,1),
\qquad
(0,1)\times(0,1),
\qquad
\mathbb S^d,
\qquad
\mathbb T^d=\mathbb R^d/\mathbb Z^d.
$$

The torus formula

$$
\mathbb T^d=\mathbb R^d/\mathbb Z^d
$$

means that two points are identified if they differ by an integer vector:

$$
x\sim y
\quad\Longleftrightarrow\quad
x-y\in\mathbb Z^d.
$$

For $d=1$,

$$
\mathbb T^1=\mathbb R/\mathbb Z.
$$

This identifies

$$
0\sim1\sim2\sim-1,
$$

so the interval $[0,1]$ has endpoints glued together, giving a circle:

$$
\mathbb T^1\cong S^1.
$$

For $d=2$,

$$
\mathbb T^2=\mathbb R^2/\mathbb Z^2,
$$

which is the usual two-dimensional torus.

## 3. Tangent spaces

A tangent vector at $x\in\mathcal M$ describes an infinitesimal direction through $x$.

If

$$
\gamma:(-s,s)\to\mathcal M
$$

is a curve with

$$
\gamma(0)=x,
$$

then its tangent vector is represented by

$$
\gamma'(0).
$$

The set of all tangent vectors at $x$ is the tangent space:

$$
T_x\mathcal M.
$$

For a $d$-dimensional manifold,

$$
T_x\mathcal M\cong\mathbb R^d.
$$

For example, if $\mathcal M=S^2\subset\mathbb R^3$, then $T_xS^2$ is the tangent plane touching the sphere at $x$.

## 4. Hyperbolic equilibria for flows

Consider an ODE

$$
\dot x=f(x),
\qquad
x\in\mathbb R^d.
$$

An equilibrium point $x_*$ satisfies

$$
f(x_*)=0.
$$

The equilibrium is **hyperbolic** if the Jacobian matrix

$$
Df(x_*)
$$

has no eigenvalue with real part zero.

If

$$
\lambda=a+bi,
$$

then

$$
\operatorname{Re}(\lambda)=a.
$$

So hyperbolicity means

$$
\operatorname{Re}(\lambda)\neq0
$$

for every eigenvalue.

Stable eigenvalues satisfy

$$
\operatorname{Re}(\lambda)<0,
$$

and unstable eigenvalues satisfy

$$
\operatorname{Re}(\lambda)>0.
$$

Thus

$$
\boxed{\text{hyperbolicity means no neutral directions.}}
$$

A hyperbolic equilibrium may be stable, unstable, or a saddle. It does not mean the Jacobian is positive definite or negative definite; positive and negative definiteness are special symmetric-matrix notions.

## 5. Stable and unstable eigenspaces

Assume without loss of generality that

$$
x_*=0.
$$

Near $0$, write

$$
\dot x=Ax+R(x),
$$

where

$$
A=Df(0),
\qquad
R(x)=f(x)-Ax.
$$

Because $0$ is hyperbolic, the space splits as

$$
\mathbb R^d=E^s(0)\oplus E^u(0).
$$

The stable eigenspace is

$$
E^s(0)=\{y\in\mathbb R^d:e^{tA}y\to0\text{ as }t\to+\infty\}.
$$

The unstable eigenspace is

$$
E^u(0)=\{y\in\mathbb R^d:e^{tA}y\to0\text{ as }t\to-\infty\}.
$$

Every nearby point decomposes uniquely as

$$
x=x_s+x_u,
$$

where

$$
x_s\in E^s,
\qquad
x_u\in E^u.
$$

There are projections

$$
P_s:\mathbb R^d\to E^s,
\qquad
P_u:\mathbb R^d\to E^u.
$$

The nonlinear remainder also splits:

$$
R(x)=R_s(x)+R_u(x),
$$

where

$$
R_s(x)=P_sR(x),
\qquad
R_u(x)=P_uR(x).
$$

## 6. Stable and unstable manifold theorem for flows

Near a hyperbolic equilibrium $x_*$, there exist local stable and unstable manifolds:

$$
W^s_{\mathrm{loc}}(x_*),
\qquad
W^u_{\mathrm{loc}}(x_*).
$$

The local stable manifold is

$$
W^s_{\mathrm{loc}}(x_*)
=
\{y\in U:\phi_t(y)\to x_*\text{ as }t\to+\infty,
\ \phi_t(y)\in U\text{ for all }t\ge0\}.
$$

The local unstable manifold is

$$
W^u_{\mathrm{loc}}(x_*)
=
\{y\in U:\phi_t(y)\to x_*\text{ as }t\to-\infty,
\ \phi_t(y)\in U\text{ for all }t\le0\}.
$$

Here $\phi_t(y)$ is the flow of the ODE.

The stable manifold is tangent to the stable eigenspace:

$$
T_{x_*}W^s_{\mathrm{loc}}(x_*)=E^s(x_*).
$$

The unstable manifold is tangent to the unstable eigenspace:

$$
T_{x_*}W^u_{\mathrm{loc}}(x_*)=E^u(x_*).
$$

At the equilibrium, the nonlinear manifold has the same first-order direction as the corresponding linear eigenspace.

## 7. Stable manifold as a graph

Near $0$, the stable manifold can be written as a graph over the stable eigenspace:

$$
W^s_{\mathrm{loc}}(0)
=
\{x=x_s+x_u:x_u=h(x_s)\}.
$$

Here

$$
h:E^s\to E^u.
$$

Thus, for every stable coordinate $x_s$, the unstable coordinate $x_u$ is not free. It must equal

$$
x_u=h(x_s).
$$

Equivalently,

$$
W^s_{\mathrm{loc}}(0)
=
\{x_s+h(x_s):x_s\in E^s\}.
$$

The function $h$ is not arbitrary. It is chosen so that the corresponding solution stays near $0$ and converges to $0$ as

$$
t\to+\infty.
$$

Usually,

$$
h(0)=0,
\qquad
Dh(0)=0.
$$

This means the stable manifold passes through $0$ and is tangent to $E^s$.

## 8. Forward-time bounded solutions

For a point to lie on the stable manifold, its orbit must not escape forward in time. Therefore the proof looks for solutions $x(t)$ that are bounded for

$$
t\ge0.
$$

The unstable component normally grows forward in time. Therefore the initial unstable coordinate must be chosen carefully to cancel that growth.

The variation-of-constants formula is

$$
x(t)=e^{tA}x_0+\int_0^t e^{(t-r)A}R(x(r))\,dr.
$$

After projecting to stable and unstable directions, one obtains the condition

$$
x_u
=
-\int_0^\infty e^{-rA}R_u(x(r))\,dr.
$$

This formula selects the unique unstable component needed to keep the orbit bounded forward in time.

Plugging this back gives

$$
x(t)
=
e^{tA}x_s
+
\int_0^t e^{(t-r)A}R_s(x(r))\,dr
-
\int_t^\infty e^{(t-r)A}R_u(x(r))\,dr.
$$

The first two terms are controlled by stable decay. The last term is controlled because it evolves the unstable direction backward in time, where it decays.

## 9. Lyapunov-Perron fixed point method

The proof rewrites the equation as a fixed-point problem.

Define

$$
P(t)=
\begin{cases}
P_s, & t>0,\\
-P_u, & t\le0.
\end{cases}
$$

Then the integral equation becomes

$$
x(t)=K(x)(t),
$$

where

$$
K(x)(t)
=
e^{tA}x_s
+
\int_0^\infty e^{(t-r)A}P(t-r)R(x(r))\,dr.
$$

The proof works in the Banach space

$$
C_b([0,\infty),\mathbb R^d)
$$

with norm

$$
\|x\|=\sup_{t\ge0}|x(t)|.
$$

Because $R$ is small near $0$, the operator $K$ becomes a contraction on a sufficiently small neighborhood.

By the Banach fixed point theorem, $K$ has a unique fixed point. That fixed point is written as

$$
\psi(t,x_s).
$$

Then the stable graph function is defined by

$$
h(y)=P_u\psi(0,y).
$$

So the stable manifold is

$$
W^s_{\mathrm{loc}}(0)=\{y+h(y):y\in E^s\}.
$$

## 10. Why $h$ is unique

For each $x_s\in E^s$, the proof finds one special bounded solution

$$
\psi(t,x_s).
$$

Then

$$
h(x_s)=P_u\psi(0,x_s).
$$

Since the fixed point is unique, $h(x_s)$ is unique.

Many functions can define graphs over $E^s$, but only one graph gives the stable manifold.

## 11. Differentiability of the stable manifold

To prove differentiability, the proof studies how

$$
\psi(t,x_s)
$$

depends on $x_s$.

Let

$$
\eta(t,x_s)=D_{x_s}\psi(t,x_s).
$$

Differentiating the fixed-point equation gives

$$
\eta(t,x_s)
=
e^{tA}P_s
+
\int_0^\infty e^{(t-r)A}P(t-r)\eta(r,x_s)\nabla R(\psi(r,x_s))\,dr.
$$

The term

$$
\nabla R(\psi(r,x_s))
$$

appears because we differentiate

$$
R(\psi(r,x_s))
$$

with respect to $x_s$, not with respect to time $r$.

By the chain rule,

$$
D_{x_s}R(\psi(r,x_s))
=
DR(\psi(r,x_s))D_{x_s}\psi(r,x_s).
$$

The same contraction argument gives existence and continuity of $\eta$, so

$$
\psi\in C^1.
$$

Therefore the stable manifold is differentiable. If $f\in C^k$, then stable and unstable manifolds are $C^k$. If $f\in C^\omega$, they are analytic.

## 12. Global stable and unstable manifolds

The theorem first gives local manifolds near $x_*$. The global stable and unstable manifolds are obtained by flowing the local manifolds:

$$
W^{s,u}(x_*)
=
\bigcup_{t\in\mathbb R}\phi_t(W^{s,u}_{\mathrm{loc}}(x_*)).
$$

The manifolds are invariant. For example,

$$
x\in W^s(x_*)
\quad\Longrightarrow\quad
\phi_t(x)\in W^s(x_*)
$$

for every time for which the expression is defined.

## 13. Stable and unstable manifolds for maps

For maps, we study

$$
x\mapsto g(x).
$$

A fixed point satisfies

$$
g(x_*)=x_*.
$$

For maps, hyperbolicity is defined using the eigenvalues of

$$
Dg(x_*).
$$

These eigenvalues are called **multipliers**. A fixed point is hyperbolic if no multiplier satisfies

$$
|\mu|=1.
$$

For maps,

$$
|\mu|<1
$$

means stable direction, because repeated iteration contracts:

$$
\mu^n\to0.
$$

Also,

$$
|\mu|>1
$$

means unstable direction.

The local stable manifold is

$$
W^s_{\mathrm{loc}}(x_*)
=
\{y\in U:g^n(y)\to x_*\text{ as }n\to+\infty,
\ g^n(y)\in U\text{ for all }n\ge0\}.
$$

The local unstable manifold is

$$
W^u_{\mathrm{loc}}(x_*)
=
\{y\in U:g^n(y)\to x_*\text{ as }n\to-\infty,
\ g^n(y)\in U\text{ for all }n\le0\}.
$$

Because $g$ is a diffeomorphism, negative iterates are defined.

## 14. Hadamard-Perron graph transform

The Hadamard-Perron method constructs the stable manifold as a fixed graph.

Assume

$$
x_*=0.
$$

Let

$$
B^s_\varepsilon=\{y\in E^s(0):|y|\le\varepsilon\}.
$$

Consider Lipschitz maps

$$
h:B^s_\varepsilon\to E^u(0)
$$

with

$$
h(0)=0.
$$

Each such $h$ gives a graph:

$$
\operatorname{graph}(h)
=
\{y+h(y):y\in B^s_\varepsilon\}.
$$

This graph is a candidate stable manifold.

The graph transform is an operator

$$
G:\mathcal L(L,\varepsilon)\to\mathcal L(L,\varepsilon).
$$

It is defined by

$$
G(\varphi)=\widetilde\varphi,
$$

where

$$
g^{-1}(\operatorname{graph}(\varphi))
=
\operatorname{graph}(\widetilde\varphi).
$$

The goal is to find a fixed point:

$$
G(\varphi)=\varphi.
$$

Then

$$
g^{-1}(\operatorname{graph}(\varphi))
=
\operatorname{graph}(\varphi),
$$

so the graph is invariant. That invariant graph is the stable manifold.

## 15. Why use $g^{-1}$ in the graph transform?

The stable manifold concerns future behavior:

$$
g^n(x)\to0
\quad\text{as }n\to+\infty.
$$

To find stable points now, we ask which points land on the candidate stable graph after one application of $g$. That set is

$$
g^{-1}(\operatorname{graph}(\varphi)).
$$

Using $g^{-1}$ is also useful because unstable directions contract backward in time, making the graph transform a contraction.

So

$$
g^{-1}(\operatorname{graph}(\varphi))
=
\{x:g(x)\in\operatorname{graph}(\varphi)\}.
$$

The graph is not the whole domain of $g^{-1}$. It is a subset of phase space, and $g^{-1}$ is applied to that subset.

## 16. Arnold's cat map

The lecture gives the linear map

$$
g(x)=Ax,
$$

where

$$
A=
\begin{pmatrix}
2&1\\
1&1
\end{pmatrix}.
$$

Thus

$$
g(x_1,x_2)=(2x_1+x_2,\ x_1+x_2).
$$

This map is considered on the torus

$$
\mathbb T^2=\mathbb R^2/\mathbb Z^2.
$$

So the torus map is

$$
g(x_1,x_2)=(2x_1+x_2,\ x_1+x_2)\pmod 1.
$$

The matrix $A$ preserves the integer lattice because it has integer entries:

$$
A\mathbb Z^2\subseteq\mathbb Z^2.
$$

Therefore, if

$$
x\sim y,
$$

then

$$
Ax\sim Ay.
$$

So the map is well-defined on the torus.

Also,

$$
\det A=1,
$$

so $A$ is invertible and $A^{-1}$ also has integer entries. Therefore the map is a toral automorphism.

## 17. Meaning of the cat map picture

The unit square

$$
[0,1]^2
$$

is a fundamental domain for the torus.

The matrix sends the corners as follows:

$$
A(0,0)=(0,0),
$$

$$
A(1,0)=(2,1),
$$

$$
A(0,1)=(1,1),
$$

$$
A(1,1)=(3,2).
$$

So the unit square becomes a slanted parallelogram. Then, because we are on the torus, points differing by integer vectors are identified, and the parallelogram is wrapped back into the unit square.

The process is

$$
\boxed{\text{stretch} + \text{shear} + \text{wrap modulo }1.}
$$

Example:

$$
x=(0.8,0.7).
$$

Then

$$
Ax=(2.3,1.5).
$$

Modulo $1$,

$$
(2.3,1.5)\sim(0.3,0.5).
$$

So

$$
g(0.8,0.7)=(0.3,0.5).
$$

## 18. Hyperbolicity of the cat map

The derivative is constant:

$$
Dg(x)=A
$$

for every point on the torus.

The eigenvalues of $A$ are

$$
\lambda_u=\frac{3+\sqrt5}{2}>1,
$$

and

$$
\lambda_s=\frac{3-\sqrt5}{2}<1.
$$

Numerically,

$$
\lambda_u\approx2.618,
\qquad
\lambda_s\approx0.382.
$$

So one direction expands and one direction contracts.

The unstable eigenspace corresponds to $\lambda_u$, and the stable eigenspace corresponds to $\lambda_s$.

Because

$$
Dg(x)=A
$$

for every point, the same stable and unstable directions exist near every point of the torus. This is why Arnold's cat map is a hyperbolic toral automorphism.

It has hyperbolicity not only at the fixed point $(0,0)$, but globally across the torus.

## 19. Why the cat map is interesting

In $\mathbb R^2$, the map

$$
x\mapsto Ax
$$

is just a linear saddle map.

But on the torus, the map cannot escape to infinity. Everything is wrapped back modulo $\mathbb Z^2$.

So the map repeatedly performs

$$
\text{stretch}\to\text{contract}\to\text{wrap back}.
$$

This creates complicated global behavior from a simple linear formula.

The local behavior is linear and hyperbolic, but the global torus wrapping produces mixing and chaotic dynamics.

Arnold's cat map is important because it is one of the simplest examples of a uniformly hyperbolic dynamical system.

## 20. Core ideas to remember

The main ideas are:

$$
\boxed{\text{A manifold is a space locally like }\mathbb R^d.}
$$

$$
\boxed{\text{Hyperbolicity means no neutral directions.}}
$$

For flows,

$$
\operatorname{Re}(\lambda)\neq0.
$$

For maps,

$$
|\mu|\neq1.
$$

Stable directions contract forward in time. Unstable directions expand forward in time but contract backward in time.

The stable manifold is the special set of points whose unstable coordinate is chosen correctly:

$$
x_u=h(x_s).
$$

The function $h$ is not arbitrary. It is determined by the bounded-solution condition.

The stable and unstable manifolds are tangent to the corresponding eigenspaces:

$$
T_{x_*}W^s=E^s,
\qquad
T_{x_*}W^u=E^u.
$$

The Lyapunov-Perron method constructs the stable manifold by solving a fixed-point equation for bounded forward-time solutions.

The Hadamard-Perron method constructs the stable manifold as a fixed point of a graph transform.

Arnold's cat map is a simple linear map on the torus that stretches, contracts, and wraps, giving a classical example of hyperbolic dynamics.
