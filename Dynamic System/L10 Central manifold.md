# L10 Central Manifold: Linearization and Eigenvalues

## 1. The correct two-step procedure

Strictly speaking, the procedure has two steps:

$$
\boxed{
\text{First compute the Jacobi matrix at the equilibrium, then compute the eigenvalues of that linearized matrix.}
}
$$

That is,

$$
A=DF(x_*,y_*),
$$

and then solve

$$
\det(A-\lambda I)=0.
$$

This is especially important in multidimensional systems. We do not usually classify the equilibrium by looking at only one derivative. We classify it using the eigenvalues of the full linearization matrix.

## 2. The example system

Consider the system

$$
x'=y-1,
$$

$$
y'=-(y-1)+px^2+x(y-1).
$$

The equilibrium is

$$
(x_*,y_*)=(0,1).
$$

The Jacobi matrix is

$$
DF(x,y)=
\begin{pmatrix}
0 & 1\\
2px+(y-1) & -1+x
\end{pmatrix}.
$$

Substituting the equilibrium $(0,1)$ gives

$$
A=DF(0,1)=
\begin{pmatrix}
0 & 1\\
0 & -1
\end{pmatrix}.
$$

This step means: evaluate all partial derivatives at the equilibrium point.

## 3. Eigenvalues of the linearization

Now compute the eigenvalues of $A$:

$$
\det(A-\lambda I)=0.
$$

Here

$$
A-\lambda I=
\begin{pmatrix}
-\lambda & 1\\
0 & -1-\lambda
\end{pmatrix}.
$$

Because this matrix is upper triangular, its determinant is the product of the diagonal entries:

$$
\det(A-\lambda I)=(-\lambda)(-1-\lambda).
$$

So

$$
(-\lambda)(-1-\lambda)=0.
$$

Therefore,

$$
\lambda=0,
\qquad
\lambda=-1.
$$

## 4. Interpretation

The correct conclusion is

$$
\boxed{
\text{We do not look at just one derivative value; we compute the eigenvalues of the Jacobi matrix at the equilibrium.}
}
$$

In a multidimensional system, the entries of the Jacobi matrix are the partial derivatives evaluated at the equilibrium point. But the stability information comes from the eigenvalues of the whole matrix.

For a one-dimensional system

$$
x'=f(x),
$$

the linearization is

$$
X'=f'(x_*)X.
$$

So the only eigenvalue is

$$
\lambda=f'(x_*).
$$

For multidimensional systems, however, the procedure is:

$$
\boxed{
\text{form the full Jacobi matrix first, then compute its eigenvalues.}
}
$$

## 5. Connection with center manifolds

In this example, the eigenvalues are

$$
0
\quad\text{and}\quad
-1.
$$

The eigenvalue $-1$ is stable because it has negative real part. The eigenvalue $0$ is a center direction because it has real part zero.

Thus the equilibrium is **non-hyperbolic**:

$$
\boxed{
\text{there is a zero eigenvalue, so linearization alone does not fully determine the nonlinear dynamics.}
}
$$

This is exactly the situation where center manifold theory becomes useful. The stable direction decays, while the center direction requires higher-order nonlinear analysis.
