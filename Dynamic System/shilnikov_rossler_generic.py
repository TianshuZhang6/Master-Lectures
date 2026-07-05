
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a = 0.1798
b = 0.2
c = 10.3084

def rossler(t, Y):
    x, y, z = Y
    return np.array([
        -(y + z),
        x + a * y,
        b + z * (x - c)
    ])

D = np.sqrt(c * c - 4 * a * b)
p = np.array([
    (c - D) / 2,
    -(c - D) / (2 * a),
    (c - D) / (2 * a),
])

J = np.array([
    [0, -1, -1],
    [1,  a,  0],
    [p[2], 0, p[0] - c]
], dtype=float)

eigvals, eigvecs = np.linalg.eig(J)

idx_s = np.argmin(eigvals.real)
v_s = eigvecs[:, idx_s].real
v_s = v_s / np.linalg.norm(v_s)

idx_u = [i for i, lam in enumerate(eigvals) if lam.real > 0 and abs(lam.imag) > 1e-10][0]
v_complex = eigvecs[:, idx_u]
u1 = v_complex.real
u1 = u1 / np.linalg.norm(u1)
u2 = v_complex.imag - np.dot(v_complex.imag, u1) * u1
u2 = u2 / np.linalg.norm(u2)

# Local 1D stable manifold
s_vals = np.linspace(-4.0, 4.0, 200)
Ws_local = p[:, None] + v_s[:, None] * s_vals[None, :]

# 2D unstable manifold patch
n_theta = 56
n_t = 170
t_u = np.linspace(0.0, 160.0, n_t)
thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
eps_u = 1e-5

X = np.full((n_t, n_theta), np.nan)
Y = np.full((n_t, n_theta), np.nan)
Z = np.full((n_t, n_theta), np.nan)

for j, theta in enumerate(thetas):
    y0 = p + eps_u * (np.cos(theta) * u1 + np.sin(theta) * u2)
    sol = solve_ivp(
        rossler,
        [0.0, t_u[-1]],
        y0,
        t_eval=t_u,
        max_step=0.25,
        rtol=1e-6,
        atol=1e-9
    )
    X[:, j] = sol.y[0]
    Y[:, j] = sol.y[1]
    Z[:, j] = sol.y[2]

# A tuned homoclinic-like orbit for visualization
theta_h = 3.9269908169872414
sol_h = solve_ivp(
    rossler,
    [0.0, 260.0],
    p + eps_u * (np.cos(theta_h) * u1 + np.sin(theta_h) * u2),
    max_step=0.1,
    rtol=1e-7,
    atol=1e-10
)

fig = plt.figure(figsize=(8.5, 7.4))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    X, Y, Z,
    color='royalblue',
    alpha=0.22,
    linewidth=0,
    antialiased=True
)

ax.plot(
    Ws_local[0], Ws_local[1], Ws_local[2],
    color='crimson',
    linewidth=3
)

ax.plot(
    sol_h.y[0], sol_h.y[1], sol_h.y[2],
    color='darkgreen',
    linewidth=3
)

ax.scatter([p[0]], [p[1]], [p[2]], color='black', s=55)

ax.text(p[0] + 4.0, p[1] + 1.0, p[2] + 4.5, r'$W^s_{\mathrm{loc}}(p)$', color='crimson', fontsize=13)
ax.text(4.0, -3.0, 3.5, r'$W^u(p)$', color='royalblue', fontsize=13)
mid = len(sol_h.t) // 3
ax.text(sol_h.y[0, mid] - 8.0, sol_h.y[1, mid] + 2.0, sol_h.y[2, mid] + 5.0,
        'homoclinic orbit', color='darkgreen', fontsize=13)

ax.set_title("Rössler Shilnikov-type numerical example\n"
             r"$\dim W^u(p)=2,\ \dim W^s(p)=1$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=22, azim=-60)

ax.set_xlim(-17, 20)
ax.set_ylim(-20, 15)
ax.set_zlim(0, 55)

plt.tight_layout()
plt.show()

print("equilibrium p =", p)
print("eigenvalues =", eigvals)
