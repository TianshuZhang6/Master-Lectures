import numpy as np
import matplotlib.pyplot as plt

# Example:
#   x' = y
#   y' = x - x^3
#
# The origin (0,0) is a saddle.
# Eigenvalues: 1 and -1
# Unstable direction: y = x
# Stable direction:   y = -x
#
# A right homoclinic loop is
#   x(t) = sqrt(2) sech(t),
#   y(t) = -sqrt(2) sech(t) tanh(t).
#
# For t < 0, the orbit is on the unstable branch W^u(0)
# For t > 0, the orbit is on the stable branch W^s(0)

def main():
    # Vector field grid
    xg = np.linspace(-0.3, 1.65, 31)
    yg = np.linspace(-1.05, 1.05, 31)
    X, Y = np.meshgrid(xg, yg)
    U = Y
    V = X - X**3

    # Homoclinic loop (right loop only)
    t = np.linspace(-8, 8, 1600)
    sech = 1 / np.cosh(t)
    xh = np.sqrt(2) * sech
    yh = -np.sqrt(2) * sech * np.tanh(t)

    # Split into unstable/stable pieces
    mask_u = t <= 0
    mask_s = t >= 0

    # Local tangent directions at the origin
    eps = 0.22
    u_line = np.linspace(0, eps, 30)
    s_line = np.linspace(0, eps, 30)

    fig, ax = plt.subplots(figsize=(8, 6.8))

    # Background phase portrait
    ax.streamplot(X, Y, U, V, density=1.0, color="0.80", linewidth=0.8, arrowsize=0.8)

    # Outgoing unstable half (red), incoming stable half (blue)
    ax.plot(xh[mask_u], yh[mask_u], color="red", linewidth=3,
            label=r"unstable manifold branch $W^u(0)$")
    ax.plot(xh[mask_s], yh[mask_s], color="blue", linewidth=3,
            label=r"stable manifold branch $W^s(0)$")

    # Local tangent directions
    ax.plot(u_line, u_line, color="red", linestyle="--", linewidth=2)
    ax.plot(s_line, -s_line, color="blue", linestyle="--", linewidth=2)

    # Mark special points
    ax.scatter([0], [0], color="black", s=40, zorder=5)
    ax.scatter([np.sqrt(2)], [0], color="black", s=25, zorder=5)

    # Direction arrows along the loop
    i1, i2 = 250, 340
    ax.annotate(
        "",
        xy=(xh[i2], yh[i2]),
        xytext=(xh[i1], yh[i1]),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    j1, j2 = 1230, 1330
    ax.annotate(
        "",
        xy=(xh[j2], yh[j2]),
        xytext=(xh[j1], yh[j1]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )

    # Labels
    ax.annotate(r"$0$", xy=(0, 0), xytext=(-0.08, -0.08), fontsize=12)
    ax.annotate(
        "homoclinic circle",
        xy=(1.18, 0.56),
        xytext=(1.20, 0.82),
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    ax.annotate(
        r"$W^u(0)$",
        xy=(0.52, 0.44),
        xytext=(0.18, 0.72),
        color="red",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.4),
    )
    ax.annotate(
        r"$W^s(0)$",
        xy=(0.58, -0.44),
        xytext=(0.18, -0.80),
        color="blue",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.4),
    )

    ax.set_title(r"Planar homoclinic loop for $\dot x = y,\ \dot y = x-x^3$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.15, 1.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
