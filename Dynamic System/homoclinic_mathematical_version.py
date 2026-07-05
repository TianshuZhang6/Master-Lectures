import numpy as np
import matplotlib.pyplot as plt

# System:
#   x' = y
#   y' = x - x^3
#   z' = -z
#
# At the origin, the linearization matrix is
#   A = [[0, 1, 0],
#        [1, 0, 0],
#        [0, 0,-1]]
# with eigenvalues 1, -1, -1.
#
# Unstable eigenspace:
#   E^u = span{(1,1,0)}
#
# Stable eigenspace:
#   E^s = span{(1,-1,0), (0,0,1)}
#
# In the (x,y)-plane, the Hamiltonian is
#   H(x,y) = 1/2 y^2 - 1/2 x^2 + 1/4 x^4.
# The homoclinic separatrix is H=0:
#   y = ± sqrt(x^2 - x^4/2), |x| <= sqrt(2).
#
# Therefore:
#   W^u(0) = {(x,y,0): (x,y) lies on the planar unstable manifold}
#   W^s(0) = {(x,y,z): (x,y) lies on the planar stable manifold, z arbitrary}

def main():
    sqrt2 = np.sqrt(2)

    # Stable manifold sheets
    x = np.linspace(-sqrt2, sqrt2, 320)
    z = np.linspace(-1.5, 1.5, 120)
    X, Z = np.meshgrid(x, z)
    rad = np.maximum(X**2 - 0.5 * X**4, 0.0)
    Y_plus = np.sqrt(rad)
    Y_minus = -np.sqrt(rad)

    # Unstable manifold / homoclinic loop in z=0
    t = np.linspace(-8, 8, 1400)
    sech = 1 / np.cosh(t)

    x_right = sqrt2 * sech
    y_right = -sqrt2 * sech * np.tanh(t)
    z_right = np.zeros_like(t)

    x_left = -sqrt2 * sech
    y_left = sqrt2 * sech * np.tanh(t)
    z_left = np.zeros_like(t)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # W^s(0): two sheets
    ax.plot_surface(X, Y_plus, Z, alpha=0.22, linewidth=0, antialiased=True)
    ax.plot_surface(X, Y_minus, Z, alpha=0.22, linewidth=0, antialiased=True)

    # W^u(0): two branches
    ax.plot(x_right, y_right, z_right, linewidth=3)
    ax.plot(x_left, y_left, z_left, linewidth=3)

    # Equilibrium at the origin
    ax.scatter([0], [0], [0], s=60)

    # Local tangent directions
    eps = 0.40
    u = np.linspace(0, eps, 20)
    s = np.linspace(-eps, eps, 20)

    ax.plot(u, u, 0*u, linestyle="--", linewidth=2)          # E^u
    ax.plot(s, -s, 0*s, linestyle="--", linewidth=2)         # first stable direction
    ax.plot([0, 0], [0, 0], [-eps, eps], linestyle="--", linewidth=2)  # z stable direction

    # Labels
    ax.text(0.92, 0.75, 0.05, r"$W^u(0)$", fontsize=13)
    ax.text(-1.20, 0.72, 1.05, r"$W^s(0)$", fontsize=13)
    ax.text(0.05, 0.05, 0.07, r"$0$", fontsize=12)
    ax.text(0.27, 0.30, 0.02, r"$E^u=\mathrm{span}\{(1,1,0)\}$", fontsize=11)
    ax.text(0.18, -0.48, 0.10, r"$E^s_1=\mathrm{span}\{(1,-1,0)\}$", fontsize=11)
    ax.text(0.04, 0.04, 0.48, r"$E^s_2=\mathrm{span}\{(0,0,1)\}$", fontsize=11)

    ax.set_title("Homoclinic loop with stable/unstable manifolds", pad=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=23, azim=-57)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
