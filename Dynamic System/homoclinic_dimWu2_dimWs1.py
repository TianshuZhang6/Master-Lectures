import numpy as np
import matplotlib.pyplot as plt

# Example:
#   x' = -y
#   y' = -x + x^3
#   z' =  z
#
# At the origin:
#   A = [[0,-1,0],
#        [-1,0,0],
#        [0, 0,1]]
# eigenvalues: 1, 1, -1
# Hence dim W^u(0)=2, dim W^s(0)=1.
#
# In the (x,y)-plane, H(x,y)=1/2 y^2 - 1/2 x^2 + 1/4 x^4 is conserved.
# The homoclinic loop(s) satisfy H=0:
#   y = ± sqrt(x^2 - x^4/2), |x| <= sqrt(2).
#
# For this explicit model:
#   W^s(0) = {(x,y,0): (x,y) on the planar stable separatrix}  (1D)
#   W^u(0) = {(x,y,z): (x,y) on the planar unstable separatrix, z in R}  (2D)

def main():
    sqrt2 = np.sqrt(2)

    # Surface for W^u(0): extrude the planar separatrix in z-direction
    x = np.linspace(-sqrt2, sqrt2, 320)
    z = np.linspace(-1.5, 1.5, 140)
    X, Z = np.meshgrid(x, z)
    rad = np.maximum(X**2 - 0.5 * X**4, 0.0)
    Y_plus = np.sqrt(rad)
    Y_minus = -np.sqrt(rad)

    # 1D stable manifold W^s(0): the planar homoclinic curves in z=0
    t = np.linspace(-8, 8, 1500)
    sech = 1 / np.cosh(t)

    # Right homoclinic loop
    x_right = sqrt2 * sech
    y_right = sqrt2 * sech * np.tanh(t)
    z_right = np.zeros_like(t)

    # Left homoclinic loop
    x_left = -sqrt2 * sech
    y_left = -sqrt2 * sech * np.tanh(t)
    z_left = np.zeros_like(t)

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    # Plot unstable manifold surface W^u(0)
    ax.plot_surface(
        X, Y_plus, Z,
        color="royalblue", alpha=0.25, linewidth=0, antialiased=True
    )
    ax.plot_surface(
        X, Y_minus, Z,
        color="royalblue", alpha=0.25, linewidth=0, antialiased=True
    )

    # Plot stable manifold curve W^s(0)
    ax.plot(x_right, y_right, z_right, color="crimson", linewidth=3)
    ax.plot(x_left, y_left, z_left, color="crimson", linewidth=3)

    # Equilibrium
    ax.scatter([0], [0], [0], color="black", s=50)

    # Labels on the picture
    ax.text(1.02, 0.60, 0.95, r"$W^u(0)$", color="royalblue", fontsize=14)
    ax.text(-1.15, -0.10, 0.06, r"$W^s(0)$", color="crimson", fontsize=14)
    ax.text(0.05, 0.05, 0.07, r"$0$", color="black", fontsize=12)

    # Mark one homoclinic circle explicitly
    ax.text(0.95, 0.28, 0.02, "homoclinic circle", color="darkgreen", fontsize=13)
    ax.plot([0.88, 1.08], [0.22, 0.48], [0.02, 0.02], color="darkgreen", linewidth=2)

    # Axes / view
    ax.set_title("Example with dim $W^u(0)=2$ and dim $W^s(0)=1$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=23, azim=-58)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
