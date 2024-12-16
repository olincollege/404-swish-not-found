import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(file_path):
    # Include columns for RGB values
    return pd.read_csv(file_path, header=None, names=["x", "y", "z", "r", "g", "b"])


def plot_sphere(ax, x_c, y_c, z_c, radius):
    # Create a meshgrid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = x_c + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = y_c + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = z_c + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere, color="lightblue", alpha=0.6, rstride=5, cstride=5
    )


# 3D plot function with data sampling option
def plot_3d(
    data, xlim=None, ylim=None, zlim=None, percent=100, trajectory=None, spheres=None
):
    # Ensure percent is between 0 and 100
    if percent < 0 or percent > 100:
        raise ValueError("Percent must be between 0 and 100.")

    # Randomly sample the specified percentage of data
    if percent < 100:
        data = data.sample(frac=percent / 100, random_state=42).reset_index(drop=True)

    # Create RGB colors by normalizing the values to [0, 1]
    colors = data[["r", "g", "b"]].values / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with RGB colors
    ax.scatter(data["x"], data["y"], data["z"], c=colors, marker="o", s=10)
    ax.scatter(0, 0, 0, c="r", marker="o", s=100)  # Plot a red dot at the origin
    if trajectory is not None:
        ax.plot(trajectory["x"], trajectory["y"], trajectory["z"], c="r")

    if spheres is not None:
        for i in range(spheres.shape[0]):
            plot_sphere(ax, spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3])

    # Setting limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # ax.axis("equal")

    # Labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    plt.show()


# Usage example
if __name__ == "__main__":
    add_reference = False
    all_frames = True
    add_trajectory = True
    add_spheres = True

    ref_path = "points/reference.csv"
    file_path = "points/points_76.csv"
    traj_path = "trajectories/4.csv"
    sphere_path = "spheres.csv"

    percent = 100  # Choose what percent of data to plot

    if all_frames:
        file_path = "points/points_0.csv"

    data = load_csv(file_path)

    if all_frames:
        for i in range(99):
            file_path = f"points/points_{i}.csv"
            data = pd.concat([data, load_csv(file_path)])

    if add_reference:
        ref = load_csv(ref_path)
        data = pd.concat([data, ref])

    sphere_points = None
    if add_spheres:
        sphere_points = np.loadtxt(sphere_path, delimiter=",")

    trajectory = None

    if add_trajectory:
        trajectory = pd.read_csv(traj_path, header=None, names=["x", "y", "z"])

    # Optional: Set your limits here
    if all_frames or add_trajectory:
        xlim = (-1.500, 1.500)
        ylim = (1.000, 4.000)
        zlim = (-1.000, 2.000)
    else:
        xlim = None
        ylim = None
        zlim = None

    # Plot with specified limits and percent sample
    plot_3d(
        data,
        trajectory=trajectory,
        spheres=sphere_points,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
