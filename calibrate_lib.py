import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

DEPTH_OFFSET = 0


def select_rectangle(data, title):
    _, ax = plt.subplots()

    colors = data[:, 3:] / 255

    ax.scatter(data[:, 0], data[:, 2], c=colors, marker="o", s=10)
    ax.set_title(title)
    ax.axis("equal")

    result = {"rectangle_vertices": None, "selected_points": None}

    # Function to handle selection
    def on_select(eclick, erelease):
        # Get the coordinates of the rectangle
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])

        result["rectangle_vertices"] = [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
        ]

        # Find points inside the rectangle
        result["selected_points"] = data[
            (data[:, 0] >= x_min)
            & (data[:, 0] <= x_max)
            & (data[:, 2] >= y_min)
            & (data[:, 2] <= y_max)
        ]

    # Initialize RectangleSelector
    selector = RectangleSelector(
        ax,
        onselect=on_select,
        useblit=True,
        button=[1],
        interactive=True,
    )

    plt.show()

    if result["rectangle_vertices"] is None:
        raise ValueError("No rectangle was selected.")

    return result["rectangle_vertices"], np.array(result["selected_points"])


def calibrate_depth(data):
    _, depth_points = select_rectangle(data, "Select A Rectangle of Depth Points")
    return np.mean(depth_points[:, 1]) + DEPTH_OFFSET


def calibrate_hoop_coords(data):
    """
    [x_min, z_min], [x_max, z_max]
    """
    vertices, _ = select_rectangle(data, "Select The Basketball Hoop Movable Area")
    return vertices[0], vertices[2]
