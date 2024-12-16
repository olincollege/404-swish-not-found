"""
Library to extract and model ball trajectory
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

HALF_GRAVITY = 4.9
BALL_LOWER_RGB = (70, 15, 10)
BALL_UPPER_RGB = (200, 120, 100)
POINTS_IN_AREA_DIST = 0.15


def fit_sphere(xyz_points):
    """
    Fits a sphere to 3d point cloud

    Args:
        xyz_points: (n, 3) numpy array of points

    Returns:
        List with x center, y center, z center and radius
    """
    A = np.hstack((-2 * xyz_points, np.ones((xyz_points.shape[0], 1))))
    b = np.sum(-1 * np.square(xyz_points), axis=1)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(np.sum(np.square(x[:3])) - x[3])

    return [x[0], x[1], x[2], r]


def _fit_linear(inputs, outputs):
    """
    Fits a line to data

    Args:
        inputs: (n, 1) numpy array of input values
        outputs: (n, 1) numpy array of outputs values

    Returns:
        Slope and bias as floats
    """
    assert inputs.shape[1] == 1
    assert outputs.shape[1] == 1

    A = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
    b = outputs
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return x[0], x[1]


def _fit_quadratic(inputs, outputs):
    """
    Fits a quadratic to data

    Args:
        inputs: (n, 1) numpy array of input values
        outputs: (n, 1) numpy array of outputs values

    Returns:
        Quadratic coeff, linear coeff, and bias as floats
    """
    assert inputs.shape[1] == 1
    assert outputs.shape[1] == 1

    A = np.hstack((np.square(inputs), inputs, np.ones((inputs.shape[0], 1))))
    b = outputs
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return x[0], x[1], x[2]


def fit_trajectory_regression(xyz_points, times):
    """
    Fits a trajectory to a ball arc

    Args:
        xyz_points: (n, 3) numpy array of points
        times: (n, 1) numpy array of times

    Returns:
        Function with time (int) as input and (1, 3) numpy array as output
    """
    times = np.array(times).reshape(-1, 1)
    x_slope, x_intercept = _fit_linear(times, xyz_points[:, 0].reshape(-1, 1))
    y_slope, y_intercept = _fit_linear(times, xyz_points[:, 1].reshape(-1, 1))
    z_quad, z_lin, z_intercept = _fit_quadratic(times, xyz_points[:, 2].reshape(-1, 1))

    # print(f"x_slope: {x_slope}, x_intercept: {x_intercept}")
    # print(f"y_slope: {y_slope}, y_intercept: {y_intercept}")
    # print(f"z_quad: {z_quad}, z_lin: {z_lin}, z_intercept: {z_intercept}")

    def trajectory(time):
        """
        Returns the position of the ball at specific time

        Args:
            time: scalar int for time or (n, 1) array of times

        Returns:
            (n, 3) numpy array of x, y, an z positions
        """
        time = np.array(time).reshape(-1, 1)
        x = x_slope * time + x_intercept
        y = y_slope * time + y_intercept
        z = z_quad * (time**2) + z_lin * time + z_intercept

        return np.hstack((x, y, z))

    def land_position(axis, landing_position):
        assert axis in ["x", "y", "z"]
        if axis == "x":
            time = (landing_position - x_intercept) / x_slope
        elif axis == "y":
            time = (landing_position - y_intercept) / y_slope
        else:
            a = z_quad
            b = z_lin
            c = z_intercept - landing_position
            time = (-b + np.sqrt(b**2 - (4 * a * c))) / (2 * a)

        return trajectory(time), time

    return trajectory, land_position


def fit_trajectory_physics(xyz_points, times):
    """
    Fits a trajectory to a ball arc

    Args:
        xyz_points: (2, 3) numpy array of points
        times: (n, 1) numpy array of times

    Returns:
        Function with time (int) as input and (1, 3) numpy array as output
    """
    times = np.array(times).reshape(-1, 1)
    elapsed_time = times[1, 0] - times[0, 0]
    v_x_0 = (xyz_points[1, 0] - xyz_points[0, 0]) / elapsed_time
    v_y_0 = (xyz_points[1, 1] - xyz_points[0, 1]) / elapsed_time
    v_z_0 = (
        (xyz_points[1, 2] - xyz_points[0, 2] + HALF_GRAVITY * (elapsed_time**2))
    ) / elapsed_time

    print(f"x_slope: {v_x_0}, x_intercept: {xyz_points[0, 0]}")
    print(f"y_slope: {v_y_0}, y_intercept: {xyz_points[0, 1]}")
    print(f"z_quad: {4.9}, z_lin: {v_z_0}, z_intercept: {xyz_points[0, 2]}")

    def trajectory(time):

        time = np.array(time).reshape(-1, 1) - times[0, 0]
        x = xyz_points[0, 0] + v_x_0 * time
        y = xyz_points[0, 1] + v_y_0 * time
        z = xyz_points[0, 2] + v_z_0 * time - (HALF_GRAVITY * (time**2))

        return np.hstack((x, y, z))

    def land_position(axis, landing_position):
        assert axis in ["x", "y", "z"]
        if axis == "x":
            time = (landing_position - xyz_points[0, 0]) / v_x_0
        elif axis == "y":
            time = (landing_position - xyz_points[0, 1]) / v_y_0
        else:
            a = -HALF_GRAVITY
            b = v_z_0
            c = xyz_points[0, 2] - landing_position
            time = (-b + np.sqrt(b**2 - (4 * a * c))) / (2 * a)

        time += times[0, 0]

        return trajectory(time), time

    return trajectory, land_position


def _points_in_area(reference, cluster, distance):
    x_min = cluster["x"].min()
    x_max = cluster["x"].max()
    y_min = cluster["y"].min()
    y_max = cluster["y"].max()
    z_min = cluster["z"].min()
    z_max = cluster["z"].max()

    return (
        (reference["x"] > x_min - distance)
        & (reference["x"] < x_max + distance)
        & (reference["y"] > y_min - distance)
        & (reference["y"] < y_max + distance)
        & (reference["z"] > z_min - distance)
        & (reference["z"] < z_max + distance)
    ).sum()


def identify_ball_by_background(background, clusters):
    if len(clusters) > 1:

        least_points = _points_in_area(background, clusters[0], POINTS_IN_AREA_DIST)
        least_point_cluster = clusters[0]

        for index in range(1, len(clusters)):
            points = _points_in_area(background, clusters[index], POINTS_IN_AREA_DIST)
            if points < least_points:
                least_points = points
                least_point_cluster = clusters[index]
    else:
        least_point_cluster = clusters[0]

    return least_point_cluster


def verify_ball(ball, min_color, max_color, pixel_threshold):

    r_min, g_min, b_min = min_color
    r_max, g_max, b_max = max_color

    qty_in_range = (
        (ball["r"] >= r_min)
        & (ball["r"] <= r_max)
        & (ball["g"] >= g_min)
        & (ball["g"] <= g_max)
        & (ball["b"] >= b_min)
        & (ball["b"] <= b_max)
    ).sum()

    return qty_in_range >= pixel_threshold


def get_ball_position(ball, offset_size, offset_dim=1):

    mean_x = ball["x"].mean()
    mean_y = ball["y"].mean()
    mean_z = ball["z"].mean()

    position = [mean_x, mean_z, mean_y]

    position[offset_dim] += offset_size

    return position


def process_ball(data, reference, return_clusters=False):

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=["x", "y", "z", "r", "g", "b"])
        reference = pd.DataFrame(reference, columns=["x", "y", "z", "r", "g", "b"])

    # data = data[data["y"] < 2.5]

    if data.shape[0] > 5:
        dbscan = DBSCAN(eps=0.05, min_samples=20)
        x_z = data[["x", "y", "z"]].to_numpy()
        data["c"] = dbscan.fit_predict(x_z)
        data = data[data["c"] != -1]

        # Make cluster list of points from labels returned by fit_predict
        cluster_points = []
        for _, value in enumerate(data["c"].unique()):
            cluster_points.append(
                data[data["c"] == value][["x", "y", "z", "r", "g", "b"]]
            )

        # If clusters exist
        if len(cluster_points) >= 1:
            ball_cluster = identify_ball_by_background(reference, cluster_points)
            is_ball = verify_ball(ball_cluster, BALL_LOWER_RGB, BALL_UPPER_RGB, 5)

            if is_ball:
                sphere = fit_sphere(np.array(ball_cluster[["x", "y", "z"]]))

                if return_clusters:
                    return cluster_points, sphere
                else:
                    return sphere

    if return_clusters:
        return cluster_points, None
    else:
        return None
