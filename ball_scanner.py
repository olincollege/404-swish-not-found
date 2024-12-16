"""
Library to take scans using Kinect
"""

import sys
import time
from PIL import Image

import numpy as np
import cv2

from pylibfreenect2 import (
    Freenect2,
    SyncMultiFrameListener,
    FrameType,
    Registration,
    Frame,
    createConsoleLogger,
    setGlobalLogger,
    LoggerLevel,
    OpenGLPacketPipeline,
)

from ball_process_lib import (
    process_ball,
    fit_trajectory_physics,
    fit_trajectory_regression,
)
from calibrate_lib import calibrate_depth, calibrate_hoop_coords


class BallScanner:

    points_folder = "points"
    image_folder = "img"
    calibrate_path = "calibrate.csv"
    landing_dim = "y"
    end_throw_threshold = 4.15
    reset_time_threshold = 3
    reset_pause_time = 7

    def __init__(self, save_history=False, calibrate=False, reset=True):
        self.pipeline = OpenGLPacketPipeline()

        print("Packet pipeline:", type(self.pipeline).__name__)

        self.logger = createConsoleLogger(LoggerLevel.Debug)
        setGlobalLogger(self.logger)

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial, pipeline=self.pipeline)

        self.listener = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth
        )

        # Register listeners
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)

        self.device.start()

        # Must be called after device.start()
        self.registration = Registration(
            self.device.getIrCameraParams(), self.device.getColorCameraParams()
        )

        self.undistorted = Frame(512, 424, 4)
        self.registered = Frame(512, 424, 4)
        _, self.reference_img, _, self.reference_depth, _ = self.get_scan(
            first_frame=True
        )

        self.min_calibration_point, self.max_calibration_point = self.calibrate(
            calibrate
        )

        self.should_reset = reset

        self.scan_times = []
        self.start_time = time.time()
        self.save_history = save_history
        self.ball_history = []
        self.land_history = []
        self.land_history_hoop_coords = []
        self.traj_function_history = []

        if self.save_history:
            self.img_history = []
            self.mask_img_history = []
            self.depth_history = []

    def get_scan(self, first_frame=False):
        frames = self.listener.waitForNewFrame()

        color = frames["color"]
        depth = frames["depth"]

        self.registration.apply(
            color,
            depth,
            self.undistorted,
            self.registered,
        )

        registered_img = self.registered.asarray(dtype=np.uint8)[:, :, [2, 1, 0]]

        last_scan_time = None

        if first_frame:
            masked_registered_img = None
            depth = self._calculate_depth(mask=None)
            ball = None
        else:
            masked_registered_img, mask = self._mask_frame(registered_img)
            depth = self._calculate_depth(mask=mask)
            ball = process_ball(depth, self.reference_depth)

            last_scan_time = time.time() - self.start_time
            self.scan_times.append(last_scan_time)
            print(f"Scan time: {last_scan_time}")

            if ball is not None:
                print(ball)
                ball.append(last_scan_time)
                self.ball_history.append(ball)

            if self.save_history:
                self.img_history.append(registered_img)
                self.mask_img_history.append(masked_registered_img)
                self.depth_history.append(depth)

            if self.should_reset:
                self.check_for_reset()

        self.listener.release(frames)

        return last_scan_time, registered_img, masked_registered_img, depth, ball

    def _calculate_depth(self, mask):
        # Get depth and color images
        depth_image = self.undistorted.asarray(np.float32)  # Shape: (512, 424)
        color_image = self.registered.asarray(np.uint8)  # Shape: (512, 424, 4)

        # Get intrinsic parameters
        ir_params = self.device.getIrCameraParams()
        fx = ir_params.fx
        fy = ir_params.fy
        cx = ir_params.cx
        cy = ir_params.cy

        # Get the shape of the depth image
        height, width = depth_image.shape  # Should be (512, 424)

        # Create coordinate grids
        u = np.arange(width)  # u corresponds to cols (width)
        v = np.arange(height)  # v corresponds to rows (height)
        u_grid, v_grid = np.meshgrid(u, v)  # Shapes: (height, width)

        # Compute x, y, z
        x = (u_grid - cx) * depth_image / fx
        y = (v_grid - cy) * depth_image / fy
        z = depth_image

        # Stack x, y, z into points array
        points = np.stack((x, y, z), axis=2)  # Shape: (height, width, 3)

        if mask is not None:
            # Create mask for valid depth
            valid = (depth_image > 0) & (~np.isnan(depth_image)) & (mask > 1)
        else:
            valid = (depth_image > 0) & (~np.isnan(depth_image))

        # Extract valid points and colors
        valid_points = points[valid] / 1000  # Shape: (N_valid, 3) ["x", "z", "y"]
        valid_colors = color_image[
            valid, :3
        ]  # Assuming RGB channels are first three ["b", "g", "r"]

        # Combine points and colors ['x', 'y', 'z', 'r', 'b', 'g']
        final_points = np.column_stack(
            (
                valid_points[:, 0],
                valid_points[:, 2],
                valid_points[:, 1],
                valid_colors[:, 2],
                valid_colors[:, 1],
                valid_colors[:, 0],
            )
        )

        # final_points[:, 0] = -1 * final_points[:, 0]
        final_points[:, 2] = -1 * final_points[:, 2]

        return final_points  # Shape: (N_valid, 6)

    def get_landing_position(self):
        if len(self.ball_history) <= 1:
            return None
        if len(self.ball_history) < 5:
            np_bh = np.array(self.ball_history)
            traj_function, landing_position = fit_trajectory_physics(
                np_bh[[0, len(self.ball_history) - 1], :3],
                np_bh[[0, len(self.ball_history) - 1], 4],
            )
        else:
            np_bh = np.array(self.ball_history)
            traj_function, landing_position = fit_trajectory_regression(
                np_bh[:, :3], np_bh[:, 4]
            )

        landing_coords, landing_time = landing_position(
            BallScanner.landing_dim, self.min_calibration_point[1]
        )

        landing_coords = landing_coords[0]

        print(f"Landing coords: {landing_coords} at {landing_time} sec")

        if (
            landing_coords[0] >= self.min_calibration_point[0] - 0.2
            and landing_coords[0] <= self.max_calibration_point[0] + 0.2
            and landing_coords[2] >= self.min_calibration_point[2] - 0.2
            and landing_coords[2] <= self.max_calibration_point[2] + 0.2
        ):
            landing_coords_hoop_frame = [
                abs(landing_coords[0] - self.max_calibration_point[0]),
                landing_coords[2] - self.min_calibration_point[2],
                len(self.traj_function_history),
                len(self.ball_history),
            ]
            self.land_history_hoop_coords.append(landing_coords_hoop_frame)
            print(
                f"Will Hit at {landing_coords_hoop_frame}, Traj Index: {len(self.traj_function_history)}"
            )
        else:
            landing_coords_hoop_frame = None
            print("Won't Hit")

        self.land_history.append(landing_coords)
        self.traj_function_history.append(traj_function)

        return landing_coords, landing_coords_hoop_frame, landing_time

    def _mask_frame(self, new_frame):
        gray_ref = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference
        diff = cv2.absdiff(gray_ref, gray_new)

        # Threshold the difference to create a binary mask of the changes
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Optional: Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Apply the mask to the new image to isolate changed regions+
        return cv2.bitwise_and(new_frame, new_frame, mask=mask), mask

    def _save_scans(self):
        for index, arr in enumerate(self.img_history):
            image = Image.fromarray(arr).convert("RGB")
            image.save(f"img/{index}_reg.png")

        for index, arr in enumerate(self.mask_img_history):
            image = Image.fromarray(arr).convert("RGB")
            image.save(f"img/{index}_mask.png")

        for index, points in enumerate(self.depth_history):
            np.savetxt(f"points/points_{index}.csv", points, delimiter=",")

        np.savetxt("times.csv", np.array(self.scan_times), delimiter=",")

        np.savetxt(
            "points/reference.csv",
            self.reference_depth,
            delimiter=",",
        )

        if len(self.ball_history) > 0:
            np.savetxt("spheres.csv", np.array(self.ball_history), delimiter=",")

        if len(self.land_history_hoop_coords) > 0:
            np.savetxt(
                "land_points_hoop.csv",
                np.array(self.land_history_hoop_coords),
                delimiter=",",
            )

        for index, trajectory_function in enumerate(self.traj_function_history):
            times = np.arange(0, self.scan_times[-1], 0.05)
            np.savetxt(
                f"trajectories/{index}.csv", trajectory_function(times), delimiter=","
            )

    def calibrate(self, calibrate):
        """
        Returns:
            [x_min, y, z_min], [x_max, y, z_max]
        """
        if not calibrate:
            calibration_points = np.loadtxt(BallScanner.calibrate_path, delimiter=",")
            return calibration_points[:3], calibration_points[3:6]
        else:
            y = calibrate_depth(self.reference_depth)
            mins, maxs = calibrate_hoop_coords(self.reference_depth)
            mins.insert(1, y)
            maxs.insert(1, y)
            np.savetxt(BallScanner.calibrate_path, np.array(mins + maxs), delimiter=",")
            return mins, maxs

    def reset(self):
        self.scan_times = []
        # self.start_time = time.time()
        self.ball_history = []
        self.land_history = []
        self.land_history_hoop_coords = []
        self.traj_function_history = []

    def check_for_reset(self):
        try:
            if len(self.ball_history) > 0:
                if (
                    time.time() - self.ball_history[-1][-1] - self.start_time
                    > BallScanner.reset_time_threshold
                ):
                    print(
                        f"Reset because of time since pickup: {time.time()} - {self.ball_history[-1][-1]}"
                    )
                    self.reset()
                elif self.ball_history[-1][1] > BallScanner.end_throw_threshold:
                    print("Reset because of end throw distance")
                    self.reset()
                    time.sleep(BallScanner.reset_pause_time)
        except:
            print(f"Can't get history: Ball history: {self.ball_history}")

    def close(self):
        if self.save_history:
            self._save_scans()

        self.device.stop()
        self.device.close()
