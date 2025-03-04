# 404 Swish Not Found

An automated basketball hoop that uses computer vision to track shots, predict trajectory, and actuate the basketball hoop.

[Watch the video demo!](https://www.youtube.com/watch?v=4lS1lKCNuzg)

## Overview

404 Swish Not Found is an open-source project that combines hardware and software to create an automated basketball hoop capable of tracking shots using computer vision and moving the hoop to make (almost) every shot go in.

## Features

- **Real-Time Tracking**: Utilizes an Xbox Kinect for real-time ball tracking, then it predicts its trajectory.
- **Open-Source Software**: Written in Python with various libraries for image processing and data analysis.
- **Dynamic Hoop Positioning**: Uses stepper motors controlled via an Arduino to reposition the hoop based on ball trajectory predictions.

## Software Design

### Architecture

The software is organized into several components:

1. **Run Scanner Script**: Main script that initializes the system, connects to the Arduino over a serial connection, and manages the scanning process.
2. **BallScanner Class**: Handles the Kinect connection, processes scans into image and point cloud data, and predicts the ball's trajectory.
3. **Ball Process Library**: Manages ball detection from point cloud data, sphere fitting, and trajectory calculations.
4. **Calibration Library**: Manages initial calibration for the basketball hoop's depth and coordinate system, guiding the user through the process using Matplotlib.
5. **Plot Scans Script**: Provides visualization tools to plot and analyze scan data, aiding in debugging and system calibration.
6. **Point to Actuation**: Translates target coordinates into precise motor movements, actuating the basketball hoop to move to the predicted position.

### Processing Pipeline

1. **Initial Setup and Calibration**: Captures a reference image and point scan as baseline data. Loads pre-existing calibration data or guides the user through calibration to establish reference frame coordinates.
2. **Image Processing and Registration**: Uses PyLibFreenect2 to obtain aligned RGB and infrared data from the Kinect. Compares the current frame against the reference image to create a mask highlighting changed pixels for motion detection.
3. **3D Point Cloud Generation**: Projects masked 2D pixel coordinates into three-dimensional space, creating an RGB point cloud of detected changes.
4. **Ball Detection and Clustering**: Applies the DBSCAN algorithm to identify distinct clusters in the point cloud. Selects the most isolated cluster, verifies its orange color to confirm it's the basketball, and fits a sphere to determine the ball's center coordinates.
5. **Trajectory Analysis and Prediction**: Calculates velocity components and applies kinematic equations to predict the ball's path. Uses regression analysis to model the ball's trajectory based on collected
6. **Actuation**: Calculates precise motor steps to move the hoop based on predicted ball positions. Controls stepper motors using the AccelStepper and MultiStepper libraries.

## External Software Dependencies

- **PyLibFreenect2**: Manages connection to Xbox Kinect One and basic processing of Kinect data.
- **OpenCV (cv2)**: Provides image processing capabilities for ball detection.
- **Pandas**: Used for storing and processing point data.
- **NumPy**: Assists in processing data for sphere fitting, trajectory prediction, and general computations.
- **Scikit-learn**: Facilitates clustering of 3D point cloud data.
- **Matplotlib**: Enables visualization of point cloud data and interactive calibration.
- **AccelStepper**: Arduino library for controlling stepper motors.
- **MultiStepper**: Arduino library for synchronized control of multiple steppers.
