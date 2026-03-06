import cv2
import numpy as np
from importCalibration import cameraIntrinsics
from importCalibration import stereoCameraCalibrationData
from importCalibration import importCalibration
from enum import Enum

class CoordinatesSystem(Enum):
    leftCamera = 0
    rightCamera = 1

class Triangulation:
    def __init__(self, calibrationPath):
        # Read in calibration at initialization
        self.stereoCamera = importCalibration(calibrationPath)
        # Camera intrinsic parameters
        self.K1 = np.array([[self.stereoCamera.left.fx, 0, self.stereoCamera.left.cx],
               [0, self.stereoCamera.left.fy, self.stereoCamera.left.cy],
               [0, 0, 1]], dtype=np.float64)
        self.K2 = K2 = np.array([[self.stereoCamera.right.fx, 0, self.stereoCamera.right.cx],
               [0, self.stereoCamera.right.fy, self.stereoCamera.right.cy],
               [0, 0, 1]], dtype=np.float64)
        self.D1 = np.array(self.stereoCamera.left.d, dtype=np.float64)
        self.D2 = np.array(self.stereoCamera.right.d, dtype=np.float64)
        self.R = R = np.array(self.stereoCamera.extrinsics.r, dtype=np.float64)
        self.T = T = np.array(self.stereoCamera.extrinsics.t, dtype=np.float64)
        # Projection matrices
        self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P2 = self.K2 @ np.hstack((R, T))
        self.coordinatesSystem = CoordinatesSystem.leftCamera # left camera by default

    def setCoordinatesSystem(self, coordinateSystem):
        self.coordinatesSystem = coordinateSystem
    
    def triangulate(self, points1, points2):
        #Undistort points
        points1 = cv2.undistortPoints(points1, self.K1, self.D1, P=self.K1)
        points2 = cv2.undistortPoints(points2, self.K2, self.D2, P=self.K2)
        
        # Triangulate points
        homogeneous_points_4d = cv2.triangulatePoints(self.P1, self.P2, points1, points2)

        # Convert from homogeneous coordinates to 3D
        points_3d = homogeneous_points_4d[:3] / homogeneous_points_4d[3]

        if (self.coordinatesSystem == CoordinatesSystem.rightCamera):
            invR = np.linalg.inv(self.R)
            points_3d = np.matmul(invR, points_3d) - self.T

        return points_3d.T