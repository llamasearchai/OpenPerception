import numpy as np
import cv2
import os
import glob
from typing import Tuple, List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)

class CameraCalibrator:
    """Camera calibration using chessboard patterns"""
    
    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """Initialize calibrator
        
        Args:
            chessboard_size: Number of inner corners in the chessboard (width, height)
            square_size: Size of the chessboard squares in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def add_image(self, image: np.ndarray) -> bool:
        """Add an image for calibration
        
        Args:
            image: Input image with chessboard
            
        Returns:
            True if chessboard was detected, False otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            
            logger.info(f"Added image with {len(corners)} corners")
            return True
        else:
            logger.warning("Chessboard not found in image")
            return False
            
    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform camera calibration
        
        Returns:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        if len(self.imgpoints) < 3:
            raise ValueError("Need at least 3 images with detected chessboards")
            
        # Get image size from first image point set
        img_size = (self.imgpoints[0].shape[1], self.imgpoints[0].shape[0])
        
        # Calibrate camera
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )
        
        logger.info(f"Camera calibrated with RMS error: {ret}")
        
        return self.camera_matrix, self.dist_coeffs
        
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using calibration results
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
            
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        
    def save_calibration(self, filename: str):
        """Save calibration results to file
        
        Args:
            filename: Output JSON file
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
            
        calibration = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_count': len(self.imgpoints),
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration, f, indent=2)
            
    def load_calibration(self, filename: str):
        """Load calibration results from file
        
        Args:
            filename: Input JSON file
        """
        with open(filename, 'r') as f:
            calibration = json.load(f)
            
        self.camera_matrix = np.array(calibration['camera_matrix'])
        self.dist_coeffs = np.array(calibration['dist_coeffs'])
        self.chessboard_size = tuple(calibration['chessboard_size'])
        self.square_size = calibration['square_size']
        
    def draw_chessboard_corners(self, image: np.ndarray) -> np.ndarray:
        """Draw detected chessboard corners on an image
        
        Args:
            image: Input image
            
        Returns:
            Image with drawn corners
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            result = cv2.drawChessboardCorners(image.copy(), self.chessboard_size, corners, ret)
            return result
        else:
            return image.copy() 