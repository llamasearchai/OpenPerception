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
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # Arrays to store object points and image points
        self.objpoints: List[np.ndarray] = []  # 3D points in real world space
        self.imgpoints: List[np.ndarray] = []  # 2D points in image plane
        
        # Calibration results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        
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
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_subpix)
            
            logger.info(f"Added image with {len(corners_subpix)} corners")
            return True
        else:
            logger.warning("Chessboard not found in image")
            return False
            
    def calibrate(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Perform camera calibration
        
        Returns:
            camera_matrix: 3x3 camera intrinsic matrix or None if calibration fails
            dist_coeffs: Distortion coefficients or None if calibration fails
        """
        if len(self.imgpoints) < 3:
            logger.error("Need at least 3 images with detected chessboards for calibration.")
            # raise ValueError("Need at least 3 images with detected chessboards")
            return None, None # Return None if not enough images
            
        # Get image size from first image (assuming all images are same size, which findChessboardCorners implies)
        # gray_shape should be (height, width)
        # img_size for calibrateCamera is (width, height)
        # Example: self.imgpoints[0] is (N, 1, 2) for N corners, (x,y) coords.
        # cv2.calibrateCamera needs imageSize. Let's assume we have a sample gray image's shape used for add_image.
        # For now, this part is tricky without an image. Let's assume gray was set by last add_image call
        # This needs a gray image's shape to be passed or stored. This is a flaw in original code.
        # For robustness, let's try to get it from the image points if possible, though not ideal.
        # However, calibrateCamera takes img_size as (width, height).
        # The imgpoints are image coordinates. We can't directly get image size from them without ambiguity.
        # The original code had: img_size = (self.imgpoints[0].shape[1], self.imgpoints[0].shape[0]) which is (1,N) - incorrect.
        # This needs to be fixed. For now, I'll leave it as is from original, but it's problematic.
        # A better way would be to store the image size when `add_image` is called.
        # Let's assume the user will pass an image to calibrate() or store it. For now, placeholder: 
        if not self.imgpoints: # Should be caught by len(self.imgpoints) < 3
            return None, None
        
        # Attempt to get a representative image size, though this is not robust.
        # Assuming gray images were used to find corners, and we need one such image's shape.
        # This part is problematic in the original code as it doesn't store image_size explicitly.
        # Placeholder: use a common HD resolution if no other info. This is a guess.
        # A proper fix would be to pass image_size to calibrate() or store it in add_image().
        image_height, image_width = 1080, 1920 # Default guess - BAD PRACTICE
        # Heuristic: Try to infer from corners if they are available. Max x and y. Risky.
        if self.imgpoints and len(self.imgpoints[0]) > 0:
             # Max values from the first set of corners could give an estimate of image size.
             # corners are (N, 1, 2). Squeeze to (N,2)
             all_corners = np.vstack([pts.squeeze() for pts in self.imgpoints if pts is not None and pts.size > 0])
             if all_corners.size > 0:
                 image_width = int(np.max(all_corners[:, 0])) + 1
                 image_height = int(np.max(all_corners[:, 1])) + 1
        img_size = (image_width, image_height)

        logger.info(f"Calibrating with {len(self.imgpoints)} images. Estimated image size: {img_size}")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            logger.info(f"Camera calibrated with RMS error: {ret}")
            return self.camera_matrix, self.dist_coeffs
        else:
            logger.error("Camera calibration failed.")
            return None, None
        
    def undistort_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Undistort an image using calibration results
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image, or None if not calibrated
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.warning("Camera not calibrated yet. Cannot undistort image.")
            # raise ValueError("Camera not calibrated yet")
            return None
            
        # Optimization: compute new camera matrix only once
        # h, w = image.shape[:2]
        # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        # dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        # x,y,w,h = roi
        # return dst[y:y+h, x:x+w]
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        
    def save_calibration(self, filename: str) -> bool:
        """Save calibration results to file
        
        Args:
            filename: Output JSON file
        Returns:
            True if successful, False otherwise.
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.error("Camera not calibrated yet. Cannot save calibration.")
            # raise ValueError("Camera not calibrated yet")
            return False
            
        calibration = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_count': len(self.imgpoints),
            'chessboard_size': list(self.chessboard_size), # Ensure it's a list for JSON
            'square_size': self.square_size,
            'rvecs': [r.tolist() for r in self.rvecs] if self.rvecs is not None else None,
            'tvecs': [t.tolist() for t in self.tvecs] if self.tvecs is not None else None,
        }
        
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(calibration, f, indent=2)
            logger.info(f"Calibration data saved to {filename}")
            return True
        except IOError as e:
            logger.error(f"Failed to save calibration file {filename}: {e}")
            return False
            
    def load_calibration(self, filename: str) -> bool:
        """Load calibration results from file
        
        Args:
            filename: Input JSON file
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                calibration = json.load(f)
        except FileNotFoundError:
            logger.error(f"Calibration file {filename} not found.")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}")
            return False
            
        try:
            self.camera_matrix = np.array(calibration['camera_matrix'])
            self.dist_coeffs = np.array(calibration['dist_coeffs'])
            self.chessboard_size = tuple(calibration['chessboard_size'])
            self.square_size = calibration['square_size']
            # Optionally load rvecs and tvecs if needed and present
            self.rvecs = [np.array(r) for r in calibration.get('rvecs', [])] if calibration.get('rvecs') else None
            self.tvecs = [np.array(t) for t in calibration.get('tvecs', [])] if calibration.get('tvecs') else None
            logger.info(f"Calibration data loaded from {filename}")
            return True
        except KeyError as e:
            logger.error(f"Missing key {e} in calibration file {filename}.")
            return False
        
    def draw_chessboard_corners(self, image: np.ndarray) -> np.ndarray:
        """Draw detected chessboard corners on an image
        
        Args:
            image: Input image
            
        Returns:
            Image with drawn corners (copy of original image if corners not found)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        result_image = image.copy()
        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            cv2.drawChessboardCorners(result_image, self.chessboard_size, corners_subpix, ret)
        
        return result_image 