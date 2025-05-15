import unittest
import numpy as np
import cv2
import os
import tempfile
import json
from pathlib import Path

from openperception.calibration.camera_calibration import CameraCalibrator

class TestCameraCalibrator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = CameraCalibrator(chessboard_size=(7, 6), square_size=0.025)
        
        # Create a synthetic chessboard image for testing
        self.image_size = (640, 480)
        self.test_image = self._create_chessboard_image(self.image_size)
        
    def _create_chessboard_image(self, image_size):
        """Create a synthetic chessboard image"""
        # Create a simple chessboard pattern
        width, height = image_size
        chessboard_size = self.calibrator.chessboard_size
        
        # Simple projection matrix
        K = np.array([
            [500, 0, width/2],
            [0, 500, height/2],
            [0, 0, 1]
        ])
        
        # Create an image with a projected chessboard
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Project chessboard corners
        for i in range(chessboard_size[0]):
            for j in range(chessboard_size[1]):
                # 3D point
                x = i * self.calibrator.square_size
                y = j * self.calibrator.square_size
                z = 0
                
                # Project to image
                X = np.array([x, y, z, 1])
                x, y, w = K @ np.array([x, y, 1])
                x = int(x / w)
                y = int(y / w)
                
                # Draw a dot
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(img, (x, y), 5, 255, -1)
        
        # Convert to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    
    def test_add_image(self):
        """Test adding an image to the calibrator"""
        result = self.calibrator.add_image(self.test_image)
        
        # Should return True when chessboard is found
        self.assertTrue(result)
        
        # Should have added points
        self.assertEqual(len(self.calibrator.objpoints), 1)
        self.assertEqual(len(self.calibrator.imgpoints), 1)
    
    def test_calibrate(self):
        """Test calibration with synthetic images"""
        # Add multiple images from different viewpoints
        for _ in range(3):
            self.calibrator.add_image(self.test_image)
        
        # Calibrate
        camera_matrix, dist_coeffs = self.calibrator.calibrate()
        
        # Check results
        self.assertIsNotNone(camera_matrix)
        self.assertIsNotNone(dist_coeffs)
        self.assertEqual(camera_matrix.shape, (3, 3))
    
    def test_save_load_calibration(self):
        """Test saving and loading calibration results"""
        # Add images and calibrate
        for _ in range(3):
            self.calibrator.add_image(self.test_image)
        
        camera_matrix, dist_coeffs = self.calibrator.calibrate()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.calibrator.save_calibration(tmp_path)
            
            # Create a new calibrator and load the results
            new_calibrator = CameraCalibrator()
            new_calibrator.load_calibration(tmp_path)
            
            # Check loaded values
            np.testing.assert_array_almost_equal(camera_matrix, new_calibrator.camera_matrix)
            np.testing.assert_array_almost_equal(dist_coeffs, new_calibrator.dist_coeffs)
            self.assertEqual(self.calibrator.chessboard_size, new_calibrator.chessboard_size)
            self.assertEqual(self.calibrator.square_size, new_calibrator.square_size)
        
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_draw_chessboard_corners(self):
        """Test drawing chessboard corners"""
        # Draw corners
        result_img = self.calibrator.draw_chessboard_corners(self.test_image)
        
        # Should return an image
        self.assertIsNotNone(result_img)
        self.assertEqual(result_img.shape, self.test_image.shape)
    
    def test_undistort_image(self):
        """Test image undistortion"""
        # Add images and calibrate
        for _ in range(3):
            self.calibrator.add_image(self.test_image)
        
        self.calibrator.calibrate()
        
        # Undistort
        undistorted = self.calibrator.undistort_image(self.test_image)
        
        # Should return an image of the same size
        self.assertEqual(undistorted.shape, self.test_image.shape)

if __name__ == '__main__':
    unittest.main() 