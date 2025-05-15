import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from glob import glob

logger = logging.getLogger(__name__)

class StructureFromMotion:
    """Structure from Motion implementation for 3D reconstruction from images"""
    
    def __init__(self, use_gpu: bool = True, feature_type: str = "sift"):
        """Initialize Structure from Motion module
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            feature_type: Feature detector to use ('sift', 'orb', 'surf')
        """
        self.use_gpu = use_gpu
        self.feature_type = feature_type.lower()
        
        # Initialize feature detector
        if self.feature_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif self.feature_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
        elif self.feature_type == 'surf':
            try:
                self.detector = cv2.xfeatures2d.SURF_create(400)
            except AttributeError:
                logger.warning("SURF not available, falling back to SIFT")
                self.detector = cv2.SIFT_create()
        else:
            logger.warning(f"Unknown feature type: {feature_type}, using SIFT")
            self.detector = cv2.SIFT_create()
        
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        # Data structures
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.camera_matrices = []
        self.point_cloud = None
        self.point_colors = None
        
    def add_image(self, image_path: str) -> int:
        """Add an image to the SfM pipeline
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Index of the added image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp, desc = self.detector.detectAndCompute(gray, None)
        
        logger.info(f"Found {len(kp)} features in image: {image_path}")
        
        # Store data
        self.images.append(image)
        self.keypoints.append(kp)
        self.descriptors.append(desc)
        
        # Return index of added image
        return len(self.images) - 1
    
    def match_features(self, idx1: int, idx2: int, ratio_test: float = 0.7) -> List[cv2.DMatch]:
        """Match features between two images
        
        Args:
            idx1: Index of first image
            idx2: Index of second image
            ratio_test: Ratio for Lowe's ratio test
            
        Returns:
            List of matches
        """
        # Get descriptors
        desc1 = self.descriptors[idx1]
        desc2 = self.descriptors[idx2]
        
        # Match features
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
        
        logger.info(f"Found {len(good_matches)} matches between images {idx1} and {idx2}")
        
        return good_matches
    
    def estimate_pose(self, idx1: int, idx2: int, K: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Estimate relative pose between two images
        
        Args:
            idx1: Index of first image
            idx2: Index of second image
            K: Camera intrinsic matrix (if None, use default)
            
        Returns:
            success: Whether pose estimation was successful
            R: Rotation matrix
            t: Translation vector
        """
        # Use default camera matrix if not provided
        if K is None:
            # Assume unit camera matrix with principal point at image center
            h, w = self.images[idx1].shape[:2]
            f = max(h, w)
            K = np.array([
                [f, 0, w/2],
                [0, f, h/2],
                [0, 0, 1]
            ])
        
        # Match features
        matches = self.match_features(idx1, idx2)
        
        if len(matches) < 8:
            logger.warning(f"Not enough matches ({len(matches)}) for pose estimation")
            return False, None, None
        
        # Extract matched keypoints
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            logger.warning("Failed to estimate essential matrix")
            return False, None, None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        
        return True, R, t
    
    def triangulate_points(self, idx1: int, idx2: int, R: np.ndarray, t: np.ndarray, K: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Triangulate 3D points from two views
        
        Args:
            idx1: Index of first image
            idx2: Index of second image
            R: Rotation matrix
            t: Translation vector
            K: Camera intrinsic matrix (if None, use default)
            
        Returns:
            points_3d: 3D points
            colors: Colors of 3D points
        """
        # Use default camera matrix if not provided
        if K is None:
            # Assume unit camera matrix with principal point at image center
            h, w = self.images[idx1].shape[:2]
            f = max(h, w)
            K = np.array([
                [f, 0, w/2],
                [0, f, h/2],
                [0, 0, 1]
            ])
        
        # Match features
        matches = self.match_features(idx1, idx2)
        
        # Extract matched keypoints
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        
        # Compute projection matrices
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        
        # Triangulate points
        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous coordinates
        pts_3d = pts_4d[:3, :] / pts_4d[3, :]
        
        # Get colors from first image
        colors = np.zeros((pts_3d.shape[1], 3))
        for i, (x, y) in enumerate(pts1.astype(int)):
            if 0 <= x < self.images[idx1].shape[1] and 0 <= y < self.images[idx1].shape[0]:
                colors[i] = self.images[idx1][y, x][::-1]  # BGR to RGB
        
        return pts_3d.T, colors
    
    def run_from_images(self, image_dir: str, output_dir: str):
        """Run Structure from Motion on a directory of images
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find image files
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob(os.path.join(image_dir, f'*.{ext}')))
            image_files.extend(glob(os.path.join(image_dir, f'*.{ext.upper()}')))
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")
        
        logger.info(f"Found {len(image_files)} images")
        
        # Add images to pipeline
        for image_file in image_files:
            self.add_image(image_file)
        
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for SfM")
        
        # Initialize camera poses
        # Start with first camera at origin
        camera_poses = [np.eye(4)]
        
        # Estimate pose of second camera relative to first
        success, R, t = self.estimate_pose(0, 1)
        if not success:
            raise RuntimeError("Failed to estimate pose between first two images")
        
        # Create transformation matrix for second camera
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        camera_poses.append(T)
        
        # Triangulate initial point cloud
        points_3d, colors = self.triangulate_points(0, 1, R, t)
        
        logger.info(f"Initial triangulation: {points_3d.shape[0]} points")
        
        # TODO: Add bundle adjustment and more sophisticated pipeline
        
        # Store results
        self.point_cloud = points_3d
        self.point_colors = colors
        
        # Save point cloud
        self.save_point_cloud(os.path.join(output_dir, 'point_cloud.ply'))
        
        logger.info(f"Results saved to: {output_dir}")
    
    def save_point_cloud(self, filename: str):
        """Save point cloud to PLY file
        
        Args:
            filename: Output PLY file path
        """
        if self.point_cloud is None or self.point_colors is None:
            raise ValueError("No point cloud to save")
        
        # Create header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(self.point_cloud)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header"
        ]
        
        # Write to file
        with open(filename, 'w') as f:
            f.write('\n'.join(header) + '\n')
            
            for i in range(len(self.point_cloud)):
                x, y, z = self.point_cloud[i]
                r, g, b = self.point_colors[i].astype(int)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        logger.info(f"Point cloud saved to: {filename}")
    
    def visualize_matches(self, idx1: int, idx2: int, output_path: Optional[str] = None) -> np.ndarray:
        """Visualize matches between two images
        
        Args:
            idx1: Index of first image
            idx2: Index of second image
            output_path: Path to save visualization (if None, don't save)
            
        Returns:
            Visualization image
        """
        # Match features
        matches = self.match_features(idx1, idx2)
        
        # Draw matches
        img_matches = cv2.drawMatches(
            self.images[idx1], self.keypoints[idx1],
            self.images[idx2], self.keypoints[idx2],
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Save if requested
        if output_path:
            cv2.imwrite(output_path, img_matches)
        
        return img_matches 