"""
Visualization module for OpenPerception.

This module provides visualization capabilities for SLAM, SfM, 
and sensor fusion results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualization class for OpenPerception outputs."""
    
    def __init__(self, max_points: int = 100000, point_size: float = 2.0, show_trajectory: bool = True):
        """Initialize the visualizer.
        
        Args:
            max_points: Maximum number of points to visualize
            point_size: Size of points in visualization
            show_trajectory: Whether to show camera/sensor trajectory
        """
        self.max_points = max_points
        self.point_size = point_size
        self.show_trajectory = show_trajectory
        
    def visualize_slam_results(self, trajectory: np.ndarray, map_points: np.ndarray, 
                             block: bool = True, save_path: Optional[str] = None):
        """Visualize SLAM results in 3D.
        
        Args:
            trajectory: Nx4x4 array of camera poses (transformation matrices)
            map_points: Mx3 array of 3D map points
            block: Whether to block execution until the plot window is closed
            save_path: Path to save the visualization image. If None, image is not saved.
        """
        if map_points.shape[0] == 0:
            logger.warning("No map points to visualize")
            return
            
        if trajectory.shape[0] == 0:
            logger.warning("No trajectory points to visualize")
            return
            
        # Limit number of points to visualize
        if map_points.shape[0] > self.max_points:
            logger.info(f"Limiting visualization to {self.max_points} out of {map_points.shape[0]} points")
            # Randomly select points
            indices = np.random.choice(map_points.shape[0], self.max_points, replace=False)
            map_points = map_points[indices]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot map points
        ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], 
                  c=map_points[:, 2], cmap='viridis', s=self.point_size, alpha=0.5)
        
        # Plot camera trajectory
        if self.show_trajectory:
            camera_positions = np.array([pose[:3, 3] for pose in trajectory])
            ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                   'r-', linewidth=2, label='Camera Trajectory')
            
            # Plot camera orientations (every 10th camera for clarity)
            for i in range(0, len(trajectory), 10):
                pose = trajectory[i]
                position = pose[:3, 3]
                # Draw coordinate axes
                axis_length = 0.5  # Length of coordinate axes
                axes = np.array([
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length]
                ])
                for j, color in enumerate(['r', 'g', 'b']):
                    direction = pose[:3, :3] @ axes[j]
                    ax.quiver(position[0], position[1], position[2],
                             direction[0], direction[1], direction[2],
                             color=color, length=axis_length, normalize=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SLAM Results: 3D Map and Camera Trajectory')
        
        # Set equal aspect ratio
        max_range = np.array([
            map_points[:, 0].max() - map_points[:, 0].min(),
            map_points[:, 1].max() - map_points[:, 1].min(),
            map_points[:, 2].max() - map_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (map_points[:, 0].max() + map_points[:, 0].min()) * 0.5
        mid_y = (map_points[:, 1].max() + map_points[:, 1].min()) * 0.5
        mid_z = (map_points[:, 2].max() + map_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show(block=block)
        
    def visualize_sfm_results(self, cameras: Dict[str, Any], points3d: np.ndarray, 
                            block: bool = True, save_path: Optional[str] = None):
        """Visualize Structure from Motion results in 3D.
        
        Args:
            cameras: Dictionary of camera information including poses
            points3d: Mx3 array of 3D points
            block: Whether to block execution until the plot window is closed
            save_path: Path to save the visualization image. If None, image is not saved.
        """
        if points3d.shape[0] == 0:
            logger.warning("No 3D points to visualize")
            return
            
        if len(cameras) == 0:
            logger.warning("No cameras to visualize")
            return
            
        # Limit number of points to visualize
        if points3d.shape[0] > self.max_points:
            logger.info(f"Limiting visualization to {self.max_points} out of {points3d.shape[0]} points")
            # Randomly select points
            indices = np.random.choice(points3d.shape[0], self.max_points, replace=False)
            points3d = points3d[indices]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], 
                  c=points3d[:, 2], cmap='viridis', s=self.point_size, alpha=0.5)
        
        # Plot camera positions and orientations
        if self.show_trajectory:
            camera_positions = []
            for cam_id, cam_data in cameras.items():
                if 'pose' in cam_data:
                    pose = cam_data['pose']
                    position = pose[:3, 3]
                    camera_positions.append(position)
                    
                    # Draw coordinate axes (for some cameras for clarity)
                    if len(camera_positions) % 5 == 0:  # Every 5th camera
                        axis_length = 0.5  # Length of coordinate axes
                        axes = np.array([
                            [axis_length, 0, 0],
                            [0, axis_length, 0],
                            [0, 0, axis_length]
                        ])
                        for j, color in enumerate(['r', 'g', 'b']):
                            direction = pose[:3, :3] @ axes[j]
                            ax.quiver(position[0], position[1], position[2],
                                    direction[0], direction[1], direction[2],
                                    color=color, length=axis_length, normalize=True)
            
            # Plot camera trajectory if we have positions
            if camera_positions:
                camera_positions = np.array(camera_positions)
                ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                      'r-', linewidth=2, label='Camera Trajectory')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Structure from Motion Results: 3D Point Cloud and Cameras')
        
        # Set equal aspect ratio
        max_range = np.array([
            points3d[:, 0].max() - points3d[:, 0].min(),
            points3d[:, 1].max() - points3d[:, 1].min(),
            points3d[:, 2].max() - points3d[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points3d[:, 0].max() + points3d[:, 0].min()) * 0.5
        mid_y = (points3d[:, 1].max() + points3d[:, 1].min()) * 0.5
        mid_z = (points3d[:, 2].max() + points3d[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show(block=block)
    
    def visualize_sensor_fusion(self, fused_state: Dict[str, Any], sensor_data: Dict[str, Any] = None,
                              block: bool = True, save_path: Optional[str] = None):
        """Visualize sensor fusion results.
        
        Args:
            fused_state: Dictionary containing fused state information
            sensor_data: Dictionary containing sensor data for visualization
            block: Whether to block execution until the plot window is closed
            save_path: Path to save the visualization image. If None, image is not saved.
        """
        if not fused_state:
            logger.warning("No fused state to visualize")
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # For 3D position and trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        
        # If we have position history, plot the trajectory
        if 'position_history' in fused_state and len(fused_state['position_history']) > 0:
            positions = np.array(fused_state['position_history'])
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2, label='Fused Trajectory')
            
            # Mark current position
            current_pos = fused_state['position']
            ax1.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], c='blue', s=100, label='Current Position')
        else:
            # Just plot current position
            pos = fused_state['position']
            ax1.scatter([pos[0]], [pos[1]], [pos[2]], c='blue', s=100, label='Position')
        
        # If we have raw sensor positions, plot them too
        if sensor_data and 'positions' in sensor_data:
            for sensor_id, positions in sensor_data['positions'].items():
                if len(positions) > 0:
                    pos_array = np.array(positions)
                    ax1.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], alpha=0.3, label=f'{sensor_id} Data')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Fused Trajectory and Position')
        ax1.legend()
        
        # For velocity visualization
        ax2 = fig.add_subplot(222)
        if 'velocity_history' in fused_state and len(fused_state['velocity_history']) > 0:
            velocities = np.array(fused_state['velocity_history'])
            times = np.arange(len(velocities))
            ax2.plot(times, velocities[:, 0], 'r-', label='X Velocity')
            ax2.plot(times, velocities[:, 1], 'g-', label='Y Velocity')
            ax2.plot(times, velocities[:, 2], 'b-', label='Z Velocity')
            
            # Compute speed
            speeds = np.linalg.norm(velocities, axis=1)
            ax2.plot(times, speeds, 'k-', label='Speed')
        else:
            # Just plot current velocity
            vel = fused_state['velocity']
            ax2.bar(['X', 'Y', 'Z'], vel, color=['r', 'g', 'b'])
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax2.set_title('Velocity Components')
        ax2.set_xlabel('Time/Component')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        
        # For orientation visualization
        ax3 = fig.add_subplot(223)
        if 'orientation_history' in fused_state and len(fused_state['orientation_history']) > 0:
            # Convert quaternions to Euler angles
            from ..utils.transformations import quaternion_to_euler
            
            orientations = np.array(fused_state['orientation_history'])
            euler_angles = []
            
            for quat in orientations:
                roll, pitch, yaw = quaternion_to_euler(quat)
                euler_angles.append([roll, pitch, yaw])
                
            euler_angles = np.degrees(np.array(euler_angles))  # Convert to degrees
            times = np.arange(len(euler_angles))
            
            ax3.plot(times, euler_angles[:, 0], 'r-', label='Roll')
            ax3.plot(times, euler_angles[:, 1], 'g-', label='Pitch')
            ax3.plot(times, euler_angles[:, 2], 'b-', label='Yaw')
        else:
            # Just plot current orientation as Euler angles
            from ..utils.transformations import quaternion_to_euler
            
            quat = fused_state['orientation']
            roll, pitch, yaw = quaternion_to_euler(quat)
            euler_degrees = np.degrees([roll, pitch, yaw])
            ax3.bar(['Roll', 'Pitch', 'Yaw'], euler_degrees, color=['r', 'g', 'b'])
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax3.set_title('Orientation (Euler Angles)')
        ax3.set_xlabel('Time/Angle')
        ax3.set_ylabel('Degrees')
        ax3.legend()
        
        # For uncertainty/covariance visualization
        ax4 = fig.add_subplot(224)
        if 'covariance' in fused_state:
            cov = fused_state['covariance']
            # Get position uncertainty (first 3 diagonal elements typically)
            position_uncertainty = np.sqrt(np.diag(cov)[:3])
            
            ax4.bar(['X', 'Y', 'Z'], position_uncertainty, color=['r', 'g', 'b'], alpha=0.7)
            ax4.set_title('Position Uncertainty (Std. Dev.)')
            ax4.set_xlabel('Axis')
            ax4.set_ylabel('Standard Deviation (m)')
        else:
            ax4.text(0.5, 0.5, 'Covariance data not available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Uncertainty Information')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show(block=block)
    
    def visualize_lidar_camera_fusion(self, image: np.ndarray, projected_points: np.ndarray, 
                                    point_colors: Optional[np.ndarray] = None, 
                                    save_path: Optional[str] = None):
        """Visualize LiDAR points projected onto a camera image.
        
        Args:
            image: Input camera image (H, W, 3)
            projected_points: Nx2 array of projected LiDAR points on image
            point_colors: Nx3 array of RGB colors for points. If None, uses distance-based coloring.
            save_path: Path to save the visualization image. If None, image is not saved.
        """
        if projected_points.shape[0] == 0:
            logger.warning("No projected points to visualize")
            return
            
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Draw points on image
        for i, pt in enumerate(projected_points):
            x, y = int(pt[0]), int(pt[1])
            
            # Skip points outside image
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                continue
                
            # Determine point color
            if point_colors is not None and i < len(point_colors):
                color = (int(point_colors[i][2]), int(point_colors[i][1]), int(point_colors[i][0]))  # BGR for OpenCV
            else:
                # Default to green
                color = (0, 255, 0)
                
            # Draw the point
            cv2.circle(vis_image, (x, y), int(self.point_size), color, -1)
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'LiDAR-Camera Fusion: {projected_points.shape[0]} Points')
        plt.axis('off')
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
            logger.info(f"Visualization saved to {save_path}")
            
        plt.show()
    
    def visualize_calibration_results(self, images: List[np.ndarray], corners: List[np.ndarray], 
                                    pattern_size: Tuple[int, int], save_dir: Optional[str] = None):
        """Visualize camera calibration results.
        
        Args:
            images: List of calibration images
            corners: List of detected checkerboard corners for each image
            pattern_size: Size of the checkerboard pattern (width, height)
            save_dir: Directory to save visualization images. If None, images are not saved.
        """
        if len(images) == 0 or len(corners) == 0:
            logger.warning("No images or corners to visualize")
            return
            
        if len(images) != len(corners):
            logger.error("Number of images and corners lists must match")
            return
            
        # Create directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Visualize each image with detected corners
        for i, (image, img_corners) in enumerate(zip(images, corners)):
            # Create a copy for visualization
            vis_image = image.copy()
            
            # Draw the corners
            cv2.drawChessboardCorners(vis_image, pattern_size, img_corners, True)
            
            # Add image number
            cv2.putText(vis_image, f"Image {i+1}/{len(images)}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Calibration Image {i+1}: Corner Detection')
            plt.axis('off')
            
            # Save if directory provided
            if save_dir:
                output_path = os.path.join(save_dir, f"calibration_image_{i+1}.jpg")
                cv2.imwrite(output_path, vis_image)
                
            plt.show(block=False)
            plt.pause(0.5)  # Short pause to allow viewing multiple images
            
        plt.show()  # Final show to block until closed
        
    def clear(self):
        """Clear all plots."""
        plt.close('all') 