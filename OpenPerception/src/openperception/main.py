#!/usr/bin/env python3
"""
OpenPerception: Computer vision and perception framework for aerial robotics and drones
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import time
import cv2
import numpy as np

from .config import load_config, Config
from .slam.visual_slam import VisualSLAM
from .sfm.structure_from_motion import StructureFromMotion
from .sensor_fusion.fusion import SensorFusion
from .web_service.api import WebService
from .data_pipeline.dataset_manager import DatasetManager
from .mission_planner.planner import MissionPlanner
from .ros2_interface.ros_bridge import ROS2Bridge
from .deployment.jetson_deployment import JetsonDeployment
from .utils.transformations import euler_to_rotation_matrix
from .visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

class OpenPerception:
    """Main interface for OpenPerception framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the OpenPerception framework.
        
        Args:
            config_path: Path to configuration file. If None, default configuration is used.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Configure logging
        self._configure_logging()
        
        # Initialize components based on configuration
        self._components = {}
        self._initialize_components()
        
    def _configure_logging(self):
        """Configure logging based on configuration."""
        log_level = getattr(logging, self.config.general.log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('openperception.log')
            ]
        )
        
        logger.info(f"Logging configured with level {self.config.general.log_level}")
        
    def _initialize_components(self):
        """Initialize components based on configuration."""
        # Create output directories
        os.makedirs(self.config.general.data_dir, exist_ok=True)
        os.makedirs(self.config.general.output_dir, exist_ok=True)
        
        # Initialize SLAM if enabled
        if self.config.slam.enabled:
            try:
                self._components['slam'] = VisualSLAM(
                    feature_type=self.config.slam.feature_type,
                    num_features=self.config.slam.num_features,
                    use_gpu=self.config.slam.use_gpu
                )
                logger.info("SLAM component initialized")
            except Exception as e:
                logger.error(f"Error initializing SLAM: {e}")
        
        # Initialize SfM if enabled
        if self.config.sfm.enabled:
            try:
                self._components['sfm'] = StructureFromMotion(
                    feature_type=self.config.sfm.feature_type,
                    matcher_type=self.config.sfm.matcher_type,
                    use_gpu=self.config.sfm.use_gpu
                )
                logger.info("SfM component initialized")
            except Exception as e:
                logger.error(f"Error initializing SfM: {e}")
        
        # Initialize Sensor Fusion if enabled
        if self.config.sensor_fusion.enabled:
            try:
                self._components['sensor_fusion'] = SensorFusion(
                    max_buffer_size=self.config.sensor_fusion.max_buffer_size
                )
                logger.info("Sensor Fusion component initialized")
            except Exception as e:
                logger.error(f"Error initializing Sensor Fusion: {e}")
        
        # Initialize Web Service if enabled
        if self.config.web_service.enabled:
            try:
                self._components['web_service'] = WebService(
                    host=self.config.web_service.host,
                    port=self.config.web_service.port,
                    enable_cors=self.config.web_service.enable_cors,
                    allowed_origins=self.config.web_service.allowed_origins,
                    workers=self.config.web_service.workers
                )
                logger.info("Web Service component initialized")
            except Exception as e:
                logger.error(f"Error initializing Web Service: {e}")
        
        # Initialize Data Pipeline if enabled
        if self.config.data_pipeline.enabled:
            try:
                self._components['data_pipeline'] = DatasetManager(
                    dataset_dir=self.config.data_pipeline.dataset_dir,
                    auto_annotate=self.config.data_pipeline.auto_annotate,
                    annotation_confidence=self.config.data_pipeline.annotation_confidence
                )
                logger.info("Data Pipeline component initialized")
            except Exception as e:
                logger.error(f"Error initializing Data Pipeline: {e}")
        
        # Initialize Mission Planner if enabled
        if self.config.mission_planner.enabled:
            try:
                self._components['mission_planner'] = MissionPlanner(
                    openai_api_key=self.config.mission_planner.openai_api_key
                )
                logger.info("Mission Planner component initialized")
            except Exception as e:
                logger.error(f"Error initializing Mission Planner: {e}")
        
        # Initialize ROS Interface if enabled
        if self.config.ros_interface.enabled:
            try:
                self._components['ros_interface'] = ROS2Bridge(
                    node_name=self.config.ros_interface.node_name,
                    use_composition=self.config.ros_interface.use_composition
                )
                logger.info("ROS Interface component initialized")
            except Exception as e:
                logger.error(f"Error initializing ROS Interface: {e}")
        
        # Initialize Visualization if enabled
        if self.config.visualization.enabled:
            try:
                self._components['visualizer'] = Visualizer(
                    max_points=self.config.visualization.max_points,
                    point_size=self.config.visualization.point_size,
                    show_trajectory=self.config.visualization.show_trajectory
                )
                logger.info("Visualization component initialized")
            except Exception as e:
                logger.error(f"Error initializing Visualization: {e}")
    
    def run_slam_from_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run SLAM on a video file.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results. If None, uses config.general.output_dir
            
        Returns:
            Dictionary with SLAM results
        """
        if 'slam' not in self._components:
            logger.error("SLAM component not initialized")
            return {'error': 'SLAM component not initialized'}
        
        slam = self._components['slam']
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {'error': f'Video file not found: {video_path}'}
        
        if output_dir is None:
            output_dir = os.path.join(self.config.general.output_dir, 'slam_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return {'error': f'Error opening video file: {video_path}'}
        
        # Process video frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with SLAM
            slam.process_frame(frame)
            frame_count += 1
            
            # Optionally visualize progress
            if self.config.general.debug and frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        # Get SLAM results
        trajectory = slam.get_trajectory()
        map_points = slam.get_map_points()
        
        # Save results
        np.save(os.path.join(output_dir, 'trajectory.npy'), trajectory)
        np.save(os.path.join(output_dir, 'map_points.npy'), map_points)
        
        # Optionally visualize results
        if self.config.visualization.enabled and 'visualizer' in self._components:
            visualizer = self._components['visualizer']
            visualizer.visualize_slam_results(trajectory, map_points)
        
        logger.info(f"SLAM completed on {video_path}. Processed {frame_count} frames.")
        logger.info(f"Results saved to {output_dir}")
        
        return {
            'trajectory': trajectory,
            'map_points': map_points,
            'frame_count': frame_count,
            'output_dir': output_dir
        }
    
    def run_sfm_from_images(self, image_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run Structure from Motion on a directory of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results. If None, uses config.general.output_dir
            
        Returns:
            Dictionary with SfM results
        """
        if 'sfm' not in self._components:
            logger.error("SfM component not initialized")
            return {'error': 'SfM component not initialized'}
        
        sfm = self._components['sfm']
        
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return {'error': f'Image directory not found: {image_dir}'}
        
        if output_dir is None:
            output_dir = os.path.join(self.config.general.output_dir, 'sfm_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of images
        image_files = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not image_files:
            logger.error(f"No image files found in {image_dir}")
            return {'error': f'No image files found in {image_dir}'}
        
        # Process images with SfM
        for i, image_file in enumerate(image_files):
            # Load image
            image = cv2.imread(image_file)
            if image is None:
                logger.warning(f"Error loading image: {image_file}")
                continue
            
            # Add image to SfM
            sfm.add_image(image, image_file)
            
            # Optionally visualize progress
            if self.config.general.debug and (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        # Run reconstruction
        reconstruction = sfm.reconstruct()
        
        # Save results
        np.save(os.path.join(output_dir, 'cameras.npy'), reconstruction['cameras'])
        np.save(os.path.join(output_dir, 'points3d.npy'), reconstruction['points3d'])
        
        # Optionally visualize results
        if self.config.visualization.enabled and 'visualizer' in self._components:
            visualizer = self._components['visualizer']
            visualizer.visualize_sfm_results(reconstruction['cameras'], reconstruction['points3d'])
        
        logger.info(f"SfM completed on {image_dir}. Processed {len(image_files)} images.")
        logger.info(f"Results saved to {output_dir}")
        
        return reconstruction
    
    def start_web_service(self):
        """Start the web service."""
        if 'web_service' not in self._components:
            logger.error("Web Service component not initialized")
            return {'error': 'Web Service component not initialized'}
        
        web_service = self._components['web_service']
        web_service.start()
    
    def create_mission_plan(self, mission_description: str, environment_description: str, 
                          constraints: str) -> str:
        """Create a mission plan.
        
        Args:
            mission_description: Description of the mission
            environment_description: Description of the environment
            constraints: Constraints for the mission
            
        Returns:
            Mission plan as a string
        """
        if 'mission_planner' not in self._components:
            logger.error("Mission Planner component not initialized")
            return "Error: Mission Planner component not initialized"
        
        mission_planner = self._components['mission_planner']
        return mission_planner.create_mission_plan(
            mission_description, environment_description, constraints
        )
    
    def deploy_to_jetson(self, target_ip: Optional[str] = None, 
                       username: Optional[str] = None) -> Dict[str, Any]:
        """Deploy the application to a Jetson device.
        
        Args:
            target_ip: Target IP address. If None, uses config.deployment.target.ip
            username: Target username. If None, uses config.deployment.target.username
            
        Returns:
            Dictionary with deployment results
        """
        try:
            deployment = JetsonDeployment(config_path=None)  # Use config from self.config
            
            # Override config if provided
            if target_ip:
                deployment.config["target"]["ip"] = target_ip
            if username:
                deployment.config["target"]["username"] = username
            
            # Prepare deployment package
            output_dir = os.path.join(self.config.general.output_dir, 'deployment')
            deployment.prepare_deployment_package(output_dir=output_dir)
            
            # TODO: Implement actual deployment to device
            # This would typically involve SCP/SSH to copy files and run setup script
            
            logger.info(f"Deployment package prepared at {output_dir}")
            logger.info(f"To deploy, copy the package to the Jetson device and run setup.sh")
            
            return {
                'status': 'prepared',
                'output_dir': output_dir
            }
        except Exception as e:
            logger.error(f"Error preparing deployment: {e}")
            return {'error': f'Error preparing deployment: {e}'}
    
    def get_component(self, name: str) -> Any:
        """Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component object
        """
        return self._components.get(name)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='OpenPerception: Computer vision and perception framework')
    
    # General arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # SLAM command
    slam_parser = subparsers.add_parser('slam', help='Run SLAM on a video file')
    slam_parser.add_argument('video', type=str, help='Path to video file')
    slam_parser.add_argument('--output', type=str, help='Directory to save results')
    
    # SfM command
    sfm_parser = subparsers.add_parser('sfm', help='Run Structure from Motion on images')
    sfm_parser.add_argument('images', type=str, help='Directory containing images')
    sfm_parser.add_argument('--output', type=str, help='Directory to save results')
    
    # Web service command
    web_parser = subparsers.add_parser('web', help='Start the web service')
    
    # Mission planner command
    mission_parser = subparsers.add_parser('mission', help='Create a mission plan')
    mission_parser.add_argument('--mission', type=str, required=True, help='Mission description')
    mission_parser.add_argument('--environment', type=str, required=True, help='Environment description')
    mission_parser.add_argument('--constraints', type=str, required=True, help='Constraints')
    
    # Deployment command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to Jetson')
    deploy_parser.add_argument('--ip', type=str, help='Target IP address')
    deploy_parser.add_argument('--username', type=str, help='Target username')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create OpenPerception instance
    app = OpenPerception(config_path=args.config)
    
    # Override debug mode if specified
    if args.debug:
        app.config.general.debug = True
    
    # Run command
    if args.command == 'slam':
        app.run_slam_from_video(args.video, args.output)
    elif args.command == 'sfm':
        app.run_sfm_from_images(args.images, args.output)
    elif args.command == 'web':
        app.start_web_service()
    elif args.command == 'mission':
        plan = app.create_mission_plan(args.mission, args.environment, args.constraints)
        print(plan)
    elif args.command == 'deploy':
        app.deploy_to_jetson(args.ip, args.username)
    else:
        print("No command specified. Use --help to see available commands.")

if __name__ == '__main__':
    main() 