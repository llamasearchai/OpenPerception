#!/usr/bin/env python3
"""
OpenPerception - Main Application

This module serves as the entry point for the OpenPerception framework, providing
a unified interface to all components and functionalities.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import cv2

# Import OpenPerception modules
from openperception.config import load_config, Config
from openperception.slam.visual_slam import VisualSLAM
from openperception.sfm.structure_from_motion import StructureFromMotion
from openperception.sensor_fusion.fusion import SensorFusion
from openperception.mission_planner.planner import MissionPlanner
from openperception.web_service.api import WebServiceAPI, set_openperception_app_instance
from openperception.benchmarking.benchmark import PerformanceBenchmark
from openperception.data_pipeline.dataset_manager import DatasetManager
from openperception.deployment.jetson_deployment import JetsonDeployment

# Optional imports
try:
    from openperception.ros2_interface.ros_bridge import ROS2Bridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Setup logging
logger = logging.getLogger("openperception")

class OpenPerception:
    """Main application class for OpenPerception framework"""

    def __init__(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        """Initialize OpenPerception framework
        
        Args:
            config_path: Path to the configuration file. If None, the default config is used.
            config_data: Configuration dictionary. If provided, overrides config_path.
        """
        # Load configuration
        self.config = load_config(config_path) if config_data is None else Config.from_dict(config_data)
        
        # Setup logging
        self._setup_logging()
        logger.info(f"OpenPerception v{self.__version__} initializing...")
        
        # Initialize components based on configuration
        self.slam_system = None
        self.sfm_system = None
        self.sensor_fusion = None
        self.mission_planner = None
        self.ros2_bridge = None
        self.web_service = None
        self.dataset_manager = None
        self.deep_learning_models = {}
        
        # Initialize enabled components
        self._initialize_components()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.general.output_dir, exist_ok=True)
        
        logger.info("OpenPerception framework initialized successfully.")
    
    @property
    def __version__(self) -> str:
        """Return the version of OpenPerception"""
        from openperception import __version__
        return __version__
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.general.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.config.general.output_dir, "openperception.log"), mode='a')
            ]
        )
    
    def _initialize_components(self):
        """Initialize components based on configuration"""
        # Initialize SLAM if enabled
        if self.config.slam.enabled:
            logger.info("Initializing VisualSLAM module...")
            # Actual initialization happens when run_slam_* methods are called
            # since we need camera parameters
        
        # Initialize SfM if enabled
        if self.config.sfm.enabled:
            logger.info("Initializing Structure from Motion module...")
            # Actual initialization happens when run_sfm_* methods are called
            # since we need camera parameters
        
        # Initialize Sensor Fusion if enabled
        if self.config.sensor_fusion.enabled:
            logger.info("Initializing Sensor Fusion module...")
            self.sensor_fusion = SensorFusion(self.config.sensor_fusion)
        
        # Initialize Mission Planner if enabled
        if self.config.mission_planner.enabled:
            logger.info("Initializing Mission Planner module...")
            self.mission_planner = MissionPlanner(self.config.mission_planner)
        
        # Initialize ROS2 Bridge if enabled and available
        if self.config.get("ros2_interface", {}).get("enabled", False) and ROS2_AVAILABLE:
            logger.info("Initializing ROS2 Bridge...")
            self.ros2_bridge = ROS2Bridge(node_name=self.config.ros2_interface.node_name)
        
        # Initialize Dataset Manager if data pipeline is configured
        if hasattr(self.config, "data_pipeline") and getattr(self.config.data_pipeline, "dataset_dir", None):
            logger.info("Initializing Dataset Manager...")
            self.dataset_manager = DatasetManager(self.config.data_pipeline.dataset_dir)
        
        # Initialize Web Service if enabled
        if self.config.web_service.enabled:
            logger.info("Initializing Web Service API...")
            self.web_service = WebServiceAPI(self, self.config.web_service)
            # Register this instance with the FastAPI routes
            set_openperception_app_instance(self)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the OpenPerception framework
        
        Returns:
            Dictionary with status information
        """
        status = {
            "version": self.__version__,
            "active_modules": [],
            "status": "running"
        }
        
        if self.slam_system:
            status["active_modules"].append("SLAM")
        if self.sfm_system:
            status["active_modules"].append("SfM")
        if self.sensor_fusion:
            status["active_modules"].append("SensorFusion")
        if self.mission_planner:
            status["active_modules"].append("MissionPlanner")
        if self.ros2_bridge:
            status["active_modules"].append("ROS2Bridge")
        if self.web_service:
            status["active_modules"].append("WebService")
        
        return status
    
    def run_slam_from_video(self, video_path: str, output_dir: Optional[str] = None, 
                           camera_matrix: Optional[np.ndarray] = None, 
                           dist_coeffs: Optional[np.ndarray] = None,
                           benchmark_mode: bool = False) -> Dict[str, Any]:
        """Run Visual SLAM on a video file
        
        Args:
            video_path: Path to the video file
            output_dir: Output directory for results. If None, uses config default
            camera_matrix: Optional camera matrix (3x3). If None, estimated from video
            dist_coeffs: Optional distortion coefficients. If None, assumes no distortion
            benchmark_mode: If True, runs in benchmark mode without saving results
            
        Returns:
            Dictionary with results information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.config.general.output_dir, "slam", 
                                     f"{Path(video_path).stem}_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Estimate camera matrix if not provided
        if camera_matrix is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Use a default camera matrix based on the video dimensions
            camera_matrix = np.array([
                [width, 0, width/2],
                [0, width, height/2],
                [0, 0, 1]
            ])
            logger.warning(f"Using estimated camera matrix for {video_path}")
        
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)
        
        # Initialize SLAM
        self.slam_system = VisualSLAM(camera_matrix, dist_coeffs, self.config.slam)
        
        # Process video frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get timestamp (use frame count / fps as an approximation)
            timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
            
            # Process frame
            pose = self.slam_system.process_frame(frame, timestamp)
            
            # Update progress
            if frame_count % 10 == 0:
                logger.info(f"Processed frame {frame_count}")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"SLAM processing completed: {frame_count} frames processed")
        
        # Save results if not in benchmark mode
        results = {"output_dir": output_dir, "frames_processed": frame_count}
        
        if not benchmark_mode:
            # Get trajectory and map
            trajectory = self.slam_system.get_current_trajectory()
            map_points, keyframe_poses = self.slam_system.get_map_points_and_poses()
            
            # Save trajectory as a simple CSV
            if trajectory:
                trajectory_path = os.path.join(output_dir, "trajectory.csv")
                with open(trajectory_path, "w") as f:
                    f.write("tx,ty,tz,qx,qy,qz,qw\n")
                    for pose in trajectory:
                        # Convert to position and quaternion
                        tx, ty, tz = pose[:3, 3]
                        try:
                            from scipy.spatial.transform import Rotation
                            r = Rotation.from_matrix(pose[:3, :3])
                            qx, qy, qz, qw = r.as_quat()
                        except ImportError:
                            # Simplified quaternion conversion (not accurate but a fallback)
                            qx, qy, qz, qw = 0, 0, 0, 1
                        
                        f.write(f"{tx},{ty},{tz},{qx},{qy},{qz},{qw}\n")
                
                results["trajectory_path"] = trajectory_path
            
            # Save map points if available
            if map_points is not None:
                map_path = os.path.join(output_dir, "map_points.csv")
                np.savetxt(map_path, map_points, delimiter=",", header="x,y,z")
                results["map_path"] = map_path
        
        return results
    
    def run_sfm_from_images(self, image_dir: str, output_dir: Optional[str] = None,
                           camera_matrix: Optional[np.ndarray] = None,
                           dist_coeffs: Optional[np.ndarray] = None,
                           benchmark_mode: bool = False) -> Dict[str, Any]:
        """Run Structure from Motion on a directory of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory for results. If None, uses config default
            camera_matrix: Optional camera matrix (3x3)
            dist_coeffs: Optional distortion coefficients
            benchmark_mode: If True, runs in benchmark mode without saving results
            
        Returns:
            Dictionary with results information
        """
        # Implementation would be similar to run_slam_from_video but for SfM
        # This is a placeholder function; actual implementation would be more complex
        logger.info(f"Running SfM on image directory: {image_dir}")
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.config.general.output_dir, "sfm", 
                                    f"{Path(image_dir).name}_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize SfM
        # Actual implementation would scan for images, sort them, etc.
        
        return {"output_dir": output_dir, "status": "success"}

    def create_mission_plan(self, mission_description: str, environment_description: str, 
                           constraints: str) -> str:
        """Create a mission plan using the mission planner
        
        Args:
            mission_description: Description of the mission
            environment_description: Description of the environment
            constraints: Mission constraints
            
        Returns:
            Mission plan as a string
        """
        if not self.mission_planner:
            raise RuntimeError("Mission Planner is not initialized or enabled in config")
        
        return self.mission_planner.create_mission_plan(
            mission_description, environment_description, constraints
        )
    
    def run_server(self):
        """Run the web service server"""
        if not self.web_service:
            raise RuntimeError("Web Service is not initialized or enabled in config")
        
        logger.info("Starting OpenPerception Web Service...")
        self.web_service.run_server()
    
    def shutdown(self):
        """Shutdown all components"""
        logger.info("Shutting down OpenPerception...")
        
        # Shutdown components in reverse initialization order
        if self.web_service:
            logger.info("Shutting down Web Service...")
            self.web_service.cleanup()
        
        if self.ros2_bridge:
            logger.info("Shutting down ROS2 Bridge...")
            self.ros2_bridge.stop()
        
        if self.slam_system:
            logger.info("Shutting down SLAM system...")
            self.slam_system.shutdown()
        
        logger.info("OpenPerception shutdown complete.")

def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="OpenPerception: Computer Vision and Perception Framework for Aerial Robotics")
    
    # Main mode arguments
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Web service command
    web_parser = subparsers.add_parser("web", help="Run the web service")
    web_parser.add_argument("--host", type=str, help="Web service host")
    web_parser.add_argument("--port", type=int, help="Web service port")
    
    # SLAM command
    slam_parser = subparsers.add_parser("slam", help="Run SLAM on a video file")
    slam_parser.add_argument("video_path", type=str, help="Path to the video file")
    slam_parser.add_argument("--output-dir", type=str, help="Output directory for results")
    
    # SfM command
    sfm_parser = subparsers.add_parser("sfm", help="Run Structure from Motion on images")
    sfm_parser.add_argument("image_dir", type=str, help="Directory containing images")
    sfm_parser.add_argument("--output-dir", type=str, help="Output directory for results")
    
    # Mission planning command
    mission_parser = subparsers.add_parser("mission", help="Create a mission plan")
    mission_parser.add_argument("--description", type=str, required=True, help="Mission description")
    mission_parser.add_argument("--environment", type=str, required=True, help="Environment description")
    mission_parser.add_argument("--constraints", type=str, required=True, help="Mission constraints")
    
    # General arguments
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Create OpenPerception instance
    app = OpenPerception(config_path=args.config)
    
    # Override log level if specified
    if args.log_level:
        logging.getLogger().setLevel(args.log_level)
    
    try:
        # Execute command
        if args.command == "web":
            # Override web service settings if provided
            if args.host:
                app.config.web_service.host = args.host
            if args.port:
                app.config.web_service.port = args.port
            
            app.run_server()
            
        elif args.command == "slam":
            results = app.run_slam_from_video(args.video_path, args.output_dir)
            print(f"SLAM processing completed. Results saved to: {results['output_dir']}")
            
        elif args.command == "sfm":
            results = app.run_sfm_from_images(args.image_dir, args.output_dir)
            print(f"SfM processing completed. Results saved to: {results['output_dir']}")
            
        elif args.command == "mission":
            plan = app.create_mission_plan(args.description, args.environment, args.constraints)
            print("Mission Plan:")
            print(plan)
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    finally:
        app.shutdown()

if __name__ == "__main__":
    main() 