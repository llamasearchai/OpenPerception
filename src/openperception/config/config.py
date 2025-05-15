"""
Configuration module for OpenPerception.

This module provides utilities for loading, validating, and accessing
configuration settings across the framework.
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Type, TypeVar, cast

import toml  # For TOML parsing

logger = logging.getLogger(__name__)

# Type variable for generic Config classes
T = TypeVar('T', bound='Config')

# Default config paths
DEFAULT_CONFIG_FILE = "config/default_config.toml"
USER_CONFIG_FILE = "config/config.toml"
ENV_CONFIG_PATH = "OPENPERCEPTION_CONFIG"

@dataclass
class GeneralConfig:
    """General configuration settings."""
    version: str = "0.1.0"
    log_level: str = "INFO"
    output_dir: str = "output"
    enable_gpu: bool = True
    gpu_device: int = 0

@dataclass
class SLAMConfig:
    """SLAM configuration settings."""
    enabled: bool = True
    feature_detector: str = "ORB"
    max_features: int = 2000
    min_features_for_initialization: int = 100
    min_matches_for_pose_estimation: int = 15
    max_frames_between_keyframes: int = 20
    keyframe_distance_threshold: float = 0.5
    ransac_threshold_pixels: float = 1.0
    max_mapping_threads: int = 2
    mapping_rate: int = 5
    matcher_cross_check: bool = True

@dataclass
class SfMConfig:
    """Structure from Motion configuration settings."""
    enabled: bool = True
    feature_detector: str = "SIFT"
    max_features: int = 5000
    min_matches_for_reconstruction: int = 20
    bundle_adjustment_max_iterations: int = 100
    matcher_algorithm: str = "FLANN"
    filter_point_cloud: bool = True
    max_reprojection_error: float = 4.0

@dataclass
class SensorFusionConfig:
    """Sensor fusion configuration settings."""
    enabled: bool = True
    imu_weight: float = 0.7
    camera_weight: float = 0.3
    max_prediction_time: float = 0.5
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1
    buffer_size: int = 200

@dataclass
class ROS2InterfaceConfig:
    """ROS2 interface configuration settings."""
    enabled: bool = False
    node_name: str = "openperception_node"
    publish_map_topic: str = "/openperception/map"
    publish_pose_topic: str = "/openperception/pose"
    subscribe_camera_topic: str = "/camera/image_raw"
    subscribe_depth_topic: str = "/camera/depth/image_rect_raw"
    subscribe_imu_topic: str = "/imu/data"
    subscribe_lidar_topic: str = "/points2"
    subscribe_gps_topic: str = "/gps/fix"
    qos_profile: str = "SENSOR_DATA"

@dataclass
class WebServiceConfig:
    """Web service configuration settings."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class MissionPlannerConfig:
    """Mission planner configuration settings."""
    enabled: bool = True
    use_nlp: bool = True
    max_waypoints: int = 100
    safety_margin: float = 5.0
    obstacle_avoidance_threshold: float = 2.0
    default_altitude: float = 10.0
    default_speed: float = 5.0

@dataclass
class DeploymentTargetConfig:
    """Deployment target configuration."""
    ip: str = "192.168.1.100"
    username: str = "jetson"
    ssh_key: str = "~/.ssh/id_rsa"
    deploy_path: str = "/home/jetson/OpenPerception"

@dataclass
class DeploymentDependenciesConfig:
    """Deployment dependencies configuration."""
    apt: List[str] = field(default_factory=list)
    pip: List[str] = field(default_factory=list)

@dataclass
class DeploymentServicesConfig:
    """Deployment services configuration."""
    enable_systemd: bool = True
    service_name: str = "openperception"
    user: str = "jetson"
    startup: bool = True

@dataclass
class DeploymentLoggingConfig:
    """Deployment logging configuration."""
    log_path: str = "/var/log/openperception"
    log_level: str = "INFO"

@dataclass
class DeploymentConfig:
    """Deployment configuration settings."""
    enable_tensorrt: bool = False
    pytorch_version: str = ""
    target: DeploymentTargetConfig = field(default_factory=DeploymentTargetConfig)
    dependencies: DeploymentDependenciesConfig = field(default_factory=DeploymentDependenciesConfig)
    services: DeploymentServicesConfig = field(default_factory=DeploymentServicesConfig)
    logging: DeploymentLoggingConfig = field(default_factory=DeploymentLoggingConfig)

@dataclass
class CalibrationConfig:
    """Calibration configuration settings."""
    enabled: bool = True
    chessboard_size: Tuple[int, int] = (9, 6)
    square_size: float = 0.025
    min_images_for_calibration: int = 10

@dataclass
class DataPipelineConfig:
    """Data pipeline configuration settings."""
    dataset_dir: str = "datasets"
    image_width: int = 640
    image_height: int = 480
    image_format: str = "jpg"
    export_formats: List[str] = field(default_factory=lambda: ["coco", "yolo"])

@dataclass
class YOLOConfig:
    """YOLO model configuration."""
    model_type: str = "yolov5s"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)

@dataclass
class DeepLearningConfig:
    """Deep learning configuration settings."""
    enabled: bool = True
    model_dir: str = "models"
    batch_size: int = 8
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_split: float = 0.2
    augmentation: bool = True
    yolo: YOLOConfig = field(default_factory=YOLOConfig)

@dataclass
class VisualizationConfig:
    """Visualization configuration settings."""
    enabled: bool = True
    map_point_size: float = 2.0
    keyframe_size: float = 0.1
    trajectory_line_width: float = 2.0
    background_color: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    point_cloud_max_points: int = 100000

@dataclass
class BenchmarkingConfig:
    """Benchmarking configuration settings."""
    output_dir: str = "benchmarks_output"
    iterations: int = 3
    compare_with_previous: bool = True
    save_metrics: List[str] = field(default_factory=lambda: ["time", "memory", "cpu", "gpu"])
    save_charts: bool = True

@dataclass
class Config:
    """Main configuration class that combines all sub-configs."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    sensor_fusion: SensorFusionConfig = field(default_factory=SensorFusionConfig)
    ros2_interface: ROS2InterfaceConfig = field(default_factory=ROS2InterfaceConfig)
    web_service: WebServiceConfig = field(default_factory=WebServiceConfig)
    mission_planner: MissionPlannerConfig = field(default_factory=MissionPlannerConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    benchmarking: BenchmarkingConfig = field(default_factory=BenchmarkingConfig)
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Config instance
        """
        # Create a copy to avoid modifying the input dictionary
        config_dict = config_dict.copy()
        
        # Process the dictionary based on our nested dataclass structure
        # This recursively converts dict sections to appropriate dataclass instances
        config = cls()
        
        # Handle each section that exists in the input dict
        if "general" in config_dict:
            config.general = _dict_to_dataclass(GeneralConfig, config_dict["general"])
        
        if "slam" in config_dict:
            config.slam = _dict_to_dataclass(SLAMConfig, config_dict["slam"])
        
        if "sfm" in config_dict:
            config.sfm = _dict_to_dataclass(SfMConfig, config_dict["sfm"])
        
        if "sensor_fusion" in config_dict:
            config.sensor_fusion = _dict_to_dataclass(SensorFusionConfig, config_dict["sensor_fusion"])
        
        if "ros2_interface" in config_dict:
            config.ros2_interface = _dict_to_dataclass(ROS2InterfaceConfig, config_dict["ros2_interface"])
        
        if "web_service" in config_dict:
            config.web_service = _dict_to_dataclass(WebServiceConfig, config_dict["web_service"])
        
        if "mission_planner" in config_dict:
            config.mission_planner = _dict_to_dataclass(MissionPlannerConfig, config_dict["mission_planner"])
        
        if "deployment" in config_dict:
            deployment_dict = config_dict["deployment"]
            deployment = DeploymentConfig()
            
            # Copy top-level fields
            for key, value in deployment_dict.items():
                if key not in ["target", "dependencies", "services", "logging"]:
                    setattr(deployment, key, value)
            
            # Handle nested sections
            if "target" in deployment_dict:
                deployment.target = _dict_to_dataclass(DeploymentTargetConfig, deployment_dict["target"])
            
            if "dependencies" in deployment_dict:
                deployment.dependencies = _dict_to_dataclass(DeploymentDependenciesConfig, deployment_dict["dependencies"])
            
            if "services" in deployment_dict:
                deployment.services = _dict_to_dataclass(DeploymentServicesConfig, deployment_dict["services"])
            
            if "logging" in deployment_dict:
                deployment.logging = _dict_to_dataclass(DeploymentLoggingConfig, deployment_dict["logging"])
            
            config.deployment = deployment
        
        if "calibration" in config_dict:
            config.calibration = _dict_to_dataclass(CalibrationConfig, config_dict["calibration"])
        
        if "data_pipeline" in config_dict:
            config.data_pipeline = _dict_to_dataclass(DataPipelineConfig, config_dict["data_pipeline"])
        
        if "deep_learning" in config_dict:
            dl_dict = config_dict["deep_learning"]
            dl_config = DeepLearningConfig()
            
            # Copy top-level fields
            for key, value in dl_dict.items():
                if key != "yolo":
                    setattr(dl_config, key, value)
            
            # Handle YOLO config if present
            if "yolo" in dl_dict:
                dl_config.yolo = _dict_to_dataclass(YOLOConfig, dl_dict["yolo"])
            
            config.deep_learning = dl_config
        
        if "visualization" in config_dict:
            config.visualization = _dict_to_dataclass(VisualizationConfig, config_dict["visualization"])
        
        if "benchmarking" in config_dict:
            config.benchmarking = _dict_to_dataclass(BenchmarkingConfig, config_dict["benchmarking"])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return {
            "general": asdict(self.general),
            "slam": asdict(self.slam),
            "sfm": asdict(self.sfm),
            "sensor_fusion": asdict(self.sensor_fusion),
            "ros2_interface": asdict(self.ros2_interface),
            "web_service": asdict(self.web_service),
            "mission_planner": asdict(self.mission_planner),
            "deployment": {
                **{k: v for k, v in asdict(self.deployment).items() 
                   if k not in ["target", "dependencies", "services", "logging"]},
                "target": asdict(self.deployment.target),
                "dependencies": asdict(self.deployment.dependencies),
                "services": asdict(self.deployment.services),
                "logging": asdict(self.deployment.logging),
            },
            "calibration": asdict(self.calibration),
            "data_pipeline": asdict(self.data_pipeline),
            "deep_learning": {
                **{k: v for k, v in asdict(self.deep_learning).items() if k != "yolo"},
                "yolo": asdict(self.deep_learning.yolo),
            },
            "visualization": asdict(self.visualization),
            "benchmarking": asdict(self.benchmarking),
        }
    
    def get(self, section: str, default: Any = None) -> Any:
        """Get a configuration section.
        
        Args:
            section: Section name
            default: Default value if the section doesn't exist
            
        Returns:
            Configuration section or default
        """
        return getattr(self, section, default)

def _dict_to_dataclass(cls: Type[T], data_dict: Dict[str, Any]) -> T:
    """Convert a dictionary to a dataclass instance.
    
    Args:
        cls: Dataclass type
        data_dict: Dictionary with data
        
    Returns:
        Dataclass instance
    """
    # Create an instance with default values
    instance = cls()
    
    # Update fields from dictionary
    for key, value in data_dict.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            logger.warning(f"Unknown config field '{key}' for {cls.__name__}")
    
    return instance

def find_config_file(config_path: Optional[str] = None) -> Optional[str]:
    """Find the configuration file to use.
    
    Args:
        config_path: Path to the configuration file. If None, search for the config file.
        
    Returns:
        Path to the configuration file, or None if not found
    """
    # If a specific path is provided, use it
    if config_path:
        if os.path.exists(config_path):
            return config_path
        logger.warning(f"Specified config file not found: {config_path}")
        return None
    
    # Check environment variable
    env_path = os.environ.get(ENV_CONFIG_PATH)
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Check for user config
    if os.path.exists(USER_CONFIG_FILE):
        return USER_CONFIG_FILE
    
    # Use default config
    if os.path.exists(DEFAULT_CONFIG_FILE):
        return DEFAULT_CONFIG_FILE
    
    # Try relative to the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    default_path = os.path.join(project_root, DEFAULT_CONFIG_FILE)
    if os.path.exists(default_path):
        return default_path
    
    logger.warning("No configuration file found")
    return None

def load_config(config_path: Optional[str] = None) -> Config:
    """Load the configuration.
    
    Args:
        config_path: Path to the configuration file. If None, the default config is used.
        
    Returns:
        Loaded configuration
    """
    # Find config file
    config_file = find_config_file(config_path)
    
    if config_file:
        try:
            # Load TOML config
            with open(config_file, "r") as f:
                config_dict = toml.load(f)
            
            logger.info(f"Loaded configuration from {config_file}")
            return Config.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
    
    # Return default config if no file found or error loading
    logger.info("Using default configuration")
    return Config() 