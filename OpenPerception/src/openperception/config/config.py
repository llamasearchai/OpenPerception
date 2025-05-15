"""
Configuration management for OpenPerception.
"""

import os
import toml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Type, TypeVar, cast
import logging

logger = logging.getLogger(__name__)

# Define the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

# Default config paths
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default_config.toml"

# Type variable for dataclass types
T = TypeVar('T')

@dataclass
class SLAMConfig:
    """Configuration for SLAM module."""
    enabled: bool = True
    use_gpu: bool = True
    num_features: int = 1000
    feature_type: str = "orb"
    max_keyframes: int = 20
    keyframe_threshold: int = 30

@dataclass
class SfMConfig:
    """Configuration for Structure from Motion module."""
    enabled: bool = True
    use_gpu: bool = True
    feature_type: str = "sift"
    matcher_type: str = "flann"
    min_matches: int = 15
    triangulation_threshold: float = 2.0

@dataclass
class SensorFusionConfig:
    """Configuration for Sensor Fusion module."""
    enabled: bool = True
    max_buffer_size: int = 100
    fusion_method: str = "kalman"
    predict_frequency: int = 30
    default_camera_intrinsics: List[List[float]] = field(default_factory=lambda: [
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    default_distortion_coeffs: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    default_camera_extrinsics: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    default_lidar_extrinsics: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@dataclass
class CalibrationConfig:
    """Configuration for Calibration module."""
    checkerboard_size: List[int] = field(default_factory=lambda: [9, 6])
    square_size: float = 0.025
    auto_detect: bool = True
    min_images: int = 15

@dataclass
class WebServiceConfig:
    """Configuration for Web Service module."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:1420"])
    workers: int = 4

@dataclass
class DataPipelineConfig:
    """Configuration for Data Pipeline module."""
    enabled: bool = True
    dataset_dir: str = "datasets"
    auto_annotate: bool = True
    annotation_confidence: float = 0.7

@dataclass
class DeepLearningConfig:
    """Configuration for Deep Learning module."""
    model_path: str = "models"
    batch_size: int = 8
    use_cuda: bool = True
    fp16: bool = False
    input_size: List[int] = field(default_factory=lambda: [224, 224])
    pretrained: bool = True
    backbone: str = "resnet50"

@dataclass
class MissionPlannerConfig:
    """Configuration for Mission Planner module."""
    enabled: bool = True
    openai_api_key: str = ""
    planning_algorithm: str = "rrt_star"
    replanning_interval: int = 5

@dataclass
class RosInterfaceConfig:
    """Configuration for ROS Interface module."""
    enabled: bool = False
    node_name: str = "open_perception_node"
    use_composition: bool = True
    topics: List[Dict[str, str]] = field(default_factory=lambda: [])

@dataclass
class VisualizationConfig:
    """Configuration for Visualization module."""
    enabled: bool = True
    max_points: int = 100000
    point_size: float = 2.0
    show_trajectory: bool = True

@dataclass
class DeploymentConfig:
    """Configuration for Deployment module."""
    target: Dict[str, str] = field(default_factory=lambda: {
        "ip": "192.168.1.100",
        "username": "jetson",
        "ssh_key": "~/.ssh/id_rsa",
        "deploy_path": "/home/jetson/open_perception"
    })
    dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        "apt": [
            "python3-pip",
            "python3-dev",
            "build-essential",
            "cmake",
            "libopencv-dev",
            "libsm6",
            "libxext6"
        ],
        "pip": [
            "numpy",
            "scipy",
            "opencv-python",
            "torch",
            "fastapi",
            "uvicorn"
        ]
    })
    optimization: Dict[str, bool] = field(default_factory=lambda: {
        "enable_tensorrt": True,
        "enable_cuda": True,
        "enable_cudnn": True
    })

@dataclass
class PathPlanningConfig:
    """Configuration for Path Planning module."""
    algorithm: str = "rrt_star"
    max_iterations: int = 1000
    goal_sample_rate: float = 0.1
    step_size: float = 0.2
    goal_threshold: float = 0.5
    obstacle_threshold: float = 0.8

@dataclass
class GeneralConfig:
    """General configuration for the application."""
    data_dir: str = "data"
    output_dir: str = "output"
    debug: bool = False
    log_level: str = "INFO"

@dataclass
class Config:
    """Main configuration class."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    sensor_fusion: SensorFusionConfig = field(default_factory=SensorFusionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    web_service: WebServiceConfig = field(default_factory=WebServiceConfig)
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    mission_planner: MissionPlannerConfig = field(default_factory=MissionPlannerConfig)
    ros_interface: RosInterfaceConfig = field(default_factory=RosInterfaceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config object from dictionary."""
        config = cls()

        def update_dataclass(dataclass_obj: T, data: Dict[str, Any]) -> T:
            """Update dataclass from dictionary."""
            for key, value in data.items():
                if hasattr(dataclass_obj, key):
                    setattr(dataclass_obj, key, value)
            return dataclass_obj

        if 'general' in config_dict:
            config.general = update_dataclass(config.general, config_dict['general'])
        if 'slam' in config_dict:
            config.slam = update_dataclass(config.slam, config_dict['slam'])
        if 'sfm' in config_dict:
            config.sfm = update_dataclass(config.sfm, config_dict['sfm'])
        if 'sensor_fusion' in config_dict:
            config.sensor_fusion = update_dataclass(config.sensor_fusion, config_dict['sensor_fusion'])
        if 'calibration' in config_dict:
            config.calibration = update_dataclass(config.calibration, config_dict['calibration'])
        if 'web_service' in config_dict:
            config.web_service = update_dataclass(config.web_service, config_dict['web_service'])
        if 'data_pipeline' in config_dict:
            config.data_pipeline = update_dataclass(config.data_pipeline, config_dict['data_pipeline'])
        if 'deep_learning' in config_dict:
            config.deep_learning = update_dataclass(config.deep_learning, config_dict['deep_learning'])
        if 'mission_planner' in config_dict:
            config.mission_planner = update_dataclass(config.mission_planner, config_dict['mission_planner'])
        if 'ros_interface' in config_dict:
            config.ros_interface = update_dataclass(config.ros_interface, config_dict['ros_interface'])
        if 'visualization' in config_dict:
            config.visualization = update_dataclass(config.visualization, config_dict['visualization'])
        if 'deployment' in config_dict:
            config.deployment = update_dataclass(config.deployment, config_dict['deployment'])
        if 'path_planning' in config_dict:
            config.path_planning = update_dataclass(config.path_planning, config_dict['path_planning'])

        return config

    @classmethod
    def from_toml(cls, toml_path: Union[str, Path]) -> 'Config':
        """Load configuration from TOML file."""
        toml_path = Path(toml_path)
        if not toml_path.exists():
            logger.warning(f"Config file {toml_path} not found. Using default configuration.")
            return cls()

        try:
            with open(toml_path, 'r') as f:
                config_dict = toml.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {toml_path}: {e}")
            logger.info("Using default configuration.")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'general': self._dataclass_to_dict(self.general),
            'slam': self._dataclass_to_dict(self.slam),
            'sfm': self._dataclass_to_dict(self.sfm),
            'sensor_fusion': self._dataclass_to_dict(self.sensor_fusion),
            'calibration': self._dataclass_to_dict(self.calibration),
            'web_service': self._dataclass_to_dict(self.web_service),
            'data_pipeline': self._dataclass_to_dict(self.data_pipeline),
            'deep_learning': self._dataclass_to_dict(self.deep_learning),
            'mission_planner': self._dataclass_to_dict(self.mission_planner),
            'ros_interface': self._dataclass_to_dict(self.ros_interface),
            'visualization': self._dataclass_to_dict(self.visualization),
            'deployment': self._dataclass_to_dict(self.deployment),
            'path_planning': self._dataclass_to_dict(self.path_planning),
        }

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items()}
        return obj

    def save(self, toml_path: Union[str, Path]) -> None:
        """Save configuration to TOML file."""
        toml_path = Path(toml_path)
        toml_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(toml_path, 'w') as f:
                toml.dump(self.to_dict(), f)
            logger.info(f"Configuration saved to {toml_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {toml_path}: {e}")

# Global configuration object
_CONFIG: Optional[Config] = None

def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get global configuration object."""
    global _CONFIG

    if _CONFIG is None:
        # If config_path is provided, use it
        if config_path:
            _CONFIG = Config.from_toml(config_path)
        else:
            # Try default path
            _CONFIG = Config.from_toml(DEFAULT_CONFIG_PATH)

    return _CONFIG

def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file."""
    return get_config(config_path) 