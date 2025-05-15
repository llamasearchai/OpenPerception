import os
import yaml
import toml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union

@dataclass
class SLAMConfig:
    enabled: bool = True
    use_gpu: bool = True
    num_features: int = 2000
    feature_type: str = "ORB"
    ransac_threshold: float = 2.5
    keyframe_threshold: float = 0.6
    loop_closure_enabled: bool = True
    mapping_rate: int = 5  # Process mapping every N frames
    max_keyframes: int = 20

@dataclass
class SensorConfig:
    camera_topic: str = "/camera/image_raw"
    lidar_topic: str = "/lidar/points"
    imu_topic: str = "/imu/data"
    depth_topic: str = "/camera/depth/image_raw"
    camera_info_topic: str = "/camera/camera_info"
    
@dataclass
class DeepLearningConfig:
    model_path: str = "models"
    batch_size: int = 8
    use_cuda: bool = True
    fp16: bool = False
    
@dataclass
class WebServiceConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    enable_cors: bool = True
    allowed_origins: List[str] = None
    workers: int = 4
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:1420"]
    
@dataclass
class ROS2Config:
    enabled: bool = False
    node_name: str = "open_perception_node"
    use_sim_time: bool = False
    use_composition: bool = True
    
@dataclass
class SfMConfig:
    enabled: bool = True
    use_gpu: bool = True
    feature_type: str = "sift"
    matcher_type: str = "flann"
    min_matches: int = 15
    triangulation_threshold: float = 2.0

@dataclass
class DataPipelineConfig:
    enabled: bool = True
    dataset_dir: str = "datasets"
    auto_annotate: bool = True
    annotation_confidence: float = 0.7
    
@dataclass
class MissionPlannerConfig:
    enabled: bool = True
    openai_api_key: Optional[str] = None
    use_local_model: bool = False
    local_model_path: Optional[str] = None
    planning_algorithm: str = "rrt_star"
    replanning_interval: int = 5
    
@dataclass
class VisualizationConfig:
    enabled: bool = True
    max_points: int = 100000
    point_size: float = 2.0
    show_trajectory: bool = True
    
@dataclass
class GeneralConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    debug: bool = False
    log_level: str = "INFO"

@dataclass
class Config:
    general: GeneralConfig = GeneralConfig()
    slam: SLAMConfig = SLAMConfig()
    sensors: SensorConfig = SensorConfig()
    deep_learning: DeepLearningConfig = DeepLearningConfig()
    web_service: WebServiceConfig = WebServiceConfig()
    ros2: ROS2Config = ROS2Config()
    sfm: SfMConfig = SfMConfig()
    data_pipeline: DataPipelineConfig = DataPipelineConfig()
    mission_planner: MissionPlannerConfig = MissionPlannerConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """Load configuration from file (YAML or TOML)"""
        _, ext = os.path.splitext(file_path)
        
        with open(file_path, 'r') as f:
            if ext.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif ext.lower() == '.toml':
                config_dict = toml.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
            
        config = cls()
        
        if 'general' in config_dict:
            config.general = GeneralConfig(**config_dict['general'])
            
        if 'slam' in config_dict:
            config.slam = SLAMConfig(**config_dict['slam'])
            
        if 'sensors' in config_dict:
            config.sensors = SensorConfig(**config_dict['sensors'])
            
        if 'deep_learning' in config_dict:
            config.deep_learning = DeepLearningConfig(**config_dict['deep_learning'])
            
        if 'web_service' in config_dict:
            config.web_service = WebServiceConfig(**config_dict['web_service'])
            
        if 'ros2' in config_dict:
            config.ros2 = ROS2Config(**config_dict['ros2'])
            
        if 'sfm' in config_dict:
            config.sfm = SfMConfig(**config_dict['sfm'])
            
        if 'data_pipeline' in config_dict:
            config.data_pipeline = DataPipelineConfig(**config_dict['data_pipeline'])
            
        if 'mission_planner' in config_dict:
            config.mission_planner = MissionPlannerConfig(**config_dict['mission_planner'])
            
        if 'visualization' in config_dict:
            config.visualization = VisualizationConfig(**config_dict['visualization'])
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'general': self.general.__dict__,
            'slam': self.slam.__dict__,
            'sensors': self.sensors.__dict__,
            'deep_learning': self.deep_learning.__dict__,
            'web_service': self.web_service.__dict__,
            'ros2': self.ros2.__dict__,
            'sfm': self.sfm.__dict__,
            'data_pipeline': self.data_pipeline.__dict__,
            'mission_planner': self.mission_planner.__dict__,
            'visualization': self.visualization.__dict__,
        }
    
    def save(self, file_path: str) -> None:
        """Save configuration to file (YAML or TOML)"""
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        _, ext = os.path.splitext(file_path)
        with open(file_path, 'w') as f:
            if ext.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif ext.lower() == '.toml':
                toml.dump(config_dict, f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
            
# Global configuration singleton
_CONFIG: Optional[Config] = None

def get_config() -> Config:
    """Get global configuration singleton"""
    global _CONFIG
    
    if _CONFIG is None:
        # Try to load from default locations
        config_paths = [
            Path.cwd() / "config.toml",
            Path.cwd() / "config.yaml",
            Path.cwd() / "config" / "default_config.toml",
            Path.home() / ".config" / "openperception" / "config.toml",
            Path("/etc/openperception/config.toml"),
        ]
        
        for path in config_paths:
            if path.exists():
                _CONFIG = Config.from_file(str(path))
                break
        
        # If no config file found, use defaults
        if _CONFIG is None:
            _CONFIG = Config()
            
    return _CONFIG 