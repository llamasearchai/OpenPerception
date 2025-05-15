import os
import toml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class GeneralConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    debug: bool = False
    log_level: str = "INFO"

@dataclass
class SLAMConfig:
    enabled: bool = True
    use_gpu: bool = True
    num_features: int = 1000
    feature_type: str = "orb"
    max_keyframes: int = 20
    keyframe_threshold: int = 30

@dataclass
class SFMConfig:
    enabled: bool = True
    use_gpu: bool = True
    feature_type: str = "sift"
    matcher_type: str = "flann"
    min_matches: int = 15
    triangulation_threshold: float = 2.0

@dataclass
class SensorFusionConfig:
    enabled: bool = True
    max_buffer_size: int = 100
    fusion_method: str = "kalman"
    predict_frequency: int = 30

@dataclass
class WebServiceConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:1420"])
    workers: int = 4

@dataclass
class DataPipelineConfig:
    enabled: bool = True
    dataset_dir: str = "datasets"
    auto_annotate: bool = True
    annotation_confidence: float = 0.7

@dataclass
class MissionPlannerConfig:
    enabled: bool = True
    openai_api_key: Optional[str] = ""
    planning_algorithm: str = "rrt_star"
    replanning_interval: int = 5

@dataclass
class ROSTopicConfig:
    name: str = ""
    type: str = ""
    direction: str = "subscribe"

@dataclass
class ROSInterfaceConfig:
    enabled: bool = False
    node_name: str = "open_perception_node"
    use_composition: bool = True
    topics: List[ROSTopicConfig] = field(default_factory=list)

@dataclass
class VisualizationConfig:
    enabled: bool = True
    max_points: int = 100000
    point_size: float = 2.0
    show_trajectory: bool = True

@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    sfm: SFMConfig = field(default_factory=SFMConfig)
    sensor_fusion: SensorFusionConfig = field(default_factory=SensorFusionConfig)
    web_service: WebServiceConfig = field(default_factory=WebServiceConfig)
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    mission_planner: MissionPlannerConfig = field(default_factory=MissionPlannerConfig)
    ros_interface: ROSInterfaceConfig = field(default_factory=ROSInterfaceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_toml(cls, toml_file: str) -> 'Config':
        config_dict = toml.load(toml_file)
        
        config = cls()

        config.general = GeneralConfig(**config_dict.get('general', {}))
        config.slam = SLAMConfig(**config_dict.get('slam', {}))
        config.sfm = SFMConfig(**config_dict.get('sfm', {}))
        config.sensor_fusion = SensorFusionConfig(**config_dict.get('sensor_fusion', {}))
        config.web_service = WebServiceConfig(**config_dict.get('web_service', {}))
        config.data_pipeline = DataPipelineConfig(**config_dict.get('data_pipeline', {}))
        config.mission_planner = MissionPlannerConfig(**config_dict.get('mission_planner', {}))
        
        ros_config_dict = config_dict.get('ros_interface', {})
        ros_topics = [ROSTopicConfig(**topic) for topic in ros_config_dict.get('topics', [])]
        config.ros_interface = ROSInterfaceConfig(
            enabled=ros_config_dict.get('enabled', False),
            node_name=ros_config_dict.get('node_name', 'open_perception_node'),
            use_composition=ros_config_dict.get('use_composition', True),
            topics=ros_topics
        )
        config.visualization = VisualizationConfig(**config_dict.get('visualization', {}))
            
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            'general': self.general.__dict__,
            'slam': self.slam.__dict__,
            'sfm': self.sfm.__dict__,
            'sensor_fusion': self.sensor_fusion.__dict__,
            'web_service': self.web_service.__dict__,
            'data_pipeline': self.data_pipeline.__dict__,
            'mission_planner': self.mission_planner.__dict__,
            'ros_interface': {
                **{k: v for k, v in self.ros_interface.__dict__.items() if k != 'topics'},
                'topics': [t.__dict__ for t in self.ros_interface.topics]
            },
            'visualization': self.visualization.__dict__,
        }

    def save(self, toml_file: str) -> None:
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(toml_file), exist_ok=True)
        with open(toml_file, 'w') as f:
            toml.dump(config_dict, f)

_CONFIG: Optional[Config] = None

def get_config() -> Config:
    global _CONFIG
    
    if _CONFIG is None:
        # Default path relative to this file's location (config/config.py)
        default_config_path = Path(__file__).parent / "default_config.toml"
        
        config_paths_to_check = [
            Path.cwd() / "config.toml", # User-defined project root config
            default_config_path,        # Default config in the package
            Path.home() / ".config" / "openperception" / "config.toml",
            Path("/etc/openperception/config.toml"),
        ]
        
        loaded_path = None
        for path in config_paths_to_check:
            if path.exists():
                try:
                    _CONFIG = Config.from_toml(str(path))
                    loaded_path = path
                    break
                except Exception as e:
                    print(f"Warning: Could not load config from {path}: {e}")
        
        if _CONFIG is None:
            print("Warning: No configuration file found. Using default values.")
            _CONFIG = Config() # Initialize with default dataclass values
            if default_config_path.exists():
                 print(f"Attempting to load bundled default: {default_config_path}")
                 try:
                    _CONFIG = Config.from_toml(str(default_config_path))
                 except Exception as e:
                    print(f"Warning: Could not load bundled default_config.toml: {e}. Falling back to empty Config.")
                    _CONFIG = Config() # Fallback to empty dataclasses
            else:
                print(f"Warning: Bundled {default_config_path} not found. Using empty Config.")
                _CONFIG = Config()


    return _CONFIG

if __name__ == '__main__':
    # Example usage:
    config = get_config()
    print("Loaded Configuration:")
    print(f"  General Data Dir: {config.general.data_dir}")
    print(f"  SLAM enabled: {config.slam.enabled}")
    print(f"  SFM feature_type: {config.sfm.feature_type}")
    print(f"  WebService host: {config.web_service.host}")
    if config.ros_interface.topics:
      print(f"  First ROS topic name: {config.ros_interface.topics[0].name}")

    # To save a configuration (e.g., current one to a new file)
    # config.save("my_config.toml") 