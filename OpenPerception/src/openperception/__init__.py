"""
OpenPerception - Computer vision and perception framework for aerial robotics and drones
"""

__version__ = "0.1.0"

from . import calibration
from . import slam
from . import sfm
from . import sensor_fusion
from . import mission_planner
from . import utils
from . import visualization
from . import web_service
from . import deep_learning
from . import path_planning
from . import deployment
from . import ros2_interface
from . import data_pipeline

# Expose key classes for easier imports
from .calibration.camera_calibration import CameraCalibrator
from .sensor_fusion.fusion import SensorFusion
from .mission_planner.planner import MissionPlanner
from .deep_learning.models import detect_objects, segment_image
from .path_planning.path_planner import plan_path 