"""
OpenPerception: A comprehensive computer vision and perception framework for aerial robotics

This package provides algorithms and tools for visual SLAM, structure from motion,
sensor fusion, mission planning, and more for aerial robotics and drones.
"""

__version__ = "0.1.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

from .config import load_config

# Import main application class when implemented
# from .core import OpenPerception

# Import major submodules for easy access
from . import (
    benchmarking,
    calibration,
    data_pipeline,
    ros2_interface,
    sensor_fusion,
    sfm,
    slam,
    web_service,
)

__all__ = [
    "load_config",
    # "OpenPerception",  # Uncomment when implemented
    "benchmarking",
    "calibration",
    "data_pipeline",
    "ros2_interface",
    "sensor_fusion",
    "sfm",
    "slam",
    "web_service",
]

# Placeholder for future imports of key functionalities
# from .slam import VisualSLAM
# from .sfm import StructureFromMotion
# from .sensor_fusion import SensorFusion
# from .calibration import CameraCalibrator

import logging

# Setup a default logger for the library
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) 