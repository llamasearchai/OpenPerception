"""
Simultaneous Localization and Mapping (SLAM) module for OpenPerception.

This module contains implementations of visual SLAM algorithms, enabling
the framework to build maps and track camera pose in real-time.
"""

from .visual_slam import VisualSLAM, KeyFrame, MapPoint, Frame

__all__ = [
    'VisualSLAM',
    'KeyFrame',
    'MapPoint',
    'Frame'
]
