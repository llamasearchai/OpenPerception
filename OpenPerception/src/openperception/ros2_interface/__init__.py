"""
ROS2 Interface for OpenPerception.

This module provides a bridge for communicating with ROS2, allowing the framework
to subscribe to sensor topics and publish results.
"""

from .ros_bridge import ROS2Bridge

__all__ = [
    'ROS2Bridge'
]
