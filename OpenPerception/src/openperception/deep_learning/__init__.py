"""
Deep learning module for OpenPerception.

This module provides deep learning capabilities for computer vision tasks
such as object detection, segmentation, and classification.
"""

from .models import (
    ModelInterface,
    DetectionResult,
    SegmentationResult,
    get_model,
    list_available_models,
    detect_objects,
    segment_image
)

__all__ = [
    'ModelInterface',
    'DetectionResult',
    'SegmentationResult',
    'get_model',
    'list_available_models',
    'detect_objects',
    'segment_image'
]
