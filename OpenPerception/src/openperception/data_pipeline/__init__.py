"""
Data pipeline module for OpenPerception.

This module handles dataset management, data loading, and preprocessing.
"""

from .dataset_manager import DatasetManager, DatasetItem, ImageAnnotation

__all__ = [
    'DatasetManager',
    'DatasetItem',
    'ImageAnnotation'
]
