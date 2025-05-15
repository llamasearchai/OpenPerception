"""
Deep learning models for OpenPerception.

This module provides deep learning model implementations for
tasks like object detection, segmentation, and classification.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Class for object detection results."""
    bboxes: np.ndarray  # Format: [x_min, y_min, x_max, y_max]
    labels: List[int]
    scores: np.ndarray
    class_names: List[str]
    
    def __post_init__(self):
        """Validate detection results."""
        if len(self.bboxes) != len(self.labels) or len(self.labels) != len(self.scores):
            raise ValueError("Number of bboxes, labels, and scores must match")
            
@dataclass
class SegmentationResult:
    """Class for segmentation results."""
    masks: np.ndarray  # Shape: [H, W] for semantic or [num_objects, H, W] for instance
    class_ids: np.ndarray  # Class ID for each pixel or instance
    class_names: List[str]
    scores: Optional[np.ndarray] = None  # For instance segmentation only

class ModelRegistry:
    """Registry for managing available models."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a model class."""
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def get_model(cls, name: str, **kwargs) -> Any:
        """Get a model by name."""
        if name not in cls._registry:
            raise ValueError(f"Model {name} not registered")
        return cls._registry[name](**kwargs)
        
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._registry.keys())

class ModelInterface:
    """Common interface for all models."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """Initialize model interface.
        
        Args:
            model_path: Path to model weights
            device: Device to run inference on ("cpu", "cuda", "mps")
        """
        self.model_path = model_path
        
        if TORCH_AVAILABLE:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            elif device == "mps" and not (hasattr(torch, "backends") and 
                                        hasattr(torch.backends, "mps") and 
                                        torch.backends.mps.is_available()):
                logger.warning("MPS not available, falling back to CPU")
                device = "cpu"
                
            self.device = torch.device(device)
        else:
            if device != "cpu":
                logger.warning("PyTorch not available, using CPU")
            self.device = "cpu"
        
        self.model = None
        self.class_names = []
        
    def load_model(self):
        """Load model from path."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocess image for inference."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def predict(self, image: np.ndarray) -> Any:
        """Run inference on an image."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def postprocess(self, outputs: Any) -> Any:
        """Postprocess model outputs."""
        raise NotImplementedError("Subclasses must implement this method")

@ModelRegistry.register("yolov5")
class YOLOv5Detector(ModelInterface):
    """YOLOv5 object detector implementation."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 classes: Optional[List[str]] = None,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 img_size: int = 640,
                 device: str = "cpu"):
        """Initialize YOLOv5 detector.
        
        Args:
            model_path: Path to YOLOv5 model weights
            classes: List of class names
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            img_size: Input image size
            device: Device to run inference on
        """
        super().__init__(model_path=model_path, device=device)
        
        if not TORCH_AVAILABLE:
            raise ImportError("YOLOv5 requires PyTorch, but it is not installed")
            
        self.classes = classes or []
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        if model_path:
            self.load_model()
            
    def load_model(self):
        """Load YOLOv5 model from path."""
        try:
            # YOLOv5 can be loaded directly from the repo or custom weights
            try:
                # Try loading from local weights first
                if self.model_path and os.path.exists(self.model_path):
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                              path=self.model_path, device=self.device)
                else:
                    # Fall back to pretrained model from hub
                    model_type = self.model_path or 'yolov5s'
                    logger.info(f"Loading pretrained {model_type} from torch hub")
                    self.model = torch.hub.load('ultralytics/yolov5', model_type, 
                                              pretrained=True, device=self.device)
            except Exception as e:
                logger.error(f"Error loading YOLOv5 from torch hub: {e}")
                # As a fallback, try to import and load directly
                import sys
                sys.path.append(str(Path(self.model_path).parent))
                from models.experimental import attempt_load
                self.model = attempt_load(self.model_path, device=self.device)
                                    
            # Set model parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            elif self.classes:
                self.class_names = self.classes
            else:
                logger.warning("No class names provided or found in model")
                self.class_names = [f"class_{i}" for i in range(1000)]  # Default placeholder
                
            logger.info(f"Loaded YOLOv5 model with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model: {e}")
            raise
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLOv5 inference.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image
        """
        # YOLOv5 hub model handles preprocessing internally
        return image  # Just return original image
        
    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run YOLOv5 inference on an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")
            
        # Convert BGR to RGB (YOLOv5 expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(image_rgb, size=self.img_size)
        
        # Extract predictions
        pred = results.pred[0].cpu().numpy()
        
        # Convert to DetectionResult format
        if len(pred) > 0:
            bboxes = pred[:, :4]  # x1, y1, x2, y2
            scores = pred[:, 4]
            labels = pred[:, 5].astype(int)
        else:
            bboxes = np.zeros((0, 4))
            scores = np.zeros(0)
            labels = []
            
        return DetectionResult(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            class_names=[self.class_names[i] for i in labels]
        )
        
    def postprocess(self, outputs: DetectionResult) -> Dict[str, Any]:
        """Format YOLOv5 outputs for visualization.
        
        Args:
            outputs: Detection results
            
        Returns:
            Dictionary with formatted detection results
        """
        return {
            "num_detections": len(outputs.bboxes),
            "bboxes": outputs.bboxes,
            "labels": outputs.labels,
            "scores": outputs.scores,
            "class_names": outputs.class_names
        }

@ModelRegistry.register("simple_segmentation")
class SimpleSegmentationModel(ModelInterface):
    """Simple segmentation model using OpenCV for basic color-based segmentation when deep learning is not available."""
    
    def __init__(self, model_path: Optional[str] = None, 
                num_classes: int = 2, 
                device: str = "cpu"):
        """Initialize simple segmentation model.
        
        Args:
            model_path: Not used for this model
            num_classes: Number of classes to segment
            device: Not used for this model
        """
        super().__init__(model_path=None, device="cpu")
        self.num_classes = num_classes
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        
    def load_model(self):
        """Nothing to load for this simple model."""
        pass
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for segmentation.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image
        """
        # Convert to HSV color space for better color segmentation
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run simple segmentation on an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Segmentation results
        """
        # Preprocess
        hsv = self.preprocess(image)
        
        # Simple K-means segmentation as fallback when no deep learning model is available
        pixel_values = hsv.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Define K-means criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = min(self.num_classes, 10)  # Cap at 10 to avoid excessive computation
        
        # Apply K-means
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to image
        labels = labels.flatten()
        segmented_image = labels.reshape(image.shape[:2])
        
        return SegmentationResult(
            masks=segmented_image,
            class_ids=np.unique(segmented_image),
            class_names=[f"segment_{i}" for i in range(k)]
        )
        
    def postprocess(self, outputs: SegmentationResult) -> Dict[str, Any]:
        """Format segmentation outputs.
        
        Args:
            outputs: Segmentation results
            
        Returns:
            Dictionary with segmentation masks and metadata
        """
        return {
            "segmentation_mask": outputs.masks,
            "num_classes": len(outputs.class_ids),
            "class_ids": outputs.class_ids,
            "class_names": outputs.class_names
        }

def get_model(model_name: str, **kwargs) -> ModelInterface:
    """Helper function to get a model by name.
    
    Args:
        model_name: Name of the model to get
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Model instance
    """
    return ModelRegistry.get_model(model_name, **kwargs)

def list_available_models() -> List[str]:
    """List all available models.
    
    Returns:
        List of model names
    """
    return ModelRegistry.list_models()

def detect_objects(image: np.ndarray, model_name: str = "yolov5", **kwargs) -> DetectionResult:
    """Detect objects in an image.
    
    Args:
        image: Input image
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Detection results
    """
    model = get_model(model_name, **kwargs)
    
    if not hasattr(model, 'model') or model.model is None:
        model.load_model()
        
    return model.predict(image)

def segment_image(image: np.ndarray, model_name: str = "simple_segmentation", **kwargs) -> SegmentationResult:
    """Segment an image.
    
    Args:
        image: Input image
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Segmentation results
    """
    model = get_model(model_name, **kwargs)
    
    if hasattr(model, 'load_model'):
        model.load_model()
        
    return model.predict(image) 