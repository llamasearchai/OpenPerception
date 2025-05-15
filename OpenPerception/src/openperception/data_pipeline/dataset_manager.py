import os
import json
import cv2
import numpy as np
import shutil
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import uuid
from dataclasses import dataclass, asdict
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ImageAnnotation:
    """Image annotation data"""
    bounding_boxes: List[Dict[str, Any]]  # [{label, x1, y1, x2, y2, confidence}, ...]
    keypoints: List[Dict[str, Any]]  # [{x, y, label}, ...]
    segmentation_mask: Optional[str]  # Path to segmentation mask
    metadata: Dict[str, Any]  # Additional metadata
    
@dataclass
class DatasetItem:
    """Dataset item"""
    id: str
    image_path: str
    timestamp: float
    annotations: Optional[ImageAnnotation] = None
    metadata: Dict[str, Any] = None
    
class DatasetManager:
    """Dataset manager for collecting, annotating, and curating datasets"""
    
    def __init__(self, dataset_dir: str):
        """Initialize dataset manager
        
        Args:
            dataset_dir: Directory to store datasets
        """
        self.dataset_dir = dataset_dir
        self.datasets = {}
        self.current_dataset = None
        
        # Create directory if not exists
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Load existing datasets
        self._load_datasets()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def _load_datasets(self):
        """Load existing datasets"""
        for dataset_name in os.listdir(self.dataset_dir):
            dataset_path = os.path.join(self.dataset_dir, dataset_name)
            
            if os.path.isdir(dataset_path):
                metadata_path = os.path.join(dataset_path, "metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            
                        self.datasets[dataset_name] = metadata
                        logger.info(f"Loaded dataset: {dataset_name}")
                    except Exception as e:
                        logger.error(f"Error loading dataset {dataset_name}: {e}")
                        
    def create_dataset(self, name: str, description: str = "") -> str:
        """Create a new dataset
        
        Args:
            name: Dataset name
            description: Dataset description
            
        Returns:
            Dataset ID
        """
        with self.lock:
            # Create dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Create dataset directory
            dataset_dir = os.path.join(self.dataset_dir, dataset_id)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create subdirectories
            os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "masks"), exist_ok=True)
            
            # Create dataset metadata
            metadata = {
                "id": dataset_id,
                "name": name,
                "description": description,
                "created_at": time.time(),
                "updated_at": time.time(),
                "item_count": 0
            }
            
            # Save metadata
            with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Add to datasets
            self.datasets[dataset_id] = metadata
            
            # Set as current dataset
            self.current_dataset = dataset_id
            
            logger.info(f"Created dataset: {name} ({dataset_id})")
            
            return dataset_id
            
    def set_current_dataset(self, dataset_id: str):
        """Set current dataset
        
        Args:
            dataset_id: Dataset ID
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        self.current_dataset = dataset_id
        
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset information
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset metadata
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        return self.datasets[dataset_id].copy()
        
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets
        
        Returns:
            List of dataset metadata
        """
        return [metadata.copy() for metadata in self.datasets.values()]
        
    def add_image(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """Add an image to the current dataset
        
        Args:
            image: Image to add
            metadata: Additional metadata
            
        Returns:
            Item ID
        """
        if self.current_dataset is None:
            raise ValueError("No current dataset selected")
            
        with self.lock:
            # Create item ID
            item_id = str(uuid.uuid4())
            
            # Get dataset directory
            dataset_dir = os.path.join(self.dataset_dir, self.current_dataset)
            
            # Save image
            image_path = os.path.join(dataset_dir, "images", f"{item_id}.jpg")
            cv2.imwrite(image_path, image)
            
            # Create item metadata
            item_metadata = {
                "id": item_id,
                "image_path": image_path,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # Save item metadata
            with open(os.path.join(dataset_dir, "annotations", f"{item_id}.json"), "w") as f:
                json.dump(item_metadata, f, indent=2)
                
            # Update dataset metadata
            self.datasets[self.current_dataset]["item_count"] += 1
            self.datasets[self.current_dataset]["updated_at"] = time.time()
            
            # Save dataset metadata
            with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                json.dump(self.datasets[self.current_dataset], f, indent=2)
                
            logger.info(f"Added image {item_id} to dataset {self.current_dataset}")
            
            return item_id
            
    def add_annotation(self, item_id: str, annotation: ImageAnnotation):
        """Add annotation to a dataset item
        
        Args:
            item_id: Item ID
            annotation: Annotation data
        """
        if self.current_dataset is None:
            raise ValueError("No current dataset selected")
            
        with self.lock:
            # Get dataset directory
            dataset_dir = os.path.join(self.dataset_dir, self.current_dataset)
            
            # Get item metadata file
            metadata_path = os.path.join(dataset_dir, "annotations", f"{item_id}.json")
            
            if not os.path.exists(metadata_path):
                raise ValueError(f"Item {item_id} not found")
                
            # Load item metadata
            with open(metadata_path, "r") as f:
                item_metadata = json.load(f)
                
            # Add annotation
            item_metadata["annotation"] = asdict(annotation)
            
            # Save item metadata
            with open(metadata_path, "w") as f:
                json.dump(item_metadata, f, indent=2)
                
            logger.info(f"Added annotation to item {item_id}")
            
    def get_item(self, item_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get a dataset item
        
        Args:
            item_id: Item ID
            
        Returns:
            Tuple of (image, metadata)
        """
        if self.current_dataset is None:
            raise ValueError("No current dataset selected")
            
        # Get dataset directory
        dataset_dir = os.path.join(self.dataset_dir, self.current_dataset)
        
        # Get item metadata file
        metadata_path = os.path.join(dataset_dir, "annotations", f"{item_id}.json")
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"Item {item_id} not found")
            
        # Load item metadata
        with open(metadata_path, "r") as f:
            item_metadata = json.load(f)
            
        # Load image
        image_path = os.path.join(dataset_dir, "images", f"{item_id}.jpg") # Assumes jpg, could be param
        image = cv2.imread(image_path)
        
        return image, item_metadata
            
    def export_dataset(self, dataset_id: str, output_dir: str, format: str = "coco"):
        """Export dataset in specified format
        
        Args:
            dataset_id: Dataset ID
            output_dir: Output directory
            format: Export format (coco, yolo, etc.)
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataset directory
        dataset_dir = os.path.join(self.dataset_dir, dataset_id)
        
        if format.lower() == "coco":
            self._export_coco(dataset_id, dataset_dir, output_dir)
        elif format.lower() == "yolo":
            self._export_yolo(dataset_id, dataset_dir, output_dir)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def _export_coco(self, dataset_id: str, dataset_dir: str, output_dir: str):
        """Export dataset in COCO format"""
        # Create COCO structure
        coco_output = {
            "info": {
                "description": self.datasets[dataset_id]["description"],
                "version": "1.0", # Add version or other info
                "year": int(time.strftime("%Y")),
                "contributor": "OpenPerception DatasetManager", # Add contributor
                "date_created": time.strftime("%Y/%m/%d")
            },
            "licenses": [], # Add license info if available
            "images": [],
            "annotations": [],
            "categories": [] 
        }
        
        # Create images directory in output
        coco_images_dir = os.path.join(output_dir, "images")
        os.makedirs(coco_images_dir, exist_ok=True)
        
        # Process all items
        annotation_files_dir = os.path.join(dataset_dir, "annotations")
        coco_annotation_id = 1
        categories_map = {} # Map label name to category_id
        category_counter = 1

        for item_filename in os.listdir(annotation_files_dir):
            if not item_filename.endswith(".json"):
                continue
                
            with open(os.path.join(annotation_files_dir, item_filename), "r") as f:
                item_metadata = json.load(f)
                
            item_id_str = item_metadata["id"] # Use string ID as per DatasetItem
            
            # Copy image and add to COCO images list
            src_image_path = os.path.join(dataset_dir, "images", f"{item_id_str}.jpg")
            dst_image_path = os.path.join(coco_images_dir, f"{item_id_str}.jpg")
            
            if not os.path.exists(src_image_path):
                logger.warning(f"Image {src_image_path} not found for item {item_id_str}. Skipping.")
                continue

            shutil.copy(src_image_path, dst_image_path)
            
            img = cv2.imread(src_image_path)
            height, width = img.shape[:2]
            
            coco_output["images"].append({
                "id": item_id_str, # Use string ID
                "file_name": f"{item_id_str}.jpg",
                "width": width,
                "height": height,
                "license": None, # Add if available
                "flickr_url": None, # Add if available
                "coco_url": None, # Add if available
                "date_captured": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(item_metadata["timestamp"]))
            })
            
            # Add annotations if available
            if "annotation" in item_metadata and item_metadata["annotation"]:
                annotation_data = item_metadata["annotation"]
                
                if "bounding_boxes" in annotation_data:
                    for bbox in annotation_data["bounding_boxes"]:
                        label = bbox["label"]
                        if label not in categories_map:
                            categories_map[label] = category_counter
                            coco_output["categories"].append({
                                "id": category_counter,
                                "name": label,
                                "supercategory": "object" # Or derive from label
                            })
                            category_counter += 1
                        
                        category_id = categories_map[label]
                        
                        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        coco_output["annotations"].append({
                            "id": coco_annotation_id,
                            "image_id": item_id_str, # Use string ID
                            "category_id": category_id,
                            "bbox": [x1, y1, bbox_width, bbox_height],
                            "area": float(bbox_width * bbox_height),
                            "segmentation": [], # Add segmentation if available
                            "iscrowd": 0 # Default to not crowd
                        })
                        coco_annotation_id += 1
                        
        # Save COCO JSON
        with open(os.path.join(output_dir, "annotations.json"), "w") as f:
            json.dump(coco_output, f, indent=2)
            
        logger.info(f"Exported dataset {dataset_id} in COCO format to {output_dir}")

    def _export_yolo(self, dataset_id: str, dataset_dir: str, output_dir: str):
        """Export dataset in YOLO format."""
        yolo_images_dir = os.path.join(output_dir, "images")
        yolo_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(yolo_images_dir, exist_ok=True)
        os.makedirs(yolo_labels_dir, exist_ok=True)

        annotation_files_dir = os.path.join(dataset_dir, "annotations")
        class_names = [] # List to store unique class names for classes.txt

        for item_filename in os.listdir(annotation_files_dir):
            if not item_filename.endswith(".json"):
                continue

            with open(os.path.join(annotation_files_dir, item_filename), "r") as f:
                item_metadata = json.load(f)

            item_id_str = item_metadata["id"]
            src_image_path = os.path.join(dataset_dir, "images", f"{item_id_str}.jpg")
            
            if not os.path.exists(src_image_path):
                logger.warning(f"Image {src_image_path} not found for item {item_id_str}. Skipping.")
                continue

            img = cv2.imread(src_image_path)
            img_height, img_width = img.shape[:2]

            # Copy image
            shutil.copy(src_image_path, os.path.join(yolo_images_dir, f"{item_id_str}.jpg"))
            
            yolo_annotations = []
            if "annotation" in item_metadata and item_metadata["annotation"]:
                annotation_data = item_metadata["annotation"]
                if "bounding_boxes" in annotation_data:
                    for bbox in annotation_data["bounding_boxes"]:
                        label = bbox["label"]
                        if label not in class_names:
                            class_names.append(label)
                        
                        class_id = class_names.index(label)
                        
                        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        # Convert to YOLO format: (center_x, center_y, width, height) normalized
                        bbox_w = (x2 - x1) / img_width
                        bbox_h = (y2 - y1) / img_height
                        center_x = ((x1 + x2) / 2) / img_width
                        center_y = ((y1 + y2) / 2) / img_height
                        
                        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}")

            if yolo_annotations:
                with open(os.path.join(yolo_labels_dir, f"{item_id_str}.txt"), "w") as f_label:
                    f_label.write("\n".join(yolo_annotations))
        
        # Create classes.txt or dataset.yaml for YOLO
        with open(os.path.join(output_dir, "classes.txt"), "w") as f_classes:
            f_classes.write("\n".join(class_names))
        
        # Optional: Create dataset.yaml for YOLOv5+
        dataset_yaml_content = {
            "path": Path(output_dir).resolve().as_posix(), # absolute path to dataset.yaml parent dir
            "train": "images", # path to train images (relative to 'path')
            "val": "images",  # path to val images (relative to 'path') - adjust if you have splits
            "names": {i: name for i, name in enumerate(class_names)}
        }
        with open(os.path.join(output_dir, "dataset.yaml"), "w") as f_yaml:
            import yaml # Requires PyYAML
            yaml.dump(dataset_yaml_content, f_yaml, sort_keys=False)

        logger.info(f"Exported dataset {dataset_id} in YOLO format to {output_dir}") 