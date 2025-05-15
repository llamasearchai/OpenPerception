#!/usr/bin/env python3
"""
Deep Learning Example - Demonstrating object detection and segmentation.

This example shows how to:
1. Load a pre-trained model for object detection
2. Process an image or video to detect objects
3. Visualize and save the results
4. Optionally perform segmentation
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import from OpenPerception deep learning module
from openperception.deep_learning import (
    detect_objects,
    segment_image,
    list_available_models
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenPerception Deep Learning Example")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--output", type=str, default="detection_results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="yolov5", help="Model to use for detection")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--segment", action="store_true", help="Also perform segmentation")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda, mps)")
    return parser.parse_args()

def visualize_detections(image, detections, output_path=None):
    """Visualize object detections on an image.
    
    Args:
        image: Input image
        detections: Detection results
        output_path: Path to save visualization
    """
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw bounding boxes
    for i in range(len(detections.bboxes)):
        # Get box coordinates
        x1, y1, x2, y2 = detections.bboxes[i].astype(int)
        score = detections.scores[i]
        label = detections.class_names[i]
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Object Detection: {len(detections.bboxes)} objects found")
    plt.axis("off")
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
    
    plt.show()

def visualize_segmentation(image, segmentation, output_path=None):
    """Visualize segmentation results.
    
    Args:
        image: Input image
        segmentation: Segmentation results
        output_path: Path to save visualization
    """
    # Create a colormap for segmentation mask
    num_classes = len(segmentation.class_ids)
    colormap = plt.cm.get_cmap("viridis", num_classes)
    
    # Create RGB segmentation mask
    mask_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Assign colors to each segment
    for i, class_id in enumerate(segmentation.class_ids):
        mask = segmentation.masks == class_id
        color = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
        mask_rgb[mask] = color
    
    # Create alpha blended visualization
    alpha = 0.5
    blended = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    
    # Add class labels
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Plot segmentation mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_rgb)
    plt.title("Segmentation Mask")
    plt.axis("off")
    
    # Plot blended result
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Blended Result")
    plt.axis("off")
    
    # Add colorbar for class labels
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Segmentation visualization saved to: {output_path}")
    
    plt.show()

def process_image(image_path, args):
    """Process a single image.
    
    Args:
        image_path: Path to input image
        args: Command line arguments
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Detect objects
    detections = detect_objects(
        image, 
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Print detection results
    print(f"Found {len(detections.bboxes)} objects:")
    for i in range(len(detections.bboxes)):
        print(f"  - {detections.class_names[i]}: {detections.scores[i]:.2f}")
    
    # Visualize detections
    output_path = os.path.join(args.output, "detections.jpg")
    visualize_detections(image, detections, output_path)
    
    # Perform segmentation if requested
    if args.segment:
        print("Performing segmentation...")
        segmentation = segment_image(
            image,
            model_name="simple_segmentation",
            num_classes=5
        )
        
        # Visualize segmentation
        seg_output_path = os.path.join(args.output, "segmentation.jpg")
        visualize_segmentation(image, segmentation, seg_output_path)

def process_video(video_path, args):
    """Process a video file.
    
    Args:
        video_path: Path to input video
        args: Command line arguments
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Frames: {frame_count}")
    print(f"  - FPS: {fps}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize video writer
    output_path = os.path.join(args.output, "detection_results.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for speed
        if frame_idx % 5 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
            
            # Detect objects
            detections = detect_objects(
                frame, 
                model_name=args.model,
                confidence_threshold=args.confidence,
                device=args.device
            )
            
            # Draw bounding boxes
            for i in range(len(detections.bboxes)):
                # Get box coordinates
                x1, y1, x2, y2 = detections.bboxes[i].astype(int)
                score = detections.scores[i]
                label = detections.class_names[i]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with score
                label_text = f"{label}: {score:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Save sample frames
        if frame_idx % 30 == 0:
            sample_path = os.path.join(args.output, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(sample_path, frame)
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video processing complete. Output saved to: {output_path}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # List available models if requested
    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine if input is image or video
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image(args.input, args)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(args.input, args)
    else:
        print(f"Unsupported file format: {args.input}")
        print("Supported formats: jpg, jpeg, png, bmp, mp4, avi, mov, mkv")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 