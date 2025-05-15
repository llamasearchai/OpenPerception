#!/usr/bin/env python3
"""
SLAM Example - Demonstrating how to run visual SLAM on a video file.

This example shows how to:
1. Initialize the OpenPerception framework
2. Run SLAM on a video file
3. Visualize and save the results
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import OpenPerception
from openperception import OpenPerception

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenPerception SLAM Example")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="slam_results", help="Directory to save results")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Running SLAM on video: {args.video}")
    print(f"Results will be saved to: {args.output}")
    
    # Initialize OpenPerception
    app = OpenPerception(config_path=args.config)
    
    # Set debug mode if specified
    if args.debug:
        app.config.general.debug = True
    
    # Run SLAM on video
    results = app.run_slam_from_video(args.video, args.output)
    
    # Check if successful
    if not results.get('success', False):
        print("SLAM processing failed")
        if 'error' in results:
            print(f"Error: {results['error']}")
        return
    
    # Print results
    print("\nSLAM Results:")
    print(f"- Processed {results.get('frame_count', 0)} frames")
    print(f"- Generated {len(results.get('map_points', []))} map points")
    print(f"- Trajectory length: {len(results.get('trajectory', []))} poses")
    print(f"- Results saved to: {results.get('output_dir', args.output)}")
    
    # Visualize results if requested
    if args.visualize and 'visualizer' in app._components:
        visualizer = app._components['visualizer']
        
        # Plot 3D map and trajectory
        visualizer.visualize_slam_results(
            trajectory=results['trajectory'],
            map_points=results['map_points'],
            save_path=os.path.join(args.output, "slam_visualization.png")
        )
        
        print("Visualization complete. Close plot window to exit.")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 