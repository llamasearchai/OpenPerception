import time
import logging
import argparse
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import torch
import psutil
import platform
from pathlib import Path

# Updated imports for OpenPerception structure
from openperception.slam.visual_slam import VisualSLAM
from openperception.sfm.structure_from_motion import StructureFromMotion
from openperception.sensor_fusion.fusion import SensorFusion
from openperception.config import load_config # For potential future config use
from openperception import OpenPerception # For accessing main app or its components

logger = logging.getLogger(__name__)

class BenchmarkResult:
    """Benchmark result container"""
    
    def __init__(self, name: str):
        """Initialize benchmark result
        
        Args:
            name: Benchmark name
        """
        self.name = name
        self.timings = []
        self.memory_usage_rss_mb = [] # Store RSS memory in MB
        self.cpu_percent = [] # Store CPU percentage
        self.start_time = None
        self.system_info = self._get_system_info()
        self.iterations_data = [] # Store per-iteration details
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3)
        }
        try:
            info["opencv_version"] = cv2.__version__
        except AttributeError:
            info["opencv_version"] = "Not found"
        try:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except (ImportError, AttributeError):
            info["torch_version"] = "Not found or CUDA error"
            info["cuda_available"] = False
        return info
        
    def start_iteration(self):
        """Start timing and resource monitoring for an iteration."""
        self.current_iteration_start_time = time.perf_counter() # More precise for timing
        self.current_iteration_start_memory_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        self.current_iteration_start_cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
        
    def end_iteration(self):
        """Stop timing and resource monitoring for an iteration."""
        if not hasattr(self, 'current_iteration_start_time') or self.current_iteration_start_time is None:
            return
            
        elapsed_time = time.perf_counter() - self.current_iteration_start_time
        end_memory_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        # CPU percent over the interval of the iteration
        # For a more accurate CPU usage of the specific code block, would need more complex tracing
        # This captures overall process CPU usage during the iteration
        end_cpu_percent = psutil.cpu_percent(interval=None) 

        self.timings.append(elapsed_time)
        self.memory_usage_rss_mb.append(end_memory_rss_mb) # Record end memory of iteration
        self.cpu_percent.append(end_cpu_percent) # Record end CPU percent

        self.iterations_data.append({
            'time_s': elapsed_time,
            'start_mem_mb': self.current_iteration_start_memory_rss_mb,
            'end_mem_mb': end_memory_rss_mb,
            'mem_delta_mb': end_memory_rss_mb - self.current_iteration_start_memory_rss_mb,
            'start_cpu_percent': self.current_iteration_start_cpu_percent,
            'end_cpu_percent': end_cpu_percent
        })
        self.current_iteration_start_time = None # Reset for next iteration
        
    def summarize(self) -> Dict[str, Any]:
        """Create summary of benchmark results."""
        if not self.timings:
            return {"name": self.name, "system_info": self.system_info, "error": "No timing data collected"}
            
        summary = {
            "name": self.name,
            "system_info": self.system_info,
            "iterations": len(self.timings),
            "total_duration_s": np.sum(self.timings),
            "timing_per_iteration_s": {
                "mean": np.mean(self.timings),
                "median": np.median(self.timings),
                "min": np.min(self.timings),
                "max": np.max(self.timings),
                "std_dev": np.std(self.timings),
                "values": self.timings # Store all raw timings for detailed analysis
            },
            "memory_rss_end_iteration_mb": {
                "mean": np.mean(self.memory_usage_rss_mb),
                "median": np.median(self.memory_usage_rss_mb),
                "min": np.min(self.memory_usage_rss_mb),
                "max": np.max(self.memory_usage_rss_mb),
                "std_dev": np.std(self.memory_usage_rss_mb),
                "values": self.memory_usage_rss_mb
            },
            "cpu_percent_end_iteration": {
                "mean": np.mean(self.cpu_percent),
                "median": np.median(self.cpu_percent),
                "min": np.min(self.cpu_percent),
                "max": np.max(self.cpu_percent),
                "std_dev": np.std(self.cpu_percent),
                "values": self.cpu_percent
            },
            "detailed_iterations_data": self.iterations_data
        }
        return summary

class PerformanceBenchmark:
    """Performance benchmarking for OpenPerception modules."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, output_dir: str = "benchmarks_output"):
        """Initialize benchmarking tool.
        Args:
            config: Overall application configuration (passed from OpenPerceptionApp ideally).
            output_dir: Directory to save benchmark results.
        """
        self.config = config if config else load_config().to_dict() # Fallback to default config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.open_perception_app = OpenPerception(config_data=self.config) # Use the main app

    def _get_default_camera_params_for_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Default K: fx=fy=width, cx=width/2, cy=height/2
        camera_matrix = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        return camera_matrix, dist_coeffs

    def benchmark_slam_on_video(self, video_path: str, iterations: int = 1) -> BenchmarkResult:
        """Benchmark Visual SLAM performance on a video file."""
        if not self.open_perception_app.slam_system:
            logger.warning("SLAM system not initialized in OpenPerceptionApp. Skipping SLAM benchmark.")
            result = BenchmarkResult(f"visual_slam_video_{Path(video_path).stem}")
            result.error = "SLAM system not available or not configured."
            return result

        result = BenchmarkResult(f"visual_slam_video_{Path(video_path).stem}")
        logger.info(f"Starting SLAM benchmark for video: {video_path}, Iterations: {iterations}")

        # Assuming OpenPerceptionApp's SLAM can be re-initialized or reset for multiple runs
        # Or, we rely on run_slam_from_video to handle its own state for benchmarking one full pass.
        # For true iteration benchmark, the SLAM state might need careful handling.

        # This benchmarks one full processing of the video per iteration.
        for i in range(iterations):
            logger.info(f"SLAM Benchmark Iteration {i+1}/{iterations}")
            result.start_iteration()
            try:
                # This method should encapsulate the entire SLAM run on the video
                # and ideally return some metrics or ensure completion.
                self.open_perception_app.run_slam_from_video(video_path, benchmark_mode=True)
            except Exception as e:
                logger.error(f"Error during SLAM benchmark iteration {i+1}: {e}", exc_info=True)
                result.iterations_data.append({'error': str(e)})
            finally:
                result.end_iteration()
        return result

    def benchmark_sfm_on_images(self, image_dir: str, iterations: int = 1) -> BenchmarkResult:
        """Benchmark Structure from Motion performance on a directory of images."""
        if not self.open_perception_app.sfm_system:
            logger.warning("SfM system not initialized in OpenPerceptionApp. Skipping SfM benchmark.")
            result = BenchmarkResult(f"sfm_images_{Path(image_dir).name}")
            result.error = "SfM system not available or not configured."
            return result

        result = BenchmarkResult(f"sfm_images_{Path(image_dir).name}")
        logger.info(f"Starting SfM benchmark for image directory: {image_dir}, Iterations: {iterations}")

        for i in range(iterations):
            logger.info(f"SfM Benchmark Iteration {i+1}/{iterations}")
            result.start_iteration()
            try:
                # This method should encapsulate the entire SfM run on the image set.
                self.open_perception_app.run_sfm_from_images_dir(image_dir, benchmark_mode=True)
            except Exception as e:
                logger.error(f"Error during SfM benchmark iteration {i+1}: {e}", exc_info=True)
                result.iterations_data.append({'error': str(e)})
            finally:
                result.end_iteration()
        return result
    
    def benchmark_object_detection(self, image_path_or_dir: str, model_name: str = "yolov5", iterations: int = 1) -> BenchmarkResult:
        """Benchmark object detection performance."""
        if not self.open_perception_app.deep_learning_models.get(model_name):
            logger.warning(f"Object detection model '{model_name}' not available. Skipping benchmark.")
            result = BenchmarkResult(f"object_detection_{model_name}_{Path(image_path_or_dir).name}")
            result.error = f"Model {model_name} not available or configured."
            return result

        result = BenchmarkResult(f"object_detection_{model_name}_{Path(image_path_or_dir).name}")
        logger.info(f"Starting Object Detection benchmark for {image_path_or_dir} with model {model_name}, Iterations: {iterations}")

        image_files = []
        if Path(image_path_or_dir).is_dir():
            image_files = sorted([p for p in Path(image_path_or_dir).glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        elif Path(image_path_or_dir).is_file():
            image_files = [Path(image_path_or_dir)]
        
        if not image_files:
            result.error = "No valid image files found."
            return result

        # Load all images into memory once to exclude I/O from per-iteration timing if desired
        # Or read per iteration to include I/O (current approach)

        for i in range(iterations):
            logger.info(f"Object Detection Benchmark Iteration {i+1}/{iterations}")
            result.start_iteration()
            try:
                num_detections_total = 0
                for img_file in image_files:
                    detections = self.open_perception_app.detect_objects_from_file(str(img_file), model_name)
                    if detections:
                        num_detections_total += len(detections.bboxes)
                result.iterations_data[-1]['num_images_processed'] = len(image_files)
                result.iterations_data[-1]['total_detections'] = num_detections_total
            except Exception as e:
                logger.error(f"Error during Object Detection benchmark iteration {i+1}: {e}", exc_info=True)
                if result.iterations_data : result.iterations_data[-1]['error'] = str(e)
                else: result.iterations_data.append({'error': str(e)}) # if start_iteration wasn't called before error
            finally:
                result.end_iteration() # This will append a new entry if start_iteration was called
        return result

    def save_benchmark_summary(self, result: BenchmarkResult, filename_prefix: Optional[str] = None):
        """Save benchmark summary to a JSON file."""
        summary = result.summarize()
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix or result.name}_{ts}.json"
        output_path = self.output_dir / filename
        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=4, cls=NumpyEncoder) # Handle numpy types for JSON
            logger.info(f"Benchmark summary saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark summary to {output_path}: {e}")

    def visualize_benchmark_summary(self, result: BenchmarkResult, filename_prefix: Optional[str] = None, show: bool = False):
        """Visualize benchmark summary (timings and memory)."""
        summary = result.summarize()
        if 'error' in summary and summary['error'] == "No timing data collected":
            logger.warning(f"Skipping visualization for {result.name} due to no timing data.")
            return

        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(f"Benchmark Summary: {result.name}\nSystem: {summary['system_info']['platform']}", fontsize=14)

        iterations = range(1, len(result.timings) + 1)

        # Timings
        axs[0].plot(iterations, result.timings, marker='o', linestyle='-', label='Iteration Time')
        axs[0].axhline(summary['timing_per_iteration_s']['mean'], color='r', linestyle='--', label=f"Mean: {summary['timing_per_iteration_s']['mean']:.3f}s")
        axs[0].set_ylabel("Time (s)")
        axs[0].set_title("Processing Time per Iteration")
        axs[0].legend()
        axs[0].grid(True)

        # Memory Usage (End of Iteration RSS)
        axs[1].plot(iterations, result.memory_usage_rss_mb, marker='s', linestyle='-', color='green', label='Memory RSS (End)')
        axs[1].axhline(summary['memory_rss_end_iteration_mb']['mean'], color='purple', linestyle='--', label=f"Mean RSS: {summary['memory_rss_end_iteration_mb']['mean']:.2f} MB")
        axs[1].set_ylabel("Memory RSS (MB)")
        axs[1].set_title("Memory Usage (RSS) at End of Iteration")
        axs[1].legend()
        axs[1].grid(True)

        # CPU Usage (End of Iteration)
        axs[2].plot(iterations, result.cpu_percent, marker='^', linestyle='-', color='orange', label='CPU % (End)')
        axs[2].axhline(summary['cpu_percent_end_iteration']['mean'], color='blue', linestyle='--', label=f"Mean CPU: {summary['cpu_percent_end_iteration']['mean']:.1f}%")
        axs[2].set_xlabel("Iteration Number")
        axs[2].set_ylabel("CPU Utilization (%)")
        axs[2].set_title("CPU Utilization at End of Iteration")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        viz_filename = f"{filename_prefix or result.name}_visualization_{ts}.png"
        output_viz_path = self.output_dir / viz_filename
        plt.savefig(output_viz_path)
        logger.info(f"Benchmark visualization saved to: {output_viz_path}")
        
        if show:
            plt.show()
        plt.close(fig)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types. """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist() # Convert arrays to lists
        return json.JSONEncoder.default(self, obj)

def main_benchmark_cli():
    parser = argparse.ArgumentParser(description="OpenPerception Performance Benchmarking Tool")
    parser.add_argument("--config", type=str, default=None, help="Path to OpenPerception config file (optional)")
    parser.add_argument("--output-dir", type=str, default="benchmarks_output", help="Directory for benchmark results")
    parser.add_argument("--module", type=str, required=True, choices=["slam", "sfm", "detection"], help="Module to benchmark")
    parser.add_argument("--input", type=str, required=True, help="Input video file (for SLAM), image directory (for SfM), or image/dir (for detection)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--model-name", type=str, default="yolov5", help="Model name for detection benchmark (e.g., yolov5)")
    parser.add_argument("--visualize", action="store_true", help="Show visualization plots after benchmarking")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load main app config if provided, else benchmark will use its default
    app_config_data = None
    if args.config:
        app_config_data = load_config(args.config).to_dict()

    benchmarker = PerformanceBenchmark(config=app_config_data, output_dir=args.output_dir)
    result = None

    if args.module == "slam":
        if not Path(args.input).is_file():
            logger.error(f"SLAM benchmark requires a video file. Provided input: {args.input}")
            return
        result = benchmarker.benchmark_slam_on_video(video_path=args.input, iterations=args.iterations)
    elif args.module == "sfm":
        if not Path(args.input).is_dir():
            logger.error(f"SfM benchmark requires an image directory. Provided input: {args.input}")
            return
        result = benchmarker.benchmark_sfm_on_images(image_dir=args.input, iterations=args.iterations)
    elif args.module == "detection":
        result = benchmarker.benchmark_object_detection(image_path_or_dir=args.input, 
                                                        model_name=args.model_name, 
                                                        iterations=args.iterations)
    else:
        logger.error(f"Unknown module for benchmarking: {args.module}")
        return

    if result:
        benchmarker.save_benchmark_summary(result)
        benchmarker.visualize_benchmark_summary(result, show=args.visualize)
    else:
        logger.error("Benchmarking did not produce a result.")

if __name__ == "__main__":
    main_benchmark_cli() 