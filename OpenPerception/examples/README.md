# OpenPerception Examples

This directory contains example applications demonstrating the capabilities of the OpenPerception framework.

## Getting Started

Make sure you have OpenPerception installed before running these examples:

```bash
# Install OpenPerception with necessary dependencies
pip install "openperception[dev,deep_learning,ros2]"

# Or if you're working from the source code
cd /path/to/OpenPerception
pip install -e ".[dev,deep_learning,ros2]"
```

## Example Categories

The examples are organized into the following categories:

- **[SLAM](slam/)** - Visual SLAM examples for video processing and real-time mapping
- **[SfM](sfm/)** - Structure from Motion examples for 3D reconstruction
- **[Sensor Fusion](sensor_fusion/)** - Examples of fusing data from multiple sensors
- **[Deep Learning](deep_learning/)** - Object detection and segmentation examples
- **[Mission Planning](mission_planning/)** - AI-powered mission planning examples
- **[Path Planning](path_planning/)** - Path planning algorithm demonstrations

## Basic Usage

Most examples can be run directly from the command line:

```bash
# SLAM example
python examples/slam/slam_example.py --video path/to/video.mp4 --output results/slam

# Deep learning example
python examples/deep_learning/deep_learning_example.py --input path/to/image.jpg --output results/detections

# See help for any example
python examples/slam/slam_example.py --help
```

## Example Datasets

Some examples require input data. You can download sample datasets from the following sources:

- [SLAM dataset](https://github.com/llamasearchai/OpenPerception-datasets/tree/main/slam) - Sample videos for SLAM
- [SfM dataset](https://github.com/llamasearchai/OpenPerception-datasets/tree/main/sfm) - Image sequences for Structure from Motion
- [Object detection dataset](https://github.com/llamasearchai/OpenPerception-datasets/tree/main/detection) - Sample images and videos for deep learning

## Contributing New Examples

We welcome contributions of new examples! To add a new example:

1. Create a new Python file in the appropriate category directory
2. Make sure to include detailed comments and documentation
3. Update this README.md if necessary
4. Submit a pull request

## License

All example code is licensed under the same MIT License as the OpenPerception framework. 