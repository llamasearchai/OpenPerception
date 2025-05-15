# OpenPerception

<div align="center">

![OpenPerception Logo](docs/assets/logo.png)

**Advanced Computer Vision and Perception Framework for Aerial Robotics and Drones**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://openperception.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/llamasearchai/OpenPerception/actions)

</div>

## Overview

OpenPerception is a comprehensive, modular computer vision and perception framework designed for aerial robotics and drone applications. It provides a unified interface for implementing and integrating various perception tasks including SLAM, SfM, sensor fusion, mission planning, and deep learning.

### Key Features

- **Visual SLAM** - Real-time simultaneous localization and mapping
- **Structure from Motion** - 3D reconstruction from image sequences
- **Sensor Fusion** - Integrate data from multiple sensors (cameras, IMU, GPS, LiDAR)
- **Mission Planning** - AI-powered mission planning using LLMs (OpenAI integration)
- **Deep Learning** - Object detection, segmentation, and classification
- **Path Planning** - RRT, RRT*, A* implementations for autonomous navigation
- **ROS2 Integration** - Seamless integration with the Robot Operating System
- **Visualization** - Powerful 3D visualization of SLAM results, point clouds, and more
- **Web Services** - FastAPI-based interface for remote control and monitoring
- **Deployment Tools** - Utilities for deploying to edge devices like NVIDIA Jetson

## Installation

### From PyPI (Recommended)

```bash
pip install openperception
```

### From Source

```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenPerception.git
cd OpenPerception

# Install in development mode
pip install -e .

# Install additional dependencies
pip install -e ".[dev,deep_learning,ros2]"
```

## Quick Start

```python
from openperception import OpenPerception

# Initialize with default configuration
app = OpenPerception()

# Run SLAM on a video file
results = app.run_slam_from_video("path/to/video.mp4")

# Run Structure from Motion on a directory of images
reconstruction = app.run_sfm_from_images("path/to/images")

# Create a mission plan using AI
plan = app.create_mission_plan(
    mission_description="Survey a 500m x 500m agricultural field",
    environment_description="Rural area with crops, some tall trees on the perimeter",
    constraints="Maximum altitude: 120m, Battery life: 20 minutes"
)

# Print the plan
print(plan)
```

## Command-Line Interface

OpenPerception provides a user-friendly command-line interface:

```bash
# Run SLAM on a video file
openperception slam path/to/video.mp4 --output results/slam

# Run SfM on a directory of images
openperception sfm path/to/images --output results/sfm

# Start the web service
openperception web

# Create a mission plan
openperception mission --mission "Survey farmland" --environment "Rural area" --constraints "Max altitude: 120m"

# Deploy to a Jetson device
openperception deploy --ip 192.168.1.100 --username jetson
```

## Architecture

OpenPerception is designed with modularity and extensibility in mind. The framework is organized into the following key components:

```
OpenPerception/
├── config/               # Configuration management
├── src/openperception/   # Core package
│   ├── benchmarking/     # Performance benchmarking tools
│   ├── calibration/      # Camera and sensor calibration
│   ├── data_pipeline/    # Data management and processing
│   ├── deep_learning/    # Object detection and segmentation
│   ├── deployment/       # Deployment utilities (e.g., Jetson)
│   ├── mission_planner/  # AI-powered mission planning
│   ├── mvg/              # Multi-view geometry algorithms
│   ├── path_planning/    # Path planning algorithms
│   ├── ros2_interface/   # ROS2 integration
│   ├── sensor_fusion/    # Multi-sensor fusion algorithms
│   ├── sfm/              # Structure from Motion
│   ├── slam/             # Visual SLAM implementation
│   ├── utils/            # Utility functions
│   ├── visualization/    # Visualization tools
│   └── web_service/      # Web API for remote control
└── tests/                # Comprehensive test suite
```

## Documentation

Comprehensive documentation is available at [https://openperception.readthedocs.io](https://openperception.readthedocs.io), including:

- API Reference
- Tutorials
- Examples
- Developer Guide
- Deployment Guides

## Examples

Check out the `examples/` directory for detailed usage examples:

- `examples/slam_example.py` - Running SLAM on a video file
- `examples/sfm_example.py` - Creating 3D reconstructions with SfM
- `examples/sensor_fusion_example.py` - Fusing data from multiple sensors
- `examples/deep_learning_example.py` - Performing object detection and segmentation
- `examples/path_planning_example.py` - Planning paths for autonomous navigation
- `examples/web_service_example.py` - Using the web service API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use OpenPerception in your research, please cite:

```bibtex
@software{openperception2024,
  author = {Nik Jois},
  title = {OpenPerception: Computer Vision and Perception Framework for Aerial Robotics},
  url = {https://github.com/llamasearchai/OpenPerception},
  version = {0.1.0},
  year = {2024},
}
```

## Acknowledgments

- OpenCV community
- PyTorch team
- ROS2 community
- All the open-source contributors who made this project possible 