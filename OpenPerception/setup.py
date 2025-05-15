from setuptools import setup, find_packages

setup(
    name="openperception",
    version="0.1.0",
    description="Advanced Computer Vision and Perception Framework for Aerial Robotics and Drones",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    url="https://github.com/llamasearchai/OpenPerception",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.1",
        "toml>=0.10.0",
        "pillow>=8.0.0",
        "scipy>=1.6.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
            "pylint",
            "pre-commit",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser",
            "sphinx-autodoc-typehints",
        ],
        "deep_learning": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "ultralytics",
        ],
        "ros2": [
            # "rclpy>=1.0.0",
            # Add other ROS2 dependencies here if pip-installable
            # Ensure these are documented for manual installation if not via pip
        ],
    },
    entry_points={
        "console_scripts": [
            "openperception=openperception.main:main_cli",
        ],
    },
) 