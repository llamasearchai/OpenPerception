import os
from setuptools import setup, find_packages

# Read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements file
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return [line.strip() for line in req if line.strip() and not line.startswith('#')]

requirements = read_requirements() if os.path.exists('requirements.txt') else [
    'numpy>=1.20.0',
    'opencv-python>=4.5.0',
    'scipy>=1.7.0',
    'matplotlib>=3.4.0',
    'torch>=1.9.0',
    'fastapi>=0.68.0',
    'uvicorn>=0.15.0',
]

setup(
    name="openperception",
    version="0.1.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="A comprehensive computer vision and perception framework for aerial robotics and autonomous systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenPerception",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "black>=21.5b2",
            "isort>=5.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
        ],
        "ros": [
            "rclpy>=0.13.0",
        ],
        "web": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
            "python-multipart>=0.0.5",
        ],
        "ml": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "tensorboard>=2.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openperception=openperception.main:main",
            "openperception-benchmark=openperception.benchmarking.benchmark:main_benchmark_cli",
            "openperception-deploy=openperception.deployment.jetson_deployment:main_deploy_cli",
        ],
    },
) 