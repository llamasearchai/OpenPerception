"""
Benchmarking module for OpenPerception.

This module provides tools and scripts for evaluating the performance
of various perception algorithms within the framework.
"""

from .benchmark import PerformanceBenchmark, BenchmarkResult, main_benchmark_cli

__all__ = [
    'PerformanceBenchmark',
    'BenchmarkResult',
    'main_benchmark_cli'
]
