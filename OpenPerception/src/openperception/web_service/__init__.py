"""
Web service module for OpenPerception.

This module provides a FastAPI-based web API for interacting with the framework,
allowing remote control, data upload, and monitoring of perception tasks.
"""

from .api import WebServiceAPI

__all__ = [
    'WebServiceAPI'
]
