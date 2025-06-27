"""
Selfie Validator - A Python package for validating selfie image quality.

This package provides tools to analyze selfie images and validate their quality
based on various factors like lighting, angle, distance, sharpness, and face detection.

Perfect for applications that need to ensure high-quality selfie input before processing.
"""

from .validator import SelfieValidator
from .exceptions import SelfieValidationError

__version__ = "0.1.0"
__author__ = "du2x"
__email__ = "du2x@pm.me"

__all__ = ["SelfieValidator", "SelfieValidationError"]