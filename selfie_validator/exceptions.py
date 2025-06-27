"""
Exception classes for selfie validation.
"""


class SelfieValidationError(Exception):
    """Base exception for selfie validation errors."""
    pass


class InvalidImageError(SelfieValidationError):
    """Raised when the input image is invalid or cannot be processed."""
    pass


class NoFaceDetectedError(SelfieValidationError):
    """Raised when no face is detected in the image."""
    pass


class QualityCheckError(SelfieValidationError):
    """Raised when image quality checks fail."""
    pass