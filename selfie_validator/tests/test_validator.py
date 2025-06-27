"""
Tests for the SelfieValidator class.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from selfie_validator.validator import SelfieValidator
from selfie_validator.exceptions import InvalidImageError, NoFaceDetectedError


class TestSelfieValidator:
    """Test cases for SelfieValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SelfieValidator()
        
        # Create a mock image (480x480, 3 channels)
        self.mock_image = np.zeros((480, 480, 3), dtype=np.uint8)
        self.mock_image[:, :] = [100, 100, 100]  # Gray image
    
    def test_init_default_parameters(self):
        """Test validator initialization with default parameters."""
        validator = SelfieValidator()
        assert validator.min_width == 480
        assert validator.min_height == 480
        assert validator.sharpness_threshold == 100.0
        assert validator.brightness_min == 100
        assert validator.brightness_max == 180
    
    def test_init_custom_parameters(self):
        """Test validator initialization with custom parameters."""
        validator = SelfieValidator(
            min_resolution=(640, 640),
            sharpness_threshold=150.0,
            brightness_range=(80, 200),
            face_ratio_range=(0.1, 0.6),
            max_angle_deviation=10.0
        )
        assert validator.min_width == 640
        assert validator.min_height == 640
        assert validator.sharpness_threshold == 150.0
        assert validator.brightness_min == 80
        assert validator.brightness_max == 200
        assert validator.face_ratio_min == 0.1
        assert validator.face_ratio_max == 0.6
        assert validator.max_angle_deviation == 10.0
    
    def test_prepare_image_numpy_array(self):
        """Test image preparation with numpy array input."""
        result = self.validator._prepare_image(self.mock_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 480, 3)
    
    def test_prepare_image_invalid_numpy_array(self):
        """Test image preparation with invalid numpy array."""
        invalid_image = np.zeros((480, 480), dtype=np.uint8)  # 2D array
        
        with pytest.raises(InvalidImageError):
            self.validator._prepare_image(invalid_image)
    
    def test_prepare_image_invalid_type(self):
        """Test image preparation with invalid input type."""
        with pytest.raises(InvalidImageError):
            self.validator._prepare_image(123)
    
    def test_check_resolution_valid(self):
        """Test resolution check with valid image."""
        result = self.validator._check_resolution(self.mock_image)
        assert result is True
    
    def test_check_resolution_invalid(self):
        """Test resolution check with invalid image."""
        small_image = np.zeros((300, 300, 3), dtype=np.uint8)
        result = self.validator._check_resolution(small_image)
        assert result is False
    
    @patch('cv2.CascadeClassifier.detectMultiScale')
    def test_validate_no_face_strict_mode(self, mock_detect):
        """Test validation with no face detected in strict mode."""
        mock_detect.return_value = []  # No faces detected
        
        with pytest.raises(NoFaceDetectedError):
            self.validator.validate(self.mock_image, strict_mode=True)
    
    @patch('cv2.CascadeClassifier.detectMultiScale')
    def test_validate_no_face_non_strict_mode(self, mock_detect):
        """Test validation with no face detected in non-strict mode."""
        mock_detect.return_value = []  # No faces detected
        
        result = self.validator.validate(self.mock_image, strict_mode=False)
        assert result["valid"] is False
        assert result["faces_detected"] == 0
    
    @patch('cv2.CascadeClassifier.detectMultiScale')
    def test_validate_with_face_all_checks_pass(self, mock_detect):
        """Test validation with face detected and all checks passing."""
        # Mock face detection - return a face in the center
        mock_detect.side_effect = [
            [(190, 190, 100, 100)],  # Face detection
            [(10, 10, 20, 20), (60, 10, 20, 20)]  # Eye detection
        ]
        
        # Create image with good sharpness
        sharp_image = np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8)
        
        result = self.validator.validate(sharp_image, strict_mode=False)
        
        # Basic checks
        assert result["faces_detected"] == 1
        assert result["resolution_ok"] is True
        assert result["eyes_ok"] is True
    
    def test_get_validation_summary_success(self):
        """Test validation summary for successful validation."""
        results = {"valid": True}
        summary = self.validator.get_validation_summary(results)
        assert "✅ Selfie validation passed" in summary
    
    def test_get_validation_summary_failure(self):
        """Test validation summary for failed validation."""
        results = {
            "valid": False,
            "resolution_ok": False,
            "sharpness_ok": True,
            "light_ok": True,
            "distance_ok": True,
            "angle_ok": True,
            "eyes_ok": True
        }
        summary = self.validator.get_validation_summary(results)
        assert "❌ Selfie validation failed" in summary
        assert "Resolution too low" in summary
    
    def test_backward_compatibility_function(self):
        """Test the backward compatibility function."""
        from selfie_validator.validator import analyze_selfie_image
        
        with patch.object(SelfieValidator, 'validate') as mock_validate:
            mock_validate.return_value = {"valid": True}
            
            result = analyze_selfie_image(self.mock_image)
            assert result["valid"] is True
            mock_validate.assert_called_once_with(self.mock_image, strict_mode=False)


if __name__ == "__main__":
    pytest.main([__file__])