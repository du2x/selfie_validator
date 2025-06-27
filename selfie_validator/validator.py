"""
Selfie validation module with enhanced features and configurability.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Union
from selfie_validator.exceptions import InvalidImageError, NoFaceDetectedError


class SelfieValidator:
    """
    A comprehensive selfie validation class that analyzes image quality
    based on multiple factors including lighting, angle, distance, and sharpness.
    """
    
    def __init__(
        self,
        min_resolution: Tuple[int, int] = (480, 480),
        sharpness_threshold: float = 100.0,
        brightness_range: Tuple[int, int] = (100, 180),
        face_ratio_range: Tuple[float, float] = (0.18, 0.50),
        max_angle_deviation: float = 8.0
    ):
        """
        Initialize the SelfieValidator with configurable parameters.
        
        Args:
            min_resolution: Minimum required image resolution (width, height)
            sharpness_threshold: Minimum Laplacian variance for sharpness
            brightness_range: Acceptable brightness range (min, max)
            face_ratio_range: Face area to image area ratio range (min, max)
            max_angle_deviation: Maximum allowed angle deviation in degrees
        """
        self.min_width, self.min_height = min_resolution
        self.sharpness_threshold = sharpness_threshold
        self.brightness_min, self.brightness_max = brightness_range
        self.face_ratio_min, self.face_ratio_max = face_ratio_range
        self.max_angle_deviation = max_angle_deviation
        
        # Initialize OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml" # type: ignore
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml" # type: ignore
        )
    
    def validate(
        self, 
        image: Union[np.ndarray, str, bytes], 
        strict_mode: bool = True
    ) -> Dict:
        """
        Validate a selfie image and return detailed analysis results.
        
        Args:
            image: Input image as numpy array, file path string, or bytes
            strict_mode: If True, all checks must pass for valid=True
        
        Returns:
            Dict containing validation results and detailed metrics
            
        Raises:
            InvalidImageError: If image cannot be processed
            NoFaceDetectedError: If no face is detected (in strict mode)
        """
        # Convert input to numpy array
        img = self._prepare_image(image)
        
        # Perform all validation checks
        results = {
            "valid": False,
            "resolution_ok": self._check_resolution(img),
            "sharpness_ok": False,
            "laplacian_var": 0.0,
            "light_ok": False,
            "brightness": 0.0,
            "distance_ok": False,
            "face_ratio": 0.0,
            "angle_ok": False,
            "angle": None,
            "eyes_ok": False,
            "faces_detected": 0
        }
        
        # Check resolution first
        if not results["resolution_ok"]:
            if strict_mode:
                raise InvalidImageError(
                    f"Resolution too low (minimum {self.min_width}x{self.min_height})"
                )
            return results
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check sharpness
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        results["laplacian_var"] = laplacian_var
        results["sharpness_ok"] = laplacian_var > self.sharpness_threshold
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        results["faces_detected"] = len(faces)
        
        if len(faces) == 0:
            if strict_mode:
                raise NoFaceDetectedError("No face detected in the image")
            return results
        
        # Analyze the largest face (assuming it's the primary subject)
        face = max(faces, key=lambda f: f[2] * f[3])  # Sort by area
        x, y, w, h = face
        face_gray = gray[y:y + h, x:x + w]
        
        # Check lighting
        brightness = float(np.mean(face_gray))
        results["brightness"] = brightness
        results["light_ok"] = self.brightness_min < brightness < self.brightness_max
        
        # Check distance (face size relative to image)
        face_area = w * h
        image_area = img.shape[0] * img.shape[1]
        face_ratio = float(face_area) / float(image_area)
        results["face_ratio"] = face_ratio
        results["distance_ok"] = self.face_ratio_min < face_ratio < self.face_ratio_max
        
        # Check angle by eye alignment
        eyes = self.eye_cascade.detectMultiScale(face_gray)
        results["eyes_ok"] = len(eyes) >= 2
        
        if results["eyes_ok"]:
            # Sort eyes by x-coordinate and take the two leftmost
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
            
            # Calculate eye centers
            cx1, cy1 = ex1 + ew1 / 2, ey1 + eh1 / 2
            cx2, cy2 = ex2 + ew2 / 2, ey2 + eh2 / 2
            
            # Calculate angle between eyes
            angle = float(np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1)))
            results["angle"] = angle
            results["angle_ok"] = abs(angle) < self.max_angle_deviation
        
        # Overall validation
        if strict_mode:
            results["valid"] = all([
                results["resolution_ok"],
                results["sharpness_ok"],
                results["light_ok"],
                results["distance_ok"],
                results["angle_ok"],
                results["eyes_ok"]
            ])
        else:
            # In non-strict mode, allow some flexibility
            passed_checks = sum([
                results["resolution_ok"],
                results["sharpness_ok"],
                results["light_ok"],
                results["distance_ok"],
                results["angle_ok"],
                results["eyes_ok"]
            ])
            results["valid"] = passed_checks >= 4  # At least 4 out of 6 checks
        
        return results
    
    def _prepare_image(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """Convert various image input formats to numpy array."""
        if isinstance(image, np.ndarray):
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise InvalidImageError("Image must be a 3-channel (BGR) numpy array")
            return image
        
        elif isinstance(image, str):
            # File path
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError(f"Could not load image from path: {image}")
            return img
        
        elif isinstance(image, bytes):
            # Bytes data
            img_array = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError("Could not decode image from bytes")
            return img
        
        else:
            raise InvalidImageError(f"Unsupported image type: {type(image)}")
    
    def _check_resolution(self, img: np.ndarray) -> bool:
        """Check if image meets minimum resolution requirements."""
        return img.shape[0] >= self.min_height and img.shape[1] >= self.min_width
    
    def get_validation_summary(self, results: Dict) -> str:
        """
        Generate a human-readable summary of validation results.
        
        Args:
            results: Validation results from validate() method
            
        Returns:
            String summary of validation status
        """
        if results["valid"]:
            return "✅ Selfie validation passed - image quality is good!"
        
        issues = []
        if not results["resolution_ok"]:
            issues.append(f"Resolution too low (minimum {self.min_width}x{self.min_height})")
        if not results["sharpness_ok"]:
            issues.append("Image is not sharp enough")
        if not results["light_ok"]:
            issues.append("Poor lighting conditions")
        if not results["distance_ok"]:
            issues.append("Face is too close or too far")
        if not results["angle_ok"]:
            issues.append("Head angle is not straight")
        if not results["eyes_ok"]:
            issues.append("Could not detect both eyes clearly")
        
        return f"❌ Selfie validation failed: {', '.join(issues)}"


# Backward compatibility function
def analyze_selfie_image(img: np.ndarray) -> Dict:
    """
    Legacy function for backward compatibility.
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Dict with validation results
    """
    validator = SelfieValidator()
    return validator.validate(img, strict_mode=False)
