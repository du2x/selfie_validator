# Selfie Validator

A Python package for validating selfie image quality using computer vision techniques. Perfect for applications that need to ensure high-quality selfie input before processing.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Face Detection**: Automatically detects faces in selfie images
- **Quality Validation**: Checks multiple quality factors:
  - ✅ **Resolution**: Ensures minimum image resolution
  - ✅ **Sharpness**: Validates image clarity using Laplacian variance
  - ✅ **Lighting**: Analyzes brightness levels for optimal visibility
  - ✅ **Distance**: Validates face size relative to image (not too close/far)
  - ✅ **Angle**: Checks head angle alignment using eye detection
  - ✅ **Eye Detection**: Ensures both eyes are clearly visible

- **Flexible Configuration**: Customizable thresholds for all validation criteria
- **Multiple Input Formats**: Supports NumPy arrays, file paths, and byte data
- **Detailed Results**: Comprehensive validation results with specific metrics
- **Backward Compatibility**: Drop-in replacement for existing implementations

## Installation

```bash
pip install selfie-validator
```

### Development Installation

```bash
git clone https://github.com/yourusername/selfie-validator.git
cd selfie-validator
pip install -e .
```

## Quick Start

```python
from selfie_validator import SelfieValidator
import cv2

# Initialize validator with default settings
validator = SelfieValidator()

# Load an image
image = cv2.imread("path/to/selfie.jpg")

# Validate the selfie
results = validator.validate(image)

# Check if validation passed
if results["valid"]:
    print("✅ Great selfie!")
else:
    print(f"❌ Issues found: {validator.get_validation_summary(results)}")

# Access detailed metrics
print(f"Face ratio: {results['face_ratio']:.3f}")
print(f"Brightness: {results['brightness']:.1f}")
print(f"Sharpness score: {results['laplacian_var']:.1f}")
```

## Advanced Usage

### Custom Configuration

```python
from selfie_validator import SelfieValidator

# Create validator with custom thresholds
validator = SelfieValidator(
    min_resolution=(640, 640),           # Minimum image size
    sharpness_threshold=150.0,           # Higher sharpness requirement
    brightness_range=(80, 200),          # Wider brightness tolerance
    face_ratio_range=(0.15, 0.45),       # Adjust face size requirements
    max_angle_deviation=10.0             # More lenient angle tolerance
)
```

### Different Input Types

```python
# From file path
results = validator.validate("selfie.jpg")

# From bytes (e.g., uploaded file)
with open("selfie.jpg", "rb") as f:
    image_bytes = f.read()
results = validator.validate(image_bytes)

# From NumPy array
import cv2
image = cv2.imread("selfie.jpg")
results = validator.validate(image)
```

### Strict vs Non-Strict Mode

```python
# Strict mode: ALL checks must pass
try:
    results = validator.validate(image, strict_mode=True)
    print("Perfect selfie!")
except SelfieValidationError as e:
    print(f"Validation failed: {e}")

# Non-strict mode: At least 4 out of 6 checks must pass
results = validator.validate(image, strict_mode=False)
if results["valid"]:
    print("Good enough selfie!")
```

## API Reference

### SelfieValidator Class

#### Constructor Parameters

- `min_resolution` (tuple): Minimum image resolution (width, height). Default: (480, 480)
- `sharpness_threshold` (float): Minimum Laplacian variance for sharpness. Default: 100.0
- `brightness_range` (tuple): Acceptable brightness range (min, max). Default: (100, 180)
- `face_ratio_range` (tuple): Face area to image area ratio range. Default: (0.18, 0.50)
- `max_angle_deviation` (float): Maximum allowed angle deviation in degrees. Default: 8.0

#### Methods

##### `validate(image, strict_mode=True)`

Validates a selfie image and returns detailed results.

**Parameters:**
- `image`: Input image (NumPy array, file path, or bytes)
- `strict_mode` (bool): If True, all checks must pass. If False, at least 4/6 checks must pass.

**Returns:**
Dictionary with validation results:

```python
{
    "valid": bool,              # Overall validation result
    "resolution_ok": bool,      # Resolution check result
    "sharpness_ok": bool,       # Sharpness check result
    "light_ok": bool,           # Lighting check result
    "distance_ok": bool,        # Distance check result
    "angle_ok": bool,           # Angle check result
    "eyes_ok": bool,            # Eye detection result
    "faces_detected": int,      # Number of faces found
    "brightness": float,        # Average face brightness
    "face_ratio": float,        # Face area to image area ratio
    "laplacian_var": float,     # Sharpness metric
    "angle": float              # Head angle in degrees (if detectable)
}
```

##### `get_validation_summary(results)`

Returns a human-readable summary of validation results.

**Parameters:**
- `results`: Dictionary returned by `validate()`

**Returns:**
String with validation summary

### Backward Compatibility

For existing codebases, you can use the legacy function:

```python
from selfie_validator import analyze_selfie_image

# Drop-in replacement for existing implementations
results = analyze_selfie_image(image_array)
```

## Error Handling

The package provides specific exceptions for different error scenarios:

```python
from selfie_validator import SelfieValidator
from selfie_validator.exceptions import (
    SelfieValidationError,
    InvalidImageError,
    NoFaceDetectedError
)

validator = SelfieValidator()

try:
    results = validator.validate("path/to/image.jpg", strict_mode=True)
except InvalidImageError as e:
    print(f"Image processing error: {e}")
except NoFaceDetectedError as e:
    print(f"No face found: {e}")
except SelfieValidationError as e:
    print(f"Validation error: {e}")
```

## Quality Criteria Details

### Resolution Check
- Ensures image meets minimum size requirements
- Default: 480x480 pixels
- Helps guarantee sufficient detail for analysis

### Sharpness Check
- Uses Laplacian variance to measure image clarity
- Default threshold: 100.0
- Higher values indicate sharper images

### Lighting Check
- Analyzes average brightness in the face region
- Default range: 100-180 (0-255 scale)
- Ensures face is neither too dark nor overexposed

### Distance Check
- Measures face area relative to total image area
- Default range: 18%-50% of image
- Ensures face is appropriately sized (not too close/far)

### Angle Check
- Uses eye detection to measure head tilt
- Default tolerance: ±8 degrees
- Ensures face is relatively straight

### Eye Detection
- Confirms both eyes are visible and detectable
- Essential for angle calculation
- Indicates face is properly oriented

## Examples

### Integration with Web Applications

```python
from flask import Flask, request, jsonify
from selfie_validator import SelfieValidator
import cv2
import numpy as np

app = Flask(__name__)
validator = SelfieValidator()

@app.route('/validate-selfie', methods=['POST'])
def validate_selfie():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Convert uploaded file to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    try:
        results = validator.validate(image, strict_mode=False)
        return jsonify({
            'valid': results['valid'],
            'summary': validator.get_validation_summary(results),
            'details': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
```

### Batch Processing

```python
import os
from selfie_validator import SelfieValidator

validator = SelfieValidator()
image_folder = "path/to/selfies"

results = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        try:
            result = validator.validate(image_path, strict_mode=False)
            results.append({
                'filename': filename,
                'valid': result['valid'],
                'score': sum([
                    result['resolution_ok'],
                    result['sharpness_ok'],
                    result['light_ok'],
                    result['distance_ok'],
                    result['angle_ok'],
                    result['eyes_ok']
                ]) / 6.0  # Quality score 0-1
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Sort by quality score
results.sort(key=lambda x: x['score'], reverse=True)
print("Best selfies:", [r['filename'] for r in results[:5]])
```

## Requirements

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- Pillow (optional, for additional image format support)

### Development Setup

```bash
git clone https://github.com/du2x/selfie-validator.git
cd selfie-validator

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Format code
black .

# Type checking
mypy .
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0
- Initial release
- Core validation functionality
- Support for multiple input formats
- Comprehensive test suite
- Full documentation

## Support

- 📧 Email: du2x@pm.me
- 🐛 Issues: [GitHub Issues](https://github.com/du2x/selfie-validator/issues)
- 📖 Documentation: [GitHub README](https://github.com/du2x/selfie-validator#readme)

---

**Made with ❤️ for better selfie validation**