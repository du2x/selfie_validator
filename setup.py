"""
Setup configuration for selfie-validator package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="selfie-validator",
    version="0.1.0",
    author="du2x",
    author_email="du2x@pm.me",
    description="A Python package for validating selfie image quality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/du2x/selfie-validator",
    project_urls={
        "Bug Tracker": "https://github.com/du2x/selfie-validator/issues",
        "Documentation": "https://github.com/du2x/selfie-validator#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="selfie, image validation, face detection, opencv, computer vision",
)