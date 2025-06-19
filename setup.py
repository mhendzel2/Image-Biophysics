#!/usr/bin/env python3
"""
Setup script for Advanced Image Biophysics package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read dependencies
with open(os.path.join(this_directory, 'dependencies.txt'), encoding='utf-8') as f:
    dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="advanced-image-biophysics",
    version="1.0.0",
    author="Advanced Biophysics Team",
    author_email="contact@advancedbiophysics.com",
    description="Comprehensive microscopy data analysis with AI enhancement and automated reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/advanced-image-biophysics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "plotly>=5.15.0",
        "scikit-image>=0.20.0",
        "opencv-python>=4.8.0",
        "tifffile>=2023.7.10",
        "h5py>=3.9.0",
        "pims>=0.6.1",
        "multipletau>=0.3.3",
        "lmfit>=1.2.0",
        "trackpy>=0.6.1",
        "fcsfiles>=2022.9.28",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
    extras_require={
        "formats": [
            "readlif>=0.6.5",
            "pylibczirw>=3.4.0",
        ],
        "ai": [
            "tensorflow>=2.13.0",
            "cellpose>=2.2.0", 
            "stardist>=0.8.3",
            "noise2void>=0.3.0",
            "csbdeep>=0.7.4",
        ],
        "reports": [
            "reportlab>=4.0.4",
            "jinja2>=3.1.0",
            "markdown>=3.4.0",
        ],
        "all": [
            "readlif>=0.6.5",
            "pylibczirw>=3.4.0",
            "tensorflow>=2.13.0",
            "cellpose>=2.2.0",
            "stardist>=0.8.3", 
            "noise2void>=0.3.0",
            "csbdeep>=0.7.4",
            "reportlab>=4.0.4",
            "jinja2>=3.1.0",
            "markdown>=3.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "advanced-biophysics=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.md", "*.txt"],
    },
    keywords="microscopy biophysics image-analysis fcs correlation-spectroscopy ai-enhancement",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/advanced-image-biophysics/issues",
        "Source": "https://github.com/your-repo/advanced-image-biophysics",
        "Documentation": "https://github.com/your-repo/advanced-image-biophysics/wiki",
    },
)