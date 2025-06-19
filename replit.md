# replit.md

## Overview

This is a comprehensive Streamlit-based application for microscopy data analysis and biophysical research. The application provides multi-format microscopy data loading, AI-powered image enhancement, specialized physics analysis methods, and automated report generation capabilities. It's designed to handle various microscopy formats (TIFF, STK, LSM, CZI, LIF, OIF, OIB) and perform advanced analyses including fluorescence correlation spectroscopy, optical flow analysis, and nuclear biophysics studies.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-tab interface
- **User Interface**: Clean, scientific-focused UI with real-time thumbnails and interactive visualizations
- **Visualization**: Combination of Matplotlib and Plotly for static and interactive plots
- **Configuration**: Custom Streamlit theming with scientific color schemes

### Backend Architecture
- **Language**: Python 3.11+
- **Core Libraries**: NumPy, SciPy, scikit-image, OpenCV for scientific computing
- **Specialized Libraries**: multipletau (FCS), trackpy (particle tracking), fcsfiles (Zeiss data)
- **AI Enhancement**: PyTorch for deep learning, with optional TensorFlow support
- **Data Processing**: Pandas for tabular data, tifffile/PIMS for microscopy formats

### Modular Design
The application follows a clean modular architecture with separate concerns:
- Data loading and format detection
- Analysis method implementations
- AI enhancement and segmentation
- Visualization and reporting
- Utility functions and validation

## Key Components

### Data Loading System (`data_loader.py`)
- **Multi-format Support**: Handles TIFF, STK, LSM, CZI, LIF, OIF, OIB formats
- **Metadata Extraction**: Preserves pixel size, time intervals, and acquisition parameters
- **Format Detection**: Automatic format recognition with capability assessment
- **Graceful Fallbacks**: PIL-based fallback when specialized readers unavailable

### Analysis Engine (`analysis_modules.py`)
- **FCS/RICS Analysis**: Complete fluorescence correlation spectroscopy suite
- **Particle Tracking**: Integration with trackpy for single particle analysis
- **iMSD Analysis**: Image mean square displacement for diffusion mapping
- **Optical Flow**: Lucas-Kanade, Farneback, and custom correlation methods

### AI Enhancement (`ai_enhancement.py`)
- **Denoising**: Non-local means, Richardson-Lucy deconvolution
- **Segmentation**: Cellpose and StarDist integration for nuclear/cellular segmentation
- **Deep Learning**: PyTorch-based enhancement with optional GPU acceleration

### Specialized Physics (`advanced_analysis.py`, `nuclear_biophysics.py`)
- **Nuclear Dynamics**: Binding kinetics and chromatin organization analysis
- **Elastography**: Force propagation and mechanical property mapping
- **Image Correlation**: RICS, STICS, imaging FCS implementations

### Reporting System (`report_generator.py`)
- **Automated Documentation**: Data-dependent analysis report generation
- **Multiple Formats**: Markdown, HTML, JSON, CSV export capabilities
- **Comprehensive Analysis**: Method documentation and parameter recording

## Data Flow

1. **Data Import**: Multi-format file upload with thumbnail generation
2. **Format Detection**: Automatic identification and metadata extraction
3. **Analysis Selection**: User chooses from available analysis methods
4. **Parameter Configuration**: Method-specific parameter tuning
5. **Processing**: Analysis execution with progress tracking
6. **Visualization**: Interactive result display and exploration
7. **Report Generation**: Automated documentation and export

## External Dependencies

### Core Scientific Stack
- numpy (≥1.24.0) - Numerical computing
- scipy (≥1.10.0) - Scientific algorithms
- matplotlib (≥3.7.0) - Plotting and visualization
- pandas (≥2.0.0) - Data manipulation
- scikit-image (≥0.20.0) - Image processing

### Microscopy-Specific Libraries
- tifffile (≥2023.7.10) - TIFF format support
- pims (≥0.6.1) - Image sequence handling
- h5py (≥3.9.0) - HDF5 data format
- multipletau (≥0.3.3) - FCS correlation analysis
- trackpy (≥0.6.1) - Particle tracking

### Optional Format Support
- readlif (≥0.6.5) - Leica LIF files
- pylibczirw (≥3.4.0) - Zeiss CZI files
- fcsfiles (≥2022.9.28) - Zeiss Confocor data

### AI/ML Libraries
- torch (≥2.0.0) - Deep learning framework
- torchvision (≥0.15.0) - Computer vision utilities
- cellpose (optional) - Cell segmentation
- stardist (optional) - Nuclear segmentation

## Deployment Strategy

### Replit Environment
- **Python Version**: 3.11 with scientific computing packages
- **Port Configuration**: Default port 5000 for Streamlit server
- **Resource Requirements**: 4GB+ RAM recommended for large datasets
- **Package Management**: uv for dependency resolution

### Installation Options
1. **Automated Deployment**: `./deploy.sh --with-optional --debug --auto-start`
2. **Manual Installation**: Virtual environment setup with pip
3. **Docker Support**: Containerized deployment for reproducibility

### Configuration
- Streamlit server configured for headless operation
- Custom theming for scientific applications
- Matplotlib backend optimization for web display

## Recent Changes

### June 16, 2025 - Segmented FCS and Universal Alignment Integration (v1.6.1)
- **Segmented FCS Analysis**: Added specialized line-scan FCS with temporal segmentation for high-resolution diffusion measurements
- **Advanced FCS Models**: Support for 2D, 3D, and anomalous diffusion fitting with comprehensive statistics
- **Universal Nuclear Alignment**: Comprehensive alignment across all biophysical analyses (RICS, FCS, iMSD, FRAP, FLIM, SPT, N&B)
- **Enhanced Temporal Stability**: Phase correlation, optical flow, and feature-based alignment methods for robust registration
- **Reliability Assessment**: Automatic quality metrics and confidence scoring for all analysis results
- **PyTorch Conflict Resolution**: Fixed Streamlit file watcher conflicts with isolated PyTorch loading
- **Comprehensive Visualization**: Advanced plotting for segmented analysis results with quality assessment

### June 15, 2025 - Critical Biophysical Analysis Corrections (v1.5.2)
- **Fixed Critical RICS Analysis**: Replaced incorrect pixel correlation with proper FFT-based spatial autocorrelation
- **Corrected iMSD Implementation**: Implemented feature-based particle tracking instead of pixel intensity calculations
- **Enhanced AI Integration**: Re-enabled PyTorch with proper dimension handling and thread management
- **Improved Richardson-Lucy**: Fixed 3D PSF generation errors and added frame-by-frame processing
- **Validated Analysis Methods**: All biophysical calculations now scientifically accurate and quantitatively reliable

### Critical Fixes Applied
- RICS now uses G(dx,dy) = IFFT(FFT(δI) * conj(FFT(δI))) with 2D Gaussian fitting
- iMSD employs trackpy-based particle detection and trajectory linking
- Multi-dimensional image handling with proper (T,Y,X) vs (Y,X,C) detection
- Enhanced error handling with graceful fallbacks for missing libraries
- Comprehensive quality metrics and validation for all analysis results

### Previous Improvements (v1.5.1)
- Enhanced image preview with Z-slice navigation and time point controls
- Integrated alignment preview with before/after comparison functionality
- Fixed Streamlit API compatibility issues and nested expander conflicts
- Created robust error handling and diagnostic tools

## Changelog

- June 15, 2025 (v1.5.2): Critical biophysical analysis corrections - RICS, iMSD, AI integration
- June 15, 2025 (v1.5.1): Enhanced preview features, alignment integration, debug package
- June 14, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.