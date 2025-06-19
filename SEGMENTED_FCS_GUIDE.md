# Segmented FCS Analysis v1.6.1

## Overview
Segmented Fluorescence Correlation Spectroscopy (sFCS) analysis has been integrated for specialized line-scan data analysis with temporal segmentation capabilities.

## Key Features

### 1. Segmented FCS Module (`fcs_analysis.py`)
- Temporal segmentation of line-scan data
- Support for x-axis (pixel) or y-axis (line) segmentation
- Advanced autocorrelation function calculation
- Multiple FCS fitting models (2D, 3D, anomalous diffusion)
- Parameter error estimation and quality assessment

### 2. Analysis Integration (`analysis_modules.py`)
- SegmentedFCSAnalysis class with nuclear alignment support
- Comprehensive parameter handling and validation
- Multi-dimensional data support with proper preprocessing
- Quality metrics and reliability assessment

### 3. User Interface (`app.py`)
- Intuitive parameter controls for segmentation settings
- Timing parameter configuration (pixel time, line time)
- Model selection (2D, 3D, anomalous diffusion)
- Advanced visualization and results display

## Scientific Implementation

### Autocorrelation Calculation
- Proper normalized autocorrelation: G(τ) = ⟨δI(t)δI(t+τ)⟩ / ⟨I⟩²
- Configurable maximum lag as fraction of segment length
- Robust handling of different segment sizes

### FCS Models
1. **2D Diffusion**: G(τ) = G₀ / (1 + 4Dτ/w₀²)
2. **3D Diffusion**: G(τ) = G₀ / ((1 + 4Dτ/w₀²)√(1 + 4Dτ/wz²))
3. **Anomalous Diffusion**: G(τ) = G₀ / (1 + (4Dτ/w₀²)^α)

### Quality Assessment
- R-squared goodness of fit for each segment
- Parameter error estimation from covariance matrix
- Success rate calculation across all segments
- Statistical analysis of parameter distributions

## Usage

### Basic Segmented FCS Analysis
1. Load line-scan microscopy data (2D format)
2. Select "Segmented FCS" analysis
3. Configure segmentation parameters:
   - Segmentation type (x or y axis)
   - Segment length (32-8192 pixels/lines)
   - FCS model (2D, 3D, anomalous)
   - Timing parameters (pixel time, line time)
4. Run analysis and review results

### Results Interpretation
- **Segment Overview**: Parameter distributions and statistics
- **Correlation Functions**: Individual autocorrelation curves
- **Parameter Maps**: Spatial variation of diffusion parameters
- **Quality Assessment**: Fit quality and reliability metrics

## Nuclear Alignment Integration
- Automatic alignment for time-series data before analysis
- Quality metrics for alignment reliability
- Fallback to single-frame analysis for 2D data
- Comprehensive reliability assessment

## Applications
- High-resolution diffusion measurements in cellular environments
- Spatial heterogeneity analysis of molecular mobility
- Time-resolved dynamics in biological systems
- Quality-controlled biophysical parameter mapping
