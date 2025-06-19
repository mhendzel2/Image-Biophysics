# Advanced Image Biophysics

A comprehensive Streamlit-based application for microscopy data analysis, featuring AI-powered enhancement tools, specialized physics methods, and automated reporting capabilities.

## Features

### Data Loading & Preview
- **Multi-format support**: TIFF, STK, LSM, CZI, LIF, OIF, OIB formats
- **Real-time thumbnails**: Visual previews with metadata extraction
- **FCS data support**: Correlation spectroscopy data import
- **Format detection**: Automatic format recognition and capability assessment

### Analysis Methods
- **Fluorescence Correlation Spectroscopy (FCS)**: Complete correlation analysis with multipletau
- **Raster Image Correlation Spectroscopy (RICS)**: Spatial autocorrelation mapping
- **Image Mean Square Displacement (iMSD)**: Diffusion behavior analysis
- **Single Particle Tracking (SPT)**: Trackpy-based particle tracking
- **Optical Flow Analysis**: Lucas-Kanade, Farneback, DIC, phase correlation
- **Image Correlation Spectroscopy (ICS)**: RICS, STICS, imaging FCS, pair correlation

### AI Enhancement
- **Denoising**: Non-local means, Richardson-Lucy deconvolution
- **Segmentation**: Cellpose and StarDist integration (when available)
- **Noise2Void**: Self-supervised denoising (simplified implementation)
- **Advanced methods**: CARE restoration, enhanced deconvolution

### Automated Reporting
- **Adaptive reports**: Data-dependent analysis documentation
- **Multiple formats**: Markdown, JSON, HTML, CSV export
- **Comprehensive analysis**: Complete method documentation
- **Specialized reports**: Physics-focused, AI enhancement, FCS analysis

## Installation

### Requirements
- Python 3.11+
- Streamlit
- Core scientific libraries (numpy, scipy, matplotlib, pandas)
- Specialized libraries (see Optional Dependencies)

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd advanced-image-biophysics

# Install Python dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port 5000
```

### Optional Dependencies
For full functionality, install additional libraries:

```bash
# Microscopy format support
pip install tifffile readlif pylibczirw

# AI enhancement (optional)
pip install tensorflow cellpose stardist

# Advanced analysis
pip install noise2void csbdeep

# Report generation
pip install reportlab jinja2
```

## Usage

### Starting the Application
```bash
streamlit run app.py --server.port 5000
```
Access the application at `http://localhost:5000`

### Basic Workflow
1. **Load Data**: Upload microscopy files with automatic format detection
2. **Preview**: View thumbnails and metadata before analysis
3. **Analyze**: Choose from 9+ biophysical analysis methods
4. **Enhance**: Apply AI-powered image enhancement
5. **Visualize**: Interactive plots and results display
6. **Export**: Generate automated reports and download results

### Supported File Formats

| Format | Extension | Library | Description |
|--------|-----------|---------|-------------|
| MetaMorph | .tif, .stk | tifffile | Stack files and TIFF images |
| Zeiss LSM | .lsm | tifffile | Laser scanning microscopy |
| Zeiss CZI | .czi | pylibczirw | Zeiss Zen format |
| Leica LIF | .lif | readlif | Leica SP8 format |
| Olympus | .oif, .oib | tifffile | Olympus imaging |
| FCS Data | .fcs, .raw, .csv | Custom | Correlation data |

## Architecture

### Core Modules
- `app.py` - Main Streamlit interface
- `data_loader.py` - Multi-format data loading
- `analysis_modules.py` - Biophysical analysis methods
- `ai_enhancement.py` - AI-powered enhancement tools
- `advanced_analysis.py` - Specialized physics methods

### Specialized Features
- `optical_flow_analysis.py` - Elastography and flow analysis
- `image_correlation_spectroscopy.py` - Complete ICS suite
- `thumbnail_generator.py` - Format preview system
- `report_generator.py` - Automated documentation
- `fcs_data_loader.py` - FCS data handling

### Visualization
- `visualization.py` - Interactive plotting
- `utils.py` - Utility functions

## Configuration

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
base = "light"
```

### Environment Variables
- `OPENAI_API_KEY` - For AI enhancement features (optional)
- `CUDA_VISIBLE_DEVICES` - GPU selection for PyTorch (optional)

## Development

### Project Structure
```
advanced-image-biophysics/
├── app.py                              # Main application
├── data_loader.py                      # Data loading
├── analysis_modules.py                 # Core analysis
├── ai_enhancement.py                   # AI tools
├── advanced_analysis.py                # Advanced methods
├── optical_flow_analysis.py            # Flow analysis
├── image_correlation_spectroscopy.py   # ICS suite
├── thumbnail_generator.py              # Preview system
├── report_generator.py                 # Automated reports
├── fcs_data_loader.py                  # FCS support
├── visualization.py                    # Plotting
├── utils.py                           # Utilities
├── requirements.txt                    # Dependencies
├── README.md                          # Documentation
└── .streamlit/config.toml             # Configuration
```

### Adding New Analysis Methods
1. Create analysis class inheriting from `BaseAnalysis`
2. Implement `analyze()` method
3. Add to `AnalysisManager` in `analysis_modules.py`
4. Update UI in `app.py`

### Adding New File Formats
1. Add format detection to `data_loader.py`
2. Implement loader method
3. Update `thumbnail_generator.py` for previews
4. Add format to supported list

## API Reference

### Core Classes

#### DataLoader
```python
loader = DataLoader()
data_info = loader.load_file(uploaded_file)
```

#### AnalysisManager
```python
manager = AnalysisManager()
results = manager.run_analysis(analysis_type, data_info, parameters)
```

#### AIEnhancementManager
```python
enhancer = AIEnhancementManager()
enhanced = enhancer.enhance_image(image_data, method, parameters)
```

#### ThumbnailGenerator
```python
generator = ThumbnailGenerator()
thumbnail = generator.generate_format_thumbnail(file_path)
```

#### ReportGenerator
```python
reporter = ReportGenerator()
report = reporter.generate_report(data_info, results, report_type)
```

## Troubleshooting

### Common Issues

**Import Errors for Optional Libraries**
- Install missing libraries: `pip install <library-name>`
- Application continues with reduced functionality

**Memory Issues with Large Files**
- Reduce image size or use binning
- Close other applications
- Consider chunked processing

**Format Not Supported**
- Check if format-specific library is installed
- Use fallback TIFF conversion
- Contact support for new format requests

### Debug Mode
Run with debug information:
```bash
streamlit run app.py --logger.level debug --server.port 5000
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## Support

For technical support:
- Check troubleshooting section
- Review console logs for errors
- Ensure all dependencies are installed
- Verify file format compatibility

## Citation

If using this software in research, please cite:
```
Advanced Image Biophysics Analysis Platform
Version 1.0
https://github.com/your-repo/advanced-image-biophysics
```