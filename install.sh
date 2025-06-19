#!/bin/bash
# Installation script for Advanced Image Biophysics

echo "Installing Advanced Image Biophysics..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [[ $(echo "$python_version >= $required_version" | bc -l) -ne 1 ]]; then
    echo "Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "Python version check passed: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install streamlit>=1.28.0
pip install numpy>=1.24.0 scipy>=1.10.0 matplotlib>=3.7.0 pandas>=2.0.0
pip install plotly>=5.15.0 scikit-image>=0.20.0 opencv-python>=4.8.0
pip install tifffile>=2023.7.10 h5py>=3.9.0 pims>=0.6.1
pip install multipletau>=0.3.3 lmfit>=1.2.0 trackpy>=0.6.1
pip install fcsfiles>=2022.9.28
pip install torch>=2.0.0 torchvision>=0.15.0

# Optional dependencies with error handling
echo "Installing optional dependencies..."

# Format support
echo "Attempting to install format support libraries..."
pip install readlif>=0.6.5 || echo "Warning: readlif installation failed (Leica LIF support disabled)"
pip install pylibczirw>=3.4.0 || echo "Warning: pylibczirw installation failed (Zeiss CZI support disabled)"

# AI enhancement (these may fail on some systems)
echo "Attempting to install AI enhancement libraries..."
pip install tensorflow>=2.13.0 || echo "Warning: TensorFlow installation failed (AI features limited)"
pip install cellpose>=2.2.0 || echo "Warning: Cellpose installation failed (cell segmentation disabled)"
pip install stardist>=0.8.3 || echo "Warning: StarDist installation failed (nucleus segmentation disabled)"

# Report generation
echo "Attempting to install report generation libraries..."
pip install reportlab>=4.0.4 || echo "Warning: ReportLab installation failed (PDF reports disabled)"
pip install jinja2>=3.1.0 || echo "Warning: Jinja2 installation failed (template rendering limited)"
pip install markdown>=3.4.0 || echo "Warning: Markdown installation failed (markdown reports disabled)"

echo "Installation complete!"
echo ""
echo "To run the application:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py --server.port 5000"
echo ""
echo "Access the application at: http://localhost:5000"