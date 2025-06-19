#!/bin/bash
# Deployment script for Advanced Image Biophysics

echo "Advanced Image Biophysics Deployment Script"
echo "==========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking system requirements..."

if ! command_exists python3; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

if ! command_exists pip; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi

echo "✅ System requirements met"

# Parse command line arguments
INSTALL_OPTIONAL=false
RUN_DEBUG=false
AUTO_START=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-optional)
            INSTALL_OPTIONAL=true
            shift
            ;;
        --debug)
            RUN_DEBUG=true
            shift
            ;;
        --auto-start)
            AUTO_START=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --with-optional    Install optional AI and format libraries"
            echo "  --debug           Run debug diagnostics after installation"
            echo "  --auto-start      Automatically start the application"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install streamlit>=1.28.0 numpy>=1.24.0 scipy>=1.10.0 matplotlib>=3.7.0 pandas>=2.0.0
pip install plotly>=5.15.0 scikit-image>=0.20.0 opencv-python>=4.8.0
pip install tifffile>=2023.7.10 h5py>=3.9.0 pims>=0.6.1
pip install multipletau>=0.3.3 lmfit>=1.2.0 trackpy>=0.6.1
pip install fcsfiles>=2022.9.28 torch>=2.0.0 torchvision>=0.15.0

if [ $? -ne 0 ]; then
    echo "❌ Core dependency installation failed"
    exit 1
fi

echo "✅ Core dependencies installed"

# Install optional dependencies if requested
if [ "$INSTALL_OPTIONAL" = true ]; then
    echo "Installing optional dependencies..."
    
    # Format support libraries
    pip install readlif>=0.6.5 || echo "⚠️ readlif installation failed"
    pip install pylibczirw>=3.4.0 || echo "⚠️ pylibczirw installation failed"
    
    # AI enhancement libraries (may fail on some systems)
    pip install tensorflow>=2.13.0 || echo "⚠️ TensorFlow installation failed"
    pip install cellpose>=2.2.0 || echo "⚠️ Cellpose installation failed"
    pip install stardist>=0.8.3 || echo "⚠️ StarDist installation failed"
    
    # Report generation
    pip install reportlab>=4.0.4 || echo "⚠️ ReportLab installation failed"
    pip install jinja2>=3.1.0 || echo "⚠️ Jinja2 installation failed"
    pip install markdown>=3.4.0 || echo "⚠️ Markdown installation failed"
    
    echo "✅ Optional dependencies installation attempted"
fi

# Run debug check if requested
if [ "$RUN_DEBUG" = true ]; then
    echo "Running system diagnostics..."
    python3 debug_tools.py
fi

# Set up configuration if not exists
if [ ! -d ".streamlit" ]; then
    echo "Setting up Streamlit configuration..."
    mkdir -p .streamlit
fi

# Make scripts executable
chmod +x install.sh
chmod +x debug_tools.py

echo "✅ Deployment complete!"
echo ""
echo "Quick start commands:"
echo "  Source environment: source venv/bin/activate"
echo "  Run diagnostics:    python3 debug_tools.py"
echo "  Start application:  streamlit run app.py --server.port 5000"
echo ""
echo "Application URL: http://localhost:5000"

# Auto-start if requested
if [ "$AUTO_START" = true ]; then
    echo "Starting application..."
    streamlit run app.py --server.port 5000
fi