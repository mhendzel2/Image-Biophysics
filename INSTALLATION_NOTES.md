# Installation Notes - Image Biophysics

## Successful Installation ✓

A Python virtual environment has been created and the core packages have been successfully installed.

### Environment Details
- **Python Version**: 3.12.10
- **Virtual Environment**: `venv/` (already in .gitignore)
- **NumPy Version**: 1.26.4 (capped at <2.0 for AI package compatibility)

### Successfully Installed Packages

#### Core Scientific Stack
- NumPy 1.26.4, SciPy 1.16.2, Pandas 2.3.3
- Matplotlib 3.10.6, Plotly 6.3.1
- scikit-image 0.25.2, OpenCV 4.11.0.86
- tifffile, h5py, pims, trackpy

#### Imaging & Spectroscopy
- readlif, pylibczirw (Zeiss CZI support)
- fcsfiles, multipletau, lmfit

#### AI/Deep Learning
- PyTorch 2.8.0, torchvision 0.23.0
- Cellpose 4.0.6 (for cell segmentation)

#### Web Interface
- Streamlit 1.50.0

## Known Limitations & Conflicts

### 1. TensorFlow Not Installed
**Issue**: TensorFlow versions 2.13-2.15 don't support Python 3.12  
**Workaround Options**:
- Use Python 3.11 if TensorFlow <2.16 is required
- Install TensorFlow 2.16+ (for Python 3.12+): `pip install tensorflow>=2.16.0`
- Most features work without TensorFlow

### 2. Conflicting AI Packages (stardist vs n2v)
**Issue**: Dependency conflict between stardist and n2v over csbdeep version
- `stardist>=0.9.0` requires `csbdeep>=0.8.0`
- `n2v>=0.3.3` requires `csbdeep<0.8.0`

**Workaround**: Install separately based on your needs

#### Option A: Install StarDist (for star-convex segmentation)
```powershell
.\venv\Scripts\python.exe -m pip install stardist csbdeep>=0.8.0
```

#### Option B: Install N2V (for Noise2Void denoising)
```powershell
.\venv\Scripts\python.exe -m pip install n2v "csbdeep<0.8.0"
```

**Note**: You cannot have both stardist and n2v installed simultaneously in the same environment.

## Activating the Virtual Environment

### PowerShell (if execution policy allows)
```powershell
.\venv\Scripts\Activate.ps1
```

### PowerShell (alternative - bypass execution policy)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Direct Python Execution (no activation needed)
```powershell
.\venv\Scripts\python.exe your_script.py
.\venv\Scripts\streamlit.exe run app.py
```

## Running the Application

### Method 1: With activated venv
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

### Method 2: Direct execution (recommended for Windows)
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

## Recommendations for Future

1. **For Full Compatibility**: Consider using Python 3.11 instead of 3.12
   - Better support for older AI packages
   - TensorFlow 2.13-2.15 compatibility
   - More pre-built wheels available

2. **Create Separate Environments** if you need both stardist and n2v:
   ```powershell
   # Environment 1: with stardist
   python -m venv venv_stardist
   .\venv_stardist\Scripts\pip install -r requirements.txt
   .\venv_stardist\Scripts\pip install stardist
   
   # Environment 2: with n2v
   python -m venv venv_n2v
   .\venv_n2v\Scripts\pip install -r requirements.txt
   .\venv_n2v\Scripts\pip install n2v
   ```

3. **Keep Dependencies Updated**: The requirements.txt file now uses more conservative version constraints to maximize compatibility.

## Testing the Installation

Run this to verify everything works:
```powershell
.\venv\Scripts\python.exe -c "import numpy, scipy, pandas, matplotlib, streamlit, torch, cellpose; print('✓ All core packages working!')"
```

## Troubleshooting

### If you see "cannot be loaded because running scripts is disabled"
This is a PowerShell execution policy issue. Use the direct execution method:
```powershell
.\venv\Scripts\python.exe script.py
```

### If packages fail to import
Try reinstalling specific packages:
```powershell
.\venv\Scripts\python.exe -m pip install --force-reinstall <package-name>
```

### For CUDA/GPU Support
PyTorch was installed with CPU support. For GPU acceleration:
```powershell
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(Replace `cu121` with your CUDA version: cu118, cu121, etc.)
