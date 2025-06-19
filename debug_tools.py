#!/usr/bin/env python3
"""
Debug Tools for Advanced Image Biophysics
Utilities for troubleshooting and system diagnosis
"""

import sys
import os
import importlib
import platform
import subprocess
from pathlib import Path
import warnings

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    elif version >= (3, 8):
        print("✅ Python version compatible")
        return True

def check_core_dependencies():
    """Check if core dependencies are available"""
    core_deps = {
        'streamlit': '1.28.0',
        'numpy': '1.24.0', 
        'scipy': '1.10.0',
        'matplotlib': '3.7.0',
        'pandas': '2.0.0',
        'plotly': '5.15.0',
        'skimage': '0.20.0',
        'cv2': '4.8.0',
        'tifffile': '2023.7.10',
        'h5py': '3.9.0',
        'pims': '0.6.1',
        'multipletau': '0.3.3',
        'lmfit': '1.2.0',
        'trackpy': '0.6.1',
        'fcsfiles': '2022.9.28',
        'torch': '2.0.0',
        'torchvision': '0.15.0'
    }
    
    print("\n=== Core Dependencies ===")
    missing = []
    
    for package, min_version in core_deps.items():
        try:
            if package == 'skimage':
                import skimage
                version = skimage.__version__
            elif package == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {package}: {version}")
            
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            missing.append(package)
        except Exception as e:
            print(f"⚠️  {package}: ERROR - {e}")
    
    return missing

def check_optional_dependencies():
    """Check optional dependencies for extended functionality"""
    optional_deps = {
        'readlif': 'Leica LIF file support',
        'pylibczirw': 'Zeiss CZI file support', 
        'tensorflow': 'AI enhancement features',
        'cellpose': 'Cell segmentation',
        'stardist': 'Nucleus segmentation',
        'noise2void': 'Self-supervised denoising',
        'csbdeep': 'CARE restoration',
        'reportlab': 'PDF report generation',
        'jinja2': 'Template rendering',
        'markdown': 'Markdown processing'
    }
    
    print("\n=== Optional Dependencies ===")
    
    for package, description in optional_deps.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version} - {description}")
        except ImportError:
            print(f"⚠️  {package}: NOT INSTALLED - {description}")
        except Exception as e:
            print(f"❌ {package}: ERROR - {e}")

def check_file_structure():
    """Verify all required files are present"""
    required_files = [
        'app.py',
        'data_loader.py', 
        'analysis_modules.py',
        'ai_enhancement.py',
        'advanced_analysis.py',
        'optical_flow_analysis.py',
        'image_correlation_spectroscopy.py',
        'thumbnail_generator.py',
        'report_generator.py',
        'fcs_data_loader.py',
        'visualization.py',
        'utils.py'
    ]
    
    print("\n=== File Structure ===")
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file}: {size:,} bytes")
        else:
            print(f"❌ {file}: MISSING")
            missing_files.append(file)
    
    # Check config directory
    config_dir = Path('.streamlit')
    if config_dir.exists():
        print(f"✅ .streamlit/: Configuration directory exists")
        config_file = config_dir / 'config.toml'
        if config_file.exists():
            print(f"✅ .streamlit/config.toml: Configuration file exists")
        else:
            print(f"⚠️  .streamlit/config.toml: Configuration file missing")
    else:
        print(f"❌ .streamlit/: Configuration directory missing")
    
    return missing_files

def check_system_resources():
    """Check system resources and capabilities"""
    print("\n=== System Resources ===")
    
    # Platform info
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Memory info (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total Memory: {memory.total / (1024**3):.1f} GB")
        print(f"Available Memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("Memory info unavailable (psutil not installed)")
    
    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
    except ImportError:
        print("❌ PyTorch not available")

def test_streamlit_import():
    """Test if Streamlit can be imported and basic functionality works"""
    print("\n=== Streamlit Test ===")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit imported successfully: {st.__version__}")
        
        # Test basic functionality
        try:
            # This should not raise an error
            st.set_page_config(page_title="Test", layout="wide")
            print("✅ Streamlit page config test passed")
        except Exception as e:
            print(f"⚠️  Streamlit page config warning: {e}")
            
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Streamlit test warning: {e}")
    
    return True

def test_app_import():
    """Test if main app can be imported"""
    print("\n=== App Import Test ===")
    
    try:
        # Suppress warnings during import test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import app
        
        print("✅ Main app imported successfully")
        
        # Test key components
        components = ['DataLoader', 'AnalysisManager', 'AIEnhancementManager', 
                     'ThumbnailGenerator', 'ReportGenerator']
        
        for component in components:
            try:
                if hasattr(app, component):
                    print(f"✅ {component}: Available")
                else:
                    print(f"⚠️  {component}: Not found in app module")
            except Exception as e:
                print(f"❌ {component}: Error - {e}")
                
        return True
        
    except ImportError as e:
        print(f"❌ App import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  App import warning: {e}")
        return False

def generate_debug_report():
    """Generate comprehensive debug report"""
    print("=" * 60)
    print("ADVANCED IMAGE BIOPHYSICS - DEBUG REPORT")
    print("=" * 60)
    
    # Run all checks
    python_ok = check_python_version()
    missing_core = check_core_dependencies()
    check_optional_dependencies()
    missing_files = check_file_structure()
    check_system_resources()
    streamlit_ok = test_streamlit_import()
    app_ok = test_app_import()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if python_ok and not missing_core and not missing_files and streamlit_ok and app_ok:
        print("✅ ALL SYSTEMS READY - Application should run successfully")
        print("\nTo start the application:")
        print("  streamlit run app.py --server.port 5000")
    else:
        print("❌ ISSUES DETECTED:")
        
        if not python_ok:
            print("  - Python version incompatible")
        if missing_core:
            print(f"  - Missing core dependencies: {', '.join(missing_core)}")
        if missing_files:
            print(f"  - Missing files: {', '.join(missing_files)}")
        if not streamlit_ok:
            print("  - Streamlit not working")
        if not app_ok:
            print("  - App import failed")
            
        print("\nRecommended actions:")
        print("  1. Install missing dependencies: pip install <package-name>")
        print("  2. Verify all files are present")
        print("  3. Check Python version (3.8+ required)")

if __name__ == "__main__":
    generate_debug_report()