"""
Advanced Image Biophysics - Streamlit Application
Main interface for loading multi-format microscopy data and biophysical analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import io
import json
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import DataLoader, SUPPORTED_FORMATS
from fcs_data_loader import FCSDataLoader
from analysis_modules import AnalysisManager
from visualization import VisualizationManager
from ai_enhancement import AIEnhancementManager, get_enhancement_parameters
from advanced_analysis import AdvancedAnalysisManager, get_advanced_parameters
from optical_flow_analysis import OpticalFlowAnalyzer, get_optical_flow_parameters
from image_correlation_spectroscopy import ImageCorrelationSpectroscopy, get_ics_parameters
from nuclear_biophysics import NuclearBiophysicsAnalyzer, get_nuclear_analysis_parameters
from database_manager import DatabaseManager, get_database_manager
from thumbnail_generator import ThumbnailGenerator
from report_generator import AutomatedReportGenerator
from utils import format_file_size, validate_analysis_parameters

def normalize_image_for_display(image_data):
    """Normalize image data to [0, 1] range for Streamlit display"""
    import numpy as np
    if image_data is None:
        return None
    
    # Convert to numpy array if needed
    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)
    
    # Handle different data types
    if image_data.dtype == np.bool_:
        return image_data.astype(np.float32)
    
    # Normalize to [0, 1] range
    img_min = image_data.min()
    img_max = image_data.max()
    
    if img_max == img_min:
        return np.zeros_like(image_data, dtype=np.float32)
    
    return ((image_data - img_min) / (img_max - img_min)).astype(np.float32)

# Configure page
st.set_page_config(
    page_title="Advanced Image Biophysics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application interface"""
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    if 'fcs_loader' not in st.session_state:
        st.session_state.fcs_loader = FCSDataLoader()
    if 'analysis_manager' not in st.session_state:
        st.session_state.analysis_manager = AnalysisManager()
    if 'viz_manager' not in st.session_state:
        st.session_state.viz_manager = VisualizationManager()
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'data_type' not in st.session_state:
        st.session_state.data_type = 'image'
    if 'ai_enhancer' not in st.session_state:
        st.session_state.ai_enhancer = AIEnhancementManager()
    if 'enhanced_data' not in st.session_state:
        st.session_state.enhanced_data = None
    if 'advanced_analyzer' not in st.session_state:
        st.session_state.advanced_analyzer = AdvancedAnalysisManager()
    if 'advanced_results' not in st.session_state:
        st.session_state.advanced_results = None
    if 'optical_flow_analyzer' not in st.session_state:
        st.session_state.optical_flow_analyzer = OpticalFlowAnalyzer()
    if 'ics_analyzer' not in st.session_state:
        st.session_state.ics_analyzer = ImageCorrelationSpectroscopy()
    if 'nuclear_analyzer' not in st.session_state:
        st.session_state.nuclear_analyzer = NuclearBiophysicsAnalyzer()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = get_database_manager()
    if 'current_experiment_id' not in st.session_state:
        st.session_state.current_experiment_id = None
    if 'specialized_results' not in st.session_state:
        st.session_state.specialized_results = {}
    if 'thumbnail_generator' not in st.session_state:
        st.session_state.thumbnail_generator = ThumbnailGenerator()
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = AutomatedReportGenerator()
    if 'file_thumbnails' not in st.session_state:
        st.session_state.file_thumbnails = {}
    
    # Header
    st.title("🔬 Advanced Image Biophysics")
    st.markdown("*Multi-format microscopy data analysis with integrated biophysical techniques*")
    
    # Sidebar for file loading and analysis selection
    with st.sidebar:
        # Database Management Section
        if st.session_state.db_manager.database_available:
            st.header("📊 Database Management")
            
            # Experiment selection/creation
            experiments = st.session_state.db_manager.get_experiments()
            
            if experiments:
                experiment_options = {f"{exp['experiment_name']} ({exp['id']})": exp['id'] for exp in experiments}
                selected_experiment = st.selectbox(
                    "Select Experiment:",
                    options=["Create New..."] + list(experiment_options.keys()),
                    help="Choose an existing experiment or create a new one"
                )
                
                if selected_experiment != "Create New...":
                    st.session_state.current_experiment_id = experiment_options[selected_experiment]
                else:
                    st.session_state.current_experiment_id = None
            else:
                st.info("No experiments found. Create your first experiment below.")
                st.session_state.current_experiment_id = None
            
            # Create new experiment interface
            if st.session_state.current_experiment_id is None:
                with st.expander("🆕 Create New Experiment", expanded=True):
                    new_exp_name = st.text_input("Experiment Name", placeholder="e.g., Nuclear Dynamics Study")
                    new_exp_desc = st.text_area("Description", placeholder="Brief description of the experiment")
                    
                    if st.button("Create Experiment") and new_exp_name:
                        try:
                            exp_id = st.session_state.db_manager.create_experiment(
                                name=new_exp_name,
                                description=new_exp_desc,
                                metadata={"created_via": "streamlit_interface"}
                            )
                            st.session_state.current_experiment_id = exp_id
                            st.success(f"Created experiment: {new_exp_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create experiment: {str(e)}")
            
            # Current experiment info
            if st.session_state.current_experiment_id:
                current_exp = next((exp for exp in experiments if exp['id'] == st.session_state.current_experiment_id), None)
                if current_exp:
                    st.success(f"Active: {current_exp['experiment_name']}")
                    
                    # Show experiment files and results
                    files = st.session_state.db_manager.get_experiment_files(st.session_state.current_experiment_id)
                    if files:
                        st.caption(f"Files: {len(files)} | Last updated: {current_exp['updated_at'].strftime('%Y-%m-%d')}")
            
            st.divider()
        
        st.header("Data Loading")
        
        # Data type selection
        data_type = st.radio(
            "Select Data Type:",
            ["Microscopy Images", "FCS Data"],
            index=0 if st.session_state.data_type == 'image' else 1,
            help="Choose between imaging data (TIFF, CZI, LIF) or FCS correlation data (RAW, ASCII)"
        )
        
        # Update session state
        st.session_state.data_type = 'image' if data_type == "Microscopy Images" else 'fcs'
        
        # File upload interface - conditional based on data type
        if st.session_state.data_type == 'image':
            # Format support information with thumbnails
            with st.expander("📋 Supported Formats & Capabilities", expanded=False):
                st.subheader("Microscopy Format Support")
                
                # Get format support information
                format_support = st.session_state.thumbnail_generator.get_format_support_info()
                
                # Create format grid display
                cols = st.columns(3)
                format_items = list(format_support.items())
                
                for i, (ext, info) in enumerate(format_items):
                    col_idx = i % 3
                    with cols[col_idx]:
                        # Format info card
                        status_icon = "✅" if info['thumbnail_supported'] else "🔧"
                        st.markdown(f"**{status_icon} {info['name']}**")
                        st.caption(f"Extension: {ext}")
                        st.caption(f"Use: {info['typical_use']}")
                        
                        # Support status
                        if info['library_available']:
                            st.success(f"Status: {info['read_capability']}")
                        else:
                            st.warning("Limited support")
            
            # Upload mode selection
            upload_mode = st.radio(
                "Upload Mode",
                ["Single File", "TIFF Sequence"],
                help="Choose single file for stacks/multi-dimensional data, or TIFF sequence for multiple individual TIFF files"
            )
            
            if upload_mode == "Single File":
                uploaded_file = st.file_uploader(
                    "Upload microscopy data",
                    type=list(SUPPORTED_FORMATS.keys()),
                    help="Supported formats: MetaMorph, Zeiss LSM/Elyra, Leica SP8, Olympus"
                )
                uploaded_files = None
            else:
                uploaded_files = st.file_uploader(
                    "Upload TIFF sequence files",
                    type=['tif', 'tiff'],
                    accept_multiple_files=True,
                    help="Upload multiple TIFF files to create a time series. Files will be sorted by name."
                )
                uploaded_file = None
        else:
            # FCS data upload
            fcs_formats = st.session_state.fcs_loader.get_supported_formats()
            uploaded_file = st.file_uploader(
                "Upload FCS data",
                type=list(fcs_formats.keys()),
                help="Supported: Zeiss Confocor RAW/FCS, ASCII traces, CSV files"
            )
        
        if uploaded_file is not None or uploaded_files is not None:
            # Generate thumbnail preview for uploaded file(s)
            if st.session_state.data_type == 'image':
                st.subheader("📸 File Preview")
                
                if uploaded_file is not None:
                    # Single file preview
                    temp_path = f"temp_{uploaded_file.name}"
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Generate thumbnail
                        thumbnail_data = st.session_state.thumbnail_generator.generate_format_thumbnail(temp_path)
                        
                        # Display thumbnail and metadata
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if thumbnail_data.get('thumbnail') and thumbnail_data['status'] != 'error':
                                st.image(thumbnail_data['thumbnail'], caption="Format Preview", width=200)
                            else:
                                st.info(f"📄 {uploaded_file.name}\nFormat: {Path(uploaded_file.name).suffix.upper()}")
                        
                        with col2:
                            # Display metadata
                            st.markdown("**File Information:**")
                            
                            # Basic file info
                            file_size = len(uploaded_file.getbuffer())
                            st.write(f"**Filename:** {uploaded_file.name}")
                            st.write(f"**Size:** {format_file_size(file_size)}")
                            st.write(f"**Format:** {Path(uploaded_file.name).suffix.upper()}")
                            
                            # Format-specific metadata
                            if thumbnail_data['status'] == 'success' and 'metadata' in thumbnail_data:
                                metadata = thumbnail_data['metadata']
                                
                                if 'dimensions' in metadata:
                                    st.write(f"**Dimensions:** {metadata['dimensions']}")
                                if 'pixel_type' in metadata:
                                    st.write(f"**Pixel Type:** {metadata['pixel_type']}")
                                if 'estimated_channels' in metadata:
                                    st.write(f"**Channels:** {metadata['estimated_channels']}")
                                if 'estimated_timepoints' in metadata:
                                    timepoints = metadata['estimated_timepoints']
                                    if timepoints > 1:
                                        st.write(f"**Time Points:** {timepoints}")
                            
                            # Preview text
                            if 'preview_text' in thumbnail_data:
                                st.text(thumbnail_data['preview_text'])
                        
                        # Store thumbnail data for later use
                        st.session_state.file_thumbnails[uploaded_file.name] = thumbnail_data
                        
                    except Exception as e:
                        st.warning(f"Could not generate preview: {str(e)}")
                    finally:
                        # Clean up temp file
                        if Path(temp_path).exists():
                            Path(temp_path).unlink()
                
                elif uploaded_files is not None:
                    # TIFF sequence preview
                    st.markdown("**TIFF Sequence Information:**")
                    st.write(f"**Number of files:** {len(uploaded_files)}")
                    
                    # Display file list
                    total_size = sum(len(f.getbuffer()) for f in uploaded_files)
                    st.write(f"**Total size:** {format_file_size(total_size)}")
                    
                    # Show sorted file names
                    sorted_names = sorted([f.name for f in uploaded_files])
                    if len(sorted_names) > 0:
                        if len(sorted_names) == 1:
                            st.write(f"**File:** {sorted_names[0]}")
                        else:
                            st.write(f"**File range:** {sorted_names[0]} → {sorted_names[-1]}")
                    
                    # Preview first file thumbnail
                    if len(uploaded_files) > 0:
                        first_file = uploaded_files[0]
                        temp_path = f"temp_{first_file.name}"
                        try:
                            with open(temp_path, "wb") as f:
                                f.write(first_file.getbuffer())
                            
                            thumbnail_data = st.session_state.thumbnail_generator.generate_format_thumbnail(temp_path)
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if thumbnail_data.get('thumbnail') and thumbnail_data['status'] != 'error':
                                    st.image(thumbnail_data['thumbnail'], caption=f"Preview: {first_file.name}", width=200)
                            
                            with col2:
                                st.markdown("**Sample Frame Metadata:**")
                                if thumbnail_data['status'] == 'success' and 'metadata' in thumbnail_data:
                                    metadata = thumbnail_data['metadata']
                                    if 'dimensions' in metadata:
                                        st.write(f"**Frame dimensions:** {metadata['dimensions']}")
                                    if 'pixel_type' in metadata:
                                        st.write(f"**Pixel type:** {metadata['pixel_type']}")
                        except Exception as e:
                            st.warning(f"Could not preview first file: {str(e)}")
                        finally:
                            # Clean up temp file
                            if Path(temp_path).exists():
                                Path(temp_path).unlink()
            
            else:
                # Display basic file info for FCS data
                st.info(f"**File:** {uploaded_file.name}")
                st.info(f"**Size:** {format_file_size(len(uploaded_file.getvalue()))}")
            
            # Load data button
            if st.button("Load Data", type="primary"):
                loading_text = "Loading FCS data..." if st.session_state.data_type == 'fcs' else "Loading microscopy data..."
                with st.spinner(loading_text):
                    try:
                        if st.session_state.data_type == 'image':
                            # Check if we're loading single file or TIFF sequence
                            if uploaded_file is not None:
                                data_info = st.session_state.data_loader.load_file(uploaded_file)
                            elif uploaded_files is not None and len(uploaded_files) > 0:
                                data_info = st.session_state.data_loader.load_tiff_sequence(uploaded_files)
                            else:
                                raise ValueError("No files selected for loading")
                        else:
                            data_info = st.session_state.fcs_loader.load_fcs_file(uploaded_file)
                        st.session_state.loaded_data = data_info
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
        
        # AI Enhancement section (only show for image data)
        if st.session_state.loaded_data is not None and st.session_state.data_type == 'image':
            st.header("AI Enhancement")
            
            # Get available enhancement methods
            available_methods = st.session_state.ai_enhancer.get_available_methods()
            
            if available_methods:
                enhancement_method = st.selectbox(
                    "Enhancement Method",
                    options=available_methods,
                    help="AI-powered image enhancement techniques"
                )
                
                if st.button("🎨 Enhance Image", type="secondary"):
                    with st.spinner(f"Applying {enhancement_method}..."):
                        try:
                            # Get default parameters for the selected method
                            default_params = get_enhancement_parameters(enhancement_method)
                            
                            # Apply enhancement
                            image_data = st.session_state.loaded_data['image_data']
                            enhancement_result = st.session_state.ai_enhancer.enhance_image(
                                image_data, enhancement_method, default_params
                            )
                            
                            if enhancement_result.get('status') == 'success':
                                st.session_state.enhanced_data = enhancement_result
                                st.success(f"✅ {enhancement_method} applied successfully!")
                                st.rerun()
                            else:
                                st.error(f"❌ Enhancement failed: {enhancement_result.get('message', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"❌ Enhancement error: {str(e)}")
            else:
                st.info("No AI enhancement libraries available. Install PyTorch, Cellpose, or StarDist for enhanced functionality.")
        
        # Advanced Analysis section (show if data is loaded)
        if st.session_state.loaded_data is not None:
            st.header("Advanced AI Methods")
            
            # Get available advanced methods
            available_advanced = st.session_state.advanced_analyzer.get_available_methods()
            
            if available_advanced:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    advanced_method = st.selectbox(
                        "AI/Advanced Method",
                        options=available_advanced,
                        help="State-of-the-art AI and advanced analysis techniques"
                    )
                
                with col2:
                    if st.button("🚀 Apply Advanced Method", type="primary"):
                        with st.spinner(f"Applying {advanced_method}..."):
                            try:
                                # Get default parameters
                                default_params = get_advanced_parameters(advanced_method)
                                
                                # Apply advanced method
                                image_data = st.session_state.loaded_data['image_data']
                                
                                # Handle time series data for methods that need it
                                if advanced_method in ['Advanced SPT with trackpy', 'STICS Analysis', 'Nuclear Displacement Mapping']:
                                    if len(image_data.shape) < 3:
                                        st.error(f"{advanced_method} requires time series data (3D array)")
                                        st.stop()
                                
                                result = st.session_state.advanced_analyzer.apply_advanced_method(
                                    advanced_method, image_data, default_params
                                )
                                
                                if result.get('status') == 'success':
                                    st.session_state.advanced_results = result
                                    st.success(f"✅ {advanced_method} completed successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Advanced method failed: {result.get('message', 'Unknown error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Advanced analysis error: {str(e)}")
            else:
                st.info("Advanced AI methods require additional libraries. Current methods use available libraries.")
        
        # Specialized Physics Methods section
        if st.session_state.loaded_data is not None:
            st.header("Specialized Physics Methods")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌊 Optical Flow & Elastography")
                available_flow_methods = st.session_state.optical_flow_analyzer.get_available_methods()
                
                if available_flow_methods:
                    flow_method = st.selectbox(
                        "Optical Flow Method",
                        options=available_flow_methods,
                        help="Force propagation and elastography analysis"
                    )
                    
                    if st.button("🔄 Analyze Motion", type="secondary"):
                        with st.spinner(f"Analyzing {flow_method}..."):
                            try:
                                image_data = st.session_state.loaded_data['image_data']
                                
                                if len(image_data.shape) < 3:
                                    st.error("Optical flow requires time series data")
                                else:
                                    default_params = get_optical_flow_parameters(flow_method)
                                    result = st.session_state.optical_flow_analyzer.analyze_optical_flow(
                                        flow_method, image_data, default_params
                                    )
                                    
                                    if result.get('status') == 'success':
                                        st.session_state.specialized_results['optical_flow'] = result
                                        st.success(f"✅ {flow_method} completed!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Flow analysis failed: {result.get('message')}")
                            except Exception as e:
                                st.error(f"❌ Motion analysis error: {str(e)}")
                else:
                    st.info("Install OpenCV for advanced optical flow methods")
            
            with col2:
                st.subheader("📊 Image Correlation Spectroscopy")
                available_ics_methods = st.session_state.ics_analyzer.get_available_methods()
                
                ics_method = st.selectbox(
                    "ICS Method",
                    options=available_ics_methods,
                    help="RICS, STICS, imaging FCS and correlation analysis"
                )
                
                if st.button("🔬 Run ICS Analysis", type="secondary"):
                    with st.spinner(f"Running {ics_method}..."):
                        try:
                            image_data = st.session_state.loaded_data['image_data']
                            default_params = get_ics_parameters(ics_method)
                            
                            # Check data requirements
                            requires_time_series = ics_method in [
                                'RICS (Raster Image Correlation Spectroscopy)',
                                'STICS (Spatio-Temporal Image Correlation Spectroscopy)',
                                'Imaging FCS',
                                'iMSD via ICS',
                                'Temporal ICS'
                            ]
                            
                            if requires_time_series and len(image_data.shape) < 3:
                                st.error(f"{ics_method} requires time series data")
                            else:
                                result = st.session_state.ics_analyzer.analyze_ics_method(
                                    ics_method, image_data, default_params
                                )
                                
                                if result.get('status') == 'success':
                                    st.session_state.specialized_results['ics'] = result
                                    st.success(f"✅ {ics_method} completed!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ ICS analysis failed: {result.get('message')}")
                        except Exception as e:
                            st.error(f"❌ ICS analysis error: {str(e)}")
        
        # Analysis selection (only show if data is loaded)
        if st.session_state.loaded_data is not None:
            st.header("Standard Analysis Tools")
            
            # Filter analysis options based on data type
            if st.session_state.data_type == 'image':
                analysis_options = [
                    "RICS (Raster Image Correlation Spectroscopy)",
                    "FCS/sFCS/FCCS (Fluorescence Correlation Spectroscopy)",
                    "iMSD (Image Mean Square Displacement)",
                    "Elastography & PIV (Particle Image Velocimetry)",
                    "N&B (Number and Brightness)",
                    "Nuclear Binding Dynamics",
                    "Chromatin Dynamics Analysis",
                    "Nuclear Elasticity Mapping",
                    "FLIM (Fluorescence Lifetime Imaging)",
                    "SPT (Single Particle Tracking)",
                    "Fourier Transform Texture Analysis",
                    "FRAP (Fluorescence Recovery After Photobleaching)"
                ]
            else:  # FCS data
                analysis_options = [
                    "FCS/sFCS/FCCS (Fluorescence Correlation Spectroscopy)",
                    "Advanced FCS Model Fitting",
                    "Cross-Correlation Analysis (FCCS)",
                    "Photon Counting Histogram (PCH)",
                    "Temporal Analysis"
                ]
            
            selected_analysis = st.selectbox(
                "Select Analysis Method",
                options=analysis_options,
                help="Choose the biophysical analysis technique to apply"
            )
            
            # Analysis-specific parameters
            st.subheader("Parameters")
            params = render_analysis_parameters(selected_analysis)
            
            # Channel selection for analysis if multichannel data
            if st.session_state.loaded_data.get('channels', 1) > 1:
                st.subheader("Channel Selection")
                channel_names = st.session_state.loaded_data.get('channel_names', [f"Channel {i+1}" for i in range(st.session_state.loaded_data['channels'])])
                
                analysis_channels = st.multiselect(
                    "Select channels for analysis",
                    options=channel_names,
                    default=[channel_names[0]] if channel_names else [],
                    help="Choose which channels to include in the analysis"
                )
                
                params['analysis_channels'] = [channel_names.index(ch) for ch in analysis_channels if ch in channel_names]
                params['channel_names'] = analysis_channels
            
            # Run analysis button
            if st.button("Run Analysis", type="primary"):
                run_analysis(selected_analysis, params)
    
    # Main content area
    if st.session_state.loaded_data is None:
        # Welcome screen
        render_welcome_screen()
    else:
        # Data overview and analysis interface
        render_data_interface()

def render_welcome_screen():
    """Render the welcome screen with supported formats and features"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to Advanced Image Biophysics")
        
        st.markdown("""
        This application provides a unified interface for analyzing microscopy data 
        using various biophysical techniques. Upload your data to get started.
        """)
        
        st.subheader("Supported Analysis Techniques")
        
        techniques = {
            "🔄 RICS": "Raster Image Correlation Spectroscopy - Measures diffusion rates and molecular flow",
            "📊 FCS/sFCS/FCCS": "Fluorescence Correlation Spectroscopy - Quantifies diffusion and interactions",
            "📈 iMSD": "Image Mean Square Displacement - Extracts diffusion and transport behavior",
            "🔧 Elastography & PIV": "Estimates viscoelastic properties and motion mapping",
            "🔢 N&B Analysis": "Number and Brightness - Determines molecular aggregation states",
            "⏱️ FLIM": "Fluorescence Lifetime Imaging - Maps molecular interactions",
            "🎯 SPT": "Single Particle Tracking - Tracks individual particle motion",
            "🌊 Fourier Analysis": "Texture analysis for chromatin organization",
            "💫 FRAP": "Fluorescence Recovery After Photobleaching - Exchange rate analysis"
        }
        
        for technique, description in techniques.items():
            st.markdown(f"**{technique}**: {description}")
    
    with col2:
        st.subheader("Supported Formats")
        
        formats = [
            "MetaMorph (.tif, .stk)",
            "Zeiss LSM 700 (.lsm, .czi)",
            "Zeiss Elyra 7 (.czi)",
            "Leica SP8 (.lif, .tif)",
            "Olympus Spinning Disk (.tif, .oif)"
        ]
        
        for fmt in formats:
            st.markdown(f"✅ {fmt}")
        
        st.info("💡 **Tip**: All analysis results use standardized visualization methods for cross-compatibility")

def render_data_interface():
    """Render the main data analysis interface"""
    
    data_info = st.session_state.loaded_data
    
    # Data overview tabs - add AI Enhancement tab for image data
    if st.session_state.data_type == 'image':
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data Overview", "🖼️ Image Preview", "🎨 AI Enhancement", "📊 Analysis Results", "💾 Export"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 FCS Preview", "📊 Analysis Results", "💾 Export"])
    
    with tab1:
        render_data_overview(data_info)
    
    with tab2:
        render_image_preview(data_info)
    
    if st.session_state.data_type == 'image':
        with tab3:
            # AI Enhancement Section
            st.subheader("🎨 AI-Powered Image Enhancement")
            
            # Get available enhancement methods
            available_methods = st.session_state.ai_enhancer.get_available_methods()
            
            if available_methods:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    enhancement_method = st.selectbox(
                        "Enhancement Method",
                        options=available_methods,
                        help="Select AI enhancement technique"
                    )
                    
                    # Simple parameter controls
                    st.write("**Parameters:**")
                    
                    if enhancement_method == 'Non-local Means Denoising':
                        patch_size = st.slider("Patch Size", 3, 15, 5)
                        patch_distance = st.slider("Patch Distance", 3, 15, 6)
                        parameters = {'patch_size': patch_size, 'patch_distance': patch_distance, 'fast_mode': True}
                    
                    elif enhancement_method == 'Richardson-Lucy Deconvolution':
                        iterations = st.slider("Iterations", 10, 100, 30)
                        parameters = {'iterations': iterations, 'psf_size': 5, 'psf_sigma': 1.0}
                    
                    else:
                        parameters = {}
                    
                    # Apply enhancement button
                    if st.button("🚀 Apply Enhancement", type="primary", key="enhance_btn"):
                        with st.spinner(f"Applying {enhancement_method}..."):
                            try:
                                image_data = st.session_state.loaded_data['image_data']
                                
                                # Handle time series data - use first frame
                                if len(image_data.shape) > 2:
                                    if len(image_data.shape) == 3:
                                        enhancement_data = image_data[0] if image_data.shape[0] < 100 else image_data
                                    else:
                                        enhancement_data = image_data
                                else:
                                    enhancement_data = image_data
                                
                                # Apply enhancement
                                enhancement_result = st.session_state.ai_enhancer.enhance_image(
                                    enhancement_data, enhancement_method, parameters
                                )
                                
                                if enhancement_result.get('status') == 'success':
                                    st.session_state.enhanced_data = enhancement_result
                                    st.success(f"✅ {enhancement_method} completed successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Enhancement failed: {enhancement_result.get('message', 'Unknown error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Enhancement error: {str(e)}")
                
                with col2:
                    # Display enhancement results
                    if st.session_state.enhanced_data is not None:
                        st.write("**Enhancement Results:**")
                        enhancement_data = st.session_state.enhanced_data
                        
                        if 'enhanced_image' in enhancement_data:
                            normalized_img = normalize_image_for_display(enhancement_data['enhanced_image'])
                            st.image(normalized_img, caption="Enhanced Image", use_container_width=True)
                        elif 'segmentation_masks' in enhancement_data:
                            normalized_masks = normalize_image_for_display(enhancement_data['segmentation_masks'])
                            st.image(normalized_masks, caption="Segmentation", use_container_width=True)
                        
                        if 'num_objects' in enhancement_data:
                            st.metric("Objects Detected", enhancement_data['num_objects'])
                        
                        # Option to use enhanced image for analysis
                        if st.button("✅ Use Enhanced for Analysis", key="use_enhanced_btn"):
                            if 'enhanced_image' in enhancement_data:
                                st.session_state.loaded_data['image_data'] = enhancement_data['enhanced_image']
                                st.success("Enhanced image set as active data")
                                st.rerun()
                    else:
                        st.info("Apply an enhancement method to see results")
            else:
                st.warning("No AI enhancement libraries available. Install PyTorch, Cellpose, or StarDist for enhanced functionality.")
            
            # Analysis Section
            st.divider()
            st.subheader("📊 Standard Analysis Methods")
            
            analysis_methods = st.session_state.analysis_manager.get_available_analyses()
            
            if analysis_methods:
                selected_analysis = st.selectbox(
                    "Select Analysis Method",
                    options=analysis_methods,
                    help="Choose biophysical analysis technique"
                )
                
                # Quick analysis button
                if st.button("🔬 Run Analysis", type="primary", key="run_analysis_btn"):
                    with st.spinner(f"Running {selected_analysis} analysis..."):
                        try:
                            # Use default parameters for quick analysis
                            default_params = {
                                'region_size': 50,
                                'max_tau': 100,
                                'binning': 1,
                                'threshold': 0.1
                            }
                            
                            data_info = st.session_state.loaded_data
                            result = st.session_state.analysis_manager.run_analysis(
                                selected_analysis, data_info, default_params
                            )
                            
                            if result.get('status') == 'success':
                                st.session_state.analysis_results[selected_analysis] = result
                                st.success(f"✅ {selected_analysis} analysis completed!")
                                st.rerun()
                            else:
                                st.error(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"❌ Analysis error: {str(e)}")
        
        with tab4:
            render_analysis_results()
            # Display specialized physics results if available
            if st.session_state.specialized_results:
                render_specialized_results()
        
        with tab5:
            render_export_options()
            
            # Add automated report generation
            if st.session_state.loaded_data is not None:
                st.divider()
                render_automated_reports()
    else:
        with tab3:
            render_analysis_results()
            # Display specialized physics results if available
            if st.session_state.specialized_results:
                render_specialized_results()
        
        with tab4:
            render_export_options()

def render_data_overview(data_info):
    """Render data overview information"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        
        info_data = {
            "Filename": data_info.get('filename', 'Unknown'),
            "Format": data_info.get('format', 'Unknown'),
            "Dimensions": f"{data_info.get('shape', 'Unknown')}",
            "Data Type": data_info.get('dtype', 'Unknown'),
            "Pixel Size": f"{data_info.get('pixel_size', 'Unknown')} µm",
            "Time Points": data_info.get('time_points', 'Unknown'),
            "Channels": data_info.get('channels', 'Unknown')
        }
        
        # Add channel names if available
        if data_info.get('channels', 1) > 1:
            channel_names = data_info.get('channel_names', [])
            if channel_names:
                info_data["Channel Names"] = ", ".join(channel_names)
        
        for key, value in info_data.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("Acquisition Parameters")
        
        acq_params = data_info.get('acquisition_params', {})
        if acq_params:
            for param, value in acq_params.items():
                st.text(f"{param}: {value}")
        else:
            st.info("No acquisition parameters available")

def render_image_preview(data_info):
    """Render image preview with basic visualization"""
    
    st.subheader("Image Preview")
    
    if 'image_data' in data_info:
        image_data = data_info['image_data']
        
        # Preview controls - Enhanced with Z-slice navigation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Time point navigation for time series data
            if len(image_data.shape) > 3:  # Time series (T,Z,Y,X,C or T,Y,X,C)
                time_point = st.slider("Time Point", 0, image_data.shape[0]-1, 0)
            else:
                time_point = 0
        
        with col2:
            # Z-slice navigation - determine if Z dimension exists
            z_slice = 0
            if len(image_data.shape) == 5:  # T,Z,Y,X,C
                z_slice = st.slider("Z-Slice", 0, image_data.shape[1]-1, 0)
            elif len(image_data.shape) == 4:
                # Check if this is Z,Y,X,C or T,Y,X,C based on metadata
                if data_info.get('z_slices', 1) > 1 and not data_info.get('time_points', 0) > 1:
                    z_slice = st.slider("Z-Slice", 0, image_data.shape[0]-1, 0)
                    time_point = 0  # Override time point for Z-stack
            elif len(image_data.shape) == 3:
                # Check if this is Z,Y,X based on metadata
                if data_info.get('z_slices', 1) > 1 and data_info.get('channels', 1) == 1:
                    z_slice = st.slider("Z-Slice", 0, image_data.shape[0]-1, 0)
        
        with col3:
            # Enhanced channel selection for multichannel data
            if data_info.get('channels', 1) > 1:
                channel_names = data_info.get('channel_names', [f"Channel {i+1}" for i in range(data_info['channels'])])
                
                # Create channel selection options
                if len(channel_names) <= 10:  # Show as selectbox for up to 10 channels
                    channel_options = {name: i for i, name in enumerate(channel_names)}
                    selected_channel_name = st.selectbox("Channel", list(channel_options.keys()))
                    channel = channel_options[selected_channel_name]
                else:  # Use slider for more than 10 channels
                    channel = st.slider("Channel", 0, len(channel_names)-1, 0)
                    st.caption(f"Selected: {channel_names[channel]}")
            else:
                channel = 0
        
        with col4:
            contrast_mode = st.selectbox("Contrast", ["Auto", "Manual", "Percentile"])
        
        # Add multichannel composite option
        if data_info.get('channels', 1) > 1:
            show_composite = st.checkbox("Show Multichannel Composite", value=False)
        else:
            show_composite = False
        
        # Display image with proper multichannel and Z-slice handling
        try:
            # Enhanced logic for extracting correct image slice with Z-slice support
            if len(image_data.shape) == 5:  # T, Z, Y, X, C
                if channel < image_data.shape[-1]:
                    display_image = image_data[time_point, z_slice, :, :, channel]
                else:
                    display_image = image_data[time_point, z_slice, :, :, 0]
            elif len(image_data.shape) == 4:
                # Check if this is Z,Y,X,C or T,Y,X,C based on metadata
                if data_info.get('z_slices', 1) > 1 and not data_info.get('time_points', 0) > 1:
                    # Z, Y, X, C format
                    if channel < image_data.shape[-1]:
                        display_image = image_data[z_slice, :, :, channel]
                    else:
                        display_image = image_data[z_slice, :, :, 0]
                else:
                    # T, Y, X, C format
                    if channel < image_data.shape[-1]:
                        display_image = image_data[time_point, :, :, channel]
                    else:
                        display_image = image_data[time_point, :, :, 0]
            elif len(image_data.shape) == 3:
                # Use data_info to determine if it's multichannel, time series, or Z-stack
                if data_info.get('z_slices', 1) > 1 and data_info.get('channels', 1) == 1:
                    # Z, Y, X format
                    display_image = image_data[z_slice, :, :]
                elif data_info.get('channels', 1) > 1 and image_data.shape[-1] == data_info['channels']:
                    # Y, X, C format
                    if channel < image_data.shape[-1]:
                        display_image = image_data[:, :, channel]
                    else:
                        display_image = image_data[:, :, 0]
                else:
                    # T, Y, X format or ambiguous - use time_point
                    display_image = image_data[time_point, :, :]
            else:  # 2D
                display_image = image_data
            
            if show_composite and data_info.get('channels', 1) > 1:
                # Display multichannel composite
                channel_data_list = []
                channel_names = data_info.get('channel_names', [f"Channel {i+1}" for i in range(data_info['channels'])])
                channel_colors = data_info.get('channel_colors', ['#FF0000', '#00FF00', '#0000FF'])
                
                # Extract all channels for current time point
                for ch in range(data_info['channels']):
                    if len(image_data.shape) == 4:  # T, Y, X, C
                        if ch < image_data.shape[-1]:
                            ch_data = image_data[time_point, :, :, ch]
                        else:
                            continue
                    elif len(image_data.shape) == 3 and image_data.shape[-1] == data_info['channels']:
                        # Y, X, C format
                        if ch < image_data.shape[-1]:
                            ch_data = image_data[:, :, ch]
                        else:
                            continue
                    else:
                        # Single channel data
                        ch_data = image_data[time_point, :, :] if len(image_data.shape) == 3 else image_data
                    
                    channel_data_list.append(ch_data)
                
                # Display composite
                st.session_state.viz_manager.display_multichannel_composite(
                    channel_data_list[:len(channel_names)],
                    channel_names[:len(channel_data_list)],
                    channel_colors[:len(channel_data_list)],
                    title=f"Multichannel Composite - Frame {time_point}"
                )
            else:
                # Display single channel
                channel_name = "Single Channel"
                channel_color = None
                
                if data_info.get('channels', 1) > 1:
                    channel_names = data_info.get('channel_names', [f"Channel {i+1}" for i in range(data_info['channels'])])
                    channel_colors = data_info.get('channel_colors', ['#FF0000', '#00FF00', '#0000FF'])
                    
                    if channel < len(channel_names):
                        channel_name = channel_names[channel]
                        if channel < len(channel_colors):
                            channel_color = channel_colors[channel]
                    else:
                        channel_name = f"Channel {channel + 1}"
                
                title = f"Frame {time_point}, {channel_name}" if data_info.get('time_points', 1) > 1 else channel_name
                
                st.session_state.viz_manager.display_image(
                    display_image, 
                    title=title,
                    contrast_mode=contrast_mode,
                    channel_color=channel_color
                )
            
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    else:
        st.warning("No image data available for preview")

def render_analysis_results():
    """Render analysis results with shared visualization"""
    
    # Check for both standard and advanced analysis results
    has_standard_results = bool(st.session_state.analysis_results)
    has_advanced_results = st.session_state.advanced_results is not None
    
    if not has_standard_results and not has_advanced_results:
        st.info("No analysis results available. Run an analysis from the sidebar to see results here.")
        return
    
    st.subheader("Analysis Results")
    
    # Display advanced AI method results
    if has_advanced_results:
        advanced_results = st.session_state.advanced_results
        method_name = advanced_results.get('method', 'Advanced Analysis')
        
        with st.expander(f"🤖 {method_name}", expanded=True):
            
            # Method-specific result display
            if 'enhanced_image' in advanced_results:
                # Enhancement methods
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image**")
                    if 'original_image' in advanced_results:
                        st.image(advanced_results['original_image'], use_container_width=True)
                
                with col2:
                    st.write("**Enhanced Image**")
                    st.image(advanced_results['enhanced_image'], use_container_width=True)
                    
                # Display enhancement statistics
                if 'parameters_used' in advanced_results:
                    st.write("**Parameters Used:**")
                    for param, value in advanced_results['parameters_used'].items():
                        st.text(f"{param}: {value}")
            
            elif 'segmentation_masks' in advanced_results:
                # Segmentation methods
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image**")
                    if 'original_image' in advanced_results:
                        st.image(advanced_results['original_image'], use_container_width=True)
                
                with col2:
                    st.write("**Segmentation Results**")
                    if 'colored_overlay' in advanced_results:
                        st.image(advanced_results['colored_overlay'], use_container_width=True)
                    else:
                        st.image(advanced_results['segmentation_masks'], use_container_width=True)
                
                # Display segmentation statistics
                if 'num_objects' in advanced_results:
                    st.metric("Objects Detected", advanced_results['num_objects'])
                elif 'num_nuclei' in advanced_results:
                    st.metric("Nuclei Detected", advanced_results['num_nuclei'])
            
            elif 'trajectories' in advanced_results:
                # SPT and tracking methods
                trajectories = advanced_results['trajectories']
                
                st.write(f"**Tracking Results**: {advanced_results.get('num_particles', 0)} particles tracked")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display trajectory statistics
                    if 'avg_trajectory_length' in advanced_results:
                        st.metric("Average Trajectory Length", f"{advanced_results['avg_trajectory_length']:.1f} frames")
                    
                    # Show sample trajectories data
                    if not trajectories.empty:
                        st.write("**Sample Trajectory Data**")
                        st.dataframe(trajectories.head(10))
                
                with col2:
                    # Display MSD results if available
                    if 'ensemble_msd' in advanced_results:
                        msd_data = advanced_results['ensemble_msd']
                        
                        # Create MSD plot
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        
                        lag_times = msd_data.index
                        fig.add_trace(go.Scatter(
                            x=lag_times,
                            y=msd_data.values,
                            mode='lines+markers',
                            name='Ensemble MSD',
                            line=dict(width=2)
                        ))
                        
                        fig.update_layout(
                            title="Mean Square Displacement",
                            xaxis_title="Lag Time",
                            yaxis_title="MSD",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            elif 'displacement_vectors' in advanced_results:
                # Nuclear displacement mapping
                displacement_df = advanced_results['displacement_vectors']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Displacement Statistics**")
                    if 'avg_displacement' in advanced_results:
                        st.metric("Average Displacement", f"{advanced_results['avg_displacement']:.2f} pixels")
                    if 'num_tracked_nuclei' in advanced_results:
                        st.metric("Tracked Nuclei", advanced_results['num_tracked_nuclei'])
                    
                    # Show displacement data sample
                    if not displacement_df.empty:
                        st.write("**Sample Displacement Data**")
                        st.dataframe(displacement_df.head(10))
                
                with col2:
                    # Create displacement magnitude histogram
                    if not displacement_df.empty:
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=displacement_df['displacement_magnitude'],
                            nbinsx=20,
                            name='Displacement Distribution'
                        ))
                        
                        fig.update_layout(
                            title="Displacement Magnitude Distribution",
                            xaxis_title="Displacement (pixels)",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            elif 'correlation_function' in advanced_results:
                # STICS analysis
                stics_data = advanced_results['correlation_function']
                
                st.write("**STICS Correlation Analysis**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'zero_lag_correlation' in advanced_results:
                        st.metric("Zero-lag Correlation", f"{advanced_results['zero_lag_correlation']:.4f}")
                    
                    st.write(f"Spatial lag range: ±{advanced_results.get('max_spatial_lag', 'N/A')}")
                    st.write(f"Temporal lag range: {advanced_results.get('max_temporal_lag', 'N/A')}")
                
                with col2:
                    # Display correlation function visualization
                    st.write("**Correlation Function Shape**")
                    st.write(f"Dimensions: {stics_data.shape}")
                    
                    # Show zero temporal lag spatial correlation
                    if len(stics_data.shape) == 3 and stics_data.shape[0] > 0:
                        spatial_corr = stics_data[0]  # tau=0
                        # Normalize image data for display
                        spatial_corr_norm = normalize_image_for_display(spatial_corr)
                        st.image(spatial_corr_norm, caption="Spatial correlation at τ=0", use_container_width=True)
            
            # Display common information for all methods
            if 'parameters_used' in advanced_results and method_name not in ['Enhanced Richardson-Lucy', 'Noise2Void (Enhanced NLM)']:
                st.write("**Method Parameters:**")
                for param, value in advanced_results['parameters_used'].items():
                    st.text(f"{param}: {value}")
    
    # Display standard analysis results
    if has_standard_results:
        st.subheader("Standard Analysis Results")
        
        for analysis_name, results in st.session_state.analysis_results.items():
            with st.expander(f"📊 {analysis_name}", expanded=True):
                # Use visualization manager for consistent result display
                st.session_state.viz_manager.display_analysis_results(analysis_name, results)

def render_export_options():
    """Render data and results export options"""
    
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Export Analysis Results**")
        
        if st.session_state.analysis_results:
            export_format = st.selectbox("Result Format", ["CSV", "Excel", "JSON", "HDF5"])
            
            if st.button("Export Results"):
                try:
                    exported_data = export_analysis_results(export_format)
                    st.download_button(
                        label=f"Download {export_format}",
                        data=exported_data,
                        file_name=f"analysis_results.{export_format.lower()}",
                        mime=get_mime_type(export_format)
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        else:
            st.info("No results to export")
    
    with col2:
        st.markdown("**🖼️ Export Visualizations**")
        
        if st.session_state.analysis_results:
            viz_format = st.selectbox("Image Format", ["PNG", "SVG", "PDF"])
            
            if st.button("Export Plots"):
                try:
                    exported_plots = export_visualizations(viz_format)
                    st.success("Visualizations exported successfully!")
                except Exception as e:
                    st.error(f"Visualization export failed: {str(e)}")
        else:
            st.info("No visualizations to export")

def render_analysis_parameters(analysis_type):
    """Render analysis-specific parameter inputs"""
    
    params = {}
    
    if "RICS" in analysis_type:
        params['tau_max'] = st.slider("Maximum lag time (τ_max)", 1, 100, 20)
        params['pixel_size'] = st.number_input("Pixel size (µm)", value=0.1, step=0.01)
        params['time_interval'] = st.number_input("Time interval (s)", value=0.1, step=0.01)
        
    elif "Segmented FCS" in analysis_type:
        params['segmentation_type'] = st.selectbox("Segmentation Type", ['x', 'y'], 
                                                   help="Segment along x-axis (pixels) or y-axis (lines)")
        params['segment_length'] = st.number_input("Segment Length", min_value=32, max_value=8192, value=128,
                                                   help="Length of each segment in pixels or lines")
        params['model_type'] = st.selectbox("FCS Model", ['2d', '3d', 'anomalous'],
                                           help="Choose FCS fitting model")
        params['max_lag_fraction'] = st.slider("Max Lag Fraction", 0.1, 0.5, 0.25,
                                              help="Maximum lag as fraction of segment length")
        
        # Timing parameters
        st.subheader("Timing Parameters")
        params['pixel_time'] = st.number_input("Pixel Time (s)", value=3.05e-6, format="%.2e",
                                               help="Time per pixel acquisition")
        params['line_time'] = st.number_input("Line Time (s)", value=0.56e-3, format="%.2e",
                                              help="Time per line acquisition")
        params['pixel_size'] = st.number_input("Pixel Size (µm)", value=0.05, step=0.001,
                                               help="Physical pixel size")
        
    elif "FCS" in analysis_type and "Segmented" not in analysis_type:
        params['bleach_correction'] = st.checkbox("Apply bleach correction", value=True)
        params['binning'] = st.slider("Temporal binning", 1, 10, 1)
        params['correlation_window'] = st.slider("Correlation window size", 8, 64, 16)
        
    elif "iMSD" in analysis_type:
        params['max_displacement'] = st.slider("Maximum displacement (pixels)", 5, 50, 20)
        params['min_track_length'] = st.slider("Minimum track length", 3, 20, 5)
        params['localization_error'] = st.number_input("Localization error (nm)", value=30.0, step=1.0)
        
    elif "Elastography" in analysis_type or "PIV" in analysis_type:
        params['window_size'] = st.slider("Analysis window size", 8, 64, 16)
        params['overlap_ratio'] = st.slider("Window overlap", 0.0, 0.8, 0.5)
        params['force_threshold'] = st.number_input("Force threshold", value=1.0, step=0.1)
        
    elif "N&B" in analysis_type:
        params['background_subtraction'] = st.checkbox("Subtract background", value=True)
        params['brightness_threshold'] = st.slider("Brightness threshold", 0.1, 10.0, 1.0)
        params['aggregation_cutoff'] = st.number_input("Aggregation cutoff", value=2.0, step=0.1)
        
    elif "FLIM" in analysis_type:
        params['lifetime_range'] = st.slider("Lifetime range (ns)", 0.1, 10.0, (0.5, 5.0))
        params['fitting_model'] = st.selectbox("Fitting model", ["Single exponential", "Double exponential", "Stretched exponential"])
        params['chi_squared_threshold'] = st.number_input("χ² threshold", value=2.0, step=0.1)
        
    elif "SPT" in analysis_type:
        params['detection_threshold'] = st.slider("Detection threshold", 1, 20, 5)
        params['linking_distance'] = st.slider("Linking distance (pixels)", 1, 10, 3)
        params['gap_closing'] = st.slider("Gap closing frames", 0, 5, 1)
        
    elif "Fourier" in analysis_type:
        params['frequency_cutoff'] = st.slider("Frequency cutoff", 0.1, 2.0, 1.0)
        params['window_function'] = st.selectbox("Window function", ["Hanning", "Hamming", "Blackman"])
        params['detrend'] = st.checkbox("Apply detrending", value=True)
        
    elif "FRAP" in analysis_type:
        params['bleach_roi_size'] = st.slider("Bleach ROI size", 5, 50, 20)
        params['normalization_method'] = st.selectbox("Normalization", ["Double normalization", "Full scale", "Background"])
        params['fitting_model'] = st.selectbox("Recovery model", ["Single exponential", "Double exponential", "Anomalous diffusion"])
    
    elif "Nuclear Binding" in analysis_type:
        params['two_component_model'] = st.checkbox("Use two-component FCS model", value=True)
        params['correlation_window_size'] = st.slider("Correlation window size (pixels)", 3, 15, 5)
        params['max_lag_time'] = st.slider("Maximum lag time (seconds)", 1.0, 20.0, 10.0)
        
    elif "Chromatin Dynamics" in analysis_type:
        params['msd_max_lag'] = st.slider("MSD maximum lag (frames)", 5, 50, 20)
        params['texture_analysis'] = st.checkbox("Include texture analysis", value=True)
        params['nb_analysis'] = st.checkbox("Include N&B analysis", value=True)
        
    elif "Nuclear Elasticity" in analysis_type:
        params['applied_force'] = st.number_input("Applied force (pN)", value=1.0, step=0.1)
        params['nuclear_area'] = st.number_input("Nuclear area (μm²)", value=100.0, step=1.0)
        params['force_application_frame'] = st.slider("Force application frame", 1, 50, 10)
        params['tracking_search_range'] = st.slider("Tracking search range (pixels)", 10, 100, 50)
    
    return params

def run_analysis(analysis_type, parameters):
    """Execute the selected analysis with given parameters"""
    
    import time
    start_time = time.time()
    
    with st.spinner(f"Running {analysis_type} analysis..."):
        try:
            # Validate parameters
            if not validate_analysis_parameters(analysis_type, parameters):
                st.error("Invalid parameters. Please check your inputs.")
                return
            
            # Run analysis using the analysis manager
            results = st.session_state.analysis_manager.run_analysis(
                analysis_type, 
                st.session_state.loaded_data, 
                parameters
            )
            
            execution_time = time.time() - start_time
            
            # Store results
            st.session_state.analysis_results[analysis_type] = results
            
            # Store in database if available
            if (st.session_state.db_manager.database_available and 
                st.session_state.current_experiment_id and
                st.session_state.loaded_data.get('database_file_id')):
                
                try:
                    analysis_id = st.session_state.db_manager.store_analysis_results(
                        file_id=st.session_state.loaded_data['database_file_id'],
                        experiment_id=st.session_state.current_experiment_id,
                        analysis_type=analysis_type,
                        results=results,
                        parameters=parameters,
                        execution_time=execution_time
                    )
                    st.success(f"✅ {analysis_type} analysis completed successfully!")
                    st.info("📊 Results stored in database")
                    
                except Exception as db_error:
                    st.warning(f"Analysis completed but database storage failed: {str(db_error)}")
                    st.success(f"✅ {analysis_type} analysis completed successfully!")
            else:
                st.success(f"✅ {analysis_type} analysis completed successfully!")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

def export_analysis_results(format_type):
    """Export analysis results in specified format"""
    
    results = st.session_state.analysis_results
    
    if format_type == "CSV":
        # Convert results to CSV format
        output = io.StringIO()
        # Implementation would depend on specific result structure
        return output.getvalue().encode()
    
    elif format_type == "JSON":
        import json
        return json.dumps(results, indent=2, default=str).encode()
    
    # Add other format implementations
    return b"Export not implemented for this format"

def export_visualizations(format_type):
    """Export current visualizations"""
    # Implementation would use the visualization manager
    return st.session_state.viz_manager.export_plots(format_type)

def get_mime_type(format_type):
    """Get MIME type for file format"""
    mime_types = {
        "CSV": "text/csv",
        "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "JSON": "application/json",
        "HDF5": "application/x-hdf",
        "PNG": "image/png",
        "SVG": "image/svg+xml",
        "PDF": "application/pdf"
    }
    return mime_types.get(format_type, "application/octet-stream")

def render_ai_enhancement_interface():
    """Render AI enhancement interface with interactive controls"""
    
    st.subheader("AI-Powered Image Enhancement")
    
    # Get available enhancement methods
    available_methods = st.session_state.ai_enhancer.get_available_methods()
    
    if not available_methods:
        st.warning("No AI enhancement libraries detected. Install PyTorch, Cellpose, or StarDist for enhanced functionality.")
        return
    
    # Enhancement method selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        enhancement_method = st.selectbox(
            "Enhancement Method",
            options=available_methods,
            help="Select AI enhancement technique"
        )
        
        # Method-specific parameter controls
        st.subheader("Parameters")
        
        parameters = {}
        
        if enhancement_method == 'Non-local Means Denoising':
            parameters['patch_size'] = st.slider("Patch Size", 3, 15, 5, help="Size of patches for comparison")
            parameters['patch_distance'] = st.slider("Patch Distance", 3, 15, 6, help="Search distance for patches")
            parameters['fast_mode'] = st.checkbox("Fast Mode", True, help="Use approximation for speed")
            parameters['auto_sigma'] = st.checkbox("Auto Estimate Noise", True, help="Automatically estimate noise level")
            if not parameters['auto_sigma']:
                parameters['h'] = st.slider("Denoising Strength (h)", 0.01, 0.5, 0.1, help="Denoising parameter")
        
        elif enhancement_method == 'Richardson-Lucy Deconvolution':
            parameters['iterations'] = st.slider("Iterations", 10, 100, 30, help="Number of deconvolution iterations")
            parameters['psf_size'] = st.slider("PSF Size", 3, 15, 5, help="Point spread function size")
            parameters['psf_sigma'] = st.slider("PSF Sigma", 0.5, 3.0, 1.0, help="Gaussian PSF standard deviation")
        
        elif 'Cellpose' in enhancement_method:
            parameters['diameter'] = st.number_input("Cell Diameter (pixels)", value=None, help="Expected cell diameter, None for auto")
            parameters['use_gpu'] = st.checkbox("Use GPU", False, help="Use GPU acceleration if available")
            
            # Channel configuration
            st.write("Channel Configuration:")
            parameters['channels'] = [
                st.selectbox("Cytoplasm Channel", [0, 1, 2], 0),
                st.selectbox("Nucleus Channel", [0, 1, 2], 0)
            ]
        
        elif enhancement_method == 'StarDist Nucleus Segmentation':
            parameters['model_name'] = st.selectbox(
                "Model", 
                ['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'],
                help="Pretrained model selection"
            )
            parameters['prob_thresh'] = st.slider("Probability Threshold", 0.1, 0.9, 0.5, help="Detection probability threshold")
            parameters['nms_thresh'] = st.slider("NMS Threshold", 0.1, 0.9, 0.4, help="Non-maximum suppression threshold")
        
        # Apply enhancement button
        if st.button("Apply Enhancement", type="primary"):
            with st.spinner(f"Applying {enhancement_method}..."):
                try:
                    image_data = st.session_state.loaded_data['image_data']
                    
                    # Handle time series data - use first frame for enhancement
                    if len(image_data.shape) > 3:
                        enhancement_data = image_data[0]  # Use first time point
                        st.info("Using first time point for enhancement demonstration")
                    else:
                        enhancement_data = image_data
                    
                    # Apply enhancement
                    enhancement_result = st.session_state.ai_enhancer.enhance_image(
                        enhancement_data, enhancement_method, parameters
                    )
                    
                    if enhancement_result.get('status') == 'success':
                        st.session_state.enhanced_data = enhancement_result
                        st.success(f"{enhancement_method} applied successfully!")
                        st.rerun()
                    else:
                        st.error(f"Enhancement failed: {enhancement_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Enhancement error: {str(e)}")
    
    with col2:
        # Display enhancement results
        if st.session_state.enhanced_data is not None:
            enhancement_data = st.session_state.enhanced_data
            
            st.subheader("Enhancement Results")
            
            # Show before/after comparison
            comparison_mode = st.radio("Display Mode", ["Side by Side", "Original", "Enhanced"])
            
            if comparison_mode == "Side by Side":
                col_orig, col_enh = st.columns(2)
                
                with col_orig:
                    st.write("**Original**")
                    if 'original_image' in enhancement_data:
                        normalized_orig = normalize_image_for_display(enhancement_data['original_image'])
                        st.image(normalized_orig, use_container_width=True)
                
                with col_enh:
                    st.write("**Enhanced**")
                    if 'enhanced_image' in enhancement_data:
                        normalized_enh = normalize_image_for_display(enhancement_data['enhanced_image'])
                        st.image(normalized_enh, use_container_width=True)
                    elif 'segmentation_masks' in enhancement_data:
                        normalized_masks = normalize_image_for_display(enhancement_data['segmentation_masks'])
                        st.image(normalized_masks, use_container_width=True)
            
            elif comparison_mode == "Original":
                if 'original_image' in enhancement_data:
                    normalized_orig = normalize_image_for_display(enhancement_data['original_image'])
                    st.image(normalized_orig, use_container_width=True)
            
            else:  # Enhanced
                if 'enhanced_image' in enhancement_data:
                    normalized_enh = normalize_image_for_display(enhancement_data['enhanced_image'])
                    st.image(normalized_enh, use_container_width=True)
                elif 'segmentation_masks' in enhancement_data:
                    normalized_masks = normalize_image_for_display(enhancement_data['segmentation_masks'])
                    st.image(normalized_masks, use_container_width=True)
            
            # Display enhancement statistics
            st.subheader("Enhancement Details")
            
            method_used = enhancement_data.get('method', 'Unknown')
            st.info(f"Method: {method_used}")
            
            if 'parameters_used' in enhancement_data:
                st.write("**Parameters Used:**")
                for param, value in enhancement_data['parameters_used'].items():
                    st.text(f"{param}: {value}")
            
            # Segmentation-specific results
            if 'num_objects' in enhancement_data:
                st.metric("Objects Detected", enhancement_data['num_objects'])
            elif 'num_nuclei' in enhancement_data:
                st.metric("Nuclei Detected", enhancement_data['num_nuclei'])
            
            # Option to use enhanced image for analysis
            if st.button("Use Enhanced Image for Analysis"):
                # Replace the original image data with enhanced version
                if 'enhanced_image' in enhancement_data:
                    st.session_state.loaded_data['image_data'] = enhancement_data['enhanced_image']
                    st.success("Enhanced image set as active data for analysis")
                    st.rerun()
        else:
            st.info("Apply an enhancement method to see results here")

def render_specialized_results():
    """Render specialized physics analysis results"""
    st.header("🔬 Specialized Physics Analysis Results")
    
    # Display optical flow results
    if 'optical_flow' in st.session_state.specialized_results:
        flow_result = st.session_state.specialized_results['optical_flow']
        st.subheader("🌊 Optical Flow & Elastography Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Method", flow_result.get('method', 'Unknown'))
            if 'avg_displacement' in flow_result:
                st.metric("Average Displacement", f"{flow_result['avg_displacement']:.3f} pixels")
            if 'max_displacement' in flow_result:
                st.metric("Maximum Displacement", f"{flow_result['max_displacement']:.3f} pixels")
        
        with col2:
            if 'num_tracked_points' in flow_result:
                st.metric("Tracked Points", flow_result['num_tracked_points'])
            
            # Display alignment quality metrics
            if 'alignment_quality' in flow_result:
                st.metric("Alignment Quality", f"{flow_result['alignment_quality']:.3f}")
            if 'total_drift' in flow_result:
                st.metric("Total Drift", f"{flow_result['total_drift']:.3f} pixels")
            
            # Display displacement vectors summary
            if 'displacement_vectors' in flow_result:
                vectors_df = flow_result['displacement_vectors']
                if hasattr(vectors_df, 'shape') and len(vectors_df) > 0:
                    st.metric("Total Vectors", len(vectors_df))
                    
                    # Show displacement distribution
                    if hasattr(vectors_df, 'columns') and 'magnitude' in vectors_df.columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(vectors_df['magnitude'], bins=20, alpha=0.7, color='blue')
                        ax.set_xlabel('Displacement Magnitude (pixels)')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Displacement Magnitude Distribution')
                        st.pyplot(fig)
                        plt.close()
        
        # Alignment preview integration
        if 'aligned_images' in flow_result:
            st.subheader("🎯 Image Alignment Results")
            
            aligned_images = flow_result['aligned_images']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Aligned Image Series Preview:**")
                
                # Preview controls for aligned images
                preview_col1, preview_col2 = st.columns(2)
                
                with preview_col1:
                    if len(aligned_images.shape) > 2:
                        preview_frame = st.slider("Preview Frame", 0, len(aligned_images)-1, 0, 
                                                key="aligned_preview_frame")
                    else:
                        preview_frame = 0
                
                with preview_col2:
                    show_comparison = st.checkbox("Show Before/After", value=False,
                                                key="show_alignment_comparison")
                
                # Display aligned image
                if show_comparison and 'image_data' in st.session_state.loaded_data:
                    # Show side-by-side comparison
                    comparison_col1, comparison_col2 = st.columns(2)
                    
                    with comparison_col1:
                        st.write("**Original:**")
                        original_data = st.session_state.loaded_data['image_data']
                        if len(original_data.shape) > 2:
                            if preview_frame < len(original_data):
                                original_frame = original_data[preview_frame]
                                if len(original_frame.shape) == 3:  # multichannel
                                    original_frame = original_frame[:, :, 0]
                                normalized_orig = normalize_image_for_display(original_frame)
                                st.image(normalized_orig, caption=f"Original Frame {preview_frame}", 
                                       use_container_width=True)
                    
                    with comparison_col2:
                        st.write("**Aligned:**")
                        if preview_frame < len(aligned_images):
                            aligned_frame = aligned_images[preview_frame]
                            if len(aligned_frame.shape) == 3:  # multichannel
                                aligned_frame = aligned_frame[:, :, 0]
                            normalized_aligned = normalize_image_for_display(aligned_frame)
                            st.image(normalized_aligned, caption=f"Aligned Frame {preview_frame}", 
                                   use_container_width=True)
                else:
                    # Show only aligned image
                    if preview_frame < len(aligned_images):
                        aligned_frame = aligned_images[preview_frame]
                        if len(aligned_frame.shape) == 3:  # multichannel
                            aligned_frame = aligned_frame[:, :, 0]
                        normalized_aligned = normalize_image_for_display(aligned_frame)
                        st.image(normalized_aligned, caption=f"Aligned Frame {preview_frame}", 
                               use_container_width=True)
            
            with col2:
                st.write("**Use Aligned Data:**")
                
                # Button to use aligned images for analysis
                if st.button("📊 Use Aligned for Analysis", type="primary", key="use_aligned_btn"):
                    # Update session state with aligned images
                    st.session_state.loaded_data['image_data'] = aligned_images
                    st.session_state.enhanced_data = {
                        'enhanced_image': aligned_images,
                        'method': 'Image Alignment',
                        'status': 'success',
                        'alignment_quality': flow_result.get('alignment_quality', 1.0)
                    }
                    st.success("Aligned image series set as active data for analysis and preview")
                    st.rerun()
                
                # Option to save aligned images
                if st.button("💾 Save Aligned Series", type="secondary", key="save_aligned_btn"):
                    # Store aligned images for export
                    if 'exports' not in st.session_state:
                        st.session_state.exports = {}
                    
                    st.session_state.exports['aligned_images'] = {
                        'data': aligned_images,
                        'method': flow_result.get('method', 'Unknown'),
                        'alignment_quality': flow_result.get('alignment_quality', 1.0),
                        'total_drift': flow_result.get('total_drift', 0)
                    }
                    st.success("Aligned images saved for export")
                
                # Display alignment statistics
                st.write("**Alignment Statistics:**")
                if 'registration_results' in flow_result:
                    reg_results = flow_result['registration_results']
                    successful_alignments = sum(1 for r in reg_results if r.get('displacement', float('inf')) != float('inf'))
                    st.metric("Successful Alignments", f"{successful_alignments}/{len(reg_results)}")
                    
                    if successful_alignments > 0:
                        avg_error = np.mean([r.get('registration_error', 0) for r in reg_results 
                                           if r.get('registration_error', float('inf')) != float('inf')])
                        st.metric("Avg Registration Error", f"{avg_error:.6f}")
        
        # Enhanced displacement visualization
        if 'displacement_vectors' in flow_result:
            st.subheader("🎯 Displacement Field Visualization")
            
            vectors_data = flow_result['displacement_vectors']
            if hasattr(vectors_data, 'shape') and len(vectors_data) > 0:
                # Create displacement field plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if hasattr(vectors_data, 'columns'):
                    # DataFrame format
                    if all(col in vectors_data.columns for col in ['x', 'y', 'dx', 'dy']):
                        ax.quiver(vectors_data['x'], vectors_data['y'], 
                                vectors_data['dx'], vectors_data['dy'],
                                vectors_data.get('magnitude', np.sqrt(vectors_data['dx']**2 + vectors_data['dy']**2)),
                                cmap='viridis', scale=50, alpha=0.7)
                        ax.set_title('Displacement Vector Field')
                        ax.set_xlabel('X Position (pixels)')
                        ax.set_ylabel('Y Position (pixels)')
                        ax.set_aspect('equal')
                        plt.colorbar(ax.collections[0], ax=ax, label='Displacement Magnitude (pixels)')
                        st.pyplot(fig)
                
                plt.close()
        
        # Display strain results for DIC
        if 'strain_results' in flow_result and flow_result['strain_results']:
            st.subheader("📊 Strain Analysis")
            strain_data = flow_result['strain_results']
            
            if 'strain_fields' in strain_data:
                strain_fields = strain_data['strain_fields']
                if strain_fields:
                    # Create strain metrics display
                    strain_metrics = []
                    for field in strain_fields:
                        strain_metrics.append({
                            'Frame': field.get('frame', 0),
                            'εxx': field.get('epsilon_xx', 0),
                            'εyy': field.get('epsilon_yy', 0),
                            'γxy': field.get('gamma_xy', 0),
                            'Von Mises': field.get('von_mises_strain', 0)
                        })
                    
                    strain_df = pd.DataFrame(strain_metrics)
                    st.dataframe(strain_df, use_container_width=True)
    
    # Display ICS results
    if 'ics' in st.session_state.specialized_results:
        ics_result = st.session_state.specialized_results['ics']
        st.subheader("📊 Image Correlation Spectroscopy Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Method", ics_result.get('method', 'Unknown'))
            
            # Display method-specific results
            if 'diffusion_coefficient' in ics_result:
                D_value = ics_result['diffusion_coefficient']
                st.metric("Diffusion Coefficient", f"{D_value:.6f} μm²/s")
            
            if 'flow_magnitude' in ics_result:
                flow_mag = ics_result['flow_magnitude']
                st.metric("Flow Velocity", f"{flow_mag:.3f} μm/s")
        
        with col2:
            if 'number_of_particles' in ics_result:
                N_value = ics_result['number_of_particles']
                st.metric("Particle Number", f"{N_value:.1f}")
            
            if 'fit_quality' in ics_result:
                r2_value = ics_result['fit_quality']
                st.metric("Fit Quality (R²)", f"{r2_value:.3f}")
            
            if 'mobile_fraction' in ics_result:
                mobile_frac = ics_result['mobile_fraction']
                st.metric("Mobile Fraction", f"{mobile_frac:.3f}")
        
        # Display correlation maps for imaging methods
        if 'diffusion_map' in ics_result:
            st.subheader("🗺️ Diffusion Map")
            diffusion_map = ics_result['diffusion_map']
            
            if hasattr(diffusion_map, 'shape') and len(diffusion_map.shape) == 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(diffusion_map, cmap='viridis', aspect='auto')
                ax.set_title('Spatial Diffusion Coefficient Map')
                ax.set_xlabel('X Position (pixels)')
                ax.set_ylabel('Y Position (pixels)')
                plt.colorbar(im, ax=ax, label='Diffusion Coefficient (μm²/s)')
                st.pyplot(fig)
                plt.close()
        
        # Display flow direction for pCF analysis
        if 'preferred_flow_direction' in ics_result:
            st.subheader("🌊 Flow Direction Analysis")
            
            direction_deg = ics_result.get('preferred_flow_direction', 0)
            anisotropy = ics_result.get('flow_anisotropy', 0)
            strength = ics_result.get('correlation_strength', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Flow Direction", f"{direction_deg:.1f}°")
            with col2:
                st.metric("Anisotropy", f"{anisotropy:.3f}")
            with col3:
                st.metric("Correlation Strength", f"{strength:.3f}")
        
        # Display STICS flow results
        if 'flow_velocity_x' in ics_result and 'flow_velocity_y' in ics_result:
            st.subheader("🔄 STICS Flow Analysis")
            
            vx = ics_result['flow_velocity_x']
            vy = ics_result['flow_velocity_y']
            vmag = ics_result.get('flow_magnitude', np.sqrt(vx**2 + vy**2))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Velocity X", f"{vx:.3f} μm/s")
            with col2:
                st.metric("Velocity Y", f"{vy:.3f} μm/s")
            with col3:
                st.metric("Speed", f"{vmag:.3f} μm/s")
    
    # Download results button
    if st.session_state.specialized_results:
        st.subheader("💾 Export Specialized Results")
        
        if st.button("📄 Download Analysis Summary", type="secondary"):
            # Create comprehensive summary
            summary_data = {}
            
            for method_type, results in st.session_state.specialized_results.items():
                summary_data[method_type] = {
                    'method': results.get('method', 'Unknown'),
                    'status': results.get('status', 'Unknown'),
                    'parameters': results.get('parameters_used', {})
                }
                
                # Add key metrics
                if method_type == 'optical_flow':
                    summary_data[method_type].update({
                        'avg_displacement': results.get('avg_displacement', 0),
                        'max_displacement': results.get('max_displacement', 0),
                        'num_tracked_points': results.get('num_tracked_points', 0)
                    })
                elif method_type == 'ics':
                    summary_data[method_type].update({
                        'diffusion_coefficient': results.get('diffusion_coefficient', 0),
                        'flow_magnitude': results.get('flow_magnitude', 0),
                        'fit_quality': results.get('fit_quality', 0)
                    })
            
            summary_json = json.dumps(summary_data, indent=2, default=str)
            
            st.download_button(
                label="📄 Download JSON Summary",
                data=summary_json,
                file_name="specialized_physics_analysis.json",
                mime="application/json"
            )

def render_automated_reports():
    """Render automated report generation interface"""
    st.header("📋 Automated Report Generation")
    st.markdown("Generate comprehensive analysis reports based on your data and results")
    
    # Report type selection
    report_types = [
        'comprehensive',
        'microscopy_analysis', 
        'fcs_analysis',
        'specialized_physics',
        'ai_enhancement'
    ]
    
    report_type_names = {
        'comprehensive': '🔬 Comprehensive Analysis Report',
        'microscopy_analysis': '📷 Microscopy-Focused Report',
        'fcs_analysis': '📊 FCS Analysis Report',
        'specialized_physics': '⚛️ Physics Methods Report',
        'ai_enhancement': '🤖 AI Enhancement Report'
    }
    
    selected_type = st.selectbox(
        "Report Type",
        options=report_types,
        format_func=lambda x: report_type_names.get(x, x),
        help="Choose the type of report to generate based on your analysis"
    )
    
    # Report generation button
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating automated report..."):
            try:
                # Generate report based on current data and results
                report = st.session_state.report_generator.generate_report(
                    data_info=st.session_state.loaded_data,
                    analysis_results=st.session_state.analysis_results or {},
                    specialized_results=st.session_state.specialized_results,
                    enhanced_data=st.session_state.enhanced_data,
                    report_type=selected_type
                )
                
                # Display report preview
                st.subheader("📖 Report Preview")
                with st.expander("View Generated Report", expanded=True):
                    st.markdown(report['report_content'])
                
                # Export options
                st.subheader("💾 Export Report")
                export_formats = st.session_state.report_generator.export_report_formats(report)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'markdown' in export_formats:
                        st.download_button(
                            label="📝 Download Markdown",
                            data=export_formats['markdown'],
                            file_name=f"analysis_report_{selected_type}.md",
                            mime="text/markdown"
                        )
                
                with col2:
                    if 'json' in export_formats:
                        st.download_button(
                            label="📊 Download JSON",
                            data=export_formats['json'],
                            file_name=f"analysis_data_{selected_type}.json", 
                            mime="application/json"
                        )
                
                with col3:
                    if 'html' in export_formats:
                        st.download_button(
                            label="🌐 Download HTML",
                            data=export_formats['html'],
                            file_name=f"analysis_report_{selected_type}.html",
                            mime="text/html"
                        )
                
                with col4:
                    if 'csv' in export_formats:
                        st.download_button(
                            label="📈 Download CSV",
                            data=export_formats['csv'],
                            file_name=f"analysis_summary_{selected_type}.csv",
                            mime="text/csv"
                        )
                
                # Report summary
                st.success(f"Report generated successfully! Report type: {report_type_names.get(selected_type, selected_type)}")
                
                # Key findings summary
                if report['report_data'].get('analysis_summary', {}).get('key_findings'):
                    st.subheader("🔍 Key Findings")
                    findings = report['report_data']['analysis_summary']['key_findings']
                    for key, value in findings.items():
                        st.metric(key.replace('_', ' ').title(), value)
                
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
    
    # Report templates information
    with st.expander("ℹ️ Report Templates Information"):
        st.markdown("""
        **Available Report Types:**
        
        - **Comprehensive**: Complete analysis including all methods and results
        - **Microscopy-Focused**: Dataset characteristics and imaging parameters  
        - **FCS Analysis**: Correlation spectroscopy and dynamics results
        - **Specialized Physics**: Advanced optical flow and ICS methods
        - **AI Enhancement**: AI-powered processing and improvement results
        
        Reports automatically adapt to your loaded data and applied analysis methods.
        """)

if __name__ == "__main__":
    main()
