"""
Advanced Image Biophysics - Main Streamlit Application

A comprehensive microscopy data analysis platform with AI enhancement
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
import sys
import io
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Image Biophysics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import application modules with error handling
def safe_import(module_name, display_name=None):
    """Safely import a module with error handling"""
    if display_name is None:
        display_name = module_name
    try:
        return __import__(module_name)
    except ImportError as e:
        st.sidebar.warning(f"⚠️ {display_name} module not available: {str(e)}")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {display_name}: {str(e)}")
        return None

# Import modules
data_loader = safe_import('data_loader', 'Data Loader')
visualization = safe_import('visualization', 'Visualization')
ai_enhancement = safe_import('ai_enhancement', 'AI Enhancement')
report_generator = safe_import('report_generator', 'Report Generator')
utils = safe_import('utils', 'Utilities')
fcs_analysis = safe_import('fcs_analysis', 'FCS Analysis')
advanced_analysis = safe_import('advanced_analysis', 'Advanced Analysis')
segmented_fcs = safe_import('segmented_fcs', 'Segmented FCS')
optical_flow_analysis = safe_import('optical_flow_analysis', 'Optical Flow Analysis')
image_correlation_spectroscopy = safe_import('image_correlation_spectroscopy', 'Image Correlation Spectroscopy')
nuclear_biophysics = safe_import('nuclear_biophysics', 'Nuclear Biophysics')
thumbnail_generator = safe_import('thumbnail_generator', 'Thumbnail Generator')
batch_processing = safe_import('batch_processing', 'Batch Processing')
number_and_brightness = safe_import('number_and_brightness', 'Number & Brightness')
pair_correlation_function = safe_import('pair_correlation_function', 'Pair Correlation Function')


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_data = None
        st.session_state.analysis_results = {}
        st.session_state.current_file = None
        st.session_state.processing = False
        st.session_state.image = None


        # Initialize AI enhancer if available
        if ai_enhancement is not None and hasattr(ai_enhancement, 'AIEnhancementManager'):
            try:
                st.session_state.ai_enhancer = ai_enhancement.AIEnhancementManager()
            except Exception as e:
                st.session_state.ai_enhancer = None
                st.sidebar.warning(f"AI Enhancement initialization failed: {str(e)}")
        else:
            st.session_state.ai_enhancer = None

        # Initialize data loader if available
        if data_loader is not None and hasattr(data_loader, 'DataLoader'):
            try:
                st.session_state.data_loader = data_loader.DataLoader()
            except Exception as e:
                st.session_state.data_loader = None
                st.sidebar.warning(f"Data Loader initialization failed: {str(e)}")
        else:
            st.session_state.data_loader = None

        # Initialize visualization manager if available
        if visualization is not None and hasattr(visualization, 'VisualizationManager'):
            try:
                st.session_state.visualizer = visualization.VisualizationManager()
            except Exception as e:
                st.session_state.visualizer = None
                st.sidebar.warning(f"Visualization initialization failed: {str(e)}")
        else:
            st.session_state.visualizer = None

        # Initialize analysis manager if available
        if advanced_analysis is not None and hasattr(advanced_analysis, 'AnalysisManager'):
            try:
                st.session_state.analyzer = advanced_analysis.AnalysisManager()
            except Exception as e:
                st.session_state.analyzer = None
                st.sidebar.warning(f"Analysis initialization failed: {str(e)}")
        else:
            st.session_state.analyzer = None

        # Initialize N&B analyzer
        if number_and_brightness is not None and hasattr(number_and_brightness, 'NumberAndBrightness'):
            st.session_state.nb_analyzer = number_and_brightness.NumberAndBrightness()
        else:
            st.session_state.nb_analyzer = None

        # Initialize PCF analyzer
        if pair_correlation_function is not None and hasattr(pair_correlation_function, 'PairCorrelationFunction'):
            st.session_state.pcf_analyzer = pair_correlation_function.PairCorrelationFunction()
        else:
            st.session_state.pcf_analyzer = None

# Main application function
def main():
    """Main application entry point"""

    # Initialize session state
    initialize_session_state()

    # Application header
    st.title("🔬 Advanced Image Biophysics")
    st.markdown("""
    Comprehensive microscopy data analysis platform with AI-powered enhancement tools,
    specialized physics methods, and automated reporting capabilities.
    """)

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            [
                "🏠 Home",
                "📁 Data Loading",
                "📊 Analysis",
                "🎨 AI Enhancement",
                "📈 Visualization",
                "📄 Reports",
                "⚙️ Batch Processing"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        # Display system information
        st.subheader("System Status")

        # Check available modules
        modules_status = {
            "Data Loader": data_loader is not None,
            "Visualization": visualization is not None,
            "AI Enhancement": ai_enhancement is not None,
            "Report Generator": report_generator is not None,
            "Utilities": utils is not None,
            "FCS Analysis": fcs_analysis is not None,
            "Advanced Analysis": advanced_analysis is not None,
            "Segmented FCS": segmented_fcs is not None,
            "Optical Flow": optical_flow_analysis is not None,
            "ICS": image_correlation_spectroscopy is not None,
            "Nuclear Biophysics": nuclear_biophysics is not None,
            "Batch Processing": batch_processing is not None,
            "Number & Brightness": number_and_brightness is not None,
            "Pair Correlation Function": pair_correlation_function is not None
        }

        for module, available in modules_status.items():
            if available:
                st.success(f"✓ {module}")
            else:
                st.error(f"✗ {module}")

        # Display current file info
        if st.session_state.current_file:
            st.divider()
            st.subheader("Current File")
            st.text(st.session_state.current_file)

    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Data Loading":
        show_data_loading_page()
    elif page == "📊 Analysis":
        show_analysis_page()
    elif page == "🎨 AI Enhancement":
        show_ai_enhancement_page()
    elif page == "📈 Visualization":
        show_visualization_page()
    elif page == "📄 Reports":
        show_reports_page()
    elif page == "⚙️ Batch Processing":
        show_batch_processing_page()

def show_home_page():
    """Display the home/welcome page"""
    st.header("Welcome to Advanced Image Biophysics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Features")
        st.markdown("""
        ### Data Loading & Preview
        - Multi-format support (TIFF, CZI, LIF, etc.)
        - Real-time thumbnails
        - FCS data support
        - Automatic format detection
        ### Analysis Methods
        - Fluorescence Correlation Spectroscopy (FCS)
        - Raster Image Correlation Spectroscopy (RICS)
        - Image Mean Square Displacement (iMSD)
        - Single Particle Tracking (SPT)
        - Optical Flow Analysis
        - Image Correlation Spectroscopy (ICS)
        - Number & Brightness (N&B)
        - Pair Correlation Function (PCF)
        ### AI Enhancement
        - Denoising (Non-local means, Richardson-Lucy)
        - Segmentation (Cellpose, StarDist)
        - Noise2Void self-supervised denoising
        - CARE restoration
        """)

    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        ### Getting Started
        1. **Load Data**: Navigate to 📁 Data Loading
        2. **Upload Files**: Supports multiple microscopy formats
        3. **Preview**: View thumbnails and metadata
        4. **Analyze**: Choose analysis method
        5. **Enhance**: Apply AI-powered enhancements
        6. **Export**: Generate reports and download results
        ### Supported Formats
        - **TIFF/STK**: MetaMorph, Leica, Olympus
        - **CZI**: Zeiss LSM 700, Elyra 7
        - **LIF**: Leica SP8
        - **OIF/OIB**: Olympus imaging
        - **FCS**: Correlation spectroscopy data
        ### Tips
        - Start with Data Loading page
        - Preview files before analysis
        - Use AI Enhancement for noisy data
        - Generate reports for documentation
        """)

    st.divider()

    # Display recent activity or statistics
    st.subheader("📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Files Loaded", len(st.session_state.analysis_results))
    with col2:
        st.metric("Analyses Complete",
                  sum(1 for v in st.session_state.analysis_results.values() if v))
    with col3:
        status = "Ready" if st.session_state.current_data is not None else "No Data"
        st.metric("Status", status)
    with col4:
        ai_status = "Available" if st.session_state.ai_enhancer else "Unavailable"
        st.metric("AI Enhancement", ai_status)

def show_data_loading_page():
    """Display the data loading page"""
    st.header("📁 Data Loading")

    if st.session_state.data_loader is None:
        st.error("❌ Data Loader module is not available. Please check installation.")
        return

    st.markdown("Upload microscopy data files for analysis.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['tif', 'tiff', 'stk', 'lsm', 'czi', 'lif', 'oif', 'oib', 'nd2', 'fcs'],
        help="Supported formats: TIFF, STK, LSM, CZI, LIF, OIF, OIB, ND2, FCS"
    )

    if uploaded_file is not None:
        st.session_state.current_file = uploaded_file.name

        with st.spinner(f"Loading {uploaded_file.name}..."):
            try:
                # Store the file in a buffer in session state
                st.session_state.uploaded_file_buffer = io.BytesIO(uploaded_file.getvalue())

                with st.spinner("Reading image metadata..."):
                    st.session_state.channel_count = st.session_state.data_loader.get_channel_count(st.session_state.uploaded_file_buffer)

                # UI for channel selection if multi-channel
                selected_channel = 0
                if st.session_state.channel_count > 1:
                    selected_channel = st.selectbox("Select Channel to Display", list(range(st.session_state.channel_count)))

                with st.spinner(f"Loading channel {selected_channel}..."):
                    load_result = st.session_state.data_loader.load_image(st.session_state.uploaded_file_buffer, channel=selected_channel)
                    if load_result['status'] == 'success':
                        st.session_state.image = load_result['image_data']
                        st.session_state.voxel_size = load_result['voxel_size']
                        st.success(f"Channel {selected_channel} loaded successfully!")
                        # Reset other states
                        for key in ['enhanced_result', 'segmentation_mask', 'analysis_results', 'percolation_results', 'colocalization_results']:
                            st.session_state[key] = None
                    else:
                        st.error("Failed to load image channel.")

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Upload a file to begin")

def show_analysis_page():
    """Display the analysis page"""
    st.header("📊 Analysis")

    if st.session_state.image is None:
        st.warning("⚠️ Please load data first from the Data Loading page.")
        return

    st.markdown("Select and configure analysis methods for your data.")

    render_analysis_controls()
    render_colocalization_controls()

def show_ai_enhancement_page():
    """Display the AI enhancement page"""
    st.header("🎨 AI Enhancement")

    if st.session_state.image is None:
        st.warning("⚠️ Please load data first from the Data Loading page.")
        return

    if st.session_state.ai_enhancer is None:
        st.error("❌ AI Enhancement module is not available. Install AI dependencies for full functionality.")
        st.info("Install with: `pip install cellpose stardist tensorflow`")
        return

    st.markdown("Apply AI-powered image enhancement and restoration methods.")

    render_ai_enhancement_controls()

def show_visualization_page():
    """Display the visualization page"""
    st.header("📈 Visualization")

    if st.session_state.image is None:
        st.warning("⚠️ No analysis results to visualize. Run an analysis first.")
        return

    st.markdown("Interactive visualization of analysis results.")

    if st.session_state.visualizer:
        st.session_state.visualizer.display_interactive_3d_volume(st.session_state.image)
    
    if "Number & Brightness" in st.session_state.analysis_results:
        st.subheader("Number & Brightness Results")
        nb_results = st.session_state.analysis_results["Number & Brightness"]
        if st.session_state.visualizer:
            st.session_state.visualizer.display_image(nb_results['number_map'], title="Number Map")
            st.session_state.visualizer.display_image(nb_results['brightness_map'], title="Brightness Map")
            
    if "Pair Correlation Function" in st.session_state.analysis_results:
        st.subheader("Pair Correlation Function Results")
        pcf_results = st.session_state.analysis_results["Pair Correlation Function"]
        if st.session_state.visualizer:
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Scatter(x=pcf_results['radius'], y=pcf_results['pcf'], mode='lines'))
            fig.update_layout(title="Pair Correlation Function", xaxis_title="Radius (pixels)", yaxis_title="g(r)")
            st.plotly_chart(fig)


def show_reports_page():
    """Display the reports generation page"""
    st.header("📄 Reports")

    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results to report. Run an analysis first.")
        return

    st.markdown("Generate comprehensive analysis reports.")

    # Report format selection
    report_format = st.selectbox(
        "Report Format",
        ["Markdown", "HTML", "PDF", "JSON", "CSV"]
    )

    # Report content options
    st.subheader("Report Contents")
    include_metadata = st.checkbox("Include Metadata", value=True)
    include_parameters = st.checkbox("Include Analysis Parameters", value=True)
    include_plots = st.checkbox("Include Plots", value=True)
    include_statistics = st.checkbox("Include Statistics", value=True)

    # Generate report button
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                st.success("✓ Report generated successfully!")
                st.info("⚠️ Report generation implementation in progress.")

                # Placeholder for download button
                st.download_button(
                    label="⬇️ Download Report",
                    data="Report content placeholder",
                    file_name=f"analysis_report.{report_format.lower()}",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"❌ Report generation failed: {str(e)}")
                st.exception(e)

def show_batch_processing_page():
    """Display the batch processing page"""
    st.header("⚙️ Batch Processing")
    render_batch_controls()

def render_batch_controls():
    """Renders controls for batch processing."""

    if not batch_processing:
        st.error("Batch processing module not available.")
        return

    input_dir = st.text_input("Input Directory", "/path/to/your/images")
    output_dir = st.text_input("Output Directory", "/path/to/your/results")

    st.subheader("Processing Pipeline")

    # Use a separate key for batch AI enhancement controls
    render_ai_enhancement_controls(context='batch')

    analysis_tasks = st.multiselect("Analysis Tasks", ['morphometrics'], key='batch_analysis_tasks')

    if st.button("Run Batch", key='run_batch_processing'):
        if not os.path.isdir(input_dir):
            st.error("Input directory does not exist.")
            return
        if not os.path.isdir(output_dir):
            st.info(f"Output directory does not exist. It will be created.")
            os.makedirs(output_dir)

        enhancement_method = st.session_state.get('ai_enhancement_batch_method', None)
        # This part needs to be more robust to gather the right params based on the selected method
        enhancement_params = {}
        # A more complete implementation would fetch the correct parameters
        # from the session state based on the enhancement_method

        analysis_params = {
            'voxel_size': st.session_state.get('voxel_size', (1.0, 1.0, 1.0)) # Example
        }

        processor = batch_processing.BatchProcessor(
            enhancement_method=enhancement_method,
            enhancement_params=enhancement_params,
            analysis_tasks=analysis_tasks,
            analysis_params=analysis_params
        )

        st.info("Starting batch processing... Check the console for progress.")
        # In a real app, you might use a thread or subprocess to avoid blocking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(i, total, message):
            progress_bar.progress(i / total)
            status_text.text(f"[{i}/{total}] {message}")

        processor.run(input_dir, output_dir, callback=progress_callback)
        st.success("Batch processing finished!")

def render_colocalization_controls(context='main'):
    """Renders controls for colocalization analysis."""
    key_prefix = f"coloc_{context}"

    if 'channel_count' in st.session_state and st.session_state.channel_count > 1:
        st.subheader("Colocalization Analysis")

        channel_options = list(range(st.session_state.channel_count))

        c1, c2 = st.columns(2)
        with c1:
            ch1_select = st.selectbox("Channel 1", channel_options, key=f"{key_prefix}_ch1")
        with c2:
            ch2_select = st.selectbox("Channel 2", channel_options, index=min(1, len(channel_options)-1), key=f"{key_prefix}_ch2")

        use_mask = st.checkbox("Use Segmentation Mask", key=f"{key_prefix}_use_mask")

        t1, t2 = st.columns(2)
        with t1:
            thresh1 = st.number_input("Channel 1 Threshold", min_value=0, value=0, key=f"{key_prefix}_thresh1")
        with t2:
            thresh2 = st.number_input("Channel 2 Threshold", min_value=0, value=0, key=f"{key_prefix}_thresh2")

        if st.button("📈 Analyze Colocalization", key=f"{key_prefix}_run"):
            if ch1_select == ch2_select:
                st.warning("Please select two different channels.")
            else:
                with st.spinner("Running colocalization analysis..."):
                    # Reload the specific channels
                    file_buffer = st.session_state.uploaded_file_buffer
                    file_buffer.seek(0)
                    ch1_data = st.session_state.data_loader.load_image(file_buffer, channel=ch1_select)['image_data']
                    file_buffer.seek(0)
                    ch2_data = st.session_state.data_loader.load_image(file_buffer, channel=ch2_select)['image_data']

                    mask = st.session_state.segmentation_mask if use_mask else None

                    if st.session_state.analyzer:
                        coloc_results = st.session_state.analyzer.calculate_colocalization(
                            ch1_data, ch2_data, mask=mask, threshold1=thresh1, threshold2=thresh2
                        )

                        if coloc_results['status'] == 'success':
                            st.session_state.colocalization_results = coloc_results
                            st.success("Colocalization analysis complete!")
                        else:
                            st.error(f"Analysis failed: {coloc_results['message']}")

    if 'colocalization_results' in st.session_state and st.session_state.colocalization_results:
        st.write("### Colocalization Results")
        res = st.session_state.colocalization_results
        st.metric("Pearson's Coefficient", f"{res['pearson_coefficient']:.3f}")
        st.metric("Mander's M1 (Ch1 overlap Ch2)", f"{res['manders_m1']:.3f}")
        st.metric("Mander's M2 (Ch2 overlap Ch1)", f"{res['manders_m2']:.3f}")

def render_ai_enhancement_controls(context='main'):
    """Renders AI enhancement controls, adaptable for different contexts."""
    if 'ai_enhancer' not in st.session_state:
        st.session_state.ai_enhancer = ai_enhancement.AIEnhancementManager()

    available_methods = st.session_state.ai_enhancer.get_available_methods()
    if not available_methods:
        st.info("No AI enhancement libraries available.")
        return

    key_prefix = f"ai_enhancement_{context}"
    enhancement_method = st.selectbox(
        "Enhancement Method",
        options=available_methods,
        key=f"{key_prefix}_method"
    )

    parameters = {}
    if ai_enhancement:
        defaults = ai_enhancement.get_enhancement_parameters(enhancement_method)

        if enhancement_method == 'Non-local Means Denoising':
            st.subheader("Denoising Parameters")
            parameters['patch_size'] = st.slider("Patch Size", 3, 15, defaults.get('patch_size', 5), key=f"{key_prefix}_patch_size")
            parameters['patch_distance'] = st.slider("Patch Distance", 3, 15, defaults.get('patch_distance', 6), key=f"{key_prefix}_patch_distance")
            parameters['auto_sigma'] = st.checkbox("Automatically estimate noise", value=defaults.get('auto_sigma', True), key=f"{key_prefix}_auto_sigma")
            parameters['h'] = st.number_input("Denoising strength (h)", 0.01, 1.0, defaults.get('h', 0.1), 0.01, disabled=parameters['auto_sigma'], key=f"{key_prefix}_h_value")

        elif enhancement_method in ['Richardson-Lucy Deconvolution', 'Richardson-Lucy with Total Variation', 'FISTA Deconvolution', 'ISTA Deconvolution', 'Iterative Constraint Tikhonov-Miller']:
            pass

    if st.button(f"🎨 Enhance Image", key=f"{key_prefix}_run"):
        if 'image' in st.session_state and st.session_state.image is not None:
            with st.spinner(f'Running {enhancement_method}...'):
                result = st.session_state.ai_enhancer.enhance_image(
                    st.session_state.image,
                    enhancement_method,
                    parameters
                )
                if result.get('status') == 'success':
                    st.session_state.enhanced_result = result
                    st.success(f"Enhancement with {enhancement_method} completed successfully!")
                    if 'segmentation_masks' in result:
                        st.session_state.segmentation_mask = result['segmentation_masks']
                else:
                    st.error(f"Enhancement failed: {result.get('message', 'Unknown error')}")
        else:
            st.warning("Please load an image before applying enhancement.")


def render_analysis_controls(context='main'):
    """Renders controls for analysis methods."""
    key_prefix = f"analysis_{context}"

    analysis_methods = ["Number & Brightness", "Pair Correlation Function"]
    
    analysis_method = st.selectbox(
        "Select Analysis Method",
        analysis_methods,
        key=f"{key_prefix}_method"
    )

    params = {}
    if analysis_method == "Number & Brightness":
        st.subheader("N&B Parameters")
        params['window_size'] = st.slider("Window Size", min_value=8, max_value=128, value=32, step=8, key=f"{key_prefix}_nb_window_size")
    elif analysis_method == "Pair Correlation Function":
        st.subheader("PCF Parameters")
        params['max_radius'] = st.slider("Max Radius", min_value=10, max_value=200, value=50, step=10, key=f"{key_prefix}_pcf_max_radius")

    if st.button("📈 Run Analysis", key=f"{key_prefix}_run"):
        if st.session_state.image is None:
            st.warning("Please load an image first.")
            return

        with st.spinner(f"Running {analysis_method}..."):
            results = None
            if analysis_method == "Number & Brightness":
                if st.session_state.nb_analyzer:
                    results = st.session_state.nb_analyzer.analyze(st.session_state.image, **params)
            elif analysis_method == "Pair Correlation Function":
                if st.session_state.pcf_analyzer:
                    # PCF works on 2D images. If we have a 3D stack, we analyze the mean projection.
                    image_2d = st.session_state.image
                    if image_2d.ndim == 3:
                        image_2d = np.mean(image_2d, axis=0)
                    results = st.session_state.pcf_analyzer.analyze(image_2d, **params)

            if results and results['status'] == 'success':
                st.session_state.analysis_results[analysis_method] = results
                st.success(f"{analysis_method} analysis complete!")
            elif results:
                st.error(f"Analysis failed: {results['message']}")
            else:
                st.error("Analysis could not be performed.")



# Application entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page or contact support if the problem persists.")
