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
import importlib

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
        return importlib.import_module(module_name)
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
analysis_core = safe_import('analysis', 'Core Analysis')
utils = safe_import('utils', 'Utilities')
fcs_analysis = safe_import('fcs_analysis', 'FCS Analysis')
advanced_analysis = safe_import('advanced_analysis', 'Advanced Analysis')
segmented_fcs = safe_import('segmented_fcs', 'Segmented FCS')
displacement_correlation_spectroscopy = safe_import('displacement_correlation_spectroscopy', 'DCS Analysis')
deformation_microscopy = safe_import('deformation_microscopy', 'Deformation Microscopy')
two_domain_elastography = safe_import('two_domain_elastography', 'Two-Domain Elastography')
allen_segmenter_ui = safe_import('allen_segmenter.ui', 'Allen Segmenter UI')
optical_flow_analysis = safe_import('optical_flow_analysis', 'Optical Flow Analysis')
image_correlation_spectroscopy = safe_import('image_correlation_spectroscopy', 'Image Correlation Spectroscopy')
nuclear_biophysics = safe_import('nuclear_biophysics', 'Nuclear Biophysics')
thumbnail_generator = safe_import('thumbnail_generator', 'Thumbnail Generator')
batch_processing = safe_import('batch_processing', 'Batch Processing')
number_and_brightness = safe_import('number_and_brightness', 'Number & Brightness')
pair_correlation_function = safe_import('pair_correlation_function', 'Pair Correlation Function')
population_analysis = safe_import('population_analysis', 'Population Analysis')
statistics_analyzer = safe_import('statistics_analyzer', 'Statistics Analyzer')
material_mechanics = safe_import('material_mechanics', 'Material Mechanics')


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
        st.session_state.population_data = None
        st.session_state.report_content = None

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
        if analysis_core is not None and hasattr(analysis_core, 'AnalysisManager'):
            try:
                st.session_state.analyzer = analysis_core.AnalysisManager()
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
        
        # Initialize Population analyzer
        if population_analysis is not None and hasattr(population_analysis, 'PopulationAnalyzer'):
            st.session_state.population_analyzer = population_analysis.PopulationAnalyzer()
        else:
            st.session_state.population_analyzer = None

        # Initialize Statistics analyzer
        if statistics_analyzer is not None and hasattr(statistics_analyzer, 'StatisticsAnalyzer'):
            st.session_state.statistics_analyzer = statistics_analyzer.StatisticsAnalyzer()
        else:
            st.session_state.statistics_analyzer = None
        
        # Initialize Report Generator
        if report_generator is not None and hasattr(report_generator, 'ReportGenerator'):
            st.session_state.report_generator = report_generator.ReportGenerator()
        else:
            st.session_state.report_generator = None

        # Initialize specialized analyzers
        if segmented_fcs is not None and hasattr(segmented_fcs, 'SegmentedFCSAnalyzer'):
            st.session_state.segmented_fcs_analyzer = segmented_fcs.SegmentedFCSAnalyzer()
        else:
            st.session_state.segmented_fcs_analyzer = None

        if optical_flow_analysis is not None and hasattr(optical_flow_analysis, 'OpticalFlowAnalyzer'):
            st.session_state.optical_flow_analyzer = optical_flow_analysis.OpticalFlowAnalyzer()
        else:
            st.session_state.optical_flow_analyzer = None

        if image_correlation_spectroscopy is not None and hasattr(image_correlation_spectroscopy, 'ImageCorrelationSpectroscopy'):
            st.session_state.ics_analyzer = image_correlation_spectroscopy.ImageCorrelationSpectroscopy()
        else:
            st.session_state.ics_analyzer = None

        if displacement_correlation_spectroscopy is not None and hasattr(displacement_correlation_spectroscopy, 'DisplacementCorrelationSpectroscopy'):
            st.session_state.dcs_analyzer = displacement_correlation_spectroscopy.DisplacementCorrelationSpectroscopy()
        else:
            st.session_state.dcs_analyzer = None

        if deformation_microscopy is not None and hasattr(deformation_microscopy, 'DeformationMicroscopy'):
            st.session_state.dm_analyzer = deformation_microscopy.DeformationMicroscopy()
        else:
            st.session_state.dm_analyzer = None

        if two_domain_elastography is not None and hasattr(two_domain_elastography, 'TwoDomainNuclearElastography'):
            st.session_state.elastography_analyzer = two_domain_elastography.TwoDomainNuclearElastography()
        else:
            st.session_state.elastography_analyzer = None

        if nuclear_biophysics is not None and hasattr(nuclear_biophysics, 'NuclearBiophysicsAnalyzer'):
            st.session_state.nuclear_analyzer = nuclear_biophysics.NuclearBiophysicsAnalyzer()
        else:
            st.session_state.nuclear_analyzer = None

        if material_mechanics is not None and hasattr(material_mechanics, 'MaterialMechanics'):
            st.session_state.material_mechanics_analyzer = material_mechanics.MaterialMechanics()
        else:
            st.session_state.material_mechanics_analyzer = None

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
                "🎨 AI Enhancement",
                "👥 Population Analysis",
                "📊 Analysis",
                "📈 Visualization",
                "📄 Reports",
                "⚙️ Batch Processing",
                "🧬 Allen Cell Segmenter"
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
            "Material Mechanics": material_mechanics is not None,
            "Batch Processing": batch_processing is not None,
            "Number & Brightness": number_and_brightness is not None,
            "Pair Correlation Function": pair_correlation_function is not None,
            "Population Analysis": population_analysis is not None,
            "Statistics Analyzer": statistics_analyzer is not None,
            "Allen Cell Segmenter": allen_segmenter_ui is not None
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
    elif page == "👥 Population Analysis":
        show_population_analysis_page()
    elif page == "📈 Visualization":
        show_visualization_page()
    elif page == "📄 Reports":
        show_reports_page()
    elif page == "⚙️ Batch Processing":
        show_batch_processing_page()
    elif page == "🧬 Allen Cell Segmenter":
        if allen_segmenter_ui:
            allen_segmenter_ui.show_allen_segmenter_page()
        else:
            st.error("Allen Cell Segmenter module is not available.")

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
        - Population Analysis (Morphometry, Statistics)
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
        2. **Segment Objects**: Use `🎨 AI Enhancement` to generate a mask
        3. **Population Analysis**: Go to `👥 Population Analysis` to extract features and perform statistical tests.
        4. **Analyze**: Choose other analysis methods
        5. **Export**: Generate reports and download results
        """)

    st.divider()

    # Display recent activity or statistics
    st.subheader("📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_objects = len(st.session_state.population_data) if st.session_state.population_data is not None else 0
        st.metric("Objects Analyzed", num_objects)
    with col2:
        st.metric("Analyses Complete",
                  sum(1 for v in st.session_state.analysis_results.values() if v))
    with col3:
        status = "Ready" if st.session_state.image is not None else "No Data"
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
                st.session_state.uploaded_file_buffer = io.BytesIO(uploaded_file.getvalue())
                st.session_state.uploaded_file_buffer.seek(0)
                st.session_state.channel_count = st.session_state.data_loader.get_channel_count(st.session_state.uploaded_file_buffer)

                selected_channel = 0
                if st.session_state.channel_count > 1:
                    selected_channel = st.selectbox("Select Channel for Analysis", list(range(st.session_state.channel_count)))

                st.session_state.uploaded_file_buffer.seek(0)
                load_result = st.session_state.data_loader.load_image(st.session_state.uploaded_file_buffer, channel=selected_channel)
                
                if load_result['status'] == 'success':
                    st.session_state.image = load_result['image_data']
                    st.session_state.voxel_size = load_result['voxel_size']
                    st.success(f"Channel {selected_channel} loaded successfully!")
                    # Reset dependent states
                    for key in ['enhanced_result', 'segmentation_mask', 'analysis_results', 'population_data', 'report_content']:
                        if key in st.session_state:
                            st.session_state[key] = {} if key == 'analysis_results' else None
                else:
                    st.error("Failed to load image channel.")

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Upload a file to begin")

def show_population_analysis_page():
    """Displays the population analysis page with an improved GUI."""
    st.header("👥 Population Analysis")

    if st.session_state.get('image') is None:
        st.warning("⚠️ Please load data first from the '📁 Data Loading' page.")
        return

    if st.session_state.get('population_analyzer') is None:
        st.error("❌ Population Analysis module is not available. Please check installation.")
        return

    if st.session_state.get('segmentation_mask') is None:
        st.warning("⚠️ No segmentation mask found. Please generate a mask first on the '🎨 AI Enhancement' page.")
        return

    # Layout: Controls on the left, Results on the right
    controls_col, results_col = st.columns((1, 2))

    with controls_col:
        st.subheader("Controls")
        if st.button("📈 Run Population Analysis", type="primary"):
            with st.spinner("Analyzing population features..."):
                image_2d = st.session_state.image
                if image_2d.ndim == 3:
                    image_2d = np.mean(image_2d, axis=0)
                
                result = st.session_state.population_analyzer.analyze(
                    image=image_2d,
                    mask=st.session_state.segmentation_mask
                )

                if result['status'] == 'success':
                    st.session_state.population_data = result['data']
                    st.success(result['message'])
                else:
                    st.error(result['message'])
    
    with results_col:
        st.subheader("Results")
        if st.session_state.get('population_data') is not None:
            df = st.session_state.population_data
            
            tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Feature Distributions", "Correlation Plots", "Statistical Comparison"])

            with tab1:
                st.dataframe(df)
                st.download_button(
                    label="⬇️ Download Data as CSV",
                    data=df.to_csv().encode('utf-8'),
                    file_name='population_analysis.csv',
                    mime='text/csv',
                )

            with tab2:
                st.markdown("#### Single Feature Distribution")
                if not df.columns.empty:
                    feature = st.selectbox("Select feature to plot", options=df.columns, index=1, key='dist_feat')
                    if feature:
                        import plotly.express as px
                        fig = px.histogram(df, x=feature, title=f'Distribution of {feature}', nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.dist_fig = fig

            with tab3:
                st.markdown("#### Feature Correlation")
                if len(df.columns) > 1:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feat = st.selectbox("X-axis feature", options=df.columns, index=1, key='corr_x')
                    with col2:
                        y_feat = st.selectbox("Y-axis feature", options=df.columns, index=len(df.columns)-2 if len(df.columns) > 2 else 1, key='corr_y')
                    
                    if x_feat and y_feat:
                        import plotly.express as px
                        fig = px.scatter(df, x=x_feat, y=y_feat, title=f'{y_feat} vs. {x_feat}', hover_data=['label'])
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.corr_fig = fig
            
            with tab4:
                render_statistics_controls(df)

        else:
            st.info("Run the analysis to see the results.")


def render_statistics_controls(df):
    """Renders the controls for statistical analysis on a given dataframe."""
    st.markdown("#### Group Comparison")

    # Initialize group column if not present
    if 'group' not in st.session_state.population_data.columns:
        st.session_state.population_data['group'] = 'Group 1'

    # --- Group Assignment UI ---
    st.markdown("**1. Assign Objects to Groups**")
    group_name = st.text_input("New group name", "Group 2")
    
    all_labels = st.session_state.population_data['label'].unique().tolist()
    labels_to_group = st.multiselect("Select object labels to assign to new group", options=all_labels)
    
    if st.button(f"Assign to {group_name}"):
        st.session_state.population_data.loc[st.session_state.population_data['label'].isin(labels_to_group), 'group'] = group_name
        st.success(f"Assigned {len(labels_to_group)} objects to {group_name}")
        st.experimental_rerun() # Rerun to update the UI with new group assignments

    st.dataframe(st.session_state.population_data[['label', 'group']])

    # --- Statistical Test UI ---
    st.markdown("**2. Perform Statistical Test**")
    
    groups_in_data = sorted(st.session_state.population_data['group'].unique().tolist())
    if len(groups_in_data) < 2:
        st.info("You need at least two groups to perform a statistical test.")
        return

    col1, col2 = st.columns(2)
    with col1:
        feature_to_test = st.selectbox("Select feature to test", options=[col for col in df.columns if col not in ['label', 'group']], index=0)
    with col2:
        test_type = st.selectbox("Select test type", ["T-test", "Mann-Whitney U", "ANOVA", "Kruskal-Wallis"])

    groups_to_compare = st.multiselect("Select groups to compare", options=groups_in_data, default=groups_in_data)

    if st.button("🔬 Run Test"):
        if len(groups_to_compare) < 2:
            st.warning("Please select at least two groups to compare.")
        elif st.session_state.statistics_analyzer and feature_to_test:
            with st.spinner("Performing statistical test..."):
                test_result = st.session_state.statistics_analyzer.perform_test(
                    data=st.session_state.population_data[st.session_state.population_data['group'].isin(groups_to_compare)],
                    feature=feature_to_test,
                    groups=groups_to_compare,
                    test_type=test_type
                )

                if test_result['status'] == 'success':
                    st.session_state.test_result = test_result
                    st.success("Test complete!")
                else:
                    st.error(f"Test failed: {test_result['message']}")

    if 'test_result' in st.session_state and st.session_state.test_result:
        res = st.session_state.test_result
        st.metric(label=f"{test_type} Statistic", value=f"{res.get('statistic', 0):.4f}")
        st.metric(label="P-value", value=f"{res.get('p_value', 0):.4f}")
        if res.get('p_value', 1) < 0.05:
            st.success("The result is statistically significant (p < 0.05).")
        else:
            st.warning("The result is not statistically significant (p >= 0.05).")

    st.markdown("**3. Visualize Comparison**")
    if feature_to_test and len(groups_to_compare) > 1:
        import plotly.express as px
        fig = px.box(df[df['group'].isin(groups_to_compare)], x='group', y=feature_to_test, points="all",
                     title=f'Comparison of {feature_to_test} across groups',
                     labels={'group': 'Group', feature_to_test: feature_to_test})
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.box_fig = fig

def show_reports_page():
    """Display the reports generation page"""
    st.header("📄 Reports")

    if st.session_state.report_generator is None:
        st.error("❌ Report Generator module is not available.")
        return

    if st.session_state.get('population_data') is None:
        st.warning("⚠️ No population data to report. Please run a population analysis first.")
        return

    st.markdown("Generate a comprehensive HTML report of your analysis.")

    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating Report..."):
            report_content = st.session_state.report_generator.generate_report(
                population_data=st.session_state.population_data,
                statistics_results=st.session_state.get('test_result'),
                dist_fig=st.session_state.get('dist_fig'),
                corr_fig=st.session_state.get('corr_fig'),
                box_fig=st.session_state.get('box_fig'),
                filename=st.session_state.current_file
            )
            st.session_state.report_content = report_content
            st.success("Report generated successfully!")

    if st.session_state.get('report_content'):
        st.subheader("Report Preview")
        st.components.v1.html(st.session_state.report_content, height=600, scrolling=True)
        
        st.download_button(
            label="⬇️ Download Report",
            data=st.session_state.report_content,
            file_name="analysis_report.html",
            mime="text/html"
        )


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
        if np.asarray(st.session_state.image).ndim == 3:
            st.session_state.visualizer.display_interactive_3d_volume(st.session_state.image)
        else:
            st.session_state.visualizer.display_2d_slice(st.session_state.image)
    
    # ... (Visualization for other analysis types) ...

def show_batch_processing_page():
    """Display the batch processing page"""
    st.header("⚙️ Batch Processing")
    render_batch_controls()

def render_batch_controls():
    """Renders controls for batch processing."""
    if not batch_processing:
        st.error("Batch processing module not available.")
        return

    st.markdown("Run an enhancement + analysis pipeline over all supported images in a folder.")
    default_in = str(Path.cwd())
    default_out = str(Path.cwd() / "batch_output")
    input_dir = st.text_input("Input directory", default_in)
    output_dir = st.text_input("Output directory", default_out)

    enhancement_methods = []
    if st.session_state.ai_enhancer is not None:
        enhancement_methods = st.session_state.ai_enhancer.get_available_methods()

    if not enhancement_methods:
        st.warning("No enhancement methods available. Install optional AI dependencies.")
        return

    enhancement_method = st.selectbox("Enhancement method", enhancement_methods, key="batch_enhancement_method")
    run_morphometrics = st.checkbox("Run morphometric analysis", value=True)
    voxel_size = st.text_input("Voxel size (z,y,x)", "1.0,0.5,0.5")

    if st.button("Run Batch Processing", type="primary", key="run_batch_processing"):
        try:
            voxel_values = tuple(float(x.strip()) for x in voxel_size.split(","))
            if len(voxel_values) != 3:
                raise ValueError("Voxel size must include exactly 3 comma-separated values.")
        except Exception as exc:
            st.error(f"Invalid voxel size: {exc}")
            return

        if not os.path.isdir(input_dir):
            st.error(f"Input directory does not exist: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)

        analysis_tasks = ['morphometrics'] if run_morphometrics else []
        processor = batch_processing.BatchProcessor(
            enhancement_method=enhancement_method,
            enhancement_params={},
            analysis_tasks=analysis_tasks,
            analysis_params={'voxel_size': voxel_values}
        )

        progress = st.progress(0)
        status_text = st.empty()

        def callback(i, total, message):
            total_safe = max(total, 1)
            pct = min(int((i / total_safe) * 100), 100)
            progress.progress(pct)
            status_text.text(message)

        with st.spinner("Running batch pipeline..."):
            processor.run(input_dir=input_dir, output_dir=output_dir, callback=callback)

        st.success("Batch processing completed.")
        consolidated = Path(output_dir) / "consolidated_morphometrics.csv"
        if consolidated.exists():
            df = pd.read_csv(consolidated)
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "Download consolidated morphometrics",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=consolidated.name,
                mime="text/csv"
            )


def _get_display_frame(image: np.ndarray) -> np.ndarray:
    """Return a representative 2D frame for display."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    return np.squeeze(arr)

def render_colocalization_controls(context='main'):
    """Renders controls for colocalization analysis."""
    st.subheader("Colocalization")
    if not isinstance(st.session_state.get('analysis_results'), dict):
        st.session_state.analysis_results = {}
    analyzer = st.session_state.get('analyzer')
    if analyzer is None:
        st.info("Core analysis module unavailable for colocalization.")
        return

    channel_count = int(st.session_state.get('channel_count', 1))
    file_buffer = st.session_state.get('uploaded_file_buffer')
    if channel_count < 2 or file_buffer is None:
        st.info("Colocalization requires a loaded multi-channel image.")
        return

    col1, col2 = st.columns(2)
    with col1:
        ch1 = st.selectbox("Channel 1", list(range(channel_count)), index=0, key=f'{context}_colo_ch1')
    with col2:
        ch2_default = 1 if channel_count > 1 else 0
        ch2 = st.selectbox("Channel 2", list(range(channel_count)), index=ch2_default, key=f'{context}_colo_ch2')

    thresholds = st.columns(2)
    with thresholds[0]:
        threshold1 = st.number_input("Threshold channel 1", value=0.0, key=f'{context}_colo_thr1')
    with thresholds[1]:
        threshold2 = st.number_input("Threshold channel 2", value=0.0, key=f'{context}_colo_thr2')
    use_mask = st.checkbox("Use segmentation mask", value=True, key=f'{context}_colo_use_mask')

    if st.button("Run Colocalization", key=f'{context}_run_colocalization'):
        if ch1 == ch2:
            st.error("Select two different channels.")
            return
        try:
            file_buffer.seek(0)
            res1 = st.session_state.data_loader.load_image(file_buffer, channel=ch1)
            file_buffer.seek(0)
            res2 = st.session_state.data_loader.load_image(file_buffer, channel=ch2)
            if res1.get('status') != 'success' or res2.get('status') != 'success':
                st.error("Could not load channels for colocalization.")
                return

            mask = st.session_state.get('segmentation_mask') if use_mask else None
            result = analyzer.calculate_colocalization(
                channel1=res1['image_data'],
                channel2=res2['image_data'],
                mask=mask,
                threshold1=float(threshold1),
                threshold2=float(threshold2)
            )
            st.session_state.analysis_results['Colocalization'] = result
        except Exception as exc:
            st.error(f"Colocalization failed: {exc}")
            return

    result = st.session_state.analysis_results.get('Colocalization')
    if result and result.get('status') == 'success':
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson", f"{result.get('pearson_coefficient', 0):.4f}")
        c2.metric("Manders M1", f"{result.get('manders_m1', 0):.4f}")
        c3.metric("Manders M2", f"{result.get('manders_m2', 0):.4f}")
    elif result and result.get('status') in ('error', 'warning'):
        st.warning(result.get('message', 'Colocalization did not complete.'))

def render_ai_enhancement_controls(context='main'):
    """Renders AI enhancement controls, adaptable for different contexts."""
    enhancer = st.session_state.get('ai_enhancer')
    if enhancer is None:
        st.error("AI enhancer unavailable.")
        return

    methods = enhancer.get_available_methods()
    if not methods:
        st.warning("No enhancement methods are currently available.")
        return

    method = st.selectbox("Enhancement method", methods, key=f'{context}_ai_method')
    params: Dict[str, Any] = {}

    if "Non-local Means" in method:
        params['patch_size'] = st.slider("Patch size", 3, 11, 5, 2, key=f'{context}_nlm_patch')
        params['patch_distance'] = st.slider("Patch distance", 3, 15, 6, key=f'{context}_nlm_dist')
        params['fast_mode'] = st.checkbox("Fast mode", True, key=f'{context}_nlm_fast')
    elif "Richardson-Lucy" in method:
        params['iterations'] = st.slider("Iterations", 5, 80, 20, key=f'{context}_rl_iter')
        params['psf_size'] = st.slider("PSF size", 3, 15, 5, 2, key=f'{context}_rl_psf_size')
        params['psf_sigma'] = st.number_input("PSF sigma", min_value=0.1, value=1.0, key=f'{context}_rl_psf_sigma')
        if "Total Variation" in method:
            params['lambda_tv'] = st.number_input("TV regularization", min_value=0.0001, value=0.002, format="%.4f", key=f'{context}_rl_tv')
    elif "Cellpose" in method:
        params['diameter'] = st.number_input("Object diameter (0 = auto)", min_value=0.0, value=0.0, key=f'{context}_cp_diam')
        params['use_gpu'] = st.checkbox("Use GPU", value=False, key=f'{context}_cp_gpu')
    elif "StarDist" in method:
        params['prob_thresh'] = st.slider("Probability threshold", 0.1, 0.99, 0.5, key=f'{context}_sd_prob')
        params['nms_thresh'] = st.slider("NMS threshold", 0.1, 0.99, 0.4, key=f'{context}_sd_nms')

    if st.button("Run Enhancement", type="primary", key=f'{context}_run_enhancement'):
        with st.spinner(f"Running {method}..."):
            result = enhancer.enhance_image(st.session_state.image, method, params)
        st.session_state.enhanced_result = result

        if result.get('status') == 'success':
            if 'enhanced_image' in result:
                st.success("Enhancement completed.")
            if 'segmentation_masks' in result:
                st.session_state.segmentation_mask = (np.asarray(result['segmentation_masks']) > 0).astype(np.uint8)
                st.success("Segmentation mask generated.")
        else:
            st.error(result.get('message', 'Enhancement failed.'))

    result = st.session_state.get('enhanced_result')
    if not result or result.get('status') != 'success':
        return

    st.divider()
    st.subheader("Enhancement Results")
    if 'enhanced_image' in result:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Original")
            st.image(_get_display_frame(st.session_state.image), use_container_width=True, clamp=True)
        with col2:
            st.caption("Enhanced")
            st.image(_get_display_frame(result['enhanced_image']), use_container_width=True, clamp=True)
        if st.button("Use enhanced image as current input", key=f'{context}_use_enhanced'):
            st.session_state.image = np.asarray(result['enhanced_image'])
            st.success("Current image replaced with enhanced image.")

    if st.session_state.get('segmentation_mask') is not None:
        st.caption("Segmentation mask")
        st.image(np.asarray(st.session_state.segmentation_mask), use_container_width=True, clamp=True)

def render_analysis_controls(context='main'):
    """Renders controls for analysis methods."""
    image = st.session_state.get('image')
    if not isinstance(st.session_state.get('analysis_results'), dict):
        st.session_state.analysis_results = {}
    if image is None:
        st.warning("No image loaded.")
        return

    options = [
        "Number & Brightness",
        "Pair Correlation Function",
        "Segmented FCS",
        "Optical Flow",
        "Image Correlation Spectroscopy",
        "Displacement Correlation Spectroscopy",
        "Material Mechanics",
        "Deformation Microscopy",
        "Two-Domain Elastography",
        "Nuclear Biophysics"
    ]
    selected = st.selectbox("Analysis tool", options, key=f'{context}_analysis_tool')

    result = None
    if selected == "Number & Brightness":
        window_size = st.slider("Smoothing window", 1, 7, 1, key=f'{context}_nb_window')
        if st.button("Run N&B", key=f'{context}_run_nb'):
            if image.ndim != 3:
                st.error("N&B requires a 3D stack (time, y, x).")
            elif st.session_state.nb_analyzer is None:
                st.error("N&B module unavailable.")
            else:
                result = st.session_state.nb_analyzer.analyze(image, window_size=window_size)

    elif selected == "Pair Correlation Function":
        max_radius = st.slider("Max radius", 5, 200, 50, key=f'{context}_pcf_radius')
        if st.button("Run PCF", key=f'{context}_run_pcf'):
            if image.ndim == 3:
                work_image = np.mean(image, axis=0)
            else:
                work_image = image
            if work_image.ndim != 2:
                st.error("PCF requires a 2D image.")
            elif st.session_state.pcf_analyzer is None:
                st.error("PCF module unavailable.")
            else:
                result = st.session_state.pcf_analyzer.analyze(work_image, max_radius=max_radius)

    elif selected == "Segmented FCS":
        dt = st.number_input("Sampling interval dt (s)", min_value=1e-6, value=0.001, format="%.6f", key=f'{context}_segfcs_dt')
        model = st.selectbox("Model", ["2D", "3D", "anomalous"], index=1, key=f'{context}_segfcs_model')
        window_s = st.number_input("Window length (s)", min_value=0.1, value=5.0, key=f'{context}_segfcs_window')
        step_s = st.number_input("Step (s)", min_value=0.1, value=2.5, key=f'{context}_segfcs_step')
        if st.button("Run Segmented FCS", key=f'{context}_run_segfcs'):
            analyzer = st.session_state.get('segmented_fcs_analyzer')
            if analyzer is None:
                st.error("Segmented FCS module unavailable.")
            else:
                if image.ndim == 3:
                    intensity = np.mean(image, axis=(1, 2))
                elif image.ndim == 2:
                    intensity = np.mean(image, axis=0)
                else:
                    intensity = np.asarray(image).reshape(-1)
                result = analyzer.analyze(intensity, dt=float(dt), model=model, window_s=float(window_s), step_s=float(step_s))

    elif selected == "Optical Flow":
        analyzer = st.session_state.get('optical_flow_analyzer')
        if analyzer is None:
            st.error("Optical flow module unavailable.")
        else:
            method = st.selectbox("Optical flow method", analyzer.get_available_methods(), key=f'{context}_of_method')
            if st.button("Run Optical Flow", key=f'{context}_run_of'):
                if image.ndim != 3:
                    st.error("Optical flow requires a time sequence (T, Y, X).")
                else:
                    result = analyzer.analyze_optical_flow(method, image, {})

    elif selected == "Image Correlation Spectroscopy":
        analyzer = st.session_state.get('ics_analyzer')
        if analyzer is None:
            st.error("ICS module unavailable.")
        else:
            method = st.selectbox("ICS method", analyzer.get_available_methods(), key=f'{context}_ics_method')
            if st.button("Run ICS", key=f'{context}_run_ics'):
                result = analyzer.analyze_ics_method(method, image, {})

    elif selected == "Displacement Correlation Spectroscopy":
        if st.button("Run DCS", key=f'{context}_run_dcs'):
            analyzer = st.session_state.get('dcs_analyzer')
            if analyzer is None:
                st.error("DCS module unavailable.")
            elif image.ndim != 3:
                st.error("DCS requires a time sequence (T, Y, X).")
            else:
                mask = st.session_state.get('segmentation_mask')
                result = analyzer.analyze(image_data=image, nuclear_mask=mask, parameters={})

    elif selected == "Material Mechanics":
        analyzer = st.session_state.get('material_mechanics_analyzer')
        if analyzer is None or material_mechanics is None:
            st.error("Material mechanics module unavailable.")
        else:
            voxel = st.session_state.get('voxel_size')
            default_px = 0.1
            if isinstance(voxel, (tuple, list, np.ndarray)) and len(voxel) > 0:
                try:
                    default_px = float(voxel[-1])
                except Exception:
                    default_px = 0.1
            elif isinstance(voxel, (float, int)):
                default_px = float(voxel)

            col1, col2 = st.columns(2)
            with col1:
                pixel_size_um = st.number_input(
                    "Pixel size (um/pixel)",
                    min_value=0.001,
                    value=float(default_px),
                    format="%.4f",
                    key=f'{context}_mm_px',
                )
            with col2:
                time_interval_s = st.number_input(
                    "Frame interval (s)",
                    min_value=0.001,
                    value=1.0,
                    format="%.4f",
                    key=f'{context}_mm_dt',
                )

            run_force = st.checkbox("Force distribution (flow divergence + shear)", value=True, key=f'{context}_mm_force')
            run_stiffness = st.checkbox("Stiffness proxy (correlation length xi)", value=True, key=f'{context}_mm_stiffness')
            run_texture = st.checkbox("Texture topology (GLCM entropy + fractal)", value=True, key=f'{context}_mm_texture')
            run_boundary = st.checkbox("Boundary mechanics (sigma/kappa)", value=True, key=f'{context}_mm_boundary')
            run_fusion = st.checkbox("Fusion kinetics (viscosity tau)", value=False, key=f'{context}_mm_fusion')
            use_segmentation_mask = st.checkbox(
                "Use segmentation mask guidance",
                value=st.session_state.get('segmentation_mask') is not None,
                key=f'{context}_mm_use_mask',
            )

            with st.expander("Advanced Material Mechanics Parameters"):
                farneback_winsize = st.slider("Farneback window size", 5, 45, 15, step=2, key=f'{context}_mm_of_win')
                glcm_window = st.slider("GLCM window", 6, 48, 16, step=2, key=f'{context}_mm_glcm_window')
                fusion_percentile = st.slider("Fusion auto-mask percentile", 70.0, 99.9, 94.0, step=0.1, key=f'{context}_mm_fusion_pct')
                boundary_n_angles = st.select_slider(
                    "Boundary angular samples",
                    options=[32, 64, 96, 128, 192, 256, 384, 512],
                    value=128,
                    key=f'{context}_mm_boundary_angles',
                )

            if st.button("Run Material Mechanics", key=f'{context}_run_mm'):
                if image.ndim != 3:
                    st.error("Material mechanics requires a time sequence (T, Y, X).")
                else:
                    mask = st.session_state.get('segmentation_mask') if use_segmentation_mask else None
                    if run_fusion and mask is not None and np.asarray(mask).ndim in (2, 3):
                        compartment_mask = mask
                    else:
                        compartment_mask = None

                    st.session_state.material_mechanics_analyzer = material_mechanics.MaterialMechanics(
                        pixel_size_um=float(pixel_size_um),
                        time_interval_s=float(time_interval_s),
                    )
                    result = st.session_state.material_mechanics_analyzer.analyze(
                        image_data=image,
                        nuclear_mask=mask,
                        compartment_mask=compartment_mask,
                        parameters={
                            'run_force_distribution': run_force,
                            'run_stiffness_proxy': run_stiffness,
                            'run_texture_topology': run_texture,
                            'run_boundary_mechanics': run_boundary,
                            'run_fusion_kinetics': run_fusion,
                            'farneback_winsize': int(farneback_winsize),
                            'glcm_window': int(glcm_window),
                            'fusion_percentile': float(fusion_percentile),
                            'boundary_n_angles': int(boundary_n_angles),
                        }
                    )

    elif selected == "Deformation Microscopy":
        if st.button("Run Deformation Microscopy", key=f'{context}_run_dm'):
            analyzer = st.session_state.get('dm_analyzer')
            if analyzer is None:
                st.error("Deformation Microscopy module unavailable.")
            else:
                if image.ndim == 3 and image.shape[0] >= 2:
                    template_image, target_image = image[0], image[-1]
                elif image.ndim == 2 and st.session_state.get('enhanced_result', {}).get('enhanced_image') is not None:
                    template_image = image
                    target_image = np.asarray(st.session_state.enhanced_result['enhanced_image'])
                else:
                    st.error("DM needs two comparable 2D frames (time stack or enhanced image).")
                    template_image = None
                    target_image = None
                if template_image is not None and target_image is not None:
                    result = analyzer.analyze(template_image=template_image, target_image=target_image, parameters={})

    elif selected == "Two-Domain Elastography":
        if st.button("Run Elastography", key=f'{context}_run_elasto'):
            analyzer = st.session_state.get('elastography_analyzer')
            dm_result = st.session_state.analysis_results.get("Deformation Microscopy")
            if analyzer is None:
                st.error("Elastography module unavailable.")
            elif not dm_result or dm_result.get('status') != 'success':
                st.error("Run Deformation Microscopy first to generate displacement fields.")
            else:
                fluorescence = np.mean(image, axis=0) if image.ndim == 3 else image
                disp = dm_result.get('displacement', {})
                result = analyzer.analyze(
                    fluorescence_image=fluorescence,
                    displacement_x=disp.get('displacement_x'),
                    displacement_y=disp.get('displacement_y'),
                    nuclear_mask=st.session_state.get('segmentation_mask'),
                    parameters={}
                )

    elif selected == "Nuclear Biophysics":
        analyzer = st.session_state.get('nuclear_analyzer')
        if analyzer is None:
            st.error("Nuclear biophysics module unavailable.")
        else:
            method = st.selectbox(
                "Nuclear method",
                ["Binding (FCS-like)", "Chromatin dynamics", "Nuclear elasticity"],
                key=f'{context}_nuclear_method'
            )
            if st.button("Run Nuclear Analysis", key=f'{context}_run_nuclear'):
                if image.ndim != 3:
                    st.error("Nuclear analyses require a time sequence (T, Y, X).")
                else:
                    mask = st.session_state.get('segmentation_mask')
                    if mask is None:
                        st.error("A nuclear mask is required. Generate a segmentation mask first.")
                    elif method == "Binding (FCS-like)":
                        result = analyzer.analyze_nuclear_binding(image, mask, parameters={})
                    elif method == "Chromatin dynamics":
                        result = analyzer.analyze_chromatin_dynamics(image, mask, parameters={})
                    else:
                        force_frame = max(image.shape[0] // 2, 1)
                        result = analyzer.analyze_nuclear_elasticity(image, force_application_time=force_frame, parameters={})

    if result is not None:
        st.session_state.analysis_results[selected] = result

    latest = st.session_state.analysis_results.get(selected)
    if latest is None:
        return

    st.divider()
    status = latest.get('status', 'unknown')
    if status != 'success':
        st.warning(latest.get('message', f"{selected} returned status: {status}"))
        return

    st.success(f"{selected} completed successfully.")

    if selected == "Number & Brightness":
        c1, c2 = st.columns(2)
        c1.image(latest['number_map'], caption="Number map", use_container_width=True, clamp=True)
        c2.image(latest['brightness_map'], caption="Brightness map", use_container_width=True, clamp=True)
    elif selected == "Pair Correlation Function":
        pcf_df = pd.DataFrame({"radius": latest['radius'], "pcf": latest['pcf']})
        st.line_chart(pcf_df.set_index("radius"))
    elif selected == "Segmented FCS":
        seg_df = latest.get('segments', pd.DataFrame())
        st.dataframe(seg_df, use_container_width=True)
    elif selected == "Optical Flow":
        st.metric("Average displacement", f"{latest.get('avg_displacement', 0):.4f}")
        st.metric("Max displacement", f"{latest.get('max_displacement', 0):.4f}")
    elif selected == "Image Correlation Spectroscopy":
        for key in ('diffusion_coefficient', 'number_of_particles', 'fit_quality'):
            if key in latest:
                st.write(f"{key}: {latest[key]}")
    elif selected == "Displacement Correlation Spectroscopy":
        summary = latest.get('summary', {})
        if summary:
            st.json(summary)
    elif selected == "Material Mechanics":
        all_results = latest.get('results', {})
        summary = latest.get('summary', {})

        successful = summary.get('successful_components', [])
        failed = summary.get('failed_components', [])
        if successful:
            st.caption("Successful components: " + ", ".join(successful))
        if failed:
            st.caption("Failed components: " + ", ".join(failed))

        force_result = all_results.get('force_distribution', {})
        if force_result.get('status') == 'success':
            st.subheader("Internal Force Distribution")
            force_summary = force_result.get('summary', {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean divergence (1/s)", f"{force_summary.get('mean_divergence_per_s', np.nan):.4f}")
            m2.metric("Positive divergence fraction", f"{force_summary.get('positive_divergence_fraction', np.nan):.3f}")
            m3.metric("Mean abs shear (1/s)", f"{force_summary.get('mean_abs_shear_rate_per_s', np.nan):.4f}")
            i1, i2 = st.columns(2)
            with i1:
                st.image(force_result.get('divergence_map'), caption="Divergence map", use_container_width=True, clamp=True)
            with i2:
                st.image(force_result.get('shear_rate_map'), caption="Shear-rate map", use_container_width=True, clamp=True)

        stiffness_result = all_results.get('stiffness_proxy', {})
        if stiffness_result.get('status') == 'success':
            st.subheader("Stiffness Proxy")
            xi_um = stiffness_result.get('correlation_length_um', np.nan)
            st.metric("Correlation length xi (um)", f"{xi_um:.3f}" if np.isfinite(xi_um) else "n/a")
            st.caption(stiffness_result.get('interpretation', ''))
            radial_x = stiffness_result.get('radial_distances_um')
            radial_y = stiffness_result.get('radial_correlation')
            if radial_x is not None and radial_y is not None:
                curve_df = pd.DataFrame({"distance_um": radial_x, "correlation": radial_y})
                st.line_chart(curve_df.set_index("distance_um"))

        texture_result = all_results.get('texture_topology', {})
        if texture_result.get('status') == 'success':
            st.subheader("Texture Topology")
            entropy = texture_result.get('entropy', {})
            fractal = texture_result.get('fractal', {})
            t1, t2 = st.columns(2)
            with t1:
                st.metric("Mean GLCM entropy", f"{entropy.get('mean_entropy', np.nan):.3f}")
                if entropy.get('entropy_map') is not None:
                    st.image(entropy.get('entropy_map'), caption="Local GLCM entropy", use_container_width=True, clamp=True)
            with t2:
                fractal_dim = fractal.get('global_fractal_dimension', np.nan)
                st.metric("Fractal dimension", f"{fractal_dim:.3f}" if np.isfinite(fractal_dim) else "n/a")
                scan = fractal.get('scan', [])
                if scan:
                    st.dataframe(pd.DataFrame(scan), use_container_width=True)
            st.caption(texture_result.get('interpretation', ''))

        fusion_result = all_results.get('fusion_kinetics', {})
        if fusion_result.get('status') == 'success':
            st.subheader("Fusion Kinetics")
            st.metric("Mean tau (s)", f"{fusion_result.get('mean_tau_s', np.nan):.3f}")
            events = fusion_result.get('events', [])
            if events:
                event_df = pd.DataFrame(
                    [
                        {
                            "fusion_frame": ev.get("fusion_frame"),
                            "num_parents": ev.get("num_parents"),
                            "tau_s": ev.get("tau_s"),
                            "fit_r_squared": ev.get("fit_r_squared"),
                        }
                        for ev in events
                    ]
                )
                st.dataframe(event_df, use_container_width=True)
        elif fusion_result.get('status') == 'warning':
            st.info(fusion_result.get('message', 'No fusion events found.'))

        boundary_result = all_results.get('boundary_mechanics', {})
        if boundary_result.get('status') == 'success':
            st.subheader("Boundary Mechanics")
            b1, b2, b3 = st.columns(3)
            sigma_val = boundary_result.get('surface_tension_sigma_N_per_m', np.nan)
            kappa_val = boundary_result.get('bending_rigidity_kappa_J', np.nan)
            kappa_kbt = boundary_result.get('bending_rigidity_kappa_over_kBT', np.nan)
            b1.metric("Sigma (N/m)", f"{sigma_val:.3e}" if np.isfinite(sigma_val) else "n/a")
            b2.metric("Kappa (J)", f"{kappa_val:.3e}" if np.isfinite(kappa_val) else "n/a")
            b3.metric("Kappa / kBT", f"{kappa_kbt:.2f}" if np.isfinite(kappa_kbt) else "n/a")
            modes = boundary_result.get('mode_numbers')
            power = boundary_result.get('power_spectrum')
            if modes is not None and power is not None:
                psd_df = pd.DataFrame({"mode": modes, "power": power})
                st.line_chart(psd_df.set_index("mode"))
        elif boundary_result.get('status') == 'warning':
            st.info(boundary_result.get('message', 'Boundary mechanics did not return a fit.'))
    elif selected == "Deformation Microscopy":
        strain = latest.get('strain', {})
        st.write("Quality metrics:", latest.get('quality_metrics', {}))
        if strain.get('hydrostatic_strain') is not None:
            st.image(strain['hydrostatic_strain'], caption="Hydrostatic strain", use_container_width=True, clamp=True)
    elif selected == "Two-Domain Elastography":
        st.metric("Stiffness ratio (Eh/Ee)", f"{latest.get('stiffness_ratio', 0):.4f}")
    elif selected == "Nuclear Biophysics":
        st.write("Result keys:", ", ".join(latest.keys()))

# Application entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page or contact support if the problem persists.")
