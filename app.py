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
displacement_correlation_spectroscopy = safe_import('displacement_correlation_spectroscopy', 'DCS Analysis')
deformation_microscopy = safe_import('deformation_microscopy', 'Deformation Microscopy')
two_domain_elastography = safe_import('two_domain_elastography', 'Two-Domain Elastography')
optical_flow_analysis = safe_import('optical_flow_analysis', 'Optical Flow Analysis')
image_correlation_spectroscopy = safe_import('image_correlation_spectroscopy', 'Image Correlation Spectroscopy')
nuclear_biophysics = safe_import('nuclear_biophysics', 'Nuclear Biophysics')
thumbnail_generator = safe_import('thumbnail_generator', 'Thumbnail Generator')
batch_processing = safe_import('batch_processing', 'Batch Processing')
number_and_brightness = safe_import('number_and_brightness', 'Number & Brightness')
pair_correlation_function = safe_import('pair_correlation_function', 'Pair Correlation Function')
population_analysis = safe_import('population_analysis', 'Population Analysis')
statistics_analyzer = safe_import('statistics_analyzer', 'Statistics Analyzer')
displacement_correlation_spectroscopy = safe_import('displacement_correlation_spectroscopy', 'Displacement Correlation Spectroscopy')
deformation_microscopy = safe_import('deformation_microscopy', 'Deformation Microscopy')
two_domain_elastography = safe_import('two_domain_elastography', 'Two-Domain Elastography')


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
            "Pair Correlation Function": pair_correlation_function is not None,
            "Population Analysis": population_analysis is not None,
            "Statistics Analyzer": statistics_analyzer is not None
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
                            st.session_state[key] = None
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
        overlay = st.session_state.get('segmentation_mask')
        st.session_state.visualizer.display_interactive_3d_volume(st.session_state.image, overlay=overlay)
    
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

def render_colocalization_controls(context='main'):
    """Renders controls for colocalization analysis."""

def render_ai_enhancement_controls(context='main'):
    """Renders AI enhancement controls, adaptable for different contexts."""

def render_analysis_controls(context='main'):
    """Renders controls for analysis methods."""

# Application entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page or contact support if the problem persists.")
