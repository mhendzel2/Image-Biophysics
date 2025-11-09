
import streamlit as st
import numpy as np
from ai_enhancement import AIEnhancementManager, get_enhancement_parameters
from visualization import VisualizationManager
from analysis import AnalysisManager
from data_loader import DataLoader
from batch_processing import BatchProcessor
import io
import os

def render_batch_controls():
    """Renders controls for batch processing."""
    st.header("Batch Processing")

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

        processor = BatchProcessor(
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
        # ... (colocalization controls as before) ...
        pass

def render_ai_enhancement_controls(context='main'):
    """Renders AI enhancement controls, adaptable for different contexts."""
    if 'ai_enhancer' not in st.session_state: st.session_state.ai_enhancer = AIEnhancementManager()

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
    # ... (the rest of the enhancement controls as before)
    # Note: A robust implementation would ensure that parameters for different
    # contexts (main vs. batch) are managed separately in the session state.
    pass

# ... (other render functions as before) ...

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("AI-Powered Microscopy Image Analysis Platform")

    # Initialize session state variables
    # ... (as before)

    tab1, tab2 = st.tabs(["Interactive Analysis", "Batch Processing"])

    with tab1:
        # All the previous UI for interactive analysis goes here
        col1, col2 = st.columns([1, 2])
        with col1:
            st.header("Controls")
            with st.sidebar:
                st.header("Image Upload")
                # ... (upload logic) ...
            render_ai_enhancement_controls('main')
            render_analysis_controls('main')
        with col2:
            st.header("Visualization")
            render_visualization_controls('main')
    
    with tab2:
        render_batch_controls()
