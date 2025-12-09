"""
Streamlit UI for Allen Cell Segmenter Integration
"""

import streamlit as st
import numpy as np
from .backends import get_segmenter_backend, ClassicSegmenterBackend, MlSegmenterBackend
from .utils import supports_segmenter_ml

def show_allen_segmenter_page():
    """Display the Allen Cell Segmenter interface."""
    st.header("🧬 Allen Cell & Structure Segmenter")
    
    st.markdown("""
    Integrates the [Allen Cell & Structure Segmenter](https://www.allencell.org/segmenter.html) 
    for 3D intracellular structure segmentation.
    """)

    if 'image_data' not in st.session_state or st.session_state.image_data is None:
        st.warning("Please load an image in the 'Data Loading' tab first.")
        return

    # Backend Selection
    backend_mode = st.radio(
        "Select Backend",
        ["Classic", "ML (Deep Learning)"],
        help="Classic: Tuned workflows. ML: Pre-trained deep learning models."
    )
    
    backend_type = "classic" if backend_mode.startswith("Classic") else "ml"
    
    # Check ML support
    if backend_type == "ml" and not supports_segmenter_ml():
        st.error("ML backend is not supported in this environment (requires PyTorch + CUDA). Switching to Classic.")
        backend_type = "classic"

    # Instantiate Backend
    try:
        backend = get_segmenter_backend(backend_type)
    except Exception as e:
        st.error(f"Failed to initialize backend: {e}")
        return

    # Structure Selection
    try:
        available_structures = backend.get_available_structures()
    except Exception as e:
        st.error(f"Failed to retrieve structures: {e}")
        available_structures = []

    if not available_structures:
        st.warning("No structures/models found. Please check your installation.")
        return

    selected_structure = st.selectbox("Select Structure / Model", available_structures)

    # Configuration
    st.subheader("Configuration")
    with st.expander("Advanced Parameters"):
        # Placeholder for parameter overrides
        # In a full implementation, we would inspect the workflow config and generate widgets
        st.info("Parameter overrides can be defined here in future versions.")
        config_text = st.text_area("JSON Configuration (Optional)", "{}")
    
    import json
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        st.error("Invalid JSON configuration.")
        config = {}

    # Run Segmentation
    if st.button("Run Segmentation"):
        with st.spinner(f"Segmenting {selected_structure}..."):
            try:
                # Get image from session state
                # Assuming st.session_state.image_data is the numpy array
                # We might need to handle different dimensions here or in the backend
                image = st.session_state.image_data
                
                # Run
                mask = backend.segment(
                    image, 
                    structure_id=selected_structure, 
                    config=config
                )
                
                # Store result
                st.session_state['segmentation_result'] = mask
                st.success("Segmentation complete!")
                
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                import traceback
                st.text(traceback.format_exc())

    # Visualization
    if 'segmentation_result' in st.session_state:
        st.subheader("Results")
        
        mask = st.session_state['segmentation_result']
        image = st.session_state.image_data
        
        # Simple visualization: Middle slice
        if mask.ndim == 3:
            z_mid = mask.shape[0] // 2
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image[z_mid], caption="Original (Middle Slice)", clamp=True, use_column_width=True)
            with col2:
                st.image(mask[z_mid].astype(float), caption="Segmentation Mask", clamp=True, use_column_width=True)
                
            st.info(f"Mask Shape: {mask.shape}, Non-zero pixels: {np.count_nonzero(mask)}")
        else:
            st.write("Result is not 3D, visualization might need adjustment.")
