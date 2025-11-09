
import streamlit as st
import numpy as np
from ai_enhancement import AIEnhancementManager, get_enhancement_parameters
from visualization import VisualizationManager
from analysis import AnalysisManager
from data_loader import DataLoader
import io

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

# ... (rest of the app.py file remains largely the same) ...

def render_ai_enhancement_controls(context='main'):
    """Renders AI enhancement controls with more options."""
    if 'ai_enhancer' not in st.session_state:
        st.session_state.ai_enhancer = AIEnhancementManager()

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
    defaults = get_enhancement_parameters(enhancement_method)

    if enhancement_method == 'Non-local Means Denoising':
        st.subheader("Denoising Parameters")
        parameters['patch_size'] = st.slider("Patch Size", 3, 15, defaults.get('patch_size', 5), key=f"{key_prefix}_patch_size")
        parameters['patch_distance'] = st.slider("Patch Distance", 3, 15, defaults.get('patch_distance', 6), key=f"{key_prefix}_patch_distance")
        parameters['auto_sigma'] = st.checkbox("Automatically estimate noise", value=defaults.get('auto_sigma', True), key=f"{key_prefix}_auto_sigma")
        parameters['h'] = st.number_input("Denoising strength (h)", 0.01, 1.0, defaults.get('h', 0.1), 0.01, disabled=parameters['auto_sigma'], key=f"{key_prefix}_h_value")

    elif enhancement_method in ['Richardson-Lucy Deconvolution', 'Richardson-Lucy with Total Variation', 'FISTA Deconvolution', 'ISTA Deconvolution', 'Iterative Constraint Tikhonov-Miller']:
        # ... (deconvolution parameters) ...
        pass # Simplified for brevity

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

def render_visualization_controls(context='main'):
    """Renders visualization controls with interactive 3D volume option."""
    if 'visualizer' not in st.session_state: st.session_state.visualizer = VisualizationManager()
    # ... (visualization controls) ...
    pass # Simplified for brevity

def render_analysis_controls(context='main'):
    """Renders analysis controls for segmentation results."""
    if 'analyzer' not in st.session_state: st.session_state.analyzer = AnalysisManager()
    key_prefix = f"analysis_{context}"

    with st.expander("Morphometric Analysis", expanded=True):
        # ... (morphometric analysis controls) ...
        pass # Simplified

    with st.expander("Percolation Analysis", expanded=False):
        # ... (percolation analysis controls) ...
        pass # Simplified

    with st.expander("Colocalization Analysis", expanded=False):
        render_colocalization_controls(context)

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("AI-Powered Microscopy Image Analysis Platform")
    
    # Init session state
    state_vars = {
        'image': None, 'voxel_size': (1.0, 0.5, 0.5), 'enhanced_result': None,
        'segmentation_mask': None, 'analysis_results': None, 'percolation_results': None,
        'colocalization_results': None, 'channel_count': 1, 'uploaded_file_buffer': None
    }
    for var, val in state_vars.items():
        if var not in st.session_state: st.session_state[var] = val
    
    if 'data_loader' not in st.session_state: st.session_state.data_loader = DataLoader()
    if 'visualizer' not in st.session_state: st.session_state.visualizer = VisualizationManager()
    if 'analyzer' not in st.session_state: st.session_state.analyzer = AnalysisManager()
    if 'ai_enhancer' not in st.session_state: st.session_state.ai_enhancer = AIEnhancementManager()

    with st.sidebar:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Choose an image file", type=['tif', 'tiff', 'png', 'jpg', 'czi', 'lif', 'nd2'])
        if uploaded_file is not None:
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

    # Main layout (simplified)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Controls")
        render_ai_enhancement_controls('main')
        render_analysis_controls('main')
    with col2:
        st.header("Visualization")
        render_visualization_controls('main')
