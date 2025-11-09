
import streamlit as st
from ai_enhancement import AIEnhancementManager, get_enhancement_parameters
from visualization import VisualizationManager

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
    if enhancement_method == 'Non-local Means Denoising':
        st.subheader("Denoising Parameters")
        defaults = get_enhancement_parameters(enhancement_method)
        
        patch_size = st.slider("Patch Size", 3, 15, defaults.get('patch_size', 5), key=f"{key_prefix}_patch_size")
        patch_distance = st.slider("Patch Distance", 3, 15, defaults.get('patch_distance', 6), key=f"{key_prefix}_patch_distance")
        auto_sigma = st.checkbox("Automatically estimate noise", value=defaults.get('auto_sigma', True), key=f"{key_prefix}_auto_sigma")
        h_value = st.number_input("Denoising strength (h)", 0.01, 1.0, defaults.get('h', 0.1), 0.01, disabled=auto_sigma, key=f"{key_prefix}_h_value")
        
        parameters = {
            'patch_size': patch_size, 
            'patch_distance': patch_distance, 
            'auto_sigma': auto_sigma,
            'h': h_value
        }

    elif enhancement_method == 'Richardson-Lucy Deconvolution':
        st.subheader("Deconvolution Parameters")
        defaults = get_enhancement_parameters(enhancement_method)
        
        iterations = st.slider("Iterations", 1, 100, defaults.get('iterations', 30), key=f"{key_prefix}_iterations")
        psf_size = st.slider("PSF Size", 3, 21, defaults.get('psf_size', 5), step=2, key=f"{key_prefix}_psf_size")
        psf_sigma = st.slider("PSF Sigma", 0.1, 5.0, defaults.get('psf_sigma', 1.0), 0.1, key=f"{key_prefix}_psf_sigma")
        
        parameters = {'iterations': iterations, 'psf_size': psf_size, 'psf_sigma': psf_sigma}

    elif enhancement_method == 'Richardson-Lucy with Total Variation':
        st.subheader("Deconvolution with TV Regularization")
        defaults = get_enhancement_parameters(enhancement_method)
        
        iterations = st.slider("Iterations", 1, 50, defaults.get('iterations', 10), key=f"{key_prefix}_tv_iterations")
        lambda_tv = st.slider("Regularization (lambda)", 0.0001, 0.1, defaults.get('lambda_tv', 0.002), 0.0001, format="%.4f", key=f"{key_prefix}_lambda_tv")
        psf_size = st.slider("PSF Size", 3, 21, defaults.get('psf_size', 5), step=2, key=f"{key_prefix}_tv_psf_size")
        psf_sigma = st.slider("PSF Sigma", 0.1, 5.0, defaults.get('psf_sigma', 1.0), 0.1, key=f"{key_prefix}_tv_psf_sigma")
        
        parameters = {
            'iterations': iterations, 
            'lambda_tv': lambda_tv,
            'psf_size': psf_size,
            'psf_sigma': psf_sigma
        }

    elif 'Cellpose' in enhancement_method:
        st.subheader("Cellpose Parameters")
        defaults = get_enhancement_parameters(enhancement_method)
        
        diameter = st.number_input("Cell Diameter (pixels)", value=float(defaults.get('diameter', 30.0)), min_value=0.0, step=1.0, key=f"{key_prefix}_diameter")
        use_gpu = st.checkbox("Use GPU (if available)", value=defaults.get('use_gpu', False), key=f"{key_prefix}_use_gpu")
        
        parameters = {
            'diameter': diameter if diameter > 0 else None, 
            'use_gpu': use_gpu
        }

    elif enhancement_method == 'StarDist Nucleus Segmentation':
        st.subheader("StarDist Parameters")
        defaults = get_enhancement_parameters(enhancement_method)
        
        prob_thresh = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.05, key=f"{key_prefix}_stardist_prob")
        nms_thresh = st.slider("NMS Threshold", 0.0, 1.0, 0.3, 0.05, key=f"{key_prefix}_stardist_nms")
        
        parameters = {
            'prob_thresh': prob_thresh,
            'nms_thresh': nms_thresh
        }

    elif enhancement_method == 'AICS Cell Segmentation':
        st.subheader("AICS-Segmenter Parameters")
        defaults = get_enhancement_parameters(enhancement_method)
        
        aics_models = ['General', 'Lamin', 'Myo', 'Sox', 'Membrane', 'Sec61b']
        model_name = st.selectbox(
            "Select Model",
            options=aics_models,
            index=aics_models.index(defaults.get('model_name', 'General')),
            key=f"{key_prefix}_aics_model"
        )
        
        parameters = {'model_name': model_name}
        
    else:
        parameters = get_enhancement_parameters(enhancement_method)


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
                else:
                    st.error(f"Enhancement failed: {result.get('message', 'Unknown error')}")
        else:
            st.warning("Please load an image before applying enhancement.")
            

def render_visualization_controls(context='main'):
    """Renders visualization controls with 3D volume option."""
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = VisualizationManager()
        
    display_options = ["2D Image", "3D Volume"]
    display_choice = st.selectbox("Display As", options=display_options, key=f"vis_control_{context}")
    
    if display_choice == "3D Volume":
        if 'image' in st.session_state and st.session_state.image is not None:
            st.session_state.visualizer.display_3d_volume(st.session_state.image)
        else:
            st.warning("Please load a 3D image to use the volume display.")

if __name__ == '__main__':
    st.title("AI Enhancement and Visualization GUI")
    
    # Placeholder for image data
    if 'image' not in st.session_state:
        st.session_state.image = None 

    # --- Image Upload ---
    uploaded_file = st.file_uploader("Choose a TIFF file", type=["tif", "tiff"])
    if uploaded_file is not None:
        st.info("Image loaded (placeholder). Replace with actual image loading logic.")

    # --- Main App Layout ---
    st.header("Visualization")
    render_visualization_controls(context='main')
    
    st.header("AI Enhancement")
    render_ai_enhancement_controls(context='main')

    with st.sidebar:
        st.header("Visualization (Sidebar)")
        render_visualization_controls(context='sidebar')
        
        st.header("AI Enhancement (Sidebar)")
        render_ai_enhancement_controls(context='sidebar')
