
def render_ai_enhancement_controls(context='main'):
    """Renders AI enhancement controls in different contexts (main or sidebar)."""
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

    # Simplified parameter controls for this example
    if enhancement_method == 'Non-local Means Denoising':
        patch_size = st.slider("Patch Size", 3, 15, 5, key=f"{key_prefix}_patch_size")
        parameters = {'patch_size': patch_size}
    else:
        parameters = {}

    if st.button(f"🎨 Enhance Image", key=f"{key_prefix}_run"):
        # ... (enhancement logic) ...
        pass

# In the main app layout:
# with st.sidebar:
#    render_ai_enhancement_controls(context='sidebar')

# In the 'AI Enhancement' tab:
# render_ai_enhancement_controls(context='main')
