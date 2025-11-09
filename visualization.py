"""
Visualization Module

Handles the display of 2D and 3D image data using Streamlit, Matplotlib, and Plotly.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

class VisualizationManager:
    """Manages the creation of various data visualizations."""

    def display_2d_slice(self, image_data: np.ndarray, title: str = "2D Slice"):
        """Displays a 2D slice of a 3D image or a 2D image."""
        if image_data.ndim == 3:
            # If 3D, show the middle slice
            display_img = image_data[image_data.shape[0] // 2]
        else:
            display_img = image_data
        
        st.image(display_img, caption=title, use_column_width=True)

    def display_interactive_3d_volume(self, image_data: np.ndarray):
        """Displays an interactive 3D volume rendering of a 3D numpy array."""
        if image_data.ndim != 3:
            st.warning("3D volume rendering requires a 3D image.")
            return

        z, y, x = image_data.shape
        Z, Y, X = np.mgrid[:z, :y, :x]

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=image_data.flatten(),
            isomin=np.min(image_data),
            isomax=np.max(image_data),
            opacity=0.1,  # Adjust for better visualization
            surface_count=20, # Adjust for detail
            colorscale='viridis'
        ))
        
        fig.update_layout(
            title="Interactive 3D Volume",
            scene_xaxis_title='X',
            scene_yaxis_title='Y',
            scene_zaxis_title='Z'
        )
        
        st.plotly_chart(fig, use_container_width=True)
