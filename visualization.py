"""
Visualization Module
Shared visualization methods for different analysis types and data formats
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
import warnings

class VisualizationManager:
    """Main visualization manager with shared methods across analysis types"""
    
    def __init__(self):
        self.color_schemes = {
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'cividis': 'cividis',
            'hot': 'hot',
            'jet': 'jet'
        }
        self.default_figsize = (10, 8)
        
    def display_image(self, 
                     image_data: np.ndarray, 
                     title: str = "Image", 
                     contrast_mode: str = "Auto",
                     colormap: str = "viridis",
                     aspect_ratio: str = "equal",
                     channel_color: Optional[str] = None) -> None:
        """
        Display image with consistent formatting and controls
        
        Args:
            image_data: 2D numpy array
            title: Image title
            contrast_mode: Contrast adjustment method
            colormap: Colormap to use
            aspect_ratio: Aspect ratio setting
        """
        
        if image_data.size == 0:
            st.warning("No image data to display")
            return
        
        # Handle contrast adjustment
        if contrast_mode == "Auto":
            vmin, vmax = np.percentile(image_data, [1, 99])
        elif contrast_mode == "Percentile":
            vmin, vmax = np.percentile(image_data, [5, 95])
        else:  # Manual
            vmin, vmax = np.min(image_data), np.max(image_data)
        
        # Use channel-specific colormap if provided
        if channel_color and channel_color.startswith('#'):
            # Convert hex color to grayscale colormap
            colormap = self._create_channel_colormap(channel_color)
        
        # Create plotly figure for interactivity
        fig = px.imshow(
            image_data,
            title=title,
            color_continuous_scale=colormap,
            zmin=vmin,
            zmax=vmax,
            aspect="equal" if aspect_ratio == "equal" else "auto"
        )
        
        fig.update_layout(
            title=title,
            width=600,
            height=600,
            showlegend=False
        )
        
        fig.update_coloraxes(colorbar_title="Intensity")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display image statistics
        self._display_image_stats(image_data)
    
    def display_analysis_results(self, analysis_name: str, results: Dict[str, Any]) -> None:
        """
        Display analysis results with appropriate visualizations
        
        Args:
            analysis_name: Name of the analysis
            results: Results dictionary from analysis
        """
        
        if results.get('status') == 'placeholder':
            st.info(f"🔄 {results.get('message', 'Analysis pending implementation')}")
            return
        
        # Display based on analysis type
        if "RICS" in analysis_name:
            self._display_rics_results(results)
        elif "FCS" in analysis_name:
            self._display_fcs_results(results)
        elif "iMSD" in analysis_name:
            self._display_imsd_results(results)
        else:
            self._display_generic_results(results)
    
    def _display_rics_results(self, results: Dict[str, Any]) -> None:
        """Display RICS analysis results with multichannel support"""
        
        st.subheader("RICS Analysis Results")
        
        # Check if this is multichannel results
        multichannel_summary = results.get('multichannel_summary')
        channel_results = {k: v for k, v in results.items() 
                          if isinstance(v, dict) and 'diffusion_map' in v}
        
        if multichannel_summary:
            # Display multichannel summary first
            st.subheader("Multichannel Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Channels Analyzed", multichannel_summary.get('channels_analyzed', 0))
            with col2:
                st.metric(
                    "Mean Diffusion (All Channels)", 
                    f"{multichannel_summary.get('mean_diffusion_all_channels', 0):.3f} µm²/s"
                )
            with col3:
                st.metric(
                    "Channel Variation", 
                    f"{multichannel_summary.get('diffusion_variation_between_channels', 0):.3f} µm²/s"
                )
            
            # Channel selection for detailed view
            channel_names = list(channel_results.keys())
            selected_channel = st.selectbox("View Channel Details", channel_names)
            
            if selected_channel in channel_results:
                channel_data = channel_results[selected_channel]
                self._display_single_channel_rics(channel_data, selected_channel)
        
        else:
            # Single channel or legacy format
            if channel_results:
                # New multichannel format but single channel
                channel_name, channel_data = next(iter(channel_results.items()))
                self._display_single_channel_rics(channel_data, channel_name)
            else:
                # Legacy single channel format
                self._display_single_channel_rics(results, "Single Channel")
    
    def _display_single_channel_rics(self, results: Dict[str, Any], channel_name: str) -> None:
        """Display RICS results for a single channel"""
        
        st.subheader(f"RICS Results - {channel_name}")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mean Diffusion Coefficient", 
                f"{results.get('mean_diffusion_coefficient', 0):.3f} µm²/s"
            )
        
        with col2:
            st.metric(
                "Diffusion Std", 
                f"{results.get('diffusion_std', 0):.3f} µm²/s"
            )
        
        with col3:
            analysis_summary = results.get('analysis_summary', {})
            st.metric(
                "Pixels Analyzed", 
                analysis_summary.get('num_pixels_analyzed', 0)
            )
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Diffusion Map", "Autocorrelation", "Statistics"])
        
        with tab1:
            if 'diffusion_map' in results:
                diffusion_map = results['diffusion_map']
                self.display_image(
                    diffusion_map,
                    title="Diffusion Coefficient Map",
                    colormap="viridis"
                )
                
                # Diffusion coefficient histogram
                valid_diffusion = diffusion_map[diffusion_map > 0]
                if len(valid_diffusion) > 0:
                    fig = px.histogram(
                        x=valid_diffusion,
                        nbins=50,
                        title="Diffusion Coefficient Distribution",
                        labels={'x': 'Diffusion Coefficient (µm²/s)', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'autocorrelation_maps' in results:
                autocorr_maps = results['autocorrelation_maps']
                
                # Display autocorrelation for different lag times
                lag_time = st.slider("Lag Time", 0, len(autocorr_maps)-1, 0)
                
                self.display_image(
                    autocorr_maps[lag_time],
                    title=f"Autocorrelation Map (τ = {lag_time})",
                    colormap="RdBu_r"
                )
        
        with tab3:
            self._display_statistics_table(results)
    
    def _display_fcs_results(self, results: Dict[str, Any]) -> None:
        """Display FCS analysis results"""
        
        st.subheader("FCS Analysis Results")
        
        # Summary metrics
        analysis_summary = results.get('analysis_summary', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation Curves", analysis_summary.get('num_correlation_curves', 0))
        with col2:
            st.metric("Mean Diffusion Time", f"{analysis_summary.get('mean_diffusion_time', 0):.3f} s")
        with col3:
            st.metric("Mean Amplitude", f"{analysis_summary.get('mean_amplitude', 0):.3f}")
        
        # Display correlation curves
        if 'correlation_curves' in results and results['correlation_curves']:
            correlation_curves = results['correlation_curves']
            
            # Plot selection
            if len(correlation_curves) > 1:
                curve_idx = st.slider("Correlation Curve", 0, len(correlation_curves)-1, 0)
            else:
                curve_idx = 0
            
            if curve_idx < len(correlation_curves):
                curve = correlation_curves[curve_idx]
                time_axis = np.arange(len(curve)) * results.get('time_interval', 0.1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=curve,
                    mode='lines+markers',
                    name=f'Correlation Curve {curve_idx}'
                ))
                
                fig.update_layout(
                    title=f"FCS Correlation Function - Region {curve_idx}",
                    xaxis_title="Lag Time (s)",
                    yaxis_title="Correlation G(τ)",
                    xaxis_type="log",
                    width=800,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Fit results table
        if 'fit_results' in results:
            self._display_fcs_fit_table(results['fit_results'])
    
    def _display_imsd_results(self, results: Dict[str, Any]) -> None:
        """Display iMSD analysis results"""
        
        st.subheader("iMSD Analysis Results")
        
        # Summary metrics
        analysis_summary = results.get('analysis_summary', {})
        diffusion_analysis = results.get('diffusion_analysis', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Mean Diffusion Coefficient", 
                f"{diffusion_analysis.get('mean_diffusion', 0):.3f} µm²/s"
            )
        with col2:
            st.metric(
                "Anomalous Exponent", 
                f"{diffusion_analysis.get('mean_alpha', 1):.3f}"
            )
        with col3:
            st.metric(
                "Regions Analyzed", 
                analysis_summary.get('num_regions_analyzed', 0)
            )
        
        # MSD visualization
        if 'msd_maps' in results and results['msd_maps'].size > 0:
            msd_maps = results['msd_maps']
            
            # Time lag selection
            lag_idx = st.slider("Time Lag", 0, len(msd_maps)-1, min(5, len(msd_maps)-1))
            
            self.display_image(
                msd_maps[lag_idx],
                title=f"MSD Map (τ = {lag_idx})",
                colormap="plasma"
            )
        
        # Diffusion coefficient and anomalous exponent distributions
        if diffusion_analysis.get('diffusion_coefficients'):
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    x=diffusion_analysis['diffusion_coefficients'],
                    nbins=30,
                    title="Diffusion Coefficient Distribution",
                    labels={'x': 'D (µm²/s)', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if diffusion_analysis.get('anomalous_exponents'):
                    fig = px.histogram(
                        x=diffusion_analysis['anomalous_exponents'],
                        nbins=30,
                        title="Anomalous Exponent Distribution",
                        labels={'x': 'α', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_generic_results(self, results: Dict[str, Any]) -> None:
        """Display generic analysis results"""
        
        st.subheader("Analysis Results")
        
        # Display any numerical results
        if 'analysis_summary' in results:
            summary = results['analysis_summary']
            
            # Create metrics from summary
            if isinstance(summary, dict):
                cols = st.columns(min(len(summary), 4))
                for i, (key, value) in enumerate(summary.items()):
                    with cols[i % len(cols)]:
                        st.metric(key.replace('_', ' ').title(), f"{value}")
        
        # Display any arrays or images
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:  # 2D array - display as image
                    st.write(f"**{key.replace('_', ' ').title()}**")
                    self.display_image(value, title=key)
                elif value.ndim == 1:  # 1D array - display as line plot
                    st.write(f"**{key.replace('_', ' ').title()}**")
                    fig = px.line(y=value, title=key.replace('_', ' ').title())
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_image_stats(self, image_data: np.ndarray) -> None:
        """Display image statistics"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min", f"{np.min(image_data):.2f}")
        with col2:
            st.metric("Max", f"{np.max(image_data):.2f}")
        with col3:
            st.metric("Mean", f"{np.mean(image_data):.2f}")
        with col4:
            st.metric("Std", f"{np.std(image_data):.2f}")
    
    def _display_statistics_table(self, results: Dict[str, Any]) -> None:
        """Display results as a statistics table"""
        
        stats_data = []
        
        # Extract numerical values from results
        for key, value in results.items():
            if isinstance(value, (int, float)):
                stats_data.append({"Parameter": key.replace('_', ' ').title(), "Value": value})
            elif isinstance(value, dict) and 'analysis_summary' in key:
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, list)):
                        display_value = sub_value if not isinstance(sub_value, list) else f"Range: {sub_value}"
                        stats_data.append({
                            "Parameter": sub_key.replace('_', ' ').title(), 
                            "Value": display_value
                        })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)
    
    def _display_fcs_fit_table(self, fit_results: List[Dict[str, Any]]) -> None:
        """Display FCS fit results in a table"""
        
        if not fit_results:
            st.info("No fit results available")
            return
        
        # Convert to dataframe
        df_data = []
        for i, fit in enumerate(fit_results):
            df_data.append({
                "Region": i,
                "N Molecules": f"{fit.get('n_molecules', 0):.2f}",
                "τ_diff (s)": f"{fit.get('tau_diff', 0):.3f}",
                "S Ratio": f"{fit.get('s_ratio', 0):.3f}",
                "Amplitude": f"{fit.get('amplitude', 0):.3f}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    def create_correlation_heatmap(self, correlation_matrix: np.ndarray, 
                                 labels: Optional[List[str]] = None,
                                 title: str = "Correlation Matrix") -> go.Figure:
        """Create an interactive correlation heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            width=600,
            height=600
        )
        
        return fig
    
    def create_particle_trajectory_plot(self, trajectories: List[np.ndarray], 
                                      title: str = "Particle Trajectories") -> go.Figure:
        """Create particle trajectory visualization"""
        
        fig = go.Figure()
        
        for i, trajectory in enumerate(trajectories):
            if trajectory.shape[1] >= 2:
                fig.add_trace(go.Scatter(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode='lines+markers',
                    name=f'Track {i}',
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X Position (pixels)",
            yaxis_title="Y Position (pixels)",
            showlegend=True,
            width=700,
            height=700
        )
        
        return fig
    
    def export_plots(self, format_type: str) -> bytes:
        """Export current plots to specified format"""
        
        if format_type == "PNG":
            # Would implement PNG export
            pass
        elif format_type == "SVG":
            # Would implement SVG export
            pass
        elif format_type == "PDF":
            # Would implement PDF export
            pass
        
        return b"Export functionality to be implemented"
    
    def _create_channel_colormap(self, hex_color: str) -> str:
        """Create a grayscale-to-color colormap for channel visualization"""
        
        # Map common fluorophore colors to appropriate colormaps
        color_mappings = {
            '#FF0000': 'Reds',      # Red
            '#00FF00': 'Greens',    # Green
            '#0000FF': 'Blues',     # Blue
            '#FFFF00': 'YlOrRd',    # Yellow
            '#FF00FF': 'plasma',    # Magenta
            '#00FFFF': 'BuPu',      # Cyan
            '#FFA500': 'Oranges',   # Orange
            '#800080': 'Purples',   # Purple
            '#FFC0CB': 'RdPu',      # Pink
            '#A52A2A': 'copper'     # Brown
        }
        
        return color_mappings.get(hex_color.upper(), 'viridis')
    
    def display_multichannel_composite(self, image_data_list: List[np.ndarray], 
                                     channel_names: List[str],
                                     channel_colors: List[str],
                                     title: str = "Multichannel Composite") -> None:
        """Display multichannel data as a composite overlay"""
        
        if not image_data_list or len(image_data_list) == 0:
            st.warning("No image data to display")
            return
        
        st.subheader(title)
        
        # Controls for composite display
        col1, col2 = st.columns(2)
        
        with col1:
            display_mode = st.selectbox(
                "Display Mode",
                ["Individual Channels", "Composite Overlay", "Side by Side"]
            )
        
        with col2:
            if display_mode == "Composite Overlay":
                opacity = st.slider("Channel Opacity", 0.1, 1.0, 0.7, 0.1)
        
        if display_mode == "Individual Channels":
            # Display each channel separately with its color
            for i, (img, name, color) in enumerate(zip(image_data_list, channel_names, channel_colors)):
                self.display_image(
                    img,
                    title=f"{name}",
                    colormap=self._create_channel_colormap(color),
                    channel_color=color
                )
        
        elif display_mode == "Side by Side":
            # Display channels in columns
            num_channels = len(image_data_list)
            cols = st.columns(min(num_channels, 3))  # Max 3 columns
            
            for i, (img, name, color) in enumerate(zip(image_data_list, channel_names, channel_colors)):
                with cols[i % 3]:
                    self.display_image(
                        img,
                        title=name,
                        colormap=self._create_channel_colormap(color),
                        channel_color=color
                    )
        
        elif display_mode == "Composite Overlay":
            # Create RGB composite (for up to 3 channels) or sequential overlay
            if len(image_data_list) <= 3:
                composite = self._create_rgb_composite(image_data_list, channel_colors)
                
                fig = px.imshow(
                    composite,
                    title="RGB Composite",
                    aspect="equal"
                )
                
                fig.update_layout(
                    title="Multichannel RGB Composite",
                    width=600,
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Composite overlay supports up to 3 channels. Showing individual channels.")
                for i, (img, name, color) in enumerate(zip(image_data_list, channel_names, channel_colors)):
                    self.display_image(
                        img,
                        title=f"{name}",
                        colormap=self._create_channel_colormap(color),
                        channel_color=color
                    )
    
    def _create_rgb_composite(self, image_list: List[np.ndarray], colors: List[str]) -> np.ndarray:
        """Create RGB composite from multiple channels"""
        
        if not image_list:
            return np.zeros((100, 100, 3))
        
        # Get dimensions from first image
        height, width = image_list[0].shape
        composite = np.zeros((height, width, 3))
        
        # Color channel mapping
        color_map = {
            '#FF0000': [1, 0, 0],  # Red
            '#00FF00': [0, 1, 0],  # Green
            '#0000FF': [0, 0, 1],  # Blue
            '#FFFF00': [1, 1, 0],  # Yellow
            '#FF00FF': [1, 0, 1],  # Magenta
            '#00FFFF': [0, 1, 1],  # Cyan
        }
        
        for i, (img, color) in enumerate(zip(image_list[:3], colors[:3])):
            # Normalize image to 0-1
            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img)) if np.max(img) > np.min(img) else img
            
            # Get RGB weights for this color
            rgb_weights = color_map.get(color.upper(), [1, 1, 1])
            
            # Add to composite
            for c in range(3):
                composite[:, :, c] += img_norm * rgb_weights[c]
        
        # Normalize final composite
        composite = np.clip(composite, 0, 1)
        
        return composite
    
    def create_multi_panel_figure(self, data_dict: Dict[str, np.ndarray], 
                                titles: Optional[Dict[str, str]] = None) -> go.Figure:
        """Create multi-panel figure for comparative visualization"""
        
        n_panels = len(data_dict)
        cols = int(np.ceil(np.sqrt(n_panels)))
        rows = int(np.ceil(n_panels / cols))
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=list(titles.values()) if titles else list(data_dict.keys())
        )
        
        for i, (key, data) in enumerate(data_dict.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Heatmap(z=data, showscale=i==0),
                row=row, col=col
            )
        
        fig.update_layout(height=300*rows, width=300*cols)
        
        return fig
