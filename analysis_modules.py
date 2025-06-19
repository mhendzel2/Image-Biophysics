"""
Analysis Modules
Placeholder implementations for various biophysical analysis techniques
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings
from scipy import ndimage, optimize
from skimage import filters, measure, feature
import time

# Import specialized microscopy analysis libraries
try:
    import multipletau
    MULTIPLETAU_AVAILABLE = True
except ImportError:
    MULTIPLETAU_AVAILABLE = False
    warnings.warn("multipletau not available - FCS correlation limited")

try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    warnings.warn("lmfit not available - advanced fitting limited")

try:
    import trackpy
    TRACKPY_AVAILABLE = True
except ImportError:
    TRACKPY_AVAILABLE = False
    warnings.warn("trackpy not available - particle tracking limited")

try:
    import fcsfiles
    FCSFILES_AVAILABLE = True
except ImportError:
    FCSFILES_AVAILABLE = False
    warnings.warn("fcsfiles not available - Zeiss FCS data limited")

class AnalysisManager:
    """Main manager for all analysis modules"""
    
    def __init__(self):
        self.analysis_modules = {
            "RICS (Raster Image Correlation Spectroscopy)": RICSAnalysis(),
            "FCS/sFCS/FCCS (Fluorescence Correlation Spectroscopy)": FCSAnalysis(),
            "Segmented FCS (Line-scan FCS)": SegmentedFCSAnalysis(),
            "iMSD (Image Mean Square Displacement)": iMSDAnalysis(),
            "Elastography & PIV (Particle Image Velocimetry)": ElastographyPIVAnalysis(),
            "N&B (Number and Brightness)": NBAnalysis(),
            "FLIM (Fluorescence Lifetime Imaging)": FLIMAnalysis(),
            "SPT (Single Particle Tracking)": SPTAnalysis(),
            "Fourier Transform Texture Analysis": FourierAnalysis(),
            "FRAP (Fluorescence Recovery After Photobleaching)": FRAPAnalysis()
        }
    
    def run_analysis(self, analysis_type: str, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the specified analysis on the data
        
        Args:
            analysis_type: Type of analysis to run
            data_info: Standardized data information
            parameters: Analysis parameters
            
        Returns:
            Analysis results dictionary
        """
        
        if analysis_type not in self.analysis_modules:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        analysis_module = self.analysis_modules[analysis_type]
        
        # Run the analysis
        results = analysis_module.analyze(data_info, parameters)
        
        # Add metadata
        results['analysis_type'] = analysis_type
        results['parameters'] = parameters
        results['timestamp'] = time.time()
        
        return results
    
    def get_available_analyses(self) -> list:
        """Return list of available analysis types"""
        return list(self.analysis_modules.keys())

class BaseAnalysis:
    """Base class for all analysis modules"""
    
    def __init__(self):
        self.name = "Base Analysis"
        self.description = "Base analysis class"
    
    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method - to be implemented by subclasses
        
        Args:
            data_info: Standardized data information
            parameters: Analysis parameters
            
        Returns:
            Results dictionary
        """
        raise NotImplementedError("Subclasses must implement analyze method")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate analysis parameters"""
        return True
    
    def preprocess_data(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Common preprocessing steps"""
        # Background subtraction if requested
        if parameters.get('background_subtraction', False):
            # Simple background subtraction using minimum value
            background = np.percentile(image_data, 5)
            image_data = image_data - background
            image_data[image_data < 0] = 0
        
        return image_data

class RICSAnalysis(BaseAnalysis):
    """Raster Image Correlation Spectroscopy Analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "RICS"
        self.description = "Measures diffusion rates and molecular flow via spatial autocorrelation"
    
    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform RICS analysis
        
        RICS analyzes fluorescence fluctuations across raster-scanned images
        to build spatial autocorrelation maps
        """
        
        image_data = data_info['image_data']
        pixel_size = data_info.get('pixel_size', 0.1)
        
        # Extract parameters
        tau_max = parameters.get('tau_max', 20)
        time_interval = parameters.get('time_interval', 0.1)
        
        # Handle multichannel data
        analysis_channels = parameters.get('analysis_channels', [0])
        channel_names = parameters.get('channel_names', ['Channel 1'])
        
        results = {}
        
        for i, channel_idx in enumerate(analysis_channels):
            # Extract channel data
            if len(image_data.shape) == 4:  # T, Y, X, C
                if channel_idx < image_data.shape[-1]:
                    channel_data = image_data[:, :, :, channel_idx]
                else:
                    continue
            elif len(image_data.shape) == 3 and data_info.get('channels', 1) > 1:
                if channel_idx < image_data.shape[-1]:
                    channel_data = image_data[:, :, channel_idx][np.newaxis, :, :]
                else:
                    continue
            else:
                channel_data = image_data
            
            # Apply nuclear alignment before RICS analysis
            try:
                from alignment_integration import apply_nuclear_alignment
                aligned_data, alignment_applied, alignment_quality = apply_nuclear_alignment(
                    channel_data, parameters, "RICS"
                )
                processed_data = self.preprocess_data(aligned_data, parameters)
            except ImportError:
                processed_data = self.preprocess_data(channel_data, parameters)
                alignment_applied = False
                alignment_quality = {}
            
            # Ensure we have time series data
            if len(processed_data.shape) == 2:
                st.warning(f"RICS analysis requires time series data for {channel_names[i]}. Using single frame.")
                processed_data = processed_data[np.newaxis, :, :]
            elif len(processed_data.shape) == 3 and processed_data.shape[2] <= 4:
                # Likely channels, take first channel
                processed_data = processed_data[:, :, 0][np.newaxis, :, :]
            
            # Calculate spatial autocorrelation using corrected FFT method
            autocorr_map = self._calculate_spatial_autocorrelation(processed_data, tau_max)
            
            # Fit proper 2D Gaussian diffusion model
            diffusion_map, quality_map = self._fit_diffusion_model(autocorr_map, pixel_size, time_interval)
            
            # Calculate statistics
            mean_diffusion = np.nanmean(diffusion_map[diffusion_map > 0])
            diffusion_std = np.nanstd(diffusion_map[diffusion_map > 0])
            
            # Store channel-specific results with corrected structure
            channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
            results[channel_name] = {
                'autocorrelation_map': autocorr_map,
                'diffusion_map': diffusion_map,
                'quality_map': quality_map,
                'mean_diffusion_coefficient': mean_diffusion,
                'diffusion_std': diffusion_std,
                'pixel_size': pixel_size,
                'channel_index': channel_idx,
                'analysis_summary': {
                    'num_pixels_analyzed': np.sum(diffusion_map > 0),
                    'mean_diffusion_um2_per_s': mean_diffusion,
                    'diffusion_range': [np.nanmin(diffusion_map[diffusion_map > 0]), 
                                      np.nanmax(diffusion_map[diffusion_map > 0])] if np.any(diffusion_map > 0) else [0, 0],
                    'fitting_quality': np.nanmean(quality_map) if quality_map is not None else 0
                }
            }
        
        # Add overall summary if multiple channels
        if len(analysis_channels) > 1:
            all_diffusion_coeffs = []
            for ch_result in results.values():
                if isinstance(ch_result, dict) and 'mean_diffusion_coefficient' in ch_result:
                    all_diffusion_coeffs.append(ch_result['mean_diffusion_coefficient'])
            
            results['multichannel_summary'] = {
                'channels_analyzed': len(analysis_channels),
                'channel_names': channel_names,
                'mean_diffusion_all_channels': np.nanmean(all_diffusion_coeffs) if all_diffusion_coeffs else 0,
                'diffusion_variation_between_channels': np.nanstd(all_diffusion_coeffs) if all_diffusion_coeffs else 0
            }
        
        return results
    
    def _calculate_spatial_autocorrelation(self, image_stack: np.ndarray, tau_max: int) -> np.ndarray:
        """Calculate proper RICS spatial autocorrelation using FFT-based method"""
        
        t_frames, height, width = image_stack.shape
        
        # Calculate intensity fluctuations: delta_I = I - <I>_t
        mean_intensity = np.mean(image_stack, axis=0)
        fluctuations = image_stack - mean_intensity[np.newaxis, :, :]
        
        # Calculate spatial autocorrelation for each frame and average
        autocorr_sum = np.zeros((height, width))
        
        for t in range(t_frames):
            delta_I = fluctuations[t]
            
            # FFT-based 2D autocorrelation: G(dx,dy) = IFFT(FFT(delta_I) * conj(FFT(delta_I)))
            fft_image = np.fft.fft2(delta_I)
            autocorr_2d = np.fft.ifft2(fft_image * np.conj(fft_image)).real
            
            # Shift zero frequency to center and normalize
            autocorr_centered = np.fft.fftshift(autocorr_2d)
            autocorr_sum += autocorr_centered
        
        # Average over time
        avg_autocorr = autocorr_sum / t_frames
        
        # Normalize by zero-lag value
        center_y, center_x = height // 2, width // 2
        zero_lag_value = avg_autocorr[center_y, center_x]
        
        if zero_lag_value > 0:
            avg_autocorr = avg_autocorr / zero_lag_value
        
        return avg_autocorr
    
    def _fit_diffusion_model(self, autocorr_map: np.ndarray, pixel_size: float, time_interval: float) -> Tuple[np.ndarray, np.ndarray]:
        """Fit proper RICS 2D Gaussian diffusion model to spatial autocorrelation"""
        
        height, width = autocorr_map.shape
        center_y, center_x = height // 2, width // 2
        
        # Create coordinate grids for fitting
        y_coords, x_coords = np.ogrid[:height, :width]
        y_coords = (y_coords - center_y) * pixel_size
        x_coords = (x_coords - center_x) * pixel_size
        
        # Extract central region for fitting (avoid edge effects)
        fit_size = min(height, width) // 4
        y_min, y_max = max(0, center_y - fit_size), min(height, center_y + fit_size)
        x_min, x_max = max(0, center_x - fit_size), min(width, center_x + fit_size)
        
        autocorr_fit_region = autocorr_map[y_min:y_max, x_min:x_max]
        y_fit = y_coords[y_min:y_max, x_min:x_max]
        x_fit = x_coords[y_min:y_max, x_min:x_max]
        
        # RICS 2D Gaussian model: G(dx,dy) = G0 * exp(-(dx²/(2σx²) + dy²/(2σy²))) + offset
        def rics_2d_gaussian(coords, G0, sigma_x, sigma_y, offset):
            y, x = coords
            return G0 * np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2))) + offset
        
        try:
            from scipy.optimize import curve_fit
            
            # Prepare data for fitting
            coords = np.vstack([y_fit.ravel(), x_fit.ravel()])
            data = autocorr_fit_region.ravel()
            
            # Initial parameter guess
            G0_init = np.max(autocorr_fit_region)
            sigma_init = pixel_size * 2  # Initial guess for beam waist
            offset_init = np.mean(autocorr_fit_region[autocorr_fit_region < np.percentile(autocorr_fit_region, 10)])
            
            popt, pcov = curve_fit(
                rics_2d_gaussian, coords, data,
                p0=[G0_init, sigma_init, sigma_init, offset_init],
                bounds=([0, pixel_size*0.5, pixel_size*0.5, -np.inf], 
                       [np.inf, pixel_size*20, pixel_size*20, np.inf]),
                maxfev=2000
            )
            
            G0, sigma_x, sigma_y, offset = popt
            
            # Calculate diffusion coefficient from beam waist
            # For RICS: D = ω²/(4*τ_p) where ω is beam waist, τ_p is pixel dwell time
            avg_sigma = (sigma_x + sigma_y) / 2
            pixel_dwell_time = time_interval  # Use provided time interval as dwell time
            diffusion_coeff = (avg_sigma ** 2) / (4 * pixel_dwell_time)
            
            # Create result maps
            diffusion_map = np.full((height, width), diffusion_coeff)
            quality_map = np.full((height, width), G0)  # Amplitude as quality metric
            
            return diffusion_map, quality_map
            
        except Exception as e:
            print(f"RICS 2D Gaussian fitting failed: {e}")
            # Fallback: estimate diffusion from autocorrelation width
            try:
                # Simple width estimation from autocorrelation decay
                center_val = autocorr_map[center_y, center_x]
                half_max = center_val / 2
                
                # Find half-maximum points in x and y directions
                x_profile = autocorr_map[center_y, :]
                y_profile = autocorr_map[:, center_x]
                
                # Estimate width at half maximum
                x_width = np.sum(x_profile > half_max) * pixel_size
                y_width = np.sum(y_profile > half_max) * pixel_size
                avg_width = (x_width + y_width) / 2
                
                # Rough diffusion estimate
                diffusion_coeff = (avg_width ** 2) / (4 * time_interval) if avg_width > 0 else 0
                
                diffusion_map = np.full((height, width), diffusion_coeff)
                quality_map = np.full((height, width), center_val)
                
                return diffusion_map, quality_map
                
            except:
                return np.zeros((height, width)), np.zeros((height, width))

class SegmentedFCSAnalysis(BaseAnalysis):
    """Segmented Fluorescence Correlation Spectroscopy Analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "Segmented FCS"
        self.description = "Measures diffusion from temporal segments of line-scan data"

    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Segmented FCS analysis with nuclear alignment"""
        
        image_data = data_info.get('image_data')
        
        if image_data is None:
            return {'status': 'error', 'message': 'No image data provided'}
        
        # Handle multi-dimensional data
        if image_data.ndim == 3:
            # Take first timepoint if it's a time series
            image_data = image_data[0, :, :]
        elif image_data.ndim == 4:
            # Take first timepoint and first channel
            image_data = image_data[0, :, :, 0]
        elif image_data.ndim != 2:
            return {'status': 'error', 'message': 'Segmented FCS requires 2D line-scan data'}
        
        # Apply nuclear alignment if data is time series
        original_data = data_info.get('image_data')
        if original_data.ndim >= 3:
            try:
                from alignment_integration import apply_nuclear_alignment, create_alignment_summary
                aligned_data, alignment_applied, alignment_quality = apply_nuclear_alignment(
                    original_data, parameters, "Segmented FCS"
                )
                # Use the aligned first frame for analysis
                if aligned_data.ndim == 3:
                    image_data = aligned_data[0, :, :]
                elif aligned_data.ndim == 4:
                    image_data = aligned_data[0, :, :, 0]
            except ImportError:
                alignment_applied = False
                alignment_quality = {}
        else:
            alignment_applied = False
            alignment_quality = {}
        
        # Get parameters with defaults
        segmentation_type = parameters.get('segmentation_type', 'x')
        segment_length = parameters.get('segment_length', 128)
        model_type = parameters.get('model_type', '2d')
        max_lag_fraction = parameters.get('max_lag_fraction', 0.25)
        
        # Get timing parameters from data_info
        pixel_time = data_info.get('pixel_time', parameters.get('pixel_time', 3.05e-6))
        line_time = data_info.get('line_time', parameters.get('line_time', 0.56e-3))
        pixel_size = data_info.get('pixel_size', parameters.get('pixel_size', 0.05))
        
        try:
            from fcs_analysis import segmented_fcs_analysis, analyze_segment_statistics
            
            # Perform segmented FCS analysis
            fcs_results = segmented_fcs_analysis(
                image_data,
                segmentation_type=segmentation_type,
                segment_length=segment_length,
                pixel_time=pixel_time,
                line_time=line_time,
                pixel_size=pixel_size,
                model_type=model_type,
                max_lag_fraction=max_lag_fraction
            )
            
            if not fcs_results:
                return {'status': 'error', 'message': 'Segmented FCS analysis failed to produce results'}
            
            # Analyze statistics across segments
            segment_stats = analyze_segment_statistics(fcs_results)
            
            # Create alignment summary
            if alignment_applied:
                from alignment_integration import create_alignment_summary
                alignment_summary = create_alignment_summary(alignment_applied, alignment_quality, "Segmented FCS")
            else:
                alignment_summary = {
                    'nuclear_alignment_applied': False,
                    'alignment_status': 'Not applied to 2D data',
                    'recommendation': 'Alignment not applicable for single-frame analysis'
                }
            
            return {
                'status': 'success',
                'segment_results': fcs_results,
                'segment_statistics': segment_stats,
                'nuclear_alignment': alignment_summary,
                'analysis_summary': {
                    'segmentation_type': segmentation_type,
                    'segment_length': segment_length,
                    'model_type': model_type,
                    'num_segments_total': segment_stats['num_segments_total'],
                    'num_segments_valid': segment_stats['num_segments_valid'],
                    'success_rate': f"{segment_stats['success_rate']:.2%}",
                    'mean_diffusion_coefficient': f"{segment_stats['mean_D']:.4f} µm²/s",
                    'diffusion_std': f"{segment_stats['std_D']:.4f} µm²/s",
                    'mean_amplitude': f"{segment_stats['mean_G0']:.4f}",
                    'mean_beam_waist': f"{segment_stats['mean_w0']:.4f} µm",
                    'average_fit_quality': f"{segment_stats['mean_r_squared']:.3f}",
                    'nuclear_alignment_applied': alignment_applied,
                    'data_reliability': alignment_summary.get('reliability_impact', 'N/A for 2D data')
                }
            }
            
        except ImportError:
            return {'status': 'error', 'message': 'Segmented FCS analysis module not available'}
        except Exception as e:
            return {'status': 'error', 'message': f'Segmented FCS analysis failed: {str(e)}'}

class FCSAnalysis(BaseAnalysis):
    """Fluorescence Correlation Spectroscopy Analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "FCS"
        self.description = "Quantifies diffusion and local concentration at high resolution"
    
    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform FCS analysis using multipletau and lmfit libraries"""
        
        if not MULTIPLETAU_AVAILABLE:
            return {'status': 'error', 'message': 'multipletau library required for FCS analysis'}
        
        image_data = data_info['image_data']
        time_interval = data_info.get('time_interval', 0.1)
        
        # Extract parameters
        bleach_correction = parameters.get('bleach_correction', True)
        binning = parameters.get('binning', 1)
        correlation_window = parameters.get('correlation_window', 16)
        
        # Handle multichannel data
        analysis_channels = parameters.get('analysis_channels', [0])
        channel_names = parameters.get('channel_names', ['Channel 1'])
        
        results = {}
        
        for i, channel_idx in enumerate(analysis_channels):
            # Extract channel data
            if len(image_data.shape) == 4:  # T, Y, X, C
                if channel_idx < image_data.shape[-1]:
                    channel_data = image_data[:, :, :, channel_idx]
                else:
                    continue
            elif len(image_data.shape) == 3 and data_info.get('channels', 1) > 1:
                if channel_idx < image_data.shape[-1]:
                    channel_data = image_data[:, :, channel_idx]
                else:
                    continue
            else:
                channel_data = image_data
            
            # Preprocess data
            processed_data = self.preprocess_data(channel_data, parameters)
            
            # Extract intensity traces for FCS
            intensity_traces = self._extract_intensity_traces(processed_data, correlation_window)
            
            # Calculate correlation functions using multipletau
            correlation_results = []
            for trace in intensity_traces:
                if len(trace) > 100:  # Minimum length for meaningful correlation
                    # Apply binning if requested
                    if binning > 1:
                        trace = self._apply_binning(trace, binning)
                    
                    # Calculate autocorrelation using multipletau
                    tau, correlation = multipletau.correlate(trace, trace, normalize=True, copy=False)
                    tau *= time_interval  # Convert to real time units
                    
                    # Apply bleach correction if requested
                    if bleach_correction:
                        correlation = self._apply_exponential_bleach_correction(correlation, tau)
                    
                    correlation_results.append({'tau': tau, 'correlation': correlation})
            
            # Fit FCS model using lmfit if available
            fit_results = []
            if LMFIT_AVAILABLE and correlation_results:
                for corr_data in correlation_results:
                    fit_result = self._fit_fcs_model_lmfit(corr_data['tau'], corr_data['correlation'])
                    fit_results.append(fit_result)
            else:
                # Fallback to scipy fitting
                for corr_data in correlation_results:
                    fit_result = self._fit_fcs_model_scipy(corr_data['tau'], corr_data['correlation'])
                    fit_results.append(fit_result)
            
            # Store channel-specific results
            channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
            results[channel_name] = {
                'correlation_curves': correlation_results,
                'fit_results': fit_results,
                'time_interval': time_interval,
                'channel_index': channel_idx,
                'analysis_summary': {
                    'num_correlation_curves': len(correlation_results),
                    'mean_diffusion_time': np.mean([r.get('tau_diff', 0) for r in fit_results if r.get('tau_diff', 0) > 0]),
                    'mean_amplitude': np.mean([r.get('amplitude', 0) for r in fit_results if r.get('amplitude', 0) > 0]),
                    'mean_concentration': np.mean([r.get('concentration', 0) for r in fit_results if r.get('concentration', 0) > 0])
                }
            }
        
        # Add multichannel summary if multiple channels
        if len(analysis_channels) > 1:
            all_diffusion_times = []
            all_concentrations = []
            for ch_result in results.values():
                if isinstance(ch_result, dict) and 'analysis_summary' in ch_result:
                    summary = ch_result['analysis_summary']
                    if summary.get('mean_diffusion_time', 0) > 0:
                        all_diffusion_times.append(summary['mean_diffusion_time'])
                    if summary.get('mean_concentration', 0) > 0:
                        all_concentrations.append(summary['mean_concentration'])
            
            results['multichannel_summary'] = {
                'channels_analyzed': len(analysis_channels),
                'channel_names': channel_names,
                'mean_diffusion_time_all_channels': np.mean(all_diffusion_times) if all_diffusion_times else 0,
                'mean_concentration_all_channels': np.mean(all_concentrations) if all_concentrations else 0,
                'diffusion_variation_between_channels': np.std(all_diffusion_times) if all_diffusion_times else 0
            }
        
        return results
    
    def _extract_intensity_traces(self, image_data: np.ndarray, window_size: int) -> list:
        """Extract intensity traces from image data for FCS analysis"""
        
        traces = []
        
        if len(image_data.shape) == 2:
            # Single frame - create trace from central region
            h, w = image_data.shape
            center_trace = np.mean(image_data[h//2-window_size//2:h//2+window_size//2,
                                            w//2-window_size//2:w//2+window_size//2])
            traces.append([center_trace])
            
        elif len(image_data.shape) == 3:
            # Time series data
            t_frames, height, width = image_data.shape
            
            # Extract traces from multiple regions
            for i in range(0, height - window_size, window_size // 2):
                for j in range(0, width - window_size, window_size // 2):
                    # Extract time trace from region
                    region_trace = np.mean(image_data[:, i:i+window_size, j:j+window_size], axis=(1, 2))
                    traces.append(region_trace)
        
        return traces
    
    def _apply_binning(self, trace: np.ndarray, binning: int) -> np.ndarray:
        """Apply temporal binning to intensity trace"""
        
        n_bins = len(trace) // binning
        binned_trace = trace[:n_bins * binning].reshape(n_bins, binning).mean(axis=1)
        return binned_trace
    
    def _apply_exponential_bleach_correction(self, correlation: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Apply exponential bleach correction to correlation function"""
        
        # Simple exponential bleach correction
        # More sophisticated methods could be implemented
        try:
            # Fit exponential decay to long-time tail
            long_time_idx = tau > np.max(tau) * 0.1
            if np.sum(long_time_idx) > 5:
                long_tau = tau[long_time_idx]
                long_corr = correlation[long_time_idx]
                
                # Fit exponential: y = offset + amp * exp(-t/tau_bleach)
                popt, _ = optimize.curve_fit(
                    lambda t, offset, amp, tau_bleach: offset + amp * np.exp(-t / tau_bleach),
                    long_tau, long_corr,
                    p0=[np.min(long_corr), np.max(long_corr) - np.min(long_corr), np.max(long_tau)],
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
                )
                
                # Subtract bleaching component
                bleach_component = popt[1] * np.exp(-tau / popt[2])
                corrected = correlation - bleach_component
                return corrected
        except:
            pass
        
        return correlation
    
    def _fit_fcs_model_lmfit(self, tau: np.ndarray, correlation: np.ndarray) -> Dict[str, Any]:
        """Fit FCS model using lmfit library"""
        
        try:
            # Define FCS model for 3D diffusion with triplet state
            def fcs_model(params, tau):
                N = params['N']
                tau_diff = params['tau_diff']
                S = params['S']
                T = params['T']
                tau_T = params['tau_T']
                offset = params['offset']
                
                triplet_term = (1 + (T / (1 - T)) * np.exp(-tau / tau_T))
                diffusion_term = (1 + tau / tau_diff)**-1 * (1 + tau / (tau_diff * S**2))**-0.5
                return (1 / N) * triplet_term * diffusion_term + offset
            
            def residual(params, tau, data):
                return data - fcs_model(params, tau)
            
            # Set up parameters
            params = lmfit.Parameters()
            params.add('N', value=1.0, min=0.1, max=1000)
            params.add('tau_diff', value=0.1, min=1e-6, max=100)
            params.add('S', value=0.2, min=0.1, max=1.0)
            params.add('T', value=0.1, min=0.0, max=0.9)
            params.add('tau_T', value=1e-6, min=1e-9, max=1e-3)
            params.add('offset', value=1.0, min=0.9, max=1.1)
            
            # Perform fit
            result = lmfit.minimize(residual, params, args=(tau, correlation))
            
            # Extract results
            fit_params = result.params
            return {
                'N': fit_params['N'].value,
                'tau_diff': fit_params['tau_diff'].value,
                'S': fit_params['S'].value,
                'T': fit_params['T'].value,
                'tau_T': fit_params['tau_T'].value,
                'offset': fit_params['offset'].value,
                'amplitude': 1.0 / fit_params['N'].value,
                'concentration': fit_params['N'].value,
                'chi_squared': result.chisqr,
                'success': result.success
            }
            
        except Exception as e:
            return {
                'N': 0, 'tau_diff': 0, 'S': 0, 'T': 0, 'tau_T': 0,
                'offset': 1, 'amplitude': 0, 'concentration': 0,
                'chi_squared': np.inf, 'success': False, 'error': str(e)
            }
    
    def _fit_fcs_model_scipy(self, tau: np.ndarray, correlation: np.ndarray) -> Dict[str, Any]:
        """Fallback FCS model fitting using scipy"""
        
        try:
            # Simplified 3D diffusion model
            def fcs_model_simple(tau, N, tau_diff, S):
                return (1 / N) * (1 + tau / tau_diff)**-1 * (1 + tau / (tau_diff * S**2))**-0.5 + 1
            
            # Initial guess
            p0 = [1.0, 0.1, 0.2]
            bounds = ([0.1, 1e-6, 0.1], [1000, 100, 1.0])
            
            popt, pcov = optimize.curve_fit(
                fcs_model_simple, tau, correlation,
                p0=p0, bounds=bounds, maxfev=1000
            )
            
            return {
                'N': popt[0],
                'tau_diff': popt[1],
                'S': popt[2],
                'T': 0,
                'tau_T': 0,
                'offset': 1,
                'amplitude': 1.0 / popt[0],
                'concentration': popt[0],
                'chi_squared': np.sum((correlation - fcs_model_simple(tau, *popt))**2),
                'success': True
            }
            
        except Exception as e:
            return {
                'N': 0, 'tau_diff': 0, 'S': 0, 'T': 0, 'tau_T': 0,
                'offset': 1, 'amplitude': 0, 'concentration': 0,
                'chi_squared': np.inf, 'success': False, 'error': str(e)
            }
    
    def _calculate_correlation_functions(self, image_data: np.ndarray, window_size: int, binning: int) -> list:
        """Calculate FCS correlation functions for different regions"""
        
        correlation_curves = []
        
        if len(image_data.shape) == 2:
            # Single frame - can't do temporal correlation
            st.warning("FCS requires time series data")
            return []
        
        # Extract time series from different spatial regions
        if len(image_data.shape) == 3:
            t_frames, height, width = image_data.shape
            
            # Create grid of analysis regions
            for i in range(0, height - window_size, window_size // 2):
                for j in range(0, width - window_size, window_size // 2):
                    # Extract time trace from region
                    region_trace = np.mean(image_data[:, i:i+window_size, j:j+window_size], axis=(1, 2))
                    
                    # Calculate autocorrelation
                    correlation = self._autocorrelate(region_trace, binning)
                    correlation_curves.append(correlation)
        
        return correlation_curves
    
    def _autocorrelate(self, trace: np.ndarray, binning: int) -> np.ndarray:
        """Calculate normalized autocorrelation function"""
        
        # Apply binning
        if binning > 1:
            n_bins = len(trace) // binning
            trace = trace[:n_bins * binning].reshape(n_bins, binning).mean(axis=1)
        
        # Calculate autocorrelation
        n = len(trace)
        correlation = np.correlate(trace, trace, mode='full')
        correlation = correlation[n-1:]  # Take positive lags only
        
        # Normalize
        correlation = correlation / correlation[0]
        
        return correlation
    
    def _apply_bleach_correction(self, correlation_curves: list) -> list:
        """Apply photobleaching correction to correlation curves"""
        
        corrected_curves = []
        for curve in correlation_curves:
            # Simple linear detrending for bleach correction
            x = np.arange(len(curve))
            coeffs = np.polyfit(x, curve, 1)
            trend = np.polyval(coeffs, x)
            corrected = curve - trend + curve[0]
            corrected_curves.append(corrected)
        
        return corrected_curves
    
    def _fit_fcs_model(self, correlation_curves: list, time_interval: float) -> list:
        """Fit FCS diffusion model to correlation curves"""
        
        fit_results = []
        
        for curve in correlation_curves:
            try:
                # Time axis
                tau = np.arange(len(curve)) * time_interval
                
                # Fit 3D diffusion model: G(τ) = (1/N) * (1/(1+τ/τ_diff)) * (1/sqrt(1+τ/(s²*τ_diff)))
                def fcs_model(t, n, tau_diff, s_ratio):
                    return (1/n) * (1/(1 + t/tau_diff)) * (1/np.sqrt(1 + t/(s_ratio**2 * tau_diff)))
                
                # Initial guess
                p0 = [1.0, 1.0, 0.2]  # N, tau_diff, s_ratio
                
                popt, pcov = optimize.curve_fit(
                    fcs_model, tau[:len(curve)//2], curve[:len(curve)//2],
                    p0=p0, bounds=([0.1, 0.001, 0.1], [100, 10, 1.0])
                )
                
                fit_results.append({
                    'n_molecules': popt[0],
                    'tau_diff': popt[1],
                    's_ratio': popt[2],
                    'amplitude': 1/popt[0],
                    'fit_quality': np.sqrt(np.diag(pcov))
                })
                
            except Exception:
                fit_results.append({
                    'n_molecules': 0,
                    'tau_diff': 0,
                    's_ratio': 0,
                    'amplitude': 0,
                    'fit_quality': [0, 0, 0]
                })
        
        return fit_results

class iMSDAnalysis(BaseAnalysis):
    """Image Mean Square Displacement Analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "iMSD"
        self.description = "Extracts diffusion and anomalous transport behavior without particle tracking"
    
    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform iMSD analysis"""
        
        image_data = data_info['image_data']
        pixel_size = data_info.get('pixel_size', 0.1)
        time_interval = data_info.get('time_interval', 0.1)
        
        # Extract parameters
        max_displacement = parameters.get('max_displacement', 20)
        min_track_length = parameters.get('min_track_length', 5)
        
        # Calculate MSD maps
        msd_curves, msd_maps = self._calculate_msd_maps(image_data, max_displacement)
        
        # Analyze diffusion behavior
        diffusion_analysis = self._analyze_diffusion_behavior(msd_curves, time_interval, pixel_size)
        
        results = {
            'msd_curves': msd_curves,
            'msd_maps': msd_maps,
            'diffusion_analysis': diffusion_analysis,
            'analysis_summary': {
                'mean_diffusion_coefficient': diffusion_analysis.get('mean_diffusion', 0),
                'anomalous_exponent': diffusion_analysis.get('mean_alpha', 1),
                'num_regions_analyzed': len(msd_curves)
            }
        }
        
        return results
    
    def _calculate_msd_maps(self, image_data: np.ndarray, max_displacement: int) -> Tuple[list, np.ndarray]:
        """Calculate proper feature-based MSD using particle detection and tracking"""
        
        if len(image_data.shape) == 2:
            st.warning("iMSD requires time series data")
            return [], np.array([])
        
        if len(image_data.shape) == 3:
            t_frames, height, width = image_data.shape
        else:
            # Take first channel if multi-channel
            t_frames, height, width = image_data.shape[0], image_data.shape[1], image_data.shape[2]
            image_data = image_data[:, :, :, 0] if len(image_data.shape) == 4 else image_data
        
        try:
            # Use trackpy for proper feature-based MSD analysis
            import trackpy as tp
            
            # Detect features in each frame
            features_list = []
            for t in range(t_frames):
                frame = image_data[t]
                
                # Detect bright spots (particles/features)
                # Use adaptive parameters based on image properties
                feature_size = max(3, min(height, width) // 50)  # Adaptive feature size
                threshold = np.percentile(frame, 95)  # Top 5% intensity as threshold
                
                features = tp.locate(frame, feature_size, minmass=threshold)
                features['frame'] = t
                features_list.append(features)
            
            if not features_list or all(len(f) == 0 for f in features_list):
                # Fallback: create synthetic features from intensity maxima
                return self._fallback_intensity_fluctuation_msd(image_data, max_displacement)
            
            # Combine all features
            all_features = pd.concat(features_list, ignore_index=True)
            
            # Link features into trajectories
            search_range = max(2, min(height, width) // 100)  # Adaptive search range
            trajectories = tp.link(all_features, search_range, memory=2)
            
            # Filter trajectories by minimum length
            min_length = max(3, t_frames // 4)
            trajectories = tp.filter_stubs(trajectories, min_length)
            
            if len(trajectories) == 0:
                return self._fallback_intensity_fluctuation_msd(image_data, max_displacement)
            
            # Calculate ensemble MSD
            msd_data = tp.imsd(trajectories, mpp=1.0, fps=1.0, max_lagtime=max_displacement)
            
            # Extract MSD curves for different regions/particles
            msd_curves = []
            individual_msds = tp.imsd(trajectories, mpp=1.0, fps=1.0, max_lagtime=max_displacement, detail=True)
            
            for particle_id in individual_msds['particle'].unique():
                particle_msd = individual_msds[individual_msds['particle'] == particle_id]
                if len(particle_msd) > 2:
                    msd_curves.append(particle_msd['msd'].values)
            
            # Create spatial MSD map by binning particle locations
            msd_maps = self._create_spatial_msd_map(trajectories, individual_msds, height, width, max_displacement)
            
            return msd_curves, msd_maps
            
        except ImportError:
            st.warning("trackpy not available - using simplified intensity fluctuation analysis")
            return self._fallback_intensity_fluctuation_msd(image_data, max_displacement)
        except Exception as e:
            st.warning(f"Feature tracking failed: {e} - using intensity fluctuation fallback")
            return self._fallback_intensity_fluctuation_msd(image_data, max_displacement)
    
    def _fallback_intensity_fluctuation_msd(self, image_data: np.ndarray, max_displacement: int) -> Tuple[list, np.ndarray]:
        """Fallback method using intensity fluctuation analysis (similar to imaging FCS)"""
        
        t_frames, height, width = image_data.shape
        
        # Calculate intensity fluctuation correlation instead of incorrect pixel MSD
        # This is more appropriate for intensity-based data
        msd_curves = []
        correlation_maps = np.zeros((max_displacement, height, width))
        
        # Calculate temporal autocorrelation for intensity fluctuations
        mean_intensity = np.mean(image_data, axis=0)
        fluctuations = image_data - mean_intensity[np.newaxis, :, :]
        
        # Sample regions across the image
        region_size = max(8, min(height, width) // 10)
        step_size = region_size // 2
        
        for i in range(0, height - region_size, step_size):
            for j in range(0, width - region_size, step_size):
                # Extract region time trace
                region_trace = np.mean(fluctuations[:, i:i+region_size, j:j+region_size], axis=(1, 2))
                
                # Calculate temporal autocorrelation
                correlation = self._calculate_temporal_autocorr(region_trace, max_displacement)
                msd_curves.append(correlation)
                
                # Map correlation values spatially
                for tau_idx, corr_val in enumerate(correlation):
                    if tau_idx < max_displacement:
                        correlation_maps[tau_idx, i:i+region_size, j:j+region_size] = corr_val
        
        return msd_curves, correlation_maps
    
    def _calculate_temporal_autocorr(self, trace: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate temporal autocorrelation for intensity fluctuations"""
        
        n_points = len(trace)
        autocorr = np.zeros(min(max_lag, n_points // 2))
        
        for tau in range(len(autocorr)):
            if tau == 0:
                autocorr[tau] = np.var(trace)
            else:
                # Temporal correlation at lag tau
                valid_points = n_points - tau
                correlation_sum = 0
                for t in range(valid_points):
                    correlation_sum += trace[t] * trace[t + tau]
                autocorr[tau] = correlation_sum / valid_points
        
        # Normalize by zero-lag value
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        return autocorr
    
    def _create_spatial_msd_map(self, trajectories, individual_msds, height: int, width: int, max_displacement: int) -> np.ndarray:
        """Create spatial MSD map by binning particle trajectories"""
        
        msd_maps = np.zeros((max_displacement, height, width))
        count_maps = np.zeros((height, width))
        
        # Bin particles by their average position
        bin_size = max(5, min(height, width) // 20)
        
        for particle_id in individual_msds['particle'].unique():
            particle_data = trajectories[trajectories['particle'] == particle_id]
            particle_msd = individual_msds[individual_msds['particle'] == particle_id]
            
            if len(particle_data) == 0 or len(particle_msd) == 0:
                continue
            
            # Average position of this particle
            avg_x = int(np.mean(particle_data['x']))
            avg_y = int(np.mean(particle_data['y']))
            
            # Ensure coordinates are within bounds
            avg_x = max(0, min(width - 1, avg_x))
            avg_y = max(0, min(height - 1, avg_y))
            
            # Add MSD values to spatial bins
            y_min = max(0, avg_y - bin_size // 2)
            y_max = min(height, avg_y + bin_size // 2)
            x_min = max(0, avg_x - bin_size // 2)
            x_max = min(width, avg_x + bin_size // 2)
            
            for tau_idx, msd_val in enumerate(particle_msd['msd'].values):
                if tau_idx < max_displacement:
                    msd_maps[tau_idx, y_min:y_max, x_min:x_max] += msd_val
                    count_maps[y_min:y_max, x_min:x_max] += 1
        
        # Normalize by counts
        for tau_idx in range(max_displacement):
            mask = count_maps > 0
            msd_maps[tau_idx, mask] /= count_maps[mask]
        
        return msd_maps
    
    def _analyze_diffusion_behavior(self, msd_curves: list, time_interval: float, pixel_size: float) -> Dict[str, Any]:
        """Analyze diffusion behavior from MSD curves"""
        
        diffusion_coefficients = []
        anomalous_exponents = []
        
        for msd_curve in msd_curves:
            if len(msd_curve) < 3:
                continue
            
            # Fit power law: MSD = 4*D*t^α
            tau_values = np.arange(1, len(msd_curve)) * time_interval
            msd_values = msd_curve[1:]  # Skip tau=0
            
            if np.any(msd_values > 0):
                try:
                    # Log-log fit for power law
                    log_tau = np.log(tau_values)
                    log_msd = np.log(msd_values + 1e-10)  # Avoid log(0)
                    
                    coeffs = np.polyfit(log_tau, log_msd, 1)
                    alpha = coeffs[0]  # Anomalous exponent
                    log_d = coeffs[1]
                    
                    # Convert back to diffusion coefficient
                    D = np.exp(log_d) / 4 * (pixel_size ** 2)
                    
                    anomalous_exponents.append(alpha)
                    diffusion_coefficients.append(D)
                
                except Exception:
                    continue
        
        return {
            'mean_diffusion': np.mean(diffusion_coefficients) if diffusion_coefficients else 0,
            'std_diffusion': np.std(diffusion_coefficients) if diffusion_coefficients else 0,
            'mean_alpha': np.mean(anomalous_exponents) if anomalous_exponents else 1,
            'std_alpha': np.std(anomalous_exponents) if anomalous_exponents else 0,
            'diffusion_coefficients': diffusion_coefficients,
            'anomalous_exponents': anomalous_exponents
        }

# Placeholder implementations for remaining analysis modules
class ElastographyPIVAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.name = "Elastography & PIV"
        self.description = "Estimates viscoelastic properties and motion mapping"
        # Import optical flow analyzer for PIV functionality
        try:
            from optical_flow_analysis import OpticalFlowAnalyzer
            self.flow_analyzer = OpticalFlowAnalyzer()
            self.available = True
        except ImportError:
            self.available = False

    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform PIV and Elastography using Dense Optical Flow (Farneback) method"""
        if not self.available:
            return {
                'status': 'error',
                'message': 'Optical flow analyzer not available'
            }
            
        image_data = data_info.get('image_data')
        if image_data.ndim < 3:
            return {
                'status': 'error', 
                'message': 'This analysis requires a time-series image stack.'
            }

        # Use Farneback dense optical flow as the basis for PIV
        try:
            from optical_flow_analysis import get_optical_flow_parameters
            flow_params = get_optical_flow_parameters('Dense Optical Flow (Farneback)')
            
            # Run optical flow analysis
            flow_results = self.flow_analyzer.analyze_optical_flow(
                'Dense Optical Flow (Farneback)', image_data, flow_params
            )

            if flow_results['status'] != 'success':
                return flow_results

            # Calculate strain map from the flow field (simplified elastography)
            flow_field = flow_results['flow_fields'][0]
            flow_x, flow_y = flow_field['flow_x'], flow_field['flow_y']
            
            # Calculate gradients for strain analysis
            from scipy import ndimage
            grad_x = ndimage.sobel(flow_x, axis=1)
            grad_y = ndimage.sobel(flow_y, axis=0)
            
            # Divergence of the vector field represents local expansion/contraction
            strain_map = grad_x + grad_y
            avg_strain = np.mean(strain_map)
            
            flow_results['strain_map'] = strain_map
            flow_results['average_strain'] = avg_strain
            flow_results['analysis_summary'] = {
                'average_displacement': flow_results.get('avg_displacement', 0),
                'average_strain': avg_strain,
                'strain_std': np.std(strain_map)
            }
            
            return flow_results
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Elastography analysis failed: {str(e)}'
            }

class NBAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.name = "Number & Brightness"
        self.description = "Determines molecular aggregation states from intensity fluctuations"

    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Number and Brightness (N&B) analysis with nuclear alignment"""
        image_data = data_info.get('image_data')
        
        if image_data.ndim < 3:
            return {
                'status': 'error', 
                'message': 'N&B analysis requires a time-series image stack.'
            }
        
        # Apply nuclear alignment before N&B analysis if enabled
        enable_alignment = parameters.get('nuclear_alignment', True)
        if enable_alignment:
            try:
                import nuclear_alignment
                
                # Perform nuclear alignment using dedicated alignment module
                alignment_params = {
                    'alignment_method': 'phase_correlation',
                    'reference_frame': 0
                }
                
                aligned_result = nuclear_alignment.align_nuclear_sequence(
                    image_data, 
                    alignment_method=alignment_params['alignment_method'],
                    reference_frame=alignment_params['reference_frame']
                )
                
                if aligned_result.get('status') == 'success' and 'aligned_sequence' in aligned_result:
                    processed_data = self.preprocess_data(aligned_result['aligned_sequence'], parameters).astype(np.float32)
                    alignment_applied = True
                    alignment_quality = aligned_result.get('quality_metrics', {})
                else:
                    # Fallback to unaligned data if alignment fails
                    processed_data = self.preprocess_data(image_data, parameters).astype(np.float32)
                    alignment_applied = False
                    alignment_quality = {}
                    
            except Exception as e:
                # Fallback to unaligned data if alignment module fails
                processed_data = self.preprocess_data(image_data, parameters).astype(np.float32)
                alignment_applied = False
                alignment_quality = {}
        else:
            # Skip alignment if disabled
            processed_data = self.preprocess_data(image_data, parameters).astype(np.float32)
            alignment_applied = False
            alignment_quality = {}
        
        # Calculate mean and variance for each pixel across the time series
        mean_intensity_map = np.mean(processed_data, axis=0)
        variance_map = np.var(processed_data, axis=0)
        
        # Avoid division by zero
        mean_intensity_map[mean_intensity_map == 0] = 1e-9
        
        # Calculate apparent brightness (B) and number (N)
        # B = variance / mean
        # N = mean^2 / variance
        brightness_map = variance_map / mean_intensity_map
        number_map = np.square(mean_intensity_map) / variance_map
        
        # Handle potential NaNs or Infs from division by zero in variance
        brightness_map = np.nan_to_num(brightness_map, nan=0.0, posinf=0.0, neginf=0.0)
        number_map = np.nan_to_num(number_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate quality metrics for aligned vs unaligned comparison
        alignment_improvement = 0
        if alignment_applied and alignment_quality:
            # Estimate improvement from reduced motion artifacts
            frame_correlations = []
            for i in range(min(10, processed_data.shape[0] - 1)):
                corr = np.corrcoef(processed_data[i].flatten(), processed_data[i+1].flatten())[0,1]
                if not np.isnan(corr):
                    frame_correlations.append(corr)
            
            if frame_correlations:
                alignment_improvement = np.mean(frame_correlations)

        return {
            'status': 'success',
            'brightness_map': brightness_map,
            'number_map': number_map,
            'mean_intensity_map': mean_intensity_map,
            'variance_map': variance_map,
            'alignment_applied': alignment_applied,
            'alignment_quality': alignment_quality,
            'analysis_summary': {
                'average_brightness': np.mean(brightness_map),
                'average_number': np.mean(number_map),
                'brightness_std': np.std(brightness_map),
                'number_std': np.std(number_map),
                'nuclear_alignment_applied': alignment_applied,
                'frame_stability_score': alignment_improvement,
                'analysis_reliability': 'High' if alignment_applied else 'Moderate - recommend enabling nuclear alignment'
            }
        }

class FLIMAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.name = "FLIM"
        self.description = "Fluorescence Lifetime Imaging Microscopy analysis"

    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform FLIM analysis.
        NOTE: This treats the time-axis of the input data as photon arrival times.
        Real FLIM data would typically be TCSPC histogram for each pixel.
        """
        image_data = data_info.get('image_data')
        if image_data.ndim < 3:
            return {
                'status': 'error', 
                'message': 'FLIM analysis requires a time-series image stack.'
            }

        st.info("Simulating FLIM analysis. The time axis is treated as photon arrival time bins.")
        
        time_bins = np.arange(image_data.shape[0])
        lifetime_map = np.zeros((image_data.shape[1], image_data.shape[2]))

        # Define a single exponential decay model to fit
        def exp_decay(t, a, tau):
            return a * np.exp(-t / tau)

        # Fit decay for each pixel
        for y in range(image_data.shape[1]):
            for x in range(image_data.shape[2]):
                decay_curve = image_data[:, y, x]
                if np.sum(decay_curve) > 0:
                    try:
                        params, _ = optimize.curve_fit(
                            exp_decay, time_bins, decay_curve, 
                            p0=[np.max(decay_curve), 2.5],
                            maxfev=1000
                        )
                        lifetime_map[y, x] = params[1]  # tau
                    except (RuntimeError, ValueError):
                        lifetime_map[y, x] = 0  # Fit failed
        
        # Filter out unrealistic lifetime values
        lifetime_map[lifetime_map < 0] = 0
        lifetime_map[lifetime_map > 10] = 0  # Assuming lifetimes are in ns range

        # Calculate statistics
        valid_lifetimes = lifetime_map[lifetime_map > 0]
        
        return {
            'status': 'success',
            'lifetime_map': lifetime_map,
            'analysis_summary': {
                'average_lifetime_ns': np.mean(valid_lifetimes) if len(valid_lifetimes) > 0 else 0,
                'std_lifetime_ns': np.std(valid_lifetimes) if len(valid_lifetimes) > 0 else 0,
                'valid_pixels': len(valid_lifetimes),
                'total_pixels': lifetime_map.size
            }
        }

class SPTAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.name = "Single Particle Tracking"
        self.description = "Tracks motion of individual particles using trackpy"
    
    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SPT analysis using trackpy library"""
        
        if not TRACKPY_AVAILABLE:
            return {'status': 'error', 'message': 'trackpy library required for SPT analysis'}
        
        image_data = data_info['image_data']
        time_interval = data_info.get('time_interval', 0.1)
        pixel_size = data_info.get('pixel_size', 0.1)
        
        # Extract parameters
        particle_size = parameters.get('particle_size', 5)
        search_range = parameters.get('search_range', 5)
        memory = parameters.get('memory', 3)
        min_mass = parameters.get('min_mass', 100)
        min_trajectory_length = parameters.get('min_trajectory_length', 10)
        
        # Handle multichannel data
        analysis_channels = parameters.get('analysis_channels', [0])
        channel_names = parameters.get('channel_names', ['Channel 1'])
        
        results = {}
        
        for i, channel_idx in enumerate(analysis_channels):
            # Extract channel data
            if len(image_data.shape) == 4:  # T, Y, X, C
                if channel_idx < image_data.shape[-1]:
                    channel_data = image_data[:, :, :, channel_idx]
                else:
                    continue
            elif len(image_data.shape) == 3 and data_info.get('channels', 1) > 1:
                if channel_idx < image_data.shape[-1]:
                    channel_data = image_data[:, :, channel_idx]
                else:
                    continue
            else:
                channel_data = image_data
            
            # Ensure we have time series data
            if len(channel_data.shape) == 2:
                # Single frame - cannot do tracking
                channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
                results[channel_name] = {
                    'status': 'error',
                    'message': 'SPT requires time series data (multiple frames)'
                }
                continue
            
            # Preprocess data
            processed_data = self.preprocess_data(channel_data, parameters)
            
            try:
                # Step 1: Locate particles in each frame using trackpy
                all_features = []
                for frame_idx in range(processed_data.shape[0]):
                    frame = processed_data[frame_idx]
                    
                    # Locate particles in this frame
                    features = trackpy.locate(frame, particle_size, minmass=min_mass)
                    features['frame'] = frame_idx
                    all_features.append(features)
                
                # Combine all features
                if all_features:
                    features_df = pd.concat(all_features, ignore_index=True)
                else:
                    features_df = pd.DataFrame()
                
                if len(features_df) == 0:
                    channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
                    results[channel_name] = {
                        'status': 'error',
                        'message': 'No particles detected in any frame'
                    }
                    continue
                
                # Step 2: Link particles across frames to form trajectories
                trajectories = trackpy.link_df(features_df, search_range, memory=memory)
                
                # Step 3: Filter short trajectories
                trajectories_filtered = trackpy.filter_stubs(trajectories, min_trajectory_length)
                
                if len(trajectories_filtered) == 0:
                    channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
                    results[channel_name] = {
                        'status': 'error',
                        'message': f'No trajectories longer than {min_trajectory_length} frames found'
                    }
                    continue
                
                # Step 4: Calculate Mean Square Displacement (MSD) using trackpy
                msd_data = trackpy.imsd(trajectories_filtered, mpp=pixel_size, fps=1/time_interval)
                
                # Step 5: Calculate mobility metrics
                mobility_metrics = self._calculate_trackpy_mobility_metrics(
                    trajectories_filtered, msd_data, pixel_size, time_interval
                )
                
                # Step 6: Diffusion analysis
                diffusion_analysis = self._analyze_diffusion_behavior(
                    trajectories_filtered, msd_data, pixel_size, time_interval
                )
                
                # Store channel-specific results
                channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
                results[channel_name] = {
                    'trajectories': trajectories_filtered,
                    'msd_data': msd_data,
                    'mobility_metrics': mobility_metrics,
                    'diffusion_analysis': diffusion_analysis,
                    'channel_index': channel_idx,
                    'analysis_summary': {
                        'num_particles': len(trajectories_filtered['particle'].unique()),
                        'num_frames': len(trajectories_filtered['frame'].unique()),
                        'mean_trajectory_length': trajectories_filtered.groupby('particle').size().mean(),
                        'mean_diffusion_coefficient': diffusion_analysis.get('mean_diffusion_coefficient', 0),
                        'mean_displacement_per_frame': mobility_metrics.get('mean_displacement_per_frame', 0)
                    }
                }
                
            except Exception as e:
                channel_name = channel_names[i] if i < len(channel_names) else f"Channel {channel_idx + 1}"
                results[channel_name] = {
                    'status': 'error',
                    'message': f'SPT analysis failed: {str(e)}'
                }
        
        # Add multichannel summary if multiple channels
        if len(analysis_channels) > 1:
            successful_channels = [ch for ch, result in results.items() 
                                 if isinstance(result, dict) and 'status' not in result]
            
            if successful_channels:
                all_diffusion_coeffs = []
                all_particle_counts = []
                
                for ch in successful_channels:
                    summary = results[ch]['analysis_summary']
                    if summary.get('mean_diffusion_coefficient', 0) > 0:
                        all_diffusion_coeffs.append(summary['mean_diffusion_coefficient'])
                    all_particle_counts.append(summary['num_particles'])
                
                results['multichannel_summary'] = {
                    'channels_analyzed': len(successful_channels),
                    'successful_channels': successful_channels,
                    'total_particles_all_channels': sum(all_particle_counts),
                    'mean_diffusion_coefficient_all_channels': np.mean(all_diffusion_coeffs) if all_diffusion_coeffs else 0,
                    'diffusion_variation_between_channels': np.std(all_diffusion_coeffs) if all_diffusion_coeffs else 0
                }
        
        return results
    
    def _calculate_trackpy_mobility_metrics(self, trajectories: pd.DataFrame, msd_data: pd.DataFrame, 
                                          pixel_size: float, time_interval: float) -> Dict[str, Any]:
        """Calculate mobility metrics from trackpy trajectories"""
        
        # Basic trajectory statistics
        particle_counts = trajectories.groupby('particle').size()
        
        # Calculate step sizes
        step_sizes = []
        for particle_id in trajectories['particle'].unique():
            particle_traj = trajectories[trajectories['particle'] == particle_id].sort_values('frame')
            dx = np.diff(particle_traj['x'].values) * pixel_size
            dy = np.diff(particle_traj['y'].values) * pixel_size
            step_size = np.sqrt(dx**2 + dy**2)
            step_sizes.extend(step_size)
        
        metrics = {
            'num_particles': len(trajectories['particle'].unique()),
            'mean_trajectory_length': particle_counts.mean(),
            'std_trajectory_length': particle_counts.std(),
            'mean_step_size': np.mean(step_sizes) if step_sizes else 0,
            'std_step_size': np.std(step_sizes) if step_sizes else 0,
            'mean_displacement_per_frame': np.mean(step_sizes) if step_sizes else 0,
            'total_displacement': np.sum(step_sizes) if step_sizes else 0
        }
        
        return metrics
    
    def _analyze_diffusion_behavior(self, trajectories: pd.DataFrame, msd_data: pd.DataFrame,
                                  pixel_size: float, time_interval: float) -> Dict[str, Any]:
        """Analyze diffusion behavior from MSD data"""
        
        try:
            # Fit linear model to MSD vs time to extract diffusion coefficients
            diffusion_coefficients = []
            
            # Analyze ensemble MSD
            if 'msd' in msd_data.columns:
                # Use first few points for linear fit (typically first 25% of data)
                n_points = min(10, len(msd_data) // 4)
                if n_points >= 3:
                    time_points = msd_data.index[:n_points] * time_interval
                    msd_values = msd_data['msd'].iloc[:n_points]
                    
                    # Linear fit: MSD = 4*D*t (for 2D diffusion)
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(time_points, msd_values)
                    diffusion_coeff = slope / 4  # Convert slope to diffusion coefficient
                    
                    diffusion_coefficients.append(diffusion_coeff)
            
            analysis = {
                'mean_diffusion_coefficient': np.mean(diffusion_coefficients) if diffusion_coefficients else 0,
                'std_diffusion_coefficient': np.std(diffusion_coefficients) if diffusion_coefficients else 0,
                'num_analyzed_particles': len(diffusion_coefficients),
                'msd_linearity': r_value**2 if 'r_value' in locals() else 0
            }
            
            return analysis
            
        except Exception as e:
            return {
                'mean_diffusion_coefficient': 0,
                'std_diffusion_coefficient': 0,
                'num_analyzed_particles': 0,
                'msd_linearity': 0,
                'error': str(e)
            }

class FourierAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.name = "Fourier Transform Texture Analysis"
        self.description = "Analyzes spatial frequency content for texture and organization"

    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Fourier Transform Texture Analysis"""
        image_data = data_info.get('image_data')
        
        # If time-series, analyze the first frame
        if image_data.ndim > 2:
            st.info("Analyzing the first frame of the stack.")
            image_data = image_data[0]
            
        # Perform 2D Fast Fourier Transform
        f_transform = np.fft.fft2(image_data)
        f_transform_shifted = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_transform_shifted)**2
        log_power_spectrum = np.log(power_spectrum + 1)  # Add 1 to avoid log(0)

        # Calculate radial power spectrum
        center = (image_data.shape[0] // 2, image_data.shape[1] // 2)
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Bin the power spectrum by radius
        max_radius = min(center)
        radial_profile = np.zeros(max_radius)
        for r in range(max_radius):
            mask = (radius == r)
            if np.any(mask):
                radial_profile[r] = np.mean(power_spectrum[mask])

        return {
            'status': 'success',
            'power_spectrum': power_spectrum,
            'log_power_spectrum': log_power_spectrum,
            'radial_profile': radial_profile,
            'analysis_summary': {
                'total_power': np.sum(power_spectrum),
                'peak_frequency': np.argmax(radial_profile),
                'dominant_wavelength': 2 * np.pi / (np.argmax(radial_profile) + 1) if np.argmax(radial_profile) > 0 else 0
            }
        }

class FRAPAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.name = "FRAP"
        self.description = "Fluorescence Recovery After Photobleaching analysis"

    def analyze(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform FRAP analysis"""
        image_data = data_info.get('image_data')
        
        if image_data.ndim < 3:
            return {
                'status': 'error', 
                'message': 'FRAP analysis requires a time-series image stack.'
            }
        
        # For this implementation, assume ROI is the central 1/4 of the image
        h, w = image_data.shape[1], image_data.shape[2]
        roi_coords = [w//4, h//4, w*3//4, h*3//4]  # x_start, y_start, x_end, y_end
        st.info(f"Using a central region of interest (ROI) at {roi_coords} for analysis.")

        # Extract intensity in the ROI over time
        roi_data = image_data[:, roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
        recovery_curve = np.mean(roi_data, axis=(1, 2))

        # Find the pre-bleach intensity (average of first few frames) and bleach point
        pre_bleach_intensity = np.mean(recovery_curve[:3])
        bleach_point_index = np.argmin(recovery_curve)
        
        # Normalize the recovery curve
        post_bleach_curve = recovery_curve[bleach_point_index:]
        if len(post_bleach_curve) > 1:
            normalized_curve = (post_bleach_curve - recovery_curve[bleach_point_index]) / \
                               (pre_bleach_intensity - recovery_curve[bleach_point_index])
        else:
            normalized_curve = np.array([0])
        
        # Fit to a single exponential recovery model: F(t) = F_inf * (1 - exp(-k*t))
        time_axis = (np.arange(len(normalized_curve))) * data_info.get('time_interval', 1)
        
        def recovery_model(t, f_inf, k):
            return f_inf * (1 - np.exp(-t * k))

        try:
            if len(normalized_curve) > 3:
                params, _ = optimize.curve_fit(
                    recovery_model, time_axis, normalized_curve, 
                    p0=[0.8, 0.1], maxfev=1000
                )
                mobile_fraction = params[0]
                halftime = np.log(2) / params[1] if params[1] > 0 else 0
            else:
                mobile_fraction = 0
                halftime = 0
        except (RuntimeError, ValueError):
            mobile_fraction = 0
            halftime = 0

        return {
            'status': 'success',
            'recovery_curve': recovery_curve,
            'normalized_curve': normalized_curve,
            'time_axis': time_axis,
            'roi_coords': roi_coords,
            'analysis_summary': {
                'mobile_fraction': mobile_fraction,
                'recovery_halftime_s': halftime,
                'pre_bleach_intensity': pre_bleach_intensity,
                'bleach_point_index': bleach_point_index
            }
        }
