"""
Image Correlation Spectroscopy Module
Implementation of RICS, STICS, imaging FCS and related correlation analysis methods
Based on state-of-the-art repositories and publications
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings
from scipy import ndimage, optimize
from scipy.fft import fft2, ifft2, fftshift
from skimage.filters import gaussian
from skimage.measure import regionprops

class ImageCorrelationSpectroscopy:
    """Advanced image correlation spectroscopy analysis suite"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which ICS methods are available"""
        return {
            'RICS (Raster Image Correlation Spectroscopy)': True,
            'STICS (Spatio-Temporal Image Correlation Spectroscopy)': True,
            'Imaging FCS': True,
            'Pair Correlation Function (pCF)': True,
            'Auto-correlation ICS': True,
            'Cross-correlation ICS': True,
            'iMSD via ICS': True,
            'Temporal ICS': True
        }
    
    def get_available_methods(self) -> List[str]:
        """Return list of available ICS methods"""
        return [method for method, available in self.available_methods.items() if available]
    
    def analyze_ics_method(self, method: str, image_data: np.ndarray, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply image correlation spectroscopy method"""
        
        if method not in self.get_available_methods():
            return {
                'status': 'error',
                'message': f'Method {method} not available'
            }
        
        try:
            if method == 'RICS (Raster Image Correlation Spectroscopy)':
                return self._analyze_rics(image_data, parameters)
            elif method == 'STICS (Spatio-Temporal Image Correlation Spectroscopy)':
                return self._analyze_stics(image_data, parameters)
            elif method == 'Imaging FCS':
                return self._analyze_imaging_fcs(image_data, parameters)
            elif method == 'Pair Correlation Function (pCF)':
                return self._analyze_pair_correlation(image_data, parameters)
            elif method == 'Auto-correlation ICS':
                return self._analyze_autocorr_ics(image_data, parameters)
            elif method == 'Cross-correlation ICS':
                return self._analyze_crosscorr_ics(image_data, parameters)
            elif method == 'iMSD via ICS':
                return self._analyze_imsd_ics(image_data, parameters)
            elif method == 'Temporal ICS':
                return self._analyze_temporal_ics(image_data, parameters)
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown method: {method}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error applying {method}: {str(e)}'
            }
    
    def _analyze_rics(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Raster Image Correlation Spectroscopy analysis"""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'RICS requires time series data (3D array)'}
            
            # Parameters
            pixel_size = parameters.get('pixel_size', 0.1)  # microns
            line_time = parameters.get('line_time', 0.001)  # seconds per line
            tau_max = parameters.get('tau_max', 10)  # maximum correlation lag
            eta_max = parameters.get('eta_max', 10)  # maximum spatial lag
            
            T, H, W = image_data.shape
            
            # Calculate RICS correlation function
            rics_correlation = self._calculate_rics_correlation(
                image_data, tau_max, eta_max
            )
            
            # Fit RICS model to extract diffusion parameters
            diffusion_results = self._fit_rics_diffusion_model(
                rics_correlation, pixel_size, line_time, tau_max, eta_max
            )
            
            # Calculate additional metrics
            avg_intensity = np.mean(image_data)
            intensity_variance = np.var(image_data)
            
            return {
                'status': 'success',
                'method': 'RICS (Raster Image Correlation Spectroscopy)',
                'correlation_function': rics_correlation,
                'diffusion_coefficient': diffusion_results.get('D', 0),
                'number_of_particles': diffusion_results.get('N', 0),
                'fit_quality': diffusion_results.get('r_squared', 0),
                'avg_intensity': avg_intensity,
                'intensity_variance': intensity_variance,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'RICS analysis failed: {str(e)}'}
    
    def _calculate_rics_correlation(self, image_data: np.ndarray, 
                                   tau_max: int, eta_max: int) -> np.ndarray:
        """Calculate RICS spatial correlation function"""
        T, H, W = image_data.shape
        
        # Calculate mean intensity for each frame
        mean_intensities = np.mean(image_data, axis=(1, 2))
        
        # Calculate intensity fluctuations
        delta_I = image_data - mean_intensities[:, np.newaxis, np.newaxis]
        
        # Initialize correlation array
        correlation = np.zeros((2 * tau_max + 1, 2 * eta_max + 1))
        
        # Calculate spatial correlation for each time point
        for tau in range(-tau_max, tau_max + 1):
            for eta in range(-eta_max, eta_max + 1):
                
                correlation_sum = 0
                count = 0
                
                for t in range(T):
                    frame = delta_I[t]
                    
                    # Calculate valid region for correlation
                    y_start = max(0, -tau)
                    y_end = min(H, H - tau)
                    x_start = max(0, -eta)
                    x_end = min(W, W - eta)
                    
                    if y_start < y_end and x_start < x_end:
                        # Calculate correlation for this shift
                        ref_region = frame[y_start:y_end, x_start:x_end]
                        shifted_region = frame[y_start + tau:y_end + tau, 
                                             x_start + eta:x_end + eta]
                        
                        correlation_sum += np.sum(ref_region * shifted_region)
                        count += ref_region.size
                
                if count > 0:
                    correlation[tau + tau_max, eta + eta_max] = correlation_sum / count
        
        return correlation
    
    def _fit_rics_diffusion_model(self, correlation: np.ndarray, pixel_size: float,
                                 line_time: float, tau_max: int, eta_max: int) -> Dict[str, Any]:
        """Fit 2D diffusion model to RICS correlation data"""
        
        # Create coordinate arrays
        tau_coords = np.arange(-tau_max, tau_max + 1) * pixel_size
        eta_coords = np.arange(-eta_max, eta_max + 1) * line_time
        
        TAU, ETA = np.meshgrid(tau_coords, eta_coords, indexing='ij')
        
        # RICS diffusion model function
        def rics_model(coords, D, N, w0, offset):
            tau, eta = coords
            # 2D Gaussian diffusion model for RICS
            denominator = 1 + 4 * D * np.abs(eta) / (w0**2)
            return (N / denominator) * np.exp(-tau**2 / (w0**2 * denominator)) + offset
        
        # Flatten arrays for fitting
        tau_flat = TAU.flatten()
        eta_flat = ETA.flatten()
        corr_flat = correlation.flatten()
        
        # Initial parameter guess
        p0 = [1.0, 100, 0.3, np.min(corr_flat)]  # D, N, w0, offset
        
        try:
            # Fit the model
            popt, pcov = optimize.curve_fit(
                lambda coords, *p: rics_model(coords, *p),
                (tau_flat, eta_flat),
                corr_flat,
                p0=p0,
                maxfev=5000
            )
            
            D, N, w0, offset = popt
            
            # Calculate R-squared
            y_pred = rics_model((tau_flat, eta_flat), *popt)
            ss_res = np.sum((corr_flat - y_pred) ** 2)
            ss_tot = np.sum((corr_flat - np.mean(corr_flat)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'D': abs(D),  # Diffusion coefficient
                'N': abs(N),  # Number of particles
                'w0': abs(w0),  # Beam waist
                'offset': offset,
                'r_squared': r_squared,
                'fit_parameters': popt,
                'fit_covariance': pcov
            }
            
        except Exception:
            return {
                'D': 0,
                'N': 0,
                'w0': 0,
                'offset': 0,
                'r_squared': 0,
                'fit_parameters': None,
                'fit_covariance': None
            }
    
    def _analyze_stics(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Spatio-Temporal Image Correlation Spectroscopy analysis"""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'STICS requires time series data (3D array)'}
            
            max_spatial_lag = parameters.get('max_spatial_lag', 5)
            max_temporal_lag = parameters.get('max_temporal_lag', 5)
            pixel_size = parameters.get('pixel_size', 0.1)  # microns
            time_interval = parameters.get('time_interval', 0.1)  # seconds
            
            # Calculate STICS correlation function
            stics_correlation = self._calculate_stics_correlation_advanced(
                image_data, max_spatial_lag, max_temporal_lag
            )
            
            # Extract flow velocity maps
            flow_results = self._extract_stics_flow_velocity(
                stics_correlation, pixel_size, time_interval, 
                max_spatial_lag, max_temporal_lag
            )
            
            # Calculate diffusion maps
            diffusion_maps = self._calculate_stics_diffusion_maps(
                stics_correlation, pixel_size, time_interval
            )
            
            return {
                'status': 'success',
                'method': 'STICS (Spatio-Temporal Image Correlation Spectroscopy)',
                'correlation_function': stics_correlation,
                'flow_velocity_x': flow_results.get('velocity_x', 0),
                'flow_velocity_y': flow_results.get('velocity_y', 0),
                'flow_magnitude': flow_results.get('magnitude', 0),
                'diffusion_coefficient': diffusion_maps.get('D', 0),
                'mobile_fraction': diffusion_maps.get('mobile_fraction', 0),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'STICS analysis failed: {str(e)}'}
    
    def _calculate_stics_correlation_advanced(self, image_data: np.ndarray,
                                            max_spatial_lag: int, max_temporal_lag: int) -> np.ndarray:
        """Calculate enhanced STICS correlation function using FFT"""
        T, H, W = image_data.shape
        
        # Normalize images
        normalized_data = np.zeros_like(image_data)
        for t in range(T):
            frame = image_data[t]
            mean_val = np.mean(frame)
            if mean_val > 0:
                normalized_data[t] = (frame - mean_val) / mean_val
        
        # Initialize correlation function
        correlation_shape = (max_temporal_lag + 1, 
                           2 * max_spatial_lag + 1, 
                           2 * max_spatial_lag + 1)
        stics_correlation = np.zeros(correlation_shape)
        
        # Calculate correlation for each temporal lag
        for tau in range(max_temporal_lag + 1):
            if T - tau <= 1:
                continue
                
            correlation_sum = np.zeros((2 * max_spatial_lag + 1, 2 * max_spatial_lag + 1))
            
            for t in range(T - tau):
                frame1 = normalized_data[t]
                frame2 = normalized_data[t + tau]
                
                # Calculate spatial cross-correlation using FFT
                f1_fft = fft2(frame1, s=(H + 2*max_spatial_lag, W + 2*max_spatial_lag))
                f2_fft = fft2(frame2, s=(H + 2*max_spatial_lag, W + 2*max_spatial_lag))
                
                # Cross-correlation in frequency domain
                cross_corr = ifft2(f1_fft * np.conj(f2_fft))
                cross_corr = fftshift(np.real(cross_corr))
                
                # Extract the region of interest
                center_y = cross_corr.shape[0] // 2
                center_x = cross_corr.shape[1] // 2
                
                y_start = center_y - max_spatial_lag
                y_end = center_y + max_spatial_lag + 1
                x_start = center_x - max_spatial_lag
                x_end = center_x + max_spatial_lag + 1
                
                correlation_sum += cross_corr[y_start:y_end, x_start:x_end]
            
            # Average over time
            stics_correlation[tau, :, :] = correlation_sum / (T - tau)
        
        return stics_correlation
    
    def _extract_stics_flow_velocity(self, stics_correlation: np.ndarray,
                                   pixel_size: float, time_interval: float,
                                   max_spatial_lag: int, max_temporal_lag: int) -> Dict[str, Any]:
        """Extract flow velocity from STICS correlation function"""
        
        try:
            # Find the peak position for each temporal lag
            peak_positions = []
            
            for tau in range(1, min(max_temporal_lag + 1, stics_correlation.shape[0])):
                corr_slice = stics_correlation[tau, :, :]
                
                # Find peak position
                peak_idx = np.unravel_index(np.argmax(corr_slice), corr_slice.shape)
                
                # Convert to spatial coordinates
                y_peak = peak_idx[0] - max_spatial_lag
                x_peak = peak_idx[1] - max_spatial_lag
                
                peak_positions.append((y_peak, x_peak))
            
            if len(peak_positions) > 1:
                # Calculate velocity from peak displacement
                peak_positions = np.array(peak_positions)
                
                # Linear fit to get average velocity
                time_points = np.arange(1, len(peak_positions) + 1) * time_interval
                
                # Velocity in y-direction
                if len(time_points) > 1:
                    v_y_fit = np.polyfit(time_points, peak_positions[:, 0] * pixel_size, 1)
                    v_x_fit = np.polyfit(time_points, peak_positions[:, 1] * pixel_size, 1)
                    
                    velocity_y = v_y_fit[0]  # microns/second
                    velocity_x = v_x_fit[0]  # microns/second
                else:
                    velocity_y = 0
                    velocity_x = 0
                
                magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                
                return {
                    'velocity_x': velocity_x,
                    'velocity_y': velocity_y,
                    'magnitude': magnitude,
                    'peak_positions': peak_positions
                }
            else:
                return {
                    'velocity_x': 0,
                    'velocity_y': 0,
                    'magnitude': 0,
                    'peak_positions': []
                }
                
        except Exception:
            return {
                'velocity_x': 0,
                'velocity_y': 0,
                'magnitude': 0,
                'peak_positions': []
            }
    
    def _calculate_stics_diffusion_maps(self, stics_correlation: np.ndarray,
                                      pixel_size: float, time_interval: float) -> Dict[str, Any]:
        """Calculate diffusion coefficient maps from STICS data"""
        
        try:
            # Extract zero-lag autocorrelation
            if stics_correlation.shape[0] > 0:
                zero_lag_corr = stics_correlation[0, :, :]
                
                # Find the central peak width
                center = zero_lag_corr.shape[0] // 2
                center_line = zero_lag_corr[center, :]
                
                # Fit Gaussian to central line to estimate beam waist
                x_coords = np.arange(len(center_line)) - center
                
                def gaussian_func(x, A, w, x0, offset):
                    return A * np.exp(-2 * (x - x0)**2 / w**2) + offset
                
                try:
                    # Initial guess
                    A_guess = np.max(center_line)
                    w_guess = 2.0  # pixels
                    x0_guess = 0
                    offset_guess = np.min(center_line)
                    
                    popt, _ = optimize.curve_fit(
                        gaussian_func, x_coords, center_line,
                        p0=[A_guess, w_guess, x0_guess, offset_guess],
                        maxfev=1000
                    )
                    
                    beam_waist = abs(popt[1]) * pixel_size  # Convert to microns
                    
                    # Estimate diffusion coefficient from correlation decay
                    if stics_correlation.shape[0] > 1:
                        tau1_corr = stics_correlation[1, center, center]
                        tau0_corr = stics_correlation[0, center, center]
                        
                        if tau0_corr > 0 and tau1_corr > 0:
                            # Simple diffusion model
                            ratio = tau1_corr / tau0_corr
                            if ratio > 0 and ratio < 1:
                                D = beam_waist**2 / (4 * time_interval) * (1/ratio - 1)
                                D = max(0, D)  # Ensure positive
                            else:
                                D = 0
                        else:
                            D = 0
                    else:
                        D = 0
                    
                    # Estimate mobile fraction
                    mobile_fraction = max(0, min(1, tau1_corr / tau0_corr)) if tau0_corr > 0 else 0
                    
                    return {
                        'D': D,
                        'beam_waist': beam_waist,
                        'mobile_fraction': mobile_fraction
                    }
                    
                except Exception:
                    return {
                        'D': 0,
                        'beam_waist': 0,
                        'mobile_fraction': 0
                    }
            else:
                return {
                    'D': 0,
                    'beam_waist': 0,
                    'mobile_fraction': 0
                }
                
        except Exception:
            return {
                'D': 0,
                'beam_waist': 0,
                'mobile_fraction': 0
            }
    
    def _analyze_imaging_fcs(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Imaging FCS analysis - per-pixel correlation analysis"""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'Imaging FCS requires time series data (3D array)'}
            
            max_lag = parameters.get('max_lag', 50)
            pixel_size = parameters.get('pixel_size', 0.1)
            time_interval = parameters.get('time_interval', 0.001)
            
            T, H, W = image_data.shape
            
            # Calculate correlation for each pixel
            diffusion_map = np.zeros((H, W))
            amplitude_map = np.zeros((H, W))
            
            for y in range(H):
                for x in range(W):
                    pixel_trace = image_data[:, y, x]
                    
                    # Calculate autocorrelation
                    correlation = self._calculate_pixel_autocorrelation(pixel_trace, max_lag)
                    
                    # Fit FCS model
                    fit_results = self._fit_fcs_model_pixel(correlation, time_interval)
                    
                    diffusion_map[y, x] = fit_results.get('D', 0)
                    amplitude_map[y, x] = fit_results.get('amplitude', 0)
            
            # Calculate summary statistics
            avg_diffusion = np.mean(diffusion_map[diffusion_map > 0])
            diffusion_heterogeneity = np.std(diffusion_map[diffusion_map > 0])
            
            return {
                'status': 'success',
                'method': 'Imaging FCS',
                'diffusion_map': diffusion_map,
                'amplitude_map': amplitude_map,
                'avg_diffusion_coefficient': avg_diffusion if not np.isnan(avg_diffusion) else 0,
                'diffusion_heterogeneity': diffusion_heterogeneity if not np.isnan(diffusion_heterogeneity) else 0,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Imaging FCS analysis failed: {str(e)}'}
    
    def _calculate_pixel_autocorrelation(self, trace: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation for a single pixel trace"""
        if len(trace) < max_lag:
            max_lag = len(trace) // 2
        
        # Normalize trace
        mean_intensity = np.mean(trace)
        if mean_intensity > 0:
            delta_trace = (trace - mean_intensity) / mean_intensity
        else:
            delta_trace = trace - mean_intensity
        
        # Calculate autocorrelation
        correlation = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if len(trace) - lag > 0:
                correlation[lag] = np.mean(delta_trace[:-lag] * delta_trace[lag:]) if lag > 0 else np.mean(delta_trace**2)
        
        return correlation
    
    def _fit_fcs_model_pixel(self, correlation: np.ndarray, time_interval: float) -> Dict[str, Any]:
        """Fit simple FCS model to pixel correlation data"""
        
        if len(correlation) < 3:
            return {'D': 0, 'amplitude': 0, 'tau_diff': 0}
        
        # Time axis
        time_axis = np.arange(len(correlation)) * time_interval
        
        # Simple exponential decay model
        def fcs_model(t, amplitude, tau_diff, offset):
            return amplitude * np.exp(-t / tau_diff) + offset
        
        try:
            # Initial guess
            amplitude_guess = correlation[0] if correlation[0] > 0 else 1
            tau_guess = time_interval * 10
            offset_guess = correlation[-1] if len(correlation) > 1 else 0
            
            popt, _ = optimize.curve_fit(
                fcs_model, time_axis, correlation,
                p0=[amplitude_guess, tau_guess, offset_guess],
                maxfev=1000
            )
            
            amplitude, tau_diff, offset = popt
            
            # Estimate diffusion coefficient (assuming w0 = 0.3 microns)
            w0 = 0.3  # microns
            D = w0**2 / (4 * abs(tau_diff)) if tau_diff > 0 else 0
            
            return {
                'D': abs(D),
                'amplitude': abs(amplitude),
                'tau_diff': abs(tau_diff),
                'offset': offset
            }
            
        except Exception:
            return {
                'D': 0,
                'amplitude': 0,
                'tau_diff': 0,
                'offset': 0
            }
    
    def _analyze_pair_correlation(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Pair Correlation Function analysis for directional flow"""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'pCF requires time series data (3D array)'}
            
            max_distance = parameters.get('max_distance', 10)  # pixels
            direction_bins = parameters.get('direction_bins', 8)
            
            T, H, W = image_data.shape
            
            # Calculate directional pair correlation
            pcf_results = self._calculate_directional_pcf(image_data, max_distance, direction_bins)
            
            # Extract flow direction and magnitude
            flow_analysis = self._analyze_pcf_flow(pcf_results, direction_bins)
            
            return {
                'status': 'success',
                'method': 'Pair Correlation Function (pCF)',
                'pcf_by_direction': pcf_results,
                'preferred_flow_direction': flow_analysis.get('direction', 0),
                'flow_anisotropy': flow_analysis.get('anisotropy', 0),
                'correlation_strength': flow_analysis.get('strength', 0),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'pCF analysis failed: {str(e)}'}
    
    def _calculate_directional_pcf(self, image_data: np.ndarray, 
                                  max_distance: int, direction_bins: int) -> Dict[str, Any]:
        """Calculate pair correlation function by direction"""
        
        T, H, W = image_data.shape
        directions = np.linspace(0, 2*np.pi, direction_bins, endpoint=False)
        
        pcf_by_direction = {}
        
        for i, direction in enumerate(directions):
            direction_name = f"Direction_{i}_deg_{int(np.degrees(direction))}"
            
            # Calculate unit vector for this direction
            dx = np.cos(direction)
            dy = np.sin(direction)
            
            correlations = []
            
            for distance in range(1, max_distance + 1):
                # Calculate displacement
                offset_x = int(distance * dx)
                offset_y = int(distance * dy)
                
                correlation_sum = 0
                count = 0
                
                for t in range(T):
                    frame = image_data[t]
                    
                    # Calculate valid regions
                    if offset_x >= 0 and offset_y >= 0:
                        ref_region = frame[:-offset_y if offset_y > 0 else H, 
                                         :-offset_x if offset_x > 0 else W]
                        comp_region = frame[offset_y:, offset_x:]
                    elif offset_x >= 0 and offset_y < 0:
                        ref_region = frame[-offset_y:, :-offset_x if offset_x > 0 else W]
                        comp_region = frame[:offset_y if offset_y < 0 else H, offset_x:]
                    elif offset_x < 0 and offset_y >= 0:
                        ref_region = frame[:-offset_y if offset_y > 0 else H, -offset_x:]
                        comp_region = frame[offset_y:, :offset_x if offset_x < 0 else W]
                    else:  # both negative
                        ref_region = frame[-offset_y:, -offset_x:]
                        comp_region = frame[:offset_y, :offset_x]
                    
                    if ref_region.size > 0 and comp_region.size > 0:
                        # Ensure same size
                        min_h = min(ref_region.shape[0], comp_region.shape[0])
                        min_w = min(ref_region.shape[1], comp_region.shape[1])
                        
                        ref_crop = ref_region[:min_h, :min_w]
                        comp_crop = comp_region[:min_h, :min_w]
                        
                        # Calculate correlation
                        correlation_sum += np.corrcoef(ref_crop.flatten(), comp_crop.flatten())[0, 1]
                        count += 1
                
                if count > 0:
                    correlations.append(correlation_sum / count)
                else:
                    correlations.append(0)
            
            pcf_by_direction[direction_name] = {
                'distances': list(range(1, max_distance + 1)),
                'correlations': correlations,
                'direction_rad': direction,
                'direction_deg': np.degrees(direction)
            }
        
        return pcf_by_direction
    
    def _analyze_pcf_flow(self, pcf_results: Dict[str, Any], direction_bins: int) -> Dict[str, Any]:
        """Analyze flow direction and anisotropy from pCF results"""
        
        try:
            # Extract correlation strengths by direction
            strengths = []
            directions = []
            
            for direction_key, data in pcf_results.items():
                if 'correlations' in data and len(data['correlations']) > 0:
                    # Use correlation at distance 1 as strength measure
                    strength = data['correlations'][0] if len(data['correlations']) > 0 else 0
                    strengths.append(strength)
                    directions.append(data['direction_rad'])
            
            if len(strengths) > 0:
                strengths = np.array(strengths)
                directions = np.array(directions)
                
                # Find preferred direction
                max_idx = np.argmax(strengths)
                preferred_direction = directions[max_idx]
                max_strength = strengths[max_idx]
                
                # Calculate anisotropy (coefficient of variation)
                anisotropy = np.std(strengths) / np.mean(strengths) if np.mean(strengths) > 0 else 0
                
                return {
                    'direction': preferred_direction,
                    'direction_deg': np.degrees(preferred_direction),
                    'strength': max_strength,
                    'anisotropy': anisotropy,
                    'all_strengths': strengths,
                    'all_directions': directions
                }
            else:
                return {
                    'direction': 0,
                    'direction_deg': 0,
                    'strength': 0,
                    'anisotropy': 0,
                    'all_strengths': [],
                    'all_directions': []
                }
                
        except Exception:
            return {
                'direction': 0,
                'direction_deg': 0,
                'strength': 0,
                'anisotropy': 0,
                'all_strengths': [],
                'all_directions': []
            }
    
    def _analyze_autocorr_ics(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-correlation Image Correlation Spectroscopy"""
        try:
            # This is similar to imaging FCS but focuses on spatial autocorrelation
            if image_data.ndim == 3:
                # Average over time for spatial analysis
                avg_image = np.mean(image_data, axis=0)
            else:
                avg_image = image_data
            
            max_lag = parameters.get('max_spatial_lag', 10)
            
            # Calculate 2D spatial autocorrelation
            spatial_autocorr = self._calculate_2d_autocorrelation(avg_image, max_lag)
            
            # Fit to get characteristic length scale
            length_scale = self._extract_correlation_length(spatial_autocorr, max_lag)
            
            return {
                'status': 'success',
                'method': 'Auto-correlation ICS',
                'spatial_autocorrelation': spatial_autocorr,
                'correlation_length': length_scale,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Auto-correlation ICS failed: {str(e)}'}
    
    def _analyze_crosscorr_ics(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-correlation ICS for multi-channel data"""
        try:
            if image_data.ndim == 4 and image_data.shape[-1] >= 2:
                # Multi-channel data
                channel1 = image_data[..., 0]
                channel2 = image_data[..., 1]
            elif image_data.ndim == 3:
                # Split channels or use temporal correlation
                mid_point = image_data.shape[0] // 2
                channel1 = image_data[:mid_point]
                channel2 = image_data[mid_point:]
            else:
                return {'status': 'error', 'message': 'Cross-correlation ICS requires multi-channel data'}
            
            max_lag = parameters.get('max_spatial_lag', 10)
            
            # Calculate cross-correlation
            if channel1.ndim == 3 and channel2.ndim == 3:
                # Average over time
                avg_ch1 = np.mean(channel1, axis=0)
                avg_ch2 = np.mean(channel2, axis=0)
            else:
                avg_ch1 = channel1
                avg_ch2 = channel2
            
            cross_correlation = self._calculate_2d_crosscorrelation(avg_ch1, avg_ch2, max_lag)
            
            # Calculate colocalization metrics
            colocalization = self._calculate_colocalization_metrics(avg_ch1, avg_ch2)
            
            return {
                'status': 'success',
                'method': 'Cross-correlation ICS',
                'cross_correlation': cross_correlation,
                'colocalization_coefficient': colocalization.get('pearson_r', 0),
                'overlap_coefficient': colocalization.get('overlap', 0),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Cross-correlation ICS failed: {str(e)}'}
    
    def _analyze_imsd_ics(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Image Mean Square Displacement via ICS methods"""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'iMSD via ICS requires time series data'}
            
            max_lag = parameters.get('max_temporal_lag', 20)
            pixel_size = parameters.get('pixel_size', 0.1)
            time_interval = parameters.get('time_interval', 0.1)
            
            T, H, W = image_data.shape
            
            # Calculate MSD for each pixel using correlation approach
            msd_map = np.zeros((H, W))
            
            for y in range(H):
                for x in range(W):
                    pixel_trace = image_data[:, y, x]
                    msd_curve = self._calculate_pixel_msd_ics(pixel_trace, max_lag)
                    
                    # Fit linear slope to get effective diffusion
                    if len(msd_curve) > 2:
                        time_axis = np.arange(len(msd_curve)) * time_interval
                        slope = np.polyfit(time_axis[1:], msd_curve[1:], 1)[0]
                        msd_map[y, x] = slope
            
            # Convert to diffusion coefficient map
            diffusion_map = msd_map / 4  # 2D diffusion: MSD = 4Dt
            
            return {
                'status': 'success',
                'method': 'iMSD via ICS',
                'msd_map': msd_map,
                'diffusion_map': diffusion_map,
                'avg_diffusion': np.mean(diffusion_map[diffusion_map > 0]),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'iMSD via ICS failed: {str(e)}'}
    
    def _analyze_temporal_ics(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Temporal Image Correlation Spectroscopy"""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'Temporal ICS requires time series data'}
            
            max_lag = parameters.get('max_temporal_lag', 50)
            time_interval = parameters.get('time_interval', 0.001)
            
            T, H, W = image_data.shape
            
            # Calculate temporal correlation for the entire image
            temporal_correlation = self._calculate_temporal_correlation(image_data, max_lag)
            
            # Fit exponential decay
            decay_fit = self._fit_temporal_decay(temporal_correlation, time_interval)
            
            return {
                'status': 'success',
                'method': 'Temporal ICS',
                'temporal_correlation': temporal_correlation,
                'decay_time': decay_fit.get('tau', 0),
                'amplitude': decay_fit.get('amplitude', 0),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Temporal ICS failed: {str(e)}'}
    
    def _calculate_2d_autocorrelation(self, image: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate 2D spatial autocorrelation"""
        H, W = image.shape
        correlation = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))
        
        # Normalize image
        mean_val = np.mean(image)
        norm_image = (image - mean_val) / np.std(image) if np.std(image) > 0 else image - mean_val
        
        for dy in range(-max_lag, max_lag + 1):
            for dx in range(-max_lag, max_lag + 1):
                
                # Calculate valid regions
                y1_start = max(0, dy)
                y1_end = min(H, H + dy)
                x1_start = max(0, dx)
                x1_end = min(W, W + dx)
                
                y2_start = max(0, -dy)
                y2_end = min(H, H - dy)
                x2_start = max(0, -dx)
                x2_end = min(W, W - dx)
                
                if (y1_end > y1_start and x1_end > x1_start and 
                    y2_end > y2_start and x2_end > x2_start):
                    
                    region1 = norm_image[y1_start:y1_end, x1_start:x1_end]
                    region2 = norm_image[y2_start:y2_end, x2_start:x2_end]
                    
                    # Ensure same size
                    min_h = min(region1.shape[0], region2.shape[0])
                    min_w = min(region1.shape[1], region2.shape[1])
                    
                    region1 = region1[:min_h, :min_w]
                    region2 = region2[:min_h, :min_w]
                    
                    correlation[dy + max_lag, dx + max_lag] = np.mean(region1 * region2)
        
        return correlation
    
    def _calculate_2d_crosscorrelation(self, image1: np.ndarray, image2: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate 2D spatial cross-correlation between two images"""
        # Similar to autocorrelation but between different images
        H, W = image1.shape
        correlation = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))
        
        # Normalize images
        mean1, std1 = np.mean(image1), np.std(image1)
        mean2, std2 = np.mean(image2), np.std(image2)
        
        norm_image1 = (image1 - mean1) / std1 if std1 > 0 else image1 - mean1
        norm_image2 = (image2 - mean2) / std2 if std2 > 0 else image2 - mean2
        
        for dy in range(-max_lag, max_lag + 1):
            for dx in range(-max_lag, max_lag + 1):
                
                # Calculate valid regions (similar to autocorrelation)
                y1_start = max(0, dy)
                y1_end = min(H, H + dy)
                x1_start = max(0, dx)
                x1_end = min(W, W + dx)
                
                y2_start = max(0, -dy)
                y2_end = min(H, H - dy)
                x2_start = max(0, -dx)
                x2_end = min(W, W - dx)
                
                if (y1_end > y1_start and x1_end > x1_start and 
                    y2_end > y2_start and x2_end > x2_start):
                    
                    region1 = norm_image1[y1_start:y1_end, x1_start:x1_end]
                    region2 = norm_image2[y2_start:y2_end, x2_start:x2_end]
                    
                    # Ensure same size
                    min_h = min(region1.shape[0], region2.shape[0])
                    min_w = min(region1.shape[1], region2.shape[1])
                    
                    region1 = region1[:min_h, :min_w]
                    region2 = region2[:min_h, :min_w]
                    
                    correlation[dy + max_lag, dx + max_lag] = np.mean(region1 * region2)
        
        return correlation
    
    def _extract_correlation_length(self, autocorr: np.ndarray, max_lag: int) -> float:
        """Extract characteristic correlation length from 2D autocorrelation"""
        try:
            center = max_lag
            center_line = autocorr[center, center:]
            
            # Find half-maximum point
            max_val = center_line[0]
            half_max = max_val / 2
            
            # Find where correlation drops to half maximum
            for i, val in enumerate(center_line):
                if val < half_max:
                    return float(i)
            
            return float(max_lag)
            
        except Exception:
            return 0.0
    
    def _calculate_colocalization_metrics(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """Calculate colocalization metrics between two images"""
        try:
            # Pearson correlation coefficient
            pearson_r = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]
            
            # Overlap coefficient
            numerator = np.sum(image1 * image2)
            denominator = np.sqrt(np.sum(image1**2) * np.sum(image2**2))
            overlap = numerator / denominator if denominator > 0 else 0
            
            return {
                'pearson_r': pearson_r if not np.isnan(pearson_r) else 0,
                'overlap': overlap if not np.isnan(overlap) else 0
            }
            
        except Exception:
            return {'pearson_r': 0, 'overlap': 0}
    
    def _calculate_pixel_msd_ics(self, trace: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate MSD for a pixel trace using correlation approach"""
        if len(trace) < max_lag:
            max_lag = len(trace) // 2
        
        msd = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if len(trace) - lag > 0:
                displacements = trace[lag:] - trace[:-lag] if lag > 0 else np.zeros_like(trace)
                msd[lag] = np.mean(displacements**2)
        
        return msd
    
    def _calculate_temporal_correlation(self, image_data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate temporal correlation for entire image stack"""
        T, H, W = image_data.shape
        
        if max_lag >= T:
            max_lag = T - 1
        
        # Calculate mean intensity over time for each pixel
        pixel_means = np.mean(image_data, axis=0)
        
        # Calculate temporal correlation
        temporal_corr = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if T - lag > 0:
                # Calculate correlation between frames separated by lag
                frames1 = image_data[:-lag] if lag > 0 else image_data
                frames2 = image_data[lag:]
                
                correlation_sum = 0
                for t in range(frames1.shape[0]):
                    frame1 = frames1[t] - pixel_means
                    frame2 = frames2[t] - pixel_means
                    correlation_sum += np.mean(frame1 * frame2)
                
                temporal_corr[lag] = correlation_sum / frames1.shape[0]
        
        return temporal_corr
    
    def _fit_temporal_decay(self, correlation: np.ndarray, time_interval: float) -> Dict[str, float]:
        """Fit exponential decay to temporal correlation"""
        try:
            time_axis = np.arange(len(correlation)) * time_interval
            
            def exp_decay(t, amplitude, tau, offset):
                return amplitude * np.exp(-t / tau) + offset
            
            # Initial guess
            amplitude_guess = correlation[0] if len(correlation) > 0 else 1
            tau_guess = time_interval * 10
            offset_guess = correlation[-1] if len(correlation) > 1 else 0
            
            popt, _ = optimize.curve_fit(
                exp_decay, time_axis, correlation,
                p0=[amplitude_guess, tau_guess, offset_guess],
                maxfev=1000
            )
            
            amplitude, tau, offset = popt
            
            return {
                'amplitude': abs(amplitude),
                'tau': abs(tau),
                'offset': offset
            }
            
        except Exception:
            return {
                'amplitude': 0,
                'tau': 0,
                'offset': 0
            }

def get_ics_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for ICS methods"""
    
    parameters = {
        'RICS (Raster Image Correlation Spectroscopy)': {
            'pixel_size': 0.1,
            'line_time': 0.001,
            'tau_max': 10,
            'eta_max': 10
        },
        'STICS (Spatio-Temporal Image Correlation Spectroscopy)': {
            'max_spatial_lag': 5,
            'max_temporal_lag': 5,
            'pixel_size': 0.1,
            'time_interval': 0.1
        },
        'Imaging FCS': {
            'max_lag': 50,
            'pixel_size': 0.1,
            'time_interval': 0.001
        },
        'Pair Correlation Function (pCF)': {
            'max_distance': 10,
            'direction_bins': 8
        },
        'Auto-correlation ICS': {
            'max_spatial_lag': 10
        },
        'Cross-correlation ICS': {
            'max_spatial_lag': 10
        },
        'iMSD via ICS': {
            'max_temporal_lag': 20,
            'pixel_size': 0.1,
            'time_interval': 0.1
        },
        'Temporal ICS': {
            'max_temporal_lag': 50,
            'time_interval': 0.001
        }
    }
    
    return parameters.get(method, {})