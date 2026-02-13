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
            
            # Calculate mean intensity and variance
            avg_intensity = np.mean(image_data)
            intensity_variance = np.var(image_data)

            if avg_intensity == 0:
                 return {'status': 'error', 'message': 'Average intensity is zero.'}

            # Calculate RICS correlation function (unnormalized covariance)
            rics_cov = self._calculate_rics_correlation(
                image_data, tau_max, eta_max
            )
            
            # Normalize G = Cov / <I>^2
            # G(0,0) approx 1/N
            rics_correlation = rics_cov / (avg_intensity**2)

            # Fit RICS model to extract diffusion parameters
            diffusion_results = self._fit_rics_diffusion_model(
                rics_correlation, pixel_size, line_time, tau_max, eta_max
            )
            
            # Calculate N from G(0) (amplitude)
            # G(0) = gamma / N, where gamma depends on PSF (usually 0.35-0.5 for 3D Gaussian)
            # Here the fit returns G0 directly as 'amplitude' (or via N parameter in model)

            # The model uses N as a fitting parameter in the numerator.
            # G = (gamma / N) * ...
            # We adjusted the model to fit 'G0' directly, then calculate N.
            
            return {
                'status': 'success',
                'method': 'RICS (Raster Image Correlation Spectroscopy)',
                'correlation_function': rics_correlation,
                'diffusion_coefficient': diffusion_results.get('D', 0),
                'number_of_particles': diffusion_results.get('N', 0), # Derived from G0
                'G0': diffusion_results.get('G0', 0),
                'fit_quality': diffusion_results.get('r_squared', 0),
                'avg_intensity': avg_intensity,
                'intensity_variance': intensity_variance,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'RICS analysis failed: {str(e)}'}
    
    def _calculate_rics_correlation(self, image_data: np.ndarray, 
                                   tau_max: int, eta_max: int) -> np.ndarray:
        """Calculate RICS spatial correlation function (Autocovariance)"""
        T, H, W = image_data.shape
        
        # Calculate mean intensity for each frame to handle bleaching/fluctuations
        # Subtracting local mean (or frame mean) gives delta_I
        mean_intensities = np.mean(image_data, axis=(1, 2))
        
        # Calculate intensity fluctuations
        delta_I = image_data - mean_intensities[:, np.newaxis, np.newaxis]
        
        # Initialize correlation array
        correlation = np.zeros((2 * tau_max + 1, 2 * eta_max + 1))
        
        # Calculate spatial correlation for each time point
        # This implementation is slow (nested loops).
        # Ideally should use FFT or optimized correlation.
        # But keeping structure for now, verifying logic.

        for tau in range(-tau_max, tau_max + 1):
            for eta in range(-eta_max, eta_max + 1):
                correlation_sum = 0
                count = 0

                # Pre-compute shifted frame
                # Shift along Y (lines, slow axis) is tau? No, usually in RICS:
                # x is fast axis (pixel time), y is slow axis (line time).
                # Correlation usually G(xi, psi) where xi is pixel lag, psi is line lag.
                # Here variables are named 'tau' and 'eta'.
                # Let's assume 'tau' corresponds to Y (lines) and 'eta' to X (pixels)?
                # Standard RICS: G(xi, psi).
                # Let's check model. 4 D |eta| / w0^2. This suggests eta is time?
                # RICS time lag = xi * pixel_time + psi * line_time.

                # In this code:
                # tau loop -> shift axis 1 (Y/lines).
                # eta loop -> shift axis 2 (X/pixels).

                shifted_delta_I = np.roll(np.roll(delta_I, tau, axis=1), eta, axis=2)

                # Manual zero padding logic
                if tau > 0:
                     shifted_delta_I[:, :tau, :] = 0
                elif tau < 0:
                     shifted_delta_I[:, tau:, :] = 0

                if eta > 0:
                     shifted_delta_I[:, :, :eta] = 0
                elif eta < 0:
                     shifted_delta_I[:, :, eta:] = 0

                # Vectorized sum over T frames
                # Only overlap regions

                # Valid mask for shift
                valid_mask = np.ones((H, W), dtype=bool)
                if tau > 0: valid_mask[:tau, :] = False
                elif tau < 0: valid_mask[tau:, :] = False
                if eta > 0: valid_mask[:, :eta] = False
                elif eta < 0: valid_mask[:, eta:] = False

                # Calculate product sum over valid region
                # Sum over T, Y, X where valid
                # Since we zero-padded shifted_delta_I, we can just multiply and sum
                # But we need to count non-zero overlaps correctly

                # Optimized calculation:
                product = delta_I * shifted_delta_I
                # We only sum over the valid overlap region defined by (H-|tau|) * (W-|eta|)
                # The zero padding handles the values, but we need correct count.

                correlation_sum = np.sum(product[:, valid_mask])
                count = T * np.sum(valid_mask)

                if count > 0:
                    correlation[tau + tau_max, eta + eta_max] = correlation_sum / count
                else:
                    correlation[tau + tau_max, eta + eta_max] = 0
        return correlation
    
    def _fit_rics_diffusion_model(self, correlation: np.ndarray, pixel_size: float,
                                 line_time: float, tau_max: int, eta_max: int) -> Dict[str, Any]:
        """Fit 2D diffusion model to RICS correlation data"""
        
        # Create coordinate arrays
        # tau corresponds to axis 0 of correlation (Y shifts)
        # eta corresponds to axis 1 of correlation (X shifts)

        # Time lag at (y_shift, x_shift) = y_shift * line_time + x_shift * pixel_time
        # But fitting usually treats spatial lag.
        # Standard RICS Eq:
        # G(xi, psi) = G0 * exp(...) * ...
        # where time lag t = |psi|*line_time + |xi|*pixel_time
        # spatial lag r = sqrt((xi*dx)^2 + (psi*dy)^2)

        y_shifts = np.arange(-tau_max, tau_max + 1)
        x_shifts = np.arange(-eta_max, eta_max + 1)
        
        Y_SHIFT, X_SHIFT = np.meshgrid(y_shifts, x_shifts, indexing='ij')
        
        # RICS diffusion model function
        def rics_model(coords, D, G0, w0, offset):
            y_s, x_s = coords

            # Time lag for this pixel separation
            # Absolute value? Usually we consider correlation vs positive time lag.
            # But RICS grid is symmetric.
            # We use absolute lags for diffusion time calculation
            t = np.abs(y_s) * line_time + np.abs(x_s) * (pixel_size/100.0) # Assume pixel time is small/negligible or provided?
            # Wait, standard RICS assumes continuous scanning.
            # Let's assume pixel_time is negligible compared to line_time or provided separately.
            # The previous implementation used: 4 * D * eta / w0^2.
            # Let's try to be consistent with standard RICS.
            # We need pixel_dwell_time. It wasn't passed in.
            # Let's approximate: t = |y_s| * line_time. (Ignoring fast axis time contribution for simplicity if small)

            # Correct RICS term:
            # G(x,y) = G0 / (1 + 4Dt/w0^2) * exp( - (r^2 + 4Dt) / w0^2 ) ... this is for spatial.
            # Scanning RICS:
            # t = tau_p * xi + tau_l * psi
            # Spatial displacement r = delta_r
            # For pure diffusion:
            # G(xi, psi) = G0 * (1 + 4Dt/w0^2)^-1 * exp(- ( (xi*dx)^2 + (psi*dy)^2 ) / (w0^2 * (1+4Dt/w0^2)) )

            # Let's use the line_time as dominant time scale
            t_lag = np.abs(y_s) * line_time

            # Spatial distance squared
            r2 = (x_s * pixel_size)**2 + (y_s * pixel_size)**2 # Assuming square pixels

            denominator = 1 + 4 * D * t_lag / (w0**2)

            return (G0 / denominator) * np.exp(-r2 / (w0**2 * denominator)) + offset
        
        # Flatten arrays for fitting
        y_flat = Y_SHIFT.flatten()
        x_flat = X_SHIFT.flatten()
        corr_flat = correlation.flatten()
        
        # Initial parameter guess
        # G0 is max of correlation
        G0_guess = np.max(corr_flat)
        w0_guess = 0.3 # microns
        D_guess = 1.0 # um^2/s
        offset_guess = 0.0

        p0 = [D_guess, G0_guess, w0_guess, offset_guess]
        
        try:
            # Fit the model
            popt, pcov = optimize.curve_fit(
                lambda coords, D, G0, w0, offset: rics_model(coords, D, G0, w0, offset),
                (y_flat, x_flat),
                corr_flat,
                p0=p0,
                bounds=([0, 0, 0.1, -np.inf], [np.inf, np.inf, 2.0, np.inf]),
                maxfev=5000
            )
            
            D, G0, w0, offset = popt

            # Calculate N from G0.
            # G0 = gamma / N. Assuming gamma=0.35 (3D Gaussian).
            gamma = 0.35
            N = gamma / G0 if G0 > 0 else 0
            
            # Calculate R-squared
            y_pred = rics_model((y_flat, x_flat), *popt)
            ss_res = np.sum((corr_flat - y_pred) ** 2)
            ss_tot = np.sum((corr_flat - np.mean(corr_flat)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'D': abs(D),  # Diffusion coefficient
                'N': abs(N),  # Number of particles
                'G0': G0,
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
                'G0': 0,
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
        
        # Normalize images: (I - <I>) / <I>
        normalized_data = np.zeros_like(image_data, dtype=float)
        mean_val = np.mean(image_data)
        if mean_val > 0:
            normalized_data = (image_data - mean_val) / mean_val
        else:
            normalized_data = image_data
        
        # Initialize correlation function
        correlation_shape = (max_temporal_lag + 1, 
                           2 * max_spatial_lag + 1, 
                           2 * max_spatial_lag + 1)
        stics_correlation = np.zeros(correlation_shape)
        
        # Calculate correlation for each temporal lag
        for tau in range(max_temporal_lag + 1):
            if T - tau <= 1:
                continue
            
            # We want sum((I(t) - <I>)(I(t+tau) - <I>)) / N_pixels
            # Normalized data is (I-<I>)/<I>.
            # So sum(norm(t)*norm(t+tau)) gives sum(dI dI) / <I>^2.
            # This matches G(tau) definition.

            # FFT Cross-correlation
            # Pad to avoid circular convolution aliasing
            pad_h = H + 2 * max_spatial_lag
            pad_w = W + 2 * max_spatial_lag

            correlation_sum = np.zeros((pad_h, pad_w))

            # Accumulate correlation over all time pairs (t, t+tau)
            for t in range(T - tau):
                frame1 = normalized_data[t]
                frame2 = normalized_data[t + tau]
                
                # FFT with padding
                f1_fft = fft2(frame1, s=(pad_h, pad_w))
                f2_fft = fft2(frame2, s=(pad_h, pad_w))
                
                # Cross-correlation in frequency domain
                cross_corr = ifft2(f1_fft * np.conj(f2_fft))
                correlation_sum += np.real(cross_corr)

            # Average over time pairs
            avg_corr_full = correlation_sum / (T - tau)
            avg_corr_shifted = fftshift(avg_corr_full)

            # Crop to ROI
            center_y = avg_corr_shifted.shape[0] // 2
            center_x = avg_corr_shifted.shape[1] // 2
            
            y_start = center_y - max_spatial_lag
            y_end = center_y + max_spatial_lag + 1
            x_start = center_x - max_spatial_lag
            x_end = center_x + max_spatial_lag + 1

            # Normalize by number of pixels (H*W) to get average
            # FFT returns sum of products.
            stics_correlation[tau, :, :] = avg_corr_shifted[y_start:y_end, x_start:x_end] / (H * W)
        
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
                # Use sub-pixel localization if possible (Gaussian fit)
                peak_idx = np.unravel_index(np.argmax(corr_slice), corr_slice.shape)
                
                # Convert to spatial coordinates relative to center
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
                    
                    # Estimate diffusion coefficient from correlation decay at center (spatial lag 0)
                    if stics_correlation.shape[0] > 1:
                        # G(0, 0, tau) decay
                        temporal_decay = stics_correlation[:, center, center]
                        time_axis = np.arange(len(temporal_decay)) * time_interval

                        # Fit FCS model
                        def fcs_model(t, G0, D):
                             return G0 / (1 + 4 * D * t / beam_waist**2)

                        popt_d, _ = optimize.curve_fit(
                            fcs_model, time_axis, temporal_decay,
                            p0=[temporal_decay[0], 1.0],
                            bounds=([0, 0], [np.inf, np.inf]),
                            maxfev=1000
                        )
                        D = popt_d[1]
                        
                    else:
                        D = 0
                    
                    # Estimate mobile fraction
                    # Usually requires fitting G(infinity) offset
                    mobile_fraction = 1.0 # Placeholder
                    
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
            valid_D = diffusion_map[diffusion_map > 0]
            avg_diffusion = np.mean(valid_D) if len(valid_D) > 0 else 0
            diffusion_heterogeneity = np.std(valid_D) if len(valid_D) > 0 else 0
            
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
        if len(trace) < 10:
             return np.zeros(max_lag)

        if len(trace) < max_lag:
            max_lag = len(trace) // 2
        
        # Normalize trace: (I - <I>) / <I>
        mean_intensity = np.mean(trace)
        if mean_intensity > 0:
            delta_trace = (trace - mean_intensity) / mean_intensity
        else:
            return np.zeros(max_lag)
        
        # Calculate autocorrelation using correlate
        full_corr = np.correlate(delta_trace, delta_trace, mode='full')
        mid = len(full_corr) // 2
        acf = full_corr[mid:mid+max_lag]
        
        # Normalize by overlap
        lags = np.arange(len(acf))
        overlap = len(trace) - lags
        acf = acf / overlap
        
        return acf
    
    def _fit_fcs_model_pixel(self, correlation: np.ndarray, time_interval: float) -> Dict[str, Any]:
        """Fit simple FCS model to pixel correlation data"""
        
        if len(correlation) < 3:
            return {'D': 0, 'amplitude': 0, 'tau_diff': 0}
        
        # Time axis
        time_axis = np.arange(len(correlation)) * time_interval
        
        # Simple diffusion model
        def fcs_model(t, amplitude, tau_diff, offset):
            return amplitude / (1 + t/tau_diff) + offset
        
        try:
            # Initial guess
            amplitude_guess = correlation[0] if correlation[0] > 0 else 0.1
            tau_guess = time_interval * 10
            offset_guess = correlation[-1] if len(correlation) > 1 else 0
            
            popt, _ = optimize.curve_fit(
                fcs_model, time_axis, correlation,
                p0=[amplitude_guess, tau_guess, offset_guess],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
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
        
        # Normalize stack: (I-<I>)/<I>
        mean_I = np.mean(image_data)
        if mean_I == 0: return {}
        norm_stack = (image_data - mean_I) / mean_I

        for i, direction in enumerate(directions):
            direction_name = f"Direction_{i}_deg_{int(np.degrees(direction))}"
            
            # Calculate unit vector for this direction
            dx_dir = np.cos(direction)
            dy_dir = np.sin(direction)
            
            correlations = []
            
            for distance in range(1, max_distance + 1):
                # Calculate displacement
                offset_x = int(round(distance * dx_dir))
                offset_y = int(round(distance * dy_dir))
                
                # Pair correlation: <delta_I(x,y) * delta_I(x+dx, y+dy)>
                
                # Shift stack
                shifted_stack = np.roll(np.roll(norm_stack, -offset_y, axis=1), -offset_x, axis=2)

                # Mask out rolled regions
                # (Simple masking for now, rolling wraps around which is bad for non-periodic,
                # but masking is complex in 3D without slicing)
                # Let's use slicing instead of roll

                try:
                    # Slicing approach
                    if offset_y >= 0:
                        slice_y_ref = slice(0, H - offset_y)
                        slice_y_shift = slice(offset_y, H)
                    else:
                        slice_y_ref = slice(-offset_y, H)
                        slice_y_shift = slice(0, H + offset_y)
                        
                    if offset_x >= 0:
                        slice_x_ref = slice(0, W - offset_x)
                        slice_x_shift = slice(offset_x, W)
                    else:
                        slice_x_ref = slice(-offset_x, W)
                        slice_x_shift = slice(0, W + offset_x)

                    if (slice_y_ref.start < slice_y_ref.stop and
                        slice_x_ref.start < slice_x_ref.stop):
                        
                        ref_region = norm_stack[:, slice_y_ref, slice_x_ref]
                        shift_region = norm_stack[:, slice_y_shift, slice_x_shift]

                        # Compute correlation
                        prod = ref_region * shift_region
                        corr_val = np.mean(prod)
                        correlations.append(corr_val)
                    else:
                        correlations.append(0)
                except Exception:
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
                    # Use average correlation over first few pixels as strength measure
                    # Distance 1 might be dominated by PSF
                    strength = np.mean(data['correlations'][:3]) if len(data['correlations']) >= 3 else data['correlations'][0]
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
        """Image Mean Square Displacement via STICS Gaussian-variance method."""
        try:
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'iMSD via ICS requires time series data'}

            imsd = self.compute_imsd_stics(
                image_stack=image_data,
                mask=parameters.get('mask', None),
                max_temporal_lag=int(parameters.get('max_temporal_lag', 20)),
                max_spatial_lag=int(parameters.get('max_spatial_lag', 6)),
                pixel_size_um=float(parameters.get('pixel_size', 0.1)),
                time_interval_s=float(parameters.get('time_interval', 0.1)),
                fit_radius=int(parameters.get('fit_radius', parameters.get('max_spatial_lag', 6)))
            )
            if imsd.get('status') != 'success':
                return imsd

            return {
                'status': 'success',
                'method': 'iMSD via ICS (STICS Gaussian variance)',
                **imsd,
                'parameters_used': parameters,
                'note': 'iMSD is computed from the spatiotemporal correlation peak variance vs temporal lag.'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'iMSD via ICS failed: {str(e)}'}

    def compute_imsd_stics(self,
                           image_stack: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           max_temporal_lag: int = 20,
                           max_spatial_lag: int = 6,
                           pixel_size_um: float = 0.1,
                           time_interval_s: float = 0.1,
                           fit_radius: Optional[int] = None) -> Dict[str, Any]:
        """Compute iMSD from STICS maps by fitting Gaussian width over temporal lag.

        For each lag τ, this computes G(ξ, χ, τ), fits:
            G = A * exp(-((x-x0)^2 + (y-y0)^2) / sigma2) + B
        and returns sigma2(τ) in µm².
        """
        stack = np.asarray(image_stack, dtype=float)
        if stack.ndim != 3:
            return {'status': 'error', 'message': 'compute_imsd_stics expects image_stack shape (T,Y,X)'}

        T, H, W = stack.shape
        if T < 6:
            return {'status': 'error', 'message': 'Insufficient temporal frames for iMSD'}

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (H, W):
                return {'status': 'error', 'message': 'mask shape must match image spatial dimensions'}
            ys, xs = np.where(mask)
            if ys.size == 0:
                return {'status': 'error', 'message': 'mask has no valid pixels'}
            y0b, y1b = int(np.min(ys)), int(np.max(ys)) + 1
            x0b, x1b = int(np.min(xs)), int(np.max(xs)) + 1
            stack = stack[:, y0b:y1b, x0b:x1b]
            roi_mask = mask[y0b:y1b, x0b:x1b]
        else:
            roi_mask = np.ones((H, W), dtype=bool)

        T, H, W = stack.shape
        max_tau = int(max(1, min(max_temporal_lag, T - 2)))
        max_r = int(max(2, min(max_spatial_lag, H // 2, W // 2)))
        fit_r = int(max_r if fit_radius is None else max(2, min(fit_radius, max_r)))

        # Apply mask and mean-subtracted fluctuations.
        # Keep zero outside mask so masked pixels do not contribute to signal.
        masked_stack = stack.copy()
        for t in range(T):
            masked_stack[t, ~roi_mask] = 0.0
        roi_mean = float(np.mean(masked_stack[:, roi_mask]))
        if abs(roi_mean) < 1e-12:
            return {'status': 'error', 'message': 'ROI mean intensity is ~0; iMSD undefined'}

        delta = np.zeros_like(masked_stack, dtype=float)
        for t in range(T):
            frame_vals = masked_stack[t, roi_mask]
            frame_mean = float(np.mean(frame_vals)) if frame_vals.size else 0.0
            delta[t] = masked_stack[t] - frame_mean
            delta[t, ~roi_mask] = 0.0

        # Precompute overlap normalization for spatial lags around center.
        ones = roi_mask.astype(float)
        pad_shape = (2 * H, 2 * W)
        o_fft = fft2(ones, s=pad_shape)
        overlap_map = fftshift(ifft2(o_fft * np.conjugate(o_fft)).real)
        cy, cx = overlap_map.shape[0] // 2, overlap_map.shape[1] // 2
        overlap_crop = overlap_map[cy - max_r:cy + max_r + 1, cx - max_r:cx + max_r + 1]
        overlap_crop = np.where(overlap_crop <= 0, np.nan, overlap_crop)

        spatial_maps: List[np.ndarray] = []
        tau_s: List[float] = []
        imsd_um2: List[float] = []
        diagnostics: List[Dict[str, Any]] = []

        for tau in range(max_tau + 1):
            n_pairs = T - tau
            if n_pairs <= 0:
                continue

            acc = np.zeros((2 * max_r + 1, 2 * max_r + 1), dtype=float)
            for t in range(n_pairs):
                f1 = delta[t]
                f2 = delta[t + tau]
                c_full = fftshift(ifft2(fft2(f1, s=pad_shape) * np.conjugate(fft2(f2, s=pad_shape))).real)
                c_crop = c_full[cy - max_r:cy + max_r + 1, cx - max_r:cx + max_r + 1]
                acc += c_crop

            c_avg = acc / n_pairs
            G = c_avg / (overlap_crop * (roi_mean ** 2))
            G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
            spatial_maps.append(G)

            fit = self._fit_imsd_gaussian_peak(G, pixel_size_um=pixel_size_um, fit_radius=fit_r)
            tau_s.append(float(tau * time_interval_s))
            imsd_um2.append(float(fit['sigma2_um2']))
            diagnostics.append({
                'tau_index': int(tau),
                'tau_s': float(tau * time_interval_s),
                'A': float(fit['A']),
                'B': float(fit['B']),
                'x0_um': float(fit['x0_um']),
                'y0_um': float(fit['y0_um']),
                'sigma2_um2': float(fit['sigma2_um2']),
                'r_squared': float(fit['r_squared']),
                'residual_ss': float(fit['residual_ss']),
                'converged': bool(fit['converged'])
            })

        tau_arr = np.asarray(tau_s, dtype=float)
        sigma2_arr = np.asarray(imsd_um2, dtype=float)

        D_um2_s = np.nan
        intercept_um2 = np.nan
        if len(tau_arr) >= 3:
            valid = np.isfinite(tau_arr) & np.isfinite(sigma2_arr)
            valid &= (tau_arr >= 0)
            if np.sum(valid) >= 3:
                coeff = np.polyfit(tau_arr[valid], sigma2_arr[valid], 1)
                slope, intercept = coeff[0], coeff[1]
                D_um2_s = float(max(0.0, slope / 4.0))
                intercept_um2 = float(intercept)

        return {
            'status': 'success',
            'tau_s': tau_arr,
            'imsd_um2': sigma2_arr,
            'stics_maps': spatial_maps,
            'fit_diagnostics': diagnostics,
            'diffusion_coefficient_um2_s': D_um2_s,
            'sigma0_squared_um2': intercept_um2,
            'model_note': 'For isotropic free diffusion, iMSD(tau) ≈ 4 D tau + sigma0^2.'
        }

    def _fit_imsd_gaussian_peak(self,
                                G_tau: np.ndarray,
                                pixel_size_um: float,
                                fit_radius: int) -> Dict[str, Any]:
        """Fit 2D Gaussian + baseline around STICS peak center to extract sigma^2."""
        h, w = G_tau.shape
        cy, cx = h // 2, w // 2
        fit_radius = int(max(1, min(fit_radius, cy, cx)))

        sub = G_tau[cy - fit_radius:cy + fit_radius + 1, cx - fit_radius:cx + fit_radius + 1]
        yy, xx = np.mgrid[-fit_radius:fit_radius + 1, -fit_radius:fit_radius + 1]
        x_um = xx * pixel_size_um
        y_um = yy * pixel_size_um

        x_flat = x_um.ravel()
        y_flat = y_um.ravel()
        z_flat = sub.ravel()

        def model(coords, A, x0, y0, sigma2, B):
            x, y = coords
            return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma2) + B

        A0 = float(np.max(z_flat) - np.median(z_flat))
        B0 = float(np.median(z_flat))
        sigma0 = float(max((pixel_size_um ** 2), (2.0 * pixel_size_um) ** 2))
        p0 = [max(A0, 1e-8), 0.0, 0.0, sigma0, B0]

        center_bound = 0.5 * pixel_size_um
        sigma_min = (0.25 * pixel_size_um) ** 2
        sigma_max = (fit_radius * pixel_size_um * 3.0) ** 2
        lb = [0.0, -center_bound, -center_bound, sigma_min, -np.inf]
        ub = [np.inf, center_bound, center_bound, sigma_max, np.inf]

        try:
            popt, _ = optimize.curve_fit(
                lambda coords, A, x0, y0, sigma2, B: model(coords, A, x0, y0, sigma2, B),
                (x_flat, y_flat),
                z_flat,
                p0=p0,
                bounds=(lb, ub),
                maxfev=5000
            )
            yhat = model((x_flat, y_flat), *popt)
            ss_res = float(np.sum((z_flat - yhat) ** 2))
            ss_tot = float(np.sum((z_flat - np.mean(z_flat)) ** 2) + 1e-12)
            r2 = float(1.0 - ss_res / ss_tot)
            return {
                'converged': True,
                'A': float(popt[0]),
                'x0_um': float(popt[1]),
                'y0_um': float(popt[2]),
                'sigma2_um2': float(popt[3]),
                'B': float(popt[4]),
                'r_squared': r2,
                'residual_ss': ss_res
            }
        except Exception:
            # Robust fallback: second-moment estimate around center.
            z = sub - np.min(sub)
            z = np.clip(z, 0, None)
            wsum = float(np.sum(z))
            if wsum <= 1e-12:
                sigma2 = float((pixel_size_um ** 2))
            else:
                r2_map = x_um ** 2 + y_um ** 2
                sigma2 = float(max(sigma_min, np.sum(z * r2_map) / wsum))
            return {
                'converged': False,
                'A': float(max(A0, 0.0)),
                'x0_um': 0.0,
                'y0_um': 0.0,
                'sigma2_um2': sigma2,
                'B': float(B0),
                'r_squared': 0.0,
                'residual_ss': float(np.sum((z_flat - np.mean(z_flat)) ** 2))
            }
    
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
        """Deprecated placeholder retained for backward compatibility.

        NOTE: This is not iMSD. Use `compute_imsd_stics`.
        """
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
        
        # Normalize fluctuations
        # G(tau) = <dI(t) dI(t+tau)> / <I>^2
        # Average G over all pixels

        mean_intensity_total = np.mean(image_data)

        for lag in range(max_lag):
            if T - lag > 0:
                # Calculate correlation between frames separated by lag
                frames1 = image_data[:-lag] if lag > 0 else image_data
                frames2 = image_data[lag:]
                
                # Fluctuations
                dI1 = frames1 - pixel_means
                dI2 = frames2 - pixel_means
                
                prod = dI1 * dI2
                # Average over time and pixels
                mean_prod = np.mean(prod)

                temporal_corr[lag] = mean_prod / (mean_intensity_total**2)
        
        return temporal_corr
    
    def _fit_temporal_decay(self, correlation: np.ndarray, time_interval: float) -> Dict[str, float]:
        """Fit exponential decay to temporal correlation"""
        try:
            time_axis = np.arange(len(correlation)) * time_interval
            
            def exp_decay(t, amplitude, tau, offset):
                return amplitude * np.exp(-t / tau) + offset
            
            # Initial guess
            amplitude_guess = correlation[0] if len(correlation) > 0 else 0.1
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

    # --- Lightweight hooks for extended analyses (STICS maps & pair correlation) ---
    def stics(self, stack: np.ndarray, max_tau: int = 10) -> Dict[str, Any]:
        """Compute spatio-temporal correlation G(Δx,Δy; τ) for τ=0..max_tau via FFT.

        Parameters
        ----------
        stack : ndarray (T,Y,X)
        max_tau : int
            Maximum temporal lag.
        """
        stack = np.asarray(stack, dtype=float)
        if stack.ndim != 3:
            raise ValueError("STICS expects (T, Y, X) stack")
        T, Y, X = stack.shape
        taus = min(max_tau, T - 1)
        maps: List[np.ndarray] = []

        # Normalize
        mean_val = np.mean(stack)
        if mean_val == 0: return {'status': 'error'}
        norm_stack = (stack - mean_val) / mean_val

        for lag in range(taus + 1):
            acc = np.zeros((Y, X))

            # Need padded FFT for linear correlation
            pad_y = Y + Y
            pad_x = X + X

            f = np.fft.fft2
            ifft = np.fft.ifft2

            # Full stack correlation in one go if memory allows,
            # else loop like _calculate_stics_correlation_advanced
            # Using _calculate_stics_correlation_advanced logic

            corr = self._calculate_stics_correlation_advanced(stack, max_spatial_lag=min(Y//2, X//2), max_temporal_lag=lag)
            maps.append(corr[lag])

        return {'status': 'success', 'tau_maps': maps, 'max_tau': taus}

    def pair_correlation(self, image: np.ndarray, dy: int, dx: int) -> float:
        """Pair-correlation pCF at fixed offset (dy, dx) within a single 2D image."""
        img = np.asarray(image, dtype=float)
        mu = img.mean()
        if mu == 0:
            mu = 1e-12

        # Calculate (I - <I>) / <I>
        norm_img = (img - mu) / mu

        # Shift
        shifted = np.roll(np.roll(norm_img, -dy, axis=0), -dx, axis=1)

        # Mask wrapped edges?
        # For single image, roll wraps.
        # Just compute mean product
        return float(np.mean(norm_img * shifted))

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
            'max_spatial_lag': 6,
            'fit_radius': 6,
            'pixel_size': 0.1,
            'time_interval': 0.1
        },
        'Temporal ICS': {
            'max_temporal_lag': 50,
            'time_interval': 0.001
        }
    }
    
    return parameters.get(method, {})
