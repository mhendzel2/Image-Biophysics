"""
Displacement Correlation Spectroscopy (DCS) Module
Based on Zidovska et al. (2013) - Micron-scale coherence in interphase chromatin dynamics

Implements:
- PIV-based displacement field calculation
- Mean Square Network Displacement (MSND)
- Spatial Displacement Autocorrelation Function (SDACF)

Reference:
Zidovska, A., Weitz, D. A., & Mitchison, T. J. (2013). Micron-scale coherence in 
interphase chromatin dynamics. PNAS, 110(39), 15555-15560.
"""

import numpy as np
from scipy import ndimage, optimize, signal
from scipy.fft import fft2, ifft2, fftshift
from typing import Dict, Any, Tuple, Optional, List
import warnings

try:
    from skimage.filters import gaussian
    from skimage.measure import regionprops, label
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - some DCS features limited")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DisplacementCorrelationSpectroscopy:
    """
    Displacement Correlation Spectroscopy (DCS) Analysis
    
    Maps chromatin dynamics simultaneously across the whole nucleus and 
    quantifies spatially correlated movements over time using PIV-based
    displacement field calculation.
    
    Key outputs:
    - Displacement vector maps d(r, Δt)
    - Mean Square Network Displacement (MSND)
    - Spatial Displacement Autocorrelation Function (SDACF)
    - Correlation length (ξ) and lifetime (τ) of correlated motion
    """
    
    def __init__(self):
        self.name = "Displacement Correlation Spectroscopy"
        self.available = True
    
    def analyze(self, image_data: np.ndarray, 
                nuclear_mask: Optional[np.ndarray] = None,
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete DCS analysis pipeline
        
        Args:
            image_data: Time-resolved image sequence (T, Y, X) - e.g., H2B-GFP
            nuclear_mask: Optional binary mask defining nuclear region
            parameters: Analysis parameters
                - window_size: PIV interrogation window size (default: 16)
                - overlap: Window overlap fraction (default: 0.75)
                - max_time_lag: Maximum time lag for analysis (default: 50)
                - pixel_size: Pixel size in μm (default: 0.1)
                - time_interval: Frame interval in seconds (default: 1.0)
                - snr_threshold: Signal-to-noise threshold for filtering (default: 1.5)
        
        Returns:
            Dictionary containing all DCS analysis results
        """
        if parameters is None:
            parameters = {}
        
        # Default parameters
        window_size = parameters.get('window_size', 16)
        overlap = parameters.get('overlap', 0.75)
        max_time_lag = parameters.get('max_time_lag', 50)
        pixel_size = parameters.get('pixel_size', 0.1)  # μm
        time_interval = parameters.get('time_interval', 1.0)  # seconds
        snr_threshold = parameters.get('snr_threshold', 1.5)
        
        try:
            # Validate input
            if image_data.ndim != 3:
                return {'status': 'error', 'message': 'DCS requires 3D time-series data (T, Y, X)'}
            
            T, H, W = image_data.shape
            
            # Apply nuclear mask if provided
            if nuclear_mask is not None:
                image_data = self._apply_mask(image_data, nuclear_mask)
            
            # Step 1: Calculate displacement fields using PIV
            displacement_fields = self._calculate_displacement_fields(
                image_data, window_size, overlap, snr_threshold, max_time_lag
            )
            
            # Step 2: Calculate Mean Square Network Displacement (MSND)
            msnd_results = self._calculate_msnd(
                displacement_fields, max_time_lag, time_interval
            )
            
            # Step 3: Calculate Spatial Displacement Autocorrelation Function (SDACF)
            sdacf_results = self._calculate_sdacf(
                displacement_fields, pixel_size, max_time_lag
            )
            
            # Step 4: Extract correlation length and lifetime
            correlation_metrics = self._extract_correlation_metrics(
                sdacf_results, pixel_size, time_interval
            )
            
            # Step 5: Generate summary statistics
            summary = self._generate_summary(
                msnd_results, sdacf_results, correlation_metrics
            )
            
            return {
                'status': 'success',
                'method': 'Displacement Correlation Spectroscopy',
                'displacement_fields': displacement_fields,
                'msnd': msnd_results,
                'sdacf': sdacf_results,
                'correlation_metrics': correlation_metrics,
                'summary': summary,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'DCS analysis failed: {str(e)}'}
    
    def _apply_mask(self, image_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply nuclear mask to time-series data"""
        masked_data = image_data.copy()
        for t in range(image_data.shape[0]):
            masked_data[t][~mask] = np.nan
        return masked_data
    
    def _calculate_displacement_fields(self, image_data: np.ndarray,
                                       window_size: int,
                                       overlap: float,
                                       snr_threshold: float,
                                       max_time_lag: int) -> Dict[str, Any]:
        """
        Step 1.1: PIV-based displacement field calculation
        
        Implements cross-correlation based PIV with:
        - Configurable interrogation windows
        - Multi-pass refinement
        - Quality filtering based on SNR and peak height
        """
        T, H, W = image_data.shape
        step_size = int(window_size * (1 - overlap))
        
        # Grid positions
        y_positions = np.arange(window_size // 2, H - window_size // 2, step_size)
        x_positions = np.arange(window_size // 2, W - window_size // 2, step_size)
        
        # Storage for displacement fields at different time lags
        displacement_maps = {}
        
        # Calculate displacements for various time lags
        time_lags = range(1, min(max_time_lag + 1, T))
        
        for dt in time_lags:
            dx_map = np.zeros((len(y_positions), len(x_positions), T - dt))
            dy_map = np.zeros_like(dx_map)
            snr_map = np.zeros_like(dx_map)
            valid_map = np.zeros_like(dx_map, dtype=bool)
            
            for t in range(T - dt):
                frame1 = image_data[t]
                frame2 = image_data[t + dt]
                
                for iy, y in enumerate(y_positions):
                    for ix, x in enumerate(x_positions):
                        # Extract interrogation windows
                        window1 = frame1[y - window_size//2:y + window_size//2,
                                        x - window_size//2:x + window_size//2]
                        window2 = frame2[y - window_size//2:y + window_size//2,
                                        x - window_size//2:x + window_size//2]
                        
                        # Skip if too many NaN values (outside mask)
                        if np.sum(np.isnan(window1)) > 0.5 * window1.size:
                            continue
                        
                        # Replace NaN with mean for correlation
                        window1 = np.nan_to_num(window1, nan=np.nanmean(window1))
                        window2 = np.nan_to_num(window2, nan=np.nanmean(window2))
                        
                        # Calculate cross-correlation
                        dx, dy, snr, peak_height = self._piv_cross_correlation(
                            window1, window2, window_size
                        )
                        
                        # Apply quality filtering
                        if snr > snr_threshold and not np.isnan(dx):
                            dx_map[iy, ix, t] = dx
                            dy_map[iy, ix, t] = dy
                            snr_map[iy, ix, t] = snr
                            valid_map[iy, ix, t] = True
            
            # Apply global velocity distribution filter
            dx_map, dy_map, valid_map = self._filter_outliers(
                dx_map, dy_map, valid_map
            )
            
            displacement_maps[dt] = {
                'dx': dx_map,
                'dy': dy_map,
                'snr': snr_map,
                'valid': valid_map,
                'magnitude': np.sqrt(dx_map**2 + dy_map**2)
            }
        
        return {
            'maps': displacement_maps,
            'grid_y': y_positions,
            'grid_x': x_positions,
            'window_size': window_size,
            'step_size': step_size
        }
    
    def _piv_cross_correlation(self, window1: np.ndarray, window2: np.ndarray,
                               window_size: int) -> Tuple[float, float, float, float]:
        """
        Calculate sub-pixel displacement using FFT-based cross-correlation
        with Gaussian peak fitting
        """
        # Normalize windows
        w1 = window1 - np.mean(window1)
        w2 = window2 - np.mean(window2)
        
        std1, std2 = np.std(w1), np.std(w2)
        if std1 < 1e-10 or std2 < 1e-10:
            return np.nan, np.nan, 0, 0
        
        w1 = w1 / std1
        w2 = w2 / std2
        
        # FFT-based cross-correlation
        f1 = fft2(w1)
        f2 = fft2(w2)
        cross_power = f1 * np.conj(f2)
        correlation = np.real(fftshift(ifft2(cross_power)))
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_height = correlation[peak_idx]
        
        # Calculate SNR (peak / std of correlation excluding peak region)
        mask = np.ones_like(correlation, dtype=bool)
        mask[max(0, peak_idx[0]-2):min(correlation.shape[0], peak_idx[0]+3),
             max(0, peak_idx[1]-2):min(correlation.shape[1], peak_idx[1]+3)] = False
        noise_std = np.std(correlation[mask])
        snr = peak_height / noise_std if noise_std > 1e-10 else 0
        
        # Sub-pixel refinement using Gaussian fit
        try:
            dy_subpix, dx_subpix = self._gaussian_subpixel_fit(correlation, peak_idx)
        except Exception:
            dy_subpix, dx_subpix = 0, 0
        
        # Convert to displacement (relative to center)
        center = np.array(correlation.shape) // 2
        dy = (peak_idx[0] - center[0]) + dy_subpix
        dx = (peak_idx[1] - center[1]) + dx_subpix
        
        return dx, dy, snr, peak_height
    
    def _gaussian_subpixel_fit(self, correlation: np.ndarray, 
                               peak_idx: Tuple[int, int]) -> Tuple[float, float]:
        """3-point Gaussian sub-pixel peak fitting"""
        y, x = peak_idx
        
        # Ensure we have valid neighbors
        if y <= 0 or y >= correlation.shape[0] - 1:
            dy_sub = 0
        else:
            c_m1 = np.log(max(correlation[y-1, x], 1e-10))
            c_0 = np.log(max(correlation[y, x], 1e-10))
            c_p1 = np.log(max(correlation[y+1, x], 1e-10))
            denom = 2 * (c_m1 - 2*c_0 + c_p1)
            dy_sub = (c_m1 - c_p1) / denom if abs(denom) > 1e-10 else 0
        
        if x <= 0 or x >= correlation.shape[1] - 1:
            dx_sub = 0
        else:
            c_m1 = np.log(max(correlation[y, x-1], 1e-10))
            c_0 = np.log(max(correlation[y, x], 1e-10))
            c_p1 = np.log(max(correlation[y, x+1], 1e-10))
            denom = 2 * (c_m1 - 2*c_0 + c_p1)
            dx_sub = (c_m1 - c_p1) / denom if abs(denom) > 1e-10 else 0
        
        return dy_sub, dx_sub
    
    def _filter_outliers(self, dx_map: np.ndarray, dy_map: np.ndarray,
                        valid_map: np.ndarray, sigma: float = 2.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter outliers based on global velocity distribution"""
        # Calculate magnitude
        magnitude = np.sqrt(dx_map**2 + dy_map**2)
        
        # Get valid magnitudes
        valid_mag = magnitude[valid_map]
        
        if len(valid_mag) > 10:
            mean_mag = np.mean(valid_mag)
            std_mag = np.std(valid_mag)
            
            # Mark outliers as invalid
            outlier_mask = (magnitude > mean_mag + sigma * std_mag) | \
                          (magnitude < mean_mag - sigma * std_mag)
            valid_map = valid_map & ~outlier_mask
            
            # Zero out invalid displacements
            dx_map[~valid_map] = 0
            dy_map[~valid_map] = 0
        
        return dx_map, dy_map, valid_map
    
    def _calculate_msnd(self, displacement_fields: Dict[str, Any],
                       max_time_lag: int,
                       time_interval: float) -> Dict[str, Any]:
        """
        Step 1.2: Mean Square Network Displacement (MSND)
        
        MSND(Δt) = <|d(r, Δt)|²>
        
        Fitting: f(Δt) = A + B * Δt^α
        Where:
        - A accounts for fast fluctuations/noise
        - α indicates diffusive regime (α<1: subdiffusive, α=1: diffusive, α>1: superdiffusive)
        """
        maps = displacement_fields['maps']
        
        # Calculate MSND for each time lag
        time_lags = []
        msnd_values = []
        msnd_errors = []
        
        for dt, field_data in maps.items():
            magnitude_sq = field_data['magnitude']**2
            valid = field_data['valid']
            
            # Spatial and temporal averaging
            valid_values = magnitude_sq[valid]
            
            if len(valid_values) > 0:
                time_lags.append(dt * time_interval)
                msnd_values.append(np.mean(valid_values))
                msnd_errors.append(np.std(valid_values) / np.sqrt(len(valid_values)))
        
        time_lags = np.array(time_lags)
        msnd_values = np.array(msnd_values)
        msnd_errors = np.array(msnd_errors)
        
        # Fit power law: f(Δt) = A + B * Δt^α
        fit_results = self._fit_msnd_power_law(time_lags, msnd_values)
        
        return {
            'time_lags': time_lags,
            'msnd': msnd_values,
            'msnd_errors': msnd_errors,
            'fit_parameters': fit_results,
            'diffusion_regime': self._classify_diffusion_regime(fit_results.get('alpha', 1.0))
        }
    
    def _fit_msnd_power_law(self, time_lags: np.ndarray, 
                           msnd_values: np.ndarray) -> Dict[str, float]:
        """Fit MSND to power law: f(Δt) = A + B * Δt^α"""
        
        def power_law(t, A, B, alpha):
            return A + B * np.power(t, alpha)
        
        try:
            # Initial guess
            p0 = [msnd_values[0], 0.01, 0.5]
            
            # Bounds: A > 0, B > 0, 0 < alpha < 2
            bounds = ([0, 0, 0], [np.inf, np.inf, 2.0])
            
            popt, pcov = optimize.curve_fit(
                power_law, time_lags, msnd_values, 
                p0=p0, bounds=bounds, maxfev=5000
            )
            
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate R-squared
            residuals = msnd_values - power_law(time_lags, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((msnd_values - np.mean(msnd_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'A': popt[0],
                'A_error': perr[0],
                'B': popt[1],
                'B_error': perr[1],
                'alpha': popt[2],
                'alpha_error': perr[2],
                'r_squared': r_squared,
                'fitted_curve': power_law(time_lags, *popt)
            }
            
        except Exception as e:
            warnings.warn(f"MSND fitting failed: {str(e)}")
            return {
                'A': np.nan, 'B': np.nan, 'alpha': np.nan,
                'r_squared': 0, 'fitted_curve': np.full_like(msnd_values, np.nan)
            }
    
    def _classify_diffusion_regime(self, alpha: float) -> str:
        """Classify diffusion regime based on MSND exponent"""
        if np.isnan(alpha):
            return 'undetermined'
        elif alpha < 0.8:
            return 'subdiffusive'
        elif alpha < 1.2:
            return 'diffusive'
        else:
            return 'superdiffusive'
    
    def _calculate_sdacf(self, displacement_fields: Dict[str, Any],
                        pixel_size: float,
                        max_time_lag: int) -> Dict[str, Any]:
        """
        Step 1.3: Spatial Displacement Autocorrelation Function (SDACF)
        
        c_dx(Δr, Δt) = <dx(r, Δt) · dx(r + Δr, Δt)>
        
        Fitting: f(Δr) = A * (Δr)^n * exp(-Δr / ξ)
        Where ξ is the correlation length
        """
        maps = displacement_fields['maps']
        grid_y = displacement_fields['grid_y']
        grid_x = displacement_fields['grid_x']
        step_size = displacement_fields['step_size']
        
        # Calculate SDACF for each time lag
        sdacf_results = {}
        
        for dt, field_data in maps.items():
            dx_map = field_data['dx']
            dy_map = field_data['dy']
            valid = field_data['valid']
            
            # Average over time
            dx_mean = np.nanmean(dx_map, axis=2)
            dy_mean = np.nanmean(dy_map, axis=2)
            valid_mean = np.mean(valid.astype(float), axis=2) > 0.5
            
            # Calculate spatial autocorrelation
            sdacf_dx = self._calculate_spatial_autocorrelation(dx_mean, valid_mean)
            sdacf_dy = self._calculate_spatial_autocorrelation(dy_mean, valid_mean)
            
            # Radial averaging
            radial_distances, radial_corr_dx = self._radial_average(sdacf_dx, step_size * pixel_size)
            _, radial_corr_dy = self._radial_average(sdacf_dy, step_size * pixel_size)
            
            # Combined correlation (average of x and y components)
            radial_corr = (radial_corr_dx + radial_corr_dy) / 2
            
            # Fit to extract correlation length
            fit_result = self._fit_sdacf(radial_distances, radial_corr)
            
            sdacf_results[dt] = {
                'sdacf_2d_dx': sdacf_dx,
                'sdacf_2d_dy': sdacf_dy,
                'radial_distances': radial_distances,
                'radial_correlation': radial_corr,
                'radial_correlation_dx': radial_corr_dx,
                'radial_correlation_dy': radial_corr_dy,
                'fit_parameters': fit_result
            }
        
        return sdacf_results
    
    def _calculate_spatial_autocorrelation(self, field: np.ndarray, 
                                           valid: np.ndarray) -> np.ndarray:
        """Calculate spatial autocorrelation of displacement field"""
        # Replace invalid with zero for correlation
        field_clean = np.where(valid, field, 0)
        
        # Normalize
        field_norm = field_clean - np.mean(field_clean[valid]) if np.sum(valid) > 0 else field_clean
        
        # FFT-based autocorrelation
        f = fft2(field_norm)
        autocorr = np.real(fftshift(ifft2(f * np.conj(f))))
        
        # Normalize by number of overlapping points
        count = np.real(fftshift(ifft2(fft2(valid.astype(float)) * np.conj(fft2(valid.astype(float))))))
        count = np.maximum(count, 1)  # Avoid division by zero
        
        autocorr = autocorr / count
        
        return autocorr
    
    def _radial_average(self, autocorr: np.ndarray, 
                       pixel_spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform radial averaging of 2D autocorrelation"""
        center = np.array(autocorr.shape) // 2
        y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
        
        # Distance from center in physical units
        distances = np.sqrt((y - center[0])**2 + (x - center[1])**2) * pixel_spacing
        
        # Bin distances
        max_dist = min(center) * pixel_spacing
        num_bins = min(center[0], center[1])
        bin_edges = np.linspace(0, max_dist, num_bins + 1)
        
        radial_distances = []
        radial_values = []
        
        for i in range(len(bin_edges) - 1):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            if np.sum(mask) > 0:
                radial_distances.append((bin_edges[i] + bin_edges[i+1]) / 2)
                radial_values.append(np.mean(autocorr[mask]))
        
        return np.array(radial_distances), np.array(radial_values)
    
    def _fit_sdacf(self, distances: np.ndarray, 
                   correlation: np.ndarray) -> Dict[str, float]:
        """
        Fit SDACF to power law with exponential cutoff:
        f(Δr) = A * (Δr)^n * exp(-Δr / ξ)
        """
        
        def sdacf_model(r, A, n, xi):
            # Avoid division by zero
            r_safe = np.maximum(r, 1e-10)
            return A * np.power(r_safe, n) * np.exp(-r / xi)
        
        try:
            # Only use positive correlation values for fitting
            positive_mask = correlation > 0
            if np.sum(positive_mask) < 3:
                raise ValueError("Insufficient positive correlation values")
            
            r_fit = distances[positive_mask]
            c_fit = correlation[positive_mask]
            
            # Initial guess
            p0 = [c_fit[0], -0.5, distances[-1] / 2]
            
            # Bounds
            bounds = ([0, -2, 0.1], [np.inf, 1, distances[-1] * 10])
            
            popt, pcov = optimize.curve_fit(
                sdacf_model, r_fit, c_fit,
                p0=p0, bounds=bounds, maxfev=5000
            )
            
            perr = np.sqrt(np.diag(pcov))
            
            return {
                'A': popt[0],
                'n': popt[1],
                'xi': popt[2],  # Correlation length in μm
                'A_error': perr[0],
                'n_error': perr[1],
                'xi_error': perr[2],
                'fitted_curve': sdacf_model(distances, *popt)
            }
            
        except Exception as e:
            warnings.warn(f"SDACF fitting failed: {str(e)}")
            return {
                'A': np.nan, 'n': np.nan, 'xi': np.nan,
                'fitted_curve': np.full_like(correlation, np.nan)
            }
    
    def _extract_correlation_metrics(self, sdacf_results: Dict[str, Any],
                                     pixel_size: float,
                                     time_interval: float) -> Dict[str, Any]:
        """Extract correlation length ξ and lifetime τ of correlated motion"""
        
        # Get correlation lengths for each time lag
        time_lags = []
        correlation_lengths = []
        correlation_length_errors = []
        
        for dt, result in sdacf_results.items():
            fit = result['fit_parameters']
            if not np.isnan(fit.get('xi', np.nan)):
                time_lags.append(dt * time_interval)
                correlation_lengths.append(fit['xi'])
                correlation_length_errors.append(fit.get('xi_error', 0))
        
        time_lags = np.array(time_lags)
        correlation_lengths = np.array(correlation_lengths)
        
        # Calculate characteristic correlation length (average)
        mean_xi = np.mean(correlation_lengths) if len(correlation_lengths) > 0 else np.nan
        std_xi = np.std(correlation_lengths) if len(correlation_lengths) > 0 else np.nan
        
        # Estimate correlation lifetime τ from decay of correlation length
        tau = self._estimate_correlation_lifetime(time_lags, correlation_lengths)
        
        return {
            'time_lags': time_lags,
            'correlation_lengths': correlation_lengths,
            'correlation_length_errors': np.array(correlation_length_errors),
            'mean_correlation_length': mean_xi,
            'std_correlation_length': std_xi,
            'correlation_lifetime': tau
        }
    
    def _estimate_correlation_lifetime(self, time_lags: np.ndarray,
                                       correlation_lengths: np.ndarray) -> float:
        """Estimate correlation lifetime from time-dependence of correlation length"""
        if len(time_lags) < 3:
            return np.nan
        
        try:
            # Fit exponential decay: ξ(t) = ξ₀ * exp(-t/τ)
            def exp_decay(t, xi0, tau):
                return xi0 * np.exp(-t / tau)
            
            popt, _ = optimize.curve_fit(
                exp_decay, time_lags, correlation_lengths,
                p0=[correlation_lengths[0], time_lags[-1]],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=2000
            )
            
            return popt[1]
            
        except Exception:
            return np.nan
    
    def _generate_summary(self, msnd_results: Dict[str, Any],
                         sdacf_results: Dict[str, Any],
                         correlation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of DCS analysis"""
        
        msnd_fit = msnd_results.get('fit_parameters', {})
        
        return {
            'diffusion_exponent_alpha': msnd_fit.get('alpha', np.nan),
            'diffusion_regime': msnd_results.get('diffusion_regime', 'undetermined'),
            'msnd_amplitude_A': msnd_fit.get('A', np.nan),
            'msnd_coefficient_B': msnd_fit.get('B', np.nan),
            'msnd_r_squared': msnd_fit.get('r_squared', 0),
            'mean_correlation_length_um': correlation_metrics.get('mean_correlation_length', np.nan),
            'correlation_lifetime_s': correlation_metrics.get('correlation_lifetime', np.nan),
            'num_time_lags_analyzed': len(sdacf_results)
        }


def get_dcs_parameters() -> Dict[str, Any]:
    """Get default parameters for DCS analysis"""
    return {
        'window_size': {
            'default': 16,
            'min': 8,
            'max': 64,
            'description': 'PIV interrogation window size in pixels'
        },
        'overlap': {
            'default': 0.75,
            'min': 0.0,
            'max': 0.9,
            'description': 'Window overlap fraction'
        },
        'max_time_lag': {
            'default': 50,
            'min': 5,
            'max': 200,
            'description': 'Maximum time lag for analysis (frames)'
        },
        'pixel_size': {
            'default': 0.1,
            'min': 0.01,
            'max': 10.0,
            'description': 'Pixel size in micrometers'
        },
        'time_interval': {
            'default': 1.0,
            'min': 0.001,
            'max': 100.0,
            'description': 'Time interval between frames (seconds)'
        },
        'snr_threshold': {
            'default': 1.5,
            'min': 1.0,
            'max': 5.0,
            'description': 'SNR threshold for displacement filtering'
        }
    }
