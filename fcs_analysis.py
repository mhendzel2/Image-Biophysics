"""
Segmented Fluorescence Correlation Spectroscopy Analysis
Specialized FCS implementation for line-scan data with temporal segmentation
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
from typing import Dict, Any, List, Tuple

def fcs_model_2d(tau, G0, D, w0):
    """2D FCS diffusion model"""
    # Note: G0 is inversely proportional to N (number of particles)
    return G0 / (1 + 4 * D * tau / w0**2)

def fcs_model_3d(tau, G0, D, w0, wz):
    """3D FCS diffusion model with axial extent"""
    return G0 / ((1 + 4 * D * tau / w0**2) * np.sqrt(1 + 4 * D * tau / wz**2))

def fcs_model_anomalous(tau, G0, D, w0, alpha):
    """Anomalous diffusion FCS model"""
    return G0 / (1 + (4 * D * tau / w0**2)**alpha)

def fit_fcs_data(tau, acf, model_func=fcs_model_2d, bounds=None):
    """Fits FCS data to a given model"""
    try:
        acf = np.nan_to_num(acf, nan=0, posinf=0, neginf=0)  # Handle potential NaNs or Infs
        # Initial guesses
        G0_guess = acf[0] if len(acf) > 0 and acf[0] > 0 else 0.01
        w0_guess = 0.2  # Typical beam waist in um
        
        # Find half-maximum point for D estimate
        half_max_idx = np.argmin(np.abs(acf - G0_guess / 2)) if G0_guess > 0 else len(acf) // 4
        if half_max_idx < len(tau) and half_max_idx > 0 and tau[half_max_idx] > 0:
            D_guess = w0_guess**2 / (4 * tau[half_max_idx])
        else:
            D_guess = 1.0 # Default guess if determination fails
        
        if model_func == fcs_model_2d:
            p0 = [G0_guess, D_guess, w0_guess]
            if bounds is None:
                bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        elif model_func == fcs_model_3d:
            wz_guess = w0_guess * 5  # Typical axial extent
            p0 = [G0_guess, D_guess, w0_guess, wz_guess]
            if bounds is None:
                bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        elif model_func == fcs_model_anomalous:
            alpha_guess = 0.8  # Subdiffusive
            p0 = [G0_guess, D_guess, w0_guess, alpha_guess]
            if bounds is None:
                bounds = ([0, 0, 0, 0.1], [np.inf, np.inf, np.inf, 2.0])
        else:
            p0 = [G0_guess, D_guess, w0_guess]
            if bounds is None:
                bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        popt, pcov = curve_fit(model_func, tau, acf, p0=p0, bounds=bounds, maxfev=5000)
        
        # Calculate fit quality
        fitted_acf = model_func(tau, *popt)
        r_squared = 1 - np.sum((acf - fitted_acf)**2) / np.sum((acf - np.mean(acf))**2)
        
        return popt, pcov, r_squared
        
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"FCS fitting failed: {str(e)}")
        if model_func == fcs_model_2d:
            return [0, 0, 0], np.zeros((3, 3)), 0
        elif model_func == fcs_model_3d:
            return [0, 0, 0, 0], np.zeros((4, 4)), 0
        elif model_func == fcs_model_anomalous:
            return [0, 0, 0, 1], np.zeros((4, 4)), 0
        else:
            return [0, 0, 0], np.zeros((3, 3)), 0

def calculate_autocorrelation(intensity_trace, normalize=True, max_lag=None):
    """
    Calculate normalized autocorrelation function.
    G(tau) = <delta_I(t) * delta_I(t+tau)> / <I(t)>^2
    This normalization preserves the G(0) = 1/N relationship.
    """
    
    if len(intensity_trace) < 10:
        return np.array([0]), np.array([0])
    
    # Calculate mean intensity
    mean_intensity = np.mean(intensity_trace)
    if mean_intensity == 0:
         return np.array([0]), np.array([0])

    # Calculate fluctuations
    trace_fluctuations = intensity_trace - mean_intensity
    
    # Calculate maximum lag
    N = len(trace_fluctuations)
    if max_lag is None:
        max_lag = N // 4
    else:
        max_lag = min(max_lag, N // 2)
    
    # Calculate autocorrelation using numpy's correlate
    # mode='full' returns the convolution sum.
    # At lag tau, the sum is over N-tau elements.
    full_corr = np.correlate(trace_fluctuations, trace_fluctuations, mode='full')
    mid_point = len(full_corr) // 2
    
    # Extract the positive lags
    acf_sum = full_corr[mid_point:mid_point + max_lag]

    # Correct for the number of overlapping points (triangle bias)
    # The number of overlapping points for lag tau is N - tau
    lags = np.arange(len(acf_sum))
    normalization_factor = N - lags

    # Avoid division by zero
    normalization_factor[normalization_factor == 0] = 1

    # Calculate covariance: (1/(N-tau)) * sum(...)
    acf_cov = acf_sum / normalization_factor

    # Normalize by mean squared: G(tau) = Cov(tau) / <I>^2
    if normalize:
        acf = acf_cov / (mean_intensity**2)
    else:
        acf = acf_cov
    
    return acf

def segmented_fcs_analysis(image_data, segmentation_type='x', segment_length=128, 
                          pixel_time=3.05e-6, line_time=0.56e-3, pixel_size=0.05,
                          model_type='2d', max_lag_fraction=0.25):
    """
    Performs segmented FCS analysis on image data.
    
    Args:
        image_data: 2D array (pixels, lines) where lines typically represent time.
                    Shape: (num_pixels, num_lines) or (X, T) for line scan.
        segmentation_type: 'x' (segments along pixels axis) or 'y' (segments along lines axis)
        segment_length: Length of each segment
        pixel_time: Time per pixel (seconds) - effective time step if correlating along pixels
        line_time: Time per line (seconds) - effective time step if correlating along lines
        pixel_size: Physical pixel size (um)
        model_type: '2d', '3d', or 'anomalous'
        max_lag_fraction: Maximum lag as fraction of segment length
    
    Returns:
        List of analysis results for each segment
    """
    
    if image_data.ndim != 2:
        raise ValueError("Segmented FCS requires 2D image data (pixels, lines).")
    
    num_pixels, num_lines = image_data.shape
    
    # Select fitting model
    if model_type == '2d':
        model_func = fcs_model_2d
    elif model_type == '3d':
        model_func = fcs_model_3d
    elif model_type == 'anomalous':
        model_func = fcs_model_anomalous
    else:
        model_func = fcs_model_2d
    
    results = []
    
    if segmentation_type == 'x':
        # Segment along x-axis (pixels)
        # We assume we want to analyze temporal fluctuations within each spatial segment.
        # If image_data is (pixels, lines) and lines=time:
        # We take a chunk of pixels. We can average the pixels in the chunk to get a lower-noise time trace.
        # Or we can analyze each pixel separately and average ACFs (more rigorous but slower).
        # Here we will average the intensity spatially (axis 0 of segment) to get a time trace.

        num_segments = num_pixels // segment_length
        # Correlation is along time (lines)
        time_step = line_time
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length
            
            # Segment shape: (segment_length, num_lines)
            segment = image_data[start_idx:end_idx, :]

            # Average over spatial dimension (axis 0) to get time trace
            mean_intensity_trace = np.mean(segment, axis=0)
            
            # Calculate maximum lag
            max_lag = int(len(mean_intensity_trace) * max_lag_fraction)
            
            # Calculate ACF
            acf = calculate_autocorrelation(mean_intensity_trace, max_lag=max_lag)
            tau = np.arange(len(acf)) * time_step
            
            # Fit ACF
            params, covariance, r_squared = fit_fcs_data(tau, acf, model_func)
            
            # Calculate parameter errors
            param_errors = np.sqrt(np.diag(covariance)) if covariance.size > 0 else np.zeros_like(params)
            
            # Store results
            result  = {
                'segment_index': i,
                'segment_type': 'x (spatial segment)',
                'segment_position': start_idx * pixel_size,
                'G0': params[0],
                'D': params[1],
                'w0': params[2] if len(params) > 2 else 0,
                'G0_error': param_errors[0] if len(param_errors) > 0 else 0,
                'D_error': param_errors[1] if len(param_errors) > 1 else 0,
                'w0_error': param_errors[2] if len(param_errors) > 2 else 0,
                'r_squared': r_squared,
                'acf': acf,
                'tau': tau,
                'mean_intensity': np.mean(mean_intensity_trace),
                'intensity_std': np.std(mean_intensity_trace)
            }
            
            # Add model-specific parameters
            if model_type == '3d' and len(params) > 3:
                result['wz'] = params[3]
                result['wz_error'] = param_errors[3] if len(param_errors) > 3 else 0
            elif model_type == 'anomalous' and len(params) > 3:
                result['alpha'] = params[3]
                result['alpha_error'] = param_errors[3] if len(param_errors) > 3 else 0
            
            results.append(result)
            
    elif segmentation_type == 'y':
        # Segment along y-axis (lines/time)
        # This breaks the long time trace into shorter chunks.
        # We average over all pixels to get a single time trace, then segment that trace.
        # OR we maintain the pixels and segment the time axis of the 2D array.

        # Usually, if we segment time, we want to see how diffusion changes over time.
        # So for each time segment, we calculate one ACF (spatially averaged).

        num_segments = num_lines // segment_length
        time_step = line_time
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length
            
            # Segment shape: (num_pixels, segment_length)
            segment = image_data[:, start_idx:end_idx]

            # Average over spatial dimension (axis 0) to get time trace for this segment
            mean_intensity_trace = np.mean(segment, axis=0)
            
            # Calculate maximum lag
            max_lag = int(len(mean_intensity_trace) * max_lag_fraction)
            
            # Calculate ACF
            acf = calculate_autocorrelation(mean_intensity_trace, max_lag=max_lag)
            tau = np.arange(len(acf)) * time_step
            
            # Fit ACF
            params, covariance, r_squared = fit_fcs_data(tau, acf, model_func)
            
            # Calculate parameter errors
            param_errors = np.sqrt(np.diag(covariance)) if covariance.size > 0 else np.zeros_like(params)
            
            # Store results
            result  = {
                'segment_index': i,
                'segment_type': 'y (temporal segment)',
                'segment_position': start_idx * line_time, # Time position
                'G0': params[0],
                'D': params[1],
                'w0': params[2] if len(params) > 2 else 0,
                'G0_error': param_errors[0] if len(param_errors) > 0 else 0,
                'D_error': param_errors[1] if len(param_errors) > 1 else 0,
                'w0_error': param_errors[2] if len(param_errors) > 2 else 0,
                'r_squared': r_squared,
                'acf': acf,
                'tau': tau,
                'mean_intensity': np.mean(mean_intensity_trace),
                'intensity_std': np.std(mean_intensity_trace)
            }
            
            # Add model-specific parameters
            if model_type == '3d' and len(params) > 3:
                result['wz'] = params[3]
                result['wz_error'] = param_errors[3] if len(param_errors) > 3 else 0
            elif model_type == 'anomalous' and len(params) > 3:
                result['alpha'] = params[3]
                result['alpha_error'] = param_errors[3] if len(param_errors) > 3 else 0
            
            results.append(result)
    else:
        raise ValueError("Invalid segmentation type. Choose 'x' or 'y'.")
    
    return results

def analyze_segment_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze statistics across all segments"""
    
    if not results:
        return {'status': 'error', 'message': 'No results to analyze'}
    
    # Extract valid diffusion coefficients with additional validation
    valid_D = [r['D'] for r in results if 'D' in r and r['D'] > 0 and r.get('r_squared', 0) > 0.5]
    valid_G0 = [r['G0'] for r in results if 'G0' in r and r['G0'] > 0 and r.get('r_squared', 0) > 0.5]
    valid_w0 = [r['w0'] for r in results if 'w0' in r and r['w0'] > 0 and r.get('r_squared', 0) > 0.5]
    r_squared_values = [r['r_squared'] for r in results if 'r_squared' in r]
    
    statistics = {
        'num_segments_total': len(results),
        'num_segments_valid': len(valid_D),
        'success_rate': len(valid_D) / len(results) if len(results) > 0 else 0,
        'mean_D': float(np.mean(valid_D)) if valid_D else 0.0,
        'std_D': float(np.std(valid_D)) if valid_D else 0.0,
        'median_D': float(np.median(valid_D)) if valid_D else 0.0,
        'mean_G0': float(np.mean(valid_G0)) if valid_G0 else 0.0,
        'std_G0': float(np.std(valid_G0)) if valid_G0 else 0.0,
        'mean_w0': float(np.mean(valid_w0)) if valid_w0 else 0.0,
        'std_w0': float(np.std(valid_w0)) if valid_w0 else 0.0,
        'mean_r_squared': float(np.mean(r_squared_values)) if r_squared_values else 0.0,
        'segment_type': results[0].get('segment_type', 'unknown') if results else 'unknown'
    }
    
    # Check for anomalous diffusion parameters
    if any('alpha' in r for r in results):
        valid_alpha = [r['alpha'] for r in results if 'alpha' in r and r.get('r_squared', 0) > 0.5]
        statistics.update({
            'mean_alpha': float(np.mean(valid_alpha)) if valid_alpha else 1.0,
            'std_alpha': float(np.std(valid_alpha)) if valid_alpha else 0.0
        })
    
    # Check for 3D parameters
    if any('wz' in r for r in results):
        valid_wz = [r['wz'] for r in results if 'wz' in r and r.get('r_squared', 0) > 0.5]
        statistics.update({
            'mean_wz': float(np.mean(valid_wz)) if valid_wz else 0.0,
            'std_wz': float(np.std(valid_wz)) if valid_wz else 0.0
        })
    
    return statistics
