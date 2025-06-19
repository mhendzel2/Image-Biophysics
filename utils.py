"""
Utility Functions
Helper functions for data processing, validation, and formatting
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
import os
from pathlib import Path

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def validate_analysis_parameters(analysis_type: str, parameters: Dict[str, Any]) -> bool:
    """
    Validate analysis parameters for the given analysis type
    
    Args:
        analysis_type: Type of analysis
        parameters: Parameter dictionary
        
    Returns:
        True if parameters are valid
    """
    
    try:
        if "RICS" in analysis_type:
            return validate_rics_parameters(parameters)
        elif "FCS" in analysis_type:
            return validate_fcs_parameters(parameters)
        elif "iMSD" in analysis_type:
            return validate_imsd_parameters(parameters)
        elif "Elastography" in analysis_type or "PIV" in analysis_type:
            return validate_elastography_parameters(parameters)
        elif "N&B" in analysis_type:
            return validate_nb_parameters(parameters)
        elif "FLIM" in analysis_type:
            return validate_flim_parameters(parameters)
        elif "SPT" in analysis_type:
            return validate_spt_parameters(parameters)
        elif "Fourier" in analysis_type:
            return validate_fourier_parameters(parameters)
        elif "FRAP" in analysis_type:
            return validate_frap_parameters(parameters)
        else:
            return True  # Unknown analysis type - allow all parameters
            
    except Exception as e:
        st.error(f"Parameter validation error: {str(e)}")
        return False

def validate_rics_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate RICS analysis parameters"""
    
    required_params = ['tau_max', 'pixel_size', 'time_interval']
    
    for param in required_params:
        if param not in parameters:
            st.error(f"Missing required parameter: {param}")
            return False
    
    # Value range checks
    if parameters['tau_max'] <= 0 or parameters['tau_max'] > 1000:
        st.error("tau_max must be between 1 and 1000")
        return False
    
    if parameters['pixel_size'] <= 0 or parameters['pixel_size'] > 10:
        st.error("pixel_size must be between 0 and 10 µm")
        return False
    
    if parameters['time_interval'] <= 0 or parameters['time_interval'] > 100:
        st.error("time_interval must be between 0 and 100 seconds")
        return False
    
    return True

def validate_fcs_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate FCS analysis parameters"""
    
    if 'correlation_window' in parameters:
        if parameters['correlation_window'] < 4 or parameters['correlation_window'] > 128:
            st.error("correlation_window must be between 4 and 128 pixels")
            return False
    
    if 'binning' in parameters:
        if parameters['binning'] < 1 or parameters['binning'] > 20:
            st.error("binning must be between 1 and 20")
            return False
    
    return True

def validate_imsd_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate iMSD analysis parameters"""
    
    if 'max_displacement' in parameters:
        if parameters['max_displacement'] < 1 or parameters['max_displacement'] > 100:
            st.error("max_displacement must be between 1 and 100 pixels")
            return False
    
    if 'min_track_length' in parameters:
        if parameters['min_track_length'] < 2 or parameters['min_track_length'] > 50:
            st.error("min_track_length must be between 2 and 50 frames")
            return False
    
    if 'localization_error' in parameters:
        if parameters['localization_error'] < 0 or parameters['localization_error'] > 1000:
            st.error("localization_error must be between 0 and 1000 nm")
            return False
    
    return True

def validate_elastography_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate Elastography/PIV parameters"""
    
    if 'window_size' in parameters:
        if parameters['window_size'] < 4 or parameters['window_size'] > 128:
            st.error("window_size must be between 4 and 128 pixels")
            return False
    
    if 'overlap_ratio' in parameters:
        if parameters['overlap_ratio'] < 0 or parameters['overlap_ratio'] > 1:
            st.error("overlap_ratio must be between 0 and 1")
            return False
    
    return True

def validate_nb_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate Number & Brightness parameters"""
    
    if 'brightness_threshold' in parameters:
        if parameters['brightness_threshold'] < 0 or parameters['brightness_threshold'] > 100:
            st.error("brightness_threshold must be between 0 and 100")
            return False
    
    if 'aggregation_cutoff' in parameters:
        if parameters['aggregation_cutoff'] < 0 or parameters['aggregation_cutoff'] > 10:
            st.error("aggregation_cutoff must be between 0 and 10")
            return False
    
    return True

def validate_flim_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate FLIM parameters"""
    
    if 'lifetime_range' in parameters:
        lifetime_range = parameters['lifetime_range']
        if isinstance(lifetime_range, (list, tuple)) and len(lifetime_range) == 2:
            if lifetime_range[0] >= lifetime_range[1]:
                st.error("Invalid lifetime range: min must be less than max")
                return False
            if lifetime_range[0] < 0 or lifetime_range[1] > 100:
                st.error("Lifetime range must be between 0 and 100 ns")
                return False
    
    if 'chi_squared_threshold' in parameters:
        if parameters['chi_squared_threshold'] < 0.1 or parameters['chi_squared_threshold'] > 10:
            st.error("chi_squared_threshold must be between 0.1 and 10")
            return False
    
    return True

def validate_spt_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate Single Particle Tracking parameters"""
    
    if 'detection_threshold' in parameters:
        if parameters['detection_threshold'] < 1 or parameters['detection_threshold'] > 100:
            st.error("detection_threshold must be between 1 and 100")
            return False
    
    if 'linking_distance' in parameters:
        if parameters['linking_distance'] < 1 or parameters['linking_distance'] > 50:
            st.error("linking_distance must be between 1 and 50 pixels")
            return False
    
    if 'gap_closing' in parameters:
        if parameters['gap_closing'] < 0 or parameters['gap_closing'] > 20:
            st.error("gap_closing must be between 0 and 20 frames")
            return False
    
    return True

def validate_fourier_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate Fourier analysis parameters"""
    
    if 'frequency_cutoff' in parameters:
        if parameters['frequency_cutoff'] < 0.01 or parameters['frequency_cutoff'] > 10:
            st.error("frequency_cutoff must be between 0.01 and 10")
            return False
    
    return True

def validate_frap_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate FRAP parameters"""
    
    if 'bleach_roi_size' in parameters:
        if parameters['bleach_roi_size'] < 3 or parameters['bleach_roi_size'] > 200:
            st.error("bleach_roi_size must be between 3 and 200 pixels")
            return False
    
    return True

def standardize_image_format(image_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standardize image format for consistent processing
    
    Args:
        image_data: Input image array
        
    Returns:
        Tuple of (standardized_array, metadata_dict)
    """
    
    metadata = {
        'original_shape': image_data.shape,
        'original_dtype': image_data.dtype,
        'standardized': False
    }
    
    # Ensure float32 data type for processing
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)
        metadata['dtype_converted'] = True
    
    # Normalize to 0-1 range if needed
    if np.max(image_data) > 1.0:
        image_data = image_data / np.max(image_data)
        metadata['normalized'] = True
    
    # Handle negative values
    if np.min(image_data) < 0:
        image_data = image_data - np.min(image_data)
        metadata['baseline_corrected'] = True
    
    metadata['standardized'] = True
    metadata['final_shape'] = image_data.shape
    metadata['final_dtype'] = image_data.dtype
    
    return image_data, metadata

def calculate_image_statistics(image_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive image statistics
    
    Args:
        image_data: Input image array
        
    Returns:
        Dictionary of statistical measures
    """
    
    stats = {}
    
    # Basic statistics
    stats['mean'] = float(np.mean(image_data))
    stats['std'] = float(np.std(image_data))
    stats['min'] = float(np.min(image_data))
    stats['max'] = float(np.max(image_data))
    stats['median'] = float(np.median(image_data))
    
    # Percentiles
    percentiles = [1, 5, 25, 75, 95, 99]
    for p in percentiles:
        stats[f'percentile_{p}'] = float(np.percentile(image_data, p))
    
    # Signal-to-noise ratio (simple estimate)
    signal = np.mean(image_data)
    noise = np.std(image_data)
    stats['snr'] = float(signal / noise) if noise > 0 else float('inf')
    
    # Dynamic range
    stats['dynamic_range'] = stats['max'] - stats['min']
    
    # Coefficient of variation
    stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] > 0 else float('inf')
    
    return stats

def detect_image_properties(image_data: np.ndarray) -> Dict[str, Any]:
    """
    Automatically detect image properties
    
    Args:
        image_data: Input image array
        
    Returns:
        Dictionary of detected properties
    """
    
    properties = {}
    
    # Dimension analysis
    properties['dimensions'] = len(image_data.shape)
    properties['shape'] = image_data.shape
    
    # Time series detection
    if len(image_data.shape) >= 3:
        # Heuristic: if one dimension is much larger, likely time
        max_dim_idx = np.argmax(image_data.shape)
        max_dim_size = image_data.shape[max_dim_idx]
        other_dims = [s for i, s in enumerate(image_data.shape) if i != max_dim_idx]
        
        if max_dim_size > 2 * max(other_dims):
            properties['likely_time_series'] = True
            properties['time_axis'] = max_dim_idx
        else:
            properties['likely_time_series'] = False
    else:
        properties['likely_time_series'] = False
    
    # Multi-channel detection
    if len(image_data.shape) >= 3:
        # Check if last dimension could be channels (typically 1-4)
        if image_data.shape[-1] <= 4:
            properties['likely_multichannel'] = True
            properties['num_channels'] = image_data.shape[-1]
        else:
            properties['likely_multichannel'] = False
    else:
        properties['likely_multichannel'] = False
    
    # Data type analysis
    properties['dtype'] = str(image_data.dtype)
    properties['is_integer'] = np.issubdtype(image_data.dtype, np.integer)
    properties['is_float'] = np.issubdtype(image_data.dtype, np.floating)
    
    # Value range analysis
    properties['value_range'] = [float(np.min(image_data)), float(np.max(image_data))]
    properties['has_negative_values'] = bool(np.any(image_data < 0))
    
    return properties

def create_analysis_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary DataFrame from analysis results
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Summary DataFrame
    """
    
    summary_data = []
    
    def extract_numeric_values(data, prefix=""):
        """Recursively extract numeric values from nested dictionaries"""
        for key, value in data.items():
            full_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, (int, float)):
                summary_data.append({
                    'Parameter': full_key.replace('_', ' ').title(),
                    'Value': value,
                    'Type': type(value).__name__
                })
            elif isinstance(value, dict):
                extract_numeric_values(value, full_key)
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                if all(isinstance(x, (int, float)) for x in value):
                    summary_data.append({
                        'Parameter': f"{full_key} (mean)".replace('_', ' ').title(),
                        'Value': np.mean(value),
                        'Type': 'array_mean'
                    })
                    summary_data.append({
                        'Parameter': f"{full_key} (std)".replace('_', ' ').title(),
                        'Value': np.std(value),
                        'Type': 'array_std'
                    })
    
    # Extract values from results
    extract_numeric_values(results)
    
    # Create DataFrame
    if summary_data:
        df = pd.DataFrame(summary_data)
        return df
    else:
        return pd.DataFrame(columns=['Parameter', 'Value', 'Type'])

def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray], 
               default_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide with handling for division by zero
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)  
        default_value: Value to return when denominator is zero
        
    Returns:
        Division result with safe handling
    """
    
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, default_value, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask] if isinstance(numerator, np.ndarray) else numerator / denominator[mask]
        return result
    else:
        return numerator / denominator if denominator != 0 else default_value

def format_scientific_notation(value: float, precision: int = 2) -> str:
    """
    Format number in scientific notation with specified precision
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    
    if value == 0:
        return "0"
    elif abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def get_memory_usage(data: np.ndarray) -> str:
    """
    Get memory usage of numpy array in human-readable format
    
    Args:
        data: Numpy array
        
    Returns:
        Memory usage string
    """
    
    bytes_used = data.nbytes
    return format_file_size(bytes_used)

def validate_roi_coordinates(roi_coords: Tuple[int, int, int, int], 
                           image_shape: Tuple[int, ...]) -> bool:
    """
    Validate ROI coordinates against image dimensions
    
    Args:
        roi_coords: (x_start, y_start, x_end, y_end)
        image_shape: Image shape tuple
        
    Returns:
        True if ROI is valid
    """
    
    x_start, y_start, x_end, y_end = roi_coords
    height, width = image_shape[-2:]  # Assume last two dimensions are spatial
    
    # Check bounds
    if x_start < 0 or y_start < 0 or x_end >= width or y_end >= height:
        return False
    
    # Check that end > start
    if x_end <= x_start or y_end <= y_start:
        return False
    
    return True

def apply_roi_mask(image_data: np.ndarray, 
                  roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply ROI mask to image data
    
    Args:
        image_data: Input image array
        roi_coords: (x_start, y_start, x_end, y_end)
        
    Returns:
        Masked image data
    """
    
    x_start, y_start, x_end, y_end = roi_coords
    
    if len(image_data.shape) == 2:
        return image_data[y_start:y_end, x_start:x_end]
    elif len(image_data.shape) == 3:
        return image_data[:, y_start:y_end, x_start:x_end]
    elif len(image_data.shape) == 4:
        return image_data[:, y_start:y_end, x_start:x_end, :]
    else:
        raise ValueError(f"Unsupported image shape: {image_data.shape}")
