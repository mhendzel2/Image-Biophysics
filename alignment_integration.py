"""
Universal Nuclear Alignment Integration for All Biophysical Analyses
Provides consistent alignment preprocessing for RICS, FCS, iMSD, FRAP, FLIM, SPT, and N&B
"""

import numpy as np
from typing import Dict, Any, Tuple
import warnings

def apply_nuclear_alignment(image_data: np.ndarray, 
                          parameters: Dict[str, Any],
                          analysis_name: str = "Unknown") -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Universal nuclear alignment preprocessing for all biophysical analyses
    
    Args:
        image_data: Input image sequence (T, Y, X) or (T, Y, X, C)
        parameters: Analysis parameters including alignment settings
        analysis_name: Name of the calling analysis for logging
        
    Returns:
        Tuple of (aligned_data, alignment_applied, quality_metrics)
    """
    
    # Check if alignment is enabled
    enable_alignment = parameters.get('nuclear_alignment', True)
    
    if not enable_alignment:
        return image_data, False, {}
    
    # Validate input dimensions
    if image_data.ndim < 3:
        warnings.warn(f"{analysis_name}: Nuclear alignment requires time series data")
        return image_data, False, {}
    
    try:
        # Handle multi-channel data by using first channel for alignment
        if image_data.ndim == 4:
            # Assume (T, Y, X, C) format
            alignment_channel = parameters.get('alignment_channel', 0)
            reference_data = image_data[:, :, :, alignment_channel]
        else:
            # Assume (T, Y, X) format
            reference_data = image_data
        
        # Perform alignment using phase correlation (most robust for nuclear data)
        from nuclear_alignment import align_nuclear_sequence
        
        alignment_method = parameters.get('alignment_method', 'phase_correlation')
        reference_frame = parameters.get('reference_frame', 0)
        
        alignment_result = align_nuclear_sequence(
            reference_data,
            alignment_method=alignment_method,
            reference_frame=reference_frame
        )
        
        if alignment_result.get('status') == 'success':
            aligned_reference = alignment_result['aligned_sequence']
            
            # Apply same transformations to all channels if multi-channel
            if image_data.ndim == 4:
                aligned_data = np.zeros_like(image_data)
                shifts = alignment_result.get('shifts', [])
                
                # Apply shifts to all channels
                for c in range(image_data.shape[3]):
                    aligned_data[:, :, :, c] = _apply_shifts_to_channel(
                        image_data[:, :, :, c], shifts
                    )
            else:
                aligned_data = aligned_reference
            
            quality_metrics = alignment_result.get('quality_metrics', {})
            quality_metrics['analysis_type'] = analysis_name
            
            return aligned_data, True, quality_metrics
        else:
            warnings.warn(f"{analysis_name}: Nuclear alignment failed - {alignment_result.get('message', 'Unknown error')}")
            return image_data, False, {}
            
    except ImportError:
        warnings.warn(f"{analysis_name}: Nuclear alignment module not available")
        return image_data, False, {}
    except Exception as e:
        warnings.warn(f"{analysis_name}: Nuclear alignment error - {str(e)}")
        return image_data, False, {}

def _apply_shifts_to_channel(channel_data: np.ndarray, shifts: list) -> np.ndarray:
    """Apply pre-calculated shifts to a single channel"""
    
    from scipy import ndimage
    
    aligned_channel = np.zeros_like(channel_data)
    
    for t, shift in enumerate(shifts):
        if t < channel_data.shape[0]:
            if len(shift) >= 2:
                aligned_channel[t] = ndimage.shift(
                    channel_data[t], shift[:2], order=1, mode='nearest'
                )
            else:
                aligned_channel[t] = channel_data[t]
    
    return aligned_channel

def get_alignment_parameters(analysis_type: str) -> Dict[str, Any]:
    """Get default alignment parameters optimized for specific analysis types"""
    
    base_params = {
        'nuclear_alignment': True,
        'alignment_method': 'phase_correlation',
        'reference_frame': 0,
        'alignment_channel': 0
    }
    
    # Analysis-specific optimizations
    if analysis_type.upper() in ['RICS', 'FCS']:
        # High-frequency analyses benefit from stable alignment
        base_params.update({
            'alignment_method': 'phase_correlation',  # Most stable for correlation analyses
        })
    elif analysis_type.upper() in ['SPT', 'iMSD']:
        # Particle tracking needs feature preservation
        base_params.update({
            'alignment_method': 'feature_based',  # Preserves particle features
        })
    elif analysis_type.upper() in ['FRAP', 'FLIM']:
        # Recovery analyses need consistent regions
        base_params.update({
            'alignment_method': 'optical_flow',  # Good for intensity-based analyses
        })
    elif analysis_type.upper() == 'NB':
        # Number & Brightness is very sensitive to motion
        base_params.update({
            'alignment_method': 'phase_correlation',  # Maximum stability
        })
    
    return base_params

def validate_alignment_quality(quality_metrics: Dict[str, Any], 
                             analysis_type: str) -> Dict[str, Any]:
    """Validate alignment quality and provide recommendations"""
    
    if not quality_metrics:
        return {
            'quality_assessment': 'No alignment applied',
            'recommendation': 'Enable nuclear alignment for better results',
            'reliability_score': 0.3
        }
    
    mean_shift = quality_metrics.get('mean_shift_pixels', [0, 0])
    max_shift = quality_metrics.get('max_shift_pixels', [0, 0])
    stability_score = quality_metrics.get('stability_score', 0)
    
    # Calculate overall shift magnitude
    mean_shift_magnitude = np.sqrt(np.sum(np.array(mean_shift)**2))
    max_shift_magnitude = np.sqrt(np.sum(np.array(max_shift)**2))
    
    # Quality thresholds based on analysis type
    if analysis_type.upper() in ['RICS', 'FCS', 'NB']:
        # High-precision analyses need minimal motion
        good_threshold = 1.0
        acceptable_threshold = 3.0
    elif analysis_type.upper() in ['SPT', 'iMSD']:
        # Particle tracking can tolerate some motion
        good_threshold = 2.0
        acceptable_threshold = 5.0
    else:
        # General analyses
        good_threshold = 1.5
        acceptable_threshold = 4.0
    
    # Assess quality
    if mean_shift_magnitude <= good_threshold and max_shift_magnitude <= good_threshold * 2:
        quality_assessment = 'Excellent alignment'
        reliability_score = 0.9 + 0.1 * stability_score
        recommendation = 'Alignment quality is excellent for quantitative analysis'
    elif mean_shift_magnitude <= acceptable_threshold and max_shift_magnitude <= acceptable_threshold * 2:
        quality_assessment = 'Good alignment'
        reliability_score = 0.7 + 0.2 * stability_score
        recommendation = 'Alignment quality is good for most analyses'
    else:
        quality_assessment = 'Poor alignment'
        reliability_score = 0.3 + 0.2 * stability_score
        recommendation = 'Consider alternative alignment method or manual correction'
    
    return {
        'quality_assessment': quality_assessment,
        'recommendation': recommendation,
        'reliability_score': min(1.0, reliability_score),
        'mean_drift_pixels': mean_shift_magnitude,
        'max_drift_pixels': max_shift_magnitude,
        'frame_stability': stability_score
    }

def create_alignment_summary(alignment_applied: bool, 
                           quality_metrics: Dict[str, Any],
                           analysis_type: str) -> Dict[str, Any]:
    """Create comprehensive alignment summary for analysis results"""
    
    if not alignment_applied:
        return {
            'nuclear_alignment_applied': False,
            'alignment_status': 'Disabled or failed',
            'quality_assessment': 'No alignment',
            'recommendation': f'Enable nuclear alignment for {analysis_type} analysis',
            'reliability_impact': 'Moderate - motion artifacts may affect results'
        }
    
    quality_validation = validate_alignment_quality(quality_metrics, analysis_type)
    
    return {
        'nuclear_alignment_applied': True,
        'alignment_method': quality_metrics.get('alignment_method', 'Unknown'),
        'reference_frame': quality_metrics.get('reference_frame', 0),
        'quality_assessment': quality_validation['quality_assessment'],
        'recommendation': quality_validation['recommendation'],
        'reliability_score': quality_validation['reliability_score'],
        'drift_statistics': {
            'mean_drift_pixels': quality_validation.get('mean_drift_pixels', 0),
            'max_drift_pixels': quality_validation.get('max_drift_pixels', 0),
            'frame_stability': quality_validation.get('frame_stability', 0)
        },
        'reliability_impact': _get_reliability_impact(quality_validation['reliability_score'])
    }

def _get_reliability_impact(reliability_score: float) -> str:
    """Convert reliability score to descriptive impact assessment"""
    
    if reliability_score >= 0.8:
        return 'High - excellent data quality for quantitative analysis'
    elif reliability_score >= 0.6:
        return 'Good - suitable for most biophysical measurements'
    elif reliability_score >= 0.4:
        return 'Moderate - results should be interpreted with caution'
    else:
        return 'Low - significant motion artifacts likely affect results'