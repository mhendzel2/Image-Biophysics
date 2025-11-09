"""
Unified Analysis Module

This module provides a centralized manager for all available analysis methods,
including Segmented FCS, RICS, and Advanced AI-driven techniques.
"""

import numpy as np
from typing import Dict, Any, List

# Import project-specific analysis modules
import fcs_analysis
import advanced_analysis

class AnalysisManager:
    """
    Central manager to access and run all analysis functionalities.
    """
    def __init__(self):
        """Initializes the analysis manager."""
        self.advanced_manager = advanced_analysis.AdvancedAnalysisManager()
        self.available_analyses = self._get_all_available_analyses()

    def _get_all_available_analyses(self) -> List[str]:
        """Gathers all available analysis methods."""
        fcs_methods = [
            "Segmented FCS (Pixel-wise)",
            "Segmented FCS (Line-wise)",
            "RICS Analysis",
        ]
        advanced_methods = self.advanced_manager.get_available_methods()
        return fcs_methods + advanced_methods

    def get_available_analyses(self) -> List[str]:
        """Returns a list of all available analysis methods."""
        return self.available_analyses

    def run_analysis(self, method: str, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the selected analysis method with the given data and parameters.
        """
        try:
            if method == "Segmented FCS (Pixel-wise)":
                fcs_params = {
                    'segmentation_type': 'x',
                    'segment_length': parameters.get('segment_length', 128),
                    'pixel_time': parameters.get('pixel_time', 3.05e-6),
                    'line_time': parameters.get('line_time', 0.56e-3),
                    'pixel_size': parameters.get('pixel_size', 0.05),
                    'model_type': parameters.get('model_type', '2d'),
                    'max_lag_fraction': parameters.get('max_lag_fraction', 0.25),
                }
                results = fcs_analysis.segmented_fcs_analysis(image_data, **fcs_params)
                if not results:
                    return {'status': 'error', 'message': 'FCS analysis produced no results.'}
                stats = fcs_analysis.analyze_segment_statistics(results)
                return {'status': 'success', 'results': results, 'statistics': stats}

            elif method == "Segmented FCS (Line-wise)":
                fcs_params = {
                    'segmentation_type': 'y',
                    'segment_length': parameters.get('segment_length', 128),
                    'pixel_time': parameters.get('pixel_time', 3.05e-6),
                    'line_time': parameters.get('line_time', 0.56e-3),
                    'pixel_size': parameters.get('pixel_size', 0.05),
                    'model_type': parameters.get('model_type', '2d'),
                    'max_lag_fraction': parameters.get('max_lag_fraction', 0.25),
                }
                results = fcs_analysis.segmented_fcs_analysis(image_data, **fcs_params)
                if not results:
                    return {'status': 'error', 'message': 'FCS analysis produced no results.'}
                stats = fcs_analysis.analyze_segment_statistics(results)
                return {'status': 'success', 'results': results, 'statistics': stats}
            
            elif method == "RICS Analysis":
                if image_data.ndim != 3:
                    return {'status': 'error', 'message': 'RICS requires a 3D image stack (time, height, width).'}
                
                # The 'tau_max' parameter from the original function signature is not used in the implementation.
                autocorr_map = self._calculate_spatial_autocorrelation(image_data)
                return {
                    'status': 'success',
                    'method': 'RICS Analysis',
                    'autocorrelation_map': autocorr_map,
                    'parameters_used': parameters
                }
            
            elif method in self.advanced_manager.get_available_methods():
                return self.advanced_manager.apply_advanced_method(method, image_data, parameters)

            else:
                return {'status': 'error', 'message': f'Unknown analysis method: {method}'}
        
        except Exception as e:
            # Provide a more detailed error message
            import traceback
            tb_str = traceback.format_exc()
            return {'status': 'error', 'message': f"Error during analysis with '{method}': {str(e)}\n{tb_str}"}

    def get_default_parameters(self, method: str) -> Dict[str, Any]:
        """Gets default parameters for a given analysis method."""
        if method.startswith("Segmented FCS"):
            return {
                'segment_length': 128,
                'pixel_time': 3.05e-6,
                'line_time': 0.56e-3,
                'pixel_size': 0.05,
                'model_type': '2d',  # Options: '2d', '3d', 'anomalous'
                'max_lag_fraction': 0.25
            }
        elif method == "RICS Analysis":
            return {} # No parameters needed for the current implementation
        
        elif method in self.advanced_manager.get_available_methods():
            return advanced_analysis.get_advanced_parameters(method)
            
        else:
            return {}

    def _calculate_spatial_autocorrelation(self, image_stack: np.ndarray) -> np.ndarray:
        """
        Calculate RICS spatial autocorrelation using an FFT-based method.
        This function was recovered from the previously corrupted file.
        """
        if image_stack.ndim != 3 or image_stack.shape[0] <= 1:
            raise ValueError("RICS requires a 3D image stack with multiple time frames.")

        t_frames, height, width = image_stack.shape
        
        # Calculate intensity fluctuations: delta_I = I - <I>_t
        mean_intensity = np.mean(image_stack, axis=0, dtype=np.float64)
        fluctuations = image_stack - mean_intensity[np.newaxis, :, :]
        
        # Calculate spatial autocorrelation for each frame and average
        autocorr_sum = np.zeros((height, width), dtype=np.float64)
        
        for t in range(t_frames):
            delta_I = fluctuations[t]
            
            # FFT-based 2D autocorrelation
            fft_image = np.fft.fft2(delta_I)
            autocorr_2d = np.fft.ifft2(fft_image * np.conj(fft_image)).real
            
            autocorr_sum += np.fft.fftshift(autocorr_2d)
        
        # Average over time
        avg_autocorr = autocorr_sum / t_frames
        
        # Normalize by the zero-lag value (G(0,0))
        center_y, center_x = height // 2, width // 2
        zero_lag_value = avg_autocorr[center_y, center_x]
        
        if zero_lag_value > 1e-9: # Avoid division by zero
            return avg_autocorr / zero_lag_value
        else:
            return avg_autocorr
