"""
Optical Flow Analysis Module
Implementation of optical flow elastography and force propagation analysis
Based on state-of-the-art methods for sub-pixel displacement tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Core libraries
try:
    from scipy import ndimage
    from scipy.optimize import minimize
    from skimage import registration
    from skimage.filters import gaussian
    from skimage.measure import regionprops
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    CORE_LIBS_AVAILABLE = True
except ImportError as e:
    CORE_LIBS_AVAILABLE = False
    warnings.warn(f"Core optical flow libraries not available: {e}")

# Advanced optical flow libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available - advanced optical flow limited")

class OpticalFlowAnalyzer:
    """Advanced optical flow analysis for elastography and force propagation"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which optical flow methods are available"""
        return {
            'Lucas-Kanade Optical Flow': CV2_AVAILABLE,
            'Dense Optical Flow (Farneback)': CV2_AVAILABLE,
            'Cross-Correlation Based': True,  # scipy implementation
            'Phase Correlation': True,  # scipy implementation
            'Digital Image Correlation (DIC)': True,  # custom implementation
            'Elastic Registration': True,  # scikit-image based
        }
    
    def get_available_methods(self) -> List[str]:
        """Return list of available optical flow methods"""
        return [method for method, available in self.available_methods.items() if available]
    
    def analyze_optical_flow(self, method: str, image_sequence: np.ndarray, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optical flow analysis to image sequence"""
        
        if method not in self.get_available_methods():
            return {
                'status': 'error',
                'message': f'Method {method} not available. Check dependencies.'
            }
        
        try:
            if method == 'Lucas-Kanade Optical Flow':
                return self._lucas_kanade_flow(image_sequence, parameters)
            elif method == 'Dense Optical Flow (Farneback)':
                return self._farneback_flow(image_sequence, parameters)
            elif method == 'Cross-Correlation Based':
                return self._cross_correlation_flow(image_sequence, parameters)
            elif method == 'Phase Correlation':
                return self._phase_correlation_flow(image_sequence, parameters)
            elif method == 'Digital Image Correlation (DIC)':
                return self._digital_image_correlation(image_sequence, parameters)
            elif method == 'Elastic Registration':
                return self._elastic_registration_flow(image_sequence, parameters)
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
    
    def _lucas_kanade_flow(self, image_sequence: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Lucas-Kanade sparse optical flow"""
        if not CV2_AVAILABLE:
            return {'status': 'error', 'message': 'OpenCV not available'}
        
        try:
            # Parameters for corner detection
            feature_params = {
                'maxCorners': parameters.get('max_corners', 100),
                'qualityLevel': parameters.get('quality_level', 0.3),
                'minDistance': parameters.get('min_distance', 7),
                'blockSize': parameters.get('block_size', 7)
            }
            
            # Parameters for Lucas-Kanade optical flow
            lk_params = {
                'winSize': (parameters.get('win_size', 15), parameters.get('win_size', 15)),
                'maxLevel': parameters.get('max_level', 2),
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }
            
            # Initialize arrays for tracking
            all_trajectories = []
            displacement_vectors = []
            
            # Convert to uint8 if needed
            if image_sequence.dtype != np.uint8:
                image_sequence = ((image_sequence - image_sequence.min()) / 
                                (image_sequence.max() - image_sequence.min()) * 255).astype(np.uint8)
            
            # Track features through sequence
            for t in range(len(image_sequence) - 1):
                frame1 = image_sequence[t]
                frame2 = image_sequence[t + 1]
                
                # Detect corners in first frame
                corners = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)
                
                if corners is not None:
                    # Calculate optical flow
                    new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                        frame1, frame2, corners, None, **lk_params
                    )
                    
                    # Select good points
                    good_new = new_corners[status == 1]
                    good_old = corners[status == 1]
                    
                    # Calculate displacement vectors
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        dx = new[0] - old[0]
                        dy = new[1] - old[1]
                        displacement_vectors.append({
                            'frame': t,
                            'x_start': old[0],
                            'y_start': old[1],
                            'dx': dx,
                            'dy': dy,
                            'magnitude': np.sqrt(dx**2 + dy**2),
                            'angle': np.arctan2(dy, dx)
                        })
                        
                        all_trajectories.append({
                            'frame': t,
                            'point_id': i,
                            'x': old[0],
                            'y': old[1],
                            'x_next': new[0],
                            'y_next': new[1]
                        })
            
            trajectories_df = pd.DataFrame(all_trajectories)
            displacements_df = pd.DataFrame(displacement_vectors)
            
            # Calculate statistics
            avg_displacement = displacements_df['magnitude'].mean() if not displacements_df.empty else 0
            max_displacement = displacements_df['magnitude'].max() if not displacements_df.empty else 0
            
            return {
                'status': 'success',
                'method': 'Lucas-Kanade Optical Flow',
                'trajectories': trajectories_df,
                'displacement_vectors': displacements_df,
                'avg_displacement': avg_displacement,
                'max_displacement': max_displacement,
                'num_tracked_points': len(displacements_df),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Lucas-Kanade flow failed: {str(e)}'}
    
    def _farneback_flow(self, image_sequence: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Dense optical flow using Farneback method"""
        if not CV2_AVAILABLE:
            return {'status': 'error', 'message': 'OpenCV not available'}
        
        try:
            # Convert to uint8 if needed
            if image_sequence.dtype != np.uint8:
                image_sequence = ((image_sequence - image_sequence.min()) / 
                                (image_sequence.max() - image_sequence.min()) * 255).astype(np.uint8)
            
            flow_fields = []
            displacement_maps = []
            
            # Calculate dense optical flow between consecutive frames
            for t in range(len(image_sequence) - 1):
                frame1 = image_sequence[t]
                frame2 = image_sequence[t + 1]
                
                # Calculate dense optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    frame1, frame2,
                    None,
                    pyr_scale=parameters.get('pyr_scale', 0.5),
                    levels=parameters.get('levels', 3),
                    winsize=parameters.get('winsize', 15),
                    iterations=parameters.get('iterations', 3),
                    poly_n=parameters.get('poly_n', 5),
                    poly_sigma=parameters.get('poly_sigma', 1.2),
                    flags=0
                )
                
                # Calculate magnitude and angle
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                angle = np.arctan2(flow[..., 1], flow[..., 0])
                
                flow_fields.append({
                    'frame': t,
                    'flow_x': flow[..., 0],
                    'flow_y': flow[..., 1],
                    'magnitude': magnitude,
                    'angle': angle
                })
                
                displacement_maps.append(magnitude)
            
            # Stack displacement maps
            displacement_stack = np.stack(displacement_maps, axis=0)
            
            # Calculate statistics
            avg_displacement = np.mean(displacement_stack)
            max_displacement = np.max(displacement_stack)
            
            return {
                'status': 'success',
                'method': 'Dense Optical Flow (Farneback)',
                'flow_fields': flow_fields,
                'displacement_maps': displacement_stack,
                'avg_displacement': avg_displacement,
                'max_displacement': max_displacement,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Farneback flow failed: {str(e)}'}
    
    def _cross_correlation_flow(self, image_sequence: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-correlation based displacement tracking"""
        try:
            window_size = parameters.get('window_size', 32)
            overlap = parameters.get('overlap', 0.5)
            search_area = parameters.get('search_area', 64)
            
            displacement_fields = []
            
            # Calculate displacements between consecutive frames
            for t in range(len(image_sequence) - 1):
                frame1 = image_sequence[t].astype(np.float64)
                frame2 = image_sequence[t + 1].astype(np.float64)
                
                # Calculate displacement field using cross-correlation
                displacement_field = self._calculate_cross_correlation_displacement(
                    frame1, frame2, window_size, overlap, search_area
                )
                
                displacement_fields.append(displacement_field)
            
            # Process results
            displacement_stack = np.stack([df['magnitude'] for df in displacement_fields], axis=0)
            
            return {
                'status': 'success',
                'method': 'Cross-Correlation Based',
                'displacement_fields': displacement_fields,
                'displacement_maps': displacement_stack,
                'avg_displacement': np.mean(displacement_stack),
                'max_displacement': np.max(displacement_stack),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Cross-correlation flow failed: {str(e)}'}
    
    def _calculate_cross_correlation_displacement(self, frame1: np.ndarray, frame2: np.ndarray,
                                                window_size: int, overlap: float, search_area: int) -> Dict[str, Any]:
        """Calculate displacement using normalized cross-correlation"""
        h, w = frame1.shape
        step = int(window_size * (1 - overlap))
        
        displacement_x = np.zeros((h // step, w // step))
        displacement_y = np.zeros((h // step, w // step))
        correlation_peak = np.zeros((h // step, w // step))
        
        for i, y in enumerate(range(0, h - window_size, step)):
            for j, x in enumerate(range(0, w - window_size, step)):
                # Extract template from frame1
                template = frame1[y:y+window_size, x:x+window_size]
                
                # Define search area in frame2
                y_start = max(0, y - search_area // 2)
                y_end = min(h, y + window_size + search_area // 2)
                x_start = max(0, x - search_area // 2)
                x_end = min(w, x + window_size + search_area // 2)
                
                search_region = frame2[y_start:y_end, x_start:x_end]
                
                # Calculate normalized cross-correlation
                try:
                    correlation = registration.phase_cross_correlation(
                        template, search_region, return_error=False
                    )
                    
                    dy, dx = correlation
                    
                    displacement_x[i, j] = dx
                    displacement_y[i, j] = dy
                    
                    # Simple correlation peak estimate
                    correlation_peak[i, j] = np.max(np.real(
                        np.fft.ifft2(np.fft.fft2(template) * np.conj(np.fft.fft2(search_region)))
                    ))
                    
                except Exception:
                    # Fallback to simple correlation
                    displacement_x[i, j] = 0
                    displacement_y[i, j] = 0
                    correlation_peak[i, j] = 0
        
        magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
        angle = np.arctan2(displacement_y, displacement_x)
        
        return {
            'displacement_x': displacement_x,
            'displacement_y': displacement_y,
            'magnitude': magnitude,
            'angle': angle,
            'correlation_peak': correlation_peak
        }
    
    def _phase_correlation_flow(self, image_sequence: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Phase correlation based displacement tracking"""
        try:
            displacement_fields = []
            
            for t in range(len(image_sequence) - 1):
                frame1 = image_sequence[t].astype(np.float64)
                frame2 = image_sequence[t + 1].astype(np.float64)
                
                # Calculate phase correlation
                try:
                    shift, error, diffphase = registration.phase_cross_correlation(
                        frame1, frame2, return_error=True
                    )
                except TypeError:
                    # Fallback for older scikit-image versions
                    shift = registration.phase_cross_correlation(frame1, frame2)
                    error = 0
                
                # Create displacement field
                h, w = frame1.shape
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                
                displacement_field = {
                    'frame': t,
                    'shift_y': shift[0],
                    'shift_x': shift[1],
                    'error': error,
                    'magnitude': np.sqrt(shift[0]**2 + shift[1]**2),
                    'angle': np.arctan2(shift[0], shift[1])
                }
                
                displacement_fields.append(displacement_field)
            
            # Calculate statistics
            magnitudes = [df['magnitude'] for df in displacement_fields]
            avg_displacement = np.mean(magnitudes)
            max_displacement = np.max(magnitudes)
            
            return {
                'status': 'success',
                'method': 'Phase Correlation',
                'displacement_fields': displacement_fields,
                'avg_displacement': avg_displacement,
                'max_displacement': max_displacement,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Phase correlation failed: {str(e)}'}
    
    def _digital_image_correlation(self, image_sequence: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Digital Image Correlation (DIC) analysis"""
        try:
            subset_size = parameters.get('subset_size', 32)
            step_size = parameters.get('step_size', 16)
            correlation_threshold = parameters.get('correlation_threshold', 0.7)
            
            dic_results = []
            
            for t in range(len(image_sequence) - 1):
                frame1 = image_sequence[t].astype(np.float64)
                frame2 = image_sequence[t + 1].astype(np.float64)
                
                # Perform DIC analysis
                dic_field = self._perform_dic_analysis(
                    frame1, frame2, subset_size, step_size, correlation_threshold
                )
                
                dic_field['frame'] = t
                dic_results.append(dic_field)
            
            # Calculate strain fields
            strain_results = self._calculate_strain_fields(dic_results)
            
            return {
                'status': 'success',
                'method': 'Digital Image Correlation (DIC)',
                'dic_results': dic_results,
                'strain_results': strain_results,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'DIC analysis failed: {str(e)}'}
    
    def _perform_dic_analysis(self, frame1: np.ndarray, frame2: np.ndarray,
                            subset_size: int, step_size: int, threshold: float) -> Dict[str, Any]:
        """Perform DIC analysis between two frames"""
        h, w = frame1.shape
        
        # Initialize arrays
        displacement_x = []
        displacement_y = []
        correlation_coeff = []
        x_coords = []
        y_coords = []
        
        for y in range(subset_size//2, h - subset_size//2, step_size):
            for x in range(subset_size//2, w - subset_size//2, step_size):
                # Extract subset from reference frame
                subset = frame1[y-subset_size//2:y+subset_size//2+1,
                               x-subset_size//2:x+subset_size//2+1]
                
                # Search for best match in deformed frame
                best_corr = -1
                best_dx = 0
                best_dy = 0
                
                search_range = subset_size // 2
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        y_new = y + dy
                        x_new = x + dx
                        
                        if (y_new - subset_size//2 >= 0 and 
                            y_new + subset_size//2 < h and
                            x_new - subset_size//2 >= 0 and 
                            x_new + subset_size//2 < w):
                            
                            deformed_subset = frame2[y_new-subset_size//2:y_new+subset_size//2+1,
                                                   x_new-subset_size//2:x_new+subset_size//2+1]
                            
                            # Calculate normalized cross-correlation
                            corr = np.corrcoef(subset.flatten(), deformed_subset.flatten())[0, 1]
                            
                            if not np.isnan(corr) and corr > best_corr:
                                best_corr = corr
                                best_dx = dx
                                best_dy = dy
                
                # Store results if correlation is above threshold
                if best_corr > threshold:
                    x_coords.append(x)
                    y_coords.append(y)
                    displacement_x.append(best_dx)
                    displacement_y.append(best_dy)
                    correlation_coeff.append(best_corr)
        
        return {
            'x_coords': np.array(x_coords),
            'y_coords': np.array(y_coords),
            'displacement_x': np.array(displacement_x),
            'displacement_y': np.array(displacement_y),
            'correlation_coeff': np.array(correlation_coeff),
            'magnitude': np.sqrt(np.array(displacement_x)**2 + np.array(displacement_y)**2)
        }
    
    def _calculate_strain_fields(self, dic_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strain fields from DIC displacement results"""
        strain_fields = []
        
        for dic_field in dic_results:
            if len(dic_field['x_coords']) > 4:  # Need sufficient points for strain calculation
                
                # Calculate strain using finite differences
                x_coords = dic_field['x_coords']
                y_coords = dic_field['y_coords']
                u = dic_field['displacement_x']  # x-displacement
                v = dic_field['displacement_y']  # y-displacement
                
                # Simple strain calculation (would need proper gridding for accuracy)
                try:
                    # Normal strains
                    epsilon_xx = np.gradient(u, x_coords, axis=0)
                    epsilon_yy = np.gradient(v, y_coords, axis=0)
                    
                    # Shear strain
                    gamma_xy = 0.5 * (np.gradient(u, y_coords, axis=0) + 
                                     np.gradient(v, x_coords, axis=0))
                    
                    strain_fields.append({
                        'frame': dic_field['frame'],
                        'epsilon_xx': np.mean(epsilon_xx) if len(epsilon_xx) > 0 else 0,
                        'epsilon_yy': np.mean(epsilon_yy) if len(epsilon_yy) > 0 else 0,
                        'gamma_xy': np.mean(gamma_xy) if len(gamma_xy) > 0 else 0,
                        'principal_strain_1': np.mean(epsilon_xx + epsilon_yy) if len(epsilon_xx) > 0 else 0,
                        'von_mises_strain': np.sqrt(0.5 * ((epsilon_xx - epsilon_yy)**2 + 
                                                          epsilon_xx**2 + epsilon_yy**2 + 
                                                          6 * gamma_xy**2)).mean() if len(epsilon_xx) > 0 else 0
                    })
                except Exception:
                    strain_fields.append({
                        'frame': dic_field['frame'],
                        'epsilon_xx': 0,
                        'epsilon_yy': 0,
                        'gamma_xy': 0,
                        'principal_strain_1': 0,
                        'von_mises_strain': 0
                    })
        
        return {
            'strain_fields': strain_fields,
            'avg_von_mises_strain': np.mean([sf['von_mises_strain'] for sf in strain_fields])
        }
    
    def _elastic_registration_flow(self, image_sequence: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Elastic registration based optical flow with aligned image output"""
        try:
            from skimage import transform
            from skimage.registration import optical_flow_tvl1, optical_flow_ilk
            
            registration_results = []
            aligned_images = [image_sequence[0]]  # First frame as reference
            
            reference_frame = image_sequence[0].astype(np.float32)
            
            for t in range(1, len(image_sequence)):
                current_frame = image_sequence[t].astype(np.float32)
                
                # Perform registration using phase correlation
                try:
                    # Use phase correlation for better alignment
                    shift, error, diffphase = registration.phase_cross_correlation(
                        reference_frame, current_frame, 
                        upsample_factor=parameters.get('upsample_factor', 10)
                    )
                    
                    # Apply transformation to align the image
                    tform = transform.SimilarityTransform(translation=shift)
                    aligned_frame = transform.warp(current_frame, tform.inverse, 
                                                 output_shape=reference_frame.shape,
                                                 preserve_range=True)
                    
                    # Calculate displacement and error metrics
                    displacement = np.sqrt(np.sum(shift**2))
                    registration_error = np.mean((aligned_frame - reference_frame)**2)
                    
                    registration_results.append({
                        'frame': t,
                        'shift': shift,
                        'transformation': tform,
                        'displacement': displacement,
                        'registration_error': registration_error,
                        'cross_correlation_error': error
                    })
                    
                    aligned_images.append(aligned_frame.astype(image_sequence.dtype))
                    
                except Exception as reg_error:
                    # Fallback: use original frame if registration fails
                    aligned_images.append(current_frame.astype(image_sequence.dtype))
                    registration_results.append({
                        'frame': t,
                        'shift': [0, 0],
                        'transformation': None,
                        'displacement': 0,
                        'registration_error': float('inf'),
                        'error': str(reg_error)
                    })
            
            # Convert aligned images to numpy array
            aligned_sequence = np.array(aligned_images)
            
            # Calculate summary statistics
            avg_displacement = np.mean([r['displacement'] for r in registration_results if r['displacement'] != float('inf')])
            total_drift = np.sum([r['displacement'] for r in registration_results if r['displacement'] != float('inf')])
            
            return {
                'status': 'success',
                'method': 'Elastic Registration',
                'registration_results': registration_results,
                'aligned_images': aligned_sequence,
                'avg_displacement': avg_displacement,
                'total_drift': total_drift,
                'alignment_quality': 1.0 / (1.0 + avg_displacement),
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Elastic registration failed: {str(e)}'}

def get_optical_flow_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for optical flow methods"""
    
    parameters = {
        'Lucas-Kanade Optical Flow': {
            'max_corners': 100,
            'quality_level': 0.3,
            'min_distance': 7,
            'block_size': 7,
            'win_size': 15,
            'max_level': 2
        },
        'Dense Optical Flow (Farneback)': {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2
        },
        'Cross-Correlation Based': {
            'window_size': 32,
            'overlap': 0.5,
            'search_area': 64
        },
        'Phase Correlation': {
            'upsample_factor': 100
        },
        'Digital Image Correlation (DIC)': {
            'subset_size': 32,
            'step_size': 16,
            'correlation_threshold': 0.7
        },
        'Elastic Registration': {
            'max_iterations': 100,
            'tolerance': 1e-6
        }
    }
    
    return parameters.get(method, {})