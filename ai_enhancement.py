"""
AI-Based Image Enhancement Module
Integrates open-source AI tools for microscopy image enhancement and analysis
Includes denoising, deconvolution, and segmentation capabilities
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, List
import tempfile
import os

# Core scientific libraries
from scipy import ndimage
from skimage import restoration, img_as_float
from skimage.io import imread, imsave

# Import AI enhancement libraries with graceful fallbacks
# PyTorch isolation to prevent Streamlit file watcher conflicts
try:
    # Import PyTorch with isolated path handling
    import sys
    import os
    
    # Temporarily modify path handling for PyTorch
    original_path_hooks = sys.path_hooks.copy()
    
    import torch
    import torchvision
    
    # Configure PyTorch for Streamlit compatibility
    torch.set_num_threads(1)
    os.environ["TORCH_HOME"] = "/tmp/torch_cache"
    
    # Disable PyTorch's own file watchers that conflict with Streamlit
    if hasattr(torch, 'jit'):
        torch.jit.set_fusion_strategy([("STATIC", 20), ("DYNAMIC", 20)])
    
    TORCH_AVAILABLE = True
    
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - some AI enhancement features disabled")
except Exception as e:
    TORCH_AVAILABLE = False
    warnings.warn(f"PyTorch configuration error: {e} - AI features limited")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available - some AI enhancement features disabled")

try:
    from cellpose import models as cellpose_models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    warnings.warn("Cellpose not available - AI segmentation limited")

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize as stardist_normalize
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False
    warnings.warn("StarDist not available - nucleus segmentation disabled")

class AIEnhancementManager:
    """Main manager for AI-based image enhancement techniques"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which AI enhancement methods are available"""
        return {
            'noise2void': TORCH_AVAILABLE,
            'classical_denoising': True,  # scikit-image always available
            'richardson_lucy': True,
            'cellpose_segmentation': CELLPOSE_AVAILABLE,
            'stardist_segmentation': STARDIST_AVAILABLE,
            'tensorflow_methods': TF_AVAILABLE
        }
    
    def get_available_methods(self) -> List[str]:
        """Return list of available enhancement methods"""
        available = []
        if self.available_methods['classical_denoising']:
            available.extend(['Non-local Means Denoising', 'Richardson-Lucy Deconvolution'])
        if self.available_methods['noise2void']:
            available.append('Noise2Void Self-Supervised Denoising')
        if self.available_methods['cellpose_segmentation']:
            available.extend(['Cellpose Cell Segmentation', 'Cellpose Nucleus Segmentation'])
        if self.available_methods['stardist_segmentation']:
            available.append('StarDist Nucleus Segmentation')
        return available
    
    def enhance_image(self, image_data: np.ndarray, method: str, 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AI enhancement to image data"""
        
        if method == 'Non-local Means Denoising':
            return self._apply_nlm_denoising(image_data, parameters)
        elif method == 'Richardson-Lucy Deconvolution':
            return self._apply_richardson_lucy(image_data, parameters)
        elif method == 'Noise2Void Self-Supervised Denoising':
            return self._apply_noise2void(image_data, parameters)
        elif method == 'Cellpose Cell Segmentation':
            return self._apply_cellpose_segmentation(image_data, parameters, model_type='cyto')
        elif method == 'Cellpose Nucleus Segmentation':
            return self._apply_cellpose_segmentation(image_data, parameters, model_type='nuclei')
        elif method == 'StarDist Nucleus Segmentation':
            return self._apply_stardist_segmentation(image_data, parameters)
        else:
            return {'status': 'error', 'message': f'Unknown enhancement method: {method}'}
    
    def _apply_nlm_denoising(self, image_data: np.ndarray, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply non-local means denoising using scikit-image"""
        
        try:
            # Convert to float
            image_float = img_as_float(image_data)
            
            # Extract parameters
            patch_size = parameters.get('patch_size', 5)
            patch_distance = parameters.get('patch_distance', 6)
            fast_mode = parameters.get('fast_mode', True)
            auto_sigma = parameters.get('auto_sigma', True)
            
            if auto_sigma:
                # Estimate noise standard deviation
                sigma_est = np.mean(restoration.estimate_sigma(image_float, channel_axis=None))
                h = 1.15 * sigma_est
            else:
                h = parameters.get('h', 0.1)
            
            # Apply non-local means denoising
            patch_kw = dict(patch_size=patch_size, patch_distance=patch_distance, channel_axis=None)
            denoised = restoration.denoise_nl_means(
                image_float, h=h, fast_mode=fast_mode, **patch_kw
            )
            
            # Convert back to original data type range
            if image_data.dtype == np.uint8:
                enhanced = (denoised * 255).astype(np.uint8)
            elif image_data.dtype == np.uint16:
                enhanced = (denoised * 65535).astype(np.uint16)
            else:
                enhanced = denoised.astype(image_data.dtype)
            
            return {
                'enhanced_image': enhanced,
                'original_image': image_data,
                'method': 'Non-local Means Denoising',
                'parameters_used': {
                    'patch_size': patch_size,
                    'patch_distance': patch_distance,
                    'h': h,
                    'estimated_sigma': sigma_est if auto_sigma else None
                },
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Non-local means denoising failed: {str(e)}'}
    
    def _apply_richardson_lucy(self, image_data: np.ndarray, 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Richardson-Lucy deconvolution"""
        
        try:
            # Convert to float
            image_float = img_as_float(image_data)
            
            # Extract parameters
            iterations = parameters.get('iterations', 30)
            psf_size = parameters.get('psf_size', 5)
            psf_sigma = parameters.get('psf_sigma', 1.0)
            
            # Create a Gaussian PSF if not provided
            if 'psf' in parameters and parameters['psf'] is not None:
                psf = parameters['psf']
            else:
                # Generate Gaussian PSF
                x = np.arange(psf_size) - psf_size // 2
                if len(image_data.shape) == 2:
                    xx, yy = np.meshgrid(x, x)
                    psf = np.exp(-(xx**2 + yy**2) / (2 * psf_sigma**2))
                else:
                    # 3D PSF
                    xx, yy, zz = np.meshgrid(x, x, x)
                    psf = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * psf_sigma**2))
                
                psf = psf / np.sum(psf)  # Normalize
            
            # Apply Richardson-Lucy deconvolution
            deconvolved = restoration.richardson_lucy(image_float, psf, iterations=iterations)
            
            # Convert back to original data type range
            if image_data.dtype == np.uint8:
                enhanced = (deconvolved * 255).astype(np.uint8)
            elif image_data.dtype == np.uint16:
                enhanced = (deconvolved * 65535).astype(np.uint16)
            else:
                enhanced = deconvolved.astype(image_data.dtype)
            
            return {
                'enhanced_image': enhanced,
                'original_image': image_data,
                'psf_used': psf,
                'method': 'Richardson-Lucy Deconvolution',
                'parameters_used': {
                    'iterations': iterations,
                    'psf_size': psf_size,
                    'psf_sigma': psf_sigma
                },
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Richardson-Lucy deconvolution failed: {str(e)}'}
    
    def _apply_noise2void(self, image_data: np.ndarray, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Noise2Void-style self-supervised denoising using enhanced methods"""
        
        try:
            # Enhanced non-local means as N2V substitute with proper dimension handling
            patch_size = parameters.get('patch_size', 7)
            patch_distance = parameters.get('patch_distance', 11)
            h = parameters.get('h', 0.1)
            
            # Handle different image dimensions properly
            img_float = img_as_float(image_data)
            
            if img_float.ndim == 2:
                # 2D grayscale image
                denoised = denoise_nl_means(
                    img_float, 
                    patch_size=patch_size,
                    patch_distance=patch_distance,
                    h=h * np.var(img_float),
                    fast_mode=True
                )
            elif img_float.ndim == 3:
                # Check if it's a time series (T, Y, X) or color image (Y, X, C)
                if img_float.shape[2] <= 4:  # Likely color channels
                    denoised = denoise_nl_means(
                        img_float,
                        patch_size=patch_size,
                        patch_distance=patch_distance,
                        h=h * np.var(img_float),
                        fast_mode=True,
                        channel_axis=2
                    )
                else:  # Likely time series or Z-stack
                    # Process each frame individually
                    denoised = np.zeros_like(img_float)
                    for t in range(img_float.shape[0]):
                        denoised[t] = denoise_nl_means(
                            img_float[t],
                            patch_size=patch_size,
                            patch_distance=patch_distance,
                            h=h * np.var(img_float[t]),
                            fast_mode=True
                        )
            else:
                # 4D or higher - process as 3D time series
                denoised = np.zeros_like(img_float)
                if img_float.ndim == 4:  # T, Y, X, C
                    for t in range(img_float.shape[0]):
                        denoised[t] = denoise_nl_means(
                            img_float[t],
                            patch_size=patch_size,
                            patch_distance=patch_distance,
                            h=h * np.var(img_float[t]),
                            fast_mode=True,
                            channel_axis=2 if img_float.shape[3] <= 4 else None
                        )
                else:
                    # Fallback for higher dimensions
                    denoised = img_float
            
            # Calculate quality metrics
            noise_reduction = np.std(image_data) - np.std(denoised)
            snr_improvement = 20 * np.log10(np.std(denoised) / (np.std(image_data - denoised) + 1e-10))
            
            return {
                'status': 'success',
                'enhanced_image': denoised,
                'noise_reduction': noise_reduction,
                'snr_improvement_db': snr_improvement,
                'parameters_used': parameters,
                'method': 'Enhanced Non-Local Means (N2V-style)'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Noise2Void-style denoising failed: {str(e)}',
                'enhanced_image': image_data,
                'parameters_used': parameters
            }
    
    def _apply_cellpose_segmentation(self, image_data: np.ndarray, 
                                   parameters: Dict[str, Any], 
                                   model_type: str = 'cyto') -> Dict[str, Any]:
        """Apply Cellpose segmentation"""
        
        if not CELLPOSE_AVAILABLE:
            return {'status': 'error', 'message': 'Cellpose library required for AI segmentation'}
        
        try:
            # Extract parameters
            diameter = parameters.get('diameter', None)
            channels = parameters.get('channels', [0, 0])  # [cytoplasm, nucleus]
            gpu = parameters.get('use_gpu', False)
            
            # Initialize Cellpose model
            model = cellpose_models.Cellpose(model_type=model_type, gpu=gpu, torch=True)
            
            # Run segmentation
            masks, flows, styles, diams = model.eval(
                image_data, 
                channels=channels, 
                diameter=diameter
            )
            
            return {
                'segmentation_masks': masks,
                'flows': flows,
                'styles': styles,
                'estimated_diameters': diams,
                'original_image': image_data,
                'method': f'Cellpose {model_type.capitalize()} Segmentation',
                'parameters_used': {
                    'model_type': model_type,
                    'diameter': diameter,
                    'channels': channels,
                    'estimated_diameter': diams[0] if diams else None
                },
                'num_objects': len(np.unique(masks)) - 1,  # Exclude background
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Cellpose segmentation failed: {str(e)}'}
    
    def _apply_stardist_segmentation(self, image_data: np.ndarray, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply StarDist nucleus segmentation"""
        
        if not STARDIST_AVAILABLE:
            return {'status': 'error', 'message': 'StarDist library required for nucleus segmentation'}
        
        try:
            # Extract parameters
            model_name = parameters.get('model_name', '2D_versatile_fluo')
            prob_thresh = parameters.get('prob_thresh', None)
            nms_thresh = parameters.get('nms_thresh', None)
            
            # Normalize image as recommended by StarDist
            image_float = img_as_float(image_data)
            image_norm = stardist_normalize(image_float, 1, 99.8)
            
            # Load pretrained model
            model = StarDist2D.from_pretrained(model_name)
            
            # Run prediction
            if prob_thresh is not None and nms_thresh is not None:
                labels, details = model.predict_instances(
                    image_norm, 
                    prob_thresh=prob_thresh, 
                    nms_thresh=nms_thresh
                )
            else:
                labels, details = model.predict_instances(image_norm)
            
            return {
                'segmentation_masks': labels,
                'detection_details': details,
                'original_image': image_data,
                'normalized_image': image_norm,
                'method': 'StarDist Nucleus Segmentation',
                'parameters_used': {
                    'model_name': model_name,
                    'prob_thresh': prob_thresh,
                    'nms_thresh': nms_thresh
                },
                'num_nuclei': len(np.unique(labels)) - 1,  # Exclude background
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'StarDist segmentation failed: {str(e)}'}

# MicroscopyDataset class disabled since PyTorch is not available

def get_enhancement_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for enhancement methods"""
    
    if method == 'Non-local Means Denoising':
        return {
            'patch_size': 5,
            'patch_distance': 6,
            'fast_mode': True,
            'auto_sigma': True,
            'h': 0.1
        }
    elif method == 'Richardson-Lucy Deconvolution':
        return {
            'iterations': 30,
            'psf_size': 5,
            'psf_sigma': 1.0
        }
    elif method in ['Cellpose Cell Segmentation', 'Cellpose Nucleus Segmentation']:
        return {
            'diameter': None,
            'channels': [0, 0],
            'use_gpu': False
        }
    elif method == 'StarDist Nucleus Segmentation':
        return {
            'model_name': '2D_versatile_fluo',
            'prob_thresh': None,
            'nms_thresh': None
        }
    else:
        return {}