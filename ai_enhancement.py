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
from scipy.signal import convolve
from scipy.optimize import nnls
from scipy.fft import fftn, ifftn
from skimage import restoration, img_as_float
from skimage.io import imread, imsave
from skimage.restoration import denoise_nl_means

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
    if "ModuleNotFoundError" in str(e) or "No module named" in str(e):
        TORCH_AVAILABLE = False
        warnings.warn(f"PyTorch not available (missing modules): {e} - AI features limited")
    else:
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

try:
    from aics_segmentation.core.pre_processing_utils import image_normalization
    from aics_segmentation.core.workflows import W_General, W_Lamin, W_Myo, W_Sox, W_Membrane, W_Sec61b
    AICS_SEGMENTATION_AVAILABLE = True
except ImportError:
    AICS_SEGMENTATION_AVAILABLE = False
    warnings.warn("AICS-Segmentation not available - AI segmentation limited")


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
            'richardson_lucy_tv': True,
            'fista': True,
            'ista': True,
            'ictm': True,
            'nnls': True,
            'cellpose_segmentation': CELLPOSE_AVAILABLE,
            'stardist_segmentation': STARDIST_AVAILABLE,
            'aics_segmentation': AICS_SEGMENTATION_AVAILABLE,
            'tensorflow_methods': TF_AVAILABLE
        }
    
    def get_available_methods(self) -> List[str]:
        """Return list of available enhancement methods"""
        available = []
        if self.available_methods['classical_denoising']:
            available.extend(['Non-local Means Denoising', 'Richardson-Lucy Deconvolution'])
        if self.available_methods['richardson_lucy_tv']:
            available.append('Richardson-Lucy with Total Variation')
        if self.available_methods['fista']:
            available.append('FISTA Deconvolution')
        if self.available_methods['ista']:
            available.append('ISTA Deconvolution')
        if self.available_methods['ictm']:
            available.append('Iterative Constraint Tikhonov-Miller')
        if self.available_methods['nnls']:
            available.append('Non-negative Least Squares Deconvolution')
        if self.available_methods['noise2void']:
            available.append('Noise2Void Self-Supervised Denoising')
        if self.available_methods['cellpose_segmentation']:
            available.extend(['Cellpose Cell Segmentation', 'Cellpose Nucleus Segmentation'])
        if self.available_methods['stardist_segmentation']:
            available.append('StarDist Nucleus Segmentation')
        if self.available_methods['aics_segmentation']:
            available.append('AICS Cell Segmentation')
        return available
    
    def enhance_image(self, image_data: np.ndarray, method: str, 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AI enhancement to image data"""
        
        if method == 'Non-local Means Denoising':
            return self._apply_nlm_denoising(image_data, parameters)
        elif method == 'Richardson-Lucy Deconvolution':
            return self._apply_richardson_lucy(image_data, parameters)
        elif method == 'Richardson-Lucy with Total Variation':
            return self._apply_richardson_lucy_tv(image_data, parameters)
        elif method == 'FISTA Deconvolution':
            return self._apply_fista(image_data, parameters)
        elif method == 'ISTA Deconvolution':
            return self._apply_ista(image_data, parameters)
        elif method == 'Iterative Constraint Tikhonov-Miller':
            return self._apply_ictm(image_data, parameters)
        elif method == 'Non-negative Least Squares Deconvolution':
            return self._apply_nnls_deconvolution(image_data, parameters)
        elif method == 'Noise2Void Self-Supervised Denoising':
            return self._apply_noise2void(image_data, parameters)
        elif method == 'Cellpose Cell Segmentation':
            return self._apply_cellpose_segmentation(image_data, parameters, model_type='cyto2')
        elif method == 'Cellpose Nucleus Segmentation':
            return self._apply_cellpose_segmentation(image_data, parameters, model_type='nuclei')
        elif method == 'StarDist Nucleus Segmentation':
            return self._apply_stardist_segmentation(image_data, parameters)
        elif method == 'AICS Cell Segmentation':
            return self._apply_aics_segmentation(image_data, parameters)
        else:
            return {'status': 'error', 'message': f'Unknown enhancement method: {method}'}
    
    def _generate_psf(self, shape: Tuple[int, ...], psf_size: int, psf_sigma: float) -> np.ndarray:
        """Generate a Gaussian Point Spread Function."""
        x = np.arange(psf_size) - psf_size // 2
        if len(shape) == 2:
            xx, yy = np.meshgrid(x, x)
            psf = np.exp(-(xx**2 + yy**2) / (2 * psf_sigma**2))
        else:
            xx, yy, zz = np.meshgrid(x, x, x)
            psf = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * psf_sigma**2))
        psf /= np.sum(psf)
        return psf

    def _apply_nlm_denoising(self, image_data: np.ndarray, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply non-local means denoising using scikit-image"""
        
        try:
            image_float = img_as_float(image_data)
            patch_size = parameters.get('patch_size', 5)
            patch_distance = parameters.get('patch_distance', 6)
            fast_mode = parameters.get('fast_mode', True)
            auto_sigma = parameters.get('auto_sigma', True)
            
            if auto_sigma:
                sigma_est = np.mean(restoration.estimate_sigma(image_float, channel_axis=None))
                h = 1.15 * sigma_est
            else:
                h = parameters.get('h', 0.1)
            
            patch_kw = dict(patch_size=patch_size, patch_distance=patch_distance, channel_axis=None)
            denoised = restoration.denoise_nl_means(
                image_float, h=h, fast_mode=fast_mode, **patch_kw
            )
            
            enhanced = (denoised * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)
            
            return {
                'enhanced_image': enhanced,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Non-local means denoising failed: {str(e)}'}
    
    def _apply_richardson_lucy(self, image_data: np.ndarray, 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Richardson-Lucy deconvolution"""
        
        try:
            image_float = img_as_float(image_data)
            iterations = parameters.get('iterations', 30)
            psf = self._generate_psf(image_data.shape, parameters.get('psf_size', 5), parameters.get('psf_sigma', 1.0))
            
            deconvolved = restoration.richardson_lucy(image_float, psf, iterations=iterations)
            
            enhanced = (deconvolved * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)
            
            return {
                'enhanced_image': enhanced,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Richardson-Lucy deconvolution failed: {str(e)}'}
            
    def _apply_richardson_lucy_tv(self, image_data: np.ndarray, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Richardson-Lucy deconvolution with Total Variation regularization."""
        try:
            image_float = img_as_float(image_data)
            iterations = parameters.get('iterations', 10)
            lambda_tv = parameters.get('lambda_tv', 0.002)
            psf = self._generate_psf(image_data.shape, parameters.get('psf_size', 5), parameters.get('psf_sigma', 1.0))

            deconvolved = restoration.richardson_lucy(image_float, psf, iterations=iterations, clip=False)
            
            for _ in range(iterations):
                convolved = convolve(deconvolved, psf, mode='same')
                relative_blur = image_float / (convolved + 1e-10)
                correction = convolve(relative_blur, psf[::-1, ::-1], mode='same')

                grad_x = np.gradient(deconvolved, axis=1)
                grad_y = np.gradient(deconvolved, axis=0)
                norm = np.sqrt(grad_x**2 + grad_y**2 + 1e-10)
                div = np.gradient(grad_x / norm, axis=1) + np.gradient(grad_y / norm, axis=0)

                deconvolved *= correction / (1 - lambda_tv * div)
                deconvolved = np.clip(deconvolved, 0, 1)

            enhanced = (deconvolved * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)

            return {
                'enhanced_image': enhanced,
                'status': 'success'
            }
        except Exception as e:
            return {'status': 'error', 'message': f'RL-TV deconvolution failed: {e}'}

    def _soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _apply_fista(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply FISTA deconvolution."""
        try:
            image_float = img_as_float(image_data)
            iterations = parameters.get('iterations', 50)
            lambda_reg = parameters.get('lambda_reg', 0.1)
            psf = self._generate_psf(image_data.shape, parameters.get('psf_size', 5), parameters.get('psf_sigma', 1.0))

            x_k = np.copy(image_float)
            y_k = np.copy(image_float)
            t_k = 1.0
            psf_otf = fftn(np.fft.ifftshift(psf), s=image_float.shape)
            psf_adj_otf = np.conj(psf_otf)
            
            for _ in range(iterations):
                x_k_prev = np.copy(x_k)
                t_k_prev = t_k

                grad = ifftn(fftn(y_k) * psf_otf - fftn(image_float) * psf_otf, axes=image_float.shape).real
                x_k = self._soft_threshold(y_k - grad, lambda_reg)
                
                t_k = (1 + np.sqrt(1 + 4 * t_k_prev**2)) / 2
                y_k = x_k + ((t_k_prev - 1) / t_k) * (x_k - x_k_prev)
            
            enhanced = (np.clip(x_k, 0, 1) * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)

            return {'enhanced_image': enhanced, 'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': f'FISTA deconvolution failed: {e}'}

    def _apply_ista(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ISTA deconvolution."""
        try:
            image_float = img_as_float(image_data)
            iterations = parameters.get('iterations', 50)
            lambda_reg = parameters.get('lambda_reg', 0.1)
            psf = self._generate_psf(image_data.shape, parameters.get('psf_size', 5), parameters.get('psf_sigma', 1.0))

            x_k = np.copy(image_float)
            psf_otf = fftn(np.fft.ifftshift(psf), s=image_float.shape)

            for _ in range(iterations):
                grad = ifftn(fftn(x_k) * psf_otf - fftn(image_float), axes=image_float.shape).real
                x_k = self._soft_threshold(x_k - grad, lambda_reg)
            
            enhanced = (np.clip(x_k, 0, 1) * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)

            return {'enhanced_image': enhanced, 'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': f'ISTA deconvolution failed: {e}'}

    def _apply_ictm(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Iterative Constraint Tikhonov-Miller deconvolution."""
        try:
            image_float = img_as_float(image_data)
            iterations = parameters.get('iterations', 30)
            reg_param = parameters.get('reg_param', 0.01)
            psf = self._generate_psf(image_data.shape, parameters.get('psf_size', 5), parameters.get('psf_sigma', 1.0))

            x_k = np.copy(image_float)
            psf_otf = fftn(np.fft.ifftshift(psf), s=image_float.shape)
            psf_otf_conj = np.conj(psf_otf)
            
            for _ in range(iterations):
                numerator = fftn(image_float) * psf_otf_conj
                denominator = fftn(x_k) * psf_otf * psf_otf_conj + reg_param
                x_k = ifftn(numerator / denominator, axes=image_float.shape).real
                x_k = np.clip(x_k, 0, 1)

            enhanced = (x_k * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)
            return {'enhanced_image': enhanced, 'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': f'ICTM deconvolution failed: {e}'}

    def _apply_nnls_deconvolution(self, image_data: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Non-Negative Least Squares deconvolution."""
        try:
            image_flat = image_data.flatten()
            psf_size = parameters.get('psf_size', 5)
            psf_sigma = parameters.get('psf_sigma', 1.0)
            psf = self._generate_psf(image_data.shape, psf_size, psf_sigma)

            # Create the convolution matrix
            from scipy.linalg import toeplitz
            psf_flat = psf.flatten()
            h, w = image_data.shape
            A_rows = []
            for i in range(h):
                for j in range(w):
                    row = np.zeros(h * w)
                    for r in range(psf.shape[0]):
                        for c in range(psf.shape[1]):
                            if 0 <= i-r < h and 0 <= j-c < w:
                                row[(i-r)*w + (j-c)] = psf[r,c]
                    A_rows.append(row)
            A = np.array(A_rows)
            
            # Solve using nnls
            x, _ = nnls(A, image_flat)
            enhanced_flat = x
            enhanced = enhanced_flat.reshape(image_data.shape)
            enhanced = (enhanced / enhanced.max() * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)
            return {'enhanced_image': enhanced, 'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': f'NNLS deconvolution failed: {e}'}

    def _apply_noise2void(self, image_data: np.ndarray, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Noise2Void-style self-supervised denoising using enhanced methods"""
        
        try:
            patch_size = parameters.get('patch_size', 7)
            patch_distance = parameters.get('patch_distance', 11)
            h = parameters.get('h', 0.1)
            
            img_float = img_as_float(image_data)
            
            denoised = denoise_nl_means(
                img_float, 
                patch_size=patch_size,
                patch_distance=patch_distance,
                h=h * np.var(img_float),
                fast_mode=True,
                channel_axis=-1 if image_data.ndim > 2 else None
            )
            
            noise_reduction = np.std(image_data) - np.std(denoised)
            snr_improvement = 20 * np.log10(np.std(denoised) / (np.std(image_data - denoised) + 1e-10))
            
            enhanced = (denoised * (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1)).astype(image_data.dtype)

            return {
                'status': 'success',
                'enhanced_image': enhanced,
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Noise2Void-style denoising failed: {str(e)}',
            }
    
    def _apply_cellpose_segmentation(self, image_data: np.ndarray, 
                                   parameters: Dict[str, Any], 
                                   model_type: str = 'cyto') -> Dict[str, Any]:
        """Apply Cellpose segmentation"""
        
        if not CELLPOSE_AVAILABLE:
            return {'status': 'error', 'message': 'Cellpose library required for AI segmentation'}
        
        try:
            diameter = parameters.get('diameter', None)
            channels = parameters.get('channels', [0, 0])
            gpu = parameters.get('use_gpu', False)
            
            model = cellpose_models.Cellpose(model_type=model_type, gpu=gpu, torch=True)
            
            masks, flows, styles, diams = model.eval(
                image_data, 
                channels=channels, 
                diameter=diameter
            )
            
            return {
                'segmentation_masks': masks,
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
            model_name = parameters.get('model_name', '2D_versatile_fluo')
            prob_thresh = parameters.get('prob_thresh', None)
            nms_thresh = parameters.get('nms_thresh', None)
            
            image_norm = stardist_normalize(img_as_float(image_data), 1, 99.8)
            
            model = StarDist2D.from_pretrained(model_name)
            
            labels, details = model.predict_instances(
                image_norm, 
                prob_thresh=prob_thresh, 
                nms_thresh=nms_thresh
            )
            
            return {
                'segmentation_masks': labels,
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'StarDist segmentation failed: {str(e)}'}

    def _apply_aics_segmentation(self, image_data: np.ndarray,
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AICS-Cell-Segmenter"""
        if not AICS_SEGMENTATION_AVAILABLE:
            return {'status': 'error', 'message': 'AICS-Segmentation library required for AI segmentation'}
        try:
            model_name = parameters.get('model_name', 'General')
            
            workflow_map = {
                'Lamin': W_Lamin(),
                'Myo': W_Myo(),
                'Sox': W_Sox(),
                'Membrane': W_Membrane(),
                'Sec61b': W_Sec61b(),
                'General': W_General()
            }
            workflow = workflow_map.get(model_name, W_General())

            normalized_image = image_normalization(image_data)
            segmentation_result = workflow.execute_flow(normalized_image)
            
            return {
                'segmentation_masks': segmentation_result,
                'status': 'success'
            }
        except Exception as e:
            return {'status': 'error', 'message': f'AICS segmentation failed: {str(e)}'}

def get_enhancement_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for enhancement methods"""
    
    defaults = {
        'Non-local Means Denoising': {
            'patch_size': 5, 'patch_distance': 6, 'fast_mode': True, 'auto_sigma': True, 'h': 0.1
        },
        'Richardson-Lucy Deconvolution': {
            'iterations': 30, 'psf_size': 5, 'psf_sigma': 1.0
        },
        'Richardson-Lucy with Total Variation': {
            'iterations': 10, 'lambda_tv': 0.002, 'psf_size': 5, 'psf_sigma': 1.0
        },
        'FISTA Deconvolution': {
            'iterations': 50, 'lambda_reg': 0.05, 'psf_size': 5, 'psf_sigma': 1.0
        },
        'ISTA Deconvolution': {
            'iterations': 50, 'lambda_reg': 0.05, 'psf_size': 5, 'psf_sigma': 1.0
        },
        'Iterative Constraint Tikhonov-Miller': {
            'iterations': 30, 'reg_param': 0.01, 'psf_size': 5, 'psf_sigma': 1.0
        },
        'Non-negative Least Squares Deconvolution': {
            'psf_size': 5, 'psf_sigma': 1.0
        },
        'Cellpose Cell Segmentation': {
            'diameter': 30, 'channels': [0, 0], 'use_gpu': False
        },
        'Cellpose Nucleus Segmentation': {
            'diameter': 30, 'channels': [0, 0], 'use_gpu': False
        },
        'StarDist Nucleus Segmentation': {
            'model_name': '2D_versatile_fluo', 'prob_thresh': None, 'nms_thresh': None
        },
        'AICS Cell Segmentation': {
            'model_name': 'General'
        }
    }
    return defaults.get(method, {})
