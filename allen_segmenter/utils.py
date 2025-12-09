"""
Utility functions for Allen Cell Segmenter integration.
"""

import importlib.util
import warnings

def supports_segmenter_ml() -> bool:
    """
    Check if the environment supports ML segmentation (torch + CUDA).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available. ML segmentation will be disabled.")
            return False
        return True
    except ImportError:
        warnings.warn("PyTorch is not installed. ML segmentation will be disabled.")
        return False
    except Exception as e:
        warnings.warn(f"Error checking for ML support: {e}")
        return False

def to_segmenter_array(image, channel_axis=None):
    """
    Convert an image to the format expected by aicssegmentation.
    Expected format is typically (Z, Y, X) or (C, Z, Y, X).
    
    Args:
        image (np.ndarray): Input image.
        channel_axis (int, optional): Index of the channel axis. 
                                      If None, assumes (Z, Y, X) for 3D or (Y, X) for 2D.
    
    Returns:
        np.ndarray: Image in (Z, Y, X) format (if single channel) or (C, Z, Y, X).
    """
    import numpy as np
    
    img = np.asarray(image)
    
    # Handle 2D images by adding Z dimension
    if img.ndim == 2:
        return img[np.newaxis, :, :]
        
    # If 3D and no channel axis specified, assume (Z, Y, X)
    if img.ndim == 3 and channel_axis is None:
        return img
        
    # If 4D, ensure C is first
    if img.ndim == 4:
        if channel_axis is not None and channel_axis != 0:
            img = np.moveaxis(img, channel_axis, 0)
        return img
        
    return img
