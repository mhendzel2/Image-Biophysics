"""
Number & Brightness (N&B) Analysis Module

This module provides functions for performing Number & Brightness (N&B) analysis on time-series images.
N&B is a fluorescence fluctuation spectroscopy technique used to determine the average number of particles (N)
and the molecular brightness (B) of fluorescently labeled molecules in a sample.

The core principle is to relate the statistical moments (mean and variance) of the fluorescence intensity
fluctuations to the number and brightness of the particles.

Key formulas:
- Apparent Brightness (B) = variance / mean
- Apparent Number (N) = mean^2 / variance

- True Molecular Brightness (epsilon) = B - 1 (for photon counting detectors)
- True Number of Particles (n) = mean / epsilon = mean^2 / (variance - mean)

References:
- Digman, M.A. et al. (2008) "The Number and Brightness of Molecules: A New Method for Sizing and Aggregation" Biophys J 94(7):L2320-32
"""

import numpy as np
from scipy import ndimage

class NumberAndBrightness:
    """
    A class to perform Number & Brightness (N&B) analysis.
    """

    def analyze(self, image_stack: np.ndarray, window_size: int = 1):
        """
        Performs N&B analysis on a stack of images.

        Args:
            image_stack (np.ndarray): A 3D numpy array representing the time-series of images (time, height, width).
            window_size (int): The size of the sliding window for spatial averaging of the results (optional smoothing).
                               Default is 1 (pixel-wise analysis).

        Returns:
            dict: A dictionary containing the N and B maps, with the following keys:
                  - 'status': 'success' or 'error'
                  - 'number_map': A 2D numpy array with the apparent number of particles (N).
                  - 'brightness_map': A 2D numpy array with the apparent brightness (B).
                  - 'true_brightness_map': A 2D numpy array with true brightness (B-1).
                  - 'true_number_map': A 2D numpy array with true number (n).
                  - 'message': A message indicating the result of the analysis.
        """
        if image_stack.ndim != 3:
            return {'status': 'error', 'message': 'Input must be a 3D image stack (time, height, width).'}

        n_frames, height, width = image_stack.shape

        # Calculate mean and variance along the time axis (axis 0)
        # This gives pixel-wise statistics
        mean_map = np.mean(image_stack, axis=0)
        var_map = np.var(image_stack, axis=0)

        # Calculate Apparent Brightness (B) and Number (N)
        # B = sigma^2 / <k>
        # N = <k>^2 / sigma^2

        # Initialize maps
        brightness_map = np.zeros_like(mean_map)
        number_map = np.zeros_like(mean_map)

        # Avoid division by zero
        mask = mean_map > 0

        brightness_map[mask] = var_map[mask] / mean_map[mask]

        # Avoid division by zero for number calculation (if variance is 0)
        var_mask = var_map > 0
        valid_mask = mask & var_mask

        number_map[valid_mask] = (mean_map[valid_mask]**2) / var_map[valid_mask]

        # Calculate True Brightness (epsilon) and True Number (n)
        # Assumes shot noise = 1 (Poissonian)
        # epsilon = B - 1
        # n = <k> / epsilon

        true_brightness_map = brightness_map - 1
        true_number_map = np.zeros_like(number_map)

        true_b_mask = true_brightness_map > 0
        valid_true_mask = mask & true_b_mask

        true_number_map[valid_true_mask] = mean_map[valid_true_mask] / true_brightness_map[valid_true_mask]

        # Optional spatial smoothing (if window_size > 1)
        if window_size > 1:
            # Use uniform filter for smoothing
            brightness_map = ndimage.uniform_filter(brightness_map, size=window_size)
            number_map = ndimage.uniform_filter(number_map, size=window_size)
            true_brightness_map = ndimage.uniform_filter(true_brightness_map, size=window_size)
            true_number_map = ndimage.uniform_filter(true_number_map, size=window_size)
        
        return {
            'status': 'success',
            'number_map': number_map,
            'brightness_map': brightness_map,
            'true_number_map': true_number_map,
            'true_brightness_map': true_brightness_map,
            'mean_intensity_map': mean_map,
            'variance_map': var_map,
            'message': 'N&B analysis completed successfully.'
        }
