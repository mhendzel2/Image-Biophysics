"""
Number & Brightness (N&B) Analysis Module

This module provides functions for performing Number & Brightness (N&B) analysis on time-series images.
N&B is a fluorescence fluctuation spectroscopy technique used to determine the average number of particles (N)
and the molecular brightness (B) of fluorescently labeled molecules in a sample.

The core principle is to relate the statistical moments (mean and variance) of the fluorescence intensity
fluctuations to the number and brightness of the particles.

Key formulas:
- Brightness (B) = variance / mean
- Number (N) = mean^2 / variance

References:
- Digman, M.A. et al. (2008) "The Number and Brightness of Molecules: A New Method for Sizing and Aggregation" Biophys J 94(7):L2320-32
"""

import numpy as np

class NumberAndBrightness:
    """
    A class to perform Number & Brightness (N&B) analysis.
    """

    def analyze(self, image_stack: np.ndarray, window_size: int = 32):
        """
        Performs N&B analysis on a stack of images.

        The analysis is performed over sliding windows of the specified size to generate
        maps of the number of particles (N) and their brightness (B).

        Args:
            image_stack (np.ndarray): A 3D numpy array representing the time-series of images (time, height, width).
            window_size (int): The size of the sliding window for the analysis.

        Returns:
            dict: A dictionary containing the N and B maps, with the following keys:
                  - 'status': 'success' or 'error'
                  - 'number_map': A 2D numpy array with the number of particles (N) for each window.
                  - 'brightness_map': A 2D numpy array with the brightness (B) for each window.
                  - 'message': A message indicating the result of the analysis.
        """
        if image_stack.ndim != 3:
            return {'status': 'error', 'message': 'Input must be a 3D image stack (time, height, width).'}

        n_frames, height, width = image_stack.shape

        # Initialize the N and B maps
        number_map = np.zeros((height - window_size, width - window_size))
        brightness_map = np.zeros((height - window_size, width - window_size))

        for y in range(height - window_size):
            for x in range(width - window_size):
                # Extract the window from the image stack
                window = image_stack[:, y:y+window_size, x:x+window_size]

                # Calculate the mean and variance over time
                mean_intensity = np.mean(window, axis=0)
                variance_intensity = np.var(window, axis=0)

                # Calculate the average mean and variance over the window
                avg_mean = np.mean(mean_intensity)
                avg_variance = np.mean(variance_intensity)
                
                if avg_mean > 0:
                    # Calculate N and B for the window
                    brightness = avg_variance / avg_mean
                    number = avg_mean / brightness if brightness > 0 else 0
                else:
                    brightness = 0
                    number = 0

                number_map[y, x] = number
                brightness_map[y, x] = brightness
        
        return {
            'status': 'success',
            'number_map': number_map,
            'brightness_map': brightness_map,
            'message': 'N&B analysis completed successfully.'
        }
