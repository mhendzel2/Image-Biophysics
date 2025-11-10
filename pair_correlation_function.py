"""
Pair Correlation Function (PCF) Analysis Module

This module provides functions for calculating the Pair Correlation Function (PCF) from image data.
The PCF, g(r), describes the probability of finding a particle at a distance r from another particle.
It is a powerful tool for quantifying the spatial organization of molecules, such as clustering or dispersion.

The PCF is calculated by analyzing the spatial distribution of intensity fluctuations in an image.

Key formula:
- g(r) = <I(x)I(x+r)> / <I(x)>^2
  where I(x) is the intensity at position x, and <> denotes an average over all positions.

This implementation uses a fast Fourier transform (FFT) based approach for efficient computation.

References:
- "Introduction to the P-Correlation Function and its Application to Image-Based Quantification of Protein Aggregation" - a resource by the Laboratory for Fluorescence Dynamics.
"""

import numpy as np
from scipy.signal import fftconvolve

class PairCorrelationFunction:
    """
    A class to calculate the Pair Correlation Function (PCF) from an image.
    """

    def analyze(self, image: np.ndarray, max_radius: int = 50):
        """
        Calculates the Pair Correlation Function (PCF) for a 2D image.

        Args:
            image (np.ndarray): A 2D numpy array representing the image.
            max_radius (int): The maximum radius (in pixels) for the PCF calculation.

        Returns:
            dict: A dictionary containing the PCF results, with the following keys:
                  - 'status': 'success' or 'error'
                  - 'radius': A 1D numpy array of radius values (r).
                  - 'pcf': A 1D numpy array of the PCF values, g(r).
                  - 'message': A message indicating the result of the analysis.
        """
        if image.ndim != 2:
            return {'status': 'error', 'message': 'Input must be a 2D image.'}

        # Normalize the image by its mean
        mean_intensity = np.mean(image)
        if mean_intensity == 0:
            return {'status': 'error', 'message': 'Image mean is zero, cannot calculate PCF.'}
        normalized_image = image / mean_intensity - 1

        # Calculate the autocorrelation using FFT
        autocorr = fftconvolve(normalized_image, normalized_image[::-1, ::-1], mode='same')
        
        # Get the dimensions of the autocorrelation image
        height, width = autocorr.shape
        center_y, center_x = height // 2, width // 2

        # Radial averaging
        y, x = np.indices((height, width))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Bin the autocorrelation values by radius
        tbin = np.bincount(r.ravel(), autocorr.ravel())
        nr = np.bincount(r.ravel())
        
        # Avoid division by zero
        radial_profile = tbin / np.where(nr == 0, 1, nr)

        # The PCF is the radial profile of the normalized autocorrelation
        pcf = radial_profile[:max_radius] + 1
        radius = np.arange(max_radius)

        return {
            'status': 'success',
            'radius': radius,
            'pcf': pcf,
            'message': 'PCF analysis completed successfully.'
        }
