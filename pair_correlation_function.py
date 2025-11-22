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
        # g(r) - 1 = <dI(x) dI(x+r)> / <I>^2

        mean_intensity = np.mean(image)
        if mean_intensity == 0:
            return {'status': 'error', 'message': 'Image mean is zero, cannot calculate PCF.'}
        
        # Fluctuation image normalized by mean
        normalized_image = (image - mean_intensity) / mean_intensity

        # Calculate the autocorrelation of fluctuations using FFT
        # fftconvolve returns sum(norm[i] * norm[i+r])
        # We need to divide by the number of overlapping pixels for each lag r

        # Standard fftconvolve with 'same' uses zero-padding, so edges are affected.
        # To get proper normalization, we also convolve a mask of ones.

        # 1. Autocorrelation of normalized fluctuations (numerator sum)
        autocorr_sum = fftconvolve(normalized_image, normalized_image[::-1, ::-1], mode='full')

        # 2. Autocorrelation of the window/mask (denominator count)
        mask = np.ones_like(normalized_image)
        overlap_count = fftconvolve(mask, mask[::-1, ::-1], mode='full')

        # Avoid division by zero
        overlap_count[overlap_count < 1] = 1

        # Normalized autocorrelation map
        autocorr_map = autocorr_sum / overlap_count

        # Extract the center part corresponding to lags
        # For mode='full', the center is at (H-1, W-1) if shape is (H,W)
        h, w = normalized_image.shape
        center_y, center_x = h - 1, w - 1

        # Radial averaging
        # Create a grid of radii corresponding to the full correlation map
        y, x = np.indices(autocorr_map.shape)
        r_grid = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # We only care about integer radii
        r_int = np.round(r_grid).astype(int)

        # Limit to max_radius
        mask_radius = r_int <= max_radius
        
        # Bin the autocorrelation values by radius
        # Use bincount for efficiency
        
        # Flatten masked arrays
        r_values = r_int[mask_radius]
        corr_values = autocorr_map[mask_radius]

        # Sum of correlations for each radius bin
        tbin = np.bincount(r_values, weights=corr_values)
        # Count of pixels in each radius bin
        nr = np.bincount(r_values)

        # Calculate radial profile (average)
        # Handle cases where nr might be 0 (though unlikely for small radii)
        valid_bins = nr > 0
        radial_profile = np.zeros_like(tbin, dtype=float)
        radial_profile[valid_bins] = tbin[valid_bins] / nr[valid_bins]

        # The result is g(r) - 1. So add 1 to get g(r).
        pcf = radial_profile + 1

        # Crop to max_radius (bincount might go slightly larger due to rounding)
        if len(pcf) > max_radius + 1:
            pcf = pcf[:max_radius + 1]

        radius = np.arange(len(pcf))

        return {
            'status': 'success',
            'radius': radius,
            'pcf': pcf,
            'message': 'PCF analysis completed successfully.'
        }
