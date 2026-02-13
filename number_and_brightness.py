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
from typing import Any, Dict, Optional

class NumberAndBrightness:
    """
    A class to perform Number & Brightness (N&B) analysis.
    """

    def analyze(self,
                image_stack: np.ndarray,
                window_size: int = 1,
                detrend_method: str = "none",
                detrend_kwargs: Optional[Dict[str, Any]] = None,
                camera_offset_adu: float = 0.0,
                camera_gain_e_per_adu: Optional[float] = None,
                adu_per_e: Optional[float] = None,
                read_noise_e: Optional[float] = None,
                excess_noise_factor: Optional[float] = None,
                block_size_frames: Optional[int] = None,
                roi_mask: Optional[np.ndarray] = None):
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

        detrend_kwargs = detrend_kwargs or {}

        n_frames, height, width = image_stack.shape
        stack = np.asarray(image_stack, dtype=float)

        if roi_mask is None:
            roi_mask = np.ones((height, width), dtype=bool)
        else:
            roi_mask = np.asarray(roi_mask, dtype=bool)
            if roi_mask.shape != (height, width):
                return {'status': 'error', 'message': 'roi_mask shape must match image spatial dimensions.'}

        # 1) Convert ADU -> electrons and subtract baseline camera offset.
        # Convention: camera_gain_e_per_adu (preferred) OR adu_per_e.
        if camera_gain_e_per_adu is not None and adu_per_e is not None:
            return {'status': 'error', 'message': 'Provide only one of camera_gain_e_per_adu or adu_per_e.'}

        if camera_gain_e_per_adu is None:
            if adu_per_e is not None:
                if adu_per_e <= 0:
                    return {'status': 'error', 'message': 'adu_per_e must be > 0.'}
                gain_e_per_adu = 1.0 / adu_per_e
            else:
                gain_e_per_adu = 1.0
        else:
            if camera_gain_e_per_adu <= 0:
                return {'status': 'error', 'message': 'camera_gain_e_per_adu must be > 0.'}
            gain_e_per_adu = camera_gain_e_per_adu

        stack_e = (stack - float(camera_offset_adu)) * gain_e_per_adu

        # 2) Optional global detrending from ROI mean intensity trajectory.
        detrended_stack_e, trend = self._apply_global_detrend(stack_e, roi_mask, detrend_method, detrend_kwargs)

        # 3) Compute raw moments and corrected variance moments.
        mean_raw_map = np.mean(stack_e, axis=0)
        var_raw_map = np.var(stack_e, axis=0)

        mean_map = np.mean(detrended_stack_e, axis=0)
        var_map = np.var(detrended_stack_e, axis=0)

        read_var = float(read_noise_e ** 2) if read_noise_e is not None else 0.0
        enf = float(excess_noise_factor) if excess_noise_factor is not None else 1.0
        enf = max(enf, 1e-8)

        # Approximate correction:
        # var_signal ≈ (var_measured - read_noise^2) / ENF^2
        var_corrected_map = (var_map - read_var) / (enf ** 2)
        var_corrected_map = np.clip(var_corrected_map, 0.0, None)

        brightness_map = np.zeros_like(mean_map)
        number_map = np.zeros_like(mean_map)

        mask = mean_map > 0

        brightness_map[mask] = var_corrected_map[mask] / mean_map[mask]

        var_mask = var_corrected_map > 0
        valid_mask = mask & var_mask

        number_map[valid_mask] = (mean_map[valid_mask] ** 2) / var_corrected_map[valid_mask]

        # Calculate True Brightness (epsilon) and True Number (n)
        # Assumes shot noise = 1 (Poissonian)
        # epsilon = B - 1
        # n = <k> / epsilon

        true_brightness_map = brightness_map - 1
        true_number_map = np.zeros_like(number_map)

        true_b_mask = true_brightness_map > 0
        valid_true_mask = mask & true_b_mask

        true_number_map[valid_true_mask] = mean_map[valid_true_mask] / true_brightness_map[valid_true_mask]

        if window_size > 1:
            brightness_map = ndimage.uniform_filter(brightness_map, size=window_size)
            number_map = ndimage.uniform_filter(number_map, size=window_size)
            true_brightness_map = ndimage.uniform_filter(true_brightness_map, size=window_size)
            true_number_map = ndimage.uniform_filter(true_number_map, size=window_size)

        # 4) Block-based uncertainty proxy.
        brightness_std_map = None
        number_std_map = None
        brightness_se_map = None
        number_se_map = None
        n_blocks = 0
        if block_size_frames is not None and block_size_frames >= 2:
            n_blocks = n_frames // int(block_size_frames)
            if n_blocks >= 2:
                b_maps = []
                n_maps = []
                for b in range(n_blocks):
                    i0 = b * int(block_size_frames)
                    i1 = i0 + int(block_size_frames)
                    blk = detrended_stack_e[i0:i1]
                    blk_mean = np.mean(blk, axis=0)
                    blk_var = np.var(blk, axis=0)
                    blk_var_corr = np.clip((blk_var - read_var) / (enf ** 2), 0.0, None)
                    blk_b = np.zeros_like(blk_mean)
                    blk_n = np.zeros_like(blk_mean)
                    blk_mask = blk_mean > 0
                    blk_b[blk_mask] = blk_var_corr[blk_mask] / blk_mean[blk_mask]
                    blk_valid = blk_mask & (blk_var_corr > 0)
                    blk_n[blk_valid] = (blk_mean[blk_valid] ** 2) / blk_var_corr[blk_valid]
                    b_maps.append(blk_b)
                    n_maps.append(blk_n)
                b_arr = np.stack(b_maps, axis=0)
                n_arr = np.stack(n_maps, axis=0)
                brightness_std_map = np.std(b_arr, axis=0, ddof=1)
                number_std_map = np.std(n_arr, axis=0, ddof=1)
                brightness_se_map = brightness_std_map / np.sqrt(n_blocks)
                number_se_map = number_std_map / np.sqrt(n_blocks)

        # Apply ROI masking to outputs for transparency.
        def _mask_out(x):
            arr = np.asarray(x)
            out = np.full_like(arr, np.nan, dtype=float)
            out[roi_mask] = arr[roi_mask]
            return out
        
        return {
            'status': 'success',
            'number_map': _mask_out(number_map),
            'brightness_map': _mask_out(brightness_map),
            'true_number_map': _mask_out(true_number_map),
            'true_brightness_map': _mask_out(true_brightness_map),
            'mean_intensity_map': _mask_out(mean_map),
            'variance_map': _mask_out(var_map),
            'raw_mean_intensity_map': _mask_out(mean_raw_map),
            'raw_variance_map': _mask_out(var_raw_map),
            'corrected_variance_map': _mask_out(var_corrected_map),
            'trend_trace': trend,
            'detrend_method': detrend_method,
            'camera_model': {
                'camera_offset_adu': float(camera_offset_adu),
                'gain_e_per_adu': float(gain_e_per_adu),
                'read_noise_e': float(read_noise_e) if read_noise_e is not None else 0.0,
                'excess_noise_factor': float(enf)
            },
            'brightness_std_map': _mask_out(brightness_std_map) if brightness_std_map is not None else None,
            'number_std_map': _mask_out(number_std_map) if number_std_map is not None else None,
            'brightness_se_map': _mask_out(brightness_se_map) if brightness_se_map is not None else None,
            'number_se_map': _mask_out(number_se_map) if number_se_map is not None else None,
            'n_blocks': int(n_blocks),
            'message': 'N&B analysis completed successfully.'
        }

    def _apply_global_detrend(self,
                              stack_e: np.ndarray,
                              roi_mask: np.ndarray,
                              detrend_method: str,
                              detrend_kwargs: Dict[str, Any]):
        """Detrend stack globally from ROI-mean trajectory and rescale frames."""
        method = str(detrend_method).lower()
        if method == 'none':
            return stack_e, np.ones(stack_e.shape[0], dtype=float)

        T = stack_e.shape[0]
        trace = np.array([np.mean(stack_e[t][roi_mask]) for t in range(T)], dtype=float)
        t = np.arange(T, dtype=float)
        eps = 1e-12

        if method == 'global_exponential':
            pos = trace > eps
            if np.sum(pos) < 3:
                trend = np.full(T, np.mean(trace[trace > eps]) if np.any(trace > eps) else 1.0)
            else:
                coeff = np.polyfit(t[pos], np.log(trace[pos]), 1)
                trend = np.exp(coeff[1] + coeff[0] * t)
        elif method == 'global_polynomial':
            degree = int(detrend_kwargs.get('degree', 2))
            degree = max(1, min(degree, 5))
            coeff = np.polyfit(t, trace, degree)
            trend = np.polyval(coeff, t)
        elif method == 'rolling_mean':
            window = int(detrend_kwargs.get('window', max(3, T // 10)))
            window = max(3, min(window, T))
            kernel = np.ones(window, dtype=float) / window
            trend = np.convolve(trace, kernel, mode='same')
        else:
            raise ValueError(f'Unknown detrend_method: {detrend_method}')

        trend = np.clip(trend, eps, None)
        ref = float(np.median(trend))
        scale = ref / trend
        out = stack_e * scale[:, None, None]
        return out, trend
