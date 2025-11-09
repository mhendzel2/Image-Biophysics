"""
Image Analysis Module
Provides functions for volumetric, morphometric, and colocalization analysis.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, label
from typing import Dict, Any, Optional, Tuple

class AnalysisManager:
    """Manages post-segmentation analysis tasks."""

    def __init__(self):
        pass

    def calculate_colocalization(
        self, 
        channel1: np.ndarray, 
        channel2: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        threshold1: float = 0,
        threshold2: float = 0
    ) -> Dict[str, Any]:
        """
        Calculates colocalization coefficients between two channels.

        Args:
            channel1: Numpy array for the first channel.
            channel2: Numpy array for the second channel (must be same shape as channel1).
            mask: Optional boolean or binary mask. Analysis is restricted to this region.
            threshold1: Intensity threshold for channel 1 to be considered 'present'.
            threshold2: Intensity threshold for channel 2 to be considered 'present'.

        Returns:
            A dictionary with Pearson's and Mander's coefficients.
        """
        if channel1.shape != channel2.shape:
            return {'status': 'error', 'message': 'Input channels must have the same shape.'}

        try:
            # Apply mask if provided
            if mask is not None:
                if mask.shape != channel1.shape:
                    return {'status': 'error', 'message': 'Mask must have the same shape as the channels.'}
                # Ensure mask is boolean
                mask = mask > 0
                ch1 = channel1[mask]
                ch2 = channel2[mask]
            else:
                ch1 = channel1.flatten()
                ch2 = channel2.flatten()

            if ch1.size == 0 or ch2.size == 0:
                return {'status': 'warning', 'message': 'Masked region is empty.'}

            # --- Pearson's Correlation Coefficient (PCC) ---
            # np.corrcoef returns a 2x2 matrix, we need the value at [0, 1]
            pcc = np.corrcoef(ch1, ch2)[0, 1]
            if np.isnan(pcc):
                pcc = 0 # Handle case of zero standard deviation

            # --- Mander's Overlap Coefficients (MOC) ---
            # Apply thresholds
            ch1_thresh = ch1 > threshold1
            ch2_thresh = ch2 > threshold2

            # Sum of intensities where the other channel is present
            sum_ch1_colocalized = np.sum(ch1[ch2_thresh])
            sum_ch2_colocalized = np.sum(ch2[ch1_thresh])

            total_sum_ch1 = np.sum(ch1)
            total_sum_ch2 = np.sum(ch2)

            m1 = sum_ch1_colocalized / total_sum_ch1 if total_sum_ch1 > 0 else 0
            m2 = sum_ch2_colocalized / total_sum_ch2 if total_sum_ch2 > 0 else 0

            return {
                'status': 'success',
                'pearson_coefficient': pcc,
                'manders_m1': m1, # Fraction of Ch1 overlapping with Ch2
                'manders_m2': m2  # Fraction of Ch2 overlapping with Ch1
            }
        except Exception as e:
            return {'status': 'error', 'message': f'Colocalization analysis failed: {e}'}

    def calculate_morphometrics(self, 
                              segmentation_mask: np.ndarray, 
                              original_image: Optional[np.ndarray] = None, 
                              voxel_size: Optional[Tuple[float, ...]] = None) -> Dict[str, Any]:
        """
        Calculate volumetric and morphometric properties for each segmented object.
        """
        # ... (rest of the function remains the same) ...
        if segmentation_mask.max() == 0:
            return {
                'status': 'warning',
                'message': 'Segmentation mask is empty. No objects to analyze.',
                'results_df': pd.DataFrame()
            }

        labeled_mask = label(segmentation_mask, background=0)
        properties_to_measure = [
            'label', 'area', 'perimeter', 'solidity', 'eccentricity',
            'equivalent_diameter', 'orientation', 'major_axis_length', 'minor_axis_length'
        ]
        if original_image is not None:
            properties_to_measure.extend(['mean_intensity', 'max_intensity', 'min_intensity'])

        try:
            props_table = regionprops_table(
                labeled_mask, 
                intensity_image=original_image, 
                properties=properties_to_measure
            )
            results_df = pd.DataFrame(props_table)

            if 'area' in results_df and voxel_size:
                if len(voxel_size) == 2:
                    pixel_area = voxel_size[0] * voxel_size[1]
                    results_df['physical_area'] = results_df['area'] * pixel_area
                elif len(voxel_size) == 3:
                    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
                    results_df.rename(columns={'area': 'volume_voxels'}, inplace=True)
                    results_df['volume_microns^3'] = results_df['volume_voxels'] * voxel_volume

            if 'perimeter' in results_df and 'area' in results_df:
                perimeter = results_df['perimeter']
                area = results_df['area']
                results_df['circularity'] = (4 * np.pi * area) / (perimeter**2)
                results_df['circularity'].fillna(0, inplace=True)

            summary_stats = {
                'Number of Objects': len(results_df),
                'Total Area/Volume (pixels/voxels)': results_df['area' if 'area' in results_df else 'volume_voxels'].sum(),
            }
            if 'physical_area' in results_df:
                summary_stats['Total Physical Area (microns^2)'] = results_df['physical_area'].sum()
            if 'volume_microns^3' in results_df:
                summary_stats['Total Volume (microns^3)'] = results_df['volume_microns^3'].sum()

            return {
                'status': 'success',
                'results_df': results_df,
                'summary_stats': summary_stats
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Morphometric analysis failed: {str(e)}',
                'results_df': pd.DataFrame()
            }

    def calculate_percolation(self, segmentation_mask: np.ndarray, 
                              axis: int = 0) -> Dict[str, Any]:
        """
        Calculates the percolation volume of the largest connected component in a 3D mask.
        """
        # ... (rest of the function remains the same) ...
        if segmentation_mask.ndim != 3:
            return {'status': 'error', 'message': 'Percolation analysis requires a 3D mask.'}

        try:
            labeled_mask, num_features = label(segmentation_mask, return_num=True, background=0)
            if num_features == 0:
                return {'status': 'success', 'percolates': False, 'percolation_volume': 0, 'largest_component_volume': 0, 'total_volume': 0}

            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_component_label = np.argmax(component_sizes) + 1
            largest_component_volume = component_sizes.max()
            total_volume = np.sum(component_sizes)

            axis_len = segmentation_mask.shape[axis]
            component_on_start = np.unique(np.take(labeled_mask, 0, axis=axis))
            component_on_end = np.unique(np.take(labeled_mask, axis_len - 1, axis=axis))

            percolates = largest_component_label in component_on_start and \
                         largest_component_label in component_on_end
            
            percolation_volume = largest_component_volume if percolates else 0

            return {
                'status': 'success',
                'percolates': percolates,
                'percolation_volume_voxels': percolation_volume,
                'largest_component_volume_voxels': largest_component_volume,
                'total_volume_voxels': total_volume,
                'percolating_fraction': (percolation_volume / total_volume) if total_volume > 0 else 0
            }
        except Exception as e:
            return {'status': 'error', 'message': f'Percolation analysis failed: {e}'}
