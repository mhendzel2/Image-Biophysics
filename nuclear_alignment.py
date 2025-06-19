"""
Nuclear Alignment Module for Biophysical Analysis
Provides robust alignment algorithms specifically designed for nuclear timelapse data
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import registration, feature, measure
from typing import Dict, Any, Tuple, Optional
import warnings

def align_nuclear_sequence(image_sequence: np.ndarray, 
                          alignment_method: str = 'phase_correlation',
                          reference_frame: int = 0) -> Dict[str, Any]:
    """
    Align nuclear time series using multiple robust methods
    
    Args:
        image_sequence: 3D array (T, Y, X) of nuclear images
        alignment_method: 'phase_correlation', 'optical_flow', or 'feature_based'
        reference_frame: Frame to use as alignment reference
        
    Returns:
        Dictionary with aligned sequence and quality metrics
    """
    
    if image_sequence.ndim != 3:
        return {
            'status': 'error',
            'message': 'Nuclear alignment requires 3D time series (T, Y, X)',
            'aligned_sequence': image_sequence
        }
    
    t_frames, height, width = image_sequence.shape
    
    if reference_frame >= t_frames:
        reference_frame = 0
    
    reference = image_sequence[reference_frame].astype(np.float32)
    aligned_sequence = np.zeros_like(image_sequence)
    aligned_sequence[reference_frame] = reference
    
    shifts = []
    quality_scores = []
    
    try:
        if alignment_method == 'phase_correlation':
            aligned_sequence, shifts, quality_scores = _phase_correlation_alignment(
                image_sequence, reference, reference_frame
            )
        elif alignment_method == 'optical_flow':
            aligned_sequence, shifts, quality_scores = _optical_flow_alignment(
                image_sequence, reference, reference_frame
            )
        elif alignment_method == 'feature_based':
            aligned_sequence, shifts, quality_scores = _feature_based_alignment(
                image_sequence, reference, reference_frame
            )
        else:
            # Fallback to phase correlation
            aligned_sequence, shifts, quality_scores = _phase_correlation_alignment(
                image_sequence, reference, reference_frame
            )
        
        # Calculate alignment quality metrics
        mean_shift = np.mean(np.abs(shifts), axis=0) if shifts else [0, 0]
        max_shift = np.max(np.abs(shifts), axis=0) if shifts else [0, 0]
        stability_score = np.mean(quality_scores) if quality_scores else 0
        
        return {
            'status': 'success',
            'aligned_sequence': aligned_sequence,
            'shifts': shifts,
            'quality_metrics': {
                'mean_shift_pixels': mean_shift,
                'max_shift_pixels': max_shift,
                'stability_score': stability_score,
                'alignment_method': alignment_method,
                'reference_frame': reference_frame
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Nuclear alignment failed: {str(e)}',
            'aligned_sequence': image_sequence
        }

def _phase_correlation_alignment(image_sequence: np.ndarray, 
                               reference: np.ndarray, 
                               reference_frame: int) -> Tuple[np.ndarray, list, list]:
    """Phase correlation-based alignment using scikit-image"""
    
    t_frames, height, width = image_sequence.shape
    aligned_sequence = np.zeros_like(image_sequence)
    aligned_sequence[reference_frame] = reference
    
    shifts = []
    quality_scores = []
    
    for t in range(t_frames):
        if t == reference_frame:
            shifts.append([0, 0])
            quality_scores.append(1.0)
            continue
        
        current_frame = image_sequence[t].astype(np.float32)
        
        # Compute phase correlation
        shift, error, diffphase = registration.phase_cross_correlation(
            reference, current_frame, upsample_factor=10
        )
        
        # Apply shift using scipy.ndimage
        aligned_frame = ndimage.shift(current_frame, shift, order=1, mode='nearest')
        aligned_sequence[t] = aligned_frame
        
        shifts.append(shift)
        # Quality score based on phase correlation error (lower is better)
        quality_score = 1.0 / (1.0 + error)
        quality_scores.append(quality_score)
    
    return aligned_sequence, shifts, quality_scores

def _optical_flow_alignment(image_sequence: np.ndarray, 
                          reference: np.ndarray, 
                          reference_frame: int) -> Tuple[np.ndarray, list, list]:
    """Optical flow-based alignment using Lucas-Kanade"""
    
    t_frames, height, width = image_sequence.shape
    aligned_sequence = np.zeros_like(image_sequence)
    aligned_sequence[reference_frame] = reference
    
    shifts = []
    quality_scores = []
    
    # Convert reference to uint8 for OpenCV
    ref_uint8 = (reference * 255 / np.max(reference)).astype(np.uint8)
    
    for t in range(t_frames):
        if t == reference_frame:
            shifts.append([0, 0])
            quality_scores.append(1.0)
            continue
        
        current_frame = image_sequence[t].astype(np.float32)
        curr_uint8 = (current_frame * 255 / np.max(current_frame)).astype(np.uint8)
        
        # Detect features in reference frame
        corners = cv2.goodFeaturesToTrack(
            ref_uint8, maxCorners=100, qualityLevel=0.01, 
            minDistance=10, blockSize=3
        )
        
        if corners is not None and len(corners) > 5:
            # Calculate optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2, 
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                ref_uint8, curr_uint8, corners, None, **lk_params
            )
            
            # Select good points
            good_new = next_pts[status == 1]
            good_old = corners[status == 1]
            
            if len(good_new) > 5:
                # Calculate mean displacement
                displacement = good_new - good_old
                mean_displacement = np.mean(displacement, axis=0)
                
                # Apply transformation
                transformation_matrix = np.float32([[1, 0, mean_displacement[0]], 
                                                  [0, 1, mean_displacement[1]]])
                aligned_frame = cv2.warpAffine(curr_uint8, transformation_matrix, (width, height))
                
                # Convert back to float
                aligned_sequence[t] = aligned_frame.astype(np.float32) * np.max(current_frame) / 255
                
                shifts.append(-mean_displacement)  # Negative because we're correcting
                quality_score = len(good_new) / len(corners)
                quality_scores.append(quality_score)
            else:
                # Fallback to no alignment
                aligned_sequence[t] = current_frame
                shifts.append([0, 0])
                quality_scores.append(0.1)
        else:
            # Fallback to no alignment
            aligned_sequence[t] = current_frame
            shifts.append([0, 0])
            quality_scores.append(0.1)
    
    return aligned_sequence, shifts, quality_scores

def _feature_based_alignment(image_sequence: np.ndarray, 
                           reference: np.ndarray, 
                           reference_frame: int) -> Tuple[np.ndarray, list, list]:
    """Feature-based alignment using ORB descriptors"""
    
    t_frames, height, width = image_sequence.shape
    aligned_sequence = np.zeros_like(image_sequence)
    aligned_sequence[reference_frame] = reference
    
    shifts = []
    quality_scores = []
    
    # Convert reference to uint8
    ref_uint8 = (reference * 255 / np.max(reference)).astype(np.uint8)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Find keypoints and descriptors for reference
    kp1, des1 = orb.detectAndCompute(ref_uint8, None)
    
    if des1 is None:
        # Fallback to phase correlation if no features found
        return _phase_correlation_alignment(image_sequence, reference, reference_frame)
    
    # Initialize matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    for t in range(t_frames):
        if t == reference_frame:
            shifts.append([0, 0])
            quality_scores.append(1.0)
            continue
        
        current_frame = image_sequence[t].astype(np.float32)
        curr_uint8 = (current_frame * 255 / np.max(current_frame)).astype(np.uint8)
        
        # Find keypoints and descriptors for current frame
        kp2, des2 = orb.detectAndCompute(curr_uint8, None)
        
        if des2 is not None:
            # Match descriptors
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 10:
                # Extract matched points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Find transformation matrix using RANSAC
                transformation_matrix, mask = cv2.findHomography(
                    dst_pts, src_pts, cv2.RANSAC, 5.0
                )
                
                if transformation_matrix is not None:
                    # Apply transformation
                    aligned_frame = cv2.warpPerspective(curr_uint8, transformation_matrix, (width, height))
                    aligned_sequence[t] = aligned_frame.astype(np.float32) * np.max(current_frame) / 255
                    
                    # Extract translation components
                    translation = transformation_matrix[:2, 2]
                    shifts.append(translation)
                    
                    # Quality score based on inlier ratio
                    quality_score = np.sum(mask) / len(mask) if mask is not None else 0.1
                    quality_scores.append(quality_score)
                else:
                    # Fallback
                    aligned_sequence[t] = current_frame
                    shifts.append([0, 0])
                    quality_scores.append(0.1)
            else:
                # Fallback
                aligned_sequence[t] = current_frame
                shifts.append([0, 0])
                quality_scores.append(0.1)
        else:
            # Fallback
            aligned_sequence[t] = current_frame
            shifts.append([0, 0])
            quality_scores.append(0.1)
    
    return aligned_sequence, shifts, quality_scores

def detect_nuclear_regions(image: np.ndarray, 
                         min_area: int = 50,
                         max_area: int = 5000) -> Dict[str, Any]:
    """
    Detect nuclear regions for alignment validation
    
    Args:
        image: 2D nuclear image
        min_area: Minimum nuclear area in pixels
        max_area: Maximum nuclear area in pixels
        
    Returns:
        Dictionary with detected regions and properties
    """
    
    try:
        # Normalize image
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Apply Gaussian filter
        filtered = ndimage.gaussian_filter(normalized, sigma=1.0)
        
        # Threshold using Otsu's method
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(filtered)
        binary = filtered > threshold
        
        # Remove small objects and fill holes
        from skimage.morphology import remove_small_objects, remove_small_holes
        cleaned = remove_small_objects(binary, min_size=min_area)
        cleaned = remove_small_holes(cleaned, area_threshold=min_area // 2)
        
        # Label regions
        labeled = measure.label(cleaned)
        regions = measure.regionprops(labeled)
        
        # Filter by area
        valid_regions = [r for r in regions if min_area <= r.area <= max_area]
        
        # Extract centroids
        centroids = [r.centroid for r in valid_regions]
        areas = [r.area for r in valid_regions]
        
        return {
            'status': 'success',
            'num_nuclei': len(valid_regions),
            'centroids': centroids,
            'areas': areas,
            'binary_mask': cleaned,
            'labeled_image': labeled
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Nuclear detection failed: {str(e)}',
            'num_nuclei': 0,
            'centroids': [],
            'areas': []
        }