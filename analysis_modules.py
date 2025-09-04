
def _calculate_spatial_autocorrelation(self, image_stack: np.ndarray, tau_max: int) -> np.ndarray:
    """Calculate proper RICS spatial autocorrelation using FFT-based method"""
    
    t_frames, height, width = image_stack.shape
    
    # Calculate intensity fluctuations: delta_I = I - <I>_t
    mean_intensity = np.mean(image_stack, axis=0)
    fluctuations = image_stack - mean_intensity[np.newaxis, :, :]
    
    # Calculate spatial autocorrelation for each frame and average
    autocorr_sum = np.zeros((height, width))
    
    for t in range(t_frames):
        delta_I = fluctuations[t]
        
        # FFT-based 2D autocorrelation: G(dx,dy) = IFFT(FFT(delta_I) * conj(FFT(delta_I)))
        fft_image = np.fft.fft2(delta_I)
        autocorr_2d = np.fft.ifft2(fft_image * np.conj(fft_image)).real
        
        # Shift zero frequency to center and normalize
        autocorr_centered = np.fft.fftshift(autocorr_2d)
        autocorr_sum += autocorr_centered
    
    # Average over time
    avg_autocorr = autocorr_sum / t_frames
    
    # Normalize by zero-lag value
    center_y, center_x = height // 2, width // 2
    zero_lag_value = avg_autocorr[center_y, center_x]
    
    if zero_lag_value > 0:
        avg_autocorr = avg_autocorr / zero_lag_value
    
    return avg_autocorr
