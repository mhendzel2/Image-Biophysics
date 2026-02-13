"""
Nuclear Biophysics Analysis Module
Advanced methods for nuclear diffusion, chromatin dynamics, and elasticity analysis
"""

import numpy as np
import pandas as pd
from scipy import optimize, ndimage
from typing import Dict, Any, Tuple, Optional, List
import warnings
from correlation_utils import acf_fft
from number_and_brightness import NumberAndBrightness
from image_correlation_spectroscopy import ImageCorrelationSpectroscopy

try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except ImportError:
    TRACKPY_AVAILABLE = False
    warnings.warn("trackpy not available - some nuclear tracking features disabled")

try:
    from skimage import measure, segmentation, filters, morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - nuclear segmentation limited")


class NuclearBiophysicsAnalyzer:
    """
    Advanced nuclear biophysics analysis for studying:
    - Nuclear diffusion and molecular binding
    - Chromatin dynamics and organization
    - Nuclear elasticity and mechanical properties
    """
    
    def __init__(self):
        self.name = "Nuclear Biophysics Analyzer"
        self.available_methods = self._check_available_methods()
        self._nb_analyzer = NumberAndBrightness()
        self._ics_analyzer = ImageCorrelationSpectroscopy()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which nuclear analysis methods are available"""
        return {
            'nuclear_fcs_binding': True,  # Always available with scipy
            'chromatin_dynamics': SKIMAGE_AVAILABLE,
            'nuclear_elasticity': TRACKPY_AVAILABLE and SKIMAGE_AVAILABLE,
            'nuclear_strain_mapping': SKIMAGE_AVAILABLE
        }
    
    def analyze_nuclear_binding(self, image_data: np.ndarray, 
                               nuclear_mask: np.ndarray,
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze nuclear diffusion and binding using masked FCS analysis
        
        Args:
            image_data: Time-lapse microscopy data (T, Y, X) or (T, Y, X, C)
            nuclear_mask: Binary mask defining nuclear region
            parameters: Analysis parameters including pixel_size, time_interval
        """
        
        try:
            pixel_size = parameters.get('pixel_size', 0.1)  # μm
            time_interval = parameters.get('time_interval', 0.1)  # seconds
            use_two_component = parameters.get('two_component_model', True)
            mode = parameters.get('mode', 'roi_mean')
            intensity_weighted = bool(parameters.get('intensity_weighted', False))
            max_lag_frames = parameters.get('max_lag_frames', None)
            
            # Apply nuclear mask to image data
            masked_data = self._apply_nuclear_mask(image_data, nuclear_mask)
            
            # Calculate correlation functions within nucleus
            correlation_results = self._calculate_nuclear_correlations(
                masked_data=masked_data,
                nuclear_mask=nuclear_mask,
                pixel_size=pixel_size,
                time_interval=time_interval,
                use_two_component=use_two_component,
                mode=mode,
                intensity_weighted=intensity_weighted,
                max_lag_frames=max_lag_frames
            )
            
            # Analyze binding kinetics
            binding_analysis = self._analyze_binding_kinetics(correlation_results)
            
            # Generate nuclear diffusion maps
            diffusion_maps = self._generate_nuclear_diffusion_maps(
                correlation_results, nuclear_mask
            )
            
            return {
                'status': 'success',
                'method': 'Nuclear FCS Binding Analysis',
                'correlation_results': correlation_results,
                'binding_kinetics': binding_analysis,
                'diffusion_maps': diffusion_maps,
                'nuclear_mask': nuclear_mask,
                'experimental': True,
                'validation_note': 'Two-component temporal correlation fit is experimental and should not be interpreted as a direct kinetic binding model without additional validation.',
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Nuclear binding analysis failed: {str(e)}'}
    
    def analyze_chromatin_dynamics(self, image_data: np.ndarray,
                                 nuclear_mask: np.ndarray,
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze chromatin dynamics using N&B, iMSD, and texture analysis
        
        Args:
            image_data: Time-lapse data of fluorescently tagged histones
            nuclear_mask: Nuclear region mask
            parameters: Analysis parameters
        """
        
        if not SKIMAGE_AVAILABLE:
            return {'status': 'error', 'message': 'scikit-image required for chromatin dynamics analysis'}
        
        try:
            # Apply nuclear mask
            masked_data = self._apply_nuclear_mask(image_data, nuclear_mask)
            
            # Number & Brightness analysis for oligomerization
            nb_results = self._calculate_nuclear_nb(masked_data, nuclear_mask)
            
            # iMSD analysis for chromatin mobility
            imsd_results = self._calculate_nuclear_imsd(masked_data, nuclear_mask, parameters)
            
            # Texture analysis for chromatin organization
            texture_results = self._analyze_chromatin_texture(masked_data)
            
            # Combine results
            chromatin_state = self._classify_chromatin_state(
                nb_results, imsd_results, texture_results
            )
            
            return {
                'status': 'success',
                'method': 'Chromatin Dynamics Analysis',
                'nb_analysis': nb_results,
                'imsd_analysis': imsd_results,
                'texture_analysis': texture_results,
                'chromatin_state': chromatin_state,
                'nuclear_mask': nuclear_mask,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Chromatin dynamics analysis failed: {str(e)}'}
    
    def analyze_nuclear_elasticity(self, image_data: np.ndarray,
                                 force_application_time: int,
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze nuclear elasticity from deformation under applied force
        
        Args:
            image_data: Time-lapse data during force application
            force_application_time: Frame number when force was applied
            parameters: Analysis parameters including force magnitude
        """
        
        if not (TRACKPY_AVAILABLE and SKIMAGE_AVAILABLE):
            return {'status': 'error', 'message': 'trackpy and scikit-image required for elasticity analysis'}
        
        try:
            # Segment nuclei in each frame
            nuclear_masks = self._segment_nuclei_timeseries(image_data)
            
            # Track nuclear displacement
            displacement_results = self._track_nuclear_displacement(
                nuclear_masks, force_application_time
            )
            
            # Calculate strain fields
            strain_results = self._calculate_nuclear_strain_fields(
                displacement_results, parameters
            )
            
            # Measure elastic modulus
            elasticity_metrics = self._calculate_nuclear_elasticity(
                strain_results, parameters
            )
            
            return {
                'status': 'success',
                'method': 'Nuclear Elasticity Analysis',
                'displacement_analysis': displacement_results,
                'strain_analysis': strain_results,
                'elasticity_metrics': elasticity_metrics,
                'nuclear_masks': nuclear_masks,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Nuclear elasticity analysis failed: {str(e)}'}
    
    def _apply_nuclear_mask(self, image_data: np.ndarray, 
                           nuclear_mask: np.ndarray) -> np.ndarray:
        """Apply nuclear mask to time-lapse data"""
        
        if image_data.ndim == 3:  # T, Y, X
            masked_data = image_data.copy()
            for t in range(image_data.shape[0]):
                masked_data[t][~nuclear_mask] = 0
        elif image_data.ndim == 4:  # T, Y, X, C
            masked_data = image_data.copy()
            for t in range(image_data.shape[0]):
                for c in range(image_data.shape[3]):
                    masked_data[t, :, :, c][~nuclear_mask] = 0
        else:
            raise ValueError("Image data must be 3D (T,Y,X) or 4D (T,Y,X,C)")
        
        return masked_data
    
    def _calculate_nuclear_correlations(self, masked_data: np.ndarray,
                                      nuclear_mask: np.ndarray,
                                      pixel_size: float,
                                      time_interval: float,
                                      use_two_component: bool,
                                      mode: str = "roi_mean",
                                      intensity_weighted: bool = False,
                                      max_lag_frames: Optional[int] = None) -> Dict[str, Any]:
        """Calculate temporal correlations within nuclear mask.

        Modes
        -----
        roi_mean:
            Single trace from ROI mean intensity over time.
        pixelwise_mean_acf:
            ACF per pixel trace, then mean ACF across pixels.
        pixelwise_fit_map:
            Pixel-wise ACF and per-pixel fit maps (slow/experimental).
        """
        if masked_data.ndim == 4:
            data_tyx = masked_data[..., 0]
        elif masked_data.ndim == 3:
            data_tyx = masked_data
        else:
            raise ValueError("Data must be 3D (T,Y,X) or 4D (T,Y,X,C)")

        if nuclear_mask.shape != data_tyx.shape[1:]:
            raise ValueError("nuclear_mask shape must match spatial dimensions of image data")

        coords = np.argwhere(nuclear_mask)
        if coords.size == 0:
            raise ValueError("nuclear_mask contains no True pixels")

        T = data_tyx.shape[0]
        if T < 8:
            raise ValueError("Insufficient frames for temporal correlation analysis")

        y_idx = coords[:, 0]
        x_idx = coords[:, 1]
        traces = data_tyx[:, y_idx, x_idx].T  # (n_pixels, T)
        trace_means = traces.mean(axis=1)
        valid_trace_mask = np.isfinite(trace_means) & (trace_means > 0)
        traces = traces[valid_trace_mask]

        if traces.shape[0] == 0:
            raise ValueError("No valid masked pixel traces with positive mean intensity")

        max_lag = max(4, min(T // 4, 100))
        if max_lag_frames is not None:
            max_lag = int(max(2, min(max_lag, max_lag_frames)))

        tau_s = np.arange(max_lag) * time_interval

        mode = str(mode).lower()
        correlation_curves: List[Tuple[np.ndarray, np.ndarray]] = []
        fitting_results: List[Dict[str, Any]] = []
        per_pixel_fit_map: Optional[Dict[str, np.ndarray]] = None

        if mode == "roi_mean":
            roi_trace = data_tyx[:, nuclear_mask].mean(axis=1)
            _, G = self._calculate_autocorrelation(roi_trace, time_interval)
            G = G[:max_lag]
            correlation_curves.append((tau_s, G))
            if use_two_component:
                fit_result = self._fit_two_component_fcs(tau_s, G, pixel_size, time_interval)
            else:
                fit_result = self._fit_single_component_fcs(tau_s, G, pixel_size, time_interval)
            if fit_result.get('status') == 'success':
                fitting_results.append(fit_result)

        elif mode in ("pixelwise_mean_acf", "pixelwise_fit_map"):
            acfs = []
            weights = []
            for tr in traces:
                _, G = self._calculate_autocorrelation(tr, time_interval)
                G = G[:max_lag]
                if np.all(np.isfinite(G)):
                    acfs.append(G)
                    weights.append(float(np.mean(tr)))

            if not acfs:
                raise ValueError("No valid ACF traces in masked region")

            acf_arr = np.vstack(acfs)
            if intensity_weighted:
                w = np.asarray(weights, dtype=float)
                w = np.clip(w, 0, np.inf)
                if np.sum(w) <= 0:
                    w = np.ones_like(w)
                G_mean = np.average(acf_arr, axis=0, weights=w)
            else:
                G_mean = acf_arr.mean(axis=0)

            correlation_curves.append((tau_s, G_mean))
            if use_two_component:
                fit_result = self._fit_two_component_fcs(tau_s, G_mean, pixel_size, time_interval)
            else:
                fit_result = self._fit_single_component_fcs(tau_s, G_mean, pixel_size, time_interval)
            if fit_result.get('status') == 'success':
                fitting_results.append(fit_result)

            if mode == "pixelwise_fit_map":
                H, W = data_tyx.shape[1:]
                tau_char_map = np.full((H, W), np.nan, dtype=float)
                D_char_map = np.full((H, W), np.nan, dtype=float)
                for idx, tr in enumerate(traces):
                    _, G = self._calculate_autocorrelation(tr, time_interval)
                    G = G[:max_lag]
                    fr = self._fit_single_component_fcs(tau_s, G, pixel_size, time_interval)
                    if fr.get('status') == 'success':
                        y = int(coords[valid_trace_mask][idx, 0])
                        x = int(coords[valid_trace_mask][idx, 1])
                        tau_char_map[y, x] = fr.get('tau_char_s', np.nan)
                        D_char_map[y, x] = fr.get('diffusion_coefficient_um2_s', np.nan)
                per_pixel_fit_map = {
                    'tau_char_s_map': tau_char_map,
                    'diffusion_coefficient_um2_s_map': D_char_map
                }
        else:
            raise ValueError(f"Unknown correlation mode: {mode}")

        G_mean = correlation_curves[0][1]
        qc = {
            'fraction_masked_pixels_used': float(traces.shape[0] / max(1, coords.shape[0])),
            'mean_intensity': float(np.mean(data_tyx[:, nuclear_mask])),
            'drift_metric_cv_roi_mean': float(
                np.std(data_tyx[:, nuclear_mask].mean(axis=1)) / (np.mean(data_tyx[:, nuclear_mask].mean(axis=1)) + 1e-12)
            ),
            'mode': mode,
            'intensity_weighted': bool(intensity_weighted)
        }

        out = {
            'tau_s': tau_s,
            'G_mean': G_mean,
            'n_traces_used': int(1 if mode == 'roi_mean' else traces.shape[0]),
            'correlation_curves': correlation_curves,
            'fitting_results': fitting_results,
            'average_results': self._average_fcs_results(fitting_results),
            'qc': qc
        }
        if per_pixel_fit_map is not None:
            out['pixelwise_fit_maps'] = per_pixel_fit_map
        return out
    
    def _fit_two_component_fcs(self, tau: np.ndarray, correlation: np.ndarray,
                              pixel_size: float, time_interval: float) -> Dict[str, Any]:
        """
        Fit two-component FCS model for bound/free analysis
        G(τ) = (1/N) * [F_free/(1 + τ/τ_free) + (1-F_free)/(1 + τ/τ_bound)]
        """
        
        try:
            def fcs_model_two_component(tau, N, F_free, tau_free, tau_bound):
                free_component = F_free / (1.0 + tau / tau_free)
                bound_component = (1.0 - F_free) / (1.0 + tau / tau_bound)
                return (1.0 / N) * (free_component + bound_component)
            
            # Initial parameter guesses
            N_guess = 1.0 / np.max(correlation)
            F_free_guess = 0.7  # 70% free fraction
            tau_free_guess = tau[np.argmax(np.gradient(correlation))] * 0.5
            tau_bound_guess = tau_free_guess * 10.0
            
            # Parameter bounds: [N, F_free, tau_free, tau_bound]
            bounds = ([0.1, 0.0, 0.001, 0.01], [1000.0, 1.0, 5.0, 50.0])
            
            # Fit the model
            popt, pcov = optimize.curve_fit(
                fcs_model_two_component, tau, correlation,
                p0=[N_guess, F_free_guess, tau_free_guess, tau_bound_guess],
                bounds=bounds,
                maxfev=2000
            )
            
            N_fitted, F_free_fitted, tau_free_fitted, tau_bound_fitted = popt
            
            # Calculate diffusion coefficients
            w0 = pixel_size  # Confocal volume waist
            D_free = (w0**2) / (4 * tau_free_fitted * time_interval)
            D_bound = (w0**2) / (4 * tau_bound_fitted * time_interval)
            
            return {
                'status': 'success',
                'model_type': 'two_component',
                'N': N_fitted,
                'fraction_fast_component': F_free_fitted,
                'fraction_slow_component': 1.0 - F_free_fitted,
                'diffusion_fast_um2_s': D_free,
                'diffusion_slow_um2_s': D_bound,
                'tau_fast_s': tau_free_fitted,
                'tau_slow_s': tau_bound_fitted,
                'tau_char_s': tau_free_fitted,
                'experimental': True,
                'validation_note': 'Two-component diffusion fit is not a directly identifiable binding kinetics model.',
                'parameter_errors': np.sqrt(np.diag(pcov)),
                'fitted_curve': fcs_model_two_component(tau, *popt)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _fit_single_component_fcs(self, tau: np.ndarray, correlation: np.ndarray,
                                 pixel_size: float, time_interval: float) -> Dict[str, Any]:
        """Fit single-component FCS model"""
        
        try:
            def fcs_model(tau, N, tau_diff):
                return (1.0 / N) * (1.0 / (1.0 + tau / tau_diff))
            
            N_guess = 1.0 / np.max(correlation)
            tau_diff_guess = tau[np.argmax(np.gradient(correlation))]
            
            bounds = ([0.1, 0.001], [1000.0, 10.0])
            
            popt, pcov = optimize.curve_fit(
                fcs_model, tau, correlation,
                p0=[N_guess, tau_diff_guess],
                bounds=bounds,
                maxfev=1000
            )
            
            N_fitted, tau_diff_fitted = popt
            w0 = pixel_size
            D = (w0**2) / (4 * tau_diff_fitted * time_interval)
            
            return {
                'status': 'success',
                'model_type': 'single_component',
                'N': N_fitted,
                'diffusion_coefficient_um2_s': D,
                'tau_char_s': tau_diff_fitted,
                'experimental': True,
                'parameter_errors': np.sqrt(np.diag(pcov)),
                'fitted_curve': fcs_model(tau, *popt)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_autocorrelation(self, trace: np.ndarray, time_interval: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate FCS-style ACF: <deltaI(t)deltaI(t+tau)> / <I>^2."""
        n = len(trace)
        max_lag = max(4, min(n // 4, 100))
        correlation = acf_fft(np.asarray(trace, dtype=float))[:max_lag]
        tau = np.arange(max_lag) * time_interval
        return tau, correlation
    
    def _calculate_nuclear_nb(self, masked_data: np.ndarray, nuclear_mask: np.ndarray) -> Dict[str, Any]:
        """Calculate Number & Brightness for nuclear region using shared analyzer."""
        if masked_data.ndim != 3:
            raise ValueError("N&B analysis requires 3D data (T,Y,X)")

        nb = self._nb_analyzer.analyze(masked_data, roi_mask=nuclear_mask)
        if nb.get('status') != 'success':
            raise ValueError(nb.get('message', 'N&B failed'))

        brightness_map = np.where(nuclear_mask, nb['brightness_map'], np.nan)
        number_map = np.where(nuclear_mask, nb['number_map'], np.nan)

        return {
            **nb,
            'brightness_map': brightness_map,
            'number_map': number_map,
            'mean_brightness': float(np.nanmean(brightness_map)) if np.any(nuclear_mask) else 0.0,
            'mean_number': float(np.nanmean(number_map)) if np.any(nuclear_mask) else 0.0,
            'oligomerization_index': float(np.nanmean(brightness_map)) if np.any(nuclear_mask) else 0.0,
            'experimental': True
        }
    
    def _calculate_nuclear_imsd(self, masked_data: np.ndarray,
                               nuclear_mask: np.ndarray,
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate nuclear iMSD via STICS Gaussian-variance approach."""
        if masked_data.ndim != 3:
            raise ValueError("iMSD analysis requires 3D data (T,Y,X)")

        max_temporal_lag = int(parameters.get('max_temporal_lag', min(masked_data.shape[0] // 4, 20)))
        max_spatial_lag = int(parameters.get('max_spatial_lag', 6))
        fit_radius = int(parameters.get('imsd_fit_radius', max_spatial_lag))
        pixel_size = float(parameters.get('pixel_size', 0.1))
        time_interval = float(parameters.get('time_interval', 0.1))

        imsd = self._ics_analyzer.compute_imsd_stics(
            image_stack=masked_data,
            mask=nuclear_mask,
            max_temporal_lag=max_temporal_lag,
            max_spatial_lag=max_spatial_lag,
            pixel_size_um=pixel_size,
            time_interval_s=time_interval,
            fit_radius=fit_radius
        )
        imsd['experimental'] = True
        return imsd
    
    def _analyze_chromatin_texture(self, masked_data: np.ndarray) -> Dict[str, Any]:
        """Analyze chromatin texture using Fourier analysis"""
        
        if masked_data.ndim == 3:
            # Average over time for texture analysis
            mean_image = np.mean(masked_data, axis=0)
        else:
            mean_image = masked_data
        
        # Apply 2D FFT
        fft_image = np.fft.fft2(mean_image)
        power_spectrum = np.abs(fft_image)**2
        
        # Calculate radial profile
        center = np.array(power_spectrum.shape) // 2
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        max_radius = min(center[0], center[1])
        radial_profile = []
        
        for radius in range(1, max_radius):
            mask = (r >= radius - 0.5) & (r < radius + 0.5)
            if np.any(mask):
                radial_profile.append(np.mean(power_spectrum[mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # Find dominant frequency
        if len(radial_profile) > 0:
            dominant_freq_idx = np.argmax(radial_profile[1:]) + 1  # Skip DC component
            dominant_wavelength = mean_image.shape[0] / dominant_freq_idx if dominant_freq_idx > 0 else 0
        else:
            dominant_wavelength = 0
        
        return {
            'power_spectrum': power_spectrum,
            'radial_profile': radial_profile,
            'dominant_wavelength': dominant_wavelength,
            'texture_entropy': -np.sum(radial_profile * np.log(radial_profile + 1e-10))
        }
    
    def _classify_chromatin_state(self, nb_results: Dict[str, Any],
                                imsd_results: Dict[str, Any],
                                texture_results: Dict[str, Any]) -> Dict[str, Any]:
        """Classify chromatin state based on combined analysis"""
        
        # Thresholds for classification (these would be calibrated experimentally)
        brightness_threshold = 1.5  # Higher brightness indicates oligomerization
        diffusion_threshold = 0.01  # Lower diffusion indicates confinement
        alpha_threshold = 0.8  # Lower alpha indicates subdiffusion/confinement
        
        brightness = nb_results.get('mean_brightness', 0)
        diffusion = imsd_results.get('mean_diffusion', 0)
        alpha = imsd_results.get('mean_anomalous_exponent', 1)
        
        # Classification logic
        if brightness > brightness_threshold and diffusion < diffusion_threshold and alpha < alpha_threshold:
            state = 'Heterochromatin (Condensed)'
            confidence = 'High'
        elif brightness < brightness_threshold and diffusion > diffusion_threshold and alpha > 0.9:
            state = 'Euchromatin (Open)'
            confidence = 'High'
        else:
            state = 'Intermediate'
            confidence = 'Medium'
        
        return {
            'chromatin_state': state,
            'confidence': confidence,
            'brightness_score': brightness,
            'mobility_score': diffusion,
            'confinement_score': 1 - alpha,
            'condensation_index': brightness * (1 - alpha) / (diffusion + 1e-10)
        }
    
    def _segment_nuclei_timeseries(self, image_data: np.ndarray) -> List[np.ndarray]:
        """Segment nuclei in each frame of time-lapse data"""
        
        nuclear_masks = []
        
        for t in range(image_data.shape[0]):
            frame = image_data[t] if image_data.ndim == 3 else image_data[t, :, :, 0]
            
            # Simple threshold-based segmentation
            # In practice, you might use more sophisticated methods
            threshold = filters.threshold_otsu(frame)
            binary_mask = frame > threshold
            
            # Clean up mask
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=100)
            binary_mask = morphology.closing(binary_mask, morphology.disk(3))
            
            nuclear_masks.append(binary_mask)
        
        return nuclear_masks
    
    def _track_nuclear_displacement(self, nuclear_masks: List[np.ndarray],
                                  force_application_time: int) -> Dict[str, Any]:
        """Track nuclear displacement before and after force application"""
        
        if not TRACKPY_AVAILABLE:
            raise ImportError("trackpy required for nuclear displacement tracking")
        
        # Calculate nuclear centroids for each frame
        centroids = []
        
        for mask in nuclear_masks:
            labeled_mask = measure.label(mask)
            props = measure.regionprops(labeled_mask)
            
            frame_centroids = []
            for prop in props:
                frame_centroids.append([prop.centroid[1], prop.centroid[0]])  # x, y
            
            centroids.append(frame_centroids)
        
        # Convert to trackpy-compatible format
        if centroids:
            all_spots = []
            for t, frame_centroids in enumerate(centroids):
                for centroid in frame_centroids:
                    all_spots.append({
                        'x': centroid[0],
                        'y': centroid[1],
                        'frame': t
                    })
            
            if all_spots:
                spots_df = pd.DataFrame(all_spots)
                
                # Link trajectories
                trajectories = tp.link_df(spots_df, search_range=50, memory=3)
                
                # Calculate displacement vectors
                displacement_vectors = self._calculate_displacement_vectors(
                    trajectories, force_application_time
                )
                
                return {
                    'trajectories': trajectories,
                    'displacement_vectors': displacement_vectors,
                    'centroids': centroids
                }
        
        return {'trajectories': None, 'displacement_vectors': None, 'centroids': centroids}
    
    def _calculate_displacement_vectors(self, trajectories: pd.DataFrame,
                                      force_time: int) -> Dict[str, Any]:
        """Calculate displacement vectors before and after force application"""
        
        displacement_data = {}
        
        for particle_id in trajectories['particle'].unique():
            particle_traj = trajectories[trajectories['particle'] == particle_id]
            
            # Get positions before and after force application
            before_force = particle_traj[particle_traj['frame'] < force_time]
            after_force = particle_traj[particle_traj['frame'] >= force_time]
            
            if len(before_force) > 0 and len(after_force) > 0:
                # Calculate average position before and after
                pos_before = [float(before_force['x'].mean()), float(before_force['y'].mean())]
                pos_after = [float(after_force['x'].mean()), float(after_force['y'].mean())]
                
                # Calculate displacement vector
                displacement = [pos_after[0] - pos_before[0], pos_after[1] - pos_before[1]]
                magnitude = np.sqrt(displacement[0]**2 + displacement[1]**2)
                
                displacement_data[str(particle_id)] = {
                    'displacement_x': float(displacement[0]),
                    'displacement_y': float(displacement[1]),
                    'magnitude': float(magnitude),
                    'position_before': pos_before,
                    'position_after': pos_after
                }
        
        return displacement_data
    
    def _calculate_nuclear_strain_fields(self, displacement_results: Dict[str, Any],
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate strain fields from displacement vectors"""
        
        displacement_vectors = displacement_results.get('displacement_vectors')
        if not displacement_vectors:
            return {'strain_field': None, 'strain_magnitude': 0}
        
        # Create displacement field grid
        positions = []
        displacements = []
        
        for particle_id, data in displacement_vectors.items():
            positions.append(data['position_before'])
            displacements.append([data['displacement_x'], data['displacement_y']])
        
        if len(positions) < 3:
            return {'strain_field': None, 'strain_magnitude': 0}
        
        positions = np.array(positions)
        displacements = np.array(displacements)
        
        # Calculate strain tensor components using finite differences
        # This is a simplified approach - more sophisticated methods could be used
        try:
            # Calculate gradients of displacement field
            dudx = np.gradient(displacements[:, 0])[0] if len(displacements) > 1 else 0
            dudy = np.gradient(displacements[:, 0])[0] if len(displacements) > 1 else 0
            dvdx = np.gradient(displacements[:, 1])[0] if len(displacements) > 1 else 0
            dvdy = np.gradient(displacements[:, 1])[0] if len(displacements) > 1 else 0
            
            # Strain tensor components
            strain_xx = dudx
            strain_yy = dvdy
            strain_xy = 0.5 * (dudy + dvdx)
            
            # Principal strains
            strain_trace = strain_xx + strain_yy
            strain_det = strain_xx * strain_yy - strain_xy**2
            
            principal_strain_1 = 0.5 * (strain_trace + np.sqrt(strain_trace**2 - 4*strain_det))
            principal_strain_2 = 0.5 * (strain_trace - np.sqrt(strain_trace**2 - 4*strain_det))
            
            strain_magnitude = np.sqrt(strain_xx**2 + strain_yy**2 + 2*strain_xy**2)
            
            return {
                'strain_xx': strain_xx,
                'strain_yy': strain_yy,
                'strain_xy': strain_xy,
                'principal_strain_1': principal_strain_1,
                'principal_strain_2': principal_strain_2,
                'strain_magnitude': strain_magnitude,
                'volumetric_strain': strain_trace
            }
            
        except Exception as e:
            return {'strain_field': None, 'strain_magnitude': 0, 'error': str(e)}
    
    def _calculate_nuclear_elasticity(self, strain_results: Dict[str, Any],
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate nuclear elastic modulus from strain analysis"""
        
        applied_force = parameters.get('applied_force', 1.0)  # pN
        nuclear_area = parameters.get('nuclear_area', 100.0)  # μm²
        
        strain_magnitude = strain_results.get('strain_magnitude', 0)
        
        if strain_magnitude > 0:
            # Calculate stress (force per unit area)
            stress = applied_force / nuclear_area  # pN/μm²
            
            # Calculate Young's modulus (stress/strain)
            youngs_modulus = stress / strain_magnitude  # pN/μm²
            
            # Convert to more standard units (Pa)
            youngs_modulus_pa = youngs_modulus * 1e-6  # Pa
            
            return {
                'youngs_modulus': youngs_modulus_pa,
                'strain_magnitude': strain_magnitude,
                'applied_stress': stress,
                'deformation_energy': 0.5 * youngs_modulus * strain_magnitude**2,
                'elasticity_classification': self._classify_nuclear_elasticity(youngs_modulus_pa)
            }
        else:
            return {
                'youngs_modulus': 0,
                'strain_magnitude': 0,
                'applied_stress': 0,
                'deformation_energy': 0,
                'elasticity_classification': 'Unable to determine'
            }
    
    def _classify_nuclear_elasticity(self, youngs_modulus: float) -> str:
        """Classify nuclear elasticity based on Young's modulus"""
        
        # Typical ranges for nuclear elasticity (these are approximate)
        if youngs_modulus < 100:
            return 'Very Soft'
        elif youngs_modulus < 500:
            return 'Soft'
        elif youngs_modulus < 2000:
            return 'Normal'
        elif youngs_modulus < 5000:
            return 'Stiff'
        else:
            return 'Very Stiff'
    
    def _average_fcs_results(self, fitting_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average FCS fitting results across multiple correlation curves"""
        
        if not fitting_results:
            return {'status': 'error', 'message': 'No successful fits'}
        
        successful_fits = [result for result in fitting_results if result['status'] == 'success']
        
        if not successful_fits:
            return {'status': 'error', 'message': 'No successful fits'}
        
        # Separate by model type
        two_component_fits = [fit for fit in successful_fits if fit.get('model_type') == 'two_component']
        single_component_fits = [fit for fit in successful_fits if fit.get('model_type') == 'single_component']
        
        averaged_results = {'status': 'success'}
        
        if two_component_fits:
            averaged_results['two_component'] = {
                'mean_fraction_fast_component': np.mean([fit['fraction_fast_component'] for fit in two_component_fits]),
                'mean_fraction_slow_component': np.mean([fit['fraction_slow_component'] for fit in two_component_fits]),
                'mean_diffusion_fast_um2_s': np.mean([fit['diffusion_fast_um2_s'] for fit in two_component_fits]),
                'mean_diffusion_slow_um2_s': np.mean([fit['diffusion_slow_um2_s'] for fit in two_component_fits]),
                'mean_tau_fast_s': np.mean([fit['tau_fast_s'] for fit in two_component_fits]),
                'mean_tau_slow_s': np.mean([fit['tau_slow_s'] for fit in two_component_fits]),
                'std_fraction_fast_component': np.std([fit['fraction_fast_component'] for fit in two_component_fits]),
                'n_fits': len(two_component_fits)
            }
        
        if single_component_fits:
            averaged_results['single_component'] = {
                'mean_diffusion_coefficient_um2_s': np.mean([fit['diffusion_coefficient_um2_s'] for fit in single_component_fits]),
                'std_diffusion_coefficient_um2_s': np.std([fit['diffusion_coefficient_um2_s'] for fit in single_component_fits]),
                'mean_tau_char_s': np.mean([fit['tau_char_s'] for fit in single_component_fits]),
                'n_fits': len(single_component_fits)
            }
        
        return averaged_results
    
    def _analyze_binding_kinetics(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize characteristic times from temporal correlation fits.

        Note: this does not provide a validated reaction-kinetics binding estimate.
        """
        
        average_results = correlation_results.get('average_results', {})
        
        if 'two_component' in average_results:
            two_comp = average_results['two_component']

            frac_fast = two_comp['mean_fraction_fast_component']
            frac_slow = two_comp['mean_fraction_slow_component']
            D_fast = two_comp['mean_diffusion_fast_um2_s']
            D_slow = two_comp['mean_diffusion_slow_um2_s']

            return {
                'fraction_fast_component': frac_fast,
                'fraction_slow_component': frac_slow,
                'diffusion_fast_um2_s': D_fast,
                'diffusion_slow_um2_s': D_slow,
                'tau_fast_s': two_comp.get('mean_tau_fast_s', np.nan),
                'tau_slow_s': two_comp.get('mean_tau_slow_s', np.nan),
                'characteristic_correlation_time_s': two_comp.get('mean_tau_fast_s', np.nan),
                'experimental': True,
                'validation_note': 'Do not interpret as direct binding kinetics without a validated reaction-diffusion model and identifiability analysis.'
            }
        if 'single_component' in average_results:
            single = average_results['single_component']
            return {
                'diffusion_coefficient_um2_s': single.get('mean_diffusion_coefficient_um2_s', np.nan),
                'characteristic_correlation_time_s': single.get('mean_tau_char_s', np.nan),
                'experimental': True,
                'validation_note': 'Single-component fit reports characteristic temporal correlation only.'
            }
        else:
            return {'binding_analysis': 'No successful temporal correlation fits'}
    
    def _generate_nuclear_diffusion_maps(self, correlation_results: Dict[str, Any],
                                       nuclear_mask: np.ndarray) -> Dict[str, Any]:
        """Generate spatial maps of diffusion parameters within nucleus"""
        if 'pixelwise_fit_maps' in correlation_results:
            maps = correlation_results['pixelwise_fit_maps']
            return {
                'diffusion_coefficient_um2_s_map': maps.get('diffusion_coefficient_um2_s_map'),
                'tau_char_s_map': maps.get('tau_char_s_map'),
                'nuclear_mask': nuclear_mask,
                'experimental': True
            }

        return {
            'nuclear_mask': nuclear_mask,
            'maps_available': False,
            'deprecation_note': 'Spatial maps are returned only in mode="pixelwise_fit_map". Constant placeholder maps were removed.',
            'experimental': True
        }


def get_nuclear_analysis_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for nuclear biophysics analysis methods"""
    
    if method == 'nuclear_binding':
        return {
            'pixel_size': 0.1,  # μm
            'time_interval': 0.1,  # seconds
            'two_component_model': True,
            'mode': 'roi_mean',  # roi_mean | pixelwise_mean_acf | pixelwise_fit_map
            'intensity_weighted': False,
            'correlation_window_size': 5,  # pixels
            'max_lag_time': 10.0  # seconds
        }
    elif method == 'chromatin_dynamics':
        return {
            'pixel_size': 0.1,  # μm
            'time_interval': 1.0,  # seconds
            'msd_max_lag': 20,  # frames
            'texture_analysis': True,
            'nb_analysis': True
        }
    elif method == 'nuclear_elasticity':
        return {
            'applied_force': 1.0,  # pN
            'nuclear_area': 100.0,  # μm²
            'force_application_frame': 10,
            'tracking_search_range': 50,  # pixels
            'memory': 3  # frames
        }
    else:
        return {
            'pixel_size': 0.1,
            'time_interval': 0.1
        }