"""
Material Mechanics Analysis Module

Continuum mechanics and fluctuation-based analysis for standard fluorescence
microscopy time series without particle tracking.

Implements:
- Chromatin Image Velocimetry (CIV) via dense optical flow
- Internal stress maps from divergence and shear strain-rate tensors
- Displacement correlation spectroscopy stiffness proxy (correlation length xi)
- Texture topology proxies (GLCM entropy, box-counting fractal dimension)
- Compartment fusion kinetics for viscosity proxy (tau from AR relaxation)
- Nuclear envelope fluctuation spectroscopy for sigma/kappa estimation
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, optimize
from scipy.fft import fft2, fftshift, ifft2

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available - dense optical flow features disabled")

try:
    from skimage.feature import graycomatrix
    from skimage.filters import gaussian, threshold_otsu
    from skimage.measure import find_contours
    from skimage.segmentation import active_contour

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - some texture/boundary features limited")


class MaterialMechanics:
    """
    Continuum mechanics analyzer for nucleus/chromatin image series.
    """

    def __init__(self, pixel_size_um: float = 0.1, time_interval_s: float = 1.0, temperature_k: float = 310.0):
        self.pixel_size_um = float(pixel_size_um)
        self.time_interval_s = float(time_interval_s)
        self.temperature_k = float(temperature_k)
        self.name = "Material Mechanics"
        self.available = True

    def analyze(
        self,
        image_data: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        compartment_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run selected mechanics analyses on a time-lapse stack (T, Y, X).
        """
        params = parameters or {}
        try:
            stack = self._prepare_stack(image_data)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        run_force = bool(params.get("run_force_distribution", True))
        run_stiffness = bool(params.get("run_stiffness_proxy", True))
        run_texture = bool(params.get("run_texture_topology", True))
        run_fusion = bool(params.get("run_fusion_kinetics", False))
        run_boundary = bool(params.get("run_boundary_mechanics", True))

        component_results: Dict[str, Dict[str, Any]] = {}
        flow_result: Optional[Dict[str, Any]] = None

        if run_force or run_stiffness:
            flow_result = self.compute_internal_stress(stack, nuclear_mask=nuclear_mask, parameters=params)
            if run_force:
                component_results["force_distribution"] = flow_result

        if run_stiffness:
            if flow_result is None or flow_result.get("status") != "success":
                component_results["stiffness_proxy"] = {
                    "status": "error",
                    "message": "Stiffness proxy requires successful optical flow.",
                }
            else:
                component_results["stiffness_proxy"] = self.estimate_stiffness_proxy(
                    flow_result["velocity_fields"]["vx"],
                    flow_result["velocity_fields"]["vy"],
                    nuclear_mask=nuclear_mask,
                    parameters=params,
                )

        if run_texture:
            component_results["texture_topology"] = self.analyze_texture_topology(
                stack, nuclear_mask=nuclear_mask, parameters=params
            )

        if run_fusion:
            component_results["fusion_kinetics"] = self.analyze_compartment_fusion(
                image_stack=stack, compartment_mask=compartment_mask, parameters=params
            )

        if run_boundary:
            component_results["boundary_mechanics"] = self.analyze_envelope_mechanics(
                image_stack=stack, nuclear_mask=nuclear_mask, parameters=params
            )

        successful = [k for k, v in component_results.items() if v.get("status") == "success"]
        failed = [k for k, v in component_results.items() if v.get("status") == "error"]
        warnings_list = [k for k, v in component_results.items() if v.get("status") == "warning"]

        if not component_results:
            return {"status": "error", "message": "No analysis components were selected."}

        overall_status = "success" if successful else "error"
        return {
            "status": overall_status,
            "method": "Material Mechanics",
            "results": component_results,
            "summary": {
                "successful_components": successful,
                "failed_components": failed,
                "warning_components": warnings_list,
            },
            "parameters_used": params,
        }

    def compute_internal_stress(
        self,
        image_stack: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Map internal force distribution using dense optical flow divergence/shear.
        """
        params = parameters or {}
        if not CV2_AVAILABLE:
            return {"status": "error", "message": "OpenCV is required for optical flow (cv2.calcOpticalFlowFarneback)."}

        try:
            stack = self._prepare_stack(image_stack)
            flow_fields = self._compute_dense_flow(stack, params)
            tensors = self._compute_strain_rate_tensors(flow_fields["vx"], flow_fields["vy"])

            divergence = tensors["divergence"]
            shear_rate = tensors["shear_rate"]
            divergence_map = np.mean(divergence, axis=0)
            shear_map = np.mean(np.abs(shear_rate), axis=0)
            mask_2d = self._resolve_2d_mask(nuclear_mask, divergence_map.shape)

            summary = self._summarize_force_maps(divergence_map, shear_map, mask_2d)
            return {
                "status": "success",
                "method": "Chromatin Image Velocimetry",
                "velocity_fields": flow_fields,
                "strain_rate_tensors": tensors,
                "divergence_map": divergence_map,
                "shear_rate_map": shear_map,
                "summary": summary,
            }
        except Exception as exc:
            return {"status": "error", "message": f"Internal stress mapping failed: {exc}"}

    def estimate_stiffness_proxy(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate stiffness proxy from displacement correlation decay length xi.
        """
        params = parameters or {}
        try:
            if vx.ndim != 3 or vy.ndim != 3 or vx.shape != vy.shape:
                raise ValueError("Velocity fields must be matching 3D arrays (T, Y, X).")

            mask_2d = self._resolve_2d_mask(nuclear_mask, vx.shape[1:])
            radial_distances = None
            radial_curves: List[np.ndarray] = []
            corr2d_list: List[np.ndarray] = []

            for t in range(vx.shape[0]):
                corr_x = self._masked_autocorrelation(vx[t], mask_2d)
                corr_y = self._masked_autocorrelation(vy[t], mask_2d)
                corr_xy = 0.5 * (corr_x + corr_y)
                center = corr_xy[corr_xy.shape[0] // 2, corr_xy.shape[1] // 2]
                if np.isfinite(center) and abs(center) > 1e-12:
                    corr_xy = corr_xy / center

                dist_um, radial = self._radial_average(corr_xy, self.pixel_size_um)
                corr2d_list.append(corr_xy)
                radial_curves.append(radial)
                radial_distances = dist_um

            if not radial_curves or radial_distances is None:
                return {"status": "error", "message": "Could not compute radial correlation curves."}

            radial_matrix = np.vstack(radial_curves)
            mean_radial = np.nanmean(radial_matrix, axis=0)
            fit = self._fit_exponential_decay(radial_distances, mean_radial, min_points=int(params.get("min_corr_points", 6)))

            return {
                "status": "success",
                "method": "Displacement Correlation Spectroscopy",
                "correlation_map": np.nanmean(np.stack(corr2d_list, axis=0), axis=0),
                "radial_distances_um": radial_distances,
                "radial_correlation": mean_radial,
                "correlation_length_um": fit.get("xi_um", np.nan),
                "fit": fit,
                "interpretation": self._interpret_stiffness(fit.get("xi_um", np.nan)),
            }
        except Exception as exc:
            return {"status": "error", "message": f"Stiffness proxy failed: {exc}"}

    def analyze_texture_topology(
        self,
        image_stack: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Texture-derived compaction proxies: local GLCM entropy + fractal dimension.
        """
        params = parameters or {}
        try:
            stack = self._prepare_stack(image_stack)
            frame = np.mean(stack, axis=0)
            mask_2d = self._resolve_2d_mask(nuclear_mask, frame.shape)

            entropy_result = self._compute_glcm_entropy(frame, mask_2d, params)
            fractal_result = self._compute_fractal_metrics(frame, mask_2d, params)

            return {
                "status": "success",
                "method": "Texture Topology",
                "entropy": entropy_result,
                "fractal": fractal_result,
                "interpretation": self._interpret_texture(entropy_result, fractal_result),
            }
        except Exception as exc:
            return {"status": "error", "message": f"Texture topology analysis failed: {exc}"}

    def analyze_compartment_fusion(
        self,
        image_stack: np.ndarray,
        compartment_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect fusion events and fit AR relaxation to extract viscosity proxy tau.
        """
        params = parameters or {}
        try:
            stack = self._prepare_stack(image_stack)
            masks = self._resolve_compartment_masks(stack, compartment_mask, params)
            if masks is None or masks.ndim != 3:
                return {"status": "error", "message": "Compartment fusion requires a 3D mask stack (T, Y, X)."}

            min_area_px = int(params.get("fusion_min_area_px", 12))
            min_trace = int(params.get("fusion_min_trace_points", 4))
            max_frames = int(params.get("fusion_max_fit_frames", 20))
            events = self._detect_fusion_events(masks, min_area_px=min_area_px, min_trace=min_trace, max_frames=max_frames)

            if not events:
                return {
                    "status": "warning",
                    "message": "No fusion events detected with current masks/thresholds.",
                    "events": [],
                }

            tau_values = np.array([ev["tau_s"] for ev in events if np.isfinite(ev.get("tau_s", np.nan))], dtype=float)
            return {
                "status": "success",
                "method": "Compartment Fusion Kinetics",
                "events": events,
                "mean_tau_s": float(np.nanmean(tau_values)) if tau_values.size else np.nan,
                "median_tau_s": float(np.nanmedian(tau_values)) if tau_values.size else np.nan,
                "num_events": int(len(events)),
            }
        except Exception as exc:
            return {"status": "error", "message": f"Fusion kinetics analysis failed: {exc}"}

    def analyze_envelope_mechanics(
        self,
        image_stack: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Boundary fluctuation spectroscopy and Helfrich-like sigma/kappa fit.
        """
        params = parameters or {}
        try:
            stack = self._prepare_stack(image_stack)
            mask_stack = self._resolve_nuclear_mask_stack(stack, nuclear_mask)
            if mask_stack is None or mask_stack.ndim != 3:
                return {"status": "error", "message": "Unable to build nuclear mask stack for boundary analysis."}

            n_angles = int(params.get("boundary_n_angles", 128))
            use_active_contours = bool(params.get("use_active_contours", True)) and SKIMAGE_AVAILABLE
            min_frames = int(params.get("boundary_min_frames", 4))

            radial_profiles = []
            frame_ids = []
            for frame_idx in range(stack.shape[0]):
                contour = self._extract_primary_contour(mask_stack[frame_idx])
                if contour is None:
                    continue
                refined = contour
                if use_active_contours:
                    refined = self._refine_contour_with_snake(stack[frame_idx], contour, params)
                radii = self._contour_to_radial_profile(refined, n_angles=n_angles)
                if radii is None:
                    continue
                radial_profiles.append(radii)
                frame_ids.append(frame_idx)

            if len(radial_profiles) < min_frames:
                return {
                    "status": "warning",
                    "message": "Insufficient valid boundary frames for fluctuation fitting.",
                    "num_valid_frames": len(radial_profiles),
                }

            radii_um = np.asarray(radial_profiles, dtype=np.float64) * self.pixel_size_um
            mean_radius_um = float(np.mean(np.mean(radii_um, axis=1)))

            # Remove per-frame mean radius to isolate shape fluctuations.
            fluctuations = radii_um - np.mean(radii_um, axis=1, keepdims=True)
            coeff = np.fft.rfft(fluctuations, axis=1)
            power = np.mean(np.abs(coeff) ** 2, axis=0)
            modes = np.arange(power.size, dtype=np.float64)

            fit = self._fit_helfrich_spectrum(modes, power, mean_radius_um)
            return {
                "status": "success",
                "method": "Boundary Fluctuation Spectroscopy",
                "mode_numbers": modes,
                "power_spectrum": power,
                "fit": fit,
                "surface_tension_sigma_N_per_m": fit.get("sigma_N_per_m", np.nan),
                "bending_rigidity_kappa_J": fit.get("kappa_J", np.nan),
                "bending_rigidity_kappa_over_kBT": fit.get("kappa_over_kbt", np.nan),
                "mean_radius_um": mean_radius_um,
                "num_valid_frames": len(radial_profiles),
                "frame_indices": frame_ids,
            }
        except Exception as exc:
            return {"status": "error", "message": f"Boundary mechanics analysis failed: {exc}"}

    def _prepare_stack(self, image_data: np.ndarray) -> np.ndarray:
        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("Material mechanics analysis requires a 3D stack (T, Y, X).")
        if arr.shape[0] < 2:
            raise ValueError("Material mechanics analysis requires at least 2 frames.")
        if not np.any(np.isfinite(arr)):
            raise ValueError("Image stack contains no finite values.")
        return np.nan_to_num(arr, copy=False)

    def _normalize_to_u8(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        p_low = np.percentile(frame, 1.0)
        p_high = np.percentile(frame, 99.0)
        if p_high <= p_low:
            p_low = float(np.min(frame))
            p_high = float(np.max(frame))
        if p_high <= p_low:
            return np.zeros(frame.shape, dtype=np.uint8)
        clipped = np.clip(frame, p_low, p_high)
        scaled = (clipped - p_low) / (p_high - p_low)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    def _compute_dense_flow(self, image_stack: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, np.ndarray]:
        pyr_scale = float(parameters.get("farneback_pyr_scale", 0.5))
        levels = int(parameters.get("farneback_levels", 3))
        winsize = int(parameters.get("farneback_winsize", 15))
        iterations = int(parameters.get("farneback_iterations", 3))
        poly_n = int(parameters.get("farneback_poly_n", 5))
        poly_sigma = float(parameters.get("farneback_poly_sigma", 1.2))
        blur_sigma = float(parameters.get("flow_preblur_sigma", 0.0))

        n_pairs = image_stack.shape[0] - 1
        h, w = image_stack.shape[1], image_stack.shape[2]
        vx = np.zeros((n_pairs, h, w), dtype=np.float32)
        vy = np.zeros((n_pairs, h, w), dtype=np.float32)

        for t in range(n_pairs):
            prev = image_stack[t]
            curr = image_stack[t + 1]
            if blur_sigma > 0:
                prev = ndimage.gaussian_filter(prev, blur_sigma)
                curr = ndimage.gaussian_filter(curr, blur_sigma)

            prev_u8 = self._normalize_to_u8(prev)
            curr_u8 = self._normalize_to_u8(curr)

            flow = cv2.calcOpticalFlowFarneback(
                prev_u8,
                curr_u8,
                None,
                pyr_scale=pyr_scale,
                levels=levels,
                winsize=winsize,
                iterations=iterations,
                poly_n=poly_n,
                poly_sigma=poly_sigma,
                flags=0,
            )

            scale = self.pixel_size_um / max(self.time_interval_s, 1e-9)
            vx[t] = flow[..., 0] * scale
            vy[t] = flow[..., 1] * scale

        speed = np.sqrt(vx**2 + vy**2)
        return {"vx": vx, "vy": vy, "speed": speed}

    def _compute_strain_rate_tensors(self, vx: np.ndarray, vy: np.ndarray) -> Dict[str, np.ndarray]:
        px = max(self.pixel_size_um, 1e-9)
        du_dx = np.gradient(vx, axis=2) / px
        du_dy = np.gradient(vx, axis=1) / px
        dv_dx = np.gradient(vy, axis=2) / px
        dv_dy = np.gradient(vy, axis=1) / px

        divergence = du_dx + dv_dy
        shear_rate = du_dy + dv_dx
        shear_invariant = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2)

        return {
            "du_dx": du_dx,
            "du_dy": du_dy,
            "dv_dx": dv_dx,
            "dv_dy": dv_dy,
            "divergence": divergence,
            "shear_rate": shear_rate,
            "shear_invariant": shear_invariant,
        }

    def _summarize_force_maps(self, divergence_map: np.ndarray, shear_map: np.ndarray, mask_2d: Optional[np.ndarray]) -> Dict[str, float]:
        if mask_2d is None:
            div_values = divergence_map.ravel()
            shear_values = shear_map.ravel()
        else:
            div_values = divergence_map[mask_2d]
            shear_values = shear_map[mask_2d]

        if div_values.size == 0:
            div_values = divergence_map.ravel()
        if shear_values.size == 0:
            shear_values = shear_map.ravel()

        return {
            "mean_divergence_per_s": float(np.nanmean(div_values)),
            "std_divergence_per_s": float(np.nanstd(div_values)),
            "positive_divergence_fraction": float(np.mean(div_values > 0)),
            "negative_divergence_fraction": float(np.mean(div_values < 0)),
            "mean_abs_shear_rate_per_s": float(np.nanmean(np.abs(shear_values))),
            "max_abs_shear_rate_per_s": float(np.nanmax(np.abs(shear_values))),
        }

    def _resolve_2d_mask(self, mask: Optional[np.ndarray], shape_2d: Tuple[int, int]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask) > 0
        if arr.ndim == 2 and arr.shape == shape_2d:
            return arr
        if arr.ndim == 3 and arr.shape[1:] == shape_2d:
            return np.any(arr, axis=0)
        return None

    def _masked_autocorrelation(self, field: np.ndarray, mask_2d: Optional[np.ndarray]) -> np.ndarray:
        if mask_2d is None:
            work = field.astype(np.float64)
            valid = np.ones_like(work, dtype=np.float64)
        else:
            valid = mask_2d.astype(np.float64)
            work = field.astype(np.float64) * valid

        if np.any(valid > 0):
            mean_val = np.sum(work) / np.sum(valid)
            work = (work - mean_val) * valid

        f = fft2(work)
        corr = np.real(fftshift(ifft2(f * np.conj(f))))

        weight = np.real(fftshift(ifft2(fft2(valid) * np.conj(fft2(valid)))))
        weight = np.maximum(weight, 1.0)
        corr = corr / weight
        return corr

    def _radial_average(self, image_2d: np.ndarray, pixel_spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_2d.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_int = r.astype(np.int32)
        r_max = int(min(cy, cx))

        radial_vals = np.zeros(r_max, dtype=np.float64)
        counts = np.zeros(r_max, dtype=np.float64)
        for radius in range(r_max):
            region = r_int == radius
            counts[radius] = np.sum(region)
            if counts[radius] > 0:
                radial_vals[radius] = np.mean(image_2d[region])
            else:
                radial_vals[radius] = np.nan

        distances = (np.arange(r_max, dtype=np.float64) + 0.5) * pixel_spacing
        return distances, radial_vals

    def _fit_exponential_decay(self, distances_um: np.ndarray, correlation: np.ndarray, min_points: int = 6) -> Dict[str, Any]:
        valid = np.isfinite(distances_um) & np.isfinite(correlation)
        valid &= distances_um > 0
        if np.sum(valid) < max(min_points, 3):
            return {"status": "warning", "message": "Not enough valid correlation points for fitting.", "xi_um": np.nan}

        x = distances_um[valid]
        y = correlation[valid]

        # Keep pre-zero region to fit early decay where exponential model is valid.
        first_non_pos = np.where(y <= 0)[0]
        if first_non_pos.size > 0 and first_non_pos[0] > min_points:
            keep = first_non_pos[0]
            x = x[:keep]
            y = y[:keep]

        if x.size < max(min_points, 3):
            return {"status": "warning", "message": "Correlation curve too short for robust fitting.", "xi_um": np.nan}

        def model(r, a, xi, c):
            return a * np.exp(-r / xi) + c

        try:
            p0 = [max(float(y[0] - y[-1]), 1e-3), max(float(x[-1] / 3.0), 1e-3), float(y[-1])]
            bounds = ([0.0, 1e-6, -1.0], [10.0, np.inf, 1.0])
            popt, pcov = optimize.curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=5000)
            fitted = model(distances_um, *popt)
            residuals = y - model(x, *popt)
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            perr = np.sqrt(np.diag(pcov))
            return {
                "status": "success",
                "a": float(popt[0]),
                "xi_um": float(popt[1]),
                "c": float(popt[2]),
                "a_error": float(perr[0]),
                "xi_error_um": float(perr[1]),
                "c_error": float(perr[2]),
                "r_squared": float(r2),
                "fitted_curve": fitted,
            }
        except Exception as exc:
            return {"status": "warning", "message": f"Correlation length fit failed: {exc}", "xi_um": np.nan}

    def _interpret_stiffness(self, xi_um: float) -> str:
        if not np.isfinite(xi_um):
            return "undetermined"
        if xi_um >= 2.0:
            return "high mechanical coupling / stiffer-like"
        if xi_um >= 0.8:
            return "intermediate coupling"
        return "low mechanical coupling / softer-like"

    def _compute_glcm_entropy(self, frame: np.ndarray, mask_2d: Optional[np.ndarray], parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not SKIMAGE_AVAILABLE:
            return {"status": "warning", "message": "scikit-image unavailable for GLCM entropy.", "entropy_map": None}

        window = int(parameters.get("glcm_window", 16))
        step = int(parameters.get("glcm_step", max(4, window // 2)))
        levels = int(parameters.get("glcm_levels", 32))
        window = max(window, 4)
        step = max(step, 1)
        levels = max(levels, 8)

        img_u8 = self._normalize_to_u8(frame)
        quantized = np.floor(img_u8.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)

        h, w = quantized.shape
        entropy_map = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

        for y0 in range(0, h - window + 1, step):
            for x0 in range(0, w - window + 1, step):
                patch_mask = None if mask_2d is None else mask_2d[y0 : y0 + window, x0 : x0 + window]
                if patch_mask is not None and np.mean(patch_mask) < 0.25:
                    continue

                patch = quantized[y0 : y0 + window, x0 : x0 + window]
                glcm = graycomatrix(
                    patch,
                    distances=[1],
                    angles=angles,
                    levels=levels,
                    symmetric=True,
                    normed=True,
                )
                probs = glcm[:, :, 0, :]
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log2(probs)))

                entropy_map[y0 : y0 + window, x0 : x0 + window] += entropy
                counts[y0 : y0 + window, x0 : x0 + window] += 1.0

        valid = counts > 0
        entropy_map[valid] /= counts[valid]
        if np.any(~valid):
            fill_value = float(np.nanmean(entropy_map[valid])) if np.any(valid) else 0.0
            entropy_map[~valid] = fill_value

        values = entropy_map[mask_2d] if mask_2d is not None else entropy_map.ravel()
        return {
            "status": "success",
            "entropy_map": entropy_map,
            "mean_entropy": float(np.nanmean(values)),
            "std_entropy": float(np.nanstd(values)),
            "interpretation": "higher entropy suggests softer/disordered texture",
        }

    def _compute_fractal_metrics(self, frame: np.ndarray, mask_2d: Optional[np.ndarray], parameters: Dict[str, Any]) -> Dict[str, Any]:
        quantiles = parameters.get("fractal_quantiles", [0.5, 0.65, 0.8])
        if not isinstance(quantiles, Sequence) or len(quantiles) == 0:
            quantiles = [0.5, 0.65, 0.8]

        data = frame.astype(np.float64)
        if mask_2d is not None:
            pool = data[mask_2d]
            if pool.size == 0:
                pool = data.ravel()
        else:
            pool = data.ravel()

        scans = []
        for q in quantiles:
            qf = float(q)
            threshold = float(np.quantile(pool, qf))
            binary = data >= threshold
            if mask_2d is not None:
                binary = binary & mask_2d
            fd = self._box_count_fractal_dimension(binary)
            scans.append({"quantile": qf, "threshold": threshold, "fractal_dimension": float(fd)})

        values = np.array([s["fractal_dimension"] for s in scans], dtype=float)
        finite = values[np.isfinite(values)]
        global_fd = float(np.nanmedian(finite)) if finite.size else np.nan

        return {
            "status": "success",
            "global_fractal_dimension": global_fd,
            "scan": scans,
        }

    def _box_count_fractal_dimension(self, binary: np.ndarray) -> float:
        arr = np.asarray(binary, dtype=bool)
        if arr.ndim != 2 or np.sum(arr) == 0:
            return np.nan

        h, w = arr.shape
        min_side = min(h, w)
        max_pow = int(np.floor(np.log2(min_side)))
        if max_pow < 2:
            return np.nan

        sizes = 2 ** np.arange(max_pow, 1, -1)
        counts = []
        valid_sizes = []
        arr_i = arr.astype(np.int32)

        for size in sizes:
            hh = (h // size) * size
            ww = (w // size) * size
            if hh < size or ww < size:
                continue

            cropped = arr_i[:hh, :ww]
            summed = np.add.reduceat(np.add.reduceat(cropped, np.arange(0, hh, size), axis=0), np.arange(0, ww, size), axis=1)
            count = int(np.sum(summed > 0))
            if count > 0:
                counts.append(count)
                valid_sizes.append(size)

        if len(counts) < 2:
            return np.nan

        x = np.log(1.0 / np.asarray(valid_sizes, dtype=np.float64))
        y = np.log(np.asarray(counts, dtype=np.float64))
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def _interpret_texture(self, entropy_result: Dict[str, Any], fractal_result: Dict[str, Any]) -> str:
        mean_entropy = entropy_result.get("mean_entropy", np.nan)
        fractal_dim = fractal_result.get("global_fractal_dimension", np.nan)
        parts = []

        if np.isfinite(mean_entropy):
            if mean_entropy < 1.5:
                parts.append("low entropy / more compact-like texture")
            elif mean_entropy < 2.5:
                parts.append("intermediate entropy")
            else:
                parts.append("high entropy / more disordered-like texture")

        if np.isfinite(fractal_dim):
            parts.append(f"fractal dimension ~ {fractal_dim:.2f}")

        return "; ".join(parts) if parts else "undetermined"

    def _resolve_compartment_masks(
        self,
        image_stack: np.ndarray,
        compartment_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        t_frames, h, w = image_stack.shape
        if compartment_mask is not None:
            mask_arr = np.asarray(compartment_mask) > 0
            if mask_arr.ndim == 2 and mask_arr.shape == (h, w):
                mask_arr = np.repeat(mask_arr[None, :, :], t_frames, axis=0)
            if mask_arr.ndim != 3 or mask_arr.shape != (t_frames, h, w):
                return None
            return mask_arr

        percentile = float(parameters.get("fusion_percentile", 94.0))
        min_size = int(parameters.get("fusion_min_area_px", 12))
        masks = np.zeros_like(image_stack, dtype=bool)

        for t in range(t_frames):
            frame = image_stack[t]
            threshold = np.percentile(frame, percentile)
            binary = frame >= threshold
            binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)))
            binary = ndimage.binary_closing(binary, structure=np.ones((3, 3)))

            labels, num = ndimage.label(binary)
            if num == 0:
                continue

            areas = np.bincount(labels.ravel())
            keep = np.zeros_like(binary, dtype=bool)
            for label_id in range(1, num + 1):
                if label_id < areas.size and areas[label_id] >= min_size:
                    keep |= labels == label_id
            masks[t] = keep

        if np.sum(masks) == 0:
            return None
        return masks

    def _detect_fusion_events(self, masks: np.ndarray, min_area_px: int, min_trace: int, max_frames: int) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        t_frames = masks.shape[0]

        labeled = []
        counts = []
        for t in range(t_frames):
            lbl, n = ndimage.label(masks[t])
            labeled.append(lbl)
            counts.append(n)

        for t in range(t_frames - 1):
            lbl_prev, n_prev = labeled[t], counts[t]
            lbl_next, n_next = labeled[t + 1], counts[t + 1]
            if n_prev < 2 or n_next < 1:
                continue

            areas_prev = np.bincount(lbl_prev.ravel())
            for next_id in range(1, n_next + 1):
                region_next = lbl_next == next_id
                if np.sum(region_next) < min_area_px:
                    continue

                overlapping_prev = np.unique(lbl_prev[region_next])
                overlapping_prev = overlapping_prev[overlapping_prev > 0]
                overlapping_prev = np.array(
                    [pid for pid in overlapping_prev if pid < areas_prev.size and areas_prev[pid] >= min_area_px],
                    dtype=int,
                )
                if overlapping_prev.size < 2:
                    continue

                trace = self._track_component_aspect_ratio(
                    masks, start_frame=t + 1, seed_component=(t + 1, next_id), max_frames=max_frames
                )
                if len(trace["aspect_ratio"]) < min_trace:
                    continue

                fit = self._fit_aspect_ratio_relaxation(trace["time_s"], trace["aspect_ratio"])
                event = {
                    "fusion_frame": int(t + 1),
                    "num_parents": int(overlapping_prev.size),
                    "parent_labels": overlapping_prev.tolist(),
                    "trace_time_s": trace["time_s"].tolist(),
                    "trace_aspect_ratio": trace["aspect_ratio"].tolist(),
                    "tau_s": fit.get("tau_s", np.nan),
                    "fit_r_squared": fit.get("r_squared", np.nan),
                    "fit": fit,
                }
                events.append(event)

        return events

    def _track_component_aspect_ratio(
        self,
        masks: np.ndarray,
        start_frame: int,
        seed_component: Tuple[int, int],
        max_frames: int,
    ) -> Dict[str, np.ndarray]:
        frame_idx, component_label = seed_component
        lbl, _ = ndimage.label(masks[frame_idx])
        current_mask = lbl == component_label
        if np.sum(current_mask) == 0:
            return {"time_s": np.array([]), "aspect_ratio": np.array([])}

        ar_values = []
        times = []
        t_end = min(masks.shape[0], start_frame + max_frames)

        for t in range(start_frame, t_end):
            labels, n = ndimage.label(masks[t])
            if n == 0:
                break
            overlap = ndimage.sum(current_mask.astype(np.int32), labels, index=np.arange(1, n + 1))
            if overlap.size == 0:
                break
            best_idx = int(np.argmax(overlap)) + 1
            if overlap[best_idx - 1] <= 0:
                break

            region = labels == best_idx
            ar = self._aspect_ratio(region)
            if not np.isfinite(ar):
                break

            times.append((t - start_frame) * self.time_interval_s)
            ar_values.append(ar)
            current_mask = region

        return {"time_s": np.asarray(times, dtype=np.float64), "aspect_ratio": np.asarray(ar_values, dtype=np.float64)}

    def _aspect_ratio(self, binary_region: np.ndarray) -> float:
        coords = np.argwhere(binary_region)
        if coords.shape[0] < 5:
            return np.nan
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 1e-12))
        return float(np.sqrt(eigvals[-1] / eigvals[0]))

    def _fit_aspect_ratio_relaxation(self, time_s: np.ndarray, aspect_ratio: np.ndarray) -> Dict[str, Any]:
        if time_s.size < 3 or aspect_ratio.size < 3:
            return {"status": "warning", "message": "Not enough points to fit fusion relaxation.", "tau_s": np.nan}

        t = np.asarray(time_s, dtype=np.float64)
        y = np.asarray(aspect_ratio, dtype=np.float64)
        y = np.maximum(y, 1.0)

        def model(time_vals, ar0, tau):
            return 1.0 + (ar0 - 1.0) * np.exp(-time_vals / tau)

        try:
            p0 = [max(float(y[0]), 1.05), max(float(t[-1] / 2.0), self.time_interval_s)]
            bounds = ([1.0, 1e-6], [50.0, max(1000.0, t[-1] * 100.0 + 1.0)])
            popt, pcov = optimize.curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = model(t, *popt)
            ss_res = float(np.sum((y - y_fit) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            perr = np.sqrt(np.diag(pcov))
            return {
                "status": "success",
                "ar0": float(popt[0]),
                "tau_s": float(popt[1]),
                "ar0_error": float(perr[0]),
                "tau_error_s": float(perr[1]),
                "r_squared": float(r2),
                "fitted_curve": y_fit,
            }
        except Exception as exc:
            return {"status": "warning", "message": f"Fusion relaxation fit failed: {exc}", "tau_s": np.nan}

    def _resolve_nuclear_mask_stack(self, image_stack: np.ndarray, nuclear_mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        t_frames, h, w = image_stack.shape
        if nuclear_mask is not None:
            mask_arr = np.asarray(nuclear_mask) > 0
            if mask_arr.ndim == 2 and mask_arr.shape == (h, w):
                dynamic = np.zeros((t_frames, h, w), dtype=bool)
                for t in range(t_frames):
                    dynamic[t] = self._estimate_nuclear_mask(image_stack[t], fallback=mask_arr)
                return dynamic
            if mask_arr.ndim == 3 and mask_arr.shape == (t_frames, h, w):
                return mask_arr
            return None

        masks = np.zeros((t_frames, h, w), dtype=bool)
        for t in range(t_frames):
            masks[t] = self._estimate_nuclear_mask(image_stack[t], fallback=None)
        return masks if np.sum(masks) > 0 else None

    def _estimate_nuclear_mask(self, frame: np.ndarray, fallback: Optional[np.ndarray]) -> np.ndarray:
        try:
            if SKIMAGE_AVAILABLE:
                thr = threshold_otsu(frame)
            else:
                thr = np.percentile(frame, 80.0)
            binary = frame > thr
            binary = ndimage.binary_fill_holes(binary)
            binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)))
            binary = ndimage.binary_closing(binary, structure=np.ones((5, 5)))
            largest = self._largest_component(binary)
            if np.sum(largest) > 0:
                return largest
        except Exception:
            pass
        if fallback is not None:
            return fallback.astype(bool)
        return np.zeros(frame.shape, dtype=bool)

    def _largest_component(self, binary: np.ndarray) -> np.ndarray:
        labels, n = ndimage.label(binary)
        if n == 0:
            return np.zeros(binary.shape, dtype=bool)
        areas = np.bincount(labels.ravel())
        if areas.size <= 1:
            return np.zeros(binary.shape, dtype=bool)
        largest = int(np.argmax(areas[1:]) + 1)
        return labels == largest

    def _extract_primary_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        if np.sum(mask) < 16:
            return None
        if SKIMAGE_AVAILABLE:
            contours = find_contours(mask.astype(np.float32), 0.5)
            if not contours:
                return None
            contour = max(contours, key=len)
            return self._resample_closed_curve(contour, target_points=256)

        # Fallback: contour from binary erosion difference.
        edges = mask & (~ndimage.binary_erosion(mask))
        coords = np.argwhere(edges)
        if coords.shape[0] < 16:
            return None
        return self._resample_closed_curve(coords.astype(np.float64), target_points=256)

    def _resample_closed_curve(self, contour: np.ndarray, target_points: int = 256) -> np.ndarray:
        pts = np.asarray(contour, dtype=np.float64)
        if pts.shape[0] < 8:
            return pts

        if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts = np.vstack([pts, pts[0]])

        seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        total = cumulative[-1]
        if total <= 1e-9:
            return pts[:-1]

        sample = np.linspace(0.0, total, target_points, endpoint=False)
        y = np.interp(sample, cumulative, pts[:, 0])
        x = np.interp(sample, cumulative, pts[:, 1])
        return np.column_stack([y, x])

    def _refine_contour_with_snake(self, frame: np.ndarray, contour: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        if not SKIMAGE_AVAILABLE:
            return contour
        try:
            alpha = float(parameters.get("snake_alpha", 0.01))
            beta = float(parameters.get("snake_beta", 3.0))
            gamma = float(parameters.get("snake_gamma", 0.001))
            max_iter = int(parameters.get("snake_max_iter", 120))

            frame_u8 = self._normalize_to_u8(frame).astype(np.float64) / 255.0
            smoothed = gaussian(frame_u8, sigma=1.0)
            snake = active_contour(
                smoothed,
                contour,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_num_iter=max_iter,
                convergence=0.05,
                boundary_condition="periodic",
            )
            if snake is None or len(snake) < 16:
                return contour
            return self._resample_closed_curve(np.asarray(snake), target_points=len(contour))
        except Exception:
            return contour

    def _contour_to_radial_profile(self, contour: np.ndarray, n_angles: int = 128) -> Optional[np.ndarray]:
        pts = np.asarray(contour, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 16:
            return None

        cy = float(np.mean(pts[:, 0]))
        cx = float(np.mean(pts[:, 1]))
        dy = pts[:, 0] - cy
        dx = pts[:, 1] - cx
        theta = np.arctan2(dy, dx)
        radius = np.sqrt(dx**2 + dy**2)

        order = np.argsort(theta)
        theta_sorted = theta[order]
        radius_sorted = radius[order]

        theta_wrapped = np.concatenate([theta_sorted - 2 * np.pi, theta_sorted, theta_sorted + 2 * np.pi])
        radius_wrapped = np.concatenate([radius_sorted, radius_sorted, radius_sorted])
        sample_theta = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
        sampled = np.interp(sample_theta, theta_wrapped, radius_wrapped)
        return sampled

    def _fit_helfrich_spectrum(self, modes: np.ndarray, power: np.ndarray, mean_radius_um: float) -> Dict[str, Any]:
        k_b = 1.380649e-23
        kbt = k_b * max(self.temperature_k, 1.0)
        radius_m = max(mean_radius_um, 1e-6) * 1e-6

        valid = np.isfinite(modes) & np.isfinite(power) & (power > 0) & (modes >= 2)
        if np.sum(valid) < 4:
            return {
                "status": "warning",
                "message": "Insufficient spectral points for sigma/kappa fit.",
                "sigma_N_per_m": np.nan,
                "kappa_J": np.nan,
                "kappa_over_kbt": np.nan,
            }

        q = modes[valid]
        p = power[valid]

        def model(mode_num, sigma, kappa):
            denom = sigma * (mode_num**2) + (kappa / (radius_m**2)) * (mode_num**4)
            denom = np.maximum(denom, 1e-30)
            return kbt / denom

        try:
            p0 = [1e-7, 1e-19]
            bounds = ([1e-12, 1e-23], [1e-2, 1e-15])
            popt, pcov = optimize.curve_fit(model, q, p, p0=p0, bounds=bounds, maxfev=10000)
            fitted = model(modes, *popt)
            ss_res = float(np.sum((p - model(q, *popt)) ** 2))
            ss_tot = float(np.sum((p - np.mean(p)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            perr = np.sqrt(np.diag(pcov))
            return {
                "status": "success",
                "sigma_N_per_m": float(popt[0]),
                "kappa_J": float(popt[1]),
                "kappa_over_kbt": float(popt[1] / kbt),
                "sigma_error": float(perr[0]),
                "kappa_error": float(perr[1]),
                "r_squared": float(r2),
                "fitted_spectrum": fitted,
            }
        except Exception as exc:
            return {
                "status": "warning",
                "message": f"Helfrich spectrum fit failed: {exc}",
                "sigma_N_per_m": np.nan,
                "kappa_J": np.nan,
                "kappa_over_kbt": np.nan,
            }


def get_material_mechanics_parameters() -> Dict[str, Any]:
    """Default parameter ranges for MaterialMechanics UI controls."""
    return {
        "pixel_size_um": {"default": 0.1, "min": 0.01, "max": 10.0, "description": "Pixel size (um/pixel)"},
        "time_interval_s": {
            "default": 1.0,
            "min": 1e-3,
            "max": 100.0,
            "description": "Frame interval (seconds)",
        },
        "farneback_winsize": {"default": 15, "min": 5, "max": 51, "description": "Farneback window size"},
        "glcm_window": {"default": 16, "min": 6, "max": 64, "description": "GLCM entropy window size"},
        "fusion_percentile": {
            "default": 94.0,
            "min": 50.0,
            "max": 99.9,
            "description": "Auto-threshold percentile for fusion masks",
        },
        "boundary_n_angles": {"default": 128, "min": 32, "max": 512, "description": "Angular samples for boundary FFT"},
    }
