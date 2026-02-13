"""
Condensate Lineage Tracking Module

Implements lineage-aware condensate tracking using:
- Seeded watershed (H-maxima markers) to separate touching droplets
- Directed acyclic graph (DAG) lineage with IoU-based temporal linking
- Event detection (fusion, fission, nucleation)
- Fusion relaxation kinetics, growth regime fitting, and wetting analysis
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, optimize

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available - lineage graph analysis disabled")

try:
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, threshold_otsu
    from skimage.measure import regionprops
    from skimage.morphology import (
        binary_closing,
        binary_opening,
        disk,
        h_maxima,
        remove_small_objects,
    )
    from skimage.segmentation import find_boundaries, watershed

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - condensate lineage module is limited")


class CondensateLineageTracker:
    """
    DAG-based condensate tracker with event-driven material-property readouts.
    """

    def __init__(self, pixel_size_um: float = 0.1, time_interval_s: float = 1.0):
        self.pixel_size_um = float(pixel_size_um)
        self.time_interval_s = float(time_interval_s)
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.last_segmentation: Optional[List[Dict[str, Any]]] = None

    def analyze(
        self,
        image_stack: np.ndarray,
        nuclear_mask: Optional[np.ndarray] = None,
        boundary_mask: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full analysis:
        1) H-maxima seeded watershed segmentation
        2) IoU-based DAG lineage construction
        3) Event detection + physics analyses
        """
        if not NETWORKX_AVAILABLE:
            return {"status": "error", "message": "networkx is required for lineage graph tracking."}
        if not SKIMAGE_AVAILABLE:
            return {"status": "error", "message": "scikit-image is required for segmentation and shape analysis."}

        params = parameters or {}
        try:
            stack = self._prepare_stack(image_stack)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        mask_stack = self._resolve_mask_stack(nuclear_mask, stack.shape)
        boundary_stack = self._resolve_mask_stack(boundary_mask, stack.shape)

        try:
            segmentation = self.segment_stack(stack, mask_stack, params)
            self.last_segmentation = segmentation
            graph = self.build_lineage(segmentation, params)
            event_summary = self._summarize_events(graph, params)

            run_fusion = bool(params.get("run_fusion_analysis", True))
            run_growth = bool(params.get("run_growth_analysis", True))
            run_wetting = bool(params.get("run_wetting_analysis", False))

            fusion_results = self.analyze_fusion_events(graph, params) if run_fusion else []
            growth_results = self.analyze_growth_regimes(graph, params) if run_growth else []
            wetting_results = (
                self.analyze_wetting(graph, segmentation, boundary_stack, params)
                if run_wetting and boundary_stack is not None
                else []
            )

            return {
                "status": "success",
                "method": "Condensate Lineage Tracking",
                "summary": {
                    **event_summary,
                    "num_fusion_kinetic_events": len(fusion_results),
                    "num_growth_tracks": len(growth_results),
                    "num_wetting_events": len(wetting_results),
                },
                "segmentation_overview": self._summarize_segmentation(segmentation),
                "fusion_events": fusion_results,
                "growth_tracks": growth_results,
                "wetting_events": wetting_results,
                "parameters_used": params,
            }
        except Exception as exc:
            return {"status": "error", "message": f"Condensate lineage analysis failed: {exc}"}

    def segment_stack(
        self,
        image_stack: np.ndarray,
        mask_stack: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Segment each frame using seeded watershed with H-maxima markers.
        """
        results: List[Dict[str, Any]] = []
        for t in range(image_stack.shape[0]):
            frame_mask = None if mask_stack is None else mask_stack[t]
            frame_result = self._segment_frame(image_stack[t], frame_mask, parameters)
            frame_result["frame"] = t
            results.append(frame_result)
        return results

    def build_lineage(self, segmentation: List[Dict[str, Any]], parameters: Dict[str, Any]) -> nx.DiGraph:
        """
        Build DAG over segmented objects with IoU-based links that permit
        many-to-one (fusion) and one-to-many (fission) edges.
        """
        iou_threshold = float(parameters.get("iou_threshold", 0.08))
        parent_top_k = int(parameters.get("parent_top_k", 2))
        child_top_k = int(parameters.get("child_top_k", 2))
        min_area_ratio = float(parameters.get("min_area_ratio", 0.2))
        max_area_ratio = float(parameters.get("max_area_ratio", 5.0))

        graph = nx.DiGraph()

        # Add nodes first.
        for frame_data in segmentation:
            t = int(frame_data["frame"])
            for obj in frame_data["objects"]:
                node = (t, int(obj["label"]))
                attrs = {k: v for k, v in obj.items() if k not in {"local_mask"}}
                graph.add_node(node, **attrs)

        # Link adjacent frames by IoU with top-k gating in both directions.
        for t in range(len(segmentation) - 1):
            objects_a = segmentation[t]["objects"]
            objects_b = segmentation[t + 1]["objects"]
            if not objects_a or not objects_b:
                continue

            iou = self._compute_iou_matrix(objects_a, objects_b)
            if iou.size == 0:
                continue

            parent_best = {}
            child_best = {}
            for i in range(iou.shape[0]):
                order = np.argsort(iou[i])[::-1]
                keep = [j for j in order if iou[i, j] >= iou_threshold][:parent_top_k]
                parent_best[i] = keep
            for j in range(iou.shape[1]):
                order = np.argsort(iou[:, j])[::-1]
                keep = [i for i in order if iou[i, j] >= iou_threshold][:child_top_k]
                child_best[j] = keep

            for i, obj_a in enumerate(objects_a):
                for j in parent_best.get(i, []):
                    if i not in child_best.get(j, []):
                        continue

                    score = float(iou[i, j])
                    if score < iou_threshold:
                        continue

                    area_a = float(obj_a["area"])
                    area_b = float(objects_b[j]["area"])
                    ratio = area_b / max(area_a, 1e-9)
                    if ratio < min_area_ratio or ratio > max_area_ratio:
                        continue

                    src = (t, int(obj_a["label"]))
                    dst = (t + 1, int(objects_b[j]["label"]))
                    graph.add_edge(src, dst, iou=score, area_ratio=ratio)

        self.graph = graph
        return graph

    def analyze_fusion_events(self, graph: nx.DiGraph, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze many-to-one events and fit AR relaxation kinetics:
        AR(t) = 1 + A * exp(-t/tau)
        """
        mass_tolerance = float(parameters.get("mass_tolerance", 0.25))
        lookahead_frames = int(parameters.get("fusion_lookahead_frames", 20))
        min_trace_points = int(parameters.get("fusion_min_trace_points", 6))
        min_initial_ar = float(parameters.get("fusion_min_initial_ar", 1.08))

        results = []
        fusion_nodes = [n for n in graph.nodes if graph.in_degree(n) > 1]
        for node in fusion_nodes:
            parents = list(graph.predecessors(node))
            if len(parents) < 2:
                continue

            child_mass = float(graph.nodes[node].get("total_intensity", 0.0))
            parent_mass = float(sum(graph.nodes[p].get("total_intensity", 0.0) for p in parents))
            if child_mass <= 0:
                continue
            mass_error = abs(child_mass - parent_mass) / child_mass
            if mass_error > mass_tolerance:
                continue

            chain = self._forward_chain(graph, node, max_len=lookahead_frames)
            if len(chain) < min_trace_points:
                continue

            times = []
            ar_values = []
            radii = []
            for k, n in enumerate(chain):
                major = float(graph.nodes[n].get("major_axis", 0.0))
                minor = float(graph.nodes[n].get("minor_axis", 0.0))
                if minor <= 1e-9:
                    break
                ar = major / minor
                times.append(k * self.time_interval_s)
                ar_values.append(ar)
                radii.append(float(graph.nodes[n].get("equiv_radius_px", np.nan)) * self.pixel_size_um)

            if len(ar_values) < min_trace_points:
                continue
            if ar_values[0] < min_initial_ar:
                continue

            fit = self._fit_ar_decay(np.asarray(times, dtype=float), np.asarray(ar_values, dtype=float))
            if not np.isfinite(fit.get("tau_s", np.nan)):
                continue

            mean_radius = float(np.nanmedian(radii)) if radii else np.nan
            results.append(
                {
                    "event_node": node,
                    "frame": int(node[0]),
                    "num_parents": len(parents),
                    "parents": parents,
                    "mass_error_fraction": float(mass_error),
                    "relaxation_time_tau_s": float(fit["tau_s"]),
                    "fit_r_squared": float(fit.get("r_squared", np.nan)),
                    "radius_um": mean_radius,
                    "viscosity_proxy_tau_over_radius": float(fit["tau_s"] / max(mean_radius, 1e-9))
                    if np.isfinite(mean_radius)
                    else np.nan,
                    "trace_time_s": [float(x) for x in times],
                    "trace_aspect_ratio": [float(x) for x in ar_values],
                }
            )

        return results

    def analyze_growth_regimes(self, graph: nx.DiGraph, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze growing tracks using power law R ~ a * t^beta and plateau checks.
        """
        min_track_length = int(parameters.get("growth_min_track_length", 6))
        plateau_fraction = float(parameters.get("plateau_growth_fraction", 0.05))
        gamma_surface_tension = parameters.get("gamma_surface_tension_N_per_m", None)
        gamma_surface_tension = float(gamma_surface_tension) if gamma_surface_tension is not None else None

        tracks = self._extract_linear_tracks(graph, min_length=min_track_length)
        results = []

        for track_id, nodes in enumerate(tracks):
            times = np.arange(1, len(nodes) + 1, dtype=float) * self.time_interval_s
            radius_um = np.array(
                [float(graph.nodes[n].get("equiv_radius_px", np.nan)) * self.pixel_size_um for n in nodes],
                dtype=float,
            )
            valid = np.isfinite(radius_um) & (radius_um > 0)
            if np.sum(valid) < min_track_length:
                continue
            t_fit = times[valid]
            r_fit = radius_um[valid]

            fit = self._fit_power_law_growth(t_fit, r_fit)
            beta = fit.get("beta", np.nan)
            r_terminal_um = float(np.nanmedian(r_fit[-min(3, len(r_fit)) :]))

            # Plateau / elastic ripening heuristic.
            regime = "undetermined"
            if len(r_fit) >= 6:
                split = max(2, len(r_fit) // 2)
                early = np.nanmean(r_fit[:split])
                late = np.nanmean(r_fit[split:])
                growth_frac = (late - early) / max(early, 1e-9)
            else:
                growth_frac = np.nan

            if np.isfinite(growth_frac) and growth_frac < plateau_fraction:
                regime = "elastic_ripening_candidate"
            elif np.isfinite(beta):
                if 0.20 <= beta <= 0.45:
                    regime = "ostwald_ripening_like"
                elif 0.45 < beta <= 0.80:
                    regime = "coalescence_like"
                else:
                    regime = "mixed_or_noncanonical"

            shear_modulus_pa = np.nan
            if regime == "elastic_ripening_candidate" and gamma_surface_tension is not None and r_terminal_um > 0:
                # Scaling estimate G ~ gamma / R_terminal.
                shear_modulus_pa = gamma_surface_tension / (r_terminal_um * 1e-6)

            results.append(
                {
                    "track_id": int(track_id),
                    "start_frame": int(nodes[0][0]),
                    "end_frame": int(nodes[-1][0]),
                    "num_points": int(len(nodes)),
                    "beta": float(beta) if np.isfinite(beta) else np.nan,
                    "fit_r_squared": float(fit.get("r_squared", np.nan)),
                    "regime": regime,
                    "terminal_radius_um": r_terminal_um,
                    "shear_modulus_proxy_pa": shear_modulus_pa,
                    "time_s": [float(x) for x in t_fit.tolist()],
                    "radius_um": [float(x) for x in r_fit.tolist()],
                }
            )

        return results

    def analyze_wetting(
        self,
        graph: nx.DiGraph,
        segmentation: List[Dict[str, Any]],
        boundary_stack: np.ndarray,
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Estimate contact angle for condensates touching a boundary mask.
        """
        boundary_band_px = int(parameters.get("wetting_boundary_band_px", 2))
        results: List[Dict[str, Any]] = []

        # Build quick index from node to segmentation object.
        object_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for frame_data in segmentation:
            t = int(frame_data["frame"])
            for obj in frame_data["objects"]:
                object_lookup[(t, int(obj["label"]))] = obj

        for node in graph.nodes:
            t, _ = node
            if t < 0 or t >= boundary_stack.shape[0]:
                continue
            boundary = boundary_stack[t]
            if np.sum(boundary) == 0:
                continue

            obj = object_lookup.get(node)
            if obj is None:
                continue

            object_mask = self._obj_to_full_mask(obj, boundary.shape)
            if np.sum(object_mask) == 0:
                continue

            boundary_edge = self._boundary_edge(boundary, band_px=boundary_band_px)
            if not np.any(object_mask & boundary_edge):
                continue

            angle = self._estimate_contact_angle(object_mask, boundary)
            if not np.isfinite(angle):
                continue

            results.append(
                {
                    "node": node,
                    "frame": int(t),
                    "contact_angle_deg": float(angle),
                    "wetting_class": self._classify_wetting(angle),
                }
            )

        return results

    def _prepare_stack(self, image_stack: np.ndarray) -> np.ndarray:
        arr = np.asarray(image_stack, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("Condensate lineage tracking requires a 3D stack (T, Y, X).")
        if arr.shape[0] < 2:
            raise ValueError("Condensate lineage tracking requires at least 2 frames.")
        if not np.any(np.isfinite(arr)):
            raise ValueError("Image stack has no finite intensities.")
        return np.nan_to_num(arr, copy=False)

    def _resolve_mask_stack(self, mask: Optional[np.ndarray], shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        t, h, w = shape
        m = np.asarray(mask) > 0
        if m.ndim == 2 and m.shape == (h, w):
            return np.repeat(m[None, :, :], t, axis=0)
        if m.ndim == 3 and m.shape == (t, h, w):
            return m
        return None

    def _segment_frame(
        self,
        frame: np.ndarray,
        frame_mask: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        sigma = float(parameters.get("segmentation_smoothing_sigma", 1.0))
        h_val = float(parameters.get("h_maxima_h", 0.08))
        min_area = int(parameters.get("min_area_px", 12))
        threshold_mode = str(parameters.get("threshold_mode", "otsu"))
        threshold_percentile = float(parameters.get("threshold_percentile", 85.0))
        topology_mode = str(parameters.get("topography_mode", "distance"))

        img = np.asarray(frame, dtype=np.float32)
        img = (img - np.min(img)) / max(np.max(img) - np.min(img), 1e-8)
        smoothed = gaussian(img, sigma=sigma, preserve_range=True)

        if frame_mask is not None:
            smoothed = np.where(frame_mask, smoothed, 0.0)

        if threshold_mode == "percentile":
            pool = smoothed[frame_mask] if frame_mask is not None else smoothed.ravel()
            if pool.size == 0:
                threshold = 1.0
            else:
                threshold = float(np.percentile(pool, threshold_percentile))
        else:
            pool = smoothed[frame_mask] if frame_mask is not None else smoothed
            threshold = float(threshold_otsu(pool)) if np.any(pool > 0) else 1.0

        binary = smoothed >= threshold
        if frame_mask is not None:
            binary &= frame_mask
        binary = remove_small_objects(binary, min_size=max(min_area, 2))
        binary = binary_opening(binary, disk(1))
        binary = binary_closing(binary, disk(1))

        if np.sum(binary) == 0:
            return {"labels": np.zeros_like(binary, dtype=np.int32), "objects": [], "watershed_dam_fraction": 0.0}

        distance = ndimage.distance_transform_edt(binary)
        seeds = h_maxima(smoothed, h=max(h_val, 1e-5)) & binary
        markers, n_markers = ndimage.label(seeds)
        if n_markers == 0:
            peaks = peak_local_max(distance, min_distance=3, labels=binary, exclude_border=False)
            markers = np.zeros_like(binary, dtype=np.int32)
            for idx, (rr, cc) in enumerate(peaks, start=1):
                markers[rr, cc] = idx
            if len(peaks) == 0:
                markers, _ = ndimage.label(binary)

        if topology_mode == "intensity":
            topography = 1.0 - smoothed
        else:
            # Watershed on negative distance keeps dams between touching droplets.
            topography = -distance

        labels = watershed(topography, markers=markers, mask=binary, watershed_line=True)
        labels = remove_small_objects(labels, min_size=max(min_area, 2))
        labels, _, _ = self._relabel_consecutive(labels)

        objects = self._extract_objects(labels, frame)
        boundaries = find_boundaries(labels, mode="inner")
        dam_pixels = boundaries & binary
        dam_fraction = float(np.sum(dam_pixels) / max(np.sum(binary), 1))

        return {"labels": labels, "objects": objects, "watershed_dam_fraction": dam_fraction}

    def _extract_objects(self, labels: np.ndarray, intensity: np.ndarray) -> List[Dict[str, Any]]:
        objects = []
        for region in regionprops(labels, intensity_image=intensity):
            minr, minc, maxr, maxc = region.bbox
            local_mask = labels[minr:maxr, minc:maxc] == region.label
            major = float(region.major_axis_length) if region.major_axis_length is not None else np.nan
            minor = float(region.minor_axis_length) if region.minor_axis_length is not None else np.nan
            if not np.isfinite(major):
                major = np.nan
            if not np.isfinite(minor) or minor <= 1e-9:
                minor = np.nan

            local_intensity = intensity[minr:maxr, minc:maxc]
            total_intensity = float(np.sum(local_intensity[local_mask]))
            area = float(region.area)
            equiv_radius_px = float(np.sqrt(area / np.pi)) if area > 0 else np.nan

            objects.append(
                {
                    "label": int(region.label),
                    "area": area,
                    "total_intensity": total_intensity,
                    "centroid_y": float(region.centroid[0]),
                    "centroid_x": float(region.centroid[1]),
                    "major_axis": major,
                    "minor_axis": minor,
                    "equiv_radius_px": equiv_radius_px,
                    "bbox": (int(minr), int(minc), int(maxr), int(maxc)),
                    "local_mask": local_mask,
                }
            )
        return objects

    def _relabel_consecutive(self, labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
        unique = np.unique(labels)
        unique = unique[unique > 0]
        forward: Dict[int, int] = {}
        inverse: Dict[int, int] = {}
        relabeled = np.zeros_like(labels, dtype=np.int32)
        for new_id, old_id in enumerate(unique, start=1):
            forward[int(old_id)] = int(new_id)
            inverse[int(new_id)] = int(old_id)
            relabeled[labels == old_id] = new_id
        return relabeled, forward, inverse

    def _compute_iou_matrix(self, objects_a: Sequence[Dict[str, Any]], objects_b: Sequence[Dict[str, Any]]) -> np.ndarray:
        matrix = np.zeros((len(objects_a), len(objects_b)), dtype=np.float64)
        for i, a in enumerate(objects_a):
            for j, b in enumerate(objects_b):
                matrix[i, j] = self._object_iou(a, b)
        return matrix

    def _object_iou(self, obj_a: Dict[str, Any], obj_b: Dict[str, Any]) -> float:
        ar0, ac0, ar1, ac1 = obj_a["bbox"]
        br0, bc0, br1, bc1 = obj_b["bbox"]

        ir0 = max(ar0, br0)
        ic0 = max(ac0, bc0)
        ir1 = min(ar1, br1)
        ic1 = min(ac1, bc1)
        if ir1 <= ir0 or ic1 <= ic0:
            return 0.0

        a_slice = obj_a["local_mask"][ir0 - ar0 : ir1 - ar0, ic0 - ac0 : ic1 - ac0]
        b_slice = obj_b["local_mask"][ir0 - br0 : ir1 - br0, ic0 - bc0 : ic1 - bc0]
        inter = int(np.sum(a_slice & b_slice))
        if inter <= 0:
            return 0.0

        area_a = float(obj_a["area"])
        area_b = float(obj_b["area"])
        union = area_a + area_b - inter
        if union <= 1e-9:
            return 0.0
        return float(inter / union)

    def _summarize_events(self, graph: nx.DiGraph, parameters: Dict[str, Any]) -> Dict[str, Any]:
        mass_tolerance = float(parameters.get("mass_tolerance", 0.25))
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        births = []
        nucleations = []
        fusions = []
        fissions = []
        deaths = []

        for node in graph.nodes:
            indeg = graph.in_degree(node)
            outdeg = graph.out_degree(node)
            frame = int(node[0])

            if indeg == 0 and frame > 0:
                births.append(node)
                nucleations.append(node)
            if outdeg == 0 and frame < self._max_graph_frame(graph):
                deaths.append(node)

            if indeg > 1:
                parents = list(graph.predecessors(node))
                child_mass = float(graph.nodes[node].get("total_intensity", 0.0))
                parent_mass = float(sum(graph.nodes[p].get("total_intensity", 0.0) for p in parents))
                valid = child_mass > 0 and abs(child_mass - parent_mass) / child_mass <= mass_tolerance
                fusions.append({"node": node, "valid_mass_conservation": valid})

            if outdeg > 1:
                children = list(graph.successors(node))
                parent_mass = float(graph.nodes[node].get("total_intensity", 0.0))
                child_mass = float(sum(graph.nodes[c].get("total_intensity", 0.0) for c in children))
                valid = parent_mass > 0 and abs(parent_mass - child_mass) / parent_mass <= mass_tolerance
                fissions.append({"node": node, "valid_mass_conservation": valid})

        return {
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
            "num_births": int(len(births)),
            "num_nucleations": int(len(nucleations)),
            "num_fusions": int(len(fusions)),
            "num_fissions": int(len(fissions)),
            "num_deaths": int(len(deaths)),
            "num_mass_valid_fusions": int(sum(1 for x in fusions if x["valid_mass_conservation"])),
            "num_mass_valid_fissions": int(sum(1 for x in fissions if x["valid_mass_conservation"])),
        }

    def _max_graph_frame(self, graph: nx.DiGraph) -> int:
        if graph.number_of_nodes() == 0:
            return 0
        return int(max(n[0] for n in graph.nodes))

    def _forward_chain(self, graph: nx.DiGraph, start: Tuple[int, int], max_len: int) -> List[Tuple[int, int]]:
        chain = [start]
        current = start
        for _ in range(max(0, max_len - 1)):
            children = list(graph.successors(current))
            if not children:
                break
            # Prefer highest-IoU continuation and avoid branching ambiguity.
            children_sorted = sorted(
                children, key=lambda c: float(graph.edges[current, c].get("iou", 0.0)), reverse=True
            )
            next_node = children_sorted[0]
            chain.append(next_node)
            current = next_node
        return chain

    def _fit_ar_decay(self, t: np.ndarray, ar: np.ndarray) -> Dict[str, Any]:
        valid = np.isfinite(t) & np.isfinite(ar) & (ar >= 1.0)
        if np.sum(valid) < 4:
            return {"tau_s": np.nan}
        t_fit = t[valid]
        y_fit = ar[valid]

        def model(tt, amp, tau):
            return 1.0 + amp * np.exp(-tt / tau)

        try:
            p0 = [max(float(y_fit[0] - 1.0), 0.05), max(float(t_fit[-1] / 2.0), self.time_interval_s)]
            bounds = ([0.0, 1e-6], [100.0, max(t_fit[-1] * 100.0, 1.0e4)])
            popt, _ = optimize.curve_fit(model, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=8000)
            pred = model(t_fit, *popt)
            ss_res = float(np.sum((y_fit - pred) ** 2))
            ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            return {"amplitude": float(popt[0]), "tau_s": float(popt[1]), "r_squared": float(r2)}
        except Exception:
            return {"tau_s": np.nan}

    def _extract_linear_tracks(self, graph: nx.DiGraph, min_length: int) -> List[List[Tuple[int, int]]]:
        tracks: List[List[Tuple[int, int]]] = []
        starts = [n for n in graph.nodes if graph.in_degree(n) == 0]
        for start in sorted(starts):
            chain = [start]
            current = start
            while True:
                children = list(graph.successors(current))
                if not children:
                    break
                children_sorted = sorted(
                    children, key=lambda c: float(graph.edges[current, c].get("iou", 0.0)), reverse=True
                )
                nxt = children_sorted[0]
                if graph.in_degree(nxt) > 1:
                    break
                chain.append(nxt)
                current = nxt
            if len(chain) >= min_length:
                tracks.append(chain)
        return tracks

    def _fit_power_law_growth(self, t: np.ndarray, r: np.ndarray) -> Dict[str, Any]:
        valid = np.isfinite(t) & np.isfinite(r) & (t > 0) & (r > 0)
        if np.sum(valid) < 4:
            return {"beta": np.nan, "r_squared": np.nan}
        t_fit = t[valid]
        r_fit = r[valid]

        def model(tt, a, beta):
            return a * np.power(tt, beta)

        try:
            p0 = [max(float(r_fit[0]), 1e-4), 0.4]
            bounds = ([1e-8, 0.0], [np.inf, 2.5])
            popt, _ = optimize.curve_fit(model, t_fit, r_fit, p0=p0, bounds=bounds, maxfev=8000)
            pred = model(t_fit, *popt)
            ss_res = float(np.sum((r_fit - pred) ** 2))
            ss_tot = float(np.sum((r_fit - np.mean(r_fit)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            return {"a": float(popt[0]), "beta": float(popt[1]), "r_squared": float(r2)}
        except Exception:
            return {"beta": np.nan, "r_squared": np.nan}

    def _obj_to_full_mask(self, obj: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        r0, c0, r1, c1 = obj["bbox"]
        local = obj["local_mask"]
        mask[r0:r1, c0:c1] = local
        return mask

    def _boundary_edge(self, boundary_mask: np.ndarray, band_px: int = 2) -> np.ndarray:
        m = np.asarray(boundary_mask, dtype=bool)
        if np.sum(m) == 0:
            return np.zeros_like(m, dtype=bool)
        eroded = ndimage.binary_erosion(m, iterations=max(1, band_px))
        edge = m ^ eroded
        if np.sum(edge) == 0:
            edge = find_boundaries(m, mode="outer")
        return edge

    def _estimate_contact_angle(self, object_mask: np.ndarray, boundary_mask: np.ndarray) -> float:
        obj = np.asarray(object_mask, dtype=bool)
        bnd = np.asarray(boundary_mask, dtype=bool)
        if np.sum(obj) < 20 or np.sum(bnd) < 20:
            return np.nan

        edge_obj = obj ^ ndimage.binary_erosion(obj)
        edge_bnd = self._boundary_edge(bnd, band_px=1)
        interface = edge_obj & ndimage.binary_dilation(edge_bnd, iterations=1)
        if np.sum(interface) < 5:
            return np.nan

        # Fit circle to free edge of droplet.
        free_edge = edge_obj & ~ndimage.binary_dilation(edge_bnd, iterations=1)
        yy, xx = np.where(free_edge)
        if len(yy) < 8:
            return np.nan
        circle = self._fit_circle(xx.astype(float), yy.astype(float))
        if circle is None:
            return np.nan
        cx, cy, _ = circle

        # Representative contact point.
        yi, xi = np.where(interface)
        cp = np.array([float(np.mean(yi)), float(np.mean(xi))], dtype=float)  # (y, x)

        # Droplet normal at contact from fitted circle center -> contact point.
        n_drop = cp - np.array([cy, cx], dtype=float)
        n_drop /= max(np.linalg.norm(n_drop), 1e-9)

        # Boundary normal from signed distance gradient.
        dist_in = ndimage.distance_transform_edt(bnd)
        dist_out = ndimage.distance_transform_edt(~bnd)
        signed = ndimage.gaussian_filter(dist_in - dist_out, sigma=1.0)
        gy, gx = np.gradient(signed)
        y_idx = int(np.clip(round(cp[0]), 0, bnd.shape[0] - 1))
        x_idx = int(np.clip(round(cp[1]), 0, bnd.shape[1] - 1))
        n_bnd = np.array([gy[y_idx, x_idx], gx[y_idx, x_idx]], dtype=float)
        n_bnd /= max(np.linalg.norm(n_bnd), 1e-9)

        # Contact angle (0-180): high affinity -> lower angle.
        cos_theta = np.clip(-np.dot(n_bnd, n_drop), -1.0, 1.0)
        theta = float(np.degrees(np.arccos(cos_theta)))
        return theta

    def _fit_circle(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
        if x.size < 3:
            return None
        A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
        b = x**2 + y**2
        try:
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c0 = c
            radius_sq = c0 + cx**2 + cy**2
            if radius_sq <= 0:
                return None
            return float(cx), float(cy), float(np.sqrt(radius_sq))
        except Exception:
            return None

    def _classify_wetting(self, angle_deg: float) -> str:
        if not np.isfinite(angle_deg):
            return "undetermined"
        if angle_deg < 60:
            return "high_affinity_wetting"
        if angle_deg < 120:
            return "partial_wetting"
        return "low_affinity_or_repulsive"

    def _summarize_segmentation(self, segmentation: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = np.array([len(frame["objects"]) for frame in segmentation], dtype=float)
        dams = np.array([float(frame.get("watershed_dam_fraction", 0.0)) for frame in segmentation], dtype=float)
        return {
            "mean_objects_per_frame": float(np.nanmean(counts)) if counts.size else 0.0,
            "max_objects_per_frame": int(np.nanmax(counts)) if counts.size else 0,
            "mean_watershed_dam_fraction": float(np.nanmean(dams)) if dams.size else 0.0,
        }


def get_condensate_lineage_parameters() -> Dict[str, Any]:
    """Default parameter schema for UI controls."""
    return {
        "pixel_size_um": {"default": 0.1, "min": 0.001, "max": 10.0},
        "time_interval_s": {"default": 1.0, "min": 0.001, "max": 100.0},
        "h_maxima_h": {"default": 0.08, "min": 0.005, "max": 0.5},
        "min_area_px": {"default": 12, "min": 3, "max": 500},
        "iou_threshold": {"default": 0.08, "min": 0.01, "max": 0.95},
        "mass_tolerance": {"default": 0.25, "min": 0.05, "max": 0.6},
    }

