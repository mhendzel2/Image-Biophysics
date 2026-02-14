"""
Trajectory Utilities for SPT Inference

Utilities to normalize trajectory inputs and extract displacement
observations with frame-gap awareness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class DisplacementDataset:
    """Container for displacement observations derived from trajectories."""

    displacements_um: np.ndarray
    dt_frames: np.ndarray
    sequence_ids: np.ndarray
    sequence_start: np.ndarray

    @property
    def n_observations(self) -> int:
        """Return number of displacement observations."""
        return int(self.displacements_um.size)


def _coerce_single_track(track: np.ndarray) -> np.ndarray:
    """
    Coerce one trajectory to (N, 3): [frame, x_um, y_um].

    Accepted formats:
    - (N, 2): interpreted as [x_um, y_um] with sequential frames
    - (N, 3): interpreted as [frame, x_um, y_um]
    """
    arr = np.asarray(track, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Each trajectory must be a 2D array.")
    if arr.shape[1] == 2:
        frames = np.arange(arr.shape[0], dtype=float)[:, None]
        arr = np.concatenate([frames, arr], axis=1)
    elif arr.shape[1] != 3:
        raise ValueError("Trajectory arrays must have shape (N,2) or (N,3).")
    if arr.shape[0] < 2:
        return np.empty((0, 3), dtype=float)

    # Ensure chronological order by frame index.
    order = np.argsort(arr[:, 0], kind="mergesort")
    return arr[order]


def normalize_tracks(trajectories: Any) -> List[np.ndarray]:
    """
    Normalize user trajectories to a list of (N,3) arrays: [frame, x_um, y_um].

    Parameters
    ----------
    trajectories:
        - list/tuple of arrays
        - single array (N,2) or (N,3)
        - pandas DataFrame with columns: frame, x, y, particle
    """
    if trajectories is None:
        raise ValueError("No trajectories provided.")

    # DataFrame-like path (trackpy output compatibility) without importing pandas.
    if hasattr(trajectories, "columns") and hasattr(trajectories, "groupby"):
        required_cols = {"frame", "x", "y", "particle"}
        available = set(getattr(trajectories, "columns"))
        missing = required_cols.difference(available)
        if missing:
            raise ValueError(
                f"Trajectory DataFrame missing required columns: {sorted(missing)}"
            )

        tracks: List[np.ndarray] = []
        for _, grp in trajectories.groupby("particle"):
            arr = grp[["frame", "x", "y"]].to_numpy(dtype=float)
            arr = _coerce_single_track(arr)
            if arr.shape[0] >= 2:
                tracks.append(arr)
        return tracks

    if isinstance(trajectories, np.ndarray):
        coerced = _coerce_single_track(trajectories)
        return [coerced] if coerced.shape[0] >= 2 else []

    if isinstance(trajectories, (list, tuple)):
        tracks = []
        for track in trajectories:
            coerced = _coerce_single_track(np.asarray(track))
            if coerced.shape[0] >= 2:
                tracks.append(coerced)
        return tracks

    raise ValueError(
        "Unsupported trajectory format. Provide ndarray, list of ndarrays, or "
        "a DataFrame with frame/x/y/particle columns."
    )


def scale_track_coordinates(tracks: Sequence[np.ndarray], scale_um: float) -> List[np.ndarray]:
    """Scale x/y coordinates from pixels to microns."""
    scaled: List[np.ndarray] = []
    for track in tracks:
        arr = np.asarray(track, dtype=float).copy()
        if arr.shape[1] != 3:
            raise ValueError("Tracks must be in [frame, x, y] format.")
        arr[:, 1:] *= float(scale_um)
        scaled.append(arr)
    return scaled


def extract_displacements(
    tracks: Sequence[np.ndarray],
    max_lag_frames: int = 1,
    allow_gap_frames: int = 0,
) -> DisplacementDataset:
    """
    Extract displacement magnitudes across tracks with optional gap support.

    Parameters
    ----------
    tracks : sequence of ndarray
        Each trajectory is (N,3) with [frame, x_um, y_um].
    max_lag_frames : int
        Maximum lag (in frames) to include.
    allow_gap_frames : int
        Additional frame gap allowed beyond each lag.
    """
    if max_lag_frames < 1:
        raise ValueError("max_lag_frames must be >= 1.")
    if allow_gap_frames < 0:
        raise ValueError("allow_gap_frames must be >= 0.")

    disp: List[float] = []
    dtf: List[int] = []
    seq_ids: List[int] = []
    seq_start: List[int] = []

    for seq_id, track in enumerate(tracks):
        arr = _coerce_single_track(track)
        if arr.shape[0] < 2:
            continue

        frames = arr[:, 0].astype(int)
        xy = arr[:, 1:]
        n = arr.shape[0]

        for i in range(n - 1):
            for j in range(i + 1, n):
                delta = int(frames[j] - frames[i])
                if delta <= 0:
                    continue
                if delta > max_lag_frames + allow_gap_frames:
                    break
                if delta < 1:
                    continue
                dxy = xy[j] - xy[i]
                r = float(np.hypot(dxy[0], dxy[1]))
                disp.append(r)
                dtf.append(delta)
                seq_ids.append(seq_id)
                seq_start.append(i)

    if not disp:
        return DisplacementDataset(
            displacements_um=np.array([], dtype=float),
            dt_frames=np.array([], dtype=int),
            sequence_ids=np.array([], dtype=int),
            sequence_start=np.array([], dtype=int),
        )

    return DisplacementDataset(
        displacements_um=np.asarray(disp, dtype=float),
        dt_frames=np.asarray(dtf, dtype=int),
        sequence_ids=np.asarray(seq_ids, dtype=int),
        sequence_start=np.asarray(seq_start, dtype=int),
    )


def extract_step_sequences(
    tracks: Sequence[np.ndarray],
    allow_gap_frames: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """
    Build per-track displacement sequences for HMM fitting.

    Returns a list of dict objects containing:
    - 'r_um': displacement magnitudes
    - 'dt_frames': frame differences for each step
    """
    if allow_gap_frames < 0:
        raise ValueError("allow_gap_frames must be >= 0.")

    sequences: List[Dict[str, np.ndarray]] = []
    max_step = 1 + allow_gap_frames

    for track in tracks:
        arr = _coerce_single_track(track)
        if arr.shape[0] < 2:
            continue

        frames = arr[:, 0].astype(int)
        xy = arr[:, 1:]

        r_values: List[float] = []
        dt_values: List[int] = []

        for i in range(arr.shape[0] - 1):
            delta = int(frames[i + 1] - frames[i])
            if delta < 1 or delta > max_step:
                continue
            dxy = xy[i + 1] - xy[i]
            r_values.append(float(np.hypot(dxy[0], dxy[1])))
            dt_values.append(delta)

        if r_values:
            sequences.append(
                {
                    "r_um": np.asarray(r_values, dtype=float),
                    "dt_frames": np.asarray(dt_values, dtype=int),
                }
            )

    return sequences


def resample_tracks(
    tracks: Sequence[np.ndarray],
    random_state: int | None = None,
) -> List[np.ndarray]:
    """Bootstrap-resample tracks with replacement."""
    if len(tracks) == 0:
        return []
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(tracks), size=len(tracks))
    return [np.asarray(tracks[i], dtype=float).copy() for i in idx]
