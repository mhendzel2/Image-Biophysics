"""
Trajectory Representation Learning Utilities

Includes:
- Synthetic trajectory generation with domain randomization
- Feature embedding baseline
- Optional transformer encoder scaffold for contrastive pretraining
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SyntheticTrajectoryConfig:
    """Config for synthetic trajectory generation."""

    n_steps: int = 64
    dt_s: float = 0.01
    pixel_size_um: float = 0.1
    localization_error_um_range: Tuple[float, float] = (0.015, 0.05)
    diffusion_um2_s_range: Tuple[float, float] = (0.005, 3.0)
    directed_velocity_um_s_range: Tuple[float, float] = (0.0, 1.2)
    confinement_radius_um_range: Tuple[float, float] = (0.15, 0.8)
    anomalous_alpha_range: Tuple[float, float] = (0.35, 0.95)
    binding_k_on_range: Tuple[float, float] = (0.2, 8.0)
    binding_k_off_range: Tuple[float, float] = (0.2, 8.0)
    random_state: Optional[int] = None


def _sample_range(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def _simulate_step_noise(
    rng: np.random.Generator,
    diffusion_um2_s: float,
    dt_s: float,
) -> np.ndarray:
    sigma = np.sqrt(max(2.0 * diffusion_um2_s * dt_s, 1e-12))
    return rng.normal(0.0, sigma, size=2)


def _add_localization_noise(
    rng: np.random.Generator,
    xy: np.ndarray,
    sigma_um: float,
) -> np.ndarray:
    return xy + rng.normal(0.0, sigma_um, size=xy.shape)


def simulate_trajectory(
    mode: str,
    config: SyntheticTrajectoryConfig,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulate one trajectory in microns.

    Returns
    -------
    trajectory : ndarray (N,2)
        Position trace [x_um, y_um].
    metadata : dict
        Ground-truth parameters used for simulation.
    """
    rng = np.random.default_rng(config.random_state if random_state is None else random_state)
    n = max(int(config.n_steps), 4)
    dt = float(config.dt_s)
    sigma_loc = _sample_range(rng, config.localization_error_um_range)

    xy = np.zeros((n, 2), dtype=float)
    meta: Dict[str, float] = {"localization_error_um": sigma_loc}

    if mode == "brownian":
        d = _sample_range(rng, config.diffusion_um2_s_range)
        for t in range(1, n):
            xy[t] = xy[t - 1] + _simulate_step_noise(rng, d, dt)
        meta["diffusion_um2_s"] = d

    elif mode == "confined":
        d = _sample_range(rng, config.diffusion_um2_s_range)
        radius = _sample_range(rng, config.confinement_radius_um_range)
        k_restore = 2.0 * d / max(radius**2, 1e-6)
        for t in range(1, n):
            drift = -k_restore * xy[t - 1] * dt
            xy[t] = xy[t - 1] + drift + _simulate_step_noise(rng, d, dt)
            r = np.linalg.norm(xy[t])
            if r > radius:
                xy[t] *= radius / max(r, 1e-9)
        meta.update({"diffusion_um2_s": d, "confinement_radius_um": radius})

    elif mode == "anomalous":
        d = _sample_range(rng, config.diffusion_um2_s_range)
        alpha = _sample_range(rng, config.anomalous_alpha_range)
        for t in range(1, n):
            scale = (t + 1) ** ((alpha - 1.0) / 2.0)
            xy[t] = xy[t - 1] + _simulate_step_noise(rng, d, dt) * scale
        meta.update({"diffusion_um2_s": d, "alpha": alpha})

    elif mode == "directed":
        d = _sample_range(rng, config.diffusion_um2_s_range)
        speed = _sample_range(rng, config.directed_velocity_um_s_range)
        theta = float(rng.uniform(0, 2 * np.pi))
        velocity = speed * np.array([np.cos(theta), np.sin(theta)])
        for t in range(1, n):
            xy[t] = xy[t - 1] + velocity * dt + _simulate_step_noise(rng, d, dt)
        meta.update({"diffusion_um2_s": d, "speed_um_s": speed})

    elif mode == "binding":
        d_free = _sample_range(rng, config.diffusion_um2_s_range)
        d_bound = max(0.01 * d_free, 0.0005)
        k_on = _sample_range(rng, config.binding_k_on_range)
        k_off = _sample_range(rng, config.binding_k_off_range)
        state = 0
        states = np.zeros(n, dtype=int)
        for t in range(1, n):
            if state == 0 and rng.uniform() < 1.0 - np.exp(-k_on * dt):
                state = 1
            elif state == 1 and rng.uniform() < 1.0 - np.exp(-k_off * dt):
                state = 0
            states[t] = state
            d = d_bound if state == 1 else d_free
            xy[t] = xy[t - 1] + _simulate_step_noise(rng, d, dt)
        meta.update(
            {
                "diffusion_free_um2_s": d_free,
                "diffusion_bound_um2_s": d_bound,
                "k_on_per_s": k_on,
                "k_off_per_s": k_off,
            }
        )
    else:
        raise ValueError(f"Unknown trajectory mode: {mode}")

    xy_noisy = _add_localization_noise(rng, xy, sigma_loc)
    return xy_noisy, meta


def generate_synthetic_dataset(
    n_per_class: int,
    config: SyntheticTrajectoryConfig,
    classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a domain-randomized synthetic dataset for trajectory classification.
    """
    if classes is None:
        classes = ["brownian", "confined", "anomalous", "directed", "binding"]

    trajectories: List[np.ndarray] = []
    labels: List[str] = []
    metadata: List[Dict[str, float]] = []

    rng = np.random.default_rng(config.random_state)
    for cls in classes:
        for _ in range(int(n_per_class)):
            seed = int(rng.integers(0, 2**31 - 1))
            traj, meta = simulate_trajectory(cls, config, random_state=seed)
            trajectories.append(traj)
            labels.append(cls)
            metadata.append(meta)

    return {
        "trajectories": trajectories,
        "labels": labels,
        "metadata": metadata,
        "classes": list(classes),
    }


class TrajectoryFeatureEmbedder:
    """
    Lightweight trajectory embedder for rapid baselines and calibration checks.
    """

    def __init__(self, dt_s: float):
        self.dt_s = float(dt_s)

    def transform(self, trajectories: Sequence[np.ndarray]) -> np.ndarray:
        feats = [self._features_one(np.asarray(traj, dtype=float)) for traj in trajectories]
        return np.vstack(feats) if feats else np.empty((0, 0), dtype=float)

    def _features_one(self, xy: np.ndarray) -> np.ndarray:
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 4:
            raise ValueError("Each trajectory must have shape (N,2) with N>=4.")
        dxy = np.diff(xy, axis=0)
        step = np.linalg.norm(dxy, axis=1)
        r = np.linalg.norm(xy - xy[0], axis=1)
        msd1 = float(np.mean(step**2))
        msd2 = float(np.mean(np.sum((xy[2:] - xy[:-2]) ** 2, axis=1)))
        alpha_proxy = np.log(max(msd2, 1e-12) / max(msd1, 1e-12)) / np.log(2.0)
        turning = np.sum(dxy[:-1] * dxy[1:], axis=1) / (
            np.linalg.norm(dxy[:-1], axis=1) * np.linalg.norm(dxy[1:], axis=1) + 1e-12
        )
        straightness = float(np.linalg.norm(xy[-1] - xy[0]) / (np.sum(step) + 1e-12))
        return np.array(
            [
                float(np.mean(step)),
                float(np.std(step)),
                msd1,
                msd2,
                float(alpha_proxy),
                float(np.mean(turning)),
                straightness,
                float(np.max(r)),
            ],
            dtype=float,
        )


try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


if TORCH_AVAILABLE:

    class TrajectoryTransformerEncoder(nn.Module):
        """
        Minimal transformer encoder scaffold for trajectory embeddings.

        Input shape: (batch, seq_len, 2)
        Output shape: (batch, embed_dim)
        """

        def __init__(
            self,
            embed_dim: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_proj = nn.Linear(2, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.input_proj(x)
            z = self.encoder(z)
            z = self.norm(z)
            return z.mean(dim=1)

else:

    class TrajectoryTransformerEncoder:  # type: ignore[no-redef]
        """Placeholder that explains missing dependency."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "PyTorch is required for TrajectoryTransformerEncoder. "
                "Install torch to use transformer-based representation learning."
            )
