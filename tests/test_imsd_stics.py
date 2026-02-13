import numpy as np
from scipy import ndimage

from image_correlation_spectroscopy import ImageCorrelationSpectroscopy


def _simulate_diffusing_particles(T=80, H=48, W=48, n_particles=120, D_um2_s=0.18, dt=0.2, pixel_size_um=0.1, psf_sigma_px=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, W, size=n_particles)
    y = rng.uniform(0, H, size=n_particles)
    step_sigma_px = np.sqrt(2 * D_um2_s * dt) / pixel_size_um

    stack = np.zeros((T, H, W), dtype=float)
    for t in range(T):
        # Render particles to pixels
        xi = np.clip(np.floor(x).astype(int), 0, W - 1)
        yi = np.clip(np.floor(y).astype(int), 0, H - 1)
        frame = np.zeros((H, W), dtype=float)
        np.add.at(frame, (yi, xi), 1.0)
        frame = ndimage.gaussian_filter(frame, sigma=psf_sigma_px, mode="constant")
        frame += 0.05 * rng.normal(size=(H, W))
        frame += 5.0
        stack[t] = frame

        # Brownian step with periodic boundary conditions
        x = (x + rng.normal(0, step_sigma_px, size=n_particles)) % W
        y = (y + rng.normal(0, step_sigma_px, size=n_particles)) % H

    return stack


def test_imsd_static_field_has_near_zero_slope():
    ics = ImageCorrelationSpectroscopy()
    rng = np.random.default_rng(22)
    H, W, T = 40, 40, 70

    base = ndimage.gaussian_filter(rng.normal(size=(H, W)), sigma=2.0)
    base = base - base.min() + 10.0
    stack = np.repeat(base[None, :, :], T, axis=0)
    stack += 0.02 * rng.normal(size=stack.shape)

    out = ics.compute_imsd_stics(
        image_stack=stack,
        max_temporal_lag=10,
        max_spatial_lag=6,
        pixel_size_um=0.1,
        time_interval_s=0.2,
    )

    assert out["status"] == "success"
    assert np.isfinite(out["diffusion_coefficient_um2_s"])
    assert out["diffusion_coefficient_um2_s"] < 0.03


def test_imsd_diffusion_simulation_recovers_D_with_loose_tolerance():
    ics = ImageCorrelationSpectroscopy()
    D_true = 0.18
    stack = _simulate_diffusing_particles(D_um2_s=D_true, seed=7)

    out = ics.compute_imsd_stics(
        image_stack=stack,
        max_temporal_lag=10,
        max_spatial_lag=6,
        pixel_size_um=0.1,
        time_interval_s=0.2,
    )

    assert out["status"] == "success"
    D_est = float(out["diffusion_coefficient_um2_s"])
    assert np.isfinite(D_est)
    assert 0.3 * D_true <= D_est <= 3.0 * D_true


def test_imsd_fit_center_is_constrained_near_zero():
    ics = ImageCorrelationSpectroscopy()
    stack = _simulate_diffusing_particles(D_um2_s=0.12, seed=12)

    out = ics.compute_imsd_stics(
        image_stack=stack,
        max_temporal_lag=8,
        max_spatial_lag=5,
        pixel_size_um=0.1,
        time_interval_s=0.2,
    )

    assert out["status"] == "success"
    diags = out["fit_diagnostics"]
    converged = [d for d in diags if d.get("converged")]
    assert len(converged) > 0

    for d in converged:
        assert abs(float(d["x0_um"])) <= 0.051
        assert abs(float(d["y0_um"])) <= 0.051
