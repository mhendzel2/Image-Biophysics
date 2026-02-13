import numpy as np

from nuclear_biophysics import NuclearBiophysicsAnalyzer


def _ar1_trace(n: int, rho: float, sigma: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    noise = rng.normal(scale=sigma, size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def test_autocorrelation_sanity_ar1_characteristic_time():
    analyzer = NuclearBiophysicsAnalyzer()
    dt = 0.1
    tau_true = 0.8
    rho = float(np.exp(-dt / tau_true))
    T, H, W = 320, 8, 8

    trace = 100.0 + _ar1_trace(T, rho=rho, sigma=2.0, seed=1)
    stack = np.repeat(trace[:, None, None], H, axis=1)
    stack = np.repeat(stack, W, axis=2)
    mask = np.ones((H, W), dtype=bool)

    res = analyzer._calculate_nuclear_correlations(
        masked_data=stack,
        nuclear_mask=mask,
        pixel_size=0.1,
        time_interval=dt,
        use_two_component=False,
        mode="roi_mean"
    )

    g = np.asarray(res["G_mean"], dtype=float)
    ratio = g[1] / (g[0] + 1e-12)
    tau_est = -dt / np.log(np.clip(ratio, 1e-6, 0.999999))

    assert np.isfinite(tau_est)
    assert 0.4 * tau_true <= tau_est <= 2.2 * tau_true


def test_regression_spatial_sampling_not_used_as_temporal_trace():
    analyzer = NuclearBiophysicsAnalyzer()
    T, H, W = 80, 20, 20
    rng = np.random.default_rng(3)
    static_frame = rng.normal(loc=100.0, scale=10.0, size=(H, W))

    # Constant in time per pixel -> temporal fluctuations are zero.
    stack = np.repeat(static_frame[None, :, :], T, axis=0)
    mask = np.ones((H, W), dtype=bool)

    res = analyzer._calculate_nuclear_correlations(
        masked_data=stack,
        nuclear_mask=mask,
        pixel_size=0.1,
        time_interval=0.1,
        use_two_component=False,
        mode="pixelwise_mean_acf"
    )
    g = np.asarray(res["G_mean"], dtype=float)

    assert np.max(np.abs(g[1:])) < 1e-8


def test_supplied_mask_is_used_not_intensity_thresholding():
    analyzer = NuclearBiophysicsAnalyzer()
    T, H, W = 120, 16, 16
    dt = 0.1
    rng = np.random.default_rng(11)

    mask = np.zeros((H, W), dtype=bool)
    mask[2:8, 2:8] = True

    # Inside-mask signal: mostly white noise around positive baseline (low temporal correlation)
    inside = 100.0 + rng.normal(0, 2.0, size=(T, np.sum(mask)))

    # Outside-mask signal: strongly correlated AR(1), should be ignored.
    rho = 0.97
    outside_count = H * W - np.sum(mask)
    outside = np.zeros((T, outside_count), dtype=float)
    outside[0] = 200.0
    noise = rng.normal(0, 0.2, size=(T, outside_count))
    for t in range(1, T):
        outside[t] = rho * outside[t - 1] + noise[t] + (1 - rho) * 200.0

    stack = np.zeros((T, H, W), dtype=float)
    flat = stack.reshape(T, -1)
    flat[:, mask.ravel()] = inside
    flat[:, ~mask.ravel()] = outside

    masked = analyzer._apply_nuclear_mask(stack, mask)
    res = analyzer._calculate_nuclear_correlations(
        masked_data=masked,
        nuclear_mask=mask,
        pixel_size=0.1,
        time_interval=dt,
        use_two_component=False,
        mode="roi_mean"
    )

    g = np.asarray(res["G_mean"], dtype=float)
    ratio = g[1] / (g[0] + 1e-12)
    assert ratio < 0.4
