import numpy as np

from number_and_brightness import NumberAndBrightness


def test_camera_offset_correction_unbiases_mean():
    rng = np.random.default_rng(123)
    nb = NumberAndBrightness()

    T, H, W = 120, 24, 24
    true_mean_e = 60.0
    gain = 2.0  # e-/ADU
    offset = 100.0  # ADU

    signal_e = rng.poisson(lam=true_mean_e, size=(T, H, W)).astype(float)
    stack_adu = signal_e / gain + offset

    out_uncorrected = nb.analyze(stack_adu)
    out_corrected = nb.analyze(
        stack_adu,
        camera_offset_adu=offset,
        camera_gain_e_per_adu=gain,
    )

    mu_uncorr = float(np.nanmean(out_uncorrected["mean_intensity_map"]))
    mu_corr = float(np.nanmean(out_corrected["mean_intensity_map"]))

    assert abs(mu_corr - true_mean_e) < 3.0
    assert abs(mu_uncorr - true_mean_e) > 15.0


def test_read_noise_correction_reduces_brightness_inflation():
    rng = np.random.default_rng(99)
    nb = NumberAndBrightness()

    T, H, W = 150, 20, 20
    lam = 30.0
    read_noise = 5.0

    signal = rng.poisson(lam=lam, size=(T, H, W)).astype(float)
    noisy = signal + rng.normal(0, read_noise, size=(T, H, W))

    raw = nb.analyze(noisy)
    corrected = nb.analyze(noisy, read_noise_e=read_noise)

    b_raw = float(np.nanmean(raw["brightness_map"]))
    b_corr = float(np.nanmean(corrected["brightness_map"]))

    assert abs(b_corr - 1.0) < abs(b_raw - 1.0)
    assert b_corr < b_raw


def test_detrend_reduces_bleach_induced_brightness_bias():
    rng = np.random.default_rng(321)
    nb = NumberAndBrightness()

    T, H, W = 140, 20, 20
    base = rng.poisson(lam=40.0, size=(T, H, W)).astype(float)
    t = np.arange(T, dtype=float)
    bleach = np.exp(-t / 60.0)
    bleached = base * bleach[:, None, None]

    out_none = nb.analyze(bleached, detrend_method="none")
    out_detrended = nb.analyze(bleached, detrend_method="global_exponential")

    b_none = float(np.nanmean(out_none["brightness_map"]))
    b_det = float(np.nanmean(out_detrended["brightness_map"]))

    assert b_det < b_none
