import numpy as np

from reaction_diffusion import ReactionDiffusionModel, RDParameters


def test_simulate_fields_shapes_and_nonnegative():
    model = ReactionDiffusionModel(pixel_size_um=0.1, time_step_s=0.02)
    h, w, t = 20, 20, 12
    f0 = np.ones((h, w), dtype=float) * 10.0
    b0 = np.ones((h, w), dtype=float) * 5.0

    bleach = model.create_bleach_profile((h, w), center_yx=(10, 10), radius_px=4, depth=0.7)
    out = model.simulate_fields(
        initial_free=f0,
        initial_bound=b0,
        n_frames=t,
        params=RDParameters(D_free_um2_s=1.0, D_bound_um2_s=0.0, k_on_s_inv=0.1, k_off_s_inv=0.05),
        bleach_profile=bleach,
        psf_sigma_px=1.0,
    )

    assert out["free"].shape == (t, h, w)
    assert out["bound"].shape == (t, h, w)
    assert out["observed"].shape == (t, h, w)
    assert np.all(out["free"] >= 0)
    assert np.all(out["bound"] >= 0)


def test_profile_likelihood_runs():
    model = ReactionDiffusionModel(pixel_size_um=0.1, time_step_s=0.01)
    h, w, t = 16, 16, 8
    f0 = np.ones((h, w), dtype=float)
    b0 = np.zeros((h, w), dtype=float)

    sim = model.simulate_fields(
        initial_free=f0,
        initial_bound=b0,
        n_frames=t,
        params=RDParameters(D_free_um2_s=0.4, D_bound_um2_s=0.0, k_on_s_inv=0.2, k_off_s_inv=0.1),
    )

    prof = model.profile_likelihood(
        observed_stack=sim["observed"],
        initial_free=f0,
        initial_bound=b0,
        n_frames=t,
        parameter_name="D_free_um2_s",
        scan_values=np.linspace(0.2, 0.8, 4),
        noise_model="gaussian",
        sigma=0.2,
    )

    assert len(prof["scan_values"]) == 4
    assert prof["delta_nll"].shape[0] == 4
