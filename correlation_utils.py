"""
Shared correlation utilities.
"""

from __future__ import annotations

import numpy as np


def acf_fft(x: np.ndarray) -> np.ndarray:
    """ACF via FFT (biased, mean-subtracted, normalized to mean^2).

    This follows the commonly used imaging/FCS definition:
    G(tau) = <delta I(t) delta I(t+tau)> / <I>^2
    with finite-overlap correction.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)
    mu = float(x.mean())
    if abs(mu) < 1e-12:
        mu = 1e-12
    x = x - mu
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    s = np.fft.irfft(f * np.conjugate(f))[:n]
    norm = (np.arange(n, 0, -1) * (mu ** 2))
    return (s / norm).real
