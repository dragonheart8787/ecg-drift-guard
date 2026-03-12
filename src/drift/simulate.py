"""Drift injection — generate degraded copies of ECG data.

Supports three intensity levels (S1=mild, S2=moderate, S3=severe) per scenario.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample

# ── Intensity presets ────────────────────────────────────────────────────
# Each scenario maps intensity label → kwargs override
INTENSITY_PRESETS: dict[str, dict[str, dict]] = {
    "noise": {
        "S1": {"snr_db": 20.0},
        "S2": {"snr_db": 10.0},
        "S3": {"snr_db": 3.0},
    },
    "resample": {
        "S1": {"to_fs": 320},
        "S2": {"to_fs": 250},
        "S3": {"to_fs": 180},
    },
    "gain": {
        "S1": {"scale_range": (0.9, 1.1), "offset_range": (-0.03, 0.03)},
        "S2": {"scale_range": (0.7, 1.3), "offset_range": (-0.1, 0.1)},
        "S3": {"scale_range": (0.5, 1.5), "offset_range": (-0.2, 0.2)},
    },
}


def get_intensity_kwargs(scenario: str, intensity: str) -> dict:
    """Return kwargs for a given scenario + intensity level."""
    return INTENSITY_PRESETS.get(scenario, {}).get(intensity, {})


# ── Core drift functions ─────────────────────────────────────────────────

def apply_noise(
    X: np.ndarray,
    snr_db: float = 10.0,
    powerline_hz: float = 60.0,
    fs: int = 360,
    baseline_wander: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Add Gaussian noise + optional 60 Hz powerline + baseline wander.

    Parameters
    ----------
    X : (N, L, 1) float32
    snr_db : desired signal-to-noise ratio in dB
    """
    rng = np.random.RandomState(seed)
    Xd = X.copy()
    N, L, _ = Xd.shape
    t = np.arange(L) / fs

    for i in range(N):
        sig = Xd[i, :, 0]
        power_sig = np.mean(sig ** 2) + 1e-12
        noise_power = power_sig / (10 ** (snr_db / 10))

        noise = rng.normal(0, np.sqrt(noise_power), L).astype(np.float32)

        amp_pl = rng.uniform(0.01, 0.05)
        noise += amp_pl * np.sin(2 * np.pi * powerline_hz * t).astype(np.float32)

        if baseline_wander:
            amp_bw = rng.uniform(0.02, 0.1)
            freq_bw = rng.uniform(0.1, 0.5)
            noise += amp_bw * np.sin(2 * np.pi * freq_bw * t).astype(np.float32)

        Xd[i, :, 0] = sig + noise
    return Xd


def apply_resample(
    X: np.ndarray,
    from_fs: int = 360,
    to_fs: int = 250,
) -> np.ndarray:
    """Resample each beat from *from_fs* to *to_fs* then back to original length."""
    N, L, C = X.shape
    target_len = int(L * to_fs / from_fs)
    Xd = np.empty_like(X)
    for i in range(N):
        down = resample(X[i, :, 0], target_len)
        Xd[i, :, 0] = resample(down, L)
    return Xd


def apply_gain(
    X: np.ndarray,
    scale_range: tuple[float, float] = (0.7, 1.3),
    offset_range: tuple[float, float] = (-0.1, 0.1),
    seed: int = 42,
) -> np.ndarray:
    """Random per-beat gain scaling + DC offset shift."""
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    scales = rng.uniform(*scale_range, size=(N, 1, 1)).astype(np.float32)
    offsets = rng.uniform(*offset_range, size=(N, 1, 1)).astype(np.float32)
    return X * scales + offsets


def apply_prior_shift(
    y: np.ndarray,
    target_dist: dict[int, float],
    seed: int = 42,
) -> np.ndarray:
    """Sub-sample indices to match a shifted class prior distribution."""
    rng = np.random.RandomState(seed)
    classes = sorted(target_dist.keys())
    indices_per_class = {c: np.where(y == c)[0] for c in classes}

    total = sum(target_dist.values())
    n_target = min(
        int(len(indices_per_class[c]) / (target_dist[c] / total + 1e-12))
        for c in classes if target_dist[c] > 0
    )

    selected: list[np.ndarray] = []
    for c in classes:
        n_c = max(1, int(n_target * target_dist[c] / total))
        pool = indices_per_class[c]
        chosen = rng.choice(pool, size=min(n_c, len(pool)), replace=False)
        selected.append(chosen)

    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return idx


# ── Convenience: apply scenario with intensity ───────────────────────────

SCENARIO_FN = {
    "noise": apply_noise,
    "resample": apply_resample,
    "gain": apply_gain,
}


def apply_scenario(
    X: np.ndarray,
    scenario: str,
    intensity: str = "S2",
    fs: int = 360,
) -> np.ndarray:
    """Apply a named scenario at a given intensity level."""
    fn = SCENARIO_FN[scenario]
    kw = get_intensity_kwargs(scenario, intensity)

    if scenario == "resample":
        return fn(X, from_fs=fs, **kw)
    elif scenario == "noise":
        return fn(X, fs=fs, **kw)
    else:
        return fn(X, **kw)
