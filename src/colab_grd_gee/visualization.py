"""Coherence and SAR/coherence RGB visualization helpers."""

from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float32).eps


def to_db(linear: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.clip(linear, 0.0, None).astype(np.float32) + EPS)).astype(np.float32)


def percentile_normalize(
    array: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Robustly stretch array values to [0, 1]."""

    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    values = array[valid]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    out = np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        out = out.copy()
        out[np.asarray(mask, dtype=bool)] = 0.0
    return out


def apply_gamma(rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    gamma = max(float(gamma), EPS)
    return np.power(rgb, 1.0 / gamma).astype(np.float32)


def make_diagnostic_rgb(
    primary_linear: np.ndarray,
    secondary_linear: np.ndarray,
    coherence: np.ndarray,
    mask: np.ndarray | None = None,
    percentile_range: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Diagnostic false color: R=t2 amplitude, G=coherence, B=t1 amplitude."""

    low, high = percentile_range
    a1 = percentile_normalize(to_db(primary_linear), low, high, mask=mask)
    a2 = percentile_normalize(to_db(secondary_linear), low, high, mask=mask)
    coh = np.clip(np.nan_to_num(coherence, nan=0.0), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        coh = coh.copy()
        coh[np.asarray(mask, dtype=bool)] = 0.0
    return np.dstack([a2, coh, a1]).astype(np.float32)


def make_pseudo_natural_rgb(
    primary_linear: np.ndarray,
    secondary_linear: np.ndarray,
    coherence: np.ndarray,
    mask: np.ndarray | None = None,
    percentile_range: tuple[float, float] = (2.0, 98.0),
    gamma: float = 0.95,
    water_strength: float = 0.22,
    green_strength: float = 0.28,
    coherence_brightness: float = 0.40,
    change_redness: float = 0.20,
) -> np.ndarray:
    """Create a SAR/coherence pseudo-RGB visualization, not an optical reconstruction."""

    low, high = percentile_range
    a1_db = to_db(primary_linear)
    a2_db = to_db(secondary_linear)
    avg_amp = percentile_normalize(0.5 * (a1_db + a2_db), low, high, mask=mask)
    change = percentile_normalize(np.abs(a2_db - a1_db), low, high, mask=mask)
    coh = np.clip(np.nan_to_num(coherence, nan=0.0), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        coh = coh.copy()
        coh[np.asarray(mask, dtype=bool)] = 0.0

    water = np.power(1.0 - avg_amp, 1.65) * (0.75 + 0.25 * coh)
    decorrelated = np.power(1.0 - coh, 1.15) * (1.0 - 0.75 * water) * (0.35 + 0.65 * avg_amp)
    stable = avg_amp * coh
    changed = change * (1.0 - 0.65 * water) * (0.35 + 0.65 * avg_amp)

    brightness = np.clip(0.05 + np.power(avg_amp, 0.88) * (0.62 + coherence_brightness * coh), 0.0, 1.0)

    # Start from a muted gray/tan SAR brightness image, then add controlled semantic tints.
    red = brightness * (1.02 + 0.10 * stable)
    green = brightness * (0.97 + 0.03 * stable)
    blue = brightness * (0.91 + 0.05 * stable)

    red *= 1.0 - 0.70 * water_strength * water
    green *= 1.0 - 0.55 * water_strength * water
    blue += 0.55 * water_strength * water * (1.0 - blue)

    red *= 1.0 - 0.30 * green_strength * decorrelated
    green += 0.38 * green_strength * decorrelated * (1.0 - green)
    blue *= 1.0 - 0.18 * green_strength * decorrelated

    red += 0.55 * change_redness * changed * (1.0 - red)
    green *= 1.0 - 0.10 * change_redness * changed
    blue *= 1.0 - 0.20 * change_redness * changed

    rgb = np.clip(np.dstack([red, green, blue]), 0.0, 1.0).astype(np.float32)

    valid = np.all(np.isfinite(rgb), axis=2)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    if np.any(valid):
        channel_means = np.maximum(np.mean(rgb[valid], axis=0), EPS)
        target = float(np.mean(channel_means))
        scale = np.clip(target / channel_means, 0.86, 1.14)
        rgb = np.clip(rgb * scale.reshape(1, 1, 3), 0.0, 1.0)

        luminance = np.clip(
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
            0.0,
            1.0,
        )
        stretched_luminance = percentile_normalize(luminance, 1.0, 99.0, mask=mask)
        rgb = np.clip(rgb * (stretched_luminance / np.maximum(luminance, EPS))[..., None], 0.0, 1.0)

    rgb = apply_gamma(rgb, gamma=gamma)
    if mask is not None:
        rgb = rgb.copy()
        rgb[np.asarray(mask, dtype=bool)] = 0.0
    return rgb.astype(np.float32)


def coherence_stats(coherence: np.ndarray) -> dict[str, float | list[float]]:
    values = np.asarray(coherence, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan"), "percentiles": []}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": [float(v) for v in np.percentile(values, [1, 5, 25, 50, 75, 95, 99])],
    }


def plot_histograms(primary_linear: np.ndarray, secondary_linear: np.ndarray, normalized_tensor: np.ndarray | None = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3 if normalized_tensor is not None else 2, figsize=(15, 4), constrained_layout=True)
    axes[0].hist(to_db(primary_linear).ravel(), bins=80, color="0.25")
    axes[0].set_title("t1 sigma0 dB")
    axes[1].hist(to_db(secondary_linear).ravel(), bins=80, color="0.25")
    axes[1].set_title("t2 sigma0 dB")
    if normalized_tensor is not None:
        axes[2].hist(normalized_tensor.ravel(), bins=80, color="0.25")
        axes[2].set_title("model-normalized inputs")
    return fig, axes
