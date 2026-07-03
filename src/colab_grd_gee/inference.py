"""GRD/GEE input preparation and tiled PyTorch inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model_loader import ModelBundle

EPS = np.finfo(np.float32).eps


def preprocessing_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return preprocessing defaults matching the original GRD/GEE Back2Coh path."""

    config = config or {}
    pre = config.get("preprocessing", {}) if isinstance(config.get("preprocessing", {}), dict) else {}
    tiling = config.get("tiling", {}) if isinstance(config.get("tiling", {}), dict) else {}
    return {
        "db_min": float(pre.get("db_min", pre.get("clip_min_db", -20.0))),
        "db_max": float(pre.get("db_max", pre.get("clip_max_db", 0.0))),
        "channel_order": list(pre.get("channel_order", ["t1", "t2"])),
        "patch_size": int(tiling.get("patch_size", config.get("patch_size", 128))),
        "stride": int(tiling.get("stride", config.get("stride", 32))),
        "batch_size": int(tiling.get("batch_size", config.get("batch_size", 8))),
        "aggregation": str(tiling.get("aggregation", "kaiser")),
    }


def linear_sigma0_to_model_inputs(
    primary_linear: np.ndarray,
    secondary_linear: np.ndarray,
    config: dict[str, Any] | None = None,
):
    """Convert two linear sigma0 images to the GRD/GEE model tensor channels."""

    cfg = preprocessing_config(config)
    primary = np.asarray(primary_linear, dtype=np.float32)
    secondary = np.asarray(secondary_linear, dtype=np.float32)
    if primary.ndim != 2 or secondary.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")
    if primary.shape != secondary.shape:
        raise ValueError(f"Input shape mismatch: {primary.shape} vs {secondary.shape}.")
    primary = np.nan_to_num(primary, nan=0.0, posinf=0.0, neginf=0.0)
    secondary = np.nan_to_num(secondary, nan=0.0, posinf=0.0, neginf=0.0)

    def normalize(array):
        db = 10.0 * np.log10(np.clip(array, 0.0, None) + EPS)
        db = np.clip(db, cfg["db_min"], cfg["db_max"])
        return ((db - cfg["db_min"]) / (cfg["db_max"] - cfg["db_min"])).astype(np.float32), db.astype(np.float32)

    t1_norm, t1_db = normalize(primary)
    t2_norm, t2_db = normalize(secondary)
    channels = {"t1": t1_norm, "t2": t2_norm, "primary": t1_norm, "secondary": t2_norm}
    tensor = np.stack([channels[name] for name in cfg["channel_order"]], axis=0).astype(np.float32)
    return tensor, {"t1_db": t1_db, "t2_db": t2_db, "config": cfg}


def _patch_starts(length: int, patch_size: int, stride: int):
    if length <= patch_size:
        return [0]
    starts = list(range(0, max(length - patch_size, 0), stride))
    last = length - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _kaiser_window(patch_size: int) -> np.ndarray:
    one_d = np.kaiser(patch_size, 5).astype(np.float32)
    return np.outer(one_d, one_d).astype(np.float32)


def _extract_model_output(output):
    if isinstance(output, dict):
        for key in ("coherence", "mean", "pred", "prediction", "out", "output"):
            if key in output:
                output = output[key]
                break
        else:
            output = next(iter(output.values()))
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 4:
        if output.shape[1] <= 4:
            output = output[:, 0, :, :]
        else:
            output = output[:, :, :, 0]
    if output.ndim != 3:
        raise ValueError(f"Expected model output with shape [N,H,W] or [N,C,H,W], got {tuple(output.shape)}.")
    return output


def predict_tiled(
    model_bundle: ModelBundle,
    primary_linear: np.ndarray,
    secondary_linear: np.ndarray,
    mask: np.ndarray | None = None,
    patch_size: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
):
    """Run tiled GRD/GEE coherence inference with overlap aggregation."""

    model = model_bundle.model
    model.eval()
    cfg = preprocessing_config(model_bundle.config)
    patch_size = int(patch_size or cfg["patch_size"])
    stride = int(stride or cfg["stride"])
    batch_size = int(batch_size or cfg["batch_size"])

    tensor, debug = linear_sigma0_to_model_inputs(primary_linear, secondary_linear, model_bundle.config)
    channels, rows, cols = tensor.shape
    pad_rows = max(0, patch_size - rows)
    pad_cols = max(0, patch_size - cols)
    if pad_rows or pad_cols:
        tensor = np.pad(tensor, ((0, 0), (0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)
    _, padded_rows, padded_cols = tensor.shape
    row_starts = _patch_starts(padded_rows, patch_size, stride)
    col_starts = _patch_starts(padded_cols, patch_size, stride)
    starts = [(r, c) for r in row_starts for c in col_starts]

    window = _kaiser_window(patch_size) if cfg["aggregation"].lower() == "kaiser" else np.ones((patch_size, patch_size), np.float32)
    prediction = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    weights = np.zeros((padded_rows, padded_cols), dtype=np.float32)

    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch_np = np.stack(
                [tensor[:, r : r + patch_size, c : c + patch_size] for r, c in batch_starts],
                axis=0,
            )
            batch = torch.from_numpy(batch_np).to(model_bundle.device, non_blocking=True)
            out = _extract_model_output(model(batch)).detach().float().cpu().numpy()
            for patch, (r, c) in zip(out, batch_starts):
                prediction[r : r + patch_size, c : c + patch_size] += patch.astype(np.float32) * window
                weights[r : r + patch_size, c : c + patch_size] += window

    coherence = prediction / np.maximum(weights, EPS)
    coherence = coherence[:rows, :cols]
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        coherence = coherence.copy()
        coherence[np.asarray(mask, dtype=bool)] = np.nan
    debug["tensor_shape"] = (channels, rows, cols)
    debug["patch_count"] = len(starts)
    debug["patch_size"] = patch_size
    debug["stride"] = stride
    return coherence, debug
