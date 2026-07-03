"""Model loading for public GRD/GEE Back2Coh Colab inference."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    config: dict[str, Any]
    model_dir: Path
    checkpoint_path: Path


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Model config must parse to a mapping: {path}")
    return data


def _download_hf_snapshot(
    repo_id: str,
    revision: str | None,
    checkpoint_filename: str,
    config_filename: str,
) -> Path:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    allow_patterns = [
        checkpoint_filename,
        config_filename,
        "*.py",
        "*.yaml",
        "*.yml",
        "*.json",
        "README.md",
        "*normalization*",
        "*statistics*",
        "*stats*",
    ]
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            allow_patterns=allow_patterns,
        )
    )


def resolve_model_dir(
    source: str,
    hf_repo_id: str,
    hf_revision: str | None,
    checkpoint_filename: str,
    config_filename: str,
    local_model_dir: str | Path,
    gdrive_model_dir: str | Path,
) -> Path:
    """Resolve a model directory from Hugging Face Hub, Google Drive, or local path."""

    source_key = source.lower().strip().replace("-", "_")
    if source_key in {"huggingface", "hf", "hugging_face"}:
        return _download_hf_snapshot(hf_repo_id, hf_revision, checkpoint_filename, config_filename)
    if source_key in {"google_drive", "gdrive", "drive"}:
        return Path(gdrive_model_dir).expanduser().resolve()
    if source_key == "local":
        return Path(local_model_dir).expanduser().resolve()
    raise ValueError("MODEL_SOURCE must be one of: 'huggingface', 'google_drive', or 'local'.")


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") or config.get("architecture") or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("config.yaml field 'model' or 'architecture' must be a mapping.")
    return model_cfg


def _import_model_class(model_dir: Path, config: dict[str, Any]):
    model_cfg = _model_config(config)
    module_name = model_cfg.get("module") or config.get("model_module")
    class_name = model_cfg.get("class_name") or model_cfg.get("class") or config.get("model_class")
    if not module_name or not class_name:
        return None
    sys.path.insert(0, str(model_dir))
    try:
        module = importlib.import_module(str(module_name))
        return getattr(module, str(class_name))
    finally:
        try:
            sys.path.remove(str(model_dir))
        except ValueError:
            pass


def _checkpoint_state_dict(checkpoint: Any):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    return None


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def build_model_from_config(config: dict[str, Any], model_dir: Path) -> torch.nn.Module | None:
    """Instantiate the GRD/GEE model class specified by config.yaml, if present."""

    cls = _import_model_class(model_dir, config)
    if cls is None:
        return None
    kwargs = _model_config(config).get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("model.kwargs in config.yaml must be a mapping.")
    model = cls(**kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The configured model class must instantiate a torch.nn.Module.")
    return model


def load_model_bundle(
    source: str,
    hf_repo_id: str,
    hf_revision: str | None,
    checkpoint_filename: str,
    config_filename: str,
    local_model_dir: str | Path,
    gdrive_model_dir: str | Path,
    device: str | None = None,
) -> ModelBundle:
    """Load the GRD/GEE PyTorch model, config, and checkpoint without retraining."""

    model_dir = resolve_model_dir(
        source=source,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        checkpoint_filename=checkpoint_filename,
        config_filename=config_filename,
        local_model_dir=local_model_dir,
        gdrive_model_dir=gdrive_model_dir,
    )
    config_path = model_dir / config_filename
    checkpoint_path = model_dir / checkpoint_filename
    config = _read_config(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    map_location = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = build_model_from_config(config, model_dir)

    if model is None:
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
            model = checkpoint["model"]
        else:
            raise ValueError(
                "Could not construct a model. Add model.module and model.class_name to config.yaml, "
                "include the corresponding .py source file in the model directory/HF repo, or save a "
                "complete torch.nn.Module checkpoint."
            )
    else:
        state_dict = _checkpoint_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError("Checkpoint does not contain a recognizable PyTorch state_dict.")
        strict = bool(config.get("strict_load", True))
        model.load_state_dict(_strip_module_prefix(state_dict), strict=strict)

    model.to(map_location)
    model.eval()
    return ModelBundle(model=model, device=map_location, config=config, model_dir=model_dir, checkpoint_path=checkpoint_path)
