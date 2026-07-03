"""Model loading for public GRD/GEE Back2Coh Colab inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelBundle:
    model: Any
    framework: str
    config: dict[str, Any]
    model_dir: Path
    weights_path: Path
    runtime: str


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Model config must parse to a mapping: {path}")
    return data


def _download_hf_snapshot(
    repo_id: str,
    revision: str | None,
    weights_filename: str,
    config_filename: str,
) -> Path:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    allow_patterns = [
        weights_filename,
        config_filename,
        "README.md",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.h5",
        "*.keras",
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
    weights_filename: str,
    config_filename: str,
    local_model_dir: str | Path,
    gdrive_model_dir: str | Path,
) -> Path:
    """Resolve a model directory from Hugging Face Hub, Google Drive, or local path."""

    source_key = source.lower().strip().replace("-", "_")
    if source_key in {"huggingface", "hf", "hugging_face"}:
        return _download_hf_snapshot(hf_repo_id, hf_revision, weights_filename, config_filename)
    if source_key in {"google_drive", "gdrive", "drive"}:
        return Path(gdrive_model_dir).expanduser().resolve()
    if source_key == "local":
        return Path(local_model_dir).expanduser().resolve()
    raise ValueError("MODEL_SOURCE must be one of: 'huggingface', 'google_drive', or 'local'.")


def configure_tensorflow():
    """Import TensorFlow and enable memory growth when GPUs are present."""

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


def _architecture_config(config: dict[str, Any]) -> dict[str, Any]:
    arch = config.get("architecture") or config.get("model") or {}
    if not isinstance(arch, dict):
        raise ValueError("config.yaml field 'architecture' must be a mapping.")
    return arch


def build_tensorflow_model(config: dict[str, Any]):
    """Build the TensorFlow/Keras GRD/GEE Back2Coh model from config."""

    from .tf_model import build_resunet

    arch = _architecture_config(config)
    input_shape = tuple(arch.get("input_shape", [128, 128, 2]))
    output_channels = int(arch.get("output_channels", 2))
    name = str(arch.get("name", "resunet")).lower()
    if name not in {"resunet", "back2coh_resunet", "back2coh_grd_gee_resunet"}:
        raise ValueError(f"Unsupported GRD/GEE TensorFlow architecture: {name}")
    return build_resunet(input_shape=input_shape, output_channels=output_channels)


def _decode_h5_attr_values(values) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_legacy_keras_h5_weights_by_name(model, weights_path: str | Path) -> int:
    """Load Keras 2.x H5 weights by layer name.

    Keras 3 can reject legacy Keras 2 ``model.weights.h5`` files when using
    ``model.load_weights`` directly. The released GRD/GEE file stores layer
    names and weight names explicitly, so loading by name preserves the exact
    architecture/weight mapping without converting the model format.
    """

    import h5py

    weights_path = Path(weights_path)
    loaded_layers = 0
    with h5py.File(weights_path, "r") as handle:
        layer_names = _decode_h5_attr_values(handle.attrs.get("layer_names", []))
        for layer_name in layer_names:
            if layer_name not in handle:
                continue
            group = handle[layer_name]
            weight_names = _decode_h5_attr_values(group.attrs.get("weight_names", []))
            if not weight_names:
                continue
            try:
                layer = model.get_layer(layer_name)
            except ValueError:
                continue
            weights = [group[weight_name][()] for weight_name in weight_names]
            if len(weights) != len(layer.weights):
                raise ValueError(
                    f"Layer {layer_name!r} expected {len(layer.weights)} weights but "
                    f"the H5 file provides {len(weights)}."
                )
            layer.set_weights(weights)
            loaded_layers += 1
    if loaded_layers == 0:
        raise ValueError(f"No matching Keras layer weights were loaded from {weights_path}.")
    return loaded_layers


def load_tensorflow_weights(model, weights_path: str | Path) -> str:
    """Load TensorFlow/Keras weights and return the loading method used."""

    weights_path = Path(weights_path)
    try:
        model.load_weights(str(weights_path))
        return "keras_load_weights"
    except Exception as exc:
        if weights_path.suffix.lower() != ".h5":
            raise
        loaded_layers = load_legacy_keras_h5_weights_by_name(model, weights_path)
        return f"legacy_keras_h5_by_name:{loaded_layers}_layers"


def load_model_bundle(
    source: str,
    hf_repo_id: str,
    hf_revision: str | None,
    weights_filename: str,
    config_filename: str,
    local_model_dir: str | Path,
    gdrive_model_dir: str | Path,
    framework: str = "tensorflow",
) -> ModelBundle:
    """Load the GRD/GEE TensorFlow/Keras model and weights without retraining."""

    framework_key = framework.lower().strip().replace("-", "_")
    if framework_key not in {"tensorflow", "keras", "tensorflow_keras"}:
        raise ValueError("Only the TensorFlow/Keras GRD/GEE model is supported by the public default workflow.")

    model_dir = resolve_model_dir(
        source=source,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        weights_filename=weights_filename,
        config_filename=config_filename,
        local_model_dir=local_model_dir,
        gdrive_model_dir=gdrive_model_dir,
    )
    config_path = model_dir / config_filename
    weights_path = model_dir / weights_filename
    config = _read_config(config_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"TensorFlow/Keras weights not found: {weights_path}")

    tf = configure_tensorflow()
    model = build_tensorflow_model(config)
    load_method = load_tensorflow_weights(model, weights_path)

    runtime = f"tensorflow {tf.__version__}; weights={load_method}"
    return ModelBundle(
        model=model,
        framework="tensorflow",
        config=config,
        model_dir=model_dir,
        weights_path=weights_path,
        runtime=runtime,
    )
