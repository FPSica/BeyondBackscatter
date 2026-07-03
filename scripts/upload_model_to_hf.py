#!/usr/bin/env python3
"""Upload a GRD/GEE model package to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ALLOWED_SUFFIXES = {".h5", ".keras", ".pb", ".yaml", ".yml", ".json", ".md"}
BLOCKED_SUFFIXES = {".tif", ".tiff", ".npy", ".npz", ".zip", ".ipynb", ".env"}
BLOCKED_DIRS = {"outputs", "data", "__pycache__", ".git", ".ipynb_checkpoints", ".cache", "hf_cache"}
BLOCKED_NAME_PARTS = {"token", "secret", "password", "credential", "credentials", "private_key", ".env"}
ALLOWED_NAME_PARTS = {"config", "normalization", "statistics", "stats", "readme", "model"}


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("--private must be true or false")


def is_safe_model_file(path: Path, model_dir: Path) -> bool:
    rel = path.relative_to(model_dir)
    parts_lower = {part.lower() for part in rel.parts}
    name_lower = path.name.lower()
    suffix = path.suffix.lower()

    if any(part in BLOCKED_DIRS for part in parts_lower):
        return False
    if suffix in BLOCKED_SUFFIXES:
        return False
    if any(blocked in name_lower for blocked in BLOCKED_NAME_PARTS):
        return False
    if path.name.startswith("."):
        return False
    if "variables" in parts_lower and name_lower.startswith("variables."):
        return True
    if suffix in ALLOWED_SUFFIXES:
        return True
    return any(part in name_lower for part in ALLOWED_NAME_PARTS)


def discover_files(model_dir: Path) -> list[Path]:
    files = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and is_safe_model_file(path, model_dir):
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo ID, e.g. FPSica/beyond-backscatter-grd-gee")
    parser.add_argument("--model-dir", required=True, help="Directory containing model files to upload")
    parser.add_argument("--private", required=True, type=parse_bool, help="Whether to create the repo as private: true/false")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = discover_files(model_dir)
    if not files:
        raise RuntimeError(f"No uploadable model files found under {model_dir}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading {len(files)} file(s) to {args.repo_id}:")
    for path in files:
        path_in_repo = path.relative_to(model_dir).as_posix()
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
        )
        print(f"  {path_in_repo}")

    print(f"Done. Model repo: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
