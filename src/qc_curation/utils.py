from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML and turn all declared artifact paths into absolute paths."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(REPO_ROOT)
    for name, configured in cfg["paths"].items():
        resolved = Path(configured)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        cfg["paths"][name] = str(resolved.resolve())
    resource_root = Path(cfg["resources"]["librispeech_root"])
    if not resource_root.is_absolute():
        resource_root = REPO_ROOT / resource_root
    cfg["resources"]["librispeech_root"] = str(resource_root.resolve())
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for value in cfg["paths"].values():
        Path(value).mkdir(parents=True, exist_ok=True)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is not available.")
    return device


def stable_digest(*parts: object) -> str:
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(message).hexdigest()


def stable_int(*parts: object, modulo: int = 2**32 - 1) -> int:
    return int(stable_digest(*parts)[:16], 16) % modulo


def stable_uniform(*parts: object) -> float:
    return int(stable_digest(*parts)[:16], 16) / float(16**16 - 1)


def stable_order(values: Iterable[str], *seed_parts: object) -> list[str]:
    return sorted(values, key=lambda value: stable_digest(*seed_parts, value))


def set_determinism(seed: int, enabled: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_transcript(text: str) -> str:
    """Normalize to the fixed CTC alphabet without using an external language model."""
    text = text.upper().replace("-", " ")
    text = re.sub(r"[^A-Z' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dataframe_digest(frame: pd.DataFrame) -> str:
    """Hash a manifest canonically without comparing mixed-type CSV columns.

    CSV round trips intentionally retain empty strings for optional corruption
    metadata. Sorting a mixed numeric/string column through pandas is therefore
    not portable. Canonicalize column order, render rows, and sort their textual
    CSV representations instead; manifest row order is not semantically relevant
    to this content fingerprint.
    """
    canonical = frame.reindex(columns=sorted(frame.columns))
    lines = canonical.to_csv(index=False, na_rep="<NA>", lineterminator="\n").splitlines()
    payload = "\n".join([lines[0], *sorted(lines[1:])]) + "\n" if lines else ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(destination)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def version_or_missing(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def run_metadata(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    packages = [
        "torch", "torchaudio", "torchmetrics", "numpy", "pandas", "scipy",
        "soundfile", "librosa", "onnxruntime", "matplotlib", "jiwer",
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg["_config_path"],
        "config_sha256": sha256_file(cfg["_config_path"]),
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "package_versions": {name: version_or_missing(name) for name in packages},
    }


def update_run_metadata(cfg: dict[str, Any], updates: dict[str, Any]) -> None:
    path = Path(cfg["paths"]["results"]) / "run_metadata.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    write_json(current, path)


def source_seconds_to_budget(hours: float) -> float:
    if hours <= 0:
        raise ValueError("A curation budget must be positive.")
    return hours * 3600.0


def select_exact_duration(frame: pd.DataFrame, ordered_indices: Iterable[int], budget_seconds: float) -> pd.DataFrame:
    """Select prefix rows without exceeding a fixed duration except when one row is unavoidable."""
    chosen: list[int] = []
    total = 0.0
    for index in ordered_indices:
        duration = float(frame.loc[index, "duration_seconds"])
        if total + duration <= budget_seconds or not chosen:
            chosen.append(index)
            total += duration
        if total >= budget_seconds:
            break
    if not chosen:
        raise RuntimeError("No utterance could be selected under the requested duration budget.")
    result = frame.loc[chosen].copy()
    result["selected_duration_seconds"] = total
    return result


def lexical_coverage(selected: pd.Series, universe: pd.Series) -> dict[str, float | int]:
    def words(texts: pd.Series) -> set[str]:
        return {word for text in texts for word in normalize_transcript(str(text)).split()}

    selected_words = words(selected)
    universe_words = words(universe)
    coverage = len(selected_words & universe_words) / max(1, len(universe_words))
    return {"unique_word_types": len(selected_words), "lexical_type_coverage": coverage}
