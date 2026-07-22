from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from .audio import load_audio
from .utils import dataframe_digest, read_csv, sha256_file, update_run_metadata, write_csv


class ReferenceFreeQualityScorer:
    """Pinned implementations of two no-reference metric families used only as frozen selectors."""

    def __init__(self, cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.sample_rate = int(cfg["resources"]["sample_rate"])
        self.dnsmos_enabled = bool(cfg["metrics"]["dnsmos"]["enabled"])
        self.squim_enabled = bool(cfg["metrics"]["squim"]["enabled"])
        self._dnsmos = None
        self._squim = None

        if self.dnsmos_enabled:
            from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

            self._dnsmos = deep_noise_suppression_mean_opinion_score
        if self.squim_enabled:
            bundle_name = str(cfg["metrics"]["squim"]["objective_bundle"])
            bundle = getattr(torchaudio.pipelines, bundle_name)
            self._squim = bundle.get_model().to(device).eval()

    def score_waveform(self, waveform: torch.Tensor) -> dict[str, float]:
        result: dict[str, float] = {
            "dnsmos_p808": float("nan"),
            "dnsmos_sig": float("nan"),
            "dnsmos_bak": float("nan"),
            "dnsmos_ovrl": float("nan"),
            "squim_stoi": float("nan"),
            "squim_pesq": float("nan"),
            "squim_sisdr": float("nan"),
        }
        if self._dnsmos is not None:
            values = self._dnsmos(
                waveform.squeeze(0).cpu(),
                self.sample_rate,
                bool(self.cfg["metrics"]["dnsmos"]["personalized"]),
                device=str(self.device),
            )
            flat = values.detach().cpu().reshape(-1).tolist()
            if len(flat) != 4:
                raise RuntimeError(f"DNSMOS returned {len(flat)} outputs; expected [P808, SIG, BAK, OVRL].")
            result.update(dict(zip(["dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"], map(float, flat))))
        if self._squim is not None:
            with torch.inference_mode():
                values = self._squim(waveform.to(self.device))
            if not isinstance(values, (tuple, list)) or len(values) != 3:
                raise RuntimeError("SQUIM_OBJECTIVE did not return the expected STOI, PESQ, SI-SDR triplet.")
            flat = [float(value.detach().cpu().reshape(-1)[0]) for value in values]
            result.update(dict(zip(["squim_stoi", "squim_pesq", "squim_sisdr"], flat)))
        return result


def _record_external_model_fingerprints(cfg: dict[str, Any]) -> None:
    dns_cache = Path.home() / ".torchmetrics" / "DNSMOS"
    dns_files: dict[str, str] = {}
    if dns_cache.exists():
        for path in sorted(dns_cache.rglob("*.onnx")):
            dns_files[str(path)] = sha256_file(path)
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")).resolve()
    squim_files: dict[str, str] = {}
    if torch_home.exists():
        # Torchaudio names the downloaded bundle artifact with ``squim``. Restrict
        # the fingerprint to that artifact rather than hashing unrelated models
        # that a user may have in the same shared Torch cache.
        for path in sorted(torch_home.rglob("*")):
            if path.is_file() and "squim" in path.name.lower():
                squim_files[str(path)] = sha256_file(path)
    update_run_metadata(
        cfg,
        {
            "metric_artifacts": {
                "dnsmos_onnx_sha256": dns_files,
                "torchaudio_bundle": str(cfg["metrics"]["squim"]["objective_bundle"]),
                "torchaudio_cache_root": str(torch_home),
                "torchaudio_squim_sha256": squim_files,
            }
        },
    )


def score_manifest(cfg: dict[str, Any], manifest_path: str | Path, name: str, device: torch.device) -> Path:
    frame = read_csv(manifest_path)
    if frame.empty:
        raise RuntimeError(f"Cannot score empty manifest {manifest_path}")
    scorer = ReferenceFreeQualityScorer(cfg, device)
    rows: list[dict[str, Any]] = []
    for row in tqdm(frame.to_dict("records"), desc=f"Scoring {name}"):
        waveform = load_audio(row["variant_audio_path"], int(cfg["resources"]["sample_rate"]))
        rows.append({**row, **scorer.score_waveform(waveform)})
    output = Path(cfg["paths"]["manifests"]) / "scored" / f"{name}.csv"
    scored = pd.DataFrame(rows)
    scored["input_manifest_sha256"] = dataframe_digest(frame)
    write_csv(scored, output)
    _record_external_model_fingerprints(cfg)
    return output


def score_all(cfg: dict[str, Any], device: torch.device) -> dict[str, str]:
    root = Path(cfg["paths"]["manifests"])
    results = {"primary_scored": str(score_manifest(cfg, root / "primary_candidates.csv", "primary_candidates_scored", device))}
    if cfg["natural"]["enabled"]:
        results["natural_scored"] = str(score_manifest(cfg, root / "natural_pool.csv", "natural_pool_scored", device))
    return results
