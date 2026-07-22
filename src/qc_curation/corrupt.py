from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, fftconvolve, sosfiltfilt

from .audio import load_audio, match_length, peak_normalize, rms, save_audio
from .utils import sha256_file, stable_digest, stable_int, stable_order, write_json


@dataclass(frozen=True)
class ResourceBank:
    noise_train: tuple[str, ...]
    noise_test: tuple[str, ...]
    rir_train: tuple[str, ...]
    rir_test: tuple[str, ...]
    noise_ids: dict[str, str]
    rir_ids: dict[str, str]

    def choices(self, kind: str, partition: str) -> tuple[str, ...]:
        if kind not in {"noise", "rir"}:
            raise ValueError(f"Unknown resource kind: {kind}")
        if partition not in {"train", "test"}:
            raise ValueError(f"Unknown resource partition: {partition}")
        return getattr(self, f"{kind}_{partition}")

    def identifier(self, kind: str, path: str) -> str:
        if kind == "noise":
            return self.noise_ids[path]
        if kind == "rir":
            return self.rir_ids[path]
        raise ValueError(f"Unknown resource kind: {kind}")


def choose_resource(bank: ResourceBank, kind: str, partition: str, *seed_parts: object) -> str:
    choices = bank.choices(kind, partition)
    if not choices:
        raise RuntimeError(f"No {kind} resources are available in the {partition} partition.")
    return choices[stable_int(*seed_parts, kind, partition, modulo=len(choices))]


def random_crop_or_repeat(noise: torch.Tensor, length: int, *seed_parts: object) -> torch.Tensor:
    signal = match_length(noise, max(length, noise.shape[-1]))
    if signal.shape[-1] == length:
        return signal
    offset = stable_int(*seed_parts, "crop", modulo=signal.shape[-1] - length + 1)
    return signal[..., offset : offset + length]


def add_noise(clean: torch.Tensor, noise: torch.Tensor, snr_db: float, *seed_parts: object) -> torch.Tensor:
    segment = random_crop_or_repeat(noise, clean.shape[-1], *seed_parts)
    scale = rms(clean) / (rms(segment) * (10.0 ** (snr_db / 20.0)))
    return clean + segment * scale


def apply_reverb(clean: torch.Tensor, rir: torch.Tensor, wet: float) -> torch.Tensor:
    impulse = rir.squeeze(0).detach().cpu().numpy()
    impulse = impulse / max(1e-7, np.max(np.abs(impulse)))
    # Keep the first two seconds: enough late energy for the supplied RIRs while
    # avoiding an unnecessarily expensive convolution on pathological recordings.
    impulse = impulse[: min(len(impulse), 32000)]
    convolved = fftconvolve(clean.squeeze(0).detach().cpu().numpy(), impulse, mode="full")[: clean.shape[-1]]
    reverb = torch.from_numpy(np.ascontiguousarray(convolved)).to(clean.dtype).unsqueeze(0)
    reverb = reverb * (rms(clean) / rms(reverb))
    wet_mix = wet / (1.0 + wet)
    return (1.0 - wet_mix) * clean + wet_mix * reverb


def apply_bandlimit(clean: torch.Tensor, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    cutoff = min(float(cutoff_hz), sample_rate * 0.45)
    if cutoff <= 50:
        raise ValueError("Bandlimit cutoff must be greater than 50 Hz.")
    sos = butter(8, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    signal = clean.squeeze(0).detach().cpu().numpy()
    # sosfiltfilt requires a short padding margin; short utterances use causal filtering.
    if signal.shape[0] > 128:
        filtered = sosfiltfilt(sos, signal)
    else:
        from scipy.signal import sosfilt

        filtered = sosfilt(sos, signal)
    return torch.from_numpy(np.ascontiguousarray(filtered)).to(clean.dtype).unsqueeze(0)


def _recipe_path(destination: Path) -> Path:
    return destination.with_suffix(".recipe.json")


def _render_recipe(
    source_path: str | Path,
    source_sha256: str | None,
    family: str,
    parameters: dict[str, float],
    partition: str,
    sample_rate: int,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    return {
        "schema_version": 3,
        "renderer_revision": "shared-resource-counterfactual-v3",
        "source_path": str(source),
        "source_sha256": source_sha256 or sha256_file(source),
        "family": family,
        "parameters": {str(key): float(value) for key, value in sorted(parameters.items())},
        "partition": partition,
        "sample_rate": int(sample_rate),
        "seed_parts": [str(value) for value in seed_parts],
    }


def _cached_render_metadata(destination: Path, recipe: dict[str, Any]) -> dict[str, str | float] | None:
    """Return a verified cached render or fail rather than silently mixing recipes."""
    if not destination.exists():
        return None
    sidecar = _recipe_path(destination)
    if not sidecar.exists():
        raise RuntimeError(
            f"{destination} exists without an auditable render recipe. "
            "Set run.overwrite_derived_audio=true or remove that derived file."
        )
    try:
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read render recipe sidecar {sidecar}") from exc
    if cached.get("recipe") != recipe:
        raise RuntimeError(
            f"Existing derived audio {destination} was made with a different recipe. "
            "Set run.overwrite_derived_audio=true or remove the stale derived file."
        )
    metadata = cached.get("render_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Render recipe sidecar {sidecar} has no valid render_metadata mapping.")
    expected_output_sha = cached.get("output_sha256")
    observed_output_sha = sha256_file(destination)
    if not isinstance(expected_output_sha, str) or expected_output_sha != observed_output_sha:
        raise RuntimeError(
            f"Derived audio checksum does not match its recipe sidecar for {destination}. "
            "Set run.overwrite_derived_audio=true or remove the corrupted derived file."
        )
    return metadata


def render_audio(
    source_path: str | Path,
    destination: str | Path,
    family: str,
    parameters: dict[str, float],
    bank: ResourceBank,
    partition: str,
    sample_rate: int,
    seed_parts: tuple[object, ...],
    overwrite: bool,
    source_sha256: str | None = None,
) -> dict[str, str | float]:
    """Render one deterministic candidate without exposing hidden parameters to a selector."""
    destination = Path(destination)
    recipe = _render_recipe(source_path, source_sha256, family, parameters, partition, sample_rate, seed_parts)
    if not overwrite:
        cached = _cached_render_metadata(destination, recipe)
        if cached is not None:
            return cached
    clean = load_audio(source_path, sample_rate)
    rendered = clean.clone()
    metadata: dict[str, str | float] = {
        "noise_id": "",
        "rir_id": "",
        "render_seed_sha256": stable_digest(*seed_parts),
        "noise_snr_db": float(parameters.get("noise_snr_db", np.nan)),
        "reverb_wet": float(parameters.get("reverb_wet", np.nan)),
        "bandlimit_hz": float(parameters.get("bandlimit_hz", np.nan)),
    }

    if family in {"reverb", "noise_reverb"}:
        rir_path = choose_resource(bank, "rir", partition, *seed_parts)
        rir = load_audio(rir_path, sample_rate)
        rendered = apply_reverb(rendered, rir, float(parameters["reverb_wet"]))
        metadata["rir_id"] = bank.identifier("rir", rir_path)

    if family in {"noise", "noise_reverb"}:
        noise_path = choose_resource(bank, "noise", partition, *seed_parts)
        noise = load_audio(noise_path, sample_rate)
        rendered = add_noise(rendered, noise, float(parameters["noise_snr_db"]), *seed_parts)
        metadata["noise_id"] = bank.identifier("noise", noise_path)

    if family == "bandlimit":
        rendered = apply_bandlimit(rendered, sample_rate, float(parameters["bandlimit_hz"]))
    elif family not in {"noise", "reverb", "noise_reverb", "clean"}:
        raise ValueError(f"Unsupported corruption family: {family}")

    rendered = peak_normalize(rendered)
    save_audio(destination, rendered, sample_rate)
    output_sha256 = sha256_file(destination)
    metadata["variant_audio_sha256"] = output_sha256
    write_json(
        {
            "recipe": recipe,
            "render_metadata": metadata,
            "output_sha256": output_sha256,
        },
        _recipe_path(destination),
    )
    return metadata


def make_resource_bank(musan_root: Path, rirs_root: Path, global_seed: int, split: list[float]) -> ResourceBank:
    noise_candidates = sorted(
        path for path in musan_root.rglob("*.wav") if "noise" in {part.lower() for part in path.parts}
    )
    rir_candidates = sorted(
        path
        for path in rirs_root.rglob("*.wav")
        if "rir" in str(path).lower() and path.stat().st_size > 0
    )
    if not noise_candidates or not rir_candidates:
        raise RuntimeError(
            "MUSAN noise or RIR resources were not found after extraction. "
            "Check the OpenSLR archive layout in data/raw/."
        )

    train_limit = float(split[0])
    test_start = float(split[0] + split[1])

    def partition(
        paths: list[Path], label: str, root: Path
    ) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
        root = root.resolve()
        by_id = {path.resolve().relative_to(root).as_posix(): str(path.resolve()) for path in paths}
        train_ids: list[str] = []
        test_ids: list[str] = []
        for identifier in sorted(by_id):
            value = stable_int(global_seed, label, identifier, modulo=1_000_000) / 1_000_000
            if value < train_limit:
                train_ids.append(identifier)
            elif value >= test_start:
                test_ids.append(identifier)
        if not train_ids or not test_ids:
            raise RuntimeError(f"Hash partition left an empty {label} train or test split.")
        train = tuple(by_id[identifier] for identifier in stable_order(train_ids, global_seed, label, "train"))
        test = tuple(by_id[identifier] for identifier in stable_order(test_ids, global_seed, label, "test"))
        return train, test, {path: identifier for identifier, path in by_id.items()}

    noise_train, noise_test, noise_ids = partition(noise_candidates, "noise", musan_root)
    rir_train, rir_test, rir_ids = partition(rir_candidates, "rir", rirs_root)
    return ResourceBank(noise_train, noise_test, rir_train, rir_test, noise_ids, rir_ids)
