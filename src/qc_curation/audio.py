from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: str | Path, target_sample_rate: int, max_seconds: float | None = None) -> torch.Tensor:
    """Load a mono float waveform using SoundFile, then resample deterministically."""
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform_t = torch.from_numpy(np.ascontiguousarray(waveform.T))
    waveform_t = waveform_t.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform_t = torchaudio.functional.resample(waveform_t, sample_rate, target_sample_rate)
    if max_seconds is not None:
        waveform_t = waveform_t[..., : int(target_sample_rate * max_seconds)]
    return waveform_t.contiguous()


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = waveform.detach().cpu().squeeze(0).clamp(-0.999, 0.999).numpy()
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    sf.write(str(temporary), clipped, sample_rate, subtype="PCM_16")
    temporary.replace(destination)


def audio_info(path: str | Path) -> tuple[int, int]:
    info = sf.info(str(path))
    return int(info.samplerate), int(info.frames)


def peak_normalize(waveform: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    maximum = waveform.abs().amax().clamp_min(1e-7)
    return waveform * min(1.0, peak / float(maximum))


def rms(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.pow(2).mean().sqrt().clamp_min(1e-7)


def match_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[-1] == length:
        return waveform
    if waveform.shape[-1] > length:
        return waveform[..., :length]
    repeats = (length + waveform.shape[-1] - 1) // waveform.shape[-1]
    return waveform.repeat(1, repeats)[..., :length]
