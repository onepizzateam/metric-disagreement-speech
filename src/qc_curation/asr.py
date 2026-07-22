from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audio import load_audio
from .utils import (
    dataframe_digest,
    normalize_transcript,
    read_csv,
    set_determinism,
    write_csv,
    write_json,
)


class CharacterTokenizer:
    """Fixed character vocabulary built once from the pre-specified configuration."""

    def __init__(self, alphabet: str):
        unique = []
        for char in alphabet:
            if char not in unique:
                unique.append(char)
        self.alphabet = "".join(unique)
        self.blank_id = 0
        self.id_to_char = [""] + list(self.alphabet)
        self.char_to_id = {char: index + 1 for index, char in enumerate(self.alphabet)}

    @property
    def vocabulary_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, transcript: str) -> list[int]:
        normalized = normalize_transcript(transcript)
        unknown = set(normalized) - set(self.char_to_id)
        if unknown:
            raise ValueError(f"Transcript contains characters outside the fixed vocabulary: {unknown}")
        return [self.char_to_id[char] for char in normalized]

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        previous = None
        chars: list[str] = []
        for token_id in token_ids:
            item = int(token_id)
            if item != self.blank_id and item != previous:
                chars.append(self.id_to_char[item])
            previous = item
        return normalize_transcript("".join(chars))


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, steps, channels = inputs.shape
        if channels != self.d_model:
            raise RuntimeError("Positional encoding dimension disagrees with encoder dimension.")
        positions = torch.arange(steps, device=inputs.device, dtype=inputs.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, channels, 2, device=inputs.device, dtype=inputs.dtype)
            * (-math.log(10000.0) / channels)
        )
        embedding = torch.zeros(steps, channels, device=inputs.device, dtype=inputs.dtype)
        embedding[:, 0::2] = torch.sin(positions * frequencies)
        embedding[:, 1::2] = torch.cos(positions * frequencies)
        return inputs + embedding.unsqueeze(0)


class CompactCTCEncoder(nn.Module):
    """Randomly initialized convolutional Transformer encoder used to avoid corpus-pretraining leakage."""

    def __init__(self, model_cfg: dict[str, Any], vocabulary_size: int):
        super().__init__()
        channels = list(model_cfg["conv_channels"])
        kernels = list(model_cfg["conv_kernels"])
        strides = list(model_cfg["conv_strides"])
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("Convolution channel, kernel, and stride lists must have the same length.")
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel, stride in zip(channels, kernels, strides):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=False),
                    nn.GroupNorm(1, out_channels),
                    nn.GELU(),
                )
            )
            in_channels = out_channels
        d_model = int(model_cfg["d_model"])
        self.feature_extractor = nn.ModuleList(layers)
        self.kernels = kernels
        self.strides = strides
        self.projection = nn.Linear(channels[-1], d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(model_cfg["nhead"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.position = SinusoidalPosition(d_model)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(model_cfg["num_layers"]), enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, vocabulary_size)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        output = lengths.clone()
        for kernel, stride in zip(self.kernels, self.strides):
            output = torch.div(output - kernel, stride, rounding_mode="floor") + 1
        return output.clamp_min(1)

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = waveforms.unsqueeze(1)
        for layer in self.feature_extractor:
            hidden = layer(hidden)
        output_lengths = self.output_lengths(waveform_lengths)
        hidden = self.projection(hidden.transpose(1, 2))
        hidden = self.position(hidden)
        steps = hidden.shape[1]
        padding = torch.arange(steps, device=hidden.device).unsqueeze(0) >= output_lengths.to(hidden.device).unsqueeze(1)
        encoded = self.encoder(hidden, src_key_padding_mask=padding)
        return self.classifier(encoded), output_lengths


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, sample_rate: int, tokenizer: CharacterTokenizer, max_seconds: float):
        self.rows = frame.sort_values("utterance_id", kind="mergesort").to_dict("records")
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if float(row["duration_seconds"]) > self.max_seconds:
            raise RuntimeError(
                f"{row['utterance_id']} exceeds max_audio_seconds. It should have been removed by the common validity filter."
            )
        waveform = load_audio(row["variant_audio_path"], self.sample_rate)
        return {
            "waveform": waveform.squeeze(0),
            "tokens": torch.tensor(self.tokenizer.encode(str(row["transcript"])), dtype=torch.long),
            "metadata": row,
        }


def collate_manifest(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveform_lengths = torch.tensor([item["waveform"].shape[-1] for item in batch], dtype=torch.long)
    waveforms = nn.utils.rnn.pad_sequence([item["waveform"] for item in batch], batch_first=True)
    target_lengths = torch.tensor([item["tokens"].shape[-1] for item in batch], dtype=torch.long)
    targets = torch.cat([item["tokens"] for item in batch])
    return {
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "metadata": [item["metadata"] for item in batch],
    }


def make_loader(frame: pd.DataFrame, cfg: dict[str, Any], tokenizer: CharacterTokenizer, device: torch.device) -> DataLoader:
    train_cfg = cfg["asr"]["train"]
    dataset = ManifestDataset(
        frame,
        int(cfg["asr"]["sample_rate"]),
        tokenizer,
        float(train_cfg["max_audio_seconds"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["run"]["num_workers"]),
        collate_fn=collate_manifest,
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["run"]["num_workers"]) > 0,
    )


def _learning_rate_lambda(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-4, (step + 1) / max(1, warmup_steps))
    return 1.0


def run_identifier(study: str, budget_hours: float, policy: str, seed: int) -> str:
    return f"{study}__{budget_hours:g}h__{policy}__seed-{seed}"


def train_one(
    cfg: dict[str, Any],
    selection_path: str | Path,
    study: str,
    budget_hours: float,
    policy: str,
    seed: int,
    device: torch.device,
) -> dict[str, str]:
    set_determinism(seed, bool(cfg["run"]["deterministic_algorithms"]))
    selected = read_csv(selection_path)
    tokenizer = CharacterTokenizer(str(cfg["asr"]["vocabulary"]))
    loader = make_loader(selected, cfg, tokenizer, device)
    if len(loader) == 0:
        raise RuntimeError(f"Selection {selection_path} cannot produce a non-empty DataLoader.")

    model = CompactCTCEncoder(cfg["asr"]["model"], tokenizer.vocabulary_size).to(device)
    train_cfg = cfg["asr"]["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_lambda(step, int(train_cfg["warmup_steps"]))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    max_steps = int(train_cfg["max_steps"])
    accumulation = int(train_cfg["gradient_accumulation"])
    run_id = run_identifier(study, budget_hours, policy, seed)
    checkpoint = Path(cfg["paths"]["checkpoints"]) / f"{run_id}.pt"
    log_path = Path(cfg["paths"]["results"]) / "training" / f"{run_id}.csv"
    log_rows: list[dict[str, float | int]] = []
    iterator = itertools.cycle(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in tqdm(range(max_steps), desc=f"Training {run_id}"):
        batch = next(iterator)
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, output_lengths = model(waveforms, waveform_lengths)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = F.ctc_loss(
                log_probs,
                targets,
                output_lengths.cpu(),
                target_lengths.cpu(),
                blank=tokenizer.blank_id,
                zero_infinity=True,
            ) / accumulation
        scaler.scale(loss).backward()
        if (step + 1) % accumulation == 0 or step + 1 == max_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if (step + 1) % max(1, int(train_cfg["eval_every_steps"])) == 0 or step == 0:
            log_rows.append(
                {
                    "step": step + 1,
                    "ctc_loss": float(loss.detach().cpu()) * accumulation,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": cfg["asr"]["model"],
            "alphabet": tokenizer.alphabet,
            "seed": seed,
            "selection_path": str(Path(selection_path).resolve()),
            "selection_sha256": dataframe_digest(selected),
            "run_id": run_id,
        },
        checkpoint,
    )
    write_csv(pd.DataFrame(log_rows), log_path)
    return {"run_id": run_id, "checkpoint": str(checkpoint), "selection_path": str(selection_path)}


def _primary_seeds(cfg: dict[str, Any], policy: str, budget_hours: float) -> list[int]:
    if (
        budget_hours == float(cfg["primary"]["confirmatory_budget_hours"])
        and policy in {"pair_random", str(cfg["primary"]["confirmatory_policy"])}
    ):
        return [int(seed) for seed in cfg["primary"]["confirmatory_seeds"]]
    return [int(seed) for seed in cfg["primary"]["exploratory_seeds"]]


def train_all(cfg: dict[str, Any], device: torch.device) -> dict[str, dict[str, str]]:
    index_path = Path(cfg["paths"]["manifests"]) / "selection_index.json"
    import json

    selection_index: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))
    runs: dict[str, dict[str, str]] = {}
    for key, selection_path in selection_index.items():
        study, budget_label, policy = key.split("/", maxsplit=2)
        budget_hours = float(budget_label.removesuffix("h"))
        if study == "primary":
            # Full eight-seed replication is reserved for the pre-specified clean,
            # 25-hour comparison. Budget sweeps retain the paired baseline and
            # selector at three exploratory seeds; component policies run only at
            # the confirmatory budget to keep the all-in-one study feasible.
            core = {"pair_random", str(cfg["primary"]["confirmatory_policy"])}
            if budget_hours != float(cfg["primary"]["confirmatory_budget_hours"]) and policy not in core:
                continue
            seeds = _primary_seeds(cfg, policy, budget_hours)
        else:
            seeds = [int(seed) for seed in cfg["natural"]["seeds"]]
        for seed in seeds:
            artifact = train_one(cfg, selection_path, study, budget_hours, policy, seed, device)
            runs[artifact["run_id"]] = artifact
    write_json(runs, Path(cfg["paths"]["results"]) / "training_index.json")
    return runs


def _word_error_counts(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    ref = normalize_transcript(reference).split()
    hyp = normalize_transcript(hypothesis).split()
    table: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            deletion = table[i - 1][j]
            insertion = table[i][j - 1]
            substitution = table[i - 1][j - 1]
            candidates = [
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
            ]
            table[i][j] = min(candidates, key=lambda item: item[0])
    errors, substitutions, deletions, insertions = table[-1][-1]
    return errors, substitutions, deletions, insertions


def load_checkpoint_model(checkpoint_path: str | Path, cfg: dict[str, Any], device: torch.device) -> tuple[CompactCTCEncoder, CharacterTokenizer]:
    artifact = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = CharacterTokenizer(str(artifact["alphabet"]))
    model = CompactCTCEncoder(artifact["model_config"], tokenizer.vocabulary_size).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, tokenizer


def evaluate_one(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    run_id: str,
    evaluation_name: str,
    evaluation_manifest: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    model, tokenizer = load_checkpoint_model(checkpoint_path, cfg, device)
    frame = read_csv(evaluation_manifest)
    loader = make_loader(frame, cfg, tokenizer, device)
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Decoding {run_id} / {evaluation_name}"):
            logits, lengths = model(batch["waveforms"].to(device), batch["waveform_lengths"].to(device))
            token_ids = logits.argmax(dim=-1).detach().cpu()
            for sequence, output_length, metadata in zip(token_ids, lengths.detach().cpu(), batch["metadata"]):
                hypothesis = tokenizer.decode_ctc(sequence[: int(output_length)].tolist())
                errors, substitutions, deletions, insertions = _word_error_counts(metadata["transcript"], hypothesis)
                predictions.append(
                    {
                        "run_id": run_id,
                        "evaluation": evaluation_name,
                        "utterance_id": metadata["utterance_id"],
                        "source_id": metadata["source_id"],
                        "reference": metadata["transcript"],
                        "hypothesis": hypothesis,
                        "reference_words": len(normalize_transcript(metadata["transcript"]).split()),
                        "errors": errors,
                        "substitutions": substitutions,
                        "deletions": deletions,
                        "insertions": insertions,
                    }
                )
    prediction_frame = pd.DataFrame(predictions)
    output = Path(cfg["paths"]["results"]) / "predictions" / run_id / f"{evaluation_name}.csv"
    write_csv(prediction_frame, output)
    errors = int(prediction_frame["errors"].sum())
    reference_words = int(prediction_frame["reference_words"].sum())
    return {
        "run_id": run_id,
        "evaluation": evaluation_name,
        "prediction_path": str(output),
        "errors": errors,
        "reference_words": reference_words,
        "wer": errors / max(1, reference_words),
    }


def evaluate_all(cfg: dict[str, Any], device: torch.device) -> pd.DataFrame:
    import json

    runs = json.loads((Path(cfg["paths"]["results"]) / "training_index.json").read_text(encoding="utf-8"))
    manifests = json.loads((Path(cfg["paths"]["manifests"]) / "manifest_index.json").read_text(encoding="utf-8"))
    evaluation_paths = {
        key.removeprefix("evaluation_"): value
        for key, value in manifests.items()
        if key.startswith("evaluation_")
    }
    rows: list[dict[str, Any]] = []
    for run_id, artifact in runs.items():
        for evaluation_name, manifest_path in evaluation_paths.items():
            row = evaluate_one(cfg, artifact["checkpoint"], run_id, evaluation_name, manifest_path, device)
            study, budget, policy, seed = run_id.split("__")
            row.update(
                {
                    "study": study,
                    "budget_hours": float(budget.removesuffix("h")),
                    "policy": policy,
                    "seed": int(seed.removeprefix("seed-")),
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(result, Path(cfg["paths"]["results"]) / "wer.csv")
    return result
