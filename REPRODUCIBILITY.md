# Reproducibility and audit protocol

## Scope

The artifact is designed so that `bash experiments/run.sh` is sufficient to recreate every data-derived output. It does not execute any hidden notebook, require a manual download, or depend on a proprietary metric service. Raw third-party assets are downloaded into `data/raw/`; all rendered audio, decisions, checkpoints, predictions, tables, and figures are written below the repository and are ignored by version control.

## Determinism

The renderer derives every sampling decision from a stable SHA-256 seed of the global seed, source utterance identifier, and operation name. This includes archive-relative resource splitting, noise/RIR choice, crop offset, candidate-slot permutation, and random selector decisions. Candidate-slot permutation is deliberately separate from the shared pair rendering seed, so low/high members use the same sampled resource/crop. Training uses a fresh seed for each paired run, a fixed utterance-ID order, fixed step count, and no training-time augmentation. PyTorch deterministic algorithms are requested in warning mode: this favors a completed run if an installed GPU kernel lacks a deterministic implementation, but it is a traceable best-effort control rather than a bitwise guarantee. A run records the effective configuration, software versions, CUDA availability, external-model cache files, and SHA-256 values in `results/run_metadata.json`.

Exact bitwise equality across GPU architectures is not promised: some CUDA kernels have hardware-dependent reductions. The artifact instead provides traceable seed-level replication, immutable manifests, and recorded package/model fingerprints. Re-running an already materialized manifest never re-renders audio unless `run.overwrite_derived_audio` is explicitly set to `true`.

## Leakage protections

- The recognizer is randomly initialized; no LibriSpeech-pretrained ASR checkpoint is used.
- Evaluation source speakers are disjoint under LibriSpeech's standard splits.
- MUSAN and RIR files are hash-partitioned before rendering; test resources are never available to candidate rendering.
- The selector receives only candidate waveforms and allowed source identifiers. It never reads clean audio, generated condition labels, SNR, RIR identity, or transcripts.
- Oracle severity is a diagnostic policy only and is excluded from the pre-specified primary analysis.

## Decision artifacts

Each row in the candidate manifest contains the raw source path and checksum, rendered-audio checksum, source/speaker/chapter IDs, transcript, source duration, candidate path, family, seeded generation parameters, archive-relative resource IDs, hidden severity rank, metric outputs, and selector decision. Each rendered WAV also has a recipe sidecar with its output checksum; cache reuse verifies both the recipe and checksum rather than trusting a filename. Each training run stores its input manifest checksum and produces per-utterance decoded predictions. These joins make it possible to audit every reported WER back to raw audio.

## Statistical plan

The pre-specified primary analysis is `pair_dnsmos_ovrl` versus `pair_random` on clean evaluation at the 25-hour budget, paired across the eight listed training seeds. It is pre-specified in the checked-in configuration before execution, not externally preregistered. The seed is the inferential unit. The artifact reports seed-level WERs, an exact two-sided paired sign-flip p-value (all $2^8$ signs, with zero differences retained), and hierarchical paired-bootstrap intervals that resample seed pairs and evaluation speakers. Metric components, alternate metrics, other budgets, robustness conditions, and all global-filtering policies are exploratory and receive Holm adjustment within their reported result family.

## Compute note

The full plan intentionally has a material compute footprint. It is appropriate to run on a CUDA workstation. A CPU run remains supported but can be slow. `configs/smoke.yaml` exists exclusively to test an installation and must not be substituted for the pre-specified study.
