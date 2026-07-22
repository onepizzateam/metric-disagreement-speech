# Does speech quality rank ASR training utility?

This repository is the complete artifact for the accompanying Interspeech 2027 workshop manuscript, **“Does Speech Quality Rank ASR Training Utility? Counterfactual Evaluation of Source-Calibrated Data Curation.”** It tests a deliberately narrow claim: a no-reference perceptual-quality score should not be assumed to rank the utility of a recording for training ASR merely because it ranks listener-perceived quality.

The artifact begins with raw audio and produces immutable manifests, quality scores, fixed-budget training sets, compact CTC recognizers, decoded predictions, bootstrap statistics, figures, CSV tables, and a `paper/generated/RESULTS_TO_PASTE.md` guide. It never bundles corpus audio or model weights.

## One-command reproduction

From this directory, run:

```bash
bash experiments/run.sh
```

The script creates an isolated Python environment, downloads the public resources, then executes every stage. It is intentionally conservative: it will stop on a failed download, a missing external command, or a manifest inconsistency rather than silently continue. Set `PYTHON_BIN`, `DEVICE`, or `STUDY_CONFIG` before invoking the script to override the defaults.

The default study is computationally substantial. It runs the pre-specified primary paired DNSMOS-OVRL comparison with eight paired seeds and the exploratory policies with three seeds. A small smoke-test configuration is supplied at `configs/smoke.yaml`; it checks plumbing only and must not be used for paper numbers.

## GitHub Codespaces

The included `.devcontainer/devcontainer.json` starts a Python 3.11 Codespace. Open the repository in Codespaces, open its terminal, and run the same command:

```bash
bash experiments/run.sh
```

The full study is computationally substantial; the repository does not download data or model weights until this command is run.

## What is controlled

The primary experiment renders a lower- and higher-severity candidate for every raw LibriSpeech source utterance. Members of a pair share the same sampled corruption resource and crop; only the configured severity differs, and the low/high slot is hash-permuted. Every paired policy keeps exactly one candidate per source utterance. Thus source utterance, transcript, speaker, duration, training updates, character vocabulary, initialization, batch order, and incidental resource draw are invariant; only the audio rendition can change.

The secondary experiment emulates an ordinary curation pool: it takes one candidate per source utterance and selects against the same nominal audio-duration budget with random, global-score, source-quantile, and score-stratified policies. It reports the achieved duration as well as retention and lexical/source coverage alongside downstream WER.

## Result hand-off

After the run completes:

1. Read `paper/generated/RESULTS_TO_PASTE.md` and `paper/generated/placeholder_values.csv`.
2. Replace every `\placeholder{...}` tag in `paper/main.tex` with the corresponding observed value or short result phrase.

All numbers in the manuscript that arise from this experiment are placeholders until the run has been completed. The generated figures are included automatically by `paper/main.tex` once they exist.

## Resource and license notes

The pipeline downloads LibriSpeech (SLR12, CC BY 4.0), MUSAN (SLR17, CC BY 4.0), and RIRS_NOISES (SLR28, Apache 2.0). The two augmentation archives are checked against the official OpenSLR MD5 values pinned in the configuration, then their observed SHA-256 values and canonical archive-relative resource IDs are recorded in `results/run_metadata.json` and the render manifests. DNSMOS model files are downloaded by the pinned TorchMetrics implementation at runtime and their observed SHA-256 values are saved there too. TorchAudio SQUIM model files are obtained by the pinned TorchAudio release; the cache root, bundle name, and SHA-256 values of matching SQUIM artifacts are also recorded. Review the upstream terms before redistributing any resulting audio or model artifact.

## Directory map

- `configs/` — immutable experiment plans.
- `experiments/run.sh` — the only command required to reproduce all results.
- `src/qc_curation/` — dataset, rendering, scoring, selection, ASR, analysis, and reporting code.
- `paper/` — complete LaTeX manuscript, verified bibliography, and generated result targets.
- `LITERATURE_REVIEW.md` — the literature synthesis that motivated the hypothesis.
- `REPRODUCIBILITY.md` — environment, determinism, and audit specification.

No experiment code was executed while preparing this artifact.
