from __future__ import annotations

import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torchaudio
from tqdm import tqdm

from .audio import audio_info
from .corrupt import ResourceBank, make_resource_bank, render_audio
from .utils import (
    ensure_dirs,
    normalize_transcript,
    read_csv,
    select_exact_duration,
    sha256_file,
    stable_int,
    stable_order,
    update_run_metadata,
    write_csv,
    write_json,
)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        handle.extractall(destination, members=members)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"Unsafe path in zip archive: {member.filename}")
        handle.extractall(destination)


def _valid_archive(archive: Path) -> bool:
    try:
        if archive.name.endswith(".zip") or archive.name.endswith(".zip.part"):
            with zipfile.ZipFile(archive) as handle:
                return handle.testzip() is None
        with tarfile.open(archive, "r:gz") as handle:
            handle.getmembers()
        return True
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False


def _archive_hashes(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _assert_archive_integrity(archive: Path, expected_md5: str) -> tuple[str, str]:
    if not _valid_archive(archive):
        raise RuntimeError(f"Archive failed integrity parsing: {archive}")
    observed_md5, observed_sha256 = _archive_hashes(archive)
    if observed_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"Archive checksum mismatch for {archive}: expected {expected_md5}, observed {observed_md5}."
        )
    return observed_md5, observed_sha256


def _download_and_extract(url: str, archive: Path, destination: Path, expected_md5: str) -> tuple[str, str]:
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        temporary = archive.with_suffix(archive.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        archive_md5, archive_sha = _assert_archive_integrity(temporary, expected_md5)
        temporary.replace(archive)
    else:
        archive_md5, archive_sha = _assert_archive_integrity(archive, expected_md5)
    marker = destination / ".extracted"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == archive_sha:
        return archive_md5, archive_sha
    if marker.exists():
        raise RuntimeError(
            f"Archive checksum changed after extraction for {archive}; refusing to mix contents. "
            "Remove the matching extracted cache and re-run to rebuild it."
        )
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        _safe_extract_zip(archive, destination)
    else:
        _safe_extract_tar(archive, destination)
    marker.write_text(f"{archive_sha}\n", encoding="utf-8")
    return archive_md5, archive_sha


def download_resources(cfg: dict[str, Any], required_splits: set[str]) -> ResourceBank:
    """Fetch raw corpora with library/archive downloaders; no corpus content is committed."""
    ensure_dirs(cfg)
    librispeech_root = Path(cfg["resources"]["librispeech_root"])
    for split in sorted(required_splits):
        # Instantiation triggers the official OpenSLR retrieval and is idempotent.
        torchaudio.datasets.LIBRISPEECH(root=str(librispeech_root), url=split, download=True)

    raw_root = Path(cfg["paths"]["raw"])
    musan_destination = raw_root / "musan_archive"
    rirs_destination = raw_root / "rirs_archive"
    musan_archive = raw_root / "musan.tar.gz"
    rirs_archive = raw_root / "rirs_noises.zip"
    musan_md5, musan_sha256 = _download_and_extract(
        cfg["resources"]["musan_url"], musan_archive, musan_destination, str(cfg["resources"]["musan_md5"])
    )
    rirs_md5, rirs_sha256 = _download_and_extract(
        cfg["resources"]["rirs_url"], rirs_archive, rirs_destination, str(cfg["resources"]["rirs_md5"])
    )
    update_run_metadata(
        cfg,
        {
            "resource_archives": {
                "musan_url": str(cfg["resources"]["musan_url"]),
                "musan_md5": musan_md5,
                "musan_sha256": musan_sha256,
                "rirs_url": str(cfg["resources"]["rirs_url"]),
                "rirs_md5": rirs_md5,
                "rirs_sha256": rirs_sha256,
            }
        },
    )

    # Archives have a nested top-level directory; recursive discovery avoids relying on a fragile version-specific name.
    return make_resource_bank(
        musan_destination,
        rirs_destination,
        int(cfg["run"]["seed"]),
        list(cfg["resources"]["resource_split"]),
    )


def raw_split_root(cfg: dict[str, Any], split: str) -> Path:
    root = Path(cfg["resources"]["librispeech_root"]) / "LibriSpeech" / split
    if not root.exists():
        raise FileNotFoundError(f"LibriSpeech split is missing: {root}")
    return root


def parse_librispeech_split(cfg: dict[str, Any], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = raw_split_root(cfg, split)
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            utterance_id, transcript = line.split(" ", maxsplit=1)
            speaker_id, chapter_id, _ = utterance_id.split("-", maxsplit=2)
            audio_path = transcript_path.parent / f"{utterance_id}.flac"
            sample_rate, frames = audio_info(audio_path)
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "source_id": speaker_id,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "transcript": normalize_transcript(transcript),
                    "raw_audio_path": str(audio_path.resolve()),
                    "duration_seconds": frames / sample_rate,
                    "split": split,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No transcript rows were parsed from {root}")
    return frame.sort_values("utterance_id").reset_index(drop=True)


def common_validity_filter(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply the same non-quality filter before every policy and evaluation arm."""
    maximum = float(cfg["asr"]["train"]["max_audio_seconds"])
    filtered = frame[
        (frame["duration_seconds"] >= 1.5)
        & (frame["duration_seconds"] <= maximum)
        & (frame["transcript"].str.len() > 0)
    ].copy()
    if filtered.empty:
        raise RuntimeError("The common duration/transcript validity filter removed every utterance.")
    return filtered.reset_index(drop=True)


def deterministic_pool(frame: pd.DataFrame, hours: float, seed: int, label: str) -> pd.DataFrame:
    ordered_ids = stable_order(frame["utterance_id"].tolist(), seed, "pool", label)
    by_id = frame.set_index("utterance_id", drop=False)
    ordered = by_id.loc[ordered_ids].reset_index(drop=True)
    selected = select_exact_duration(ordered, ordered.index, hours * 3600.0)
    return selected.drop(columns=["selected_duration_seconds"], errors="ignore").reset_index(drop=True)


def _severity_parameters(cfg: dict[str, Any], severity: str) -> dict[str, float]:
    values = cfg["primary"][f"{severity}_severity"]
    return {name: float(value) for name, value in values.items()}


def make_primary_candidates(cfg: dict[str, Any], pool: pd.DataFrame, bank: ResourceBank) -> pd.DataFrame:
    if int(cfg["primary"]["candidate_count"]) != 2:
        raise ValueError("The counterfactual design requires exactly two candidate renditions per source utterance.")
    output_root = Path(cfg["paths"]["derived"]) / "primary_candidates"
    sample_rate = int(cfg["resources"]["sample_rate"])
    overwrite = bool(cfg["run"]["overwrite_derived_audio"])
    families = list(cfg["primary"]["families"])
    rows: list[dict[str, Any]] = []

    for source in tqdm(pool.to_dict("records"), desc="Rendering paired training candidates"):
        utterance_id = str(source["utterance_id"])
        source_sha256 = sha256_file(source["raw_audio_path"])
        family = families[stable_int(cfg["run"]["seed"], "family", utterance_id, modulo=len(families))]
        labels = ["low", "high"]
        if stable_int(cfg["run"]["seed"], "slot-order", utterance_id, modulo=2):
            labels.reverse()
        for slot, severity in enumerate(labels):
            params = _severity_parameters(cfg, severity)
            candidate_path = output_root / family / f"{utterance_id}__candidate-{slot}.wav"
            render_metadata = render_audio(
                source["raw_audio_path"],
                candidate_path,
                family,
                params,
                bank,
                "train",
                sample_rate,
                # Candidate slot only hides the low/high label. Both members of
                # a pair share the same noise/RIR/crop draw so severity is the
                # sole intended audio intervention.
                (cfg["run"]["seed"], "primary", utterance_id),
                overwrite,
                source_sha256,
            )
            row = dict(source)
            row.update(
                {
                    "study": "primary",
                    "pair_id": utterance_id,
                    "candidate_slot": slot,
                    "variant_id": f"{utterance_id}__candidate-{slot}",
                    "variant_audio_path": str(candidate_path.resolve()),
                    "family": family,
                    "severity_label": severity,
                    "severity_rank": 0 if severity == "low" else 1,
                    "raw_audio_sha256": source_sha256,
                    **render_metadata,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["utterance_id", "candidate_slot"]).reset_index(drop=True)


def make_natural_pool(cfg: dict[str, Any], pool: pd.DataFrame) -> pd.DataFrame:
    rows = pool.copy()
    rows["study"] = "natural"
    rows["pair_id"] = ""
    rows["candidate_slot"] = -1
    rows["variant_id"] = rows["utterance_id"]
    rows["variant_audio_path"] = rows["raw_audio_path"]
    rows["family"] = "natural"
    rows["severity_label"] = "unknown"
    rows["severity_rank"] = -1
    rows["noise_id"] = ""
    rows["rir_id"] = ""
    rows["noise_snr_db"] = float("nan")
    rows["reverb_wet"] = float("nan")
    rows["bandlimit_hz"] = float("nan")
    rows["raw_audio_sha256"] = rows["raw_audio_path"].map(sha256_file)
    rows["variant_audio_sha256"] = rows["raw_audio_sha256"]
    return rows


def make_evaluation_manifests(cfg: dict[str, Any], bank: ResourceBank) -> dict[str, Path]:
    manifests_root = Path(cfg["paths"]["manifests"]) / "evaluation"
    derived_root = Path(cfg["paths"]["derived"]) / "evaluation"
    sample_rate = int(cfg["resources"]["sample_rate"])
    overwrite = bool(cfg["run"]["overwrite_derived_audio"])
    outputs: dict[str, Path] = {}

    for split, label in [
        (cfg["evaluation"]["test_clean_split"], "clean"),
        (cfg["evaluation"]["test_other_split"], "other"),
    ]:
        frame = common_validity_filter(parse_librispeech_split(cfg, split), cfg)
        frame["raw_audio_sha256"] = frame["raw_audio_path"].map(sha256_file)
        frame["variant_audio_sha256"] = frame["raw_audio_sha256"]
        frame["evaluation_name"] = label
        frame["variant_audio_path"] = frame["raw_audio_path"]
        path = manifests_root / f"{label}.csv"
        write_csv(frame, path)
        outputs[label] = path

    clean_frame = common_validity_filter(parse_librispeech_split(cfg, cfg["evaluation"]["test_clean_split"]), cfg)
    clean_frame["raw_audio_sha256"] = clean_frame["raw_audio_path"].map(sha256_file)
    clean_frame["variant_audio_sha256"] = clean_frame["raw_audio_sha256"]
    for condition in cfg["evaluation"]["synthetic_conditions"]:
        name = str(condition["name"])
        rows: list[dict[str, Any]] = []
        for source in tqdm(clean_frame.to_dict("records"), desc=f"Rendering evaluation {name}"):
            params = {
                key: float(value)
                for key, value in condition.items()
                if key not in {"name", "family"}
            }
            path = derived_root / name / f"{source['utterance_id']}.wav"
            render_metadata = render_audio(
                source["raw_audio_path"],
                path,
                str(condition["family"]),
                params,
                bank,
                "test",
                sample_rate,
                (cfg["run"]["seed"], "evaluation", name, source["utterance_id"]),
                overwrite,
                str(source["raw_audio_sha256"]),
            )
            row = dict(source)
            row.update({"evaluation_name": name, "variant_audio_path": str(path.resolve()), **render_metadata})
            rows.append(row)
        output = manifests_root / f"{name}.csv"
        write_csv(pd.DataFrame(rows), output)
        outputs[name] = output
    return outputs


def prepare_all(cfg: dict[str, Any]) -> dict[str, str]:
    required = {
        cfg["primary"]["train_split"],
        cfg["evaluation"]["test_clean_split"],
        cfg["evaluation"]["test_other_split"],
    }
    if cfg["natural"]["enabled"]:
        required.add(cfg["natural"]["train_split"])
    bank = download_resources(cfg, required)
    manifests_root = Path(cfg["paths"]["manifests"])

    primary_raw = common_validity_filter(parse_librispeech_split(cfg, cfg["primary"]["train_split"]), cfg)
    primary_pool = deterministic_pool(primary_raw, float(cfg["primary"]["pool_hours"]), int(cfg["run"]["seed"]), "primary")
    write_csv(primary_pool, manifests_root / "primary_sources.csv")
    primary_candidates = make_primary_candidates(cfg, primary_pool, bank)
    primary_path = manifests_root / "primary_candidates.csv"
    write_csv(primary_candidates, primary_path)

    outputs: dict[str, str] = {"primary_candidates": str(primary_path)}
    if cfg["natural"]["enabled"]:
        natural_raw = common_validity_filter(parse_librispeech_split(cfg, cfg["natural"]["train_split"]), cfg)
        natural_pool = deterministic_pool(natural_raw, float(cfg["natural"]["pool_hours"]), int(cfg["run"]["seed"]), "natural")
        natural_path = manifests_root / "natural_pool.csv"
        write_csv(make_natural_pool(cfg, natural_pool), natural_path)
        outputs["natural_pool"] = str(natural_path)

    outputs.update({f"evaluation_{name}": str(path) for name, path in make_evaluation_manifests(cfg, bank).items()})
    write_json(outputs, manifests_root / "manifest_index.json")
    return outputs
