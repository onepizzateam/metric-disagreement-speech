from __future__ import annotations

import argparse
from typing import Any, Callable

from .analyse import analyse_all
from .asr import evaluate_all, train_all
from .metrics import score_all
from .prepare import prepare_all
from .select import materialize_selection_manifests
from .utils import choose_device, ensure_dirs, load_config, run_metadata, update_run_metadata, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the quality-conditioned ASR curation study.")
    parser.add_argument(
        "command",
        choices=["prepare", "score", "select", "train", "evaluate", "analyse", "all"],
        help="Pipeline stage to execute. 'all' is the manuscript reproduction entry point.",
    )
    parser.add_argument("--config", required=True, help="Pre-specified YAML configuration.")
    parser.add_argument("--device", default=None, help="Override run.device (for example cuda, cuda:0, or cpu).")
    return parser.parse_args()


def _run_stage(label: str, stage: Callable[[], Any]) -> Any:
    print(f"\n{'=' * 16} {label.upper()} {'=' * 16}")
    result = stage()
    print(f"Completed {label}.")
    return result


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["run"]["device"] = args.device
    ensure_dirs(cfg)
    device = choose_device(str(cfg["run"]["device"]))
    update_run_metadata(cfg, run_metadata(cfg, device))

    stages: dict[str, Callable[[], Any]] = {
        "prepare": lambda: prepare_all(cfg),
        "score": lambda: score_all(cfg, device),
        "select": lambda: materialize_selection_manifests(cfg),
        "train": lambda: train_all(cfg, device),
        "evaluate": lambda: evaluate_all(cfg, device),
        "analyse": lambda: analyse_all(cfg),
    }
    if args.command == "all":
        execution = {name: _run_stage(name, stage) for name, stage in stages.items()}
        write_json({name: str(result) for name, result in execution.items()}, f"{cfg['paths']['results']}/pipeline_index.json")
        return
    _run_stage(args.command, stages[args.command])


if __name__ == "__main__":
    main()
