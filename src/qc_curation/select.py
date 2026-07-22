from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import (
    dataframe_digest,
    lexical_coverage,
    read_csv,
    select_exact_duration,
    source_seconds_to_budget,
    stable_digest,
    stable_int,
    stable_order,
    write_csv,
    write_json,
)


PAIR_METRICS: dict[str, tuple[str, bool]] = {
    "pair_dnsmos_ovrl": ("dnsmos_ovrl", True),
    "pair_dnsmos_sig": ("dnsmos_sig", True),
    "pair_dnsmos_bak": ("dnsmos_bak", True),
    "pair_squim_pesq": ("squim_pesq", True),
    "pair_low_dnsmos_ovrl": ("dnsmos_ovrl", False),
    # A hidden generation parameter used only as a diagnostic ceiling, never a deployable selector.
    "pair_oracle_severity": ("severity_rank", False),
}


def _primary_source_subset(candidates: pd.DataFrame, hours: float, cfg: dict[str, Any]) -> pd.DataFrame:
    sources = candidates.drop_duplicates("utterance_id").copy()
    ordered_ids = stable_order(sources["utterance_id"].tolist(), cfg["run"]["seed"], "primary-budget", hours)
    sources = sources.set_index("utterance_id", drop=False).loc[ordered_ids].reset_index(drop=True)
    selected_sources = select_exact_duration(sources, sources.index, source_seconds_to_budget(hours))
    selected_ids = set(selected_sources["utterance_id"])
    result = candidates[candidates["utterance_id"].isin(selected_ids)].copy()
    expected = result.groupby("pair_id").size()
    if not (expected == 2).all():
        raise RuntimeError("A primary source subset lost a member of a candidate pair.")
    return result


def _select_one_per_pair(frame: pd.DataFrame, policy: str, cfg: dict[str, Any]) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    if policy == "pair_random":
        score_name = "random_choice"
        ascending = False
    elif policy in PAIR_METRICS:
        score_name, ascending = PAIR_METRICS[policy]
        ascending = not ascending
    else:
        raise ValueError(f"Unknown paired policy: {policy}")

    for pair_id, group in frame.groupby("pair_id", sort=True):
        if len(group) != 2:
            raise RuntimeError(f"Pair {pair_id} has {len(group)} candidates, not two.")
        if policy == "pair_random":
            index = stable_int(cfg["run"]["seed"], "pair-random", pair_id, modulo=2)
            chosen = group.sort_values("candidate_slot").iloc[index]
            selector_score = float(index)
        else:
            ordered = group.sort_values(
                [score_name, "candidate_slot"], ascending=[ascending, True], kind="mergesort"
            )
            chosen = ordered.iloc[0]
            selector_score = float(chosen[score_name])
        chosen = chosen.copy()
        chosen["selector_score"] = selector_score
        chosen["selector_metric"] = score_name
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["selection_policy"] = policy
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _global_random(frame: pd.DataFrame, budget_seconds: float, cfg: dict[str, Any]) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_order"] = ordered["utterance_id"].map(lambda ident: stable_digest(cfg["run"]["seed"], "global-random", ident))
    ordered = ordered.sort_values("_order")
    chosen = select_exact_duration(ordered, ordered.index, budget_seconds)
    chosen["selector_score"] = np.nan
    chosen["selector_metric"] = "random"
    return chosen.drop(columns=["_order", "selected_duration_seconds"], errors="ignore")


def _global_ranked(frame: pd.DataFrame, metric: str, budget_seconds: float, descending: bool) -> pd.DataFrame:
    if frame[metric].isna().any():
        raise RuntimeError(f"Cannot rank on {metric}: at least one score is missing.")
    ordered = frame.sort_values([metric, "utterance_id"], ascending=[not descending, True], kind="mergesort")
    chosen = select_exact_duration(ordered, ordered.index, budget_seconds)
    chosen["selector_score"] = chosen[metric]
    chosen["selector_metric"] = metric
    return chosen.drop(columns=["selected_duration_seconds"], errors="ignore")


def _source_quantile(frame: pd.DataFrame, metric: str, budget_seconds: float) -> pd.DataFrame:
    """Rank within source, then allocate the budget by a duration-proportional round robin.

    The queue rule avoids an accidental global comparison of raw MOS scales while
    retaining an overall fixed-duration constraint up to the final indivisible
    utterance. At each step it serves the source with the smallest retained
    duration fraction and takes its highest-scoring remaining segment that fits
    the unspent budget.
    """
    queues: dict[str, list[int]] = {}
    source_total: dict[str, float] = {}
    for source_id, group in frame.groupby("source_id", sort=True):
        ordered = group.sort_values([metric, "utterance_id"], ascending=[False, True], kind="mergesort")
        queues[str(source_id)] = ordered.index.tolist()
        source_total[str(source_id)] = float(ordered["duration_seconds"].sum())
    selected_indices: list[int] = []
    selected_by_source = {source: 0.0 for source in queues}
    total = 0.0
    while True:
        remaining = budget_seconds - total
        feasible = [
            source
            for source, queue in queues.items()
            if any(float(frame.loc[index, "duration_seconds"]) <= remaining for index in queue)
        ]
        if not feasible:
            break
        source = min(feasible, key=lambda item: (selected_by_source[item] / source_total[item], item))
        # Preserve the within-source score order whenever possible, but skip a
        # currently unfillable long segment rather than leaving usable duration
        # idle while a shorter lower-ranked segment fits the fixed budget.
        position = next(
            index
            for index, candidate in enumerate(queues[source])
            if float(frame.loc[candidate, "duration_seconds"]) <= remaining
        )
        index = queues[source].pop(position)
        selected_indices.append(index)
        selected_by_source[source] += float(frame.loc[index, "duration_seconds"])
        total += float(frame.loc[index, "duration_seconds"])
    if not selected_indices:
        raise RuntimeError("Source-quantile selection could not fit any utterance into the budget.")
    result = frame.loc[selected_indices].copy()
    result["selector_score"] = result[metric]
    result["selector_metric"] = f"within_source_{metric}"
    return result


def _score_stratified(frame: pd.DataFrame, metric: str, budget_seconds: float, cfg: dict[str, Any]) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_bin"] = pd.qcut(ranked[metric].rank(method="first"), q=5, labels=False)
    chosen_frames: list[pd.DataFrame] = []
    for bin_index, group in ranked.groupby("_bin", sort=True):
        target = budget_seconds / 5.0
        group = group.copy()
        group["_order"] = group["utterance_id"].map(
            lambda ident: stable_digest(cfg["run"]["seed"], "stratified", metric, int(bin_index), ident)
        )
        ordered = group.sort_values("_order")
        chosen_frames.append(select_exact_duration(ordered, ordered.index, target))
    result = pd.concat(chosen_frames, ignore_index=True)
    result["selector_score"] = result[metric]
    result["selector_metric"] = f"stratified_{metric}"
    return result.drop(columns=["_bin", "_order", "selected_duration_seconds"], errors="ignore")


def _global_select(frame: pd.DataFrame, policy: str, budget_seconds: float, cfg: dict[str, Any]) -> pd.DataFrame:
    if policy == "global_random":
        result = _global_random(frame, budget_seconds, cfg)
    elif policy == "global_dnsmos_ovrl":
        result = _global_ranked(frame, "dnsmos_ovrl", budget_seconds, True)
    elif policy == "global_squim_pesq":
        result = _global_ranked(frame, "squim_pesq", budget_seconds, True)
    elif policy == "global_low_dnsmos_ovrl":
        result = _global_ranked(frame, "dnsmos_ovrl", budget_seconds, False)
    elif policy == "global_source_quantile_dnsmos":
        result = _source_quantile(frame, "dnsmos_ovrl", budget_seconds)
    elif policy == "global_score_stratified_dnsmos":
        result = _score_stratified(frame, "dnsmos_ovrl", budget_seconds, cfg)
    else:
        raise ValueError(f"Unknown global policy: {policy}")
    result["selection_policy"] = policy
    result = result.sort_values("utterance_id", kind="mergesort").reset_index(drop=True)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result


def _entropy(series: pd.Series) -> float:
    probabilities = series.value_counts(normalize=True)
    return float(-(probabilities * np.log(probabilities)).sum())


def _composition(study: str, policy: str, selected: pd.DataFrame, universe: pd.DataFrame) -> dict[str, Any]:
    source_counts = selected["source_id"].value_counts()
    effective_sources = float(np.exp(_entropy(selected["source_id"])))
    composition = {
        "study": study,
        "policy": policy,
        "utterances": len(selected),
        "selected_hours": float(selected["duration_seconds"].sum() / 3600.0),
        "unique_sources": int(selected["source_id"].nunique()),
        "effective_source_count": effective_sources,
        "source_entropy": _entropy(selected["source_id"]),
        "family_entropy": _entropy(selected["family"]),
        "mean_dnsmos_ovrl": float(selected["dnsmos_ovrl"].mean()),
        "mean_squim_pesq": float(selected["squim_pesq"].mean()),
        "selection_manifest_sha256": dataframe_digest(selected),
    }
    composition.update(lexical_coverage(selected["transcript"], universe["transcript"]))
    return composition


def materialize_selection_manifests(cfg: dict[str, Any]) -> dict[str, str]:
    manifests = Path(cfg["paths"]["manifests"])
    output_root = manifests / "selected"
    primary = read_csv(manifests / "scored" / "primary_candidates_scored.csv")
    index: dict[str, str] = {}
    composition_rows: list[dict[str, Any]] = []

    for hours in cfg["primary"]["budgets_hours"]:
        source_subset = _primary_source_subset(primary, float(hours), cfg)
        for policy in cfg["primary"]["paired_policies"]:
            selected = _select_one_per_pair(source_subset, str(policy), cfg)
            selected["selection_budget_hours"] = float(hours)
            destination = output_root / "primary" / f"{float(hours):g}h" / f"{policy}.csv"
            write_csv(selected, destination)
            key = f"primary/{float(hours):g}h/{policy}"
            index[key] = str(destination)
            composition_rows.append(_composition("primary", f"{hours}h/{policy}", selected, source_subset))

    if cfg["natural"]["enabled"]:
        natural = read_csv(manifests / "scored" / "natural_pool_scored.csv")
        budget_hours = float(cfg["natural"]["budget_hours"])
        for policy in cfg["natural"]["policies"]:
            selected = _global_select(natural, str(policy), source_seconds_to_budget(budget_hours), cfg)
            selected["selection_budget_hours"] = budget_hours
            destination = output_root / "natural" / f"{budget_hours:g}h" / f"{policy}.csv"
            write_csv(selected, destination)
            key = f"natural/{budget_hours:g}h/{policy}"
            index[key] = str(destination)
            composition_rows.append(_composition("natural", f"{budget_hours}h/{policy}", selected, natural))

    write_csv(pd.DataFrame(composition_rows), Path(cfg["paths"]["results"]) / "selection_composition.csv")
    write_json(index, manifests / "selection_index.json")
    return index
