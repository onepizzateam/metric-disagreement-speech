from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .utils import read_csv, stable_int, write_csv, write_json


METRICS = ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "squim_pesq"]


def _percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _pair_metric_behaviour(primary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    draws = int(cfg["statistics"]["bootstrap_draws"])
    confidence = float(cfg["statistics"]["confidence_level"])
    rng = np.random.default_rng(stable_int(cfg["run"]["seed"], "metric-bootstrap"))
    for metric in METRICS:
        all_pairs: list[dict[str, Any]] = []
        for family, group in primary.groupby("family", sort=True):
            pairs = []
            for pair_id, candidate_group in group.groupby("pair_id", sort=True):
                chosen = candidate_group.sort_values([metric, "candidate_slot"], ascending=[False, True]).iloc[0]
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "correct": int(chosen["severity_rank"]) == 0,
                        "score_gap": float(candidate_group.loc[candidate_group["severity_rank"] == 0, metric].iloc[0])
                        - float(candidate_group.loc[candidate_group["severity_rank"] == 1, metric].iloc[0]),
                    }
                )
            pair_frame = pd.DataFrame(pairs)
            if pair_frame.empty:
                continue
            all_pairs.extend(pairs)
            bootstrap = []
            values = pair_frame["correct"].to_numpy(dtype=float)
            for _ in range(draws):
                bootstrap.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            lower, upper = _percentile_interval(bootstrap, confidence)
            rho = spearmanr(group[metric], -group["severity_rank"]).statistic
            rows.append(
                {
                    "metric": metric,
                    "family": family,
                    "pairs": len(pair_frame),
                    "choose_lower_severity_rate": float(values.mean()),
                    "ci_low": lower,
                    "ci_high": upper,
                    "mean_low_minus_high_score": float(pair_frame["score_gap"].mean()),
                    "spearman_score_vs_negative_severity": float(rho),
                }
            )
        # The manuscript's overall score is pooled by source pair, not an
        # unweighted mean of corruption-family rates. This remains meaningful
        # if a deterministic family allocation happens not to be exactly equal.
        overall_frame = pd.DataFrame(all_pairs)
        if not overall_frame.empty:
            overall_values = overall_frame["correct"].to_numpy(dtype=float)
            overall_bootstrap = [
                float(rng.choice(overall_values, size=len(overall_values), replace=True).mean())
                for _ in range(draws)
            ]
            lower, upper = _percentile_interval(overall_bootstrap, confidence)
            rho = spearmanr(primary[metric], -primary["severity_rank"]).statistic
            rows.append(
                {
                    "metric": metric,
                    "family": "overall",
                    "pairs": len(overall_frame),
                    "choose_lower_severity_rate": float(overall_values.mean()),
                    "ci_low": lower,
                    "ci_high": upper,
                    "mean_low_minus_high_score": float(overall_frame["score_gap"].mean()),
                    "spearman_score_vs_negative_severity": float(rho),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_wer(wer: pd.DataFrame) -> pd.DataFrame:
    return (
        wer.groupby(["study", "budget_hours", "policy", "evaluation"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_wer=("wer", "mean"),
            sd_wer=("wer", "std"),
            min_wer=("wer", "min"),
            max_wer=("wer", "max"),
        )
        .fillna({"sd_wer": 0.0})
    )


def _sign_flip_p_value(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    if len(differences) == 0:
        return float("nan")
    values = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def _paired_prediction_difference(path_a: str, path_b: str, evaluation: str) -> pd.DataFrame:
    a = read_csv(Path(path_a).parent / f"{evaluation}.csv")
    b = read_csv(Path(path_b).parent / f"{evaluation}.csv")
    merged = a.merge(
        b,
        on=["utterance_id", "source_id"],
        suffixes=("_metric", "_random"),
        validate="one_to_one",
    )
    return merged


def _hierarchical_bootstrap(
    comparisons: list[pd.DataFrame], draws: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Resample paired model seeds and then evaluation speakers; preserve pairing at both levels."""
    if not comparisons:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled_seed_indices = rng.integers(0, len(comparisons), size=len(comparisons))
        per_seed: list[float] = []
        for index in sampled_seed_indices:
            frame = comparisons[int(index)]
            speakers = frame["source_id"].drop_duplicates().to_numpy()
            selected_speakers = rng.choice(speakers, size=len(speakers), replace=True)
            pieces = [frame[frame["source_id"] == speaker] for speaker in selected_speakers]
            resampled = pd.concat(pieces, ignore_index=True)
            metric_wer = resampled["errors_metric"].sum() / max(1, resampled["reference_words_metric"].sum())
            random_wer = resampled["errors_random"].sum() / max(1, resampled["reference_words_random"].sum())
            per_seed.append(float(metric_wer - random_wer))
        estimates.append(float(np.mean(per_seed)))
    return _percentile_interval(estimates, confidence)


def _policy_contrasts(wer: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute every available policy-versus-random matched-seed contrast.

    The designated primary row is unadjusted. All other contrasts are
    exploratory and Holm-adjusted within a study/budget/evaluation family.
    """
    primary_budget = float(cfg["primary"]["confirmatory_budget_hours"])
    primary_target = str(cfg["primary"]["confirmatory_policy"])
    rows: list[dict[str, Any]] = []
    for study, baseline in [("primary", "pair_random"), ("natural", "global_random")]:
        study_wer = wer[wer["study"] == study]
        if study_wer.empty:
            continue
        for budget, evaluation in study_wer[["budget_hours", "evaluation"]].drop_duplicates().itertuples(index=False):
            evaluation_rows = study_wer[
                (study_wer["budget_hours"] == budget) & (study_wer["evaluation"] == evaluation)
            ]
            baseline_rows = evaluation_rows[evaluation_rows["policy"] == baseline]
            if baseline_rows.empty:
                continue
            for policy in sorted(set(evaluation_rows["policy"]) - {baseline}):
                metric_rows = evaluation_rows[evaluation_rows["policy"] == policy]
                merged = metric_rows.merge(
                    baseline_rows, on="seed", suffixes=("_metric", "_random"), validate="one_to_one"
                )
                if merged.empty:
                    continue
                differences = (merged["wer_metric"] - merged["wer_random"]).to_numpy(dtype=float)
                comparisons = [
                    _paired_prediction_difference(row.prediction_path_metric, row.prediction_path_random, evaluation)
                    for row in merged.itertuples(index=False)
                ]
                ci_low, ci_high = _hierarchical_bootstrap(
                    comparisons,
                    int(cfg["statistics"]["bootstrap_draws"]),
                    float(cfg["statistics"]["confidence_level"]),
                    stable_int(cfg["run"]["seed"], "hierarchical", study, policy, baseline, budget, evaluation),
                )
                primary_analysis = (
                    study == "primary"
                    and policy == primary_target
                    and budget == primary_budget
                    and evaluation == str(cfg["report"]["primary_evaluation"])
                )
                rows.append(
                    {
                        "study": study,
                        "budget_hours": budget,
                        "evaluation": evaluation,
                        "metric_policy": policy,
                        "baseline_policy": baseline,
                        "seeds": len(merged),
                        "mean_difference_wer": float(differences.mean()),
                        "mean_difference_percentage_points": float(differences.mean() * 100.0),
                        "seed_sd_difference": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
                        "sign_flip_p": _sign_flip_p_value(differences),
                        "hierarchical_ci_low": ci_low,
                        "hierarchical_ci_high": ci_high,
                        "primary_analysis": primary_analysis,
                        "correction_family": f"{study}/{float(budget):g}h/{evaluation}",
                    }
                )
    contrast = pd.DataFrame(rows)
    if contrast.empty:
        return contrast
    contrast["holm_p_exploratory"] = np.nan
    exploratory = contrast[~contrast["primary_analysis"]]
    for _, family in exploratory.groupby("correction_family", sort=True):
        indices = family.index.to_numpy()
        pvals = family["sign_flip_p"].to_numpy(dtype=float)
        order = np.argsort(pvals)
        adjusted = np.empty_like(pvals)
        previous = 0.0
        for rank, position in enumerate(order):
            value = min(1.0, (len(pvals) - rank) * pvals[position])
            adjusted[position] = max(previous, value)
            previous = adjusted[position]
        contrast.loc[indices, "holm_p_exploratory"] = adjusted
    return contrast


def _plot_metric_behaviour(metric: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(8.3, 4.5))
    overall = metric[metric["family"] == "overall"].copy()
    overall = overall.rename(columns={"choose_lower_severity_rate": "rate", "ci_low": "low", "ci_high": "high"})
    labels = overall["metric"].str.replace("_", " ")
    errors = np.vstack([overall["rate"] - overall["low"], overall["high"] - overall["rate"]])
    axis.bar(labels, overall["rate"], yerr=errors, capsize=4, color="#4C78A8")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability of choosing lower-severity candidate")
    axis.set_xlabel("Frozen selector")
    axis.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_paired_wer(summary: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int) -> None:
    budget = float(cfg["primary"]["confirmatory_budget_hours"])
    relevant = summary[(summary["study"] == "primary") & (summary["budget_hours"] == budget)].copy()
    policies = ["pair_random", str(cfg["primary"]["confirmatory_policy"])]
    relevant = relevant[relevant["policy"].isin(policies)]
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for policy, group in relevant.groupby("policy"):
        group = group.sort_values("evaluation")
        axis.errorbar(
            group["evaluation"], group["mean_wer"] * 100,
            yerr=group["sd_wer"] * 100,
            marker="o", capsize=3, linewidth=1.8, label=policy.replace("pair_", "").replace("_", " "),
        )
    axis.set_ylabel("Word error rate (%)")
    axis.set_xlabel("Evaluation condition")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_global_pareto(summary: pd.DataFrame, composition: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int) -> None:
    natural = summary[(summary["study"] == "natural") & (summary["evaluation"] == "clean")].copy()
    if natural.empty:
        return
    natural["composition_key"] = natural.apply(lambda row: f"{row.budget_hours}h/{row.policy}", axis=1)
    comp = composition[composition["study"] == "natural"].copy()
    merged = natural.merge(comp, left_on="composition_key", right_on="policy", how="left", suffixes=("", "_comp"))
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    for row in merged.itertuples(index=False):
        axis.scatter(row.mean_dnsmos_ovrl, row.mean_wer * 100, s=max(30, row.effective_source_count * 4), alpha=0.85)
        axis.annotate(str(row.policy).replace("global_", ""), (row.mean_dnsmos_ovrl, row.mean_wer * 100), fontsize=8)
    axis.set_xlabel("Mean selected DNSMOS OVRL")
    axis.set_ylabel("Clean-evaluation WER (%)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _find_summary_row(summary: pd.DataFrame, study: str, budget: float, policy: str, evaluation: str) -> dict[str, float] | None:
    rows = summary[
        (summary["study"] == study)
        & np.isclose(summary["budget_hours"], budget)
        & (summary["policy"] == policy)
        & (summary["evaluation"] == evaluation)
    ]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _placeholder_values(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    metric: pd.DataFrame,
    composition: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    values: list[dict[str, str]] = []
    budget = float(cfg["primary"]["confirmatory_budget_hours"])
    evaluation = str(cfg["report"]["primary_evaluation"])
    for policy, tag_prefix in [("pair_random", "primary_random"), (str(cfg["primary"]["confirmatory_policy"]), "primary_dnsmos")]:
        row = _find_summary_row(summary, "primary", budget, policy, evaluation)
        if row:
            values.extend(
                [
                    {"tag": f"{tag_prefix}_clean_wer_percent", "value": f"{100 * row['mean_wer']:.2f}", "source": "wer_summary.csv"},
                    {"tag": f"{tag_prefix}_clean_wer_sd_percent", "value": f"{100 * row['sd_wer']:.2f}", "source": "wer_summary.csv"},
                ]
            )
    primary = contrasts[(contrasts["primary_analysis"])].copy() if not contrasts.empty else contrasts
    if not primary.empty:
        row = primary.iloc[0]
        values.extend(
            [
                {"tag": "primary_dnsmos_minus_random_percentage_points", "value": f"{row['mean_difference_percentage_points']:.2f}", "source": "policy_contrasts.csv"},
                {"tag": "primary_dnsmos_minus_random_ci_percentage_points", "value": f"[{100 * row['hierarchical_ci_low']:.2f}, {100 * row['hierarchical_ci_high']:.2f}]", "source": "policy_contrasts.csv"},
                {"tag": "primary_sign_flip_p", "value": f"{row['sign_flip_p']:.4g}", "source": "policy_contrasts.csv"},
            ]
        )
    overall_metric = metric[metric["family"] == "overall"]
    for item in overall_metric.itertuples(index=False):
        values.append(
            {
                "tag": f"{item.metric}_lower_severity_selection_percent",
                "value": f"{100 * item.choose_lower_severity_rate:.1f}",
                "source": "metric_behavior.csv",
            }
        )
    if cfg["natural"]["enabled"]:
        for policy in cfg["natural"]["policies"]:
            row = _find_summary_row(
                summary, "natural", float(cfg["natural"]["budget_hours"]), str(policy), "clean"
            )
            if row:
                values.append(
                    {
                        "tag": f"natural_{policy}_clean_wer_percent",
                        "value": f"{100 * row['mean_wer']:.2f}",
                        "source": "wer_summary.csv",
                    }
                )
        table_tags = {
            "global_random": {
                "selected_hours": "random selected hours",
                "unique_sources": "random unique sources",
                "effective_source_count": "random effective sources",
                "lexical_type_coverage": "random lexical coverage",
                "mean_dnsmos_ovrl": "random mean OVRL",
            },
            "global_dnsmos_ovrl": {
                "selected_hours": "top OVRL selected hours",
                "unique_sources": "top OVRL unique sources",
                "effective_source_count": "top OVRL effective sources",
                "lexical_type_coverage": "top OVRL lexical coverage",
                "mean_dnsmos_ovrl": "top OVRL mean",
            },
            "global_squim_pesq": {
                "selected_hours": "SQUIM selected hours",
                "unique_sources": "SQUIM unique sources",
                "effective_source_count": "SQUIM effective sources",
                "lexical_type_coverage": "SQUIM lexical coverage",
                "mean_dnsmos_ovrl": "SQUIM mean OVRL",
            },
            "global_low_dnsmos_ovrl": {
                "selected_hours": "bottom OVRL selected hours",
                "unique_sources": "bottom OVRL unique sources",
                "effective_source_count": "bottom OVRL effective sources",
                "lexical_type_coverage": "bottom OVRL lexical coverage",
                "mean_dnsmos_ovrl": "bottom OVRL mean",
            },
            "global_source_quantile_dnsmos": {
                "selected_hours": "quota selected hours",
                "unique_sources": "quota unique sources",
                "effective_source_count": "quota effective sources",
                "lexical_type_coverage": "quota lexical coverage",
                "mean_dnsmos_ovrl": "quota mean OVRL",
            },
            "global_score_stratified_dnsmos": {
                "selected_hours": "stratified selected hours",
                "unique_sources": "stratified unique sources",
                "effective_source_count": "stratified effective sources",
                "lexical_type_coverage": "stratified lexical coverage",
                "mean_dnsmos_ovrl": "stratified mean OVRL",
            },
        }
        for policy, tags in table_tags.items():
            matches = composition[
                (composition["study"] == "natural") & composition["policy"].astype(str).str.endswith(f"/{policy}")
            ]
            if matches.empty:
                continue
            row = matches.iloc[0]
            for column, tag in tags.items():
                if column in {"selected_hours", "effective_source_count", "mean_dnsmos_ovrl"}:
                    value = f"{float(row[column]):.2f}"
                elif column == "lexical_type_coverage":
                    value = f"{float(row[column]):.3f}"
                else:
                    value = str(int(row[column]))
                values.append({"tag": tag, "value": value, "source": "selection_composition.csv"})
    return pd.DataFrame(values)


def analyse_all(cfg: dict[str, Any]) -> dict[str, str]:
    results = Path(cfg["paths"]["results"])
    paper = Path(cfg["paths"]["paper_generated"])
    primary = read_csv(Path(cfg["paths"]["manifests"]) / "scored" / "primary_candidates_scored.csv")
    wer = read_csv(results / "wer.csv")
    composition = read_csv(results / "selection_composition.csv")
    metric = _pair_metric_behaviour(primary, cfg)
    summary = _aggregate_wer(wer)
    contrasts = _policy_contrasts(wer, cfg)
    write_csv(metric, results / "metric_behavior.csv")
    write_csv(summary, results / "wer_summary.csv")
    write_csv(contrasts, results / "policy_contrasts.csv")
    values = _placeholder_values(summary, contrasts, metric, composition, cfg)
    write_csv(values, paper / "placeholder_values.csv")

    dpi = int(cfg["report"]["figure_dpi"])
    _plot_metric_behaviour(metric, paper / "fig_metric_choice_accuracy.pdf", dpi)
    _plot_paired_wer(summary, cfg, paper / "fig_paired_wer.pdf", dpi)
    _plot_global_pareto(summary, composition, cfg, paper / "fig_global_pareto.pdf", dpi)

    markdown = [
        "# Results to paste into `paper/main.tex`",
        "",
        "Each row below is derived automatically from the immutable manifests and decoded predictions. "
        "Use it to replace the identically named `\\placeholder{...}` tag; review the direction/sign convention before writing prose.",
        "",
        "| Placeholder tag | Value | Source |",
        "|---|---:|---|",
    ]
    for row in values.itertuples(index=False):
        markdown.append(f"| `{row.tag}` | {row.value} | `{row.source}` |")
    markdown.extend(
        [
            "",
            "`primary_dnsmos_minus_random_percentage_points` is DNSMOS-selected WER minus random-selected WER; negative values favor DNSMOS selection.",
            "The paired sign-flip p-value is the pre-specified primary analysis only at the 25-hour clean condition; it is not an externally preregistered confirmatory test. All other p-values are exploratory and Holm-adjusted where reported.",
        ]
    )
    (paper / "RESULTS_TO_PASTE.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(
        {
            "metric_behavior": str(results / "metric_behavior.csv"),
            "wer_summary": str(results / "wer_summary.csv"),
            "policy_contrasts": str(results / "policy_contrasts.csv"),
            "placeholder_values": str(paper / "placeholder_values.csv"),
        },
        results / "analysis_index.json",
    )
    return {"analysis": str(results / "analysis_index.json")}
