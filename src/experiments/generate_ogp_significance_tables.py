#!/usr/bin/env python3
"""Generate full ECE/Kappa tables with pairwise OGP comparison shading.

For each dataset, metric, and foundation-model representation, OGP is
compared separately with every competing method using runs matched by fold.
Each competing-method cell is pastel green when OGP is better and pastel red
otherwise. Non-significant comparisons retain that color and receive a
superscript dagger. The OGP row is the unshaded comparison reference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


OGP_BETTER_COLOR = "D9EAD3"
OGP_NOT_BETTER_COLOR = "F4CCCC"
FEATURE_ORDER = (
    "CONCHV15", "GIGAPATH", "PHIKONV2", "UNI", "UNI2", "VIRCHOW", "VIRCHOW2",
)
MODEL_ORDER = (
    "LogisticRegression", "RLOrdinal", "KNN", "RandomForest", "MLP",
    "OrdinalMLP", "SVM", "SVGP_multiclass", "SORD", "Beta", "POE", "CORE",
    "DRF", "CORF", "OrdinalSVGP",
)
MODEL_NAMES = {
    "LogisticRegression": "LR", "RLOrdinal": "OLR", "KNN": "KNN",
    "RandomForest": "RF", "MLP": "MLP", "OrdinalMLP": "OMLP", "SVM": "SVM",
    "SVGP_multiclass": "MGP", "OrdinalSVGP": "OGP", "SORD": "SORD",
    "Beta": "Beta", "POE": "POE", "CORE": "CORE", "DRF": "DRF", "CORF": "CORF",
}
NEW_MODELS = {
    "sord": "SORD", "beta": "Beta", "poe": "POE", "core": "CORE",
    "drf": "DRF", "corf": "CORF",
}
METRICS = {
    "ece": ("test/ece", "min", "ECE $\\downarrow$"),
    "kappa": ("test/kappa", "max", "Kappa $\\uparrow$"),
}
RUN_LOG_PATTERN = re.compile(
    r"^(?P<dataset>sicap|crowdgleason)_(?P<feature>[^_]+)_"
    r"(?P<model>sord|beta|poe|core|drf|corf)_seed(?P<seed>\d+)\.log$"
)
WANDB_LOG_PATTERN = re.compile(r"Find logs at: \./(?P<path>wandb/run-[^/]+)/logs")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "src/notebooks/complete_results",
    )
    parser.add_argument(
        "--run-logs",
        type=Path,
        default=root / "logs/ordinal_baselines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/paper_tables",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=("sicap", "crowdgleason")
    )
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--test", choices=("paired-t", "wilcoxon"), default="paired-t"
    )
    parser.add_argument(
        "--correction", choices=("holm", "none"), default="holm"
    )
    parser.add_argument(
        "--correction-scope",
        choices=("dataset-metric", "feature"),
        default="feature",
        help=(
            "Apply multiplicity correction separately across the OGP comparisons "
            "within each feature extractor (default), or across all comparisons "
            "in a dataset/metric table."
        ),
    )
    return parser.parse_args()


def load_new_results(run_logs: Path, root: Path, datasets: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_path in sorted(run_logs.glob("*_seed*.log")):
        match = RUN_LOG_PATTERN.match(log_path.name)
        if match is None or match.group("dataset") not in datasets:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        wandb_matches = list(WANDB_LOG_PATTERN.finditer(text))
        if not wandb_matches:
            continue
        summary_path = root / wandb_matches[-1].group("path") / "files/wandb-summary.json"
        if not summary_path.is_file():
            continue
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if not all(column in summary for column, _, _ in METRICS.values()):
            continue
        rows.append({
            "dataset": match.group("dataset"),
            "model": NEW_MODELS[match.group("model")],
            "fold": int(match.group("seed")),
            "features": match.group("feature"),
            "percentage": 100.0,
            "test/ece": float(summary["test/ece"]),
            "test/kappa": float(summary["test/kappa"]),
            "source": str(summary_path.relative_to(root)),
        })
    return pd.DataFrame(rows)


def load_results(args: argparse.Namespace, root: Path) -> pd.DataFrame:
    frames = []
    for dataset in args.datasets:
        path = args.input_dir / f"results_{dataset}.csv"
        frame = pd.read_csv(path)
        frame = frame[np.isclose(frame["percentage"], args.percentage)].copy()
        frame = frame[frame["model"].isin(MODEL_ORDER)]
        frame["dataset"] = dataset
        frame["source"] = str(path.relative_to(root))
        frames.append(frame)
    new = load_new_results(args.run_logs, root, set(args.datasets))
    if not new.empty:
        frames.append(new)
    result = pd.concat(frames, ignore_index=True, sort=False)
    duplicate = result.duplicated(
        ["dataset", "model", "features", "fold"], keep=False
    )
    if duplicate.any():
        keys = ["dataset", "model", "features", "fold", "source"]
        raise ValueError(
            "Duplicate runs:\n" + result.loc[duplicate, keys].to_string(index=False)
        )
    return result


def summarize(frame: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [MODEL_ORDER, FEATURE_ORDER], names=["model", "features"]
    )
    return (
        frame.groupby(["model", "features"], observed=True)[metric_column]
        .agg(mean="mean", std="std", n="count")
        .reindex(index)
        .reset_index()
    )


def paired_p_value(first: np.ndarray, second: np.ndarray, test: str) -> float:
    if np.allclose(first - second, 0.0):
        return 1.0
    if test == "paired-t":
        return float(ttest_rel(first, second).pvalue)
    return float(wilcoxon(first, second, alternative="two-sided").pvalue)


def holm_adjust(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    order = valid.sort_values().index
    adjusted: dict[int, float] = {}
    running_max = 0.0
    total = len(valid)
    for rank, index in enumerate(order):
        running_max = max(running_max, (total - rank) * float(valid.loc[index]))
        adjusted[index] = min(running_max, 1.0)
    return pd.Series(adjusted, dtype=float).reindex(values.index)


def ogp_comparisons(
    frame: pd.DataFrame,
    metric_column: str,
    direction: str,
    test: str,
    correction: str,
    correction_scope: str,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_ORDER:
        feature_frame = frame[frame["features"] == feature]
        ogp = feature_frame[feature_frame["model"] == "OrdinalSVGP"][
            ["fold", metric_column]
        ].rename(columns={metric_column: "ogp"})

        for model in MODEL_ORDER:
            if model == "OrdinalSVGP":
                continue
            competitor = feature_frame[feature_frame["model"] == model][
                ["fold", metric_column]
            ].rename(columns={metric_column: "competitor"})
            paired = (
                ogp.merge(
                    competitor,
                    on="fold",
                    how="inner",
                    validate="one_to_one",
                )
                .dropna()
                .sort_values("fold")
            )
            p_raw = np.nan
            status = "insufficient matched runs"
            if len(paired) >= 2:
                p_raw = paired_p_value(
                    paired["ogp"].to_numpy(),
                    paired["competitor"].to_numpy(),
                    test,
                )
                status = "tested"
            ogp_mean = paired["ogp"].mean()
            competitor_mean = paired["competitor"].mean()
            if paired.empty:
                ogp_better = np.nan
            elif direction == "min":
                ogp_better = ogp_mean < competitor_mean
            else:
                ogp_better = ogp_mean > competitor_mean
            rows.append(
                {
                    "feature": feature,
                    "model": model,
                    "model_display": MODEL_NAMES[model],
                    "n_pairs": len(paired),
                    "ogp_mean": ogp_mean,
                    "competitor_mean": competitor_mean,
                    "mean_difference_ogp_minus_competitor": (
                        paired["ogp"] - paired["competitor"]
                    ).mean(),
                    "ogp_better": ogp_better,
                    "p_raw": p_raw,
                    "status": status,
                }
            )

    result = pd.DataFrame(rows)
    if correction == "none":
        result["p_adjusted"] = result["p_raw"]
    elif correction_scope == "feature":
        result["p_adjusted"] = result.groupby("feature", group_keys=False)[
            "p_raw"
        ].transform(holm_adjust)
    else:
        result["p_adjusted"] = holm_adjust(result["p_raw"])
    result["significant"] = result["p_adjusted"] < alpha
    return result


def format_value(
    row: pd.Series,
    best: bool,
    ogp_better: bool | None,
    significant: bool | None,
    decimals: int,
) -> str:
    if pd.isna(row["n"]) or int(row["n"]) == 0:
        return "--"
    mean = f"{row['mean']:.{decimals}f}"
    std = f"{row['std']:.{decimals}f}" if int(row["n"]) > 1 else "--"
    if best:
        mean = rf"\mathbf{{{mean}}}"
        if std != "--":
            std = rf"\mathbf{{{std}}}"
    value = rf"${mean} \pm {std}$"
    if ogp_better is None:
        return value

    color = OGP_BETTER_COLOR if ogp_better else OGP_NOT_BETTER_COLOR
    if significant is False:
        value = value[:-1] + r"^{\dagger}$"
    return rf"\cellcolor[HTML]{{{color}}}{value}"


def make_latex_table(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    direction: str,
    decimals: int,
) -> str:
    comparison_lookup = comparisons.set_index(["feature", "model"])
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * len(FEATURE_ORDER) + "}",
        r"\toprule",
        "Model & " + " & ".join(FEATURE_ORDER) + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model == "OrdinalSVGP":
            lines.append(r"\midrule")
        model_rows = summary[summary["model"] == model].set_index("features")
        cells = []
        for feature in FEATURE_ORDER:
            row = model_rows.loc[feature]
            available = summary[
                (summary["features"] == feature) & summary["mean"].notna()
            ]
            best_value = available["mean"].agg(direction)
            significance: bool | None = None
            ogp_better: bool | None = None
            if model != "OrdinalSVGP" and (feature, model) in comparison_lookup.index:
                comparison = comparison_lookup.loc[(feature, model)]
                if not pd.isna(comparison["ogp_better"]):
                    ogp_better = bool(comparison["ogp_better"])
                if comparison["status"] == "tested":
                    significance = bool(comparison["significant"])
            is_best = bool(np.isclose(row["mean"], best_value, equal_nan=False))
            cells.append(
                format_value(
                    row,
                    best=is_best,
                    ogp_better=ogp_better,
                    significant=significance,
                    decimals=decimals,
                )
            )
        lines.append(
            MODEL_NAMES[model] + " & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    if args.decimals < 0:
        raise ValueError("--decimals cannot be negative")

    root = Path(__file__).resolve().parents[2]
    frame = load_results(args, root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_comparisons = []

    for dataset in args.datasets:
        dataset_frame = frame[frame["dataset"] == dataset]
        for metric, (column, direction, _label) in METRICS.items():
            summary = summarize(dataset_frame, column)
            comparisons = ogp_comparisons(
                dataset_frame,
                column,
                direction,
                args.test,
                args.correction,
                args.correction_scope,
                args.alpha,
            )
            comparisons.insert(0, "metric", metric)
            comparisons.insert(0, "dataset", dataset)
            all_comparisons.append(comparisons)
            latex = make_latex_table(
                summary,
                comparisons,
                direction,
                args.decimals,
            )
            output_path = args.output_dir / f"{dataset}_ogp_comparison_{metric}_table.tex"
            output_path.write_text(latex, encoding="utf-8")
            print(f"Wrote {output_path}")

    details_path = args.output_dir / "ogp_pairwise_significance_tests.csv"
    pd.concat(all_comparisons, ignore_index=True).to_csv(details_path, index=False)
    print(f"Wrote {details_path}")
    print(r"LaTeX requirement: \usepackage[table]{xcolor}")


if __name__ == "__main__":
    main()
