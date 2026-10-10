#!/usr/bin/env python3
"""Regenerate the paper's rankability Figures 5 and 6 using all models.

The original nine methods are loaded from the complete-results CSV files. The
six newer ordinal baselines (SORD, Beta, POE, CORE, DRF, and CORF) are loaded
from the local W&B summaries referenced by ``logs/ordinal_baselines``.

Outputs:
  - figure5_rankability_vs_average_kappa_all_models.pdf
  - figure6_rankability_original_baselines.pdf
  - figure6_rankability_modern_ordinal_methods.pdf
  - figure6_summary_all_models.tex
  - figure5_data_all_models.csv
  - figure6_data_all_models.csv
  - figure6_summary_all_models.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from generate_ogp_significance_tables import (
    FEATURE_ORDER,
    MODEL_NAMES,
    MODEL_ORDER,
    load_new_results,
)


DATASETS = ("sicap", "crowdgleason")
DATASET_NAMES = {"sicap": "SICAPv2", "crowdgleason": "CrowdGleason"}
DATASET_MARKERS = {"sicap": "o", "crowdgleason": "s"}
EXPECTED_FOLDS = tuple(range(5))
ORIGINAL_MODELS = (
    "LogisticRegression",
    "RLOrdinal",
    "KNN",
    "RandomForest",
    "MLP",
    "OrdinalMLP",
    "SVM",
    "SVGP_multiclass",
    "OrdinalSVGP",
)
MODERN_ORDINAL_MODELS = (
    "SORD",
    "Beta",
    "POE",
    "CORE",
    "DRF",
    "CORF",
    "OrdinalSVGP",
)
MODEL_COLORS = dict(
    zip(MODEL_ORDER, plt.get_cmap("tab20")(np.arange(len(MODEL_ORDER))))
)
# Keep the proposed method visually identical in every comparison panel.
MODEL_COLORS["OrdinalSVGP"] = "#4D4D4D"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "src/notebooks/complete_results",
        help="Directory containing results_<dataset>.csv and rankability.csv.",
    )
    parser.add_argument(
        "--run-logs",
        type=Path,
        default=root / "logs/ordinal_baselines",
        help="Directory containing the six newer ordinal-baseline run logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/rankability_figures",
    )
    parser.add_argument("--percentage", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write PNG copies of all figures.",
    )
    return parser.parse_args()


def load_rankability(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    missing_rows = sorted(set(DATASET_NAMES.values()) - set(frame.index))
    missing_columns = sorted(set(FEATURE_ORDER) - set(frame.columns))
    if missing_rows or missing_columns:
        raise ValueError(
            "Invalid rankability table; missing rows "
            f"{missing_rows} and columns {missing_columns}."
        )
    values = frame.loc[list(DATASET_NAMES.values()), list(FEATURE_ORDER)]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Rankability scores must all be finite.")
    return frame


def load_all_runs(args: argparse.Namespace, root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        path = args.results_dir / f"results_{dataset}.csv"
        frame = pd.read_csv(path)
        frame = frame[
            np.isclose(frame["percentage"], args.percentage)
            & frame["model"].isin(MODEL_ORDER)
        ].copy()
        frame["dataset"] = dataset
        frame["source"] = str(path)
        frames.append(frame)

    newer = load_new_results(args.run_logs, root, set(DATASETS))
    if newer.empty:
        raise RuntimeError(
            "No newer ordinal-baseline results were found in "
            f"{args.run_logs}."
        )
    frames.append(newer)
    result = pd.concat(frames, ignore_index=True, sort=False)

    keys = ["dataset", "model", "features", "fold"]
    duplicates = result.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate result rows:\n"
            + result.loc[duplicates, [*keys, "source"]].to_string(index=False)
        )

    expected = {
        (dataset, model, feature, fold)
        for dataset in DATASETS
        for model in MODEL_ORDER
        for feature in FEATURE_ORDER
        for fold in EXPECTED_FOLDS
    }
    observed = set(result[keys].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        message = []
        if missing:
            preview = ", ".join("/".join(map(str, row)) for row in missing[:20])
            message.append(f"missing {len(missing)} rows: {preview}")
        if unexpected:
            preview = ", ".join(
                "/".join(map(str, row)) for row in unexpected[:20]
            )
            message.append(f"unexpected {len(unexpected)} rows: {preview}")
        raise ValueError("Incomplete result grid; " + "; ".join(message))

    return result


def make_derived_data(
    runs: pd.DataFrame, rankability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_data = (
        runs.groupby(["dataset", "model", "features"], observed=True)[
            "test/kappa"
        ]
        .mean()
        .rename("kappa")
        .reset_index()
    )
    model_data["rankability"] = [
        float(rankability.loc[DATASET_NAMES[dataset], feature])
        for dataset, feature in zip(
            model_data["dataset"], model_data["features"]
        )
    ]
    model_data["model_display"] = model_data["model"].map(MODEL_NAMES)
    model_data["dataset_display"] = model_data["dataset"].map(DATASET_NAMES)

    figure5_data = (
        model_data.groupby(
            ["dataset", "dataset_display", "features", "rankability"],
            observed=True,
            as_index=False,
        )
        .agg(
            kappa_max=("kappa", "max"),
            kappa_mean=("kappa", "mean"),
            kappa_std_across_models=("kappa", "std"),
            n_models=("model", "nunique"),
        )
    )

    summary_rows = []
    for model in MODEL_ORDER:
        subset = model_data[model_data["model"] == model]
        regression = linregress(subset["rankability"], subset["kappa"])
        summary_rows.append(
            {
                "model": model,
                "model_display": MODEL_NAMES[model],
                "average_kappa": subset["kappa"].mean(),
                "slope": regression.slope,
                "intercept": regression.intercept,
                "r_value": regression.rvalue,
                "p_value": regression.pvalue,
                "slope_standard_error": regression.stderr,
                "n_scenarios": len(subset),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return model_data, figure5_data, summary


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def write_latex_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Write the Figure 6 summary as a LaTeX snippet for a subfigure."""
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Avg. Kappa & Slope \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.model_display} & {row.average_kappa:.3f} & "
            rf"{row.slope:.3f} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(
    figure: plt.Figure, output_dir: Path, stem: str, dpi: int, png: bool
) -> None:
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    if png:
        figure.savefig(
            output_dir / f"{stem}.png", bbox_inches="tight", dpi=dpi
        )
    plt.close(figure)


def plot_figure5(
    data: pd.DataFrame, output_dir: Path, dpi: int, png: bool
) -> tuple[float, float]:
    colors = dict(
        zip(FEATURE_ORDER, plt.get_cmap("tab10")(np.arange(len(FEATURE_ORDER))))
    )
    rho, p_value = spearmanr(data["rankability"], data["kappa_mean"])

    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    for row in data.itertuples(index=False):
        axis.scatter(
            row.rankability,
            row.kappa_mean,
            marker=DATASET_MARKERS[row.dataset],
            color=colors[row.features],
            s=115,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

    feature_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            label=feature,
            markerfacecolor=colors[feature],
            markeredgecolor="white",
            markersize=10,
        )
        for feature in FEATURE_ORDER
    ]
    feature_legend = axis.legend(
        handles=feature_handles,
        title="Features",
        loc="upper left",
        frameon=True,
        fontsize=16,
        title_fontsize=18,
    )
    axis.add_artist(feature_legend)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS[dataset],
            linestyle="none",
            label=DATASET_NAMES[dataset],
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=10,
        )
        for dataset in DATASETS
    ]
    axis.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="lower right",
        frameon=True,
        fontsize=16,
        title_fontsize=18,
    )
    axis.set_xlabel("Rankability Score", fontsize=18)
    axis.set_ylabel("Average Cohen's Kappa", fontsize=18)
    axis.tick_params(labelsize=14)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.text(
        0.96,
        0.21,
        rf"Spearman $\rho$: {rho:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=17,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        "figure5_rankability_vs_average_kappa_all_models",
        dpi,
        png,
    )
    return float(rho), float(p_value)


def plot_figure6_group(
    model_data: pd.DataFrame,
    models: tuple[str, ...],
    output_dir: Path,
    output_stem: str,
    dpi: int,
    png: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 8.0))

    for model in models:
        subset = model_data[model_data["model"] == model].sort_values(
            "rankability"
        )
        axis.plot(
            subset["rankability"],
            subset["kappa"],
            color=MODEL_COLORS[model],
            marker=".",
            label=MODEL_NAMES[model],
            linewidth=2.0,
            markersize=9,
            alpha=0.95,
        )

    axis.set_xlabel("Rankability Score", fontsize=18)
    axis.set_ylabel("Average Cohen's Kappa", fontsize=18)
    # Keep identical axes across the two Figure 6 plots so that their visual
    # slopes and performance ranges remain directly comparable.
    axis.set_xlim(0.615, 0.875)
    axis.set_ylim(0.15, 0.93)
    axis.tick_params(labelsize=14)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(
        loc="lower right",
        ncol=3 if len(models) >= 9 else 2,
        frameon=True,
        fontsize=17,
        columnspacing=1.2,
        handlelength=2.2,
    )
    figure.tight_layout()
    save_figure(
        figure,
        output_dir,
        output_stem,
        dpi,
        png,
    )


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankability = load_rankability(args.results_dir / "rankability.csv")
    runs = load_all_runs(args, root)
    model_data, figure5_data, summary = make_derived_data(runs, rankability)

    model_data.to_csv(
        args.output_dir / "figure6_data_all_models.csv", index=False
    )
    figure5_data.to_csv(
        args.output_dir / "figure5_data_all_models.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "figure6_summary_all_models.csv", index=False
    )
    write_latex_summary(
        summary, args.output_dir / "figure6_summary_all_models.tex"
    )

    configure_plotting()
    rho, p_value = plot_figure5(
        figure5_data, args.output_dir, args.dpi, args.png
    )
    plot_figure6_group(
        model_data,
        ORIGINAL_MODELS,
        args.output_dir,
        "figure6_rankability_original_baselines",
        args.dpi,
        args.png,
    )
    plot_figure6_group(
        model_data,
        MODERN_ORDINAL_MODELS,
        args.output_dir,
        "figure6_rankability_modern_ordinal_methods",
        args.dpi,
        args.png,
    )

    print(f"Loaded {len(runs)} runs across {len(MODEL_ORDER)} models.")
    print(f"Figure 5 Spearman rho={rho:.6f}, p={p_value:.6g}.")
    print(f"Wrote outputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
