#!/usr/bin/env python3
"""Generate the paper's result and rankability LaTeX tables from CSV files.

For every dataset, metric, and foundation model, the method having the best
mean value is compared with the method having the second-best mean value.
Tests are paired by fold. The best mean receives a superscript star when the
corrected p-value is below alpha.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


MODEL_ORDER = (
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
MODEL_NAMES = {
    "LogisticRegression": "LR",
    "RLOrdinal": "OLR",
    "KNN": "KNN",
    "RandomForest": "RF",
    "MLP": "MLP",
    "OrdinalMLP": "OMLP",
    "SVM": "SVM",
    "SVGP_multiclass": "MGP",
    "OrdinalSVGP": "OGP",
}
FEATURE_ORDER = (
    "CONCHV15",
    "GIGAPATH",
    "PHIKONV2",
    "UNI",
    "UNI2",
    "VIRCHOW",
    "VIRCHOW2",
)
METRICS = {
    "kappa": ("test/kappa", "max", "Kappa $\\uparrow$"),
    "ece": ("test/ece", "min", "ECE $\\downarrow$"),
}
def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "src" / "notebooks" / "complete_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing results_sicap.csv and results_crowdgleason.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "paper_tables",
        help="Directory in which LaTeX tables and test details are written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("sicap", "crowdgleason"),
        help="Dataset suffixes used in results_<dataset>.csv.",
    )
    parser.add_argument("--percentage", type=float, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--test",
        choices=("paired-t", "wilcoxon"),
        default="paired-t",
        help="Paired test applied to runs matched by fold.",
    )
    parser.add_argument(
        "--correction",
        choices=("holm", "none"),
        default="holm",
        help="Multiple-testing correction across foundation models in each table.",
    )
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument(
        "--rankability-only",
        action="store_true",
        help="Generate only rankability_table.tex from rankability.csv.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    if args.decimals < 0:
        raise ValueError("--decimals cannot be negative")


def load_results(path: Path, percentage: float) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "model",
        "fold",
        "features",
        "percentage",
        *(spec[0] for spec in METRICS.values()),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    frame = frame[np.isclose(frame["percentage"], percentage)].copy()
    if frame.empty:
        raise ValueError(f"No rows in {path} have percentage={percentage:g}")

    duplicate = frame.duplicated(["features", "model", "fold"], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, ["features", "model", "fold"]]
        raise ValueError(
            "Expected one result per feature/model/fold, but found duplicates:\n"
            + keys.drop_duplicates().to_string(index=False)
        )

    missing_models = sorted(set(MODEL_ORDER).difference(frame["model"].unique()))
    missing_features = sorted(set(FEATURE_ORDER).difference(frame["features"].unique()))
    if missing_models or missing_features:
        raise ValueError(
            f"Incomplete data in {path}; missing models={missing_models}, "
            f"missing features={missing_features}"
        )
    return frame


def load_rankability(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Rankability file not found: {path}")
    frame = pd.read_csv(path, index_col=0)
    expected_columns = [*FEATURE_ORDER, "Average"]
    expected_datasets = ["SICAPv2", "CrowdGleason"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"Expected rankability columns {expected_columns}, "
            f"but found {frame.columns.tolist()}"
        )
    if frame.index.tolist() != expected_datasets:
        raise ValueError(
            f"Expected rankability datasets {expected_datasets}, "
            f"but found {frame.index.tolist()}"
        )
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Rankability scores must all be finite numbers")
    return frame


def paired_p_value(first: np.ndarray, second: np.ndarray, test: str) -> float:
    differences = first - second
    if np.allclose(differences, 0.0):
        return 1.0
    if test == "paired-t":
        return float(ttest_rel(first, second).pvalue)
    return float(wilcoxon(first, second, alternative="two-sided").pvalue)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def summarize(frame: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    return (
        frame.groupby(["features", "model"], observed=True)[metric_column]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


def significance_tests(
    frame: pd.DataFrame,
    metric_column: str,
    direction: str,
    test: str,
    correction: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_ORDER:
        feature_frame = frame[frame["features"] == feature]
        means = feature_frame.groupby("model")[metric_column].mean()
        ranked_models = means.sort_values(
            ascending=direction == "min", kind="stable"
        ).index.tolist()
        if len(ranked_models) < 2:
            raise ValueError(f"Need at least two models for {feature}")
        best_model, second_best_model = ranked_models[:2]

        paired = (
            feature_frame[feature_frame["model"] == best_model][["fold", metric_column]]
            .rename(columns={metric_column: "best"})
            .merge(
                feature_frame[feature_frame["model"] == second_best_model][
                    ["fold", metric_column]
                ].rename(columns={metric_column: "second_best"}),
                on="fold",
                how="inner",
                validate="one_to_one",
            )
            .sort_values("fold")
        )
        if len(paired) < 2:
            raise ValueError(
                f"Need at least two matched folds for {feature}: "
                f"{best_model} versus {second_best_model}"
            )
        p_raw = paired_p_value(
            paired["best"].to_numpy(), paired["second_best"].to_numpy(), test
        )
        rows.append(
            {
                "feature": feature,
                "best_model": best_model,
                "best_model_display": MODEL_NAMES[best_model],
                "second_best_model": second_best_model,
                "second_best_model_display": MODEL_NAMES[second_best_model],
                "n_pairs": len(paired),
                "best_mean": paired["best"].mean(),
                "second_best_mean": paired["second_best"].mean(),
                "mean_difference_best_minus_second": (
                    paired["best"] - paired["second_best"]
                ).mean(),
                "p_raw": p_raw,
            }
        )

    result = pd.DataFrame(rows)
    result["p_adjusted"] = (
        holm_adjust(result["p_raw"].tolist())
        if correction == "holm"
        else result["p_raw"].to_numpy()
    )
    result["significant"] = result["p_adjusted"] < alpha
    return result


def format_cell(mean: float, std: float, best: bool, significant: bool, decimals: int) -> str:
    mean_text = f"{mean:.{decimals}f}"
    std_text = f"{std:.{decimals}f}"
    if best:
        mean_text = rf"\mathbf{{{mean_text}}}"
        std_text = rf"\mathbf{{{std_text}}}"
    if significant:
        mean_text += r"^{*}"
    return rf"${mean_text} \pm {std_text}$"


def make_latex_table(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    metric_label: str,
    direction: str,
    test: str,
    correction: str,
    alpha: float,
    decimals: int,
) -> str:
    significant_winners = {
        row.feature: row.best_model
        for row in tests.loc[tests["significant"]].itertuples()
    }
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * len(FEATURE_ORDER) + "}",
        r"\toprule",
        "Model & " + " & ".join(FEATURE_ORDER) + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = []
        for feature in FEATURE_ORDER:
            feature_summary = summary[summary["features"] == feature].set_index("model")
            row = feature_summary.loc[model]
            best_value = (
                feature_summary["mean"].max()
                if direction == "max"
                else feature_summary["mean"].min()
            )
            cells.append(
                format_cell(
                    row["mean"],
                    row["std"],
                    best=np.isclose(row["mean"], best_value),
                    significant=significant_winners.get(feature) == model,
                    decimals=decimals,
                )
            )
        lines.append(MODEL_NAMES[model] + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])

    test_label = "two-sided paired $t$-test" if test == "paired-t" else "two-sided Wilcoxon signed-rank test"
    correction_label = (
        "Holm-adjusted across the seven foundation models"
        if correction == "holm"
        else "unadjusted"
    )
    lines.extend(
        [
            r"\par\smallskip",
            rf"\noindent\footnotesize Bold values indicate the best mean {metric_label} for each foundation model. "
            rf"$^{{*}}$ indicates that the best method differs significantly from the second-best method "
            rf"({test_label}, {correction_label}, $p < {alpha:g}$).",
        ]
    )
    return "\n".join(lines) + "\n"


def make_rankability_latex(frame: pd.DataFrame, decimals: int) -> str:
    columns = [*FEATURE_ORDER, "Average"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Held-out linear rankability of the foundation-model "
        r"representations. The linear ordering direction is fitted on the "
        r"complete training split, and rankability is the signed Spearman "
        r"correlation $\rho_{\mathrm{rank}}$ evaluated on the independent "
        r"test split. The average assigns equal weight to each of the seven "
        r"foundation models.}",
        r"\label{tab:rankability_scores}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * len(FEATURE_ORDER) + "|c}",
        r"\toprule",
        "Dataset & " + " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for dataset, row in frame.iterrows():
        values = [f"${float(row[column]):.{decimals}f}$" for column in columns]
        lines.append(dataset + " & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\par\smallskip",
            r"\noindent\footnotesize Higher positive values indicate stronger "
            r"correctly oriented ordinal structure along the learned linear direction.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankability = load_rankability(args.input_dir / "rankability.csv")
    rankability_path = args.output_dir / "rankability_table.tex"
    rankability_path.write_text(
        make_rankability_latex(rankability, args.decimals), encoding="utf-8"
    )
    print(f"Wrote {rankability_path}")
    if args.rankability_only:
        return

    all_tests = []

    for dataset in args.datasets:
        input_path = args.input_dir / f"results_{dataset}.csv"
        frame = load_results(input_path, args.percentage)
        for metric_name, (metric_column, direction, metric_label) in METRICS.items():
            metric_summary = summarize(frame, metric_column)
            tests = significance_tests(
                frame,
                metric_column,
                direction,
                args.test,
                args.correction,
                args.alpha,
            )
            tests.insert(0, "metric", metric_name)
            tests.insert(0, "dataset", dataset)
            all_tests.append(tests)

            latex = make_latex_table(
                metric_summary,
                tests,
                metric_label,
                direction,
                args.test,
                args.correction,
                args.alpha,
                args.decimals,
            )
            output_path = args.output_dir / f"{dataset}_{metric_name}_table.tex"
            output_path.write_text(latex, encoding="utf-8")
            print(f"Wrote {output_path}")

    test_path = args.output_dir / "significance_tests.csv"
    pd.concat(all_tests, ignore_index=True).to_csv(test_path, index=False)
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
