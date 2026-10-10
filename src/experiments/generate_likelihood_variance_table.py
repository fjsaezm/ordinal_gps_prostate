#!/usr/bin/env python3
"""Create the rebuttal LaTeX table for the likelihood-variance comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURE_ORDER = (
    "CONCHV15", "GIGAPATH", "PHIKONV2", "UNI", "UNI2", "VIRCHOW", "VIRCHOW2",
)
VALUE_COLUMNS = (
    "likelihood_variance",
    "kappa_without_likelihood_variance",
    "kappa",
    "ece_without_likelihood_variance",
    "ece",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("sicap", "crowdgleason"), default="sicap"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Comparison CSV produced by evaluate_crowdgleason_likelihood_variance.py.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def format_value(values: pd.Series, decimals: int) -> str:
    return rf"${values.mean():.{decimals}f} \pm {values.std(ddof=1):.{decimals}f}$"


def validate(frame: pd.DataFrame) -> None:
    required = {"feature", "seed", *VALUE_COLUMNS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {', '.join(missing_columns)}")
    duplicates = frame.duplicated(["feature", "seed"], keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, ["feature", "seed"]].to_dict("records")
        raise ValueError(f"Duplicate feature/seed rows: {values}")
    missing_runs = [
        f"{feature}/seed{seed}"
        for feature in FEATURE_ORDER
        for seed in range(5)
        if not ((frame["feature"] == feature) & (frame["seed"] == seed)).any()
    ]
    if missing_runs:
        raise ValueError("Missing evaluations: " + ", ".join(missing_runs))


def make_latex(frame: pd.DataFrame, dataset: str, decimals: int) -> str:
    dataset_label = "SICAP" if dataset == "sicap" else "CrowdGleason"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of retaining the learned ordinal-likelihood variance at test",
        f"time on {dataset_label}. Values are mean $\\pm$ standard deviation over five seeds.}}",
        rf"\label{{tab:{dataset}-likelihood-variance}}",
        "",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& Learned variance",
        r"& \multicolumn{2}{c}{$\kappa$}",
        r"& \multicolumn{2}{c}{ECE} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Encoder",
        r"& $\widehat{\sigma}_{\delta,\mathrm{train}}^2$",
        r"& $\sigma_{\delta,\mathrm{pred}}^2=0$",
        r"& $\sigma_{\delta,\mathrm{pred}}^2=\widehat{\sigma}_{\delta,\mathrm{train}}^2$",
        r"& $\sigma_{\delta,\mathrm{pred}}^2=0$",
        r"& $\sigma_{\delta,\mathrm{pred}}^2=\widehat{\sigma}_{\delta,\mathrm{train}}^2$ \\",
        r"\midrule",
    ]
    for feature in FEATURE_ORDER:
        feature_frame = frame[frame["feature"] == feature]
        values = [format_value(feature_frame[column], decimals) for column in VALUE_COLUMNS]
        lines.append(feature + " & " + " & ".join(values) + r" \\")
    all_values = [format_value(frame[column], decimals) for column in VALUE_COLUMNS]
    lines.extend(
        [
            r"\midrule",
            "All runs & " + " & ".join(all_values) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input or Path(
        f"results/{args.dataset}_ordinal_svgp_likelihood_variance_on.csv"
    )
    output_path = args.output or Path(
        f"rebuttal/{args.dataset}_likelihood_variance_table.tex"
    )
    frame = pd.read_csv(input_path)
    validate(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        make_latex(frame, args.dataset, args.decimals), encoding="utf-8"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
