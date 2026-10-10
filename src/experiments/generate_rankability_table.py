#!/usr/bin/env python3
"""Generate the paper's rankability LaTeX table from rankability.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_ORDER = (
    "CONCHV15", "GIGAPATH", "PHIKONV2", "UNI", "UNI2", "VIRCHOW", "VIRCHOW2",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "src/notebooks/complete_results/rankability.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/paper_tables/rankability_table.tex",
    )
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def load_rankability(path: Path) -> pd.DataFrame:
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


def make_latex(frame: pd.DataFrame, decimals: int) -> str:
    columns = [*FEATURE_ORDER, "Average"]
    summary_columns = ["Median", r"LOMO mean range"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Held-out linear rankability of the foundation-model "
        r"representations. The linear ordering direction is fitted on the "
        r"complete training split, and rankability is the signed Spearman "
        r"correlation $\rho_{\mathrm{rank}}$ evaluated on the independent "
        r"test split. The average and median summarize the seven foundation "
        r"models. The leave-one-model-out (LOMO) column reports "
        r"the range of the seven means obtained after omitting each foundation "
        r"model in turn.}",
        r"\label{tab:rankability_scores}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * len(FEATURE_ORDER) + "|ccc}",
        r"\toprule",
        "Dataset & " + " & ".join([*columns, *summary_columns]) + r" \\",
        r"\midrule",
    ]
    for dataset, row in frame.iterrows():
        values = [f"${float(row[column]):.{decimals}f}$" for column in columns]
        scores = row.loc[list(FEATURE_ORDER)].to_numpy(dtype=float)
        median = float(np.median(scores))
        lofo_means = (scores.sum() - scores) / (len(scores) - 1)
        values.extend(
            [
                f"${median:.{decimals}f}$",
                f"${lofo_means.min():.{decimals}f}$--"
                f"${lofo_means.max():.{decimals}f}$",
            ]
        )
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
    if args.decimals < 0:
        raise ValueError("--decimals cannot be negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        make_latex(load_rankability(args.input), args.decimals), encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
