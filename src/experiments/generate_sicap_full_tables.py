#!/usr/bin/env python3
"""Generate SI-CAP ECE/Kappa tables including the new ordinal baselines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_ORDER = (
    "CONCHV15",
    "GIGAPATH",
    "PHIKONV2",
    "UNI",
    "UNI2",
    "VIRCHOW",
    "VIRCHOW2",
)
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
    "SORD",
    "Beta",
    "POE",
    "CORE",
    "DRF",
    "CORF",
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
    "SORD": "SORD",
    "Beta": "Beta",
    "POE": "POE",
    "CORE": "CORE",
    "DRF": "DRF",
    "CORF": "CORF",
}
NEW_MODELS = {
    "sord": "SORD",
    "beta": "Beta",
    "poe": "POE",
    "core": "CORE",
    "drf": "DRF",
    "corf": "CORF",
}
METRICS = {
    "ece": ("test/ece", "min", "ECE $\\downarrow$"),
    "kappa": ("test/kappa", "max", "Kappa $\\uparrow$"),
}
RUN_LOG_PATTERN = re.compile(
    r"^sicap_(?P<feature>[^_]+)_(?P<model>sord|beta|poe|core|drf|corf)_seed(?P<seed>\d+)\.log$"
)
WANDB_LOG_PATTERN = re.compile(r"Find logs at: \./(?P<path>wandb/run-[^/]+)/logs")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-results",
        type=Path,
        default=root / "src/notebooks/complete_results/results_sicap.csv",
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
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def load_new_results(run_logs: Path, root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_path in sorted(run_logs.glob("sicap_*_seed*.log")):
        match = RUN_LOG_PATTERN.match(log_path.name)
        if match is None:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        wandb_matches = list(WANDB_LOG_PATTERN.finditer(text))
        if not wandb_matches:
            raise ValueError(f"Completed W&B path not found in {log_path}")
        run_dir = root / wandb_matches[-1].group("path")
        summary_path = run_dir / "files/wandb-summary.json"
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        rows.append(
            {
                "model": NEW_MODELS[match.group("model")],
                "fold": int(match.group("seed")),
                "features": match.group("feature"),
                "percentage": 100.0,
                "test/ece": float(summary["test/ece"]),
                "test/kappa": float(summary["test/kappa"]),
                "source": str(summary_path.relative_to(root)),
            }
        )
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> None:
    duplicate = frame.duplicated(["model", "features", "fold"], keep=False)
    if duplicate.any():
        raise ValueError(
            "Duplicate model/feature/fold rows:\n"
            + frame.loc[duplicate, ["model", "features", "fold"]].to_string(index=False)
        )
    counts = frame.groupby(["model", "features"]).size()
    expected_index = pd.MultiIndex.from_product(
        [MODEL_ORDER, FEATURE_ORDER], names=["model", "features"]
    )
    counts = counts.reindex(expected_index, fill_value=0)
    invalid = counts[counts != 5]
    if not invalid.empty:
        raise ValueError(f"Expected five runs per cell, found:\n{invalid}")
    for column, _, _ in METRICS.values():
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"Non-finite values found in {column}")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [MODEL_ORDER, FEATURE_ORDER], names=["model", "features"]
    )
    return (
        frame.groupby(["model", "features"], observed=True)
        .agg(
            ece_mean=("test/ece", "mean"),
            ece_std=("test/ece", "std"),
            kappa_mean=("test/kappa", "mean"),
            kappa_std=("test/kappa", "std"),
            n=("fold", "count"),
        )
        .reindex(index)
        .reset_index()
    )


def format_value(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def make_markdown(summary: pd.DataFrame, decimals: int) -> str:
    sections = [
        "# Full SI-CAP results\n",
        "Values are mean ± sample standard deviation across five seeds.\n",
    ]
    for metric, (_, direction, label) in METRICS.items():
        sections.append(f"## {label.replace('$', '')}\n")
        sections.append("| Model | " + " | ".join(FEATURE_ORDER) + " |")
        sections.append("|---|" + "---:|" * len(FEATURE_ORDER))
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        best = summary.groupby("features")[mean_column].agg(direction)
        for model in MODEL_ORDER:
            model_rows = summary[summary["model"] == model].set_index("features")
            cells = []
            for feature in FEATURE_ORDER:
                row = model_rows.loc[feature]
                value = format_value(row[mean_column], row[std_column], decimals)
                if np.isclose(row[mean_column], best.loc[feature]):
                    value = f"**{value}**"
                cells.append(value)
            sections.append(f"| {MODEL_NAMES[model]} | " + " | ".join(cells) + " |")
        sections.append("")
    sections.append("Bold indicates the best mean for each feature extractor.\n")
    return "\n".join(sections)


def make_latex(summary: pd.DataFrame, metric: str, decimals: int) -> str:
    _, direction, label = METRICS[metric]
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    best = summary.groupby("features")[mean_column].agg(direction)
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * len(FEATURE_ORDER) + "}",
        r"\toprule",
        "Model & " + " & ".join(FEATURE_ORDER) + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        model_rows = summary[summary["model"] == model].set_index("features")
        cells = []
        for feature in FEATURE_ORDER:
            row = model_rows.loc[feature]
            mean = f"{row[mean_column]:.{decimals}f}"
            std = f"{row[std_column]:.{decimals}f}"
            if np.isclose(row[mean_column], best.loc[feature]):
                mean = rf"\mathbf{{{mean}}}"
                std = rf"\mathbf{{{std}}}"
            cells.append(rf"${mean} \pm {std}$")
        lines.append(MODEL_NAMES[model] + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\par\smallskip",
            rf"\noindent\footnotesize Values are mean $\pm$ sample standard deviation over five seeds. Bold indicates the best mean {label} for each foundation model.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    base = pd.read_csv(args.base_results)
    base = base[np.isclose(base["percentage"], 100.0)].copy()
    base = base[base["model"].isin(MODEL_ORDER)]
    base["source"] = str(args.base_results.relative_to(root))
    new = load_new_results(args.run_logs, root)
    frame = pd.concat([base, new], ignore_index=True, sort=False)
    validate(frame)
    summary = summarize(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sicap_full_metrics.csv"
    markdown_path = args.output_dir / "sicap_full_results.md"
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(make_markdown(summary, args.decimals), encoding="utf-8")
    for metric in METRICS:
        path = args.output_dir / f"sicap_full_{metric}_table.tex"
        path.write_text(make_latex(summary, metric, args.decimals), encoding="utf-8")
        print(f"Wrote {path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
