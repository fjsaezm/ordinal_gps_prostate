#!/usr/bin/env python3
"""Generate t-SNE visualizations of foundation features and GP predictions.

For each real dataset, the script selects the foundation-model representation
with the highest and lowest held-out rankability score. It restores the OGP
and MGP checkpoints for a requested fold and creates views of one t-SNE projection:

1. points colored by the ground-truth ordinal class; and
2. the same points colored by the OGP posterior latent mean or predicted class;
3. the same points colored by the multiclass GP (MGP) predicted class.

Eight standalone PDF panels, combined latent-function and class-prediction PDF
figures, per-dataset lowest-rankability prediction-error figures, LaTeX snippets,
and a CSV recording the selected runs and plotting settings are written to the
output directory. PNG copies can optionally be requested with ``--png``.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from experiments.load_dataset import load_dataset  # noqa: E402
from experiments import plot_average_reliability as checkpoint_utils  # noqa: E402


DATASETS = ("sicap", "crowdgleason")
DATASET_DISPLAY = {"sicap": "SICAPv2", "crowdgleason": "CrowdGleason"}
RANKABILITY_ROW = {"sicap": "SICAPv2", "crowdgleason": "CrowdGleason"}
OGP_MODEL = "OrdinalSVGP"
MGP_MODEL = "SVGP_multiclass"
ERROR_COLORS = ("#A8D8B9", "#F6C985", "#D62728")
ERROR_LABELS = ("Correct", "Adjacent\nclass", "Non-adjacent\nclass")


@dataclass
class Projection:
    dataset: str
    feature: str
    rank: str
    rankability: float
    coordinates: np.ndarray
    labels: np.ndarray
    latent_mean: np.ndarray
    ogp_predictions: np.ndarray
    mgp_predictions: np.ndarray
    ogp_run_id: str
    mgp_run_id: str
    n_train: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rankability-csv",
        type=Path,
        default=ROOT / "src/notebooks/complete_results/rankability.csv",
    )
    parser.add_argument("--wandb-dir", type=Path, default=ROOT / "wandb")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/ogp_tsne"
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help=(
            "Maximum stratified training samples per panel (default: 5000). "
            "Use 0 to include the complete training set."
        ),
    )
    parser.add_argument(
        "--latent-batch-size",
        type=int,
        default=2048,
        help="Batch size for OGP latent and OGP/MGP class predictions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--match-tolerance", type=float, default=1e-10)
    parser.add_argument("--png", action="store_true")
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rankability(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    required_rows = set(RANKABILITY_ROW.values())
    required_columns = set(checkpoint_utils.FEATURES)
    missing_rows = sorted(required_rows - set(frame.index))
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_rows or missing_columns:
        raise ValueError(
            f"Invalid rankability table: missing rows {missing_rows} and "
            f"columns {missing_columns}."
        )
    selected = frame.loc[
        list(required_rows), list(checkpoint_utils.FEATURES)
    ].to_numpy(dtype=float)
    if not np.isfinite(selected).all():
        raise ValueError("Rankability scores must be finite.")
    return frame


def select_extremes(frame: pd.DataFrame) -> dict[str, dict[str, tuple[str, float]]]:
    result = {}
    features = list(checkpoint_utils.FEATURES)
    for dataset in DATASETS:
        scores = frame.loc[RANKABILITY_ROW[dataset], features].astype(float)
        high = str(scores.idxmax())
        low = str(scores.idxmin())
        if high == low:
            raise ValueError(f"Rankability has no distinct extremes for {dataset}.")
        result[dataset] = {
            "high": (high, float(scores[high])),
            "low": (low, float(scores[low])),
        }
    return result


def stratified_indices(
    labels: np.ndarray, max_samples: int, seed: int
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    n_samples = labels.size
    if max_samples <= 0 or max_samples >= n_samples:
        return np.arange(n_samples)

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    target = counts / counts.sum() * max_samples
    allocation = np.floor(target).astype(int)
    allocation = np.maximum(allocation, 1)

    while allocation.sum() > max_samples:
        removable = np.flatnonzero(allocation > 1)
        index = removable[np.argmax(allocation[removable] - target[removable])]
        allocation[index] -= 1
    while allocation.sum() < max_samples:
        available = np.flatnonzero(allocation < counts)
        index = available[np.argmax(target[available] - allocation[available])]
        allocation[index] += 1

    indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == label), size=count, replace=False)
            for label, count in zip(classes, allocation)
        ]
    )
    rng.shuffle(indices)
    return indices


def predict_latent_mean(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    means = []
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        mean, _ = model.predict_f(features[start:stop], full_cov=False)
        means.append(np.asarray(mean).reshape(stop - start, -1)[:, 0])
    result = np.concatenate(means)
    if not np.isfinite(result).all():
        raise RuntimeError("OGP produced non-finite posterior latent means.")
    return result


def predict_classes(
    saved_run, model, features: np.ndarray, batch_size: int
) -> np.ndarray:
    predictions = []
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        _, batch_predictions = checkpoint_utils.predict_probabilities(
            saved_run, model, features[start:stop]
        )
        predictions.append(np.asarray(batch_predictions).reshape(-1))
    result = np.concatenate(predictions).astype(int)
    if result.shape != (len(features),):
        raise RuntimeError(
            f"{saved_run.model} returned invalid prediction shape {result.shape}."
        )
    return result


def compute_tsne(
    features: np.ndarray,
    pca_components: int,
    perplexity: float,
    iterations: int,
    seed: int,
) -> np.ndarray:
    if perplexity >= len(features):
        raise ValueError(
            f"t-SNE perplexity ({perplexity}) must be below the number of "
            f"plotted samples ({len(features)})."
        )
    components = min(pca_components, features.shape[1], len(features) - 1)
    reduced = PCA(
        n_components=components, svd_solver="randomized", random_state=seed
    ).fit_transform(features)
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        n_iter=iterations,
        init="pca",
        method="barnes_hut",
        random_state=seed,
    ).fit_transform(reduced)
    if not np.isfinite(coordinates).all():
        raise RuntimeError("t-SNE produced non-finite coordinates.")
    return coordinates


def load_projection(
    dataset: str,
    feature: str,
    rank: str,
    rankability: float,
    args: argparse.Namespace,
) -> Projection:
    expected = checkpoint_utils.load_expected_results(
        dataset,
        feature,
        (OGP_MODEL, MGP_MODEL),
        (args.fold,),
        args.percentage,
    )
    runs = checkpoint_utils.discover_runs(
        args.wandb_dir,
        dataset,
        feature,
        args.percentage,
        expected,
        args.match_tolerance,
    )
    ogp_run = runs[(OGP_MODEL, args.fold)]
    mgp_run = runs[(MGP_MODEL, args.fold)]
    cfg = checkpoint_utils.evaluation_config(ogp_run)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    X_train, y_train, _, _ = load_dataset(cfg.dataset, cfg.seed)
    ogp_model = checkpoint_utils.load_saved_model(ogp_run, X_train, y_train)

    y_train = np.asarray(y_train).reshape(-1).astype(int)
    indices = stratified_indices(y_train, args.max_samples, args.seed)
    plotted_features = X_train[indices]
    labels = y_train[indices]
    latent_mean = predict_latent_mean(
        ogp_model, plotted_features, args.latent_batch_size
    )
    ogp_predictions = predict_classes(
        ogp_run, ogp_model, plotted_features, args.latent_batch_size
    )
    del ogp_model
    tf.keras.backend.clear_session()

    mgp_model = checkpoint_utils.load_saved_model(mgp_run, X_train, y_train)
    mgp_predictions = predict_classes(
        mgp_run, mgp_model, plotted_features, args.latent_batch_size
    )
    del mgp_model
    coordinates = compute_tsne(
        plotted_features,
        args.pca_components,
        args.perplexity,
        args.iterations,
        args.seed,
    )
    tf.keras.backend.clear_session()
    return Projection(
        dataset=dataset,
        feature=feature,
        rank=rank,
        rankability=rankability,
        coordinates=coordinates,
        labels=labels,
        latent_mean=latent_mean,
        ogp_predictions=ogp_predictions,
        mgp_predictions=mgp_predictions,
        ogp_run_id=ogp_run.run_id,
        mgp_run_id=mgp_run.run_id,
        n_train=len(X_train),
    )


def point_size(n_points: int) -> float:
    return float(np.clip(16000.0 / n_points, 1.2, 5.0))


def draw_panel(
    ax: plt.Axes,
    projection: Projection,
    view: str,
    latent_norm: Normalize,
    standalone: bool,
):
    coordinates = projection.coordinates
    if view in {"labels", "ogp_predictions", "mgp_predictions"}:
        classes = np.unique(projection.labels)
        norm = BoundaryNorm(
            np.arange(classes.min() - 0.5, classes.max() + 1.5),
            plt.get_cmap("viridis").N,
        )
        values = {
            "labels": projection.labels,
            "ogp_predictions": projection.ogp_predictions,
            "mgp_predictions": projection.mgp_predictions,
        }[view]
    elif view == "latent":
        classes = None
        norm = latent_norm
        values = projection.latent_mean
    else:
        raise ValueError(f"Unknown panel view: {view}")

    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        cmap="viridis",
        norm=norm,
        s=point_size(len(values)),
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    if standalone:
        descriptor = "Highest" if projection.rank == "high" else "Lowest"
        view_title = {
            "labels": "Ground-truth labels",
            "latent": "OGP latent mean",
            "ogp_predictions": "OGP predicted classes",
            "mgp_predictions": "MGP predicted classes",
        }[view]
        ax.set_title(
            f"{DATASET_DISPLAY[projection.dataset]} — {projection.feature}\n"
            f"{descriptor} rankability ($\\rho_{{\\mathrm{{rank}}}}$="
            f"{projection.rankability:.3f}) — {view_title}",
            fontsize=11,
        )
        colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        if classes is not None:
            colorbar.set_ticks(classes)
            colorbar.set_label(
                "True ordinal label" if view == "labels" else "Predicted class"
            )
        else:
            colorbar.set_label(r"OGP posterior mean $\mathbb{E}[f(\mathbf{x})]$")
    return scatter, classes


def panel_stem(projection: Projection, view: str) -> str:
    return (
        f"tsne_{projection.dataset}_{projection.rank}_"
        f"{projection.feature}_{view}"
    )


def save_standalone_panels(
    projections: list[Projection],
    latent_norms: dict[str, Normalize],
    args: argparse.Namespace,
) -> None:
    for projection in projections:
        for view in ("labels", "latent"):
            fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
            draw_panel(
                ax,
                projection,
                view,
                latent_norms[projection.dataset],
                standalone=True,
            )
            stem = panel_stem(projection, view)
            fig.savefig(args.output_dir / f"{stem}.pdf", dpi=args.dpi)
            if args.png:
                fig.savefig(args.output_dir / f"{stem}.png", dpi=args.dpi)
            plt.close(fig)


def save_combined_figure(
    projections: list[Projection],
    latent_norms: dict[str, Normalize],
    args: argparse.Namespace,
) -> Path:
    lookup = {(p.dataset, p.rank): p for p in projections}
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.8, 7.6),
        constrained_layout=True,
    )
    column_specs = (
        ("high", "labels"),
        ("high", "latent"),
        ("low", "labels"),
        ("low", "latent"),
    )
    column_titles = (
        "Highest rankability\nTrue labels",
        "Highest rankability\nOGP latent mean",
        "Lowest rankability\nTrue labels",
        "Lowest rankability\nOGP latent mean",
    )
    letters = iter("abcdefgh")
    for row, dataset in enumerate(DATASETS):
        label_mappable = None
        latent_mappable = None
        classes = None
        for column, ((rank, view), title) in enumerate(
            zip(column_specs, column_titles)
        ):
            projection = lookup[(dataset, rank)]
            mappable, panel_classes = draw_panel(
                axes[row, column],
                projection,
                view,
                latent_norms[dataset],
                standalone=False,
            )
            if view == "labels":
                label_mappable = mappable
                classes = panel_classes
            else:
                latent_mappable = mappable
            if row == 0:
                axes[row, column].set_title(title, fontsize=11, pad=7)
            axes[row, column].text(
                0.02,
                0.98,
                f"({next(letters)})",
                transform=axes[row, column].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
            axes[row, column].text(
                0.98,
                0.02,
                f"{projection.feature}\n$\\rho_{{\\mathrm{{rank}}}}$="
                f"{projection.rankability:.3f}",
                transform=axes[row, column].transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )

        axes[row, 0].annotate(
            DATASET_DISPLAY[dataset],
            xy=(-0.23, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        label_colorbar = fig.colorbar(
            label_mappable,
            ax=[axes[row, 0], axes[row, 2]],
            location="bottom",
            fraction=0.045,
            pad=0.025,
            aspect=35,
        )
        label_colorbar.set_ticks(classes)
        label_colorbar.set_label("True ordinal label", fontsize=9)
        latent_colorbar = fig.colorbar(
            latent_mappable,
            ax=[axes[row, 1], axes[row, 3]],
            location="bottom",
            fraction=0.045,
            pad=0.025,
            aspect=35,
        )
        latent_colorbar.set_label(
            r"OGP posterior mean $\mathbb{E}[f(\mathbf{x})]$", fontsize=9
        )

    output = args.output_dir / "ogp_tsne_guidance.pdf"
    fig.savefig(output, dpi=args.dpi)
    if args.png:
        fig.savefig(output.with_suffix(".png"), dpi=args.dpi)
    plt.close(fig)
    return output


def save_prediction_figure(
    projections: list[Projection],
    latent_norms: dict[str, Normalize],
    args: argparse.Namespace,
) -> Path:
    """Compare true, OGP, and MGP classes on identical t-SNE coordinates."""
    lookup = {(p.dataset, p.rank): p for p in projections}
    fig, axes = plt.subplots(
        2,
        6,
        figsize=(20.5, 7.5),
        constrained_layout=True,
    )
    column_specs = tuple(
        (rank, view)
        for rank in ("high", "low")
        for view in ("labels", "ogp_predictions", "mgp_predictions")
    )
    column_titles = tuple(
        f"{rank_title} rankability\n{view_title}"
        for rank_title in ("Highest", "Lowest")
        for view_title in ("True labels", "OGP predictions", "MGP predictions")
    )
    letters = iter("abcdefghijkl")
    for row, dataset in enumerate(DATASETS):
        mappable = None
        classes = None
        for column, ((rank, view), title) in enumerate(
            zip(column_specs, column_titles)
        ):
            projection = lookup[(dataset, rank)]
            mappable, panel_classes = draw_panel(
                axes[row, column],
                projection,
                view,
                latent_norms[dataset],
                standalone=False,
            )
            classes = panel_classes
            if row == 0:
                axes[row, column].set_title(title, fontsize=10, pad=7)
            axes[row, column].text(
                0.02,
                0.98,
                f"({next(letters)})",
                transform=axes[row, column].transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
            axes[row, column].text(
                0.98,
                0.02,
                f"{projection.feature}\n$\\rho_{{\\mathrm{{rank}}}}$="
                f"{projection.rankability:.3f}",
                transform=axes[row, column].transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )

        axes[row, 0].annotate(
            DATASET_DISPLAY[dataset],
            xy=(-0.25, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        colorbar = fig.colorbar(
            mappable,
            ax=list(axes[row, :]),
            location="bottom",
            fraction=0.045,
            pad=0.025,
            aspect=55,
        )
        colorbar.set_ticks(classes)
        colorbar.set_label("Ordinal class", fontsize=9)

    output = args.output_dir / "ogp_mgp_tsne_predictions.pdf"
    fig.savefig(output, dpi=args.dpi)
    if args.png:
        fig.savefig(output.with_suffix(".png"), dpi=args.dpi)
    plt.close(fig)
    return output


def ordinal_error_categories(
    labels: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    """Encode exact, adjacent-class, and non-adjacent-class predictions."""
    distances = np.abs(
        np.asarray(predictions, dtype=int) - np.asarray(labels, dtype=int)
    )
    return np.where(distances == 0, 0, np.where(distances == 1, 1, 2))


def save_rankability_error_figures(
    projections: list[Projection],
    latent_norms: dict[str, Normalize],
    args: argparse.Namespace,
    rank: str,
) -> dict[str, Path]:
    """Save one clean prediction/error panel per dataset for one rank extreme."""
    if rank not in {"high", "low"}:
        raise ValueError(f"Unknown rankability extreme: {rank}")
    lookup = {(p.dataset, p.rank): p for p in projections}
    error_cmap = ListedColormap(ERROR_COLORS)
    error_norm = BoundaryNorm(np.arange(-0.5, 3.5), error_cmap.N)
    column_titles = (
        "True labels",
        "OGP predictions",
        "MGP predictions",
        "OGP error map",
        "MGP error map",
    )
    outputs = {}
    for dataset in DATASETS:
        projection = lookup[(dataset, rank)]
        fig, axes = plt.subplots(
            1,
            5,
            figsize=(18.0, 3.8),
            constrained_layout=True,
        )
        class_mappable = None
        classes = None
        for column, view in enumerate(
            ("labels", "ogp_predictions", "mgp_predictions")
        ):
            class_mappable, classes = draw_panel(
                axes[column],
                projection,
                view,
                latent_norms[dataset],
                standalone=False,
            )

        error_mappable = None
        for column, predictions in enumerate(
            (projection.ogp_predictions, projection.mgp_predictions), start=3
        ):
            values = ordinal_error_categories(projection.labels, predictions)
            error_mappable = axes[column].scatter(
                projection.coordinates[:, 0],
                projection.coordinates[:, 1],
                c=values,
                cmap=error_cmap,
                norm=error_norm,
                s=point_size(len(values)),
                alpha=0.88,
                linewidths=0,
                rasterized=True,
            )
            axes[column].set_aspect("equal", adjustable="box")

        for column, title in enumerate(column_titles):
            axes[column].set_title(title, fontsize=16, pad=9)
            axes[column].set_axis_off()

        class_colorbar = fig.colorbar(
            class_mappable,
            ax=list(axes[:3]),
            location="bottom",
            fraction=0.05,
            pad=0.025,
            aspect=40,
        )
        class_colorbar.set_ticks(classes)
        class_colorbar.ax.tick_params(labelsize=13)
        class_colorbar.set_label("Ordinal class", fontsize=14)
        error_colorbar = fig.colorbar(
            error_mappable,
            ax=list(axes[3:]),
            location="bottom",
            fraction=0.05,
            pad=0.025,
            aspect=28,
        )
        error_colorbar.set_ticks(range(3))
        error_colorbar.set_ticklabels(ERROR_LABELS)
        error_colorbar.ax.tick_params(labelsize=13)
        error_colorbar.set_label("Prediction error severity", fontsize=14)

        output = (
            args.output_dir
            / f"ogp_mgp_tsne_{rank}_rankability_errors_{dataset}.pdf"
        )
        fig.savefig(output, dpi=args.dpi)
        if args.png:
            fig.savefig(output.with_suffix(".png"), dpi=args.dpi)
        plt.close(fig)
        outputs[dataset] = output
    return outputs


def write_latex(
    projections: list[Projection], output_figure: Path, args: argparse.Namespace
) -> Path:
    lookup = {(p.dataset, p.rank): p for p in projections}
    try:
        graphics_path = output_figure.relative_to(ROOT).as_posix()
    except ValueError:
        graphics_path = output_figure.as_posix()
    sample_text = (
        "the complete training set"
        if args.max_samples <= 0
        else f"a class-stratified sample of up to {args.max_samples:,} training instances"
    )
    sicap_high = lookup[("sicap", "high")]
    sicap_low = lookup[("sicap", "low")]
    crowd_high = lookup[("crowdgleason", "high")]
    crowd_low = lookup[("crowdgleason", "low")]
    caption = (
        "t-SNE visualization of foundation-model representations and the OGP "
        "ordinal mapping on "
        f"{sample_text}. For each dataset, we show the representations with the "
        "highest and lowest held-out linear rankability: "
        f"{sicap_high.feature} ({sicap_high.rankability:.3f}) and "
        f"{sicap_low.feature} ({sicap_low.rankability:.3f}) for SICAPv2, and "
        f"{crowd_high.feature} ({crowd_high.rankability:.3f}) and "
        f"{crowd_low.feature} ({crowd_low.rankability:.3f}) for CrowdGleason. "
        "Within each pair, the same t-SNE coordinates are colored either by the "
        "ground-truth ordinal label or by the OGP posterior latent mean. OGP uses "
        "fixed foundation-model features; consequently, the latter panels show "
        "the ordinal function learned over each representation rather than a new "
        "OGP embedding."
    )
    lines = [
        r"\begin{figure*}[t]",
        r"    \centering",
        rf"    \includegraphics[width=\textwidth]{{{graphics_path}}}",
        rf"    \caption{{{caption}}}",
        r"    \label{fig:ogp-tsne-guidance}",
        r"\end{figure*}",
    ]
    output = args.output_dir / "ogp_tsne_guidance.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_prediction_latex(
    projections: list[Projection], output_figure: Path, args: argparse.Namespace
) -> Path:
    lookup = {(p.dataset, p.rank): p for p in projections}
    try:
        graphics_path = output_figure.relative_to(ROOT).as_posix()
    except ValueError:
        graphics_path = output_figure.as_posix()
    sample_text = (
        "the complete training set"
        if args.max_samples <= 0
        else f"a class-stratified sample of up to {args.max_samples:,} training instances"
    )
    sicap_high = lookup[("sicap", "high")]
    sicap_low = lookup[("sicap", "low")]
    crowd_high = lookup[("crowdgleason", "high")]
    crowd_low = lookup[("crowdgleason", "low")]
    caption = (
        "t-SNE visualization of ground-truth and GP-predicted classes on "
        f"{sample_text}. For each dataset, we show the representations with the "
        "highest and lowest held-out linear rankability: "
        f"{sicap_high.feature} ({sicap_high.rankability:.3f}) and "
        f"{sicap_low.feature} ({sicap_low.rankability:.3f}) for SICAPv2, and "
        f"{crowd_high.feature} ({crowd_high.rankability:.3f}) and "
        f"{crowd_low.feature} ({crowd_low.rankability:.3f}) for CrowdGleason. "
        "Within each group, identical t-SNE coordinates and class colors are "
        "used for the ground truth, ordinal Gaussian process (OGP) predictions, "
        "and multiclass Gaussian process (MGP) predictions."
    )
    lines = [
        r"\begin{figure*}[t]",
        r"    \centering",
        rf"    \includegraphics[width=\textwidth]{{{graphics_path}}}",
        rf"    \caption{{{caption}}}",
        r"    \label{fig:ogp-mgp-tsne-predictions}",
        r"\end{figure*}",
    ]
    output = args.output_dir / "ogp_mgp_tsne_predictions.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_rankability_error_latex(
    projections: list[Projection],
    output_figures: dict[str, Path],
    args: argparse.Namespace,
    rank: str,
) -> Path:
    if rank not in {"high", "low"}:
        raise ValueError(f"Unknown rankability extreme: {rank}")
    lookup = {(p.dataset, p.rank): p for p in projections}
    graphics_paths = {}
    for dataset, output_figure in output_figures.items():
        try:
            graphics_paths[dataset] = output_figure.relative_to(ROOT).as_posix()
        except ValueError:
            graphics_paths[dataset] = output_figure.as_posix()
    sample_text = (
        "the complete training set"
        if args.max_samples <= 0
        else f"a class-stratified sample of up to {args.max_samples:,} training instances"
    )
    sicap = lookup[("sicap", rank)]
    crowd = lookup[("crowdgleason", rank)]
    rank_description = "highest" if rank == "high" else "lowest"
    caption = (
        "t-SNE visualization of GP predictions and ordinal error severity for "
        f"the {rank_description}-rankability representation in each dataset on "
        f"{sample_text}: {sicap.feature} ({sicap.rankability:.3f}) for SICAPv2 "
        f"and {crowd.feature} ({crowd.rankability:.3f}) for CrowdGleason. "
        "From left to right, panels show the ground-truth classes, OGP "
        "predictions, MGP predictions, OGP errors, and MGP errors. Green marks "
        "correct predictions, orange marks predictions one adjacent ordinal "
        "class away, and red marks errors spanning more than one class."
    )
    lines = [
        r"% Requires \usepackage{graphicx} in the document preamble.",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \begin{tabular}{@{}c@{\hspace{0.6em}}c@{}}",
        rf"        \raisebox{{-0.5\height}}{{\rotatebox[origin=c]{{90}}{{\shortstack{{\textbf{{{DATASET_DISPLAY['sicap']}}}\\{sicap.feature}\\$\rho_{{\mathrm{{rank}}}}={sicap.rankability:.3f}$}}}}}} &",
        rf"        \raisebox{{-0.5\height}}{{\includegraphics[width=0.93\textwidth]{{{graphics_paths['sicap']}}}}} \\[0.8em]",
        rf"        \raisebox{{-0.5\height}}{{\rotatebox[origin=c]{{90}}{{\shortstack{{\textbf{{{DATASET_DISPLAY['crowdgleason']}}}\\{crowd.feature}\\$\rho_{{\mathrm{{rank}}}}={crowd.rankability:.3f}$}}}}}} &",
        rf"        \raisebox{{-0.5\height}}{{\includegraphics[width=0.93\textwidth]{{{graphics_paths['crowdgleason']}}}}} \\",
        r"    \end{tabular}",
        rf"    \caption{{{caption}}}",
        rf"    \label{{fig:ogp-mgp-tsne-{rank}-rankability-errors}}",
        r"\end{figure*}",
    ]
    output = args.output_dir / f"ogp_mgp_tsne_{rank}_rankability_errors.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_metadata(projections: list[Projection], args: argparse.Namespace) -> Path:
    rows = []
    for projection in projections:
        rows.append(
            {
                "dataset": projection.dataset,
                "dataset_display": DATASET_DISPLAY[projection.dataset],
                "rank": projection.rank,
                "feature": projection.feature,
                "rankability": projection.rankability,
                "fold": args.fold,
                # Keep the historical column as an alias for downstream users.
                "run_id": projection.ogp_run_id,
                "ogp_run_id": projection.ogp_run_id,
                "mgp_run_id": projection.mgp_run_id,
                "n_train": projection.n_train,
                "n_plotted": len(projection.labels),
                "tsne_seed": args.seed,
                "perplexity": args.perplexity,
                "iterations": args.iterations,
                "pca_components": min(
                    args.pca_components, len(projection.labels) - 1
                ),
            }
        )
    output = args.output_dir / "ogp_tsne_metadata.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    args = parse_args()
    if args.iterations < 250:
        raise ValueError("--iterations must be at least 250.")
    if args.pca_components < 2:
        raise ValueError("--pca-components must be at least 2.")
    if args.latent_batch_size < 1:
        raise ValueError("--latent-batch-size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    # The checkpoint helper stores its paper-result paths relative to the
    # repository. Make them absolute so this script is safe to call elsewhere.
    checkpoint_utils.PAPER_RESULTS = {
        dataset: ROOT / path
        for dataset, path in checkpoint_utils.PAPER_RESULTS.items()
    }

    rankability = load_rankability(args.rankability_csv)
    extremes = select_extremes(rankability)
    projections = []
    for dataset in DATASETS:
        for rank in ("high", "low"):
            feature, score = extremes[dataset][rank]
            print(
                f"Processing {DATASET_DISPLAY[dataset]}/{feature} "
                f"({rank}, rankability={score:.3f})",
                flush=True,
            )
            projections.append(
                load_projection(dataset, feature, rank, score, args)
            )

    latent_norms = {}
    for dataset in DATASETS:
        values = np.concatenate(
            [p.latent_mean for p in projections if p.dataset == dataset]
        )
        lower, upper = np.quantile(values, [0.01, 0.99])
        if np.isclose(lower, upper):
            lower, upper = float(values.min()), float(values.max())
        latent_norms[dataset] = Normalize(vmin=lower, vmax=upper, clip=True)

    save_standalone_panels(projections, latent_norms, args)
    combined = save_combined_figure(projections, latent_norms, args)
    prediction_figure = save_prediction_figure(projections, latent_norms, args)
    low_error_figures = save_rankability_error_figures(
        projections, latent_norms, args, "low"
    )
    high_error_figures = save_rankability_error_figures(
        projections, latent_norms, args, "high"
    )
    latex = write_latex(projections, combined, args)
    prediction_latex = write_prediction_latex(
        projections, prediction_figure, args
    )
    low_error_latex = write_rankability_error_latex(
        projections, low_error_figures, args, "low"
    )
    high_error_latex = write_rankability_error_latex(
        projections, high_error_figures, args, "high"
    )
    metadata = write_metadata(projections, args)
    print(f"Saved 8 standalone panels to {args.output_dir}")
    print(f"Saved combined figure to {combined}")
    print(f"Saved prediction comparison to {prediction_figure}")
    for rank, outputs in (
        ("low", low_error_figures),
        ("high", high_error_figures),
    ):
        for dataset, output in outputs.items():
            print(
                f"Saved {DATASET_DISPLAY[dataset]} {rank}-rankability error "
                f"comparison to {output}"
            )
    print(f"Saved LaTeX snippet to {latex}")
    print(f"Saved prediction LaTeX snippet to {prediction_latex}")
    print(f"Saved low-rank error-map LaTeX snippet to {low_error_latex}")
    print(f"Saved high-rank error-map LaTeX snippet to {high_error_latex}")
    print(f"Saved plotting metadata to {metadata}")


if __name__ == "__main__":
    main()
