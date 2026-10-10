"""Plot fold-averaged reliability diagrams for one feature extractor.

The script restores the exact local W&B runs represented in the paper's
100%-training results tables. For every selected model, it computes top-label
reliability in common equal-width bins for each fold, then plots the mean
confidence against mean accuracy. Shaded regions show one standard deviation
of bin accuracy across folds.

This is a standalone evaluation utility: it does not change the repository's
training or prediction code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from omegaconf import OmegaConf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.build_model import build_model
from experiments.load_dataset import load_dataset
from experiments.metrics_utils import ECE
from experiments.predict import predict_class_ordinal_manual


FEATURES = (
    "CONCHV15",
    "GIGAPATH",
    "PHIKONV2",
    "UNI",
    "UNI2",
    "VIRCHOW",
    "VIRCHOW2",
)
DATASETS = ("sicap", "crowdgleason")
FOLDS = tuple(range(5))
MODELS = (
    "LogisticRegression",
    "RLOrdinal",
    "KNN",
    "RandomForest",
    "MLP",
    "OrdinalMLP",
    "SVM",
    "SVGP_multiclass",
    "SORD",
    "Beta",
    "POE",
    "CORE",
    "DRF",
    "CORF",
    "OrdinalSVGP",
)
DEFAULT_MODELS = MODELS
MODEL_LABELS = {
    "LogisticRegression": "LR",
    "RLOrdinal": "OLR",
    "KNN": "KNN",
    "RandomForest": "RF",
    "MLP": "MLP",
    "OrdinalMLP": "OMLP",
    "SVM": "SVM",
    "SVGP_multiclass": "MGP",
    "SORD": "SORD",
    "Beta": "Beta",
    "POE": "POE",
    "CORE": "CORE",
    "DRF": "DRF",
    "CORF": "CORF",
    "OrdinalSVGP": "OGP",
}
PAPER_RESULTS = {
    "sicap": Path("src/notebooks/complete_results/results_sicap.csv"),
    "crowdgleason": Path(
        "src/notebooks/complete_results/results_crowdgleason.csv"
    ),
}
SKLEARN_MODELS = {
    "LogisticRegression",
    "RLOrdinal",
    "KNN",
    "RandomForest",
    "MLP",
    "SVM",
}
GP_MODELS = {"SVGP_multiclass", "OrdinalSVGP"}
MODERN_ORDINAL_MODELS = {"SORD", "Beta", "POE", "CORE", "DRF", "CORF"}
NEW_MODEL_LOG_NAMES = {
    "ordinal_mlp": "OrdinalMLP",
    "sord": "SORD",
    "beta": "Beta",
    "poe": "POE",
    "core": "CORE",
    "drf": "DRF",
    "corf": "CORF",
}
RUN_LOG_PATTERN = re.compile(
    r"^(?P<dataset>sicap|crowdgleason)_(?P<feature>[^_]+)_"
    r"(?P<model>ordinal_mlp|sord|beta|poe|core|drf|corf)_seed(?P<fold>\d+)\.log$"
)
WANDB_LOG_PATTERN = re.compile(r"Find logs at: \./(?P<path>wandb/run-[^/]+)/logs")


@dataclass(frozen=True)
class PaperResult:
    model: str
    fold: int
    kappa: float
    ece: float


@dataclass(frozen=True)
class SavedRun:
    run_id: str
    run_dir: Path
    model: str
    fold: int
    dataset_config: dict
    model_config: dict
    stored_kappa: float
    stored_ece: float

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "files" / "weights"


def _value(config: dict, key: str, default=None):
    section = config.get(key, {})
    return section.get("value", default) if isinstance(section, dict) else default


def _run_id(run_dir: Path) -> str:
    return run_dir.name.rsplit("-", 1)[-1]


def load_expected_results(
    dataset: str,
    feature: str,
    models: tuple[str, ...],
    folds: tuple[int, ...],
    percentage: int,
):
    frame = pd.read_csv(PAPER_RESULTS[dataset])
    selected = frame[
        (frame["features"] == feature)
        & (frame["percentage"] == percentage)
        & (frame["model"].isin(models))
        & (frame["fold"].isin(folds))
    ]
    expected = {}
    for _, row in selected.iterrows():
        result = PaperResult(
            model=str(row["model"]),
            fold=int(row["fold"]),
            kappa=float(row["test/kappa"]),
            ece=float(row["test/ece"]),
        )
        expected[(result.model, result.fold)] = result

    root = Path(__file__).resolve().parents[2]
    for log_path in sorted((root / "logs/ordinal_baselines").glob("*_seed*.log")):
        match = RUN_LOG_PATTERN.match(log_path.name)
        if match is None:
            continue
        model = NEW_MODEL_LOG_NAMES[match.group("model")]
        fold = int(match.group("fold"))
        if (
            match.group("dataset") != dataset
            or match.group("feature") != feature
            or model not in models
            or fold not in folds
        ):
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        run_matches = list(WANDB_LOG_PATTERN.finditer(log_text))
        if not run_matches:
            continue
        summary_path = root / run_matches[-1].group("path") / "files/wandb-summary.json"
        if not summary_path.is_file():
            continue
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("test/kappa") is None or summary.get("test/ece") is None:
            continue
        expected[(model, fold)] = PaperResult(
            model=model,
            fold=fold,
            kappa=float(summary["test/kappa"]),
            ece=float(summary["test/ece"]),
        )

    wanted = {(model, fold) for model in models for fold in folds}
    missing = sorted(wanted - set(expected))
    if missing:
        details = ", ".join(f"{model}/fold{fold}" for model, fold in missing)
        raise RuntimeError(f"The paper results CSV is missing: {details}")
    return expected


def _checkpoint_exists(model: str, files_dir: Path):
    weights_dir = files_dir / "weights"
    if model in SKLEARN_MODELS:
        return (weights_dir / "best.pkl").exists()
    if model == "OrdinalMLP":
        # Historical best.keras files omitted the separately attached ordinal
        # likelihood. Only complete TensorFlow checkpoints are admissible.
        return tf.train.latest_checkpoint(str(weights_dir)) is not None
    if model in GP_MODELS or model in MODERN_ORDINAL_MODELS:
        return tf.train.latest_checkpoint(str(weights_dir)) is not None
    return False


def discover_runs(
    wandb_dir: Path,
    dataset: str,
    feature: str,
    percentage: int,
    expected: dict[tuple[str, int], PaperResult],
    match_tolerance: float,
):
    """Select the newest checkpoint whose stored metrics match each paper row."""
    selected: dict[tuple[str, int], SavedRun] = {}
    candidate_counts = {key: 0 for key in expected}

    for config_path in sorted(wandb_dir.glob("run-*/files/config.yaml")):
        try:
            text = config_path.read_text(encoding="utf-8")
            if f"features: {feature}" not in text:
                continue
            stored = yaml.safe_load(text)
            dataset_config = _value(stored, "dataset", {})
            model_config = _value(stored, "model", {})
            model = model_config.get("name")
            if dataset_config.get("name") != dataset:
                continue
            if dataset_config.get("features") != feature:
                continue
            if int(dataset_config.get("percentage", 100)) != percentage:
                continue
            if model not in {key[0] for key in expected}:
                continue

            historical_fold = _value(stored, "fold")
            fold = historical_fold if historical_fold is not None else _value(stored, "seed")
            fold = int(fold)
            key = (model, fold)
            if key not in expected:
                continue

            files_dir = config_path.parent
            summary_path = files_dir / "wandb-summary.json"
            if not summary_path.exists() or not _checkpoint_exists(model, files_dir):
                continue
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            if summary.get("test/kappa") is None or summary.get("test/ece") is None:
                continue
            stored_kappa = float(summary["test/kappa"])
            stored_ece = float(summary["test/ece"])
            candidate_counts[key] += 1

            paper = expected[key]
            if abs(stored_kappa - paper.kappa) > match_tolerance:
                continue
            if abs(stored_ece - paper.ece) > match_tolerance:
                continue

            run_dir = config_path.parents[1]
            candidate = SavedRun(
                run_id=_run_id(run_dir),
                run_dir=run_dir,
                model=model,
                fold=fold,
                dataset_config=dataset_config,
                model_config=model_config,
                stored_kappa=stored_kappa,
                stored_ece=stored_ece,
            )
            if key not in selected or run_dir.name > selected[key].run_dir.name:
                selected[key] = candidate
        except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            continue

    missing = sorted(set(expected) - set(selected))
    if missing:
        details = [
            f"{model}/fold{fold} ({candidate_counts[(model, fold)]} candidates found)"
            for model, fold in missing
        ]
        raise RuntimeError(
            "Could not locate checkpoints matching these paper rows:\n- "
            + "\n- ".join(details)
        )
    return selected


def evaluation_config(saved_run: SavedRun):
    model_config = dict(saved_run.model_config)
    if saved_run.model in GP_MODELS:
        model_config["inducing_points"] = dict(model_config["inducing_points"])
        # The checkpoint replaces Z, so k-means is unnecessary at load time.
        model_config["inducing_points"]["init_method"] = "random"
    return OmegaConf.create(
        {
            "seed": saved_run.fold,
            "dataset": saved_run.dataset_config,
            "model": model_config,
        }
    )


def restore_gp_checkpoint(model, checkpoint_path: str):
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_nontrivial_match()
    names = {name for name, _ in tf.train.list_variables(checkpoint_path)}
    historical_z_name = (
        "model/inducing_variable/Z/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE"
    )
    if historical_z_name in names:
        model.inducing_variable.Z.assign(
            tf.train.load_variable(checkpoint_path, historical_z_name)
        )
        status.expect_partial()
    else:
        status.assert_existing_objects_matched()


def restore_modern_ordinal_checkpoint(model, checkpoint_path: str):
    """Restore TF 2.11 checkpoints whose object graph is unreadable by Keras 3."""
    checkpoint_names = {name for name, _ in tf.train.list_variables(checkpoint_path)}
    restored = []
    optimizer_fallback_index = 0

    def assign(variable, name):
        nonlocal optimizer_fallback_index
        candidates = [f"model/{name}/.ATTRIBUTES/VARIABLE_VALUE"]
        if name.endswith("/kernel"):
            candidates.append(
                f"model/{name[:-len('kernel')]}_kernel/.ATTRIBUTES/VARIABLE_VALUE"
            )
        key = next((candidate for candidate in candidates if candidate in checkpoint_names), None)
        if key is None:
            # Keras 3 may omit layers reached through a custom `embed` method
            # from the model object graph, while retaining their values in the
            # optimizer's ordered trainable-variable list.
            fallback = (
                f"optimizer/_trainable_variables/{optimizer_fallback_index}/"
                ".ATTRIBUTES/VARIABLE_VALUE"
            )
            optimizer_fallback_index += 1
            if fallback not in checkpoint_names:
                raise RuntimeError(
                    f"Missing {', '.join(candidates)} and {fallback} in {checkpoint_path}"
                )
            key = fallback
        value = tf.train.load_variable(checkpoint_path, key)
        if tuple(value.shape) != tuple(variable.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint {value.shape}, model {variable.shape}"
            )
        variable.assign(value)
        restored.append(key)

    def restore_backbone(backbone, checkpoint_prefix):
        for index, layer in enumerate(backbone.hidden_layers):
            assign(layer.kernel, f"{checkpoint_prefix}/hidden_layers/{index}/kernel")
            assign(layer.bias, f"{checkpoint_prefix}/hidden_layers/{index}/bias")
        if backbone.output_layer.built:
            assign(backbone.output_layer.kernel, f"{checkpoint_prefix}/output_layer/kernel")
            assign(backbone.output_layer.bias, f"{checkpoint_prefix}/output_layer/bias")

    if hasattr(model, "backbone"):
        restore_backbone(model.backbone, "backbone")
    if hasattr(model, "router"):
        restore_backbone(model.router, "router")
    for attribute in ("mean_layer", "log_variance_layer", "classifier"):
        if hasattr(model, attribute):
            layer = getattr(model, attribute)
            assign(layer.kernel, f"{attribute}/kernel")
            assign(layer.bias, f"{attribute}/bias")
    for attribute in (
        "leaf_means",
        "leaf_variances",
        "leaf_cumulative_probabilities",
    ):
        if hasattr(model, attribute):
            assign(getattr(model, attribute), attribute)

    if not restored:
        raise RuntimeError(f"No model tensors restored from {checkpoint_path}")


def restore_ordinal_mlp_checkpoint(model, checkpoint_path: str):
    """Restore a complete TF 2.8 OMLP checkpoint across TF/Keras versions."""
    checkpoint_names = {name for name, _ in tf.train.list_variables(checkpoint_path)}

    def assign(variable, name):
        key = f"{name}/.ATTRIBUTES/VARIABLE_VALUE"
        if key not in checkpoint_names:
            raise RuntimeError(f"Missing {key} in {checkpoint_path}")
        value = tf.train.load_variable(checkpoint_path, key)
        if tuple(value.shape) != tuple(variable.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint {value.shape}, model {variable.shape}"
            )
        variable.assign(value)

    for index, layer in enumerate(model.hidden_layers):
        assign(layer.kernel, f"model/hidden_layers/{index}/kernel")
        assign(layer.bias, f"model/hidden_layers/{index}/bias")
    assign(model.output_layer.kernel, "model/output_layer/kernel")
    assign(model.output_layer.bias, "model/output_layer/bias")
    assign(
        model.ordinal_likelihood._bin_increments.unconstrained_variable,
        "ordinal_likelihood/_bin_increments/_pretransformed_input",
    )
    assign(
        model.ordinal_likelihood.sigma.unconstrained_variable,
        "ordinal_likelihood/sigma/_pretransformed_input",
    )


def load_saved_model(saved_run: SavedRun, X_train, y_train):
    weights_dir = saved_run.checkpoint_dir
    if saved_run.model in SKLEARN_MODELS:
        return joblib.load(weights_dir / "best.pkl")

    cfg = evaluation_config(saved_run)
    model = build_model(cfg, X_train, y_train)
    if saved_run.model in GP_MODELS:
        restore_gp_checkpoint(model, tf.train.latest_checkpoint(str(weights_dir)))
    elif saved_run.model in MODERN_ORDINAL_MODELS:
        # Materialize lazily built Keras layers before matching checkpoint variables.
        model.predict_probabilities(
            tf.convert_to_tensor(X_train[:1], dtype=tf.float64)
        )
        checkpoint_path = tf.train.latest_checkpoint(str(weights_dir))
        restore_modern_ordinal_checkpoint(model, checkpoint_path)
    elif saved_run.model == "OrdinalMLP":
        # Build dense variables, then restore both the network and the
        # separately rooted ordinal-likelihood parameters.
        model(tf.convert_to_tensor(X_train[:1], dtype=tf.float64), training=False)
        restore_ordinal_mlp_checkpoint(
            model, tf.train.latest_checkpoint(str(weights_dir))
        )
    return model


def predict_probabilities(saved_run: SavedRun, model, X_test):
    """Return probabilities and hard predictions used in the paper metrics."""
    if saved_run.model == "OrdinalSVGP":
        f_mean, f_var = model.predict_f(X_test)
        historical_default = saved_run.dataset_config.get("name") != "crowdgleason"
        use_likelihood_variance = bool(
            # Historical runs predate this flag. SICAP used the learned scale,
            # whereas CrowdGleason set it to zero because its consensus test
            # labels followed a different annotation process from training.
            saved_run.model_config.get(
                "predict_likelihood_variance", historical_default
            )
        )
        sigma = (
            model.likelihood.noise_scale(
                tf.convert_to_tensor(X_test, dtype=tf.float64)
            ).numpy()
            if use_likelihood_variance
            else 0.0
        )
        predictions, probabilities = predict_class_ordinal_manual(
            f_mean[..., :1],
            f_var[..., :1],
            model.likelihood.bin_edges,
            sigma=sigma,
        )
    elif saved_run.model == "SVGP_multiclass":
        f_mean, _ = model.predict_f(X_test)
        probabilities = tf.nn.softmax(f_mean, axis=1).numpy()
        predictions = probabilities.argmax(axis=1)
    elif saved_run.model in MODERN_ORDINAL_MODELS:
        X_tensor = tf.convert_to_tensor(X_test, dtype=tf.float64)
        probabilities = model.predict_probabilities(X_tensor).numpy()
        if saved_run.model == "CORF":
            predictions = model.predict_labels(X_tensor).numpy()
        else:
            predictions = probabilities.argmax(axis=1)
    elif saved_run.model == "OrdinalMLP":
        probabilities = model.predict_probabilities(
            tf.convert_to_tensor(X_test, dtype=tf.float64)
        ).numpy()
        predictions = probabilities.argmax(axis=1)
    else:
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"{saved_run.model} does not provide predict_proba")
        probabilities = model.predict_proba(X_test)
        # In particular, mord's LogisticAT prediction is not necessarily the
        # argmax of predict_proba; retain its reported hard-decision rule for
        # checkpoint verification while reliability still uses top confidence.
        predictions = np.asarray(model.predict(X_test)).reshape(-1)

    probabilities = np.asarray(probabilities, dtype=float)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if probabilities.ndim != 2 or not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"Invalid probabilities from {saved_run.model}")
    if np.any(row_sums <= 0.0):
        raise RuntimeError(f"Non-positive probability sum from {saved_run.model}")
    if saved_run.model == "OrdinalMLP":
        # TrainableOrdinal clips each infinite-tail CDF to 0.001/0.999 for
        # stable gradients, so OMLP outputs sum to 0.998. Preserve those exact
        # training/evaluation-time confidences to reproduce its recorded ECE.
        return probabilities, predictions
    return probabilities / row_sums, predictions


def verify_metrics(
    saved_run: SavedRun,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    tolerance: float,
):
    # POE averages a finite number of latent samples at inference, so a restored
    # checkpoint is subject to small Monte Carlo variation from the logged pass.
    if saved_run.model == "POE":
        tolerance = max(tolerance, 1e-2)
    labels = np.asarray(labels).reshape(-1).astype(int)
    predictions = np.asarray(predictions).reshape(-1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        kappa = float(cohen_kappa_score(labels, predictions, weights="quadratic"))
        ece = float(ECE(bins=15).measure(probabilities, labels))
    if abs(kappa - saved_run.stored_kappa) > tolerance:
        raise RuntimeError(
            f"{saved_run.model}/fold{saved_run.fold}: restored kappa {kappa:.12f} "
            f"does not match stored {saved_run.stored_kappa:.12f}"
        )
    if abs(ece - saved_run.stored_ece) > tolerance:
        raise RuntimeError(
            f"{saved_run.model}/fold{saved_run.fold}: restored ECE {ece:.12f} "
            f"does not match stored {saved_run.stored_ece:.12f}"
        )
    return ece


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int):
    labels = np.asarray(labels).reshape(-1).astype(int)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.digitize(confidence, edges[1:-1]), n_bins - 1)

    mean_confidence = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for index in range(n_bins):
        selected = bin_index == index
        count[index] = int(selected.sum())
        if count[index]:
            mean_confidence[index] = float(confidence[selected].mean())
            accuracy[index] = float(correctness[selected].mean())
    return mean_confidence, accuracy, count


def evaluate_curves(
    runs: dict[tuple[str, int], SavedRun],
    models: tuple[str, ...],
    folds: tuple[int, ...],
    n_bins: int,
    verification_tolerance: float,
):
    curves = {model: [] for model in models}
    for model_name in models:
        for fold in folds:
            saved_run = runs[(model_name, fold)]
            print(f"Evaluating {MODEL_LABELS[model_name]}/fold{fold}", flush=True)
            random.seed(fold)
            np.random.seed(fold)
            tf.random.set_seed(fold)
            cfg = evaluation_config(saved_run)
            X_train, y_train, X_test, y_test = load_dataset(cfg.dataset, cfg.seed)
            model = load_saved_model(saved_run, X_train, y_train)
            # Make probabilistic prediction (notably POE) reproducible independently
            # of random draws used while constructing and restoring the model.
            tf.random.set_seed(fold)
            probabilities, predictions = predict_probabilities(
                saved_run, model, X_test
            )
            ece = verify_metrics(
                saved_run,
                y_test,
                probabilities,
                predictions,
                verification_tolerance,
            )
            confidence, accuracy, count = reliability_bins(
                y_test, probabilities, n_bins
            )
            curves[model_name].append(
                {
                    "fold": fold,
                    "confidence": confidence,
                    "accuracy": accuracy,
                    "count": count,
                    "ece": ece,
                }
            )
            tf.keras.backend.clear_session()
    return curves


def nanmean_no_warning(values: np.ndarray, axis: int):
    count = np.sum(np.isfinite(values), axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )


def nanstd_no_warning(values: np.ndarray, axis: int):
    mean = nanmean_no_warning(values, axis=axis)
    expanded_mean = np.expand_dims(mean, axis=axis)
    squared_error = np.where(np.isfinite(values), (values - expanded_mean) ** 2, 0.0)
    count = np.sum(np.isfinite(values), axis=axis)
    return np.sqrt(
        np.divide(
            squared_error.sum(axis=axis),
            count,
            out=np.full_like(mean, np.nan, dtype=float),
            where=count > 0,
        )
    )


def plot_overlay(curves, models, dataset, feature, n_bins, output, show):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    colors = plt.get_cmap("tab20")(np.arange(len(models)))
    line_styles = ("-", "--", "-.", ":")
    fold_count = len(next(iter(curves.values())))

    for model_index, (color, model_name) in enumerate(zip(colors, models)):
        fold_curves = curves[model_name]
        confidence = np.stack([curve["confidence"] for curve in fold_curves])
        accuracy = np.stack([curve["accuracy"] for curve in fold_curves])
        mean_confidence = nanmean_no_warning(confidence, axis=0)
        mean_accuracy = nanmean_no_warning(accuracy, axis=0)
        std_accuracy = nanstd_no_warning(accuracy, axis=0)
        valid = np.isfinite(mean_confidence) & np.isfinite(mean_accuracy)
        eces = np.asarray([curve["ece"] for curve in fold_curves])
        ece_std = eces.std(ddof=1) if len(eces) > 1 else 0.0
        label = f"{MODEL_LABELS[model_name]} (ECE {eces.mean():.3f}±{ece_std:.3f})"

        ax.plot(
            mean_confidence[valid],
            mean_accuracy[valid],
            marker="o",
            markersize=4,
            linewidth=1.8,
            linestyle=line_styles[model_index % len(line_styles)],
            color=color,
            label=label,
        )
        ax.fill_between(
            mean_confidence[valid],
            np.clip(mean_accuracy[valid] - std_accuracy[valid], 0.0, 1.0),
            np.clip(mean_accuracy[valid] + std_accuracy[valid], 0.0, 1.0),
            color=color,
            alpha=0.10,
            linewidth=0,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2, label="Perfect calibration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(
        f"Average reliability diagram — {dataset}, {feature}\n"
        f"{fold_count} {'fold' if fold_count == 1 else 'folds'}, "
        f"{n_bins} equal-width bins"
    )
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved reliability diagram to {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_grid(curves, models, dataset, feature, n_bins, output, show):
    """Plot one reliability curve per panel to keep many-model figures legible."""
    del dataset, feature, n_bins
    # Fifteen models fit naturally as three rows of five. Retain the historical
    # seven-column layout for smaller/model-subset figures.
    n_columns = 5 if len(models) == 15 else 7
    n_rows = math.ceil(len(models) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(15.5, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    colors = plt.get_cmap("tab20")(np.arange(len(models)))
    for ax, color, model_name in zip(axes, colors, models):
        fold_curves = curves[model_name]
        confidence = np.stack([curve["confidence"] for curve in fold_curves])
        accuracy = np.stack([curve["accuracy"] for curve in fold_curves])
        mean_confidence = nanmean_no_warning(confidence, axis=0)
        mean_accuracy = nanmean_no_warning(accuracy, axis=0)
        std_accuracy = nanstd_no_warning(accuracy, axis=0)
        valid = np.isfinite(mean_confidence) & np.isfinite(mean_accuracy)
        eces = np.asarray([curve["ece"] for curve in fold_curves])
        ece_std = eces.std(ddof=1) if len(eces) > 1 else 0.0

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="0.35",
            linewidth=1.0,
            zorder=1,
        )
        ax.fill_between(
            mean_confidence[valid],
            np.clip(mean_accuracy[valid] - std_accuracy[valid], 0.0, 1.0),
            np.clip(mean_accuracy[valid] + std_accuracy[valid], 0.0, 1.0),
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            mean_confidence[valid],
            mean_accuracy[valid],
            marker="o",
            markersize=3.2,
            linewidth=2.0 if model_name == "OrdinalSVGP" else 1.6,
            color=color,
            zorder=3,
        )
        ax.set_title(
            f"{MODEL_LABELS[model_name]}  ·  ECE {eces.mean():.3f} ± {ece_std:.3f}",
            fontsize=10,
            pad=4,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.tick_params(labelsize=8)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.supxlabel("Mean confidence", fontsize=11)
    fig.supylabel("Empirical accuracy", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved reliability diagram to {output}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--feature", choices=FEATURES, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(DEFAULT_MODELS),
        help=(
            "Models to include. Defaults to all models, including OMLP; OMLP "
            "requires a complete TensorFlow checkpoint from the current trainer."
        ),
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS)
    )
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--match-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--verification-tolerance",
        type=float,
        default=1e-3,
        help="Allowed absolute difference from stored ECE and kappa.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("grid", "overlay"),
        default="grid",
        help="Use readable per-model panels (default) or one combined axis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = tuple(args.models)
    folds = tuple(args.folds)
    expected = load_expected_results(
        args.dataset,
        args.feature,
        models,
        folds,
        args.percentage,
    )
    runs = discover_runs(
        args.wandb_dir,
        args.dataset,
        args.feature,
        args.percentage,
        expected,
        args.match_tolerance,
    )
    curves = evaluate_curves(
        runs,
        models,
        folds,
        args.bins,
        args.verification_tolerance,
    )
    output = args.output or Path(
        f"results/reliability_{args.dataset}_{args.feature}.pdf"
    )
    plot_function = plot_grid if args.layout == "grid" else plot_overlay
    plot_function(
        curves,
        models,
        args.dataset,
        args.feature,
        args.bins,
        output,
        args.show,
    )


if __name__ == "__main__":
    main()
