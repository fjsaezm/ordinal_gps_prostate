"""Evaluate saved CrowdGleason OrdinalSVGP runs with learned likelihood noise.

This script does not retrain. It discovers the completed local W&B runs that
produced the paper's CrowdGleason results, restores their checkpoints, and
recomputes test predictions with ``sigma_delta`` retained. The original
``test/*`` summary values are never overwritten.
"""

import argparse
import csv
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from omegaconf import OmegaConf
from sklearn.exceptions import UndefinedMetricWarning

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.build_model import build_model
from experiments.load_dataset import load_dataset
from experiments.metrics_utils import compute_metrics
from experiments.predict import predict_class_ordinal_manual


EXPECTED_FEATURES = (
    "CONCHV15",
    "GIGAPATH",
    "PHIKONV2",
    "UNI",
    "UNI2",
    "VIRCHOW",
    "VIRCHOW2",
)
EXPECTED_SEEDS = tuple(range(5))
WANDB_PREFIX = "test_with_likelihood_variance"
CORE_METRICS = ("acc", "kappa", "mae", "brier", "ece", "nll")


@dataclass(frozen=True)
class SavedRun:
    run_id: str
    run_dir: Path
    feature: str
    seed: int
    dataset_config: dict
    model_config: dict
    stored_kappa: float

    @property
    def checkpoint_dir(self):
        return self.run_dir / "files" / "weights"


def _value(config, key, default=None):
    section = config.get(key, {})
    return section.get("value", default) if isinstance(section, dict) else default


def _run_id(run_dir):
    return run_dir.name.rsplit("-", 1)[-1]


def discover_runs(wandb_dir):
    """Return the newest completed baseline run for each feature/seed pair."""
    selected = {}
    for config_path in sorted(wandb_dir.glob("run-*/files/config.yaml")):
        try:
            config_text = config_path.read_text(encoding="utf-8")
            # Avoid YAML-parsing thousands of unrelated local runs.
            if "        name: crowdgleason\n" not in config_text:
                continue
            if "        name: OrdinalSVGP\n" not in config_text:
                continue
            if "        percentage: 100\n" not in config_text:
                continue
            stored = yaml.safe_load(config_text)
            dataset = _value(stored, "dataset", {})
            model = _value(stored, "model", {})
            if dataset.get("name") != "crowdgleason":
                continue
            if model.get("name") != "OrdinalSVGP":
                continue
            if int(dataset.get("percentage", 100)) != 100:
                continue
            # The historical paper runs did not serialize this option. A run
            # explicitly marked True is not a source for this ablation.
            if model.get("predict_likelihood_variance") is True:
                continue

            summary_path = config_path.with_name("wandb-summary.json")
            checkpoint_dir = config_path.parent / "weights"
            checkpoint_path = tf.train.latest_checkpoint(str(checkpoint_dir))
            if not summary_path.exists() or checkpoint_path is None:
                continue
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            if summary.get("test/kappa") is None:
                continue

            historical_fold = _value(stored, "fold")
            seed = (
                historical_fold
                if historical_fold is not None
                else _value(stored, "seed")
            )
            feature = dataset.get("features")
            if feature not in EXPECTED_FEATURES or seed not in EXPECTED_SEEDS:
                continue

            run_dir = config_path.parents[1]
            candidate = SavedRun(
                run_id=_run_id(run_dir),
                run_dir=run_dir,
                feature=feature,
                seed=int(seed),
                dataset_config=dataset,
                model_config=model,
                stored_kappa=float(summary["test/kappa"]),
            )
            # Lexicographic run directory order is chronological for W&B's
            # run-YYYYMMDD_HHMMSS-id naming convention.
            key = (candidate.feature, candidate.seed)
            if key not in selected or candidate.run_dir.name > selected[key].run_dir.name:
                selected[key] = candidate
        except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            continue
    return selected


def validate_coverage(runs):
    expected = {(feature, seed) for feature in EXPECTED_FEATURES for seed in EXPECTED_SEEDS}
    missing = sorted(expected - set(runs))
    if missing:
        formatted = ", ".join(f"{feature}/seed{seed}" for feature, seed in missing)
        raise RuntimeError(f"Missing completed OrdinalSVGP runs: {formatted}")


def evaluation_config(saved_run):
    model_config = dict(saved_run.model_config)
    model_config["predict_likelihood_variance"] = True
    # The inducing values are restored in full. Random initialization avoids
    # rerunning k-means merely to construct a checkpoint-compatible model.
    model_config["inducing_points"] = dict(model_config["inducing_points"])
    model_config["inducing_points"]["init_method"] = "random"
    return OmegaConf.create(
        {
            "seed": saved_run.seed,
            "dataset": saved_run.dataset_config,
            "model": model_config,
        }
    )


def restore_historical_checkpoint(model, checkpoint_path):
    """Restore a checkpoint written by the older GPflow used for the paper.

    In that version, inducing points were a GPflow ``Parameter`` and were
    serialized below ``_pretransformed_input``. In the current environment
    they are a plain TensorFlow variable. TensorFlow restores the remaining
    model parameters by object path; the inducing matrix is assigned here.
    """
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_nontrivial_match()

    variable_names = {name for name, _ in tf.train.list_variables(checkpoint_path)}
    historical_z_name = (
        "model/inducing_variable/Z/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE"
    )
    if historical_z_name in variable_names:
        inducing_points = tf.train.load_variable(checkpoint_path, historical_z_name)
        model.inducing_variable.Z.assign(inducing_points)
        status.expect_partial()
    else:
        status.assert_existing_objects_matched()


def evaluate_run(saved_run, verification_tolerance):
    random.seed(saved_run.seed)
    np.random.seed(saved_run.seed)
    tf.random.set_seed(saved_run.seed)

    cfg = evaluation_config(saved_run)
    X_train, y_train, X_test, y_test = load_dataset(cfg.dataset, cfg.seed)
    model = build_model(cfg, X_train, y_train)
    checkpoint_path = tf.train.latest_checkpoint(str(saved_run.checkpoint_dir))
    restore_historical_checkpoint(model, checkpoint_path)

    f_mean, f_var = model.predict_f(X_test)
    bin_edges = model.likelihood.bin_edges
    sigma = float(np.asarray(model.likelihood.sigma.numpy()).reshape(()))

    # Reproduce the stored setting first. This prevents silently pairing a
    # checkpoint with the wrong feature extractor or historical fold.
    y_pred_without, probabilities_without = predict_class_ordinal_manual(
        f_mean, f_var, bin_edges, sigma=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics_without, _ = compute_metrics(
            y_test, y_pred_without, probabilities_without
        )
    kappa_error = abs(metrics_without["kappa"] - saved_run.stored_kappa)
    if kappa_error > verification_tolerance:
        raise RuntimeError(
            f"checkpoint verification failed: stored kappa={saved_run.stored_kappa:.12f}, "
            f"restored kappa={metrics_without['kappa']:.12f}, difference={kappa_error:.3g}"
        )

    y_pred_with, probabilities_with = predict_class_ordinal_manual(
        f_mean, f_var, bin_edges, sigma=sigma
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics_with, _ = compute_metrics(y_test, y_pred_with, probabilities_with)
    return {
        "run_id": saved_run.run_id,
        "feature": saved_run.feature,
        "seed": saved_run.seed,
        "checkpoint": checkpoint_path,
        "likelihood_sigma": sigma,
        "likelihood_variance": sigma ** 2,
        "stored_kappa_without_likelihood_variance": saved_run.stored_kappa,
        **{
            f"{metric}_without_likelihood_variance": metrics_without[metric]
            for metric in CORE_METRICS
        },
        **{metric: metrics_with[metric] for metric in CORE_METRICS},
    }


def update_wandb_summary(api, entity, project, result):
    run_path = (
        f"{entity}/{project}/{result['run_id']}"
        if entity
        else f"{project}/{result['run_id']}"
    )
    run = api.run(run_path)
    values = {
        f"{WANDB_PREFIX}/predict_likelihood_variance": True,
        f"{WANDB_PREFIX}/likelihood_sigma": result["likelihood_sigma"],
        f"{WANDB_PREFIX}/likelihood_variance": result["likelihood_variance"],
    }
    values.update(
        {f"{WANDB_PREFIX}/{metric}": result[metric] for metric in CORE_METRICS}
    )
    for key, value in values.items():
        run.summary[key] = value
    run.summary.update()


def write_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--project", default="OGP")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/crowdgleason_ordinal_svgp_likelihood_variance_on.csv"),
    )
    parser.add_argument(
        "--update-wandb",
        action="store_true",
        help="Write the new namespaced metrics to each existing W&B run summary.",
    )
    parser.add_argument(
        "--verification-tolerance",
        type=float,
        default=1e-10,
        help="Maximum allowed difference from the stored zero-variance kappa.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N runs (useful for a smoke test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs = discover_runs(args.wandb_dir)
    validate_coverage(runs)
    ordered_runs = [runs[key] for key in sorted(runs)]
    if args.limit is not None:
        ordered_runs = ordered_runs[: args.limit]

    api = None
    if args.update_wandb:
        import wandb

        api = wandb.Api()

    results = []
    failures = []
    for index, saved_run in enumerate(ordered_runs, start=1):
        label = f"{saved_run.feature}/seed{saved_run.seed} ({saved_run.run_id})"
        print(f"[{index}/{len(ordered_runs)}] Evaluating {label}", flush=True)
        try:
            result = evaluate_run(saved_run, args.verification_tolerance)
            results.append(result)
            if api is not None:
                update_wandb_summary(api, args.entity, args.project, result)
            print(
                f"  sigma={result['likelihood_sigma']:.6f}, "
                f"kappa={result['kappa']:.6f}, ece={result['ece']:.6f}",
                flush=True,
            )
        except Exception as error:  # Continue so one damaged run does not hide the rest.
            failures.append((label, str(error)))
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)

    if results:
        write_results(args.output, results)
        print(f"Saved {len(results)} evaluations to {args.output}")
    if failures:
        details = "\n".join(f"- {label}: {message}" for label, message in failures)
        raise RuntimeError(f"{len(failures)} evaluations failed:\n{details}")


if __name__ == "__main__":
    main()
