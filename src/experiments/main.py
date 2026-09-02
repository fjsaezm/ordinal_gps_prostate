import sys
import os

sys.path.append("src/")
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from load_dataset import load_dataset
from build_model import build_model
from metrics_utils import compute_metrics
from train import train
from predict import predict
import numpy as np
import tensorflow as tf
import random
from absl import logging

# Ignore ABSL warnings
logging.set_verbosity(logging.WARNING)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def predict_and_log(cfg, model, X, y, logger, set="test"):
    y_pred, class_probs = predict(cfg, model, X, y)
    metrics, cm = compute_metrics(y, y_pred, class_probs)
    metrics = {f"{set}/{k}": v for k, v in metrics.items()}
    logger.log(metrics)
    try:
        logger.log(
            {
                f"{set}/confusion_matrix": logger.Table(
                    data=cm.tolist(), columns=[f"Pred_{i}" for i in range(cm.shape[1])]
                )
            }
        )
    except Exception as e:
        print(f"Error logging confusion matrix: {e}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function that runs the experiment using Hydra configuration.

    Args:
        cfg: OmegaConf configuration object
    """
    # Print the configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    seed_everything(cfg.seed)
    print(f"\nUsing random seed {cfg.seed}")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Initialize wandb if enabled
    if cfg.wandb.get("enabled", True):
        # Create unique run name for Hydra multirun compatibility
        heteroscedastic_suffix = ""
        if bool(cfg.model.get("heteroscedastic", False)):
            predictor = str(cfg.model.get("heteroscedastic_predictor", "mlp"))
            heteroscedastic_suffix = f"_heteroscedastic_{predictor}"
        run_name = (
            f"{cfg.model.name}_{cfg.dataset.name}_{cfg.dataset.features}_seed{cfg.seed}"
            f"{heteroscedastic_suffix}"
        )

        # Initialize wandb with proper settings for multirun
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,  # Allow reinitializing for multirun
            settings=wandb.Settings(
                start_method="thread"
            ),  # Better for multiprocessing
        )
        # Set save_weights_path in config for downstream use
        cfg.output.save_weights_path = (
            wandb.run.dir + "/weights/" if cfg.output.get("save_weights", True) else None
        )
        if cfg.output.save_weights_path is not None:
            os.makedirs(cfg.output.save_weights_path, exist_ok=True)
        wandb.config.update({"save_weights_path": cfg.output.save_weights_path})
    else:
        # Create a local directory for weights if wandb is disabled
        likelihood_variant = ""
        if bool(cfg.model.get("heteroscedastic", False)):
            predictor = cfg.model.get("heteroscedastic_predictor", "mlp")
            likelihood_variant = f"heteroscedastic_{predictor}/"
        cfg.output.save_weights_path = (
            f"weights/{cfg.model.name}/{cfg.dataset.name}/"
            f"{cfg.dataset.features}/{likelihood_variant}seed_{cfg.seed}/"
            if cfg.output.get("save_weights", True)
            else None
        )
        if cfg.output.save_weights_path is not None:
            os.makedirs(cfg.output.save_weights_path, exist_ok=True)

    # Load data
    print(
        f"\nLoading dataset: {cfg.dataset.name}, features: {cfg.dataset.get('features', 'default')}, seed: {cfg.seed}"
    )
    X_train, y_train, X_test, y_test = load_dataset(cfg.dataset, cfg.seed)

    # Build model
    print(f"Building model: {cfg.model.name}")
    model = build_model(cfg, X_train, y_train)

    # Train
    print("Training...")
    logger = wandb if cfg.wandb.get("enabled", True) else None
    model = train(cfg, model, X_train, y_train, logger=logger)
    print(
        "Training finished."
        if cfg.output.save_weights_path is None
        else "Training finished and best model saved."
    )

    # Predict and log results
    if logger:
        predict_and_log(cfg, model, X_train, y_train, logger=logger, set="train")
        predict_and_log(cfg, model, X_test, y_test, logger=logger, set="test")
        # Properly finish the wandb run
        wandb.finish()
    else:
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("test", X_test, y_test),
        ]:
            y_pred, class_probs = predict(cfg, model, X_split, y_split)
            metrics, _ = compute_metrics(y_split, y_pred, class_probs)
            print(
                f"{split_name} metrics: "
                + ", ".join(f"{name}={value:.6f}" for name, value in metrics.items())
            )
        print("Wandb disabled - metrics printed locally")


if __name__ == "__main__":
    main()
