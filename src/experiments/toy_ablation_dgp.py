"""
Toy Ablation Study for Ordinal SVGP Models

This script runs a grid search over different model configurations,
training each variant and saving:
1. Training loss plots (ELL, KL, Total Loss)
2. Latent function posterior plots
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from itertools import product

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from build_model import build_model
from train import train
from predict import predict_class_ordinal_manual

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")


def generate_data(num_data, seed=42):
    X = np.linspace(0, 1, num_data).reshape(-1, 1)
    centers = np.array([0.15, 0.35, 0.5, 0.65, 0.8, 0.9])
    class_labels = np.array([0, 3, 1, 2, 3, 0])
    spreads = np.array([0.07, 0.05, 0.05, 0.05, 0.05, 0.07])

    # Compute "affinity" for each class across X
    num_classes = 4
    logits = np.zeros((num_data, num_classes))
    for c_idx, c in enumerate(class_labels):
        logits[:, c] += np.exp(
            -0.5 * ((X[:, 0] - centers[c_idx]) / spreads[c_idx]) ** 2
        )

    # Normalize into probabilities and sample classes
    probs = logits / np.sum(logits, axis=1, keepdims=True)
    Y = np.array([np.random.choice(num_classes, p=p) for p in probs])
    return X, Y.astype(float), probs


class SimpleLogger:
    """Simple logger to track and plot training losses"""

    def __init__(self):
        self.losses = {}

    def log(self, losses, step=None):
        """Log training metrics"""
        for key, val in losses.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(val)

    def plot_losses(self, figsize=(15, 5), save_path=None):
        """Plot the training losses"""
        n_plots = len(self.losses.keys())

        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        for plot_idx, (key, values) in enumerate(self.losses.items()):
            axes[plot_idx].plot(range(1, len(values) + 1), values, "b-", linewidth=2)
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel(key)
            axes[plot_idx].set_title(f"Training {key}")
            axes[plot_idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved loss plot to {save_path}")

        plt.close()


def plot_latent_function_posterior(model, X_train, y_train, X_full, save_path=None):
    """Plot the latent function posterior with uncertainty"""

    # Generate test points for plotting
    X_plot = np.linspace(X_full.min() - 0.05, X_full.max() + 0.05, 200).reshape(-1, 1)

    # Get predictions
    f_mean, f_var = model.predict_f(X_plot)
    f_mean_np = f_mean.numpy()
    f_var_np = f_var.numpy()
    f_std = np.sqrt(f_var_np.flatten())

    # Get class probabilities
    bin_edges = model.likelihood.bin_edges
    plot_predictions, class_probs = predict_class_ordinal_manual(
        f_mean_np, f_var_np, bin_edges
    )

    train_preds, _ = predict_class_ordinal_manual(*model.predict_f(X_train), bin_edges)
    accuracy = np.mean(train_preds.flatten() == y_train.flatten())

    n_classes = class_probs.shape[1]
    colors = [
        "red",
        "orange",
        "green",
        "blue",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Create plot - Latent function posterior with uncertainty
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        X_plot.flatten(), f_mean_np.flatten(), "b-", linewidth=2, label="Predicted mean"
    )
    ax.fill_between(
        X_plot.flatten(),
        f_mean_np.flatten() - 2 * f_std,
        f_mean_np.flatten() + 2 * f_std,
        alpha=0.2,
        color="blue",
        label="±2σ",
    )
    # Plot training data points
    for i in range(n_classes):
        mask = y_train == i
        if np.any(mask):
            f_train_mean, _ = model.predict_f(X_train[mask])
            ax.scatter(
                X_train[mask].flatten(),
                y_train[mask].flatten(),
                c=colors[i % len(colors)],
                s=50,
                alpha=0.8,
                label=f"Class {i}",
                edgecolors="black",
                linewidth=0.5,
            )

    # Plot inducing points
    try:
        Z = model.inducing_variable.Z.numpy()
        f_inducing, _ = model.predict_f(Z)
        ax.scatter(
            Z.flatten(),
            np.ones_like(Z).flatten() * -1.0,
            marker="*",
            s=25,
            c="black",
            linewidth=3,
            label="Inducing points",
        )
    except Exception as e:
        print(f"Could not plot inducing points: {e}")

    # Plot bin edges
    bin_edges_np = (
        bin_edges.numpy() if hasattr(bin_edges, "numpy") else np.array(bin_edges)
    )
    for i, edge in enumerate(bin_edges_np):
        ax.axhline(
            y=edge,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Bin edges" if i == 0 else "",
        )

    ax.set_xlabel("Input X", fontsize=12)
    ax.set_ylabel("Latent Function Value", fontsize=12)
    ax.set_title(f"Train Acc: {accuracy*100:.2f}%", fontsize=14)
    # ax.legend(loc='best', fontsize=9)
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved posterior plot to {save_path}")
    else:
        plt.show()

    plt.close()


def create_config(
    lengthscale,
    likelihood,
    standardize_bins,
    learning_rate,
    epochs=500,
    num_inducing=10,
    bin_size=1.0,
    bin_range=None,
    num_layers=2,
    inner_mean_function="PCA",
):
    """Create a configuration dictionary for the model"""
    config_dict = {
        "model": {
            "name": "OrdinalDSVI_DGP",
            "type": "ordinal_dsvi_dgp",
            "kernel": {"type": "RBF", "lengthscale": lengthscale},
            "inducing_points": {"num_inducing": num_inducing, "init_method": "kmeans"},
            "bin_size": bin_size,
            "prior_temp": 1.0,
            "num_layers": num_layers,
            "gps_inner_layer": 1,
            "inner_mean_function": inner_mean_function,
            "train_kernel_params": True,
            "train_inducing_points": True,
            "train_likelihood_params": True,
            "likelihood": likelihood,
            "standardize_bins": standardize_bins,
            "bin_range": bin_range,
        },
        "training": {
            "batch_size": 100,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "patience": 200,
            "log_interval": 1,
            "lr_scheduler": "None",
            "lr_scheduler_last_epoch": 400,
        },
        "seed": 2,
        "output": {"save_weights_path": None},
    }
    return OmegaConf.create(config_dict)


def run_ablation_study(output_dir="toy_ablation/toy_ablation_results"):
    """Run ablation study over different configurations"""

    # Create output directory
    output_kld_dir = "toy_ablation/toy_ablation_losses"
    output_init = "toy_ablation/toy_init_ablation"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_kld_dir, exist_ok=True)
    os.makedirs(output_init, exist_ok=True)

    # Define parameter grid - varying lengthscale, likelihood, standardize_bins
    param_grid = {
        "lengthscale": [1.5],
        "likelihood": [
            # "Ordinal_fixed",
            "Ordinal_trainable",
            # "Ordinal_fixed_dirichlet",
            # "Ordinal_trainable_dirichlet",
            # "Ordinal_fixed_inverse",
            # "Ordinal_trainable_inverse",
        ],
        "standardize_bins": [True],
        "num_inducing": [10],
        "learning_rate": [0.001],
        "epochs": [2000],
        # "bin_size": [0.1, 1.0, 5.0, 10.0],  # Used as alphas
        # "bin_range": [1.0, 5.0, 10.0],  # Used to scale to [-K, K]
        "bin_size": [1.0],
        "bin_range": [None],
        "num_layers": [2, 5],
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f"Running ablation study with {len(combinations)} configurations...")
    print(f"Results will be saved to: {output_path.absolute()}")
    print("=" * 80)

    # Generate data once (reuse for all experiments)
    num_data = 100
    X_full, y_full, f_true = generate_data(num_data, seed=42)
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X_full, y_full, f_true, test_size=0.2, random_state=42, stratify=y_full
    )

    n_classes = len(np.unique(y_full))
    print(f"Dataset: {num_data} samples, {n_classes} classes")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print("=" * 80)

    # Run experiments
    for idx, params in enumerate(combinations):
        exp_name = (
            f"ls_{params['lengthscale']}_"
            f"lik_{params['likelihood']}_"
            f"stdbins_{params['standardize_bins']}_"
            f"binsize_{params['bin_size']}_"
            f"binrange_{params['bin_range']}_"
            f"layers_{params['num_layers']}"
        )

        print(f"\n[{idx+1}/{len(combinations)}] Running: {exp_name}")
        print(f"  Params: {params}")

        try:
            # Create config
            cfg = create_config(**params)

            # Build model
            model = build_model(cfg, X_train, y_train)

            # Save figure as PDF with exp_name.pdf
            figure_path = output_init + f"/{exp_name}_init.pdf"

            plot_latent_function_posterior(
                model, X_train, y_train, X_full, save_path=figure_path
            )

            # Create logger
            logger = SimpleLogger()

            # Train model
            trained_model = train(cfg, model, X_train, y_train, logger=logger)

            # Save figure as PDF with exp_name.pdf
            figure_path = output_path / f"{cfg.model.name}-{exp_name}.pdf"

            plot_latent_function_posterior(
                trained_model, X_train, y_train, X_full, save_path=figure_path
            )

            # Save training losses plot
            if params["epochs"] > 0:
                loss_figure_path = output_kld_dir + f"/{exp_name}_losses.pdf"
                logger.plot_losses(save_path=loss_figure_path)
            print(f"  ✓ Figures saved")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"All figures saved to: {output_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run toy ablation study for Ordinal SVGP"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="toy_ablation_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    run_ablation_study()
