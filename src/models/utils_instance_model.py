import gpflow
from .likelihoods.TrainableOrdinal import TrainableOrdinal
from .likelihoods.HeteroscedasticOrdinal import HeteroscedasticOrdinal
from .likelihoods.HeteroscedasticOrdinalSVGP import HeteroscedasticOrdinalSVGP
from gpflow.utilities import set_trainable
import numpy as np
from typing import Any
from numpy.typing import NDArray as AnyNDArray
from omegaconf import DictConfig
from gpflow.base import Parameter
from gpflow.utilities import positive


def initialize_Z(cfg: DictConfig, X_train):
    """
    Initialize inducing points based on the specified method.
    """
    num_inducing = cfg.model.inducing_points.num_inducing
    init_method = cfg.model.inducing_points.init_method

    if num_inducing > X_train.shape[0]:
        raise ValueError(
            f"The number of inducing points ({num_inducing}) cannot be greater than the number of training points ({X_train.shape[0]})."
        )

    if init_method == "kmeans":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_inducing, random_state=cfg.seed)
        kmeans.fit(X_train)
        init_Z = kmeans.cluster_centers_

    elif init_method == "random":
        init_Z = X_train[
            np.random.choice(X_train.shape[0], num_inducing, replace=False),
            :,
        ]

    return init_Z


def init_lengthscale_GP(cfg: DictConfig, X_train):
    lengthscale = cfg.model.kernel.lengthscale
    if isinstance(lengthscale, (int, float)):
        return float(lengthscale)
    elif lengthscale == "scale":
        gamma_value = 1.0 / (X_train.shape[1] * X_train.var())
        return 1.0 / np.sqrt(2.0 * float(gamma_value))
    return 1.0


def instance_kernel(cfg: DictConfig, X_train: AnyNDArray) -> gpflow.kernels.Kernel:
    lengthscale = init_lengthscale_GP(cfg, X_train)

    kernel_type = cfg.model.kernel.type
    if kernel_type == "Matern32":
        kernel = gpflow.kernels.Matern32(lengthscales=lengthscale)
    elif kernel_type == "RBF":
        kernel = gpflow.kernels.RBF(lengthscales=lengthscale, variance=1.0)
        # kernel = gpflow.kernels.RBF(lengthscales=2.0, variance=2.0)
    elif kernel_type == "linear":
        kernel = gpflow.kernels.Linear()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Control kernel parameter trainability
    train_kernel_params = cfg.model.get("train_kernel_params", True)
    if not train_kernel_params:
        set_trainable(kernel.lengthscales, False)
        set_trainable(kernel.variance, False)

    return kernel


def instance_likelihood(cfg, n_classes):
    name = cfg.model.likelihood

    if "Ordinal" in cfg.model.likelihood:
        if "dirichlet" in cfg.model.likelihood:
            rng = np.random.default_rng(cfg.seed)
            proportions = rng.dirichlet(
                cfg.model.bin_size * np.ones(n_classes - 1), size=1
            )
            # Convert to cumulative thresholds in [0, 1)
            bin_edges = np.cumsum(proportions)
            if cfg.model.bin_range is not None:
                r = cfg.model.bin_range
                bin_edges = -r + 2 * r * bin_edges
            else:
                bin_edges = bin_edges - bin_edges.mean()
            # print("Initialized bin edges (dirichlet):", bin_edges)
        elif "inverse" in cfg.model.likelihood:
            from scipy.stats import norm

            u = np.random.uniform(0.0, 1.0, size=n_classes - 1)
            z = norm.ppf(u)
            bin_edges = np.sort(z)
            # print("Initialized bin edges (uniform):", bin_edges)
        else:
            bin_size = cfg.model.bin_size
            standardize_bins = cfg.model.get("standardize_bins", False)
            bin_edges = np.array(np.arange(n_classes + 1), dtype=np.float64) * bin_size
            bin_edges = bin_edges - bin_edges.mean()
            if standardize_bins:
                bin_edges = bin_edges / bin_edges.std()

            # print("Initialized bin edges (equi-spaced):", bin_edges)
            bin_edges = bin_edges[1:-1]

        # print("Using bin edges:", bin_edges)

    use_heteroscedastic_likelihood = bool(cfg.model.get("heteroscedastic", False))
    heteroscedastic_predictor = str(
        cfg.model.get("heteroscedastic_predictor", "mlp")
    ).lower()
    if use_heteroscedastic_likelihood and cfg.model.name != "OrdinalSVGP":
        raise ValueError(
            "model.heteroscedastic=true is currently supported only for OrdinalSVGP"
        )
    if use_heteroscedastic_likelihood and "Ordinal_trainable" not in name:
        raise ValueError(
            "model.heteroscedastic=true requires likelihood=Ordinal_trainable"
        )
    if use_heteroscedastic_likelihood and heteroscedastic_predictor not in {
        "mlp",
        "svgp",
    }:
        raise ValueError(
            "model.heteroscedastic_predictor must be either 'mlp' or 'svgp'"
        )

    if (
        "Ordinal_trainable" in name
        and use_heteroscedastic_likelihood
        and heteroscedastic_predictor == "svgp"
    ):
        heteroscedastic_cfg = cfg.model.get("heteroscedastic_svgp", {})
        likelihood = HeteroscedasticOrdinalSVGP(
            initial_bin_edges=bin_edges,
            baseline_sigma=float(heteroscedastic_cfg.get("baseline_sigma", 1.0)),
            max_log_scale_deviation=float(
                heteroscedastic_cfg.get("max_log_scale_deviation", np.log(4.0))
            ),
            num_monte_carlo_points=int(
                heteroscedastic_cfg.get("num_monte_carlo_points", 32)
            ),
        )
    elif "Ordinal_trainable" in name and use_heteroscedastic_likelihood:
        heteroscedastic_cfg = cfg.model.get("heteroscedastic_mlp", {})
        likelihood = HeteroscedasticOrdinal(
            initial_bin_edges=bin_edges,
            hidden_dims=list(heteroscedastic_cfg.get("hidden_dims", [32])),
            activations=list(heteroscedastic_cfg.get("activations", ["tanh"])),
            baseline_sigma=float(heteroscedastic_cfg.get("baseline_sigma", 1.0)),
            max_log_scale_deviation=float(
                heteroscedastic_cfg.get("max_log_scale_deviation", np.log(4.0))
            ),
        )
    elif "Ordinal_trainable" in name:
        likelihood = TrainableOrdinal(initial_bin_edges=bin_edges)
    elif "Ordinal_fixed" in name:
        likelihood = gpflow.likelihoods.Ordinal(bin_edges=bin_edges)

    elif name == "Multiclass":
        likelihood = gpflow.likelihoods.Softmax(num_classes=n_classes)

    return likelihood


def instance_SVGP(cfg, init_Z, num_latent_gps, kernel, likelihood, X_train):

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=init_Z,
        num_latent_gps=num_latent_gps,
        num_data=X_train.shape[0],
    )

    # Control inducing points trainability
    train_inducing_points = cfg.model.train_inducing_points
    if not train_inducing_points:
        set_trainable(model.inducing_variable, False)

    # Control likelihood parameters trainability
    train_likelihood_params = cfg.model.train_likelihood_params
    if not train_likelihood_params:
        for param in model.likelihood.parameters:
            set_trainable(param, False)
        if hasattr(model.likelihood, "noise_scale_mlp"):
            model.likelihood.noise_scale_mlp.trainable = False

    return model


def instance_gp_based_model(cfg, X_train, y_train, versbose=False) -> Any:

    model_name = cfg.model.name
    init_Z = initialize_Z(cfg, X_train)
    lengthscale = init_lengthscale_GP(cfg, X_train)
    n_classes = len(np.unique(y_train))
    likelihood = instance_likelihood(cfg, n_classes)

    if model_name in ["OrdinalSVGP", "SVGP_multiclass"]:
        kernel = instance_kernel(cfg, X_train)
        num_latent_gps = 1 if model_name == "OrdinalSVGP" else n_classes

        if getattr(likelihood, "uses_latent_noise", False):
            noise_kernel = instance_kernel(cfg, X_train)
            kernel = gpflow.kernels.SeparateIndependent([kernel, noise_kernel])
            init_Z = gpflow.inducing_variables.SharedIndependentInducingVariables(
                gpflow.inducing_variables.InducingPoints(init_Z)
            )
            num_latent_gps = 2

        model = instance_SVGP(cfg, init_Z, num_latent_gps, kernel, likelihood, X_train)

    return model
