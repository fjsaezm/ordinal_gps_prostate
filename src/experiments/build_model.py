import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from models.utils_instance_model import instance_gp_based_model


def build_model(cfg: DictConfig, X_train, y_train):
    """
    Construye un modelo de clasificación ordinal basado en el nombre del modelo.

    Args:
        cfg: OmegaConf configuration object
        X_train: Training features
        y_train: Training labels

    Returns:
        Model: The constructed model
    """
    model_name = cfg.model.name

    if model_name in ["OrdinalSVGP", "SVGP_multiclass"]:
        model = instance_gp_based_model(cfg, X_train, y_train)

    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=cfg.training.epochs, random_state=cfg.seed)

    elif model_name == "MLP":
        from sklearn.neural_network import MLPClassifier

        hidden_layers = cfg.model.get("hidden_layer_sizes", [100])
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            learning_rate_init=cfg.training.learning_rate,
            max_iter=cfg.training.epochs,
            random_state=cfg.seed,
        )

    elif model_name == "OrdinalMLP":
        from models.MLP import OrdinalMLP

        # Get number of classes
        n_classes = len(np.unique(y_train))

        # Compute bin_edges the same way as GP models
        if "dirichlet" in cfg.model.likelihood:
            rng = np.random.default_rng(cfg.seed)
            proportions = rng.dirichlet(
                cfg.model.bin_size * np.ones(n_classes - 1), size=1
            )
            bin_edges = np.cumsum(proportions)
            if cfg.model.bin_range is not None:
                r = cfg.model.bin_range
                bin_edges = -r + 2 * r * bin_edges
            else:
                bin_edges = bin_edges - bin_edges.mean()
        elif "inverse" in cfg.model.likelihood:
            from scipy.stats import norm
            u = np.random.uniform(0.0, 1.0, size=n_classes - 1)
            z = norm.ppf(u)
            bin_edges = np.sort(z)
        else:
            bin_size = cfg.model.bin_size
            standardize_bins = cfg.model.get("standardize_bins", False)
            bin_edges = np.array(np.arange(n_classes + 1), dtype=np.float64) * bin_size
            bin_edges = bin_edges - bin_edges.mean()
            if standardize_bins:
                bin_edges = bin_edges / bin_edges.std()
            bin_edges = bin_edges[1:-1]

        model = OrdinalMLP(
            inner_dims=cfg.model.inner_dims,
            activations=cfg.model.activations,
            bin_edges=bin_edges,
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
        )

    elif model_name == "SVM":
        from sklearn.svm import SVC

        model = SVC(probability=True, random_state=cfg.seed)

    elif model_name == "RLOrdinal":
        import mord

        # Use mord's LogisticAT (All-Threshold) which is much faster than statsmodels
        model = mord.LogisticAT(alpha=1.0)  # L2 regularization for stability

    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        n_neighbors = cfg.model.get("n_neighbors", 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        n_estimators = cfg.model.get("n_estimators", 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=cfg.seed)

    elif model_name == "SORD":
        from models.SORD import SORD

        model = SORD(
            num_classes=len(np.unique(y_train)),
            inner_dims=list(cfg.model.inner_dims),
            activations=list(cfg.model.activations),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            distance_power=cfg.model.get("distance_power", 1.0),
            temperature=cfg.model.get("temperature", 1.0),
        )

    elif model_name == "Beta":
        from models.Beta import BetaOrdinal

        model = BetaOrdinal(
            num_classes=len(np.unique(y_train)),
            inner_dims=list(cfg.model.inner_dims),
            activations=list(cfg.model.activations),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            smoothing=cfg.model.get("smoothing", 1.0),
            output_scheme=cfg.model.get("output_scheme", "stick_breaking"),
        )

    elif model_name == "POE":
        from models.POE import POE

        model = POE(
            num_classes=len(np.unique(y_train)),
            inner_dims=list(cfg.model.inner_dims),
            activations=list(cfg.model.activations),
            embedding_dim=cfg.model.get("embedding_dim", 32),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            kl_weight=cfg.model.get("kl_weight", 1e-5),
            ordinal_weight=cfg.model.get("ordinal_weight", 1e-4),
            margin=cfg.model.get("margin", 5.0),
            mc_samples_train=cfg.model.get("mc_samples_train", 4),
            mc_samples_predict=cfg.model.get("mc_samples_predict", 16),
        )

    elif model_name == "CORE":
        from models.CORE import CORE

        model = CORE(
            num_classes=len(np.unique(y_train)),
            inner_dims=list(cfg.model.inner_dims),
            activations=list(cfg.model.activations),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            core_weight=cfg.model.get("core_weight", 1.0),
            feature_temperature=cfg.model.get("feature_temperature", 2.0),
            label_temperature=cfg.model.get("label_temperature", 1.0),
        )

    elif model_name == "DRF":
        from models.DRF import DRF

        model = DRF(
            num_classes=len(np.unique(y_train)),
            inner_dims=list(cfg.model.inner_dims),
            activations=list(cfg.model.activations),
            num_trees=cfg.model.get("num_trees", 5),
            depth=cfg.model.get("depth", 4),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            min_variance=cfg.model.get("min_variance", 0.05),
        )
    elif model_name == "CORF":
        from models.CORF import CORF

        model = CORF(
            num_classes=len(np.unique(y_train)),
            inner_dims=list(cfg.model.inner_dims),
            activations=list(cfg.model.activations),
            num_trees=cfg.model.get("num_trees", 5),
            depth=cfg.model.get("depth", 6),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            leaf_temperature=cfg.model.get("leaf_temperature", 0.5),
            probability_epsilon=cfg.model.get("probability_epsilon", 1e-6),
        )
    else:
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. Modelos soportados: 'OrdinalSVGP', 'SVGP_multiclass', 'LogisticRegression', etc."
        )

    return model
