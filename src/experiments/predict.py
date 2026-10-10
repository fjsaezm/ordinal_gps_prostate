import numpy as np
import tensorflow as tf
from scipy.stats import norm
from omegaconf import DictConfig


def predict_class_ordinal_manual(f_mean, f_var, bin_edges, sigma=1.0):
    """
    Predict ordinal classes from a latent Gaussian distribution.

    Parameters:
    - f_mean: array of means (N, 1)
    - f_var: array of variances (N, 1)
    - bin_edges: array of bin edges (K-1,)
    - sigma: scalar or per-observation likelihood standard deviation

    Returns:
    - y_pred: predicted classes (N,)
    - probs: class probabilities (N, K)
    """
    # ensure bin_edge is np array:
    try:
        bin_edges = bin_edges.numpy()
    except:
        bin_edges = np.array(bin_edges)

    if len(f_mean.shape) > 2:
        f_mean = f_mean[0, :, 0]
    else:
        f_mean = f_mean[:, 0]
    f_std = np.sqrt(f_var[:, 0])
    f_std = np.maximum(f_std, 1e-6)  # Avoid divide-by-zero

    N = f_mean.shape[0]
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim > 0:
        sigma = sigma.reshape(-1)
        if sigma.size == 1:
            sigma = sigma.item()
        elif sigma.size != N:
            raise ValueError(
                f"Expected one likelihood scale per observation ({N}), "
                f"but received {sigma.size}"
            )
    sigma_sq = sigma**2

    n_classes = len(bin_edges) + 1
    probs = np.zeros((N, n_classes))

    bin_edges_ext = np.concatenate(([-np.inf], bin_edges, [np.inf]))

    quotient_term = np.sqrt(sigma_sq + f_std**2)
    # quotient_term = f_std

    for k in range(n_classes):
        upper = (bin_edges_ext[k + 1] - f_mean) / quotient_term
        lower = (bin_edges_ext[k] - f_mean) / quotient_term
        probs[:, k] = norm.cdf(upper) - norm.cdf(lower)

    y_pred = np.argmax(probs, axis=1)
    return y_pred, probs


def predict_class_ordinal_latent_noise(
    f_mean,
    f_var,
    bin_edges,
    baseline_sigma=1.0,
    max_log_scale_deviation=np.log(4.0),
    num_samples=128,
    seed=0,
):
    """Predict classes when the second SVGP output models log noise.

    We integrate the location GP analytically, as in the homoscedastic
    predictor, and use Monte Carlo only for the one-dimensional noise GP.
    """
    f_mean = np.asarray(f_mean, dtype=float)
    f_var = np.asarray(f_var, dtype=float)
    if f_mean.ndim != 2 or f_mean.shape[1] != 2:
        raise ValueError("Expected f_mean with shape (N, 2)")
    if f_var.shape != f_mean.shape:
        raise ValueError("Expected f_var to have the same shape as f_mean")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")

    try:
        bin_edges = bin_edges.numpy()
    except AttributeError:
        bin_edges = np.asarray(bin_edges)

    rng = np.random.default_rng(seed)
    noise_samples = (
        f_mean[None, :, 1]
        + np.sqrt(np.maximum(f_var[None, :, 1], 0.0))
        * rng.standard_normal((int(num_samples), f_mean.shape[0]))
    )
    sigma_samples = float(baseline_sigma) * np.exp(
        float(max_log_scale_deviation) * np.tanh(noise_samples)
    )
    denominator = np.sqrt(
        sigma_samples**2 + np.maximum(f_var[None, :, 0], 1e-12)
    )

    extended_edges = np.concatenate(([-np.inf], bin_edges, [np.inf]))
    probabilities = np.empty((f_mean.shape[0], len(extended_edges) - 1))
    for class_index in range(probabilities.shape[1]):
        upper = (
            extended_edges[class_index + 1] - f_mean[None, :, 0]
        ) / denominator
        lower = (
            extended_edges[class_index] - f_mean[None, :, 0]
        ) / denominator
        probabilities[:, class_index] = np.mean(
            norm.cdf(upper) - norm.cdf(lower), axis=0
        )

    return np.argmax(probabilities, axis=1), probabilities


def predict(cfg: DictConfig, model, X, y):
    model_name = cfg.model.name

    if model_name == "OrdinalSVGP":
        f_mean, f_var = model.predict_f(X)
        bin_edges_np = model.likelihood.bin_edges
        if (
            cfg.model.predict_likelihood_variance
            and getattr(model.likelihood, "uses_latent_noise", False)
        ):
            prediction_cfg = cfg.model.get("heteroscedastic_svgp", {})
            y_pred, class_probs = predict_class_ordinal_latent_noise(
                f_mean.numpy(),
                f_var.numpy(),
                bin_edges_np,
                baseline_sigma=model.likelihood.sigma.numpy(),
                max_log_scale_deviation=(
                    model.likelihood.max_log_scale_deviation
                ),
                num_samples=int(
                    prediction_cfg.get("num_predictive_samples", 128)
                ),
                seed=int(cfg.seed),
            )
        else:
            if cfg.model.predict_likelihood_variance:
                X_tensor = tf.convert_to_tensor(X, dtype=tf.float64)
                sigma = model.likelihood.noise_scale(X_tensor).numpy()
            else:
                sigma = 0.0
            # With likelihood variance disabled, only the location output is
            # relevant even for the two-latent heteroscedastic model.
            y_pred, class_probs = predict_class_ordinal_manual(
                f_mean[..., :1], f_var[..., :1], bin_edges_np, sigma
            )

    elif model_name == "SVGP_multiclass":
        f_mean, f_var = model.predict_f(X)
        class_probs = tf.nn.softmax(f_mean, axis=1).numpy()
        y_pred = np.argmax(class_probs, axis=1)

    elif model_name in [
            "LogisticRegression",
            "MLP",
            "SVM",
            "KNN",
            "RandomForest",
    ]:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            class_probs = model.predict_proba(X)
        else:
            class_probs = np.zeros((len(y_pred), len(np.unique(y))))

    elif model_name == "OrdinalMLP":
        X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
        class_probs = model.predict_probabilities(X_tf).numpy()
        y_pred = np.argmax(class_probs, axis=1)

    elif model_name == "RLOrdinal":
        # mord models use sklearn-like API
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            class_probs = model.predict_proba(X)
        else:
            class_probs = np.zeros((len(y_pred), len(np.unique(y))))
    elif model_name == "CORF":
        X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
        class_probs = model.predict_probabilities(X_tf).numpy()
        y_pred = model.predict_labels(X_tf).numpy()
    elif model_name in ["SORD", "Beta", "POE", "CORE", "DRF"]:
        X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
        class_probs = model.predict_probabilities(X_tf).numpy()
        y_pred = np.argmax(class_probs, axis=1)
    else:
        print(f"Model {model_name} not recognized for prediction.")

    return y_pred, class_probs
