from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    log_loss,
)
try:
    from netcal.metrics import ECE
except ImportError:
    class ECE:
        """Minimal equal-width ECE fallback matching the repository setting."""

        def __init__(self, bins=15):
            self.bins = int(bins)

        def measure(self, probabilities, labels):
            import numpy as np

            probabilities = np.asarray(probabilities)
            labels = np.asarray(labels).reshape(-1)
            confidence = probabilities.max(axis=1)
            predictions = probabilities.argmax(axis=1)
            boundaries = np.linspace(0.0, 1.0, self.bins + 1)
            error = 0.0
            for index in range(self.bins):
                if index == self.bins - 1:
                    in_bin = (confidence >= boundaries[index]) & (
                        confidence <= boundaries[index + 1]
                    )
                else:
                    in_bin = (confidence >= boundaries[index]) & (
                        confidence < boundaries[index + 1]
                    )
                if np.any(in_bin):
                    accuracy = np.mean(predictions[in_bin] == labels[in_bin])
                    mean_confidence = np.mean(confidence[in_bin])
                    error += np.mean(in_bin) * abs(accuracy - mean_confidence)
            return float(error)


def multiclass_brier_score(y_true, y_prob):
    """
    Brier score for multiclass classification.

    Args:
        y_true (np.ndarray): true labels, shape [N]
        y_prob (np.ndarray): predicted probabilities, shape [N, C]

    Returns:
        float: average Brier score
    """
    num_classes = y_prob.shape[1]
    brier = 0.0
    for c in range(num_classes):
        true_binary = (y_true == c).astype(int)
        brier += brier_score_loss(true_binary, y_prob[:, c])
    return brier / num_classes


def compute_metrics(y_true, y_pred, y_probs):

    report = classification_report(y_true, y_pred, output_dict=True)

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred)
    try:
        brier = multiclass_brier_score(y_true, y_probs)
    except:
        brier = float("nan")
    try:
        ece = ECE(bins=15).measure(y_probs, y_true)
    except:
        ece = float("nan")

    try:
        logloss = log_loss(y_true, y_probs)
    except:
        logloss = float("nan")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    d = {
        "acc": report["accuracy"],
        "kappa": kappa,
        "mae": mae,
        "brier": brier,
        "ece": ece,
        "nll": logloss,
    }
    for m in ["precision", "recall", "f1-score", "support"]:
        for k, v in report.items():
            if k.isdigit():  # Only per-class metrics
                d[f"{m}_class_{k}"] = v[m]

        d[f"{m}_weighted"] = report["weighted avg"][m]

    return d, cm
