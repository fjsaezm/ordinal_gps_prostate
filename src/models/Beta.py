"""Beta-distribution unimodal regularisation with an ordinal stick head."""

import numpy as np
import tensorflow as tf
from scipy.special import betainc

from .ordinal_common import OrdinalBackbone, soft_cross_entropy


_PUBLISHED_PARAMETERS = {
    3: [(1, 4), (4, 4), (4, 1)],
    4: [(1, 6), (6, 10), (10, 6), (6, 1)],
    5: [(1, 8), (6, 14), (12, 12), (14, 6), (8, 1)],
    6: [(1, 10), (7, 20), (15, 20), (20, 15), (20, 7), (10, 1)],
    7: [(1, 12), (7, 26), (16, 28), (24, 24), (28, 16), (26, 7), (12, 1)],
    8: [(1, 14), (7, 31), (17, 37), (27, 35), (35, 27), (37, 17), (31, 7), (14, 1)],
}


def beta_target_matrix(num_classes):
    """Integrate each published beta density over the ordinal class bins."""
    if num_classes not in _PUBLISHED_PARAMETERS:
        raise ValueError("Beta regularisation supports 3 to 8 classes")
    edges = np.linspace(0.0, 1.0, num_classes + 1)
    targets = []
    for alpha, beta in _PUBLISHED_PARAMETERS[num_classes]:
        probabilities = np.diff(betainc(alpha, beta, edges))
        probabilities = probabilities / probabilities.sum()
        targets.append(probabilities)
    return np.asarray(targets, dtype=np.float64)


class BetaOrdinal(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        inner_dims,
        activations,
        dropout_rate=0.0,
        smoothing=1.0,
        output_scheme="stick_breaking",
        name="beta_ordinal",
    ):
        super().__init__(name=name, dtype=tf.float64)
        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("Beta smoothing must lie in [0, 1]")
        if output_scheme not in {"softmax", "stick_breaking"}:
            raise ValueError("output_scheme must be softmax or stick_breaking")
        self.output_scheme = output_scheme
        output_dim = self.num_classes if output_scheme == "softmax" else self.num_classes - 1
        self.backbone = OrdinalBackbone(
            inner_dims, activations, output_dim, dropout_rate, name="beta_mlp"
        )
        self.beta_targets = tf.constant(beta_target_matrix(self.num_classes))

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def probabilities_from_logits(self, logits):
        if self.output_scheme == "softmax":
            return tf.nn.softmax(logits, axis=1)
        breaks = tf.math.sigmoid(logits)
        remaining_before = tf.math.cumprod(1.0 - breaks, axis=1, exclusive=True)
        first_classes = breaks * remaining_before
        final_class = tf.reduce_prod(1.0 - breaks, axis=1, keepdims=True)
        probabilities = tf.concat([first_classes, final_class], axis=1)
        return probabilities / tf.reduce_sum(probabilities, axis=1, keepdims=True)

    def soft_targets(self, labels):
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        beta_targets = tf.gather(self.beta_targets, labels)
        one_hot = tf.one_hot(labels, self.num_classes, dtype=tf.float64)
        return (1.0 - self.smoothing) * one_hot + self.smoothing * beta_targets

    def loss_components(self, inputs, labels, training=True):
        probabilities = self.probabilities_from_logits(self(inputs, training=training))
        loss = soft_cross_entropy(self.soft_targets(labels), probabilities)
        return loss, {"loss": loss, "beta_ce": loss}

    def predict_probabilities(self, inputs, mc_samples=None):
        del mc_samples
        return self.probabilities_from_logits(self(inputs, training=False))

