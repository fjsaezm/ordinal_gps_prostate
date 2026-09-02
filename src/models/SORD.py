"""Soft Labels for Ordinal Regression (SORD)."""

import tensorflow as tf

from .ordinal_common import OrdinalBackbone, soft_cross_entropy


class SORD(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        inner_dims,
        activations,
        dropout_rate=0.0,
        distance_power=1.0,
        temperature=1.0,
        name="sord",
    ):
        super().__init__(name=name, dtype=tf.float64)
        self.num_classes = int(num_classes)
        self.distance_power = float(distance_power)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("SORD temperature must be positive")
        self.backbone = OrdinalBackbone(
            inner_dims, activations, self.num_classes, dropout_rate, name="sord_mlp"
        )

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def soft_targets(self, labels):
        labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float64)
        ranks = tf.cast(tf.range(self.num_classes)[None, :], tf.float64)
        distances = tf.pow(tf.abs(labels - ranks), self.distance_power)
        return tf.nn.softmax(-distances / self.temperature, axis=1)

    def loss_components(self, inputs, labels, training=True):
        probabilities = tf.nn.softmax(self(inputs, training=training), axis=1)
        loss = soft_cross_entropy(self.soft_targets(labels), probabilities)
        return loss, {"loss": loss, "sord_ce": loss}

    def predict_probabilities(self, inputs, mc_samples=None):
        del mc_samples
        return tf.nn.softmax(self(inputs, training=False), axis=1)

