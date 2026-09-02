"""Shared TensorFlow layers for ordinal baselines on precomputed features."""

from typing import List

import tensorflow as tf


class OrdinalBackbone(tf.keras.Model):
    """Small MLP that exposes both its embedding and output logits."""

    def __init__(
        self,
        inner_dims: List[int],
        activations: List[str],
        output_dim: int,
        dropout_rate: float = 0.0,
        name: str = "ordinal_backbone",
    ):
        super().__init__(name=name, dtype=tf.float64)
        if len(inner_dims) != len(activations):
            raise ValueError("inner_dims and activations must have the same length")
        if not inner_dims:
            raise ValueError("inner_dims cannot be empty")

        self.hidden_layers = []
        self.dropout_layers = []
        for index, (width, activation) in enumerate(zip(inner_dims, activations)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    width,
                    activation=None if activation == "None" else activation,
                    dtype=tf.float64,
                    name=f"dense_{index}",
                )
            )
            self.dropout_layers.append(
                tf.keras.layers.Dropout(dropout_rate, dtype=tf.float64)
                if dropout_rate > 0.0
                else None
            )
        self.output_layer = tf.keras.layers.Dense(
            output_dim, activation=None, dtype=tf.float64, name="output"
        )

    def embed(self, inputs, training=False):
        x = tf.cast(inputs, tf.float64)
        for layer, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = layer(x)
            if dropout is not None:
                x = dropout(x, training=training)
        return x

    def call(self, inputs, training=False):
        embedding = self.embed(inputs, training=training)
        return self.output_layer(embedding)


def soft_cross_entropy(targets, probabilities):
    probabilities = tf.clip_by_value(probabilities, 1e-12, 1.0)
    return -tf.reduce_mean(
        tf.reduce_sum(tf.cast(targets, tf.float64) * tf.math.log(probabilities), axis=1)
    )

