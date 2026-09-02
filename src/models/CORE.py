"""CORE ordinal-representation regularisation for frozen feature vectors."""

import tensorflow as tf

from .ordinal_common import OrdinalBackbone


class CORE(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        inner_dims,
        activations,
        dropout_rate=0.0,
        core_weight=1.0,
        feature_temperature=2.0,
        label_temperature=1.0,
        name="core",
    ):
        super().__init__(name=name, dtype=tf.float64)
        self.num_classes = int(num_classes)
        self.core_weight = float(core_weight)
        self.feature_temperature = float(feature_temperature)
        self.label_temperature = float(label_temperature)
        self.backbone = OrdinalBackbone(
            inner_dims, activations, self.num_classes, dropout_rate, name="core_mlp"
        )

    def call(self, inputs, training=False, return_embedding=False):
        embedding = self.backbone.embed(inputs, training=training)
        logits = self.backbone.output_layer(embedding)
        return (logits, embedding) if return_embedding else logits

    def core_loss(self, embedding, labels):
        embedding = tf.cast(embedding, tf.float64)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float64)
        feature_squared = tf.reduce_sum(tf.square(embedding), axis=1, keepdims=True)
        feature_distances = tf.maximum(
            feature_squared - 2.0 * tf.matmul(embedding, embedding, transpose_b=True)
            + tf.transpose(feature_squared),
            0.0,
        )
        label_distances = tf.square(labels[:, None] - labels[None, :])
        feature_distribution = tf.nn.softmax(
            -feature_distances / self.feature_temperature, axis=1
        )
        label_distribution = tf.nn.softmax(
            -label_distances / self.label_temperature, axis=1
        )
        feature_distribution = tf.clip_by_value(feature_distribution, 1e-12, 1.0)
        label_distribution = tf.clip_by_value(label_distribution, 1e-12, 1.0)
        return tf.reduce_mean(
            tf.reduce_sum(
                label_distribution
                * (tf.math.log(label_distribution) - tf.math.log(feature_distribution)),
                axis=1,
            )
        )

    def loss_components(self, inputs, labels, training=True):
        logits, embedding = self(inputs, training=training, return_embedding=True)
        labels_flat = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels_flat, logits, from_logits=True
            )
        )
        ordinal = self.core_loss(embedding, labels)
        loss = ce + self.core_weight * ordinal
        return loss, {"loss": loss, "core_ce": ce, "core_ordinal": ordinal}

    def predict_probabilities(self, inputs, mc_samples=None):
        del mc_samples
        return tf.nn.softmax(self(inputs, training=False), axis=1)

