"""Embedding-level TensorFlow adaptation of Deep Regression Forests."""

import numpy as np
import tensorflow as tf

from .ordinal_common import OrdinalBackbone


class DRF(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        inner_dims,
        activations,
        num_trees=5,
        depth=4,
        dropout_rate=0.0,
        min_variance=0.05,
        name="drf",
    ):
        super().__init__(name=name, dtype=tf.float64)
        self.num_classes = int(num_classes)
        self.num_trees = int(num_trees)
        self.depth = int(depth)
        self.path_depth = self.depth - 1
        self.num_leaves = 2 ** self.path_depth
        self.num_splits = self.num_leaves - 1
        self.min_variance = float(min_variance)
        self.router = OrdinalBackbone(
            inner_dims,
            activations,
            self.num_trees * self.num_splits,
            dropout_rate,
            name="drf_router",
        )
        initial_means = np.linspace(0.0, self.num_classes - 1.0, self.num_leaves)
        initial_means = np.tile(initial_means[None, :], (self.num_trees, 1))
        self.leaf_means = tf.Variable(
            initial_means, trainable=False, dtype=tf.float64, name="leaf_means"
        )
        self.leaf_variances = tf.Variable(
            tf.ones((self.num_trees, self.num_leaves), dtype=tf.float64),
            trainable=False,
            name="leaf_variances",
        )
        self._path_nodes, self._path_left = self._make_paths()

    def _make_paths(self):
        nodes = np.zeros((self.num_leaves, self.path_depth), dtype=np.int32)
        left = np.zeros((self.num_leaves, self.path_depth), dtype=bool)
        for leaf in range(self.num_leaves):
            node = 0
            for level in range(self.path_depth):
                go_right = bool((leaf >> (self.path_depth - level - 1)) & 1)
                nodes[leaf, level] = node
                left[leaf, level] = not go_right
                node = 2 * node + (2 if go_right else 1)
        return tf.constant(nodes), tf.constant(left)

    def routing_probabilities(self, inputs, training=False):
        logits = self.router(inputs, training=training)
        decisions = tf.math.sigmoid(
            tf.reshape(logits, [-1, self.num_trees, self.num_splits])
        )
        paths = []
        for leaf in range(self.num_leaves):
            node_probabilities = tf.gather(decisions, self._path_nodes[leaf], axis=2)
            directions = self._path_left[leaf][None, None, :]
            path_probabilities = tf.where(
                directions, node_probabilities, 1.0 - node_probabilities
            )
            paths.append(tf.reduce_prod(path_probabilities, axis=2))
        routing = tf.stack(paths, axis=2)
        return routing / tf.reduce_sum(routing, axis=2, keepdims=True)

    def component_density(self, labels):
        labels = tf.cast(tf.reshape(labels, [-1, 1, 1]), tf.float64)
        variance = tf.maximum(self.leaf_variances[None, :, :], self.min_variance)
        difference = labels - self.leaf_means[None, :, :]
        return tf.exp(-0.5 * tf.square(difference) / variance) / tf.sqrt(
            2.0 * np.pi * variance
        )

    def loss_components(self, inputs, labels, training=True):
        routing = self.routing_probabilities(inputs, training=training)
        density = self.component_density(labels)
        mixture_density = tf.reduce_mean(tf.reduce_sum(routing * density, axis=2), axis=1)
        nll = -tf.reduce_mean(tf.math.log(tf.maximum(mixture_density, 1e-12)))
        return nll, {"loss": nll, "drf_nll": nll}

    def update_leaf_distributions(self, inputs, labels):
        routing = self.routing_probabilities(inputs, training=False)
        density = self.component_density(labels)
        responsibility = routing * density
        responsibility /= tf.reduce_sum(responsibility, axis=2, keepdims=True) + 1e-12
        labels_3d = tf.cast(tf.reshape(labels, [-1, 1, 1]), tf.float64)
        weights = tf.reduce_sum(responsibility, axis=0) + 1e-12
        means = tf.reduce_sum(responsibility * labels_3d, axis=0) / weights
        variances = tf.reduce_sum(
            responsibility * tf.square(labels_3d - means[None, :, :]), axis=0
        ) / weights
        self.leaf_means.assign(means)
        self.leaf_variances.assign(tf.maximum(variances, self.min_variance))

    @staticmethod
    def _normal_cdf(value, mean, variance):
        z = (value - mean) / tf.sqrt(2.0 * variance)
        return 0.5 * (1.0 + tf.math.erf(z))

    def predict_probabilities(self, inputs, mc_samples=None):
        del mc_samples
        routing = self.routing_probabilities(inputs, training=False)
        means = self.leaf_means[None, :, :]
        variances = tf.maximum(self.leaf_variances[None, :, :], self.min_variance)
        boundaries = [-np.inf] + [index + 0.5 for index in range(self.num_classes - 1)] + [np.inf]
        class_probabilities = []
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            upper_cdf = tf.ones_like(means) if np.isposinf(upper) else self._normal_cdf(upper, means, variances)
            lower_cdf = tf.zeros_like(means) if np.isneginf(lower) else self._normal_cdf(lower, means, variances)
            component_probability = upper_cdf - lower_cdf
            class_probabilities.append(
                tf.reduce_mean(tf.reduce_sum(routing * component_probability, axis=2), axis=1)
            )
        probabilities = tf.stack(class_probabilities, axis=1)
        return probabilities / tf.reduce_sum(probabilities, axis=1, keepdims=True)

