"""Embedding-level adaptation of the Convolutional Ordinal Regression Forest.

The image backbone from Zhu et al. (TNNLS 2022) is replaced by the same small
MLP used by the other embedding-level neural baselines in this repository.
The ordinal forest, cumulative leaf distributions, and alternating leaf/router
optimization follow the method described in https://arxiv.org/abs/2008.03077.
"""

import numpy as np
import tensorflow as tf

from .ordinal_common import OrdinalBackbone


class CORF(tf.keras.Model):
    """Differentiable forest whose leaves contain ordinal distributions."""

    def __init__(
        self,
        num_classes,
        inner_dims,
        activations,
        num_trees=5,
        depth=6,
        dropout_rate=0.0,
        leaf_temperature=0.5,
        probability_epsilon=1e-6,
        name="corf",
    ):
        super().__init__(name=name, dtype=tf.float64)
        self.num_classes = int(num_classes)
        self.num_thresholds = self.num_classes - 1
        self.num_trees = int(num_trees)
        self.depth = int(depth)
        self.path_depth = self.depth - 1
        self.num_leaves = 2 ** self.path_depth
        self.num_splits = self.num_leaves - 1
        self.leaf_temperature = float(leaf_temperature)
        self.probability_epsilon = float(probability_epsilon)

        if self.num_classes < 2:
            raise ValueError("CORF requires at least two ordered classes")
        if self.depth < 2:
            raise ValueError("CORF depth must be at least 2")
        if self.leaf_temperature <= 0.0:
            raise ValueError("leaf_temperature must be positive")
        if not 0.0 < self.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie strictly between 0 and 0.5")

        self.router = OrdinalBackbone(
            inner_dims,
            activations,
            self.num_trees * self.num_splits,
            dropout_rate,
            name="corf_router",
        )
        self.leaf_cumulative_probabilities = tf.Variable(
            self._initial_leaf_distributions(),
            trainable=False,
            dtype=tf.float64,
            name="leaf_cumulative_probabilities",
        )
        self._path_nodes, self._path_left = self._make_paths()

    def _initial_leaf_distributions(self):
        """Initialize diverse, monotone P(y > k) vectors across the leaves."""
        locations = np.linspace(0.0, self.num_classes - 1.0, self.num_leaves)
        thresholds = np.arange(self.num_thresholds, dtype=np.float64) + 0.5
        scale = self.leaf_temperature
        logits = (locations[:, None] - thresholds[None, :]) / scale
        cumulative = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        cumulative = np.tile(cumulative[None, :, :], (self.num_trees, 1, 1))
        return np.clip(
            cumulative, self.probability_epsilon, 1.0 - self.probability_epsilon
        )

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

    def ordinal_targets(self, labels):
        labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.int32)
        thresholds = tf.range(self.num_thresholds, dtype=tf.int32)[None, :]
        return tf.cast(labels > thresholds, tf.float64)

    def tree_cumulative_probabilities(self, inputs, training=False):
        routing = self.routing_probabilities(inputs, training=training)
        return tf.einsum(
            "bml,mlk->bmk", routing, self.leaf_cumulative_probabilities
        )

    def cumulative_probabilities(self, inputs, training=False):
        return tf.reduce_mean(
            self.tree_cumulative_probabilities(inputs, training=training), axis=1
        )

    def loss_components(self, inputs, labels, training=True):
        targets = self.ordinal_targets(labels)[:, None, :]
        cumulative = tf.clip_by_value(
            self.tree_cumulative_probabilities(inputs, training=training),
            1e-12,
            1.0 - 1e-12,
        )
        binary_cross_entropy = -(
            targets * tf.math.log(cumulative)
            + (1.0 - targets) * tf.math.log(1.0 - cumulative)
        )
        # Equation (6): average the per-tree ordinal-distribution losses.
        loss = tf.reduce_mean(tf.reduce_sum(binary_cross_entropy, axis=2))
        return loss, {"loss": loss, "corf_ordinal_nll": loss}

    def update_leaf_distributions(self, inputs, labels):
        """Update leaf ordinal distributions with fixed routing parameters."""
        routing = self.routing_probabilities(inputs, training=False)
        targets = self.ordinal_targets(labels)[:, None, None, :]
        routing = routing[:, :, :, None]
        leaf_probabilities = self.leaf_cumulative_probabilities[None, :, :, :]

        positive_joint = routing * leaf_probabilities
        negative_joint = routing * (1.0 - leaf_probabilities)
        positive_total = tf.reduce_sum(positive_joint, axis=2, keepdims=True)
        negative_total = tf.reduce_sum(negative_joint, axis=2, keepdims=True)
        responsibilities = (
            targets * positive_joint / tf.maximum(positive_total, 1e-12)
            + (1.0 - targets)
            * negative_joint
            / tf.maximum(negative_total, 1e-12)
        )

        numerator = tf.reduce_sum(targets * responsibilities, axis=0)
        denominator = tf.reduce_sum(responsibilities, axis=0)
        updated = numerator / tf.maximum(denominator, 1e-12)

        # Ordinal labels are descending cumulative vectors. The paper's update
        # preserves that structure; this projection also protects it against
        # finite-precision violations that would yield negative class masses.
        updated_np = updated.numpy()
        updated_np = np.minimum.accumulate(updated_np, axis=2)
        updated_np = np.clip(
            updated_np, self.probability_epsilon, 1.0 - self.probability_epsilon
        )
        self.leaf_cumulative_probabilities.assign(updated_np)

    def predict_probabilities(self, inputs, mc_samples=None):
        del mc_samples
        cumulative = self.cumulative_probabilities(inputs, training=False)
        first = 1.0 - cumulative[:, :1]
        middle = cumulative[:, :-1] - cumulative[:, 1:]
        last = cumulative[:, -1:]
        probabilities = tf.concat([first, middle, last], axis=1)
        probabilities = tf.maximum(probabilities, 0.0)
        return probabilities / tf.reduce_sum(probabilities, axis=1, keepdims=True)

    def predict_labels(self, inputs):
        """Paper prediction rule: count cumulative outputs exceeding 0.5."""
        cumulative = self.cumulative_probabilities(inputs, training=False)
        return tf.reduce_sum(tf.cast(cumulative > 0.5, tf.int64), axis=1)
