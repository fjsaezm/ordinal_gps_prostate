"""Probabilistic Ordinal Embeddings for frozen feature vectors."""

import tensorflow as tf

from .ordinal_common import OrdinalBackbone


class POE(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        inner_dims,
        activations,
        embedding_dim=32,
        dropout_rate=0.0,
        kl_weight=1e-5,
        ordinal_weight=1e-4,
        margin=5.0,
        mc_samples_train=4,
        mc_samples_predict=16,
        name="poe",
    ):
        super().__init__(name=name, dtype=tf.float64)
        self.num_classes = int(num_classes)
        self.kl_weight = float(kl_weight)
        self.ordinal_weight = float(ordinal_weight)
        self.margin = float(margin)
        self.mc_samples_train = int(mc_samples_train)
        self.mc_samples_predict = int(mc_samples_predict)
        self.backbone = OrdinalBackbone(
            inner_dims, activations, inner_dims[-1], dropout_rate, name="poe_mlp"
        )
        self.mean_layer = tf.keras.layers.Dense(embedding_dim, dtype=tf.float64, name="mean")
        self.log_variance_layer = tf.keras.layers.Dense(
            embedding_dim, dtype=tf.float64, name="log_variance"
        )
        self.classifier = tf.keras.layers.Dense(
            self.num_classes, dtype=tf.float64, name="classifier"
        )

    def distribution(self, inputs, training=False):
        hidden = self.backbone.embed(inputs, training=training)
        mean = self.mean_layer(hidden)
        log_variance = tf.clip_by_value(self.log_variance_layer(hidden), -10.0, 8.0)
        return mean, log_variance

    def sample_logits(self, mean, log_variance, samples):
        shape = tf.concat([[samples], tf.shape(mean)], axis=0)
        epsilon = tf.random.normal(shape, dtype=tf.float64)
        z = mean[None, :, :] + tf.exp(0.5 * log_variance)[None, :, :] * epsilon
        return self.classifier(z)

    @staticmethod
    def wasserstein_distance(mean_a, var_a, mean_b, var_b):
        mean_term = tf.reduce_sum(tf.square(mean_a - mean_b), axis=1)
        variance_term = tf.reduce_sum(
            tf.square(tf.sqrt(var_a) - tf.sqrt(var_b)), axis=1
        )
        return tf.sqrt(tf.maximum(mean_term + variance_term, 1e-12))

    def ordinal_triplet_loss(self, mean, log_variance, labels):
        batch_size = tf.shape(mean)[0]

        def no_triplets():
            return tf.constant(0.0, tf.float64)

        def compute_triplets():
            labels_flat = tf.cast(tf.reshape(labels, [-1]), tf.float64)
            anchors = tf.range(batch_size)
            seconds = tf.math.mod(anchors + 1, batch_size)
            all_label_distances = tf.abs(labels_flat[:, None] - labels_flat[None, :])
            second_label_distances = tf.gather_nd(
                all_label_distances, tf.stack([anchors, seconds], axis=1)
            )
            difference = tf.abs(all_label_distances - second_label_distances[:, None])
            invalid = tf.logical_or(
                tf.eye(batch_size, dtype=tf.bool), difference < tf.constant(1e-12, tf.float64)
            )
            difference = tf.where(invalid, tf.constant(1e12, tf.float64), difference)
            thirds = tf.argmin(difference, axis=1, output_type=tf.int32)
            third_label_distances = tf.gather_nd(
                all_label_distances, tf.stack([anchors, thirds], axis=1)
            )
            signs = tf.sign(second_label_distances - third_label_distances)
            variance = tf.exp(log_variance)
            distance_12 = self.wasserstein_distance(
                mean, variance, tf.gather(mean, seconds), tf.gather(variance, seconds)
            )
            distance_13 = self.wasserstein_distance(
                mean, variance, tf.gather(mean, thirds), tf.gather(variance, thirds)
            )
            violations = tf.nn.relu((distance_13 - distance_12) * signs + self.margin)
            valid = tf.abs(signs)
            return tf.reduce_sum(violations * valid) / tf.maximum(
                tf.reduce_sum(valid), tf.constant(1.0, tf.float64)
            )

        return tf.cond(batch_size >= 3, compute_triplets, no_triplets)

    def loss_components(self, inputs, labels, training=True):
        mean, log_variance = self.distribution(inputs, training=training)
        logits = self.sample_logits(mean, log_variance, self.mc_samples_train)
        repeated_labels = tf.tile(tf.cast(tf.reshape(labels, [-1]), tf.int32), [self.mc_samples_train])
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                repeated_labels,
                tf.reshape(logits, [-1, self.num_classes]),
                from_logits=True,
            )
        )
        kl = 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(mean) + tf.exp(log_variance) - log_variance - 1.0, axis=1
            )
        )
        ordinal = self.ordinal_triplet_loss(mean, log_variance, labels)
        loss = ce + self.kl_weight * kl + self.ordinal_weight * ordinal
        return loss, {"loss": loss, "poe_ce": ce, "poe_kl": kl, "poe_ordinal": ordinal}

    def predict_probabilities(self, inputs, mc_samples=None):
        mean, log_variance = self.distribution(inputs, training=False)
        samples = self.mc_samples_predict if mc_samples is None else int(mc_samples)
        logits = self.sample_logits(mean, log_variance, samples)
        return tf.reduce_mean(tf.nn.softmax(logits, axis=-1), axis=0)
