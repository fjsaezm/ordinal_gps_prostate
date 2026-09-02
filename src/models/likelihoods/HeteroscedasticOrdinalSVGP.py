"""Ordinal-probit likelihood driven by latent location and noise GPs."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.likelihoods import MonteCarloLikelihood
from gpflow.utilities import positive, set_trainable, to_default_int

from .TrainableOrdinal import (
    default_float,
    stable_inv_probit,
    standardize_boundary,
)


class HeteroscedasticOrdinalSVGP(MonteCarloLikelihood):
    """Ordinal likelihood with separate latent location and log-noise GPs.

    The two latent outputs are ``F[..., 0] = f(x)`` and ``F[..., 1] = g(x)``.
    The observation scale is a bounded transformation of the noise latent,

    ``sigma(x) = baseline_sigma * exp(max_deviation * tanh(g(x)))``.

    Bounding the transformation retains the numerical safeguards used by the
    MLP variant while the GP prior and its variational KL regularize ``g``.
    """

    def __init__(
        self,
        initial_bin_edges: np.ndarray,
        baseline_sigma: float = 1.0,
        max_log_scale_deviation: float = float(np.log(4.0)),
        num_monte_carlo_points: int = 32,
    ) -> None:
        if baseline_sigma <= 0:
            raise ValueError("baseline_sigma must be positive")
        if max_log_scale_deviation <= 0:
            raise ValueError("max_log_scale_deviation must be positive")
        if num_monte_carlo_points <= 0:
            raise ValueError("num_monte_carlo_points must be positive")

        super().__init__(input_dim=None, latent_dim=2, observation_dim=1)
        self.num_monte_carlo_points = int(num_monte_carlo_points)
        self.num_bins = initial_bin_edges.size + 1
        self.first_edge = float(initial_bin_edges[0])
        self._bin_increments = Parameter(
            np.diff(initial_bin_edges), transform=positive()
        )
        self.sigma = Parameter(baseline_sigma, transform=positive())
        set_trainable(self.sigma, False)
        self.max_log_scale_deviation = float(max_log_scale_deviation)

    @property
    def is_heteroscedastic(self) -> bool:
        return True

    @property
    def heteroscedastic_predictor(self) -> str:
        return "svgp"

    @property
    def uses_latent_noise(self) -> bool:
        return True

    @property
    def bin_edges(self) -> tf.Tensor:
        return self.first_edge + tf.concat(
            [[0.0], tf.cumsum(self._bin_increments)], axis=0
        )

    def noise_scale_from_latent(self, noise_latent: tf.Tensor) -> tf.Tensor:
        noise_latent = tf.cast(noise_latent, default_float())
        bounded_offset = self.max_log_scale_deviation * tf.math.tanh(noise_latent)
        return tf.cast(self.sigma, default_float()) * tf.math.exp(bounded_offset)

    def noise_scale(self, X=None) -> tf.Tensor:
        del X
        raise ValueError(
            "The SVGP noise scale requires the second latent GP; "
            "use noise_scale_from_latent()."
        )

    def _class_probabilities(self, F: tf.Tensor) -> tf.Tensor:
        location = F[..., :1]
        sigma = self.noise_scale_from_latent(F[..., 1:2])
        upper_edges = tf.concat(
            [self.bin_edges, tf.constant([np.inf], dtype=default_float())], axis=0
        )
        lower_edges = tf.concat(
            [tf.constant([-np.inf], dtype=default_float()), self.bin_edges], axis=0
        )
        return stable_inv_probit(
            standardize_boundary(upper_edges, location, sigma)
        ) - stable_inv_probit(
            standardize_boundary(lower_edges, location, sigma)
        )

    def _log_prob(self, X, F, Y):
        del X
        labels = to_default_int(Y)
        location = F[..., :1]
        sigma = self.noise_scale_from_latent(F[..., 1:2])
        upper_edges = tf.concat(
            [self.bin_edges, tf.constant([np.inf], dtype=default_float())], axis=0
        )
        lower_edges = tf.concat(
            [tf.constant([-np.inf], dtype=default_float()), self.bin_edges], axis=0
        )
        selected_upper = tf.gather(upper_edges, labels)
        selected_lower = tf.gather(lower_edges, labels)
        probabilities = stable_inv_probit(
            standardize_boundary(selected_upper, location, sigma)
        ) - stable_inv_probit(
            standardize_boundary(selected_lower, location, sigma)
        )
        return tf.squeeze(tf.math.log(probabilities + 1e-6), axis=-1)

    def _conditional_mean(self, X, F):
        del X
        probabilities = self._class_probabilities(F)
        labels = tf.cast(tf.range(self.num_bins), default_float())
        return tf.reduce_sum(probabilities * labels, axis=-1, keepdims=True)

    def _conditional_variance(self, X, F):
        del X
        probabilities = self._class_probabilities(F)
        labels = tf.cast(tf.range(self.num_bins), default_float())
        mean = tf.reduce_sum(probabilities * labels, axis=-1, keepdims=True)
        second_moment = tf.reduce_sum(
            probabilities * tf.square(labels), axis=-1, keepdims=True
        )
        return second_moment - tf.square(mean)
