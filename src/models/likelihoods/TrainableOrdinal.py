import gpflow
import tensorflow as tf
import numpy as np
from gpflow.base import Parameter
from gpflow.utilities import positive, to_default_int
from gpflow.likelihoods import ScalarLikelihood
from gpflow.likelihoods.utils import inv_probit


def default_float():
    return tf.float64


def stable_inv_probit(value: tf.Tensor) -> tf.Tensor:
    """Evaluate GPflow's inverse probit without sending infinities to ``erf``."""
    value = tf.convert_to_tensor(value, dtype=default_float())
    finite_value = tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))
    probabilities = inv_probit(finite_value)
    probabilities = tf.where(
        tf.math.is_inf(value) & (value > 0),
        tf.cast(1.0 - 1e-3, default_float()),
        probabilities,
    )
    return tf.where(
        tf.math.is_inf(value) & (value < 0),
        tf.cast(1e-3, default_float()),
        probabilities,
    )


def standardize_boundary(
    boundary: tf.Tensor, latent: tf.Tensor, sigma: tf.Tensor
) -> tf.Tensor:
    """Compute ``(boundary - latent) / sigma`` with safe infinite endpoints.

    Replacing the infinite endpoints before the arithmetic avoids undefined
    ``inf / sigma`` gradients on CPU. The endpoints are restored afterwards,
    before evaluating the inverse probit.
    """
    boundary = tf.convert_to_tensor(boundary, dtype=default_float())
    finite_boundary = tf.where(
        tf.math.is_finite(boundary), boundary, tf.zeros_like(boundary)
    )
    standardized = (finite_boundary - latent) / sigma
    return tf.where(
        tf.math.is_inf(boundary),
        tf.broadcast_to(boundary, tf.shape(standardized)),
        standardized,
    )


class TrainableOrdinal(ScalarLikelihood):
    """
    Ordinal likelihood with trainable and ordered bin edges.
    """

    def __init__(self, initial_bin_edges: np.ndarray, initial_sigma: float = 1.0, **kwargs):
        """
        Args:
            initial_bin_edges: numpy array specifying where the output label
                switches. For labels 0...K, this has size (K-1).
        """
        super().__init__(**kwargs)
        self.num_bins = initial_bin_edges.size + 1

        # Fix the first bin edge
        self.first_edge = initial_bin_edges[0]

        # Trainable increments: positive differences between consecutive edges
        init_increments = np.diff(initial_bin_edges)
        self._bin_increments = Parameter(init_increments, transform=positive())

        # Trainable observation noise scale
        self.sigma = Parameter(initial_sigma, transform=positive())

    @property
    def is_heteroscedastic(self) -> bool:
        return False

    @property
    def heteroscedastic_predictor(self) -> str:
        return "none"

    @property
    def uses_latent_noise(self) -> bool:
        return False

    def noise_scale(self, X=None) -> tf.Tensor:
        """Return the likelihood standard deviation for each observation.

        ``X`` is accepted to provide a shared interface with input-dependent
        ordinal likelihoods. The homoscedastic implementation intentionally
        ignores it and returns the existing scalar parameter.
        """
        del X
        return tf.convert_to_tensor(self.sigma, dtype=default_float())

    @property
    def bin_edges(self) -> tf.Tensor:
        """
        Returns monotonically increasing bin edges, computed from trainable increments.
        """
        # Ordered edges: first_edge + cumulative sum of positive increments
        return self.first_edge + tf.concat(
            [[0.0], tf.cumsum(self._bin_increments)], axis=0
        )

    def _scalar_log_prob(self, X, F, Y):
        Y = to_default_int(Y)

        sigma = self.noise_scale(X)

        scaled_bins_left = tf.concat(
            [self.bin_edges, tf.constant([np.inf], dtype=default_float())],
            axis=0,
        )
        scaled_bins_right = tf.concat(
            [
                tf.constant([-np.inf], dtype=default_float()),
                self.bin_edges,
            ],
            axis=0,
        )

        selected_bins_left = tf.gather(scaled_bins_left, Y)
        selected_bins_right = tf.gather(scaled_bins_right, Y)

        probs = stable_inv_probit(
            standardize_boundary(selected_bins_left, F, sigma)
        ) - stable_inv_probit(
            standardize_boundary(selected_bins_right, F, sigma)
        )

        return tf.math.log(probs + 1e-6)

    def _make_phi(self, F, X=None):
        sigma = self.noise_scale(X)
        scaled_bins_left = tf.concat(
            [self.bin_edges, tf.constant([np.inf], dtype=default_float())],
            axis=0,
        )
        scaled_bins_right = tf.concat(
            [
                tf.constant([-np.inf], dtype=default_float()),
                self.bin_edges,
            ],
            axis=0,
        )

        F_flat = tf.reshape(F, (-1, 1))
        return stable_inv_probit(
            standardize_boundary(scaled_bins_left, F_flat, sigma)
        ) - stable_inv_probit(
            standardize_boundary(scaled_bins_right, F_flat, sigma)
        )

    def _conditional_mean(self, X, F):
        phi = self._make_phi(F, X)
        Ys = tf.reshape(tf.range(self.num_bins, dtype=default_float()), (-1, 1))
        return tf.reshape(tf.linalg.matmul(phi, Ys), tf.shape(F))

    def _conditional_variance(self, X, F):
        phi = self._make_phi(F, X)
        Ys = tf.reshape(tf.range(self.num_bins, dtype=default_float()), (-1, 1))
        E_y = phi @ Ys
        E_y2 = phi @ (Ys**2)
        return tf.reshape(E_y2 - E_y**2, tf.shape(F))
