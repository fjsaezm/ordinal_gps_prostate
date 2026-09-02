"""Input-dependent ordinal-probit likelihood with a deterministic MLP scale."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from gpflow.utilities import set_trainable

from .TrainableOrdinal import TrainableOrdinal, default_float


class NoiseScaleMLP(tf.keras.Model):
    """Small MLP producing a bounded log-scale offset.

    The output layer is initialized to zero, so every input starts with the
    configured baseline scale. Bounding the log-scale offset prevents the
    noise head from making the ordinal likelihood arbitrarily flat or sharp.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (32,),
        activations: Sequence[str] = ("tanh",),
        max_log_scale_deviation: float = float(np.log(4.0)),
        name: str = "noise_scale_mlp",
    ) -> None:
        super().__init__(name=name, dtype=default_float())
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        activations = tuple(activations)
        if len(hidden_dims) != len(activations):
            raise ValueError(
                "heteroscedastic hidden_dims and activations must have equal length"
            )
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("heteroscedastic hidden dimensions must be positive")
        if max_log_scale_deviation <= 0:
            raise ValueError("max_log_scale_deviation must be positive")

        self.max_log_scale_deviation = float(max_log_scale_deviation)
        self.hidden_layers = [
            tf.keras.layers.Dense(
                dim,
                activation=None if activation in (None, "None") else activation,
                dtype=default_float(),
                name=f"hidden_{index}",
            )
            for index, (dim, activation) in enumerate(zip(hidden_dims, activations))
        ]
        self.output_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            dtype=default_float(),
            name="log_scale_offset",
        )

    def call(self, inputs, training=None):
        del training
        outputs = tf.cast(inputs, default_float())
        for layer in self.hidden_layers:
            outputs = layer(outputs)
        raw_offset = self.output_layer(outputs)
        return self.max_log_scale_deviation * tf.math.tanh(raw_offset)


class HeteroscedasticOrdinal(TrainableOrdinal):
    """Ordinal likelihood whose positive scale is predicted from each input.

    The thresholds are inherited from :class:`TrainableOrdinal`. For input
    ``x``, the likelihood scale is

    ``baseline_sigma * exp(noise_scale_mlp(x))``.

    ``baseline_sigma`` is fixed to anchor the scale and keep optimization
    identifiable. The MLP and ordered threshold increments remain trainable.
    """

    def __init__(
        self,
        initial_bin_edges: np.ndarray,
        hidden_dims: Sequence[int] = (32,),
        activations: Sequence[str] = ("tanh",),
        baseline_sigma: float = 1.0,
        max_log_scale_deviation: float = float(np.log(4.0)),
        **kwargs,
    ) -> None:
        if baseline_sigma <= 0:
            raise ValueError("baseline_sigma must be positive")
        super().__init__(
            initial_bin_edges=initial_bin_edges,
            initial_sigma=baseline_sigma,
            **kwargs,
        )
        set_trainable(self.sigma, False)
        self.noise_scale_mlp = NoiseScaleMLP(
            hidden_dims=hidden_dims,
            activations=activations,
            max_log_scale_deviation=max_log_scale_deviation,
        )

    @property
    def is_heteroscedastic(self) -> bool:
        return True

    @property
    def heteroscedastic_predictor(self) -> str:
        return "mlp"

    @property
    def uses_latent_noise(self) -> bool:
        return False

    @property
    def additional_trainable_variables(self):
        """Keras variables that GPflow's module traversal does not discover."""
        return tuple(self.noise_scale_mlp.trainable_variables)

    def noise_scale(self, X=None) -> tf.Tensor:
        if X is None:
            raise ValueError("HeteroscedasticOrdinal requires input features X")
        log_scale_offset = self.noise_scale_mlp(X, training=False)
        baseline = tf.cast(self.sigma, default_float())
        return baseline * tf.math.exp(log_scale_offset)
