import tensorflow as tf
from typing import List, Union, Optional
import numpy as np
from .likelihoods.TrainableOrdinal import TrainableOrdinal


def inv_probit(x):
    """Inverse probit function (standard normal CDF)."""
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


class MLP(tf.keras.Model):
    """
    A Multi-Layer Perceptron implementation in TensorFlow.

    Args:
        inner_dims: List of integers specifying the number of units in each hidden layer
        activations: List of activation functions for each layer. Can be strings or callable functions
        output_dim: Number of output units (default: None, uses last inner_dim)
        output_activation: Activation function for output layer (default: None)
        use_bias: Whether to use bias in dense layers (default: True)
        dropout_rate: Dropout rate between layers (default: 0.0, no dropout)
        name: Name of the model (default: 'mlp')

    """

    def __init__(
        self,
        inner_dims: List[int],
        activations: List[Union[str, callable]],
        output_dim: Optional[int] = None,
        output_activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        name: str = "mlp",
        **kwargs,
    ):
        super(MLP, self).__init__(name=name, **kwargs)

        # Validation
        if len(inner_dims) != len(activations):
            raise ValueError(
                f"Length of inner_dims ({len(inner_dims)}) must match length of activations ({len(activations)})"
            )

        if not inner_dims:
            raise ValueError("inner_dims cannot be empty")

        self.inner_dims = inner_dims
        self.activations = activations
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

        # Build layers
        self.hidden_layers = []
        self.dropout_layers = []

        # Hidden layers
        for i, (dim, activation) in enumerate(zip(inner_dims, activations)):
            # Dense layer
            act = activation if activation != "None" else None
            layer = tf.keras.layers.Dense(
                units=dim,
                activation=act,
                use_bias=use_bias,
                name=f"dense_{i}",
                dtype=tf.float64,
            )
            self.hidden_layers.append(layer)

            # Add dropout layer if dropout_rate > 0
            if dropout_rate > 0.0:
                dropout_layer = tf.keras.layers.Dropout(
                    rate=dropout_rate, name=f"dropout_{i}", dtype=tf.float64
                )
                self.dropout_layers.append(dropout_layer)
            else:
                self.dropout_layers.append(None)

        # Output layer (optional)
        if output_dim is not None:
            self.output_layer = tf.keras.layers.Dense(
                units=output_dim,
                activation=output_activation,
                use_bias=use_bias,
                name="output",
            )
        else:
            self.output_layer = None

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass through the MLP.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim)
            training: Boolean indicating training mode (for dropout)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = inputs

        # Forward through hidden layers with dropout
        for layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = layer(x)
            # Apply dropout if layer exists and we're in training mode
            if dropout_layer is not None:
                x = dropout_layer(x, training=training)

        # Apply output layer if specified (no dropout after output layer)
        if self.output_layer is not None:
            x = self.output_layer(x)

        return x

    def get_config(self):
        """Return the config of the model for serialization."""
        config = super(MLP, self).get_config()
        config.update(
            {
                "inner_dims": self.inner_dims,
                "activations": self.activations,
                "output_dim": self.output_dim,
                "output_activation": self.output_activation,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config."""
        return cls(**config)

    def summary_custom(self):
        """Print a custom summary of the MLP architecture."""
        print(f"\n{self.name.upper()} Architecture:")
        print("=" * 50)

        total_params = 0

        # Hidden layers
        for i, (dim, activation) in enumerate(zip(self.inner_dims, self.activations)):
            layer = self.hidden_layers[i]
            params = layer.count_params()
            total_params += params

            print(f"Dense Layer {i+1}: {dim} units, activation={activation}")
            print(f"  Parameters: {params:,}")

            # Show dropout info
            if self.dropout_rate > 0.0:
                print(f"  Dropout: {self.dropout_rate}")

            if self.dropout_layers[i] is not None:
                dropout_params = self.dropout_layers[i].count_params()
                total_params += dropout_params

        # Output layer
        if self.output_layer is not None:
            params = self.output_layer.count_params()
            total_params += params
            print(
                f"Output Layer: {self.output_dim} units, activation={self.output_activation}"
            )
            print(f"  Parameters: {params:,}")
        else:
            print("No output layer (uses last hidden layer)")

        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print("=" * 50)


class OrdinalMLP(tf.keras.Model):
    """
    Multi-Layer Perceptron for ordinal regression using ordinal likelihood.
    """

    def __init__(
        self,
        inner_dims: List[int],
        activations: List[Union[str, callable]],
        bin_edges: np.ndarray,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        name: str = "ordinal_mlp",
        **kwargs,
    ):
        super(OrdinalMLP, self).__init__(name=name, **kwargs)

        # Validation
        if len(inner_dims) != len(activations):
            raise ValueError(
                f"Length of inner_dims ({len(inner_dims)}) must match length of activations ({len(activations)})"
            )

        if not inner_dims:
            raise ValueError("inner_dims cannot be empty")

        self.inner_dims = inner_dims
        self.activations = activations
        self.dropout_rate = dropout_rate

        # Build hidden layers
        self.hidden_layers = []
        self.dropout_layers = []

        for i, (dim, activation) in enumerate(zip(inner_dims, activations)):
            act = activation if activation != "None" else None
            layer = tf.keras.layers.Dense(
                units=dim,
                activation=act,
                use_bias=use_bias,
                name=f"dense_{i}",
                dtype=tf.float64,
            )
            self.hidden_layers.append(layer)

            if dropout_rate > 0.0:
                dropout_layer = tf.keras.layers.Dropout(
                    rate=dropout_rate, name=f"dropout_{i}", dtype=tf.float64
                )
                self.dropout_layers.append(dropout_layer)
            else:
                self.dropout_layers.append(None)

        # Output layer: single logit for ordinal regression
        self.output_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,  # No activation for logit
            use_bias=use_bias,
            name="output",
            dtype=tf.float64,
        )

        # Ordinal likelihood - use the same TrainableOrdinal as GP models
        self.ordinal_likelihood = TrainableOrdinal(bin_edges)

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass through the OrdinalMLP.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim)
            training: Boolean indicating training mode (for dropout)

        Returns:
            Logits of shape (batch_size, 1)
        """
        x = inputs

        # Forward through hidden layers with dropout
        for layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = layer(x)
            if dropout_layer is not None:
                x = dropout_layer(x, training=training)

        # Output logit
        logits = self.output_layer(x)
        return logits

    def train_step(self, data):
        """
        Custom training step for ordinal regression.

        Args:
            data: Tuple of (inputs, labels)

        Returns:
            Dict with loss and metrics
        """
        x, y = data

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            # Use TrainableOrdinal._scalar_log_prob directly
            log_probs = self.ordinal_likelihood._scalar_log_prob(None, logits, y)
            loss = -tf.reduce_mean(log_probs)

        # Get all trainable variables (model + likelihood parameters)
        trainable_vars = self.trainable_variables + list(self.ordinal_likelihood.trainable_variables)
        gradients = tape.gradient(loss, trainable_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute metrics using _make_phi
        probs = self.ordinal_likelihood._make_phi(logits)
        predictions = tf.argmax(probs, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y, tf.int64)), tf.float32))

        return {"loss": loss, "accuracy": accuracy}

    def test_step(self, data):
        """
        Custom test step for ordinal regression.

        Args:
            data: Tuple of (inputs, labels)

        Returns:
            Dict with loss and metrics
        """
        x, y = data

        logits = self(x, training=False)
        # Use TrainableOrdinal._scalar_log_prob directly
        log_probs = self.ordinal_likelihood._scalar_log_prob(None, logits, y)
        loss = -tf.reduce_mean(log_probs)

        # Compute metrics using _make_phi
        probs = self.ordinal_likelihood._make_phi(logits)
        predictions = tf.argmax(probs, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y, tf.int64)), tf.float32))

        return {"loss": loss, "accuracy": accuracy}

    def predict_probabilities(self, inputs):
        """
        Predict class probabilities.

        Args:
            inputs: Input tensor

        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self(inputs, training=False)
        return self.ordinal_likelihood._make_phi(logits)

    def predict_classes(self, inputs):
        """
        Predict class labels.

        Args:
            inputs: Input tensor

        Returns:
            Predicted classes of shape (batch_size,)
        """
        probs = self.predict_probabilities(inputs)
        return tf.argmax(probs, axis=1)

    def summary_custom(self):
        """Print a custom summary of the OrdinalMLP architecture."""
        print(f"\n{self.name.upper()} Architecture:")
        print("=" * 50)

        total_params = 0

        # Hidden layers
        for i, (dim, activation) in enumerate(zip(self.inner_dims, self.activations)):
            layer = self.hidden_layers[i]
            params = layer.count_params()
            total_params += params

            print(f"Dense Layer {i+1}: {dim} units, activation={activation}")
            print(f"  Parameters: {params:,}")

            if self.dropout_rate > 0.0:
                print(f"  Dropout: {self.dropout_rate}")

        # Output layer
        params = self.output_layer.count_params()
        total_params += params
        print(f"Output Layer: 1 unit (logit), activation=None")
        print(f"  Parameters: {params:,}")

        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Number of Classes: {self.ordinal_likelihood.num_bins}")
        print("=" * 50)

