import tensorflow as tf
from typing import Any


class DeepKernelLearning(tf.keras.Model):
    """
    Deep Kernel Learning model that combines a neural network feature extractor
    with a Gaussian Process for classification/regression.

    The architecture consists of:
    1. Pre-initialized MLP feature extractor that learns representations
    2. Pre-initialized Gaussian Process model for final predictions

    Args:
        mlp_model: Pre-initialized MLP model for feature extraction
        gp_model: Pre-initialized GPflow model (e.g., SVGP, GPR, etc.)
        name: Name of the model (default: 'deep_kernel_learning')

    """

    def __init__(
        self,
        mlp_model: tf.keras.Model,
        gp_model: Any,
        name: str = "deep_kernel_learning",
        **kwargs,
    ):
        super(DeepKernelLearning, self).__init__(name=name, **kwargs)

        # Store pre-initialized models
        self.feature_extractor = mlp_model
        self.gp_model = gp_model
        self.likelihood = gp_model.likelihood
        self.num_data = gp_model.num_data
        self.training = True

    def call(self, inputs, return_features=False, **kwargs):
        """
        Forward pass through the Deep Kernel Learning model.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim)
            training: Boolean indicating training mode
            return_features: If True, return both predictions and features

        Returns:
            If return_features=False: GP predictions (mean, var) or logits
            If return_features=True: (predictions, features)
        """
        # Extract features using MLP
        features = self.feature_extractor(inputs, training=self.training)

        # Get GP predictions
        mean, var = self.gp_model.predict_f(features)

        if return_features:
            return (mean, var), features
        else:
            return mean, var

    def predict_f(self, inputs, **kwargs):
        """
        Predict function values (latent function).

        Args:
            inputs: Input tensor

        Returns:
            Tuple of (mean, variance) predictions
        """
        features = self.feature_extractor(inputs, training=self.training)
        return self.gp_model.predict_f(features, **kwargs)

    def predict_y(self, inputs, **kwargs):
        """
        Predict output values (through likelihood).

        Args:
            inputs: Input tensor

        Returns:
            Predictions through the likelihood function
        """
        features = self.feature_extractor(inputs, training=False)
        return self.gp_model.predict_y(features, **kwargs)

    def prior_kl(self):
        """
        Compute the prior KL divergence from the GP model.

        Returns:
            Prior KL divergence value
        """
        return self.gp_model.prior_kl()

    @property
    def trainable_variables(self):
        return list(self.feature_extractor.trainable_variables) + list(
            self.gp_model.trainable_variables
        )

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
