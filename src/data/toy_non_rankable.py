"""A one-dimensional toy representation with zero ordinal rankability."""

import numpy as np


class ToyNonRankable:
    """Generate a symmetric, folded ordinal representation.

    Moving from left to right, the class order is
    ``0, 1, 2, 3, 3, 2, 1, 0``.  Samples are generated in mirrored pairs
    ``(x, 1 - x)`` with the same label.  Consequently, the Spearman
    correlation between the feature and label is zero (up to floating-point
    precision), for either orientation of the one-dimensional projection.
    """

    _SEGMENT_LABELS = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=int)

    def __init__(self, n_train=1000, n_test=100, seed=0):
        if n_train < 1 or n_test < 1:
            raise ValueError("n_train and n_test must be positive")

        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.num_classes = 4
        self._rng = np.random.RandomState(self.seed)

        self.X_train, self.Y_train = self._generate_split(self.n_train)
        self.X_test, self.Y_test = self._generate_split(self.n_test)

    @classmethod
    def _labels_from_features(cls, features):
        segment = np.minimum((features[:, 0] * 8).astype(int), 7)
        return cls._SEGMENT_LABELS[segment]

    def _generate_split(self, n_samples):
        """Generate symmetric pairs so finite samples retain zero correlation."""
        n_pairs = n_samples // 2
        left = self._rng.uniform(0.0, 0.5, size=n_pairs)
        features = np.concatenate([left, 1.0 - left])

        if n_samples % 2:
            features = np.concatenate([features, np.array([0.5])])

        X = features.reshape(-1, 1)
        Y = self._labels_from_features(X)
        permutation = self._rng.permutation(n_samples)
        return X[permutation], Y[permutation]
