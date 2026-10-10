"""A structured but globally non-rankable companion to ``ToyRankeable``."""

import numpy as np


class ToyNonRankable:
    """Generate four compact intervals with a non-monotonic class ordering.

    The construction matches the rankable toy's one-dimensional, separated
    class regions, but assigns the left-to-right zero-based label order
    ``1, 3, 0, 2`` (classes ``2, 4, 1, 3`` in the paper's one-based plot).
    Thus, each class remains a visually coherent block and both local pairs
    increase, while the complete sequence is globally non-monotonic. With
    balanced sampling, the held-out Spearman rankability is exactly zero.
    """

    _NUM_CLASSES = 4
    _CLASS_ORDER = np.array([1, 3, 0, 2], dtype=int)
    _SEGMENT_MARGIN = 0.08

    def __init__(self, n_train=1000, n_test=100, seed=0):
        if n_train < 1 or n_test < 1:
            raise ValueError("n_train and n_test must be positive")

        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.num_classes = self._NUM_CLASSES
        self._rng = np.random.RandomState(self.seed)

        self.X_train, self.Y_train = self._generate_split(self.n_train)
        self.X_test, self.Y_test = self._generate_split(self.n_test)

    def _generate_split(self, n_samples):
        """Sample evenly from the four permuted class intervals."""
        num_segments = self.num_classes
        segments = np.tile(
            np.arange(num_segments),
            int(np.ceil(n_samples / num_segments)),
        )[:n_samples]
        self._rng.shuffle(segments)

        # Keep a small gap at each boundary so the four compact class regions
        # remain visually distinct, as in the rankable companion example.
        within_segment = self._rng.uniform(
            self._SEGMENT_MARGIN,
            1.0 - self._SEGMENT_MARGIN,
            size=n_samples,
        )
        features = (segments + within_segment) / num_segments
        labels = self._CLASS_ORDER[segments]

        permutation = self._rng.permutation(n_samples)
        return features[permutation, None], labels[permutation]
