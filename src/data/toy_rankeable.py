import numpy as np
from sklearn.utils import shuffle


class ToyRankeable:

    def __init__(
        self,
        n_train=1000,
        n_test=100,
        seed=0,
    ):

        self.n_train = n_train
        self.n_test = n_test
        self.train_intervals = [
            (0.0, 0.3, 0),
            (0.3, 0.45, 1),
            (0.45, 0.7, 2),
            (0.7, 1.0, 3),
        ]

        self.test_intervals = self.train_intervals.copy()

        self.seed = seed

        np.random.seed(seed)

        # Infer number of classes
        all_labels = [lbl for _, _, lbl in self.train_intervals + self.test_intervals]
        self.num_classes = len(np.unique(all_labels))

        self.X_train, self.Y_train = self._generate_split(n_train, self.train_intervals)
        self.X_test, self.Y_test = self._generate_split(n_test, self.test_intervals)

    def _generate_split(self, n_samples, intervals):
        """
        For each sample:
          1. Sample an interval
          2. Sample x from that interval
          3. Assign label
        """

        intervals = np.array(intervals, dtype=object)
        starts = intervals[:, 0].astype(float)
        ends = intervals[:, 1].astype(float)
        labels = intervals[:, 2].astype(int)

        # Interval sampling probabilities (∝ length)
        lengths = ends - starts
        probs = lengths / lengths.sum()

        X = np.zeros((n_samples, 1))
        Y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            idx = np.random.choice(len(intervals), p=probs)
            X[i, 0] = np.random.uniform(starts[idx], ends[idx])
            Y[i] = labels[idx]

        X, Y = shuffle(X, Y, random_state=self.seed)
        return X, Y
