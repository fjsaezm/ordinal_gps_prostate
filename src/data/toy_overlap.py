import numpy as np
from sklearn.utils import shuffle


class ToyOverlap:

    def __init__(
        self,
        n_train=1000,
        n_test=100,
        seed=0,
    ):

        self.n_train = n_train
        self.n_test = n_test
        self.train_intervals = [
            (0.0, 0.25, 0),
            (0.225, 0.4, 3),
            (0.375, 0.5, 1),
            (0.475, 0.65, 2),
            #(0.625, 0.8, 3),
            #(0.775, 1.0, 0),
        ]
        self.test_intervals = [
            (0.0, 0.25, 0),
            (0.25, 0.4, 3),
            (0.4, 0.5, 1),
            (0.5, 0.65, 2),
            #(0.65, 0.8, 3),
            #(0.8, 1.0, 0),
        ]



        self.seed = seed

        np.random.seed(seed)

        # Infer number of classes
        all_labels = [
            lbl for _, _, lbl in self.train_intervals + self.test_intervals
        ]
        self.num_classes = len(np.unique(all_labels))

        self.X_train, self.Y_train = self._generate_split(
            n_train, self.train_intervals)
        self.X_test, self.Y_test = self._generate_split(
            n_test, self.test_intervals)

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


class ToyOverlapOLD:

    def __init__(self, n_train=1000, n_test=100, seed=0):
        np.random.seed(seed)

        self.n_train = n_train
        self.n_test = n_test
        self.num_data = n_train + n_test
        self.seed = seed
        # Data generation parameters
        self.centers = np.array([0.15, 0.35, 0.5, 0.65, 0.8, 0.9])
        self.class_labels = np.array([0, 3, 1, 2, 3, 0])
        self.num_classes = len(np.unique(self.class_labels))
        self.spreads = np.array([0.07, 0.05, 0.05, 0.05, 0.05, 0.07])

        X, y = self.generate_data(n_samples=self.num_data)
        self.X_train, self.Y_train = X[:n_train], y[:n_train]
        self.X_test, self.Y_test = X[n_train:], y[n_train:]

    def generate_data(self, n_samples=100):

        X = np.linspace(0, 1, n_samples).reshape(-1, 1)

        logits = np.zeros((n_samples, self.num_classes))
        for c_idx, c in enumerate(self.class_labels):
            logits[:, c] += np.exp(
                -0.5 * ((X[:, 0] - self.centers[c_idx]) / self.spreads[c_idx]) ** 2
            )

        # Normalize into probabilities and sample classes
        probs = logits / np.sum(logits, axis=1, keepdims=True)
        Y = np.array([np.random.choice(self.num_classes, p=p) for p in probs])
        X, Y = shuffle(X, Y, random_state=self.seed)
        return X, Y.astype(int)
