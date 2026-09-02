import numpy as np
import gpflow


class ToyR:

    def __init__(self, n_train=1000, n_test=100, kern_lengthscale=0.05, seed=0):
        np.random.seed(seed)

        self.n_train = n_train
        self.n_test = n_test
        self.num_data = n_train + n_test
        self.kern_lengthscale = kern_lengthscale
        self.kern = gpflow.kernels.SquaredExponential(lengthscales=kern_lengthscale)
        self.seed = seed

        X, f, y = self.generate_data(n_samples=self.num_data)
        self.X_train, self.f_train, self.Y_train = X[:n_train], f[:n_train], y[:n_train]
        self.X_test, self.f_test, self.Y_test = X[n_train:], f[n_train:], y[n_train:]

    def generate_data(self, n_samples=100):
        X = np.random.rand(n_samples, 1)  # Generate X values
        K = self.kern(X)  # + np.eye(n_samples) * 1e-6
        f = np.random.multivariate_normal(mean=np.zeros(n_samples), cov=K).reshape(
            -1, 1
        )

        # Finally convert f values into ordinal values Y
        Y = np.round((f + f.min()) * 3)
        Y = Y - Y.min()
        Y = np.asarray(Y, np.float64)

        return X, f, Y
