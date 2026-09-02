import numpy as np
import os


class CLAM_DATASET:

    def __init__(
        self,
        path="/data/data_fjaviersaezm/crowdgleason/",
        feats="features_UNI",
        X_train_name="X_train.npy",
        y_train_name="y_train.npy",
        X_test_name="X_test.npy",
        y_test_name="y_test.npy",
    ):

        self.path = path
        self.feats = feats
        load_path = self.path + self.feats + "/"
        self.X_train = np.load(os.path.join(load_path, X_train_name)).astype(np.float64)
        self.y_train = (
            np.load(os.path.join(load_path, y_train_name))
            .astype(np.int64)
            .reshape(-1, 1)
        )
        self.X_test = np.load(os.path.join(load_path, X_test_name)).astype(np.float64)
        self.y_test = (
            np.load(os.path.join(load_path, y_test_name))
            .astype(np.int64)
            .reshape(-1, 1)
        )
        self.train_names = np.load(os.path.join(load_path, "train_names.npy"))
        self.test_names = np.load(os.path.join(load_path, "test_names.npy"))
