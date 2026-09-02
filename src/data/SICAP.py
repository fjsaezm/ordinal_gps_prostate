import pickle
import numpy as np


class SICAP:

    def __init__(
        self,
        path="/home/fjaviersaezm/data/datasets/sicap/features_sicap_norm_0.pickle",
    ):
        self.path = path
        with open(self.path, "rb") as fp:
            sicap_data = pickle.load(fp)

        print(sicap_data["train"].keys())

        train, test, val = sicap_data["train"], sicap_data["test"], sicap_data["val"]
        self.X_train, self.y_train, self.train_names = self._unpack_data(train)
        self.X_val, self.y_val, self.val_names = self._unpack_data(val)
        self.X_test, self.y_test, self.test_names = self._unpack_data(test)

    def get_data(self):
        """
        Returns the dataset.
        """
        return (
            self.X_train.astype(np.float64),
            np.array(self.y_train, dtype=np.int64).reshape(-1, 1),
            self.X_test.astype(np.float64),
            np.array(self.y_test, dtype=np.int64).reshape(-1, 1),
        )

    def _unpack_data(self, data):
        """
        Unpacks the data from the dataset dictionary.
        """
        return data["features"], data["label_list"], data["names"]


if __name__ == "__main__":
    import time

    start_time = time.time()
    sicap = SICAP()
    X_train, y_train, X_test, y_test = sicap.get_data()
    print("Number of training samples:", X_train.shape[0])
    print("Number of test samples:", X_test.shape[0])
    print("Data shape", X_train[0].shape)
    print("First three names:", sicap.train_names[:3])
    print("Time taken to load data:", time.time() - start_time)
