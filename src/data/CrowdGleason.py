import numpy as np
import pickle

# Funciones para desempaquetar los datos


class CrowdGleason:

    def __init__(
        self,
        path="~/data/datasets/crowdgleason/features_grx_norm_0.pickle",
    ):

        self.path = path

        with open(self.path, "rb") as fp:
            grx_data = pickle.load(fp)

        # Load data
        train, test, val = grx_data["train"], grx_data["test"], grx_data["val"]

        # Unpack
        self.X_train, self.y_train, self.train_names, self.train_MV, self.train_EM = (
            self._unpack_data_grx_train(train)
        )

        # Decision! y_train is Majority Vote (MV) labels
        # Decision test!
        # self.y_train = self.train_MV

        self.X_val, self.y_val, self.val_names, self.val_MV, self.val_EM = (
            self._unpack_data_grx_train(val)
        )
        (
            self.X_test,
            self.y_test,
            self.test_names,
            self.test_MV,
            self.test_EM,
            self.test_GT,
        ) = self._unpack_data_grx_test(test)

        self.X_train = np.concatenate([self.X_train, self.X_val])
        self.y_train = np.concatenate([self.train_EM, self.val_EM])

        print(len(self.y_train))

        print("---")

        # Filter consensus labels
        self.index_consensus_test = np.array(self.test_MV) == np.array(self.test_GT)

        # Filter consensus data
        self.X_test_con = self.X_test[self.index_consensus_test]
        self.y_test_con = np.array(self.test_GT)[self.index_consensus_test]

    def get_data(self, consensus=True):
        """
        Returns the dataset based on the consensus flag.
        Args:
            consensus (bool): If True, returns only the test data in consensus.
        """
        if consensus:
            return (
                self.X_train.astype(np.float64),
                np.array(self.y_train, dtype=np.int64).reshape(-1, 1),
                self.X_test_con.astype(np.float64),
                np.array(self.y_test_con, dtype=np.int64).reshape(-1, 1),
            )
        else:
            return (
                self.X_train.astype(np.float64),
                np.array(self.y_train, dtype=np.int64).reshape(-1, 1),
                self.X_test.astype(np.float64),
                np.array(self.y_test, dtype=np.int64).reshape(-1, 1),
            )

    def _unpack_data_grx_train(self, data):
        return (
            data["features"],
            np.array(data["label_list"]),
            data["names"],
            data["MV"],
            data["EM"],
        )

    def _unpack_data_grx_test(self, data):
        return (
            data["features"],
            np.array(data["label_list"]),
            data["names"],
            data["MV"],
            data["EM"],
            data["GT"],
        )
