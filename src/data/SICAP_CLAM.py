import pickle
import os
import numpy as np
from .CLAM_DATASET import CLAM_DATASET


class SICAP_CLAM(CLAM_DATASET):

    def __init__(self, path="/data/data_fjaviersaezm/SICAPv2/", feats="features_UNI"):
        super().__init__(path=path, feats=feats)


if __name__ == "__main__":
    import time

    start_time = time.time()
    sicap = SICAP_CLAM()
    X_train, y_train, X_test, y_test = (
        sicap.X_train,
        sicap.y_train,
        sicap.X_test,
        sicap.y_test,
    )

    print("Number of training samples:", X_train.shape[0])
    print("Number of test samples:", X_test.shape[0])
    print("Data shape", X_train.shape)
    print("Test data shape", X_test.shape)
    print("Time taken to load data:", time.time() - start_time)
