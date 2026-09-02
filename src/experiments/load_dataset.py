import numpy as np
from sklearn.preprocessing import StandardScaler
from data import SICAP, CrowdGleason, SICAP_CLAM, CROWDGLEASON_CLAM
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from omegaconf import DictConfig

BASE_PATH = "/home/fjaviersaezm/data/datasets/"


def ordinally_contaminate_label(label, min_label, max_label):
    l = label
    if label == min_label:
        l += 1
    elif label == max_label:
        l -= 1
    else:
        l += np.random.choice([-1, 1])
    return l


def extra_contaminate_label(label, min_label, max_label):
    all_labels = np.arange(min_label, max_label + 1)
    all_labels = np.delete(all_labels, label)
    return np.expand_dims(np.random.choice(all_labels), axis=0)


def load_dataset(dataset_cfg: DictConfig,
                 seed: int,
                 scaler=None,
                 return_scaler: bool = False):
    """
    Loads a dataset using the provided experimental seed.

    Args:
    - dataset_cfg: DictConfig, dataset configuration containing name and features
    - seed: int, seed used for data selection, generation, noise, and shuffling.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """

    dataset_name = dataset_cfg.name
    features = dataset_cfg.get("features", "default")

    # Handle sicap dataset
    if "sicap" in dataset_name:
        if features == "default":
            # Original sicap dataset
            data_path = f"{BASE_PATH}sicap/features_sicap_norm_{seed}.pickle"
            dataset = SICAP(path=data_path)
            X_train, y_train, X_test, y_test = dataset.get_data()
        else:
            # Features extracted with CLAM
            dataset = SICAP_CLAM(feats=f"features_{features}")
            X_train, y_train, X_test, y_test = (
                dataset.X_train,
                dataset.y_train,
                dataset.X_test,
                dataset.y_test,
            )

        # Introduce noise in labels if specified
        if dataset_cfg.get("noise_level", 0) > 0:
            noise_level = dataset_cfg.noise_level
            n_noisy = int(noise_level * y_train.shape[0] / 100)
            noisy_indices = np.random.choice(y_train.shape[0],
                                             n_noisy,
                                             replace=False)
            cont_fn = (extra_contaminate_label if "extra" in dataset_name else
                       ordinally_contaminate_label)
            y_train[noisy_indices] = [
                cont_fn(label, min(y_train), max(y_train))
                for label in y_train[noisy_indices]
            ]
            print(
                f"Introduced noise in {n_noisy} labels ({noise_level}%) with seed {seed}."
            )
    # Handle crowdgleason dataset
    elif "crowdgleason" in dataset_name:
        print(dataset_name)
        if features == "default":
            # Original crowdgleason dataset
            data_path = f"{BASE_PATH}crowdgleason/features_grx_norm_{seed}.pickle"
            dataset = CrowdGleason(path=data_path)
            X_train, y_train, X_test, y_test = dataset.get_data(consensus=True)
        else:
            # Features extracted with CLAM
            labels = dataset_cfg.get("labels", "MV")
            labels = "y_train.npy" if labels == "MV" else "y_train_EM.npy"
            dataset = CROWDGLEASON_CLAM(feats=f"features_{features}",
                                        y_train_name=labels)

            X_train, y_train, X_test, y_test = (
                dataset.X_train,
                dataset.y_train,
                dataset.X_test,
                dataset.y_test,
            )
    elif dataset_name == "toyR":
        from data.toy_R import ToyR

        dataset = ToyR(n_train=100, n_test=20, seed=seed)
        X_train, y_train = dataset.X_train, dataset.Y_train
        X_test, y_test = dataset.X_test, dataset.Y_test

    elif dataset_name == "toy_overlap":
        from data import ToyOverlap

        dataset = ToyOverlap(n_train=100, n_test=20, seed=seed)
        X_train, y_train = dataset.X_train, dataset.Y_train
        X_test, y_test = dataset.X_test, dataset.Y_test
    elif dataset_name == "toy_rankeable":
        from data import ToyRankeable

        dataset = ToyRankeable(n_train=100, n_test=20, seed=seed)
        X_train, y_train = dataset.X_train, dataset.Y_train
        X_test, y_test = dataset.X_test, dataset.Y_test
    elif dataset_name == "toy_non_rankable":
        from data import ToyNonRankable

        dataset = ToyNonRankable(n_train=100, n_test=20, seed=seed)
        X_train, y_train = dataset.X_train, dataset.Y_train
        X_test, y_test = dataset.X_test, dataset.Y_test
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}.")

    # Data shuffling
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    # Data scaling
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    percentage = dataset_cfg.get("percentage", 100)
    if percentage < 100:
        n_samples = int(percentage * X_train.shape[0] / 100)
        X_train = X_train[:n_samples]
        y_train = y_train[:n_samples]

    if return_scaler:
        return X_train, y_train, X_test, y_test, scaler

    return X_train, y_train, X_test, y_test
