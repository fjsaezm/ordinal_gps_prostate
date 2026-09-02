# Python
import os
import hashlib

# Tensorflow
import tensorflow as tf

# Gpflow
import gpflow

# Standard
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans

# custom

# from utils_evaluation import evaluation_step_regression_DSVI_DGP
import copy
import json
from .SVGP_base import SVGP_base
import argparse


cg = {"dtype": tf.float64, "seed": 0}
cg = argparse.Namespace(**cg)

# ========================================================================================
# Interface to create GPflow inducing points classes


# Get default kernel hyperparameters
def get_kernel_hyperparams(init_params):
    params = {
        "ls": 1.0,  # length scale
        "os": 1.0,  # output scale
        "linear_param": 1.0,  # parameter of linear kernel
        "p": 1.0,  # parameter for the diagonal in the P matrix in the neuralnetwork nonstationary kernel
    }

    for k, v in init_params.items():
        params[k] = v

    return params


def wrap_inducing_points(Z, how, num_multioutput, transformation=None):

    # case in which we have just one GP process to model
    if how == "single_process":
        Z = gpflow.inducing_variables.InducingPoints(Z, transformation=transformation)

    # use a separate set of independent inducing points ( K O(N**3)  computations )
    elif how == "latent_separate_independent":
        Z = [
            gpflow.inducing_variables.InducingPoints(Z, transformation=transformation)
            for i in range(num_multioutput)
        ]
        Z = gpflow.inducing_variables.SeparateIndependentInducingVariables(Z)

    # use the same set of inducing points shared by all latent process ( O(N**3) computations )
    elif how == "latent_shared_independent":
        # Z = gpflow.inducing_variables.InducingPoints(Z, transformation=transformation)
        Z = gpflow.inducing_variables.InducingPoints(Z)
        Z = gpflow.inducing_variables.SharedIndependentInducingVariables(Z)

    # use a set of different inducing points per output process see section 4.3.1 https://arxiv.org/pdf/2003.01115.pdf
    elif how == "multioutput_processes":
        assert transformation is None, "Not implemented"
        # although similar to how == 'single_process', prefer to make it explicit here
        Z = gpflow.inducing_variables.InducingPoints(Z)

    else:
        raise ValueError("Unkown inducing point wrapper {}".how())

    return Z


# ========================================================================================
# Interface to create GPflow kernel instances


def instance_kernels(
    name,
    ard_num_dims,
    num_multioutput,
    kernel_is_shared,
    return_kernels_as_list,
    init_params={},
):

    # Initial parameters
    init_params = get_kernel_hyperparams(init_params)

    # Expand lengthscale for ard if requested
    init_ls = (
        tf.ones(ard_num_dims, dtype=cg.dtype) * init_params["ls"]
        if ard_num_dims is not None
        else init_params["ls"]
    )

    # Expand parameter of linear kernel for ard if requested
    init_var = (
        tf.ones(ard_num_dims, dtype=cg.dtype) * init_params["linear_param"]
        if ard_num_dims is not None
        else init_params["linear_param"]
    )

    # Instance a list of multioutput kernels. Note that kernel combinations can be done better by specifying kernels in a list in name.
    if name == "scale_rbf":
        kernel_list = [
            gpflow.kernels.SquaredExponential(
                variance=init_params["os"], lengthscales=init_ls
            )
            for i in range(num_multioutput)
        ]
    elif name == "scale_rbf+linear":
        kernel_list = [
            gpflow.kernels.SquaredExponential(
                variance=init_params["os"], lengthscales=init_ls
            )
            + gpflow.kernels.Linear(variance=init_var)
            for i in range(num_multioutput)
        ]
    elif name == "share_ARD_scale_rbf+linear":
        # we tie the parameters of both kernels:
        kernel_list = []
        for i in range(num_multioutput):
            SE = gpflow.kernels.SquaredExponential(
                variance=init_params["os"], lengthscales=init_ls
            )
            LI = gpflow.kernels.Linear(variance=init_var)

            # Copy parameter reference. This should necesarily tie the parameters
            LI.variance = SE.lengthscales

            kernel_list.append(SE + LI)

    elif name == "arccosine":
        kernel_list = [
            gpflow.kernels.ArcCosine(order=1, weight_variances=init_ls)
            for i in range(num_multioutput)
        ]
    else:
        raise ValueError("Unknown kernel type {}".format(name))

    # Wrap it with the multioutput kernel or return as list unless num_multioutput is 1 in case we only return kernel
    # @fjsaezm: Commented this line as Juan suggested
    # if num_multioutput == 1:
    #    return kernel_list[0]

    if return_kernels_as_list:
        # Return them as list. If shared then the same instance object is repeated
        if kernel_is_shared:
            kernel = [kernel_list[0] for i in range(num_multioutput)]
        else:
            kernel = kernel_list
    else:
        if kernel_is_shared:
            # The fact that I only run shared K and shared Z does
            kernel = gpflow.kernels.SharedIndependent(
                kernel_list[0], output_dim=num_multioutput
            )
            #  not make using sharedindependent inneficient. Only when sharedindependent
            #  Kernel and separate Z are used Gpflow is inneficient and a kernel without
            # the SharedIndependent wrapping must be used. Anyway with the refactorization
            # I have done the code is not so inneficient as the original one since
            # I have included a tiling. I return SharedIndependent wrapping since my
            # implementation of LinearCoregionalization kernel works by wrapping
            # shared/separate independent. This was easier than creating two different
            # LMC kernels (for shared and independent) or checking internally if the kernel
            # list has one kernel (shared) or K kernels (separate). This could be, perhaps,
            # a better way to make the refactorization of LMC and would allow me to return
            # a single kernel without sharedindependent wrapping that I would wrap in a
            # list and pass into the LMC kernel.
        else:
            kernel = gpflow.kernels.SeparateIndependent(kernel_list)

    return kernel


# ========================================================================================
# Interface to initialize the inducing points


def initialize_inducing_points(X, num_Z, how, num_runs, verbose=True):
    if how == "kmeans":
        if verbose:
            display_message("Running KMEANS to obtain {} clusters".format(num_Z))
        kmeans = KMeans(
            n_clusters=num_Z,
            init="k-means++",
            n_init=num_runs,
            random_state=cg.seed,
        ).fit(X)

        return kmeans.cluster_centers_

    elif how == "parallel_kmeans":
        if verbose:
            display_message(
                "Running Parallel KMEANS to obtain {} clusters".format(num_Z)
            )
        kmeans = MiniBatchKMeans(
            n_clusters=num_Z,
            init="k-means++",
            batch_size=10000,
            n_init=num_runs,
            random_state=cg.seed,
        ).fit(X)

        return kmeans.cluster_centers_

    else:
        raise ValueError(
            "Cannot initialize inducing points using method {}".format(how)
        )


# ========================================================================================
# Gets data in a tensorflow dataloader and returns it as a numpy array


def get_dataset_from_tf_loader(dataloader, total_data, batch_size, data_shape):

    batch_size = int(batch_size)
    total_data = int(total_data)

    X = np.zeros((total_data,) + data_shape)
    Y = np.zeros((total_data,))

    for idx, (x, y) in enumerate(dataloader):

        X[idx * batch_size : idx * batch_size + batch_size] = x.numpy()
        Y[idx * batch_size : idx * batch_size + batch_size] = y.numpy()

    return X, Y


# ========================================================================================
# Displays message to stdout


def display_message(message):
    print("=" * 100)
    print("||", message, "||")
    print("=" * 100)


# ========================================================================================
# Class method that can be used to wrap arguments to tf functions in a __call__
class L2_normalization_wrapper(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.l2_normalize(x, axis=1)


# ========================================================================================
# Basic block for PCA projection


def return_projection_matrix(input_dim, output_dim, X):
    if output_dim == input_dim:
        # Identity Mapping
        W = np.eye(input_dim)
    elif output_dim > input_dim:
        # Idennptity Mapping + Zero PAD
        W = np.concatenate(
            [np.eye(input_dim), np.zeros((input_dim, output_dim - input_dim))], 1
        )
    elif output_dim < input_dim:
        # PCA
        _, _, V = np.linalg.svd(X, full_matrices=False)
        W = V[:output_dim, :].T
    else:
        raise NotImplemented("Check what happened")

    return W


# ======================================================================================
# Functions used to compute the percentile distance between datapoints
def pairwise_distances(X, max_datapoints=10_000):
    if X.shape[0] > max_datapoints:
        indices = tf.random.shuffle(tf.range(X.shape[0]))[
            :max_datapoints
        ]  # Randomly select 10000 indices
        X_selected = tf.gather(
            X, indices
        )  # Select the rows based on the random indices
    else:
        X_selected = X
    # Expand dimensions of X for broadcasting to compute pairwise differences
    X_expanded_1 = tf.expand_dims(X_selected, axis=1)  # Shape: [N, 1, d]
    X_expanded_2 = tf.expand_dims(X_selected, axis=0)  # Shape: [1, N, d]

    # Compute pairwise differences using broadcasting
    pairwise_diff = X_expanded_1 - X_expanded_2  # Shape: [N, N, d]
    lower_triangular_part = tf.experimental.numpy.tril(pairwise_diff, k=-1)
    pairwise_norms = tf.norm(lower_triangular_part, axis=-1)  # Shape: [N, N]

    return pairwise_norms


def compute_distances_percentile(X, percentile=50):
    dists = pairwise_distances(X)

    # Step 6: Compute the median of distances
    median_distance = np.percentile(dists, percentile)

    return median_distance


# ========================================================================================
# Interface to instantiate the inner layers of a DGP
def compute_whitened_q_mu_initialization_datapoints(
    layer_input,
    output,
    kernel_type,
    gps_per_inner_layer,
    kernel_is_shared,
    init_kernel_params,
    jitter=1e-6,
):
    """
    When initializing the DGP using a subset of datapoints aiming to predict the identity on those datapoints and using a whitened prediction,
    the output of each layer must be adjusted solving the equation L_xx m0 = output, where x is the input and output is the desired output
    The output in the inner layers should be the same inducing locations (Z) and the value of those inducing locations in the last layer
    """

    K = instance_kernels(
        name=kernel_type,
        ard_num_dims=layer_input.shape[-1],
        num_multioutput=gps_per_inner_layer,
        kernel_is_shared=kernel_is_shared,
        return_kernels_as_list=True,
        init_params=init_kernel_params,
    )[0]
    K_zz = K(layer_input, full_cov=True) + np.eye(layer_input.shape[0]) * jitter
    L_zz = tf.linalg.cholesky(K_zz)

    output0 = tf.linalg.triangular_solve(L_zz, output)

    return output0


def select_Z_from_trainset(X, init_Z, init_mode="datapoints_random"):
    # Ensure inputs are TensorFlow tensors
    X_tf = tf.constant(X, dtype=tf.float32)
    init_Z_tf = tf.constant(init_Z, dtype=tf.float32)

    mode = init_mode.split("_")[-1]

    if mode == "random":
        idx = np.random.choice(X_tf.shape[0], size=init_Z_tf.shape[0], replace=False)
    elif "kmeans" in init_mode:
        if X_tf.shape[1] == 1:
            # Use absolute distance for similarity in 1D
            # Expand dims to broadcast
            Z_exp = tf.expand_dims(init_Z_tf, 1)  # (M, 1, 1)
            X_exp = tf.expand_dims(X_tf, 0)  # (1, N, 1)
            distance = tf.abs(Z_exp - X_exp)  # (M, N, 1)
            distance = tf.squeeze(distance, axis=-1)
            similarity = -distance
        else:
            # Use cosine similarity for multi-dimensional data
            Z_normed = tf.nn.l2_normalize(init_Z_tf, axis=1)
            X_normed = tf.nn.l2_normalize(X_tf, axis=1)
            similarity = tf.linalg.matmul(Z_normed, tf.transpose(X_normed))  # (M, N)

        # Sort indices by similarity (descending)
        sorted_indices = tf.argsort(similarity, axis=1, direction="DESCENDING")

        if "unique" in init_mode:
            selected_indices = set()
            unique_min_indices = []

            for row in sorted_indices:
                for idx in row:
                    idx_val = idx.numpy()
                    if idx_val not in selected_indices:
                        selected_indices.add(idx_val)
                        unique_min_indices.append(idx)
                        break
            idx = tf.convert_to_tensor(unique_min_indices).numpy()
        else:
            idx = sorted_indices[:, 0].numpy()

    return idx


class PCA(gpflow.mean_functions.Linear):
    ## PCA mean function is the one proposed by salimbeni et al https://arxiv.org/abs/1705.08933
    #  Works like the linear mean function but with non learnable parameters and no bias

    def __init__(self, A):

        gpflow.mean_functions.MeanFunction.__init__(self)
        b = np.zeros(A.shape[-1], dtype=np.float64)
        self.A = tf.constant(A)
        self.b = tf.constant(b)


def instance_layers_dsvi_dgp(
    X,
    Y,
    init_Z,
    num_layers=2,
    gps_per_inner_layer=1,
    gps_output_layer=1,
    kernel_type="scale_rbf",
    kernel_is_shared=False,
    init_kernel_params={"ls": 2.0, "os": 2.0},
    type_inducing_point="latent_shared_independent",
    inner_mean_function="zero",
    output_mean_function="zero",
    init_variational_parameters="zeros",
    whitened_gp=True,
    whitened_kernel=False,
    jitter=1e-6,
):

    # Creating GP instances per layer
    num_Z = init_Z.shape[0]
    if "datapoints" in init_variational_parameters:

        # "datapointsm0_same" uses the same inducing locations as zeros initialization
        # since it doest not need the "y" (zero layer output)
        if "same" in init_variational_parameters:
            init_U = tf.zeros((num_Z, Y.shape[1]), dtype=cg.dtype)
        else:
            idx = select_Z_from_trainset(X, init_Z, init_variational_parameters)
            init_Z = X[idx]
            init_U = Y[idx]

        m0 = compute_whitened_q_mu_initialization_datapoints(
            init_Z,
            init_Z,
            kernel_type,
            gps_per_inner_layer,
            kernel_is_shared,
            init_kernel_params,
            jitter,
        )
        y0 = compute_whitened_q_mu_initialization_datapoints(
            init_Z,
            init_U,
            kernel_type,
            gps_output_layer,
            kernel_is_shared,
            init_kernel_params,
            jitter,
        )
        if not whitened_gp:
            m0 = init_Z
            y0 = init_U

    GP_list = []
    for i in range(num_layers):

        if i == num_layers - 1:  # changing to instance output layer
            gps_per_inner_layer = gps_output_layer
            inner_mean_function = output_mean_function
            if "datapoints" in init_variational_parameters:
                m0 = y0

        # Wrap inducing points
        Z = wrap_inducing_points(
            Z=init_Z,
            how=type_inducing_point,
            num_multioutput=gps_per_inner_layer,
        )

        # Mean function
        W_x = return_projection_matrix(
            input_dim=X.shape[-1],
            output_dim=gps_per_inner_layer,
            X=X,
        )
        if inner_mean_function == "PCA":
            # mean_function = gpflow.mean_functions.PCA(A=W_x)
            mean_function = PCA(A=W_x)
        elif inner_mean_function == "zero":
            mean_function = gpflow.mean_functions.Zero(output_dim=gps_per_inner_layer)
        else:
            raise NotImplementedError(
                "Unknown mean function {}".format(inner_mean_function)
            )

        # Kernel
        K = instance_kernels(
            name=kernel_type,
            ard_num_dims=X.shape[-1],
            num_multioutput=gps_per_inner_layer,
            kernel_is_shared=kernel_is_shared,
            return_kernels_as_list=False,
            init_params=init_kernel_params,
        )
        if whitened_kernel:
            K = K + gpflow.kernels.White(variance=1e-5)

        # Variational parameters
        if "zeros" in init_variational_parameters:
            q_mu = tf.zeros((num_Z, gps_per_inner_layer), dtype=cg.dtype)
        elif init_variational_parameters == "random":
            q_mu = tf.random.normal([num_Z, gps_per_inner_layer], dtype=cg.dtype)
        elif init_variational_parameters == "data_mean":
            q_mu = tf.ones(
                (num_Z, gps_per_inner_layer), dtype=cg.dtype
            ) * tf.reduce_mean(Y)
        elif "datapoints" in init_variational_parameters:
            # Differenciate 2 cases: last layer also datapoints (using m0) or not
            if i == num_layers - 1:
                if "mY" in init_variational_parameters:
                    q_mu = m0
                elif "m0" in init_variational_parameters:
                    q_mu = tf.zeros((num_Z, gps_per_inner_layer), dtype=cg.dtype)
                else:
                    raise ValueError(
                        "If init_variational_parameters contains datapoints, it must contain m0 or mY for the last layer q_mu"
                    )
            else:
                q_mu = m0

        else:
            raise ValueError(
                "Initialization of variational parameters can be: choices = ['zeros', 'zeros_largestd', 'datapoints', 'datapoints_kmeans', 'datapoints_kmeansunique','datapoints_kmeansunique_largestd']"
            )

        q_sqrt = tf.linalg.cholesky(
            tf.eye(num_Z, batch_shape=[gps_per_inner_layer], dtype=cg.dtype) * 1e-5
        )

        # Last layer identity variance
        if i == num_layers - 1:
            if "largestd" in init_variational_parameters:
                q_sqrt = tf.linalg.cholesky(
                    tf.eye(num_Z, batch_shape=[gps_per_inner_layer], dtype=cg.dtype)
                )
            elif (
                "cov" in init_variational_parameters
                and "mY" in init_variational_parameters
            ):
                if whitened_gp:
                    q_sqrt = tf.linalg.cholesky(
                        tf.eye(num_Z, batch_shape=[gps_per_inner_layer], dtype=cg.dtype)
                    )
                else:
                    K_zz = (
                        tf.squeeze(K(init_Z, full_cov=True))
                        + tf.eye(init_Z.shape[0], dtype=cg.dtype) * 1e-12
                    )
                    L_zz = tf.linalg.cholesky(K_zz)
                    q_sqrt = tf.expand_dims(L_zz, axis=0)

        # Instance gp per layer
        gp = SVGP_base(
            kernel=K,
            likelihood=None,
            inducing_variable=Z,
            mean_function=mean_function,
            num_latent_gps=gps_per_inner_layer,
            q_diag=False,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whitened_gp,
            num_data=None,
        )

        GP_list.append(gp)

        # Project inducing points for next layer as in Salimbeni.
        W_z = return_projection_matrix(
            input_dim=init_Z.shape[-1],
            output_dim=gps_per_inner_layer,
            X=init_Z,
        )

        # Project inducing points to next layer
        init_Z = init_Z @ W_z

        # Project data to next layer
        X = X @ W_x

    return GP_list


def instance_dgp_config(
    X,
    Y,
    init_Z,
    num_layers=2,
    gps_per_inner_layer=1,
    gps_output_layer=1,
    kernel_type="scale_rbf",
    kernel_is_shared=False,
    init_kernel_params={"ls": 2.0, "os": 2.0},
    type_inducing_point="latent_shared_independent",
    inner_mean_function="zero",
    output_mean_function="zero",
    init_variational_parameters="zeros",
    whitened_gp=True,
    whitened_kernel=False,
    jitter=1e-6,
    likelihood=None,
):
    from .DSVI_DGP import DSVI_DGP

    GP_list = instance_layers_dsvi_dgp(
        X=X,
        Y=Y,
        init_Z=init_Z,
        num_layers=num_layers,
        gps_per_inner_layer=gps_per_inner_layer,
        gps_output_layer=gps_output_layer,
        kernel_type=kernel_type,
        kernel_is_shared=kernel_is_shared,
        init_kernel_params=init_kernel_params,
        type_inducing_point=type_inducing_point,
        inner_mean_function=inner_mean_function,
        output_mean_function=output_mean_function,
        init_variational_parameters=init_variational_parameters,
        whitened_gp=whitened_gp,
        whitened_kernel=whitened_kernel,
        jitter=jitter,
    )

    model = DSVI_DGP(
        GP_list=GP_list,
        likelihood=likelihood,
        num_data=X.shape[0],
    )

    return model


def instance_best_metric_dgp_using_lenghtscale(
    X,
    Y,
    dataset,
    init_Z,
    num_layers=2,
    gps_per_inner_layer=1,
    gps_output_layer=1,
    kernel_type="scale_rbf",
    kernel_is_shared=False,
    init_kernel_params={"ls": 2.0, "os": 2.0},
    type_inducing_point="latent_shared_independent",
    inner_mean_function="zero",
    output_mean_function="zero",
    init_variational_parameters="zeros",
    whitened_gp=True,
    whitened_kernel=False,
    likelihood=0.001,
    jitter=1e-6,
    distance_percentile=50,
    lenghtscale_scaling_factor=2.0,
    num_steps=5,
    metric="rmse",
):

    distance = compute_distances_percentile(X, percentile=distance_percentile)
    print("Median distance: ", distance)

    step_size = 0.5
    multipliers = (
        [distance - i * step_size for i in range(num_steps, 0, -1)]
        + [distance]
        + [distance + i * step_size for i in range(1, num_steps + 1)]
    )
    multipliers = np.array(multipliers)
    multipliers = multipliers[multipliers > 0.0]
    # multipliers = (
    #    [distance / (lenghtscale_scaling_factor**i) for i in range(num_steps, 0, -1)]
    #    + [distance]
    #    + [distance * (lenghtscale_scaling_factor**i) for i in range(1, num_steps + 1)]
    # )
    multipliers = np.round(multipliers, decimals=4)

    ds_train = tf.data.Dataset.from_tensor_slices((X, Y))
    buffer_size = 50000 if X.shape[0] > 50000 else X.shape[0]
    ds_train = ds_train.shuffle(
        buffer_size=buffer_size, reshuffle_each_iteration=True, seed=cg.seed
    )
    ds_train = ds_train.batch(X.shape[0])

    all_models = []
    all_metrics = []
    all_m = []
    print(
        "Searching for the lengthscale that optimizes the metric {} in the dataset {}".format(
            metric, dataset
        )
    )
    # print("Multipliers: ", multipliers)

    for m in multipliers:
        k_params = {"ls": m, "os": init_kernel_params["os"]}

        model = instance_dgp_config(
            X,
            Y,
            init_Z,
            num_layers=num_layers,
            gps_per_inner_layer=gps_per_inner_layer,
            gps_output_layer=gps_output_layer,
            kernel_type=kernel_type,
            kernel_is_shared=kernel_is_shared,
            init_kernel_params=k_params,
            type_inducing_point=type_inducing_point,
            inner_mean_function=inner_mean_function,
            output_mean_function=output_mean_function,
            init_variational_parameters=init_variational_parameters,
            whitened_gp=whitened_gp,
            whitened_kernel=whitened_kernel,
            jitter=jitter,
            likelihood=likelihood,
        )

        if metric == "rmse":
            value, _ = evaluation_step_regression_DSVI_DGP(
                ds_train,
                model,
                reshape_shape=None,
                dataset=dataset,
                SMC=10,
                return_point_preds=False,
            )
        elif metric == "ell":
            _, value, _ = model.elbo((X, Y), S_DGP=100)

        all_models.append(copy.deepcopy(model))
        all_metrics.append(value)
        all_m.append(m)

    if metric == "rmse":
        best_idx = np.argmin(all_metrics)
    elif metric == "ell":
        best_idx = np.argmax(all_metrics)

    # print("All metrics: ", all_metrics)
    # print(f"Best {metric}: ", all_metrics[best_idx])

    best_model = all_models[best_idx]
    best_m = all_m[best_idx]

    return best_model, best_m


def instance_dsvi_dgp(
    X,
    Y,
    dataset,
    init_Z,
    num_layers=2,
    gps_per_inner_layer=1,
    gps_output_layer=1,
    kernel_type="scale_rbf",
    kernel_is_shared=False,
    init_kernel_params={"ls": 2.0, "os": 2.0},
    type_inducing_point="latent_shared_independent",
    inner_mean_function="zero",
    output_mean_function="zero",
    init_variational_parameters="zeros",
    whitened_gp=True,
    whitened_kernel=False,
    jitter=1e-6,
    init_kernel_lengthscale="constant",
    likelihood=None,
):

    if init_kernel_lengthscale == "distances":
        model, used_lengthscale = instance_best_metric_dgp_using_lenghtscale(
            X,
            Y,
            dataset,
            init_Z,
            num_layers=num_layers,
            gps_per_inner_layer=gps_per_inner_layer,
            gps_output_layer=gps_output_layer,
            kernel_type=kernel_type,
            kernel_is_shared=kernel_is_shared,
            init_kernel_params=init_kernel_params,
            type_inducing_point=type_inducing_point,
            inner_mean_function=inner_mean_function,
            output_mean_function=output_mean_function,
            init_variational_parameters=init_variational_parameters,
            whitened_gp=whitened_gp,
            whitened_kernel=whitened_kernel,
            likelihood=likelihood,
            jitter=jitter,
            distance_percentile=50,
            lenghtscale_scaling_factor=2.0,
            num_steps=5,
            metric="rmse",
        )
    else:
        model = instance_dgp_config(
            X,
            Y,
            init_Z,
            num_layers=num_layers,
            gps_per_inner_layer=gps_per_inner_layer,
            gps_output_layer=gps_output_layer,
            kernel_type=kernel_type,
            kernel_is_shared=kernel_is_shared,
            init_kernel_params=init_kernel_params,
            type_inducing_point=type_inducing_point,
            inner_mean_function=inner_mean_function,
            output_mean_function=output_mean_function,
            init_variational_parameters=init_variational_parameters,
            whitened_gp=whitened_gp,
            whitened_kernel=whitened_kernel,
            likelihood=likelihood,
        )
        used_lengthscale = init_kernel_params["ls"]

    return model, used_lengthscale
