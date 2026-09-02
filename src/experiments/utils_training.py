import tensorflow as tf
import numpy as np
from tqdm import tqdm


def optimization_variables(model):
    """Return GPflow variables plus any explicitly registered likelihood weights."""
    variables = list(model.trainable_variables)
    additional = getattr(model.likelihood, "additional_trainable_variables", ())
    known_ids = {id(variable) for variable in variables}
    variables.extend(
        variable for variable in additional if id(variable) not in known_ids
    )
    return variables


def compute_class_weights(Y, jitter=1e-6):
    """
    Compute class weights inversely proportional to class frequency.

    Args:
        Y: tf.Tensor of shape [num_samples, 1] or [num_samples]
        jitter: small value to avoid division by zero
    Returns:
        tf.Tensor of shape [num_classes], normalized to sum to 1
    """
    num_classes = len(np.unique(Y))
    Y = tf.reshape(Y, [-1])  # flatten
    counts = tf.math.bincount(
        Y, minlength=num_classes, dtype=tf.float64
    )  # [num_classes]

    # Inverse frequency
    inv_freq = 1.0 / (counts + jitter)

    # Normalize so sum(weights * counts) = num_samples
    weights = inv_freq * (tf.reduce_sum(counts) / tf.reduce_sum(inv_freq * counts))

    alpha = 0.5  # smooth weights
    clip_min, clip_max = 0.9, 1.1

    weights = tf.pow(weights, alpha)
    weights = tf.clip_by_value(weights, clip_min, clip_max)
    weights /= tf.reduce_mean(weights)

    return weights  # shape [num_classes]


# @tf.function
def optimization_step_SVGP_weighted(
    model, X, Y, opt, class_weights, prior_temperature=1.0
):
    """
    Perform a single optimization step for SVGP with per-class weighting.
    """

    # Ensure class_weights is a Tensor (handles list or tensor input)
    class_weights_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float64)

    # Flatten Y to a 1-D index vector for gathering weights
    y_flat = tf.reshape(Y, [-1])
    # Make sure indices are integer type for gather
    y_idx = tf.cast(y_flat, tf.int32)
    weights = tf.cast(tf.gather(class_weights_tensor, y_idx), tf.float64)  # shape [N]

    Y_unsqueezed = tf.reshape(Y, [-1, 1])

    with tf.GradientTape() as tape:

        f_mean, f_var = model.predict_f(X)  # (N, OD) or (N,)

        ell_per_sample = model.likelihood.variational_expectations(
            X, f_mean, f_var, Y_unsqueezed
        )

        # Apply per-sample weights (both are shape [N])
        weighted_per_sample = ell_per_sample * weights  # (N,)

        # Rescale for minibatch (like standard SVGP)
        weighted_ell = tf.reduce_sum(weighted_per_sample) * (
            model.num_data / tf.cast(tf.shape(X)[0], tf.float64)
        )

        kl = model.prior_kl()
        # Minus elbo
        m_elbo = -weighted_ell + prior_temperature * kl

    trainable_variables = optimization_variables(model)
    grads = tape.gradient(m_elbo, trainable_variables)
    gradient_pairs = [
        (gradient, variable)
        for gradient, variable in zip(grads, trainable_variables)
        if gradient is not None
    ]
    opt.apply_gradients(gradient_pairs)

    return -m_elbo, weighted_ell, kl


def _modern_ordinal_optimizer(cfg, steps_per_epoch):
    learning_rate = cfg.training.learning_rate
    scheduler = str(cfg.training.get("lr_scheduler", "none")).lower()
    schedule_epochs = min(
        cfg.training.epochs, cfg.training.get("lr_scheduler_last_epoch", cfg.training.epochs)
    )
    decay_steps = max(int(schedule_epochs * steps_per_epoch), 1)
    if scheduler == "linear":
        learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
            learning_rate,
            decay_steps,
            learning_rate / 10.0,
            power=1.0,
        )
    elif scheduler == "cosine":
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            learning_rate,
            decay_steps,
            alpha=float(cfg.training.get("lr_min_factor", 0.01)),
        )
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def train_modern_ordinal(cfg, model, X_train, y_train, logger=None):
    """Shared no-validation training loop for the five ordinal additions."""
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
    y_train = tf.convert_to_tensor(np.asarray(y_train).reshape(-1), dtype=tf.int32)
    batch_size = min(int(cfg.training.batch_size), int(X_train.shape[0]))
    steps_per_epoch = int(np.ceil(int(X_train.shape[0]) / batch_size))
    dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    optimizer = _modern_ordinal_optimizer(cfg, steps_per_epoch)

    checkpoint_manager = None
    save_weights_path = cfg.output.get("save_weights_path")
    if save_weights_path is not None:
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, save_weights_path, max_to_keep=1
        )

    best_loss = float("inf")
    patience_counter = 0
    patience = int(cfg.training.patience)
    log_interval = int(cfg.training.log_interval)
    progress = tqdm(range(int(cfg.training.epochs)))

    for epoch in progress:
        component_totals = {}
        batches = 0
        for X_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                loss, components = model.loss_components(
                    X_batch, y_batch, training=True
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            gradient_pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            ]
            optimizer.apply_gradients(gradient_pairs)
            batches += 1
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.numpy())

        epoch_components = {
            name: value / batches for name, value in component_totals.items()
        }
        epoch_loss = epoch_components["loss"]

        # Differentiable forests alternate router and closed-form leaf updates.
        if hasattr(model, "update_leaf_distributions"):
            model.update_leaf_distributions(X_train, y_train)

        progress.set_description(
            f"Epoch {epoch + 1}/{cfg.training.epochs} - loss: {epoch_loss:.5f}"
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            if checkpoint_manager is not None:
                checkpoint_manager.save()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"training loss did not improve for {patience} epochs."
                )
                break

        if logger is not None and (epoch + 1) % log_interval == 0:
            logger.log(epoch_components, step=epoch)

    if checkpoint_manager is not None and checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print("Restored best model")
    return model


def train_sord(cfg, model, X_train, y_train, logger=None):
    return train_modern_ordinal(cfg, model, X_train, y_train, logger)


def train_beta(cfg, model, X_train, y_train, logger=None):
    return train_modern_ordinal(cfg, model, X_train, y_train, logger)


def train_poe(cfg, model, X_train, y_train, logger=None):
    return train_modern_ordinal(cfg, model, X_train, y_train, logger)


def train_core(cfg, model, X_train, y_train, logger=None):
    return train_modern_ordinal(cfg, model, X_train, y_train, logger)


def train_drf(cfg, model, X_train, y_train, logger=None):
    return train_modern_ordinal(cfg, model, X_train, y_train, logger)


def train_corf(cfg, model, X_train, y_train, logger=None):
    return train_modern_ordinal(cfg, model, X_train, y_train, logger)
