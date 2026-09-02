import tensorflow as tf
from tqdm import tqdm
import copy
import joblib
from omegaconf import DictConfig
import numpy as np
from utils_training import (
    compute_class_weights,
    optimization_step_SVGP_weighted,
    train_beta,
    train_corf,
    train_core,
    train_drf,
    train_poe,
    train_sord,
)


def train(cfg: DictConfig, model, X_train, y_train, logger=None):
    model_name = cfg.model.name

    # Get training parameters for all models
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.learning_rate
    epochs = cfg.training.epochs
    patience = cfg.training.patience
    log_interval = cfg.training.log_interval
    lr_scheduler = cfg.training.lr_scheduler

    # Get save weights path at the beginning for all models
    save_weights_path = cfg.output.get("save_weights_path")

    if model_name in [
            "OrdinalSVGP",
            "SVGP_multiclass",
            "OrdinalDSVI_DGP",
            "DKL_OrdinalSVGP",
    ]:

        # Linear Learning Rate Scheduler: reduces LR by 10x over 100 epochs
        steps_per_epoch = max(len(X_train) // batch_size, 1)
        print("Steps per epoch:", steps_per_epoch)

        if lr_scheduler == "linear":
            # Create linear decay scheduler (reduces from initial_lr to initial_lr/10 over 500 epochs)
            total_steps = (min(epochs, cfg.training.lr_scheduler_last_epoch) *
                           steps_per_epoch)
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=total_steps,
                end_learning_rate=learning_rate / 10.0,
                power=1.0,  # Linear decay (power=1.0)
            )
            opt = tf.optimizers.Adam(learning_rate=lr_schedule)

        elif lr_scheduler == "cosine":
            print("Using cosine decay learning rate scheduler")
            
            # Cosine decay scheduler
            total_steps = min(
                epochs, cfg.training.lr_scheduler_last_epoch) * steps_per_epoch
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=total_steps,
                alpha=learning_rate * cfg.training.
                lr_min_factor,  # Minimum learning rate as a fraction of the initial learning rate
            )
            opt = tf.optimizers.Adam(learning_rate=lr_schedule)
        else:
            opt = tf.optimizers.Adam(learning_rate=learning_rate)

        if batch_size > X_train.shape[0]:
            batch_size = X_train.shape[0]
            print(
                f"Batch size {batch_size} is larger than the dataset size. Using the full dataset for training."
            )

        # Dataset object
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (train_dataset.shuffle(
            buffer_size=len(X_train),
            seed=cfg.seed,
            reshuffle_each_iteration=True).batch(batch_size).prefetch(
                tf.data.AUTOTUNE).repeat())
        prior_temp = cfg.model.get("prior_temp", 1.0)
        prior_temperature = tf.constant(prior_temp, dtype=tf.float64)
        class_weights = (compute_class_weights(y_train)
                         if cfg.training.get("class_weights") == "compute" else
                         [1 for _ in range(len(np.unique(y_train)))])
        print("Using class weights:", class_weights)

        # Create a temporary directory to save the best model checkpoint
        if save_weights_path is not None:
            checkpoint_dir = save_weights_path
            checkpoint = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(checkpoint,
                                                 checkpoint_dir,
                                                 max_to_keep=1)
        best_elbo = float("-inf")
        patience_counter = 0

        progress_bar = tqdm(range(epochs))
        for epoch in progress_bar:
            total_losses = {"elbo": 0.0, "ell": 0.0, "kl": 0.0}
            for X_batch, Y_batch in train_dataset.take(steps_per_epoch):
                elbo, ell, kl = optimization_step_SVGP_weighted(
                    model,
                    X_batch,
                    Y_batch,
                    opt,
                    class_weights=class_weights,
                    prior_temperature=prior_temperature,
                )
                total_losses["elbo"] += elbo.numpy()
                total_losses["ell"] += ell.numpy()
                total_losses["kl"] += kl.numpy()

            # Update progress bar with current ELBO
            progress_bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - ELL: {total_losses['ell']:.4f}, KL: {total_losses['kl']:.4f}"
            )

            # Early stopping check
            if total_losses["elbo"] > best_elbo:
                best_elbo = total_losses["elbo"]
                patience_counter = 0
                if save_weights_path is not None:
                    manager.save()  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1} — no improvement in {patience} epochs."
                    )
                    break

            # Log every log_interval epochs
            if logger is not None and (epoch + 1) % log_interval == 0:
                logger.log(total_losses, step=epoch)

                if model_name in ["OrdinalSVGP", "DKL_OrdinalSVGP"]:
                    if model.likelihood.is_heteroscedastic:
                        diagnostics = {}
                        if getattr(model.likelihood, "uses_latent_noise", False):
                            latent_mean, latent_var = model.predict_f(X_batch)
                            batch_scales = model.likelihood.noise_scale_from_latent(
                                latent_mean[:, 1:2]
                            )
                            diagnostics["likelihood_noise_latent_std_mean"] = (
                                tf.reduce_mean(
                                    tf.sqrt(tf.maximum(latent_var[:, 1:2], 0.0))
                                ).numpy()
                            )
                        else:
                            batch_scales = model.likelihood.noise_scale(X_batch)
                        diagnostics.update(
                            {
                                "likelihood_sigma_mean": tf.reduce_mean(
                                    batch_scales
                                ).numpy(),
                                "likelihood_sigma_min": tf.reduce_min(
                                    batch_scales
                                ).numpy(),
                                "likelihood_sigma_max": tf.reduce_max(
                                    batch_scales
                                ).numpy(),
                            }
                        )
                        logger.log(diagnostics, step=epoch)
                    else:
                        # Preserve the historical logging key for existing runs.
                        logger.log(
                            {
                                "likelihood_variance":
                                model.likelihood.sigma.numpy()
                            },
                            step=epoch,
                        )

        # Restore the best model
        if save_weights_path is not None:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print("Restored best model")

        if cfg.model.name == "DKL_OrdinalSVGP":
            model.eval(
            )  # Set DKL model to evaluation mode after loading weights

    elif model_name == "RLOrdinal":
        # mord models use sklearn-like API (much faster than statsmodels)
        model.fit(X_train, y_train.ravel())  # Ensure y is 1D
        if save_weights_path is not None:
            joblib.dump(model, save_weights_path + "best.pkl")  # Save

    elif model_name in [
            "LogisticRegression",
            "MLP",
            "SVM",
            "KNN",
            "NaiveBayes",
            "RandomForest",
    ]:
        model.fit(X_train, y_train)
        if save_weights_path is not None:
            joblib.dump(model, save_weights_path + "best.pkl")  # Save

    elif model_name == "OrdinalMLP":

        # Convert to tensors
        X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
        y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)

        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tf, y_train_tf))
        train_dataset = train_dataset.shuffle(
            buffer_size=len(X_train), seed=cfg.seed
        ).batch(batch_size)

        # Optimizer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)

        # Callbacks
        callbacks = []
        if save_weights_path is not None:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=save_weights_path + "best.keras",
                save_best_only=True,
                monitor='loss',
                mode='min'
            )
            callbacks.append(checkpoint_callback)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # Train
        model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    elif model_name in ["SORD", "Beta", "POE", "CORE", "DRF", "CORF"]:
        trainers = {
            "SORD": train_sord,
            "Beta": train_beta,
            "POE": train_poe,
            "CORE": train_core,
            "DRF": train_drf,
            "CORF": train_corf,
        }
        model = trainers[model_name](
            cfg, model, X_train, y_train, logger=logger
        )

    return model
