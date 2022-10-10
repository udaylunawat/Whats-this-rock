import tensorflow as tf
from tensorflow.keras.optimizers import schedules
from wandb.keras import WandbCallback


def get_earlystopper(cfg) -> tf.keras.callbacks:
    """ Returns tf.keras.callbacks.EarlyStopping

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
    Configuration

    Returns
    -------
    tensorflow.keras.callbacks
        Stop training when a monitored metric has stopped improving.
    """
    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.monitor,
        patience=cfg.earlystopping.patience,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    )

    return earlystopper


def get_reduce_lr_on_plateau(cfg):
    """ Returns tf.keras.callbacks.ReduceLROnPlateau

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
    Configuration

    Returns
    -------
    tf.keras.callbacks.ReduceLROnPlateau
        Reduce learning rate when a metric has stopped improving.
    """
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=cfg.monitor,
        factor=cfg.reduce_lr.factor,
        patience=cfg.reduce_lr.patience,
        min_lr=cfg.reduce_lr.min_lr,
        verbose=1,
    )

    return reduce_lr_on_plateau


def get_callbacks(cfg):
    """ Returns a Callback List

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Configuration

    Returns
    -------
    List
        Callbacks List
    """
    wandbcallback = WandbCallback(
        monitor=cfg.monitor, mode="auto", save_model=(cfg.save_model),
    )

    callbacks = [wandbcallback]
    if cfg.earlystopping.use:
        earlystopper = get_earlystopper(cfg)
        callbacks.append(earlystopper)
    if cfg.reduce_lr.use:
        reduce_lr = get_reduce_lr_on_plateau(cfg)
        callbacks.append(reduce_lr)

    return callbacks
