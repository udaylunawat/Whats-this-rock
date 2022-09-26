import tensorflow as tf
from wandb.keras import WandbCallback


def get_earlystopper(cfg):
    """_summary_

    Parameters
    ----------
    cfg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    cfg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    """ Define WandbCallback for experiment tracking

    Parameters
    ----------
    cfg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
