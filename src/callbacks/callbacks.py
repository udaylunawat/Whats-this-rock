import tensorflow as tf
from wandb.keras import WandbCallback

def get_earlystopper(cfg):
    cfg = cfg.callback

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.monitor,
        patience=cfg.earlystopping.patience,
        verbose=1,
        mode='max',
        restore_best_weights=True)

    return earlystopper


def get_reduce_lr_on_plateau(cfg):
    cfg = cfg.callback

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=cfg.monitor,
        factor=cfg.reduce_lr.factor,
        patience=cfg.reduce_lr.patience,
        min_lr=cfg.reduce_lr.min_lr,
        )

    return reduce_lr_on_plateau


def get_callbacks(cfg):
    # Define WandbCallback for experiment tracking
    wandbcallback = WandbCallback(
        monitor=cfg.callback.monitor,
        mode='min',
        save_model=(cfg.model.save_model),
    )

    callbacks = [wandbcallback]
    if cfg.callback.earlystopping.use:
        earlystopper = get_earlystopper(cfg)
        callbacks.append(earlystopper)
    if cfg.callback.reduce_lr.use:
        reduce_lr = get_reduce_lr_on_plateau(cfg)
        callbacks.append(reduce_lr)

    return callbacks