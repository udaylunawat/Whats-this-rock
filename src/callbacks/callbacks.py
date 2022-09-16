import tensorflow as tf
from wandb.keras import WandbCallback

def get_earlystopper(cfg):

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.monitor,
        patience=cfg.earlystopping_patience,
        verbose=1,
        mode='max',
        restore_best_weights=True)

    return earlystopper


def get_reduce_lr_on_plateau(cfg):

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=cfg.monitor,
        factor=cfg.reduce_lr_factor,
        patience=cfg.reduce_lr_patience,
        min_lr=cfg.reduce_lr_min_lr,
        verbose=1,
        )

    return reduce_lr_on_plateau


def get_callbacks(cfg):
    # Define WandbCallback for experiment tracking
    wandbcallback = WandbCallback(
        monitor=cfg.monitor,
        mode='auto',
        save_model=(cfg.save_model),
    )

    callbacks = [wandbcallback]
    if cfg.use_earlystopping:
        earlystopper = get_earlystopper(cfg)
        callbacks.append(earlystopper)
    if cfg.use_reduce_lr:
        reduce_lr = get_reduce_lr_on_plateau(cfg)
        callbacks.append(reduce_lr)

    return callbacks