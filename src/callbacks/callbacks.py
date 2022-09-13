import tensorflow as tf


def get_earlystopper(cfg):
    cfg = cfg.callback

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.monitor,
        patience=cfg.earlystopping.patience,
        verbose=1,
        mode='auto',
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
