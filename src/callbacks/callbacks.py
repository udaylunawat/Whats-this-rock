import wandb
import warnings
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras.optimizers import schedules


def get_earlystopper(cfg) -> tf.keras.callbacks:
    """Return tf.keras.callbacks.EarlyStopping.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration

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
    """Return tf.keras.callbacks.ReduceLROnPlateau.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration

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


class LRLogger(tf.keras.callbacks.Callback):
    """log lr at the end of every epoch.

    Parameters
    ----------
    tf : callbacks
        callbacks
    """

    def on_epoch_end(self, epoch, logs=None):
        """Log lr on epoch end."""
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr.initial_learning_rate))  # get the current learning rate
        wandb.log({'learning_rate':lr}, commit=True)


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """Stops training when epoch > min_epoch and monitor value < value.

    Parameters
    ----------
    Callback : Tensorflow Callback
        tensorflow Callback
    """

    def __init__(self, monitor='val_accuracy', value=0.60, min_epoch=10, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.min_epoch = min_epoch
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value and epoch >= self.min_epoch:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def get_callbacks(cfg):
    """Return a Callback List.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration

    Returns
    -------
    List
        Callbacks List
    """
    wandbcallback = WandbCallback(
        monitor=cfg.monitor, mode="auto", save_model=(cfg.save_model),
    )

    callbacks = [wandbcallback, LRLogger(), CustomEarlyStopping()]
    if cfg.earlystopping.use:
        earlystopper = get_earlystopper(cfg)
        callbacks.append(earlystopper)
    else:
        cfg.earlystopping.patience = None
    if cfg.reduce_lr.use:
        reduce_lr = get_reduce_lr_on_plateau(cfg)
        callbacks.append(reduce_lr)
    else:
        cfg.reduce_lr.factor = None
        cfg.reduce_lr.min_lr = None
        cfg.reduce_lr.patience = None

    return callbacks, cfg
