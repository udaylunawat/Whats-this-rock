# AUTOGENERATED! DO NOT EDIT! File to edit: ../../notebooks/03_a_train_utils.ipynb.

# %% auto 0
__all__ = ['get_optimizer', 'get_model_weights', 'get_lr_scheduler']

# %% ../../notebooks/03_a_train_utils.ipynb 1
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import optimizers

# %% ../../notebooks/03_a_train_utils.ipynb 2
def get_optimizer(cfg, lr: str) -> optimizers:
    """Get optimizer set with an learning rate.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration
    lr : str
        learning rate

    Returns
    -------
    tensorflow.keras.optimizers
        Tensorflow optimizer

    Raises
    ------
    NotImplementedError
        Raise error if cfg.optimizer not implemented.
    """
    optimizer_dict = {
        "adam": optimizers.Adam,
        "rms": optimizers.RMSprop,
        "sgd": optimizers.SGD,
        "adamax": optimizers.Adamax,
    }
    try:
        opt = optimizer_dict[cfg.optimizer](learning_rate=lr)
    except NotImplementedError:
        raise NotImplementedError("Not implemented.")

    return opt


def get_model_weights(train_ds: tf.data.Dataset):
    """Return model weights dict.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Tensorflow Dataset.

    Returns
    -------
    dict
        Dictionary of class weights.
    """
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_ds.class_names),
        y=train_ds.class_names,
    )

    train_class_weights = dict(enumerate(class_weights))
    return train_class_weights


def get_lr_scheduler(cfg, lr) -> optimizers.schedules:
    """Return A LearningRateSchedule.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration

    Returns
    -------
    tensorflow.keras.optimizers.schedules
        A LearningRateSchedule
    """
    scheduler = {
        "cosine_decay": optimizers.schedules.CosineDecay(
            lr, decay_steps=cfg.lr_decay_steps
        ),
        "exponentialdecay": optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=cfg.lr_decay_steps,
            decay_rate=cfg.reduce_lr.factor,
            staircase=True,
        ),
        "cosine_decay_restarts": optimizers.schedules.CosineDecayRestarts(
            lr, first_decay_steps=cfg.lr_decay_steps
        ),
    }
    return scheduler[cfg.lr_schedule]
