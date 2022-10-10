import time
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend as K
from tensorflow.keras.callbacks import Callback


def get_optimizer(cfg, lr: str) -> optimizers:
    """Gets optimizer set with an learning rate

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Configuration
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
        "adamax": optimizers.Adamax
    }
    try:
        opt = optimizer_dict[cfg.optimizer](learning_rate=lr)
    except:
        raise NotImplementedError("Not implemented.")

    return opt


def get_model_weights(train_ds):
    """_summary_

    Parameters
    ----------
    train_ds : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_ds.class_names),
        y=train_ds.class_names,
    )

    train_class_weights = dict(enumerate(class_weights))
    return train_class_weights


def get_lr_scheduler(cfg) -> schedules:
    """Returns A LearningRateSchedule

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
    Configuration

    Returns
    -------
    schedules
        A LearningRateSchedule
    """
    scheduler = {
        'cosine_decay':schedules.CosineDecay(cfg.lr, decay_steps=cfg.lr_decay_steps),
        'exponentialdecay':schedules.ExponentialDecay(cfg.lr, decay_steps=100, decay_rate=0.96, staircase=True),
        'cosine_decay_restarts':schedules.CosineDecayRestarts(cfg.lr, first_decay_steps=cfg.lr_decay_steps),
    }
    return scheduler[cfg.lr_schedule]
