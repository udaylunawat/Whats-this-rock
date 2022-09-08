import time
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# import models


def get_optimizer(config):
    if config.train_config.optimizer == "adam":
        opt = optimizers.Adam(learning_rate=config.train_config.lr)
    elif config.train_config.optimizer == "rms":
        opt = optimizers.RMSprop(learning_rate=config.train_config.lr,
                                 rho=0.9,
                                 epsilon=1e-08,
                                 decay=0.0)
    elif config.train_config.optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=config.train_config.lr)
    elif config.train_config.optimizer == "adamax":
        opt = optimizers.Adamax(learning_rate=config.train_config.lr)

    return opt


def get_model_weights_ds(train_ds):
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_ds.class_names),
        y=train_ds.class_names,
    )

    train_class_weights = dict(enumerate(class_weights))
    return train_class_weights


def configure_for_performance(ds, config):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    # ds = ds.batch(config.dataset_config.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

