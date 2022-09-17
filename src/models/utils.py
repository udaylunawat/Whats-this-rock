import time
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend as K
from tensorflow.keras.callbacks import Callback


def get_optimizer(cfg, lr = cfg.lr):
    if cfg.optimizer == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    elif cfg.optimizer == "rms":
        opt = optimizers.RMSprop(learning_rate=lr,
                                 rho=0.9,
                                 epsilon=1e-08,
                                 decay=0.0)
    elif cfg.optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=lr)
    elif cfg.optimizer == "adamax":
        opt = optimizers.Adamax(learning_rate=lr)
    else:
        print(
            f"\n{cfg.optimizer} not present, using default (adam optimizer)\n")
        opt = optimizers.Adam(learning_rate=lr)
    return opt


def get_model_weights_ds(train_ds):
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_ds.class_names),
        y=train_ds.class_names,
    )

    train_class_weights = dict(enumerate(class_weights))
    return train_class_weights
