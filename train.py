#!/usr/bin/env python

"""
Trains a simple cnn on the fashion mnist dataset.
Designed to show how to do a simple wandb integration with keras.
Also demonstrates resumption of interrupted runs with
    python train.py --resume
"""
import sys

from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD

import wandb
from wandb.keras import WandbCallback


defaults = dict(
    dropout=0.2,
    hidden_layer_size=128,
    layer_1_size=16,
    layer_2_size=32,
    learn_rate=0.01,
    decay=1e-6,
    momentum=0.9,
    epochs=27,
    )

resume = sys.argv[-1] == "--resume"
wandb.init(config=defaults, resume=resume)
config = wandb.config



# build model
if wandb.run.resumed:
    print("RESUMING")
    # restore the best model
    model = load_model(wandb.restore("model-best.h5").name)
else:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications import EfficientNetV2B0
    import tensorflow_addons as tfa

    if K.image_data_format() == 'channels_first':
        input_shape = (3, config.size, config.size)
    else:
        input_shape = (config.size, config.size, 3)

    efficientnet_pretrained = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        classifier_activation="softmax",
        include_preprocessing=False,
    )

    # Freeze layers
    efficientnet_pretrained.trainable = config.pretrained_trainable
    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())
    # Add untrained final layers
    model = Sequential(
        [
            efficientnet_pretrained,
            GlobalAveragePooling2D(),
            Dense(1024),
            Dense(num_classes, activation="softmax"),
        ]
    )

    opt = Adam(args.learning_rate)

    # enable logging for validation examples
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=[tfa.metrics.F1Score(num_classes=num_classes, average='weighted', threshold=0.5),
                           'accuracy'])
callbacks = [
        # ModelCheckpoint("save_at_{epoch}_ft_0_001.h5", save_best_only=True),
        # EarlyStopping(monitor="f1_score", min_delta=0, patience=10),
        WandbCallback(training_data=train_generator, validation_data=val_generator, input_type="image", labels=labels)
    ]
model.fit(train_generator, validation_data=val_generator, epochs=config.epochs, callbacks=callbacks)
