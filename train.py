#!/usr/bin/env python

"""
Trains a simple cnn on the fashion mnist dataset.
Designed to show how to do a simple wandb integration with keras.
Also demonstrates resumption of interrupted runs with
    python train.py --resume
"""
import sys

from keras.models import Sequential, load_model

import wandb
from wandb.keras import WandbCallback


defaults = dict(learning_rate=0.01,
                epochs=30,
                size=224,
                pretrained_trainable=False,
                batch_size=64,
                optimizer='adam',
                sample_size=0.2)

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
    import pandas as pd
    from utilities import get_stratified_dataset_partitions_pd
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    data = pd.read_csv('training_data.csv', index_col=0)
    data = data.sample(frac=config.sample_size).reset_index(drop=True)
    # Splitting data into train, val and test samples using stratified splits
    train_df, val_df, test_df = get_stratified_dataset_partitions_pd(data, 0.8, 0.1, 0.1)
    train_df = pd.concat([train_df, val_df])

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 validation_split=0.2,
                                 fill_mode="nearest",
                                 zoom_range=0,
                                 brightness_range=[0.4, 1.5],
                                 width_shift_range=0,
                                 height_shift_range=0,
                                 rotation_range=0,
                                 rescale=1. / 255.)

    train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                  directory="./",
                                                  x_col="image_path",
                                                  y_col="classes",
                                                  subset="training",
                                                  batch_size=config.batch_size,
                                                  seed=42,
                                                  color_mode='rgb',
                                                  shuffle=True,
                                                  class_mode="categorical",
                                                  target_size=(config.size, config.size))

    val_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                directory="./",
                                                x_col="image_path",
                                                y_col="classes",
                                                subset="validation",
                                                batch_size=config.batch_size,
                                                seed=42,
                                                color_mode='rgb',
                                                shuffle=True,
                                                class_mode="categorical",
                                                target_size=(config.size, config.size))

    test_datagen = ImageDataGenerator(featurewise_center=False,
                                      featurewise_std_normalization=False,
                                      rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      directory="./",
                                                      x_col="image_path",
                                                      y_col='classes',
                                                      batch_size=config.batch_size,
                                                      validation_split=None,
                                                      seed=42,
                                                      shuffle=False,
                                                      color_mode='rgb',
                                                      class_mode=None,
                                                      target_size=(config.size, config.size))

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

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

    opt = config.optimizer

    # enable logging for validation examples
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=[tfa.metrics.F1Score(num_classes=num_classes, average='weighted', threshold=0.5),
                           'accuracy'])
callbacks = [  # ModelCheckpoint("save_at_{epoch}_ft_0_001.h5", save_best_only=True),
             EarlyStopping(monitor="f1_score", min_delta=0, patience=10),
             WandbCallback(training_data=train_generator, validation_data=val_generator, input_type="image", labels=labels)]
model.fit(train_generator, validation_data=val_generator, epochs=config.epochs, callbacks=callbacks)
