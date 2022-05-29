#!/usr/bin/env python

"""
Trains an EfficientNet model on the given dataset.
Deigned to show how to do a simple wandb integration with keras.
"""
import argparse
import os
import numpy as np
import pandas as pd
import random

from sklearn.metrics import classification_report

# *IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED'] = str(1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.random import set_seed
import tensorflow_addons as tfa

import wandb
from wandb.keras import WandbCallback

from utilities import get_stratified_dataset_partitions_pd
import plot


def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


#make some random data
reset_random_seeds()

# default config/hyperparameter values
# you can modify these below or via command line
# https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/cnn_train.py

PROJECT_NAME = "rock_classification"
SAMPLE_SIZE = 0.1
LEARNING_RATE = 0.00001
BATCH_SIZE = 512
EPOCHS = 50
TRAINABLE = False
# DROPOUT = 0.2
# L1_SIZE = 16
# L2_SIZE = 32
# HIDDEN_LAYER_SIZE = 128
# DECAY = 1e-6
# MOMENTUM = 0.9
MODEL_NOTES = f'''
EfficientNetV2B0
f1 macro, {SAMPLE_SIZE}% sample_size
efficientnet_pretrained.trainable = True
'''

# dataset config
img_width = 224
img_height = 224

num_classes = 4


def split_and_stratify_data(args):
    data = pd.read_csv('training_data.csv', index_col=0)
    data = data.sample(frac=args.sample_size).reset_index(drop=True)
    # Splitting data into train, val and test samples using stratified splits
    train_df, val_df, test_df = get_stratified_dataset_partitions_pd(data, 0.8, 0.1, 0.1)
    return train_df, val_df, test_df


def build_model(args, num_classes):
    """ Construct a simple categorical CNN following the Keras tutorial """
    if K.image_data_format() == 'channels_first':
        input_shape = (3, args.size, args.size)
    else:
        input_shape = (args.size, args.size, 3)

    efficientnet_pretrained = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        classifier_activation="softmax",
        include_preprocessing=False,
    )

    # Freeze layers
    efficientnet_pretrained.trainable = args.pretrained_trainable
    num_classes = num_classes
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
                  metrics=[tfa.metrics.F1Score(num_classes=num_classes, average='macro', threshold=0.5),
                           'accuracy'])
    return model


def train_efficientnet(args):
    # initialize wandb logging to your project
    run = wandb.init(project=args.project_name, notes=args.notes)
    # config = wandb.config
    # log all experimental args to wandb
    wandb.config.update(args)

    train_df, val_df, test_df = split_and_stratify_data(args)

    # datagen = ImageDataGenerator(horizontal_flip=args['aug_horizontal_flip'],
    #                              validation_split=args['aug_validation_split'],
    #                              fill_mode=args['aug_fill_mode'],
    #                              zoom_range=args['aug_zoom_range'],
    #                              brightness_range=args['aug_brightness_range'],
    #                              width_shift_range=args['aug_width_shift_range'],
    #                              height_shift_range=args['aug_height_shift_range'],
    #                              rotation_range=args['aug_rotation_range'],
    #                              rescale=args['aug_rescale'])

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 validation_split=0,
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
                                                  batch_size=args.batch_size,
                                                  seed=42,
                                                  color_mode='rgb',
                                                  shuffle=True,
                                                  class_mode="categorical",
                                                  target_size=(args.size, args.size))

    val_generator = datagen.flow_from_dataframe(dataframe=val_df,
                                                directory="./",
                                                x_col="image_path",
                                                y_col="classes",
                                                subset="validation",
                                                batch_size=args.batch_size,
                                                seed=42,
                                                color_mode='rgb',
                                                shuffle=True,
                                                class_mode="categorical",
                                                target_size=(args.size, args.size))

    test_datagen = ImageDataGenerator(featurewise_center=False,
                                      featurewise_std_normalization=False,
                                      rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      directory="./",
                                                      x_col="image_path",
                                                      y_col='classes',
                                                      batch_size=1,
                                                      seed=42,
                                                      shuffle=False,
                                                      class_mode=None,
                                                      target_size=(args.size, args.size))

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    model = build_model(args, num_classes)
    # Save best checkpoints and stop early to save time
    callbacks = [
        ModelCheckpoint("save_at_{epoch}_ft_0_001.h5", save_best_only=True),
        EarlyStopping(monitor="f1_score", min_delta=0, patience=10),
        WandbCallback(input_type="image", labels=labels, generator=val_generator)
    ]
    model.fit(train_generator, validation_data=val_generator, epochs=args.epochs, callbacks=callbacks)

    scores = model.evaluate_generator(generator=test_generator)
    print('Accuracy: ', scores)

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    pred = model.predict_generator(test_generator, steps=nb_samples, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    test_acc = sum([predicted_class_indices[i] == test_generator.classes[i] for i in range(len(test_df))]) / len(test_df)
    # Confution Matrix and Classification Report
    # print('Confusion Matrix')
    # print(confusion_matrix(test_generator.classes, predicted_class_indices))

    confusion_matrix = plot.confusion_matrix(labels, test_generator.classes, predicted_class_indices)
    wandb.log({"test_accuracy": test_acc, "Confusion Matrix": confusion_matrix})

    cl_report = classification_report(test_generator.classes, predicted_class_indices)
    print('Classification Report')
    print(cl_report)
    wandb.log({"test_accuracy": test_acc, "Classification Report": cl_report})
    run.finish()
    # save trained model
    # model.save(args.model_name + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--notes",
        type=str,
        default=MODEL_NOTES,
        help="Notes about the training run")
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        default=PROJECT_NAME,
        help="Main project name")
    parser.add_argument(
        "-sample",
        "--sample_size",
        type=float,
        default=SAMPLE_SIZE,
        help="sample_size")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="learning rate")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="batch_size")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help="number of training epochs (passes through full training data)")
    parser.add_argument(
        "-size",
        "--size",
        type=int,
        default=224,
        help="Image size")
    # parser.add_argument(
    #   "--dropout",
    #   type=float,
    #   default=DROPOUT,
    #   help="dropout before dense layers")
    #   parser.add_argument(
    #     "--hidden_layer_size",
    #     type=int,
    #     default=HIDDEN_LAYER_SIZE,
    #     help="hidden layer size")
    #   parser.add_argument(
    #     "-l1",
    #     "--layer_1_size",
    #     type=int,
    #     default=L1_SIZE,
    #     help="layer 1 size")
    #   parser.add_argument(
    #     "-l2",
    #     "--layer_2_size",
    #     type=int,
    #     default=L2_SIZE,
    #     help="layer 2 size")
    #   parser.add_argument(
    #     "--decay",
    #     type=float,
    #     default=DECAY,
    #     help="learning rate decay")
    #   parser.add_argument(
    #     "--momentum",
    #     type=float,
    #     default=MOMENTUM,
    #     help="learning rate momentum")
    parser.add_argument(
        "-q",
        "--dry_run",
        action="store_true",
        help="Dry run (do not log to wandb)")
    parser.add_argument(
        "-trainable",
        "--pretrained_trainable",
        action="store_true",
        default=TRAINABLE,
        help="Train the pretrained model")

    args = parser.parse_args()

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    train_efficientnet(args)
