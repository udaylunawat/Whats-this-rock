#!/usr/bin/env python

"""
Trains a model on the dataset.
Designed to show how to do a simple wandb integration with keras.
"""
# *IMPORANT*: Have to do this line *before* importing tensorflow
from utilities import get_data
from models import get_efficientnet, get_mobilenet, get_baseline_model, model2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow_addons as tfa
from tensorflow.random import set_seed
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from wandb.keras import WandbCallback
import wandb
import random
import pandas as pd
import numpy as np
import argparse
import sys
import os
# os.environ['PYTHONHASHSEED'] = str(1)


# import matplotlib.pyplot as plt


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--notes",
        type=str,
        default=MODEL_NOTES,
        help="Notes about the training run",
        required=False)
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        default=PROJECT_NAME,
        help="Main project name")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model")
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
        "-o",
        "--optimizer",
        type=str,
        default='Adam',
        help="optimizer")
    parser.add_argument(
        "-size",
        "--size",
        type=int,
        default=224,
        help="Image size")
    parser.add_argument(
        "-f",
        "--f1_scoring",
        type=str,
        default='macro',
        help="f1 scoring")
    parser.add_argument(
        "-fm",
        "--fill_mode",
        type=str,
        default='reflect',
        help="augmentation fill_mode")
    parser.add_argument(
        "-hsr",
        "--height_shift_range",
        type=float,
        nargs='+',
        default=0.0,
        help="Augmentation Height Shift Range")
    parser.add_argument(
        "-wsr",
        "--width_shift_range",
        type=float,
        default=0.0,
        help="Augmentation Width Shift Range")
    parser.add_argument(
        "-z",
        "--zoom_range",
        type=float,
        default=0.0,
        help="Augmentation Zoom Range")
    parser.add_argument(
        "-zca",
        "--zca_whitening",
        type=bool,
        default=False,
        help="Augmentation ZCA Whitening")
    parser.add_argument(
        "-ro",
        "--rotation_range",
        type=float,
        default=0.0,
        help="Augmentation Rotation Range")
    parser.add_argument(
        "-aug",
        "--augmentation",
        type=bool,
        default=True,
        help="Augmentation")

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
        type=bool,
        default=False,
        help="Dry run (do not log to wandb)")
    parser.add_argument(
        "-trainable",
        "--pretrained_trainable",
        type=bool,
        default=False,
        help="Train the pretrained model")

    args = parser.parse_args()
    return args


def get_compiled_model(config, model):
    if config.optimizer == 'Adam':
        opt = Adam(config.learning_rate)
    elif config.optimizer == 'RMS':
        opt = RMSprop(learning_rate=config.learning_rate,
                      rho=0.9, epsilon=1e-08, decay=0.0)
    elif config.optimizer == 'SGD':
        opt = SGD(learning_rate=config.learning_rate)

    # enable logging for validation examples
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=[
            tfa.metrics.F1Score(
                num_classes=num_classes,
                average=config.f1_scoring,
                threshold=0.5),
            'accuracy'])
    return model


# https://datamonje.com/image-data-augmentation/#cutout
# https://colab.research.google.com/drive/1on9rvQdr0s8CfqYeZBTbgk0K5I_Dha-X#scrollTo=py9wqsM05kio
def custom_augmentation(np_tensor):
    import tensorflow as tf
    def random_contrast(np_tensor):
        return np.array(tf.image.random_contrast(np_tensor, 0.5, 2))

    def random_hue(np_tensor):
        return np.array(tf.image.random_hue(np_tensor, 0.5))

    def random_saturation(np_tensor):
        return np.array(tf.image.random_saturation(np_tensor, 0.2, 3))

    def random_crop(np_tensor):
        # cropped height between 70% to 130% of an original height
        new_height = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[0])
        # cropped width between 70% to 130% of an original width
        new_width = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[1])
        # resize to new height and width
        cropped = tf.image.resize_with_crop_or_pad(np_tensor, new_height, new_width)
        return np.array(tf.image.resize(cropped, np_tensor.shape[:2]))

    def gaussian_noise(np_tensor):
        mean = 0
        # variance: randomly between 1 to 25
        var = np.random.randint(1, 26)
        # sigma is square root of the variance value
        noise = np.random.normal(mean, var**0.5, np_tensor.shape)
        return np.clip(np_tensor + noise, 0, 255).astype('int')

    def cutout(np_tensor):
        cutout_height = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[0])
        cutout_width = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[1])
        cutout_height_point = np.random.randint(np_tensor.shape[0]-cutout_height)
        cutout_width_point = np.random.randint(np_tensor.shape[1]-cutout_width)
        np_tensor[cutout_height_point:cutout_height_point+cutout_height, cutout_width_point:cutout_width_point+cutout_width, :] = 127
        return np_tensor

    if (np.random.uniform() < 0.1):
        np_tensor = random_contrast(np_tensor)
    if (np.random.uniform() < 0.1):
        np_tensor = random_hue(np_tensor)
    if (np.random.uniform() < 0.1):
        np_tensor = random_saturation(np_tensor)
    if (np.random.uniform() < 0.2):
        np_tensor = random_crop(np_tensor)
    if (np.random.uniform() < 0.2):
        np_tensor = gaussian_noise(np_tensor)
    if (np.random.uniform() < 0.3):
        np_tensor = cutout(np_tensor)
    return np.array(np_tensor)


def get_generators(config, train_data, test_data):
    if config.augmentation is True:

        datagen = ImageDataGenerator(horizontal_flip=True,
                                     featurewise_center=False,
                                     featurewise_std_normalization=False,
                                     zca_whitening=config.zca_whitening,
                                     validation_split=0.2,
                                     fill_mode=config.fill_mode,
                                     zoom_range=[0.8, 1.25],  # config.zoom_range,
                                     brightness_range=[0.5, 1.2],
                                     width_shift_range=0.1,  # config.width_shift_range,
                                     height_shift_range=0.1,  # config.height_shift_range,
                                     rotation_range=30,  # config.rotation_range,
                                     shear_range=30,
                                     rescale=1. / 255.)
    elif config.augmentation is False:
        datagen = ImageDataGenerator(rescale=1. / 255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="./",
        x_col="image_path",
        y_col="classes",
        subset="training",
        batch_size=config.batch_size,
        seed=42,
        color_mode='rgb',
        shuffle=True,
        class_mode="categorical",
        preprocessing_function = custom_augmentation,
        target_size=(
            config.size,
            config.size))

    val_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="./",
        x_col="image_path",
        y_col="classes",
        subset="validation",
        batch_size=config.batch_size,
        seed=42,
        color_mode='rgb',
        shuffle=True,
        class_mode="categorical",
        target_size=(
            config.size,
            config.size))

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory="./",
        x_col="image_path",
        y_col='classes',
        batch_size=config.batch_size,
        validation_split=None,
        seed=42,
        shuffle=False,
        color_mode='rgb',
        class_mode=None,
        target_size=(
            config.size,
            config.size))

    return train_generator, val_generator, test_generator


# def finetune(config, model):
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, config.size, config.size)
#     else:
#         input_shape = (config.size, config.size, 3)

#     feature_extractor = EfficientNetV2B0(
#         include_top=False,
#         weights="imagenet",
#         input_shape=input_shape,
#         classifier_activation="softmax",
#         include_preprocessing=False,
#     )
#     feature_extractor.trainable = True

#     # Fine-tune from this layer onwards
#     fine_tune_at = 150

#     # Freeze all the layers before the `fine_tune_at` layer
#     for layer in feature_extractor.layers[:fine_tune_at]:
#         layer.trainable = False

#     # Add untrained final layers
#     model = Sequential(
#         [
#             feature_extractor,
#             GlobalAveragePooling2D(),
#             Dense(1024),
#             Dense(num_classes, activation="softmax"),
#         ]
#     )

#     model.summary()
#     model.compile(
#         optimizer=Adam(1e-5),
#         loss="categorical_crossentropy",
#         metrics=[
#             tfa.metrics.F1Score(
#                 num_classes=num_classes,
#                 average='weighted',
#                 threshold=0.5),
#             'accuracy'])

#     epochs = 10
#     callbacks = [
#         # ModelCheckpoint("save_at_{epoch}_ft_0_001.h5", save_best_only=True),
#         # EarlyStopping(monitor="val_f1_score", min_delta=0.01, patience=10),
#         WandbCallback(
#             training_data=train_generator,
#             validation_data=val_generator,
#             input_type="image",
#             labels=labels)
#     ]
#     history_tune = model.fit(train_generator,
#                              validation_data=val_generator,
#                              epochs=epochs,
#                              callbacks=callbacks,
#                              initial_epoch=history.epoch[-1])


# make some random data
# reset_random_seeds()

# default config/hyperparameter values
# you can modify these below or via command line
# https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/cnn_train.py

PROJECT_NAME = "rock_classification"
MODEL_NAME = "efficientnet"
SAMPLE_SIZE = 0.2
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 20
SIZE = 224,
TRAINABLE = False
# DROPOUT = 0.2
# L1_SIZE = 16
# L2_SIZE = 32
# HIDDEN_LAYER_SIZE = 128
# DECAY = 1e-6
# MOMENTUM = 0.9
MODEL_NOTES = f'''
Minerals Removed
'''


def SMOTE_Data(train_df):
    from imblearn.over_sampling import SMOTE
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    X_train = []
    for img in train_df['image_path']:
        loaded_img = load_img(os.path.join(IMAGE_DIR, img), target_size=(config.size, config.size))
        img_arr = img_to_array(loaded_img)
        X_train.append(img_arr)

    print(np.array(X_train).shape)
    y_train = train_df.drop('image_path', axis=1, inplace=False)
    print(y_train.head())
    y_train = np.array(y_train.values)
    X_train = np.array(X_train)
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train.reshape((-1, config.size * config.size * 3)), y_train)
    X_train.reshape(-1, config.size, config.size, 3)
    return X_train, y_train


if __name__ == "__main__":
    args = get_parser()

    resume = sys.argv[-1] == "--resume"

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    run = wandb.init(
        project=args.project_name,
        notes=args.notes,
        resume=resume)

    config = args
    print(config)
    # config.optimizer = 'Adam'
    # config.f1_scoring = 'weighted'
    # config.zoom_range = [0.5, 1.0]
    # config.fill_mode = 'reflect'
    # config.width_shift_range = [0, 0.3]
    # config.height_shift_range = [0, 0.3]
    # config.rotation_range = 90
    # config.zca_whitening = False
    # wandb.config.update(config)

    # build model
    if wandb.run.resumed:
        print("RESUMING")
        # restore the best model
        model = load_model(wandb.restore("model-best.h5").name)
    else:
        train_df, test_df = get_data(config.sample_size)


        train_generator, val_generator, test_generator = get_generators(
            config, train_df, test_df)

        num_classes = len(train_generator.class_indices)
        labels = list(train_generator.class_indices.keys())

        if config.model == "efficientnet":
            model = get_efficientnet(config, num_classes)
        elif config.model == "baseline":
            model = baseline_model(config.size, 3, num_classes)
        elif config.model == "baseline_cnn":
            model = model2(config.size, config.size, num_classes)
        elif config.model == "mobilenet":
            model = get_mobilenet(config, num_classes)

    model = get_compiled_model(config, model)
    model.summary()

    class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_generator.classes),
                y=train_generator.classes)

    train_class_weights = dict(enumerate(class_weights))

    callbacks = [WandbCallback(training_data=train_generator, validation_data=val_generator, input_type="image", labels=labels),
                 # ModelCheckpoint("save_at_{epoch}_ft_0_001.h5", save_best_only=True),
                 EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=10)
    ]
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.epochs,
        class_weight=train_class_weights,
        callbacks=callbacks)

    if config.pretrained_trainable:
        finetune(config, model)

        # Confution Matrix and Classification Report
    Y_pred = model.predict(
        val_generator,
        len(val_generator) // config.batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    print('Confusion Matrix with Validation data')
    print(confusion_matrix(val_generator.classes, y_pred))
    print('Classification Report')
    cl_report = classification_report(
        val_generator.classes,
        y_pred,
        target_names=labels,
        output_dict=True)
    # print(pd.DataFrame(cl_report))
    print(cl_report)

    run.finish()
