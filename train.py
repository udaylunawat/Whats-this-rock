#!/usr/bin/env python

"""
Trains a model on the dataset.

Designed to show how to do a simple wandb integration with keras.
"""

from data_utilities import get_data_tfds, prepare_dataset, get_generators
from model_utilities import get_model, get_optimizer, get_best_checkpoint, get_model_weights, delete_checkpoints, LRA


# from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import losses
from tensorflow.keras.backend import clear_session
import tensorflow_addons as tfa
from keras.models import load_model

import wandb
from wandb.keras import WandbCallback

import os
import argparse
import random
import numpy as np
import json


# import matplotlib.pyplot as plt

# *IMPORANT*: Have to do this line *before* importing tensorflow
# os.environ['PYTHONHASHSEED'] = str(1)


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


# read config file
with open('config.json') as config_file:
    config = json.load(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--notes",
        type=str,
        required=True,
        help="Wandb Notes")
    args = parser.parse_args()
    reset_random_seeds()

    print(f"config:- {json.dumps(config, indent=2)}")
    run = wandb.init(config=config, allow_val_change=True)

    config = wandb.config
    IMAGE_SIZE = (config["image_size"], config["image_size"])

    train_dataset, val_dataset = get_generators(config)
    labels = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']
    class_weights = get_model_weights(train_dataset)

    # build model
    clear_session()
    model = get_model(config)

    best_model = get_best_checkpoint()
    if best_model:
        if best_model.split('-')[3] == config['model_name']:
            print(f"Loading {best_model}.")
            model = load_model(os.path.join('checkpoints', best_model))

    opt = get_optimizer(config)

    config.metrics.append(tfa.metrics.F1Score(
        num_classes=config.num_classes,
        average='macro',
        threshold=0.5))

    # Notice that we use label_smoothing=0.1 in the loss function.
    # When using MixUp, label smoothing is highly recommended.

    model.compile(loss=config['loss_fn'],
                  optimizer=opt,
                  metrics=config["metrics"])
    model_checkpoint = ModelCheckpoint("checkpoints/"+
                                       f"{wandb.run.name}-"+config["model_name"]+
                                       "-epoch-{epoch}_val_accuracy-{val_accuracy:.2f}.hdf5", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=config['lr_reduce_factor'], patience=config['lr_reduce_patience'], verbose=1)
    earlystopper = EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], verbose=1, mode='auto', min_delta=config['earlystopping_min_delta'],
        restore_best_weights=True
    )
    # Define WandbCallback for experiment tracking
    wandbcallback = WandbCallback(monitor="val_loss",
                                  training_data=train_dataset,
                                  validation_data=val_dataset,
                                  input_type="images",
                                  output_type="label",
                                  labels=labels,
                                  save_model=(False),
                                  save_graph=(False)
                                  )
    # callbacks = [wandbcallback, earlystopper, model_checkpoint, reduce_lr, delete_checkpoints()]
    callbacks=[LRA(model=model,patience=config.lr_reduce_patience, stop_patience=config.earlystopping_patience , threshold=.9,
    factor=config.lr_reduce_factor,dwell=False, model_name=config.model_name, freeze=False, initial_epoch=0),
    wandbcallback]
    LRA.tepochs=config.max_epochs  # used to determine value of last epoch for printing

    history = model.fit(
        train_dataset,
        epochs=config["max_epochs"],
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=-1,
        verbose=0,
    )

    # scores = model.evaluate_generator(generator=test_generator)
    # print('Accuracy: ', scores)

    # filenames = test_generator.filenames
    # nb_samples = len(filenames)
    # pred = model.predict_generator(test_generator, steps=nb_samples, verbose=1)
    # predicted_class_indices = np.argmax(pred, axis=1)
    # test_acc = sum([predicted_class_indices[i] == test_generator.classes[i] for i in range(len(test_df))]) / len(test_df)
    # # Confution Matrix and Classification Report
    # # print('Confusion Matrix')
    # # print(confusion_matrix(test_generator.classes, predicted_class_indices))

    # confusion_matrix = plot.confusion_matrix(labels, test_generator.classes, predicted_class_indices)
    # wandb.log({"test_accuracy": test_acc, "Confusion Matrix": confusion_matrix})

    # cl_report = classification_report(test_generator.classes, predicted_class_indices)
    # print('Classification Report')
    # print(cl_report)
    # wandb.log({"test_accuracy": test_acc, "Classification Report": cl_report})

    run.finish()