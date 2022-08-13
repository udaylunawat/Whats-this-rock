#!/usr/bin/env python

"""
Trains a model on the dataset.

Designed to show how to do a simple wandb integration with keras.
"""

from data_utilities import get_data_tfds, prepare_dataset, get_generators
from model_utilities import get_model, get_optimizer


# from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import losses
from tensorflow.keras.backend import clear_session
import tensorflow_addons as tfa


from wandb.keras import WandbCallback
import wandb
import random
import numpy as np
import json
import os
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

    reset_random_seeds()
    print(f"config:- {config}")
    run = wandb.init(config=config)

    config = wandb.config
    IMAGE_SIZE = (config["image_size"], config["image_size"])

    train_dataset, val_dataset = get_generators(config)

    # build model
    clear_session()
    model = get_model(config)

    opt = get_optimizer(config)

    config.metrics.append(tfa.metrics.F1Score(
        num_classes=config.num_classes,
        average='macro',
        threshold=0.5))

    # Notice that we use label_smoothing=0.1 in the loss function.
    # When using MixUp, label smoothing is highly recommended.

    model.compile(loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
                  optimizer=opt,
                  metrics=config["metrics"])

    # model.summary()
    # def decay_schedule(epoch, lr):
    #     # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    #     if (epoch % 5 == 0) and (epoch != 0):
    #         lr = lr * 0.1
    #     return lr

    # lr_scheduler = LearningRateScheduler(decay_schedule)
    model_checkpoint = ModelCheckpoint("checkpoints/"+
                                       f"{wandb.run.name}-"+config["model_name"]+
                                       "-epoch-{epoch}_val_accuracy-{val_accuracy:.2f}.hdf5", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=config['lr_reduce_factor'], patience=config['lr_reduce_patience'], verbose=1)
    earlystopper = EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], verbose=1, mode='auto', min_delta=config['earlystopping_min_delta'],
        restore_best_weights=True
    )

    wandbcallback = WandbCallback()

    # Define WandbCallback for experiment tracking

    callbacks = [wandbcallback, earlystopper, model_checkpoint, reduce_lr]

    history = model.fit(
        train_dataset,
        epochs=config["max_epochs"],
        validation_data=val_dataset,
        callbacks=callbacks,
        workers=-1
    )

    # Confusion Matrix and Classification Report
    # Y_pred = model.predict(
    #     val_generator,
    #     len(val_generator) // config.batch_size + 1)
    # y_pred = np.argmax(Y_pred, axis=1)

    # print('Confusion Matrix with Validation data')
    # print(confusion_matrix(val_generator.classes, y_pred))
    # print('Classification Report')
    # cl_report = classification_report(
    #     val_generator.classes,
    #     y_pred,
    #     target_names=labels,
    #     output_dict=True)
    # # print(pd.DataFrame(cl_report))
    # print(cl_report)

    run.finish()