#!/usr/bin/env python

"""
Trains a model on the dataset.
Designed to show how to do a simple wandb integration with keras.
"""
from data_utilities import get_data_tfds, prepare_dataset
from model_utilities import get_model, get_optimizer
from augment_utilities import apply_rand_augment, cut_mix_and_mix_up, preprocess_for_model

# from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
import tensorflow_addons as tfa
from tensorflow.data import AUTOTUNE

from wandb.keras import WandbCallback
import wandb
import random
import numpy as np
import json
import argparse
import sys
import os
# import matplotlib.pyplot as plt

# *IMPORANT*: Have to do this line *before* importing tensorflow
# os.environ['PYTHONHASHSEED'] = str(1)

config = dict(
    root_dir="data/4_tfds_dataset",
    project_name="rock-classification-with-keras-cv",
    model_name="resnet",
    num_classes=3,
    sample_size=1.0,
    augment=True,
    optimizer="adam",
    init_learning_rate=0.0001,
    batch_size=64,
    max_epochs=5,
    image_size=224,
    trainable=False,
    # lr_decay_rate = 0.7,
    loss_fn="categoricalcrossentropy",
    metrics=['accuracy'],
    earlystopping_patience=5,
    lr_reduce_patience=20,
    notes="keras-cv augment run"
)


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


def load_dataset(split="train"):
    dataset = data[split]
    return prepare_dataset(dataset, split)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--notes",
        type=str,
        default="",
        help="Notes about the training run",
        required=False)
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        default=config.project_name,
        help="Main project name")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=config.model_name,
        help="Model")
    parser.add_argument(
        "-sample",
        "--sample_size",
        type=float,
        default=config.sample_size,
        help="sample_size")
    parser.add_argument(
        "-init_lr",
        "--init_learning_rate",
        type=float,
        default=config.init_learning_rate,
        help="learning rate")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=config.batch_size,
        help="batch_size")
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        default=config.max_epochs,
        help="number of training epochs (passes through full training data)")
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default=config.optimizer,
        help="optimizer")
    parser.add_argument(
        "-size",
        "--image_size",
        type=int,
        default=config.image_size,
        help="Image size")
    parser.add_argument(
        "-q",
        "--dry_run",
        default=False,
        action='store_true',
        help="Dry run (do not log to wandb)")
    parser.add_argument(
        "-trainable",
        "--pretrained_trainable",
        action='store_true',
        default=False,
        help="Train the pretrained model")

    args = parser.parse_args()
    return args


# save dictionary to config.json file
with open('config.json', 'w') as f:
    json.dump(config, f)

from box import Box
# using addict to allow for easy access to dictionary keys using dot notation
config = Box(config)


if __name__ == "__main__":

    reset_random_seeds()

    args = get_parser()
    resume = sys.argv[-1] == "--resume"

    print("Arguments:", args)
    print(f"\n\nConfig before update: {config}\n\n")

    # combine args and config
    config.update(vars(args))
    print(f"\n\nConfig after update: {config}\n\n")

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    run = wandb.init(
        project=args.project_name,
        entity='udaylunawat',
        notes=args.notes,
        resume=resume)

    # # if arguments are passed in, override config
    # if len(sys.argv) > 1:
    #     config = args

    wandb.config.update(config)

    # build model
    if wandb.run.resumed:
        print("RESUMING")
        # restore the best model
        model = load_model(wandb.restore("model-best.h5").name)
    else:
        data, builder = get_data_tfds()

        num_classes = builder.info.features['label'].num_classes
        config["num_classes"] = num_classes

        IMAGE_SIZE = (config["image_size"], config["image_size"])

        # AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = (
            load_dataset()
            .map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
            .map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)
        )

        # visualize_dataset(train_dataset, "CutMix, MixUp and RandAugment")

        train_dataset = train_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

        val_dataset = load_dataset(split="val")
        val_dataset = val_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

        test_dataset = load_dataset(split="test")
        test_dataset = test_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

        labels = builder.info.features['label'].names

        # build model
        clear_session()
        model = get_model(config, config.num_classes)

    opt = get_optimizer(config)

    config.metrics.append(tfa.metrics.F1Score(
        num_classes=config.num_classes,
        average='macro',
        threshold=0.5))

    # Notice that we use label_smoothing=0.1 in the loss function.
    # When using MixUp, label smoothing is highly recommended.
    model.compile(
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=opt,
        metrics=config["metrics"],
    )

    # model.summary()
    def decay_schedule(epoch, lr):
        # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
        if (epoch % 5 == 0) and (epoch != 0):
            lr = lr * 0.1
        return lr

    lr_scheduler = LearningRateScheduler(decay_schedule)
    # model_checkpoint = ModelCheckpoint("checkpoints/"+
    #                                    f"{wandb.run.name}-"+config["model_name"]+
    #                                    "-epoch-{epoch}_val_accuracy-{val_accuracy:.2f}.hdf5", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=config.lr_reduce_patience, verbose=1)
    earlystopper = EarlyStopping(
        monitor='val_loss', patience=config['earlystopping_patience'], verbose=1, mode='auto',
        restore_best_weights=True
    )

    wandbcallback = WandbCallback(training_data=train_dataset,
                                  labels=labels,
                                  save_model=True,
                                  monitor='val_loss',
                                  log_weights=True)

    # Define WandbCallback for experiment tracking

    callbacks = [earlystopper, reduce_lr, wandbcallback]
    history = model.fit(
        train_dataset,
        epochs=config["max_epochs"],
        validation_data=val_dataset,
        callbacks=callbacks
    )

    if config.pretrained_trainable:
        print("Not implemented")
        pass

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
