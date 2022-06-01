#!/usr/bin/env python

"""
Trains a model on the dataset.
Designed to show how to do a simple wandb integration with keras.
"""
from data_utilities import get_data
from model_utilities import get_generators, finetune, get_model, get_optimizer

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.addons import tfa

from wandb.keras import WandbCallback
import wandb
import random
import numpy as np
import json
import argparse
import sys
import os
import shutil
# import matplotlib.pyplot as plt

# *IMPORANT*: Have to do this line *before* importing tensorflow
# os.environ['PYTHONHASHSEED'] = str(1)

with open('config.json') as f:
    config = json.load(f)


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
        "-remove",
        "--remove_class",
        type=str,
        default=None,
        help="Root Folder")
    parser.add_argument(
        "-m",
        "--model",
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
        "-lr",
        "--learning_rate",
        type=float,
        default=config.learning_rate,
        help="learning rate")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=config.batch_size,
        help="batch_size")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=config.epochs,
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
        default=config.image_size,
        help="Image size")
    # https://stackoverflow.com/a/60999928/9292995
    parser.add_argument(
        "-aug",
        "--augmentation",
        action='store_true',
        help="Augmentation")
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


if __name__ == "__main__":

    reset_random_seeds()

    args = get_parser()
    resume = sys.argv[-1] == "--resume"

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    run = wandb.init(
        project=args.project_name,
        notes=args.notes,
        resume=resume)

    # if arguments are passed in, override config
    if len(sys.argv) == 1:
        config = args
    else:
        wandb.config.update(config)

    print(f'Augmentation - {config.augmentation is False}')

    if args.remove:
        shutil.rmtree(os.path.join(args.root, args.remove))

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

        model = get_model(config, num_classes)

    opt = get_optimizer(config, model)

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=[
            tfa.metrics.F1Score(
                num_classes=num_classes,
                average=config.f1_scoring,
                threshold=0.5),
            'accuracy'])
    model.summary()

    # model_checkpoint = ModelCheckpoint("save_at_{epoch}_ft_0_001.h5", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0)
    early_stop = EarlyStopping(monitor="val_loss", patience=7, verbose=0, restore_best_weights=True)
    wandbcallback = WandbCallback(training_data=train_generator, validation_data=val_generator, input_type="image", labels=labels)

    callbacks = [reduce_lr, early_stop, wandbcallback]
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.epochs,
        # class_weight=train_class_weights,
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
