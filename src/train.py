#!/usr/bin/env python
"""
Trains a model on rocks dataset
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import subprocess
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow_addons as tfa

# speed improvements
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import wandb
from wandb.keras import WandbCallback

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from src.preprocess import process_data
from src.models import get_model
from src.data_utilities import get_tfds_from_dir, prepare
from src.model_utilities import (
    get_optimizer,
    get_model_weights_ds
)
from src.download_data import get_data
from src.builtin_callbacks import get_earlystopper, get_reduce_lr_on_plateau
from src import plot

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config", "configs/baseline.py")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool("log_eval", False,
                  "Log model prediction, needs --wandb argument as well.")


def seed_everything(seed):
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(config, train_dataset, val_dataset, labels):

    tf.keras.backend.clear_session()

    model = get_model(config)
    model.summary()

    print(f"\nModel loaded: {config.model_config.backbone}.\n\n")

    config.train_config.metrics.append(
        tfa.metrics.F1Score(num_classes=config.dataset_config.num_classes,
                            average="macro",
                            threshold=0.5))

    class_weights = None
    if config.train_config.use_class_weights:
        class_weights = get_model_weights_ds(train_dataset)

    optimizer = get_optimizer(config)
    # speed improvements
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=config.train_config.loss,
        metrics=config.train_config.metrics,
    )

    # Define WandbCallback for experiment tracking
    wandbcallback = WandbCallback(
        monitor=config.callback_config.monitor,
        save_model=(config.callback_config.save_model),
    )
    if config.callback_config.use_earlystopping:
        earlystopper = get_earlystopper(config)

    if config.callback_config.use_reduce_lr_on_plateau:
        reduce_lr = get_reduce_lr_on_plateau(config)

    callbacks = [wandbcallback, earlystopper, reduce_lr,]
    verbose=1
    #
    train_dataset = prepare(train_dataset, config, shuffle=True, augment=config.train_config.use_augmentations)
    val_dataset = prepare(val_dataset, config)

    history = model.fit(
        train_dataset,
        epochs=config.train_config.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=-1,
        verbose=verbose,
    )

    return model, history


def evaluate(config, model, history, test_dataset, labels):
    # Scores
    test_dataset = prepare(test_dataset, config)
    scores = model.evaluate(test_dataset, return_dict=True)
    print("Scores: ", scores)

    # Predict
    # TODO (udaylunawat): CM and CR giving wrong results, try changing labels=infered
    y_true = tf.concat([y for x, y in test_dataset], axis=0)
    true_categories = tf.argmax(y_true, axis=1)
    y_pred = model.predict(test_dataset, verbose=1)
    predicted_categories = tf.argmax(y_pred, axis=1)
    # Confusion Matrix
    cm = plot.plot_confusion_matrix(labels, true_categories,
                                    predicted_categories)

    # Classification Report
    cl_report = classification_report(
        true_categories,
        predicted_categories,
        labels=[0, 1, 2, 3, 4, 5, 6],
        target_names=labels,
        output_dict=True,
    )
    print(cl_report)

    cr = sns.heatmap(pd.DataFrame(cl_report).iloc[:-1, :].T, annot=True)
    plt.savefig("cr.png", dpi=400)

    wandb.log({"Test Accuracy": scores["accuracy"]})
    wandb.log({"Test F1 Score": scores["f1_score"]})

    # average of val and test f1 score
    wandb.log({
        "Avg VT F1 Score":
        (scores["f1_score"] + max(history.history["val_f1_score"])) / 2
    })
    wandb.log({"Confusion Matrix": cm})
    wandb.log({
        "Classification Report Image:":
        wandb.Image("cr.png", caption="Classification Report")
    })


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)
    seed_everything(config.seed)

    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            config=config.to_dict(),
            allow_val_change=True,
        )

    artifact = wandb.Artifact('rocks', type='files')
    artifact.add_dir('src/')
    wandb.log_artifact(artifact)

    print(f"\nDatasets used for Training:- {config.dataset_config.train_dataset}")

    for dataset_id in config.dataset_config.train_dataset:
        get_data(dataset_id)

    if not os.path.exists('data/4_tfds_dataset/train'):
        process_data(config)

    train_dataset, val_dataset, test_dataset = get_tfds_from_dir(config)
    labels = [
        "Basalt",
        "Coal",
        "Granite",
        "Limestone",
        "Marble",
        "Quartzite",
        "Sandstone",
    ]
    # TODO (udaylunawat): Update the `num_classes` to train_dataset.classes (not working)
    config.dataset_config.num_classes = 7
    if wandb.run is not None:
        wandb.config.update(
            {"dataset_config.num_classes": config.dataset_config.num_classes})
    model, history = train(config, train_dataset, val_dataset, labels)
    evaluate(config, model, history, test_dataset, labels)

    del model
    _ = gc.collect()

    run.finish()


if __name__ == "__main__":
    app.run(main)
