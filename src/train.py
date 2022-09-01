#!/usr/bin/env python
"""
Trains a model on rocks dataset
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models import get_model
from data_utilities import get_generators
from model_utilities import (
    get_optimizer,
    get_model_weights,
    LRA,
)
import plot

from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow_addons as tfa

import wandb
from wandb.keras import WandbCallback

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config", "configs/baseline.py")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool(
    "log_eval", False, "Log model prediction, needs --wandb argument as well."
)


def seed_everything(seed):
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(config, train_dataset, val_dataset, labels):

    tf.keras.backend.clear_session()
    # file_name = "model-best.h5"
    # if config["finetune"]:
    #     if os.path.exists(file_name):
    #         os.remove("model-best.h5")

    #     api = wandb.Api()
    #     run = api.run(
    #         config["pretrained_model_link"]
    #     )  # different-sweep-34-efficientnet-epoch-3-val_f1_score-0.71.hdf5
    #     run.file(file_name).download()
    #     model = tf.keras.models.load_model(file_name)
    #     pml = config["pretrained_model_link"]
    #     print(f"Downloaded Trained model: {pml},\nfinetuning...")
    # else:
    #     # build model
    #     model = get_model(config)
    model = get_model(config)
    model.summary()

    print(f"Model loaded: {config.model_config.backbone}.\n\n")

    config.train_config.metrics.append(
        tfa.metrics.F1Score(
            num_classes=config.dataset_config.num_classes, average="macro", threshold=0.5
        )
    )

    class_weights = get_model_weights(train_dataset)

    opt = get_optimizer(config)
    # Compile the model
    model.compile(
        optimizer=config.train_config.optimizer,
        loss=config.train_config.loss,
        metrics=config.train_config.metrics,
    )

    # Define WandbCallback for experiment tracking
    wandbcallback = WandbCallback(
        monitor="val_f1_score",
        save_model=(True),
        save_graph=(False),
        log_evaluation=True,
        generator=val_dataset,
        validation_steps=val_dataset.samples // config.dataset_config.batch_size,
    )
    # callbacks = [wandbcallback, earlystopper, model_checkpoint, reduce_lr,]
    # verbose=1
    callbacks = [
        LRA(
            wandb=wandb,
            model=model,
            patience=config.callback_config.rlrp_patience,
            stop_patience=config.callback_config.early_patience,
            threshold=0.9,
            factor=config.callback_config.rlrp_factor,
            dwell=False,
            model_name=config.model_config.backbone,
            freeze=False,
            initial_epoch=0,
        ),
        wandbcallback,
    ]
    verbose = 0
    LRA.tepochs = config.train_config.epochs  # used to determine value of last epoch for printing

    # TODO (udaylunawat): add steps_per_epoch and validation_steps
    history = model.fit(
        train_dataset,
        # steps_per_epoch=train_dataset.samples // config.dataset_config.batch_size,
        epochs=config.train_config.epochs,
        validation_data=val_dataset,
        # validation_steps=val_dataset.samples // config.dataset_config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=-1,
        verbose=verbose,
    )

    return model, history


def evaluate(config, model, history, test_dataset, labels):
    # Scores
    scores = model.evaluate(test_dataset, return_dict=True)
    print("Scores: ", scores)

    # Predict
    pred = model.predict(test_dataset, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    # Confusion Matrix
    cm = plot.plot_confusion_matrix(
        labels, test_dataset.classes, predicted_class_indices
    )

    # Classification Report
    cl_report = classification_report(
        test_dataset.classes,
        predicted_class_indices,
        labels=[0, 1, 2, 3, 4, 5, 6],
        target_names=labels,
        output_dict=True,
    )
    print(cl_report)

    cr = sns.heatmap(pd.DataFrame(cl_report).iloc[:-1, :].T, annot=True)
    os.makedirs('imgs')
    plt.savefig("imgs/cr.png", dpi=400)

    wandb.log({"Test Accuracy": scores["accuracy"]})
    wandb.log({"Test F1 Score": scores["f1_score"]})

    # average of val and test f1 score
    wandb.log(
        {
            "Avg VT F1 Score": (
                scores["f1_score"] + max(history.history["val_f1_score"])
            )
            / 2
        }
    )
    wandb.log({"Confusion Matrix": cm})
    wandb.log(
        {
            "Classification Report Image:": wandb.Image(
                "imgs/cr.png", caption="Classification Report"
            )
        }
    )


def main(_):
    # Get configs from the config file.
    config = CONFIG.value

    seed_everything(config.seed)

    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            config=config.to_dict(),
            allow_val_change=True,
        )

    train_dataset, val_dataset, test_dataset = get_generators(config)
    labels = [
        "Basalt",
        "Coal",
        "Granite",
        "Limestone",
        "Marble",
        "Quartzite",
        "Sandstone",
    ]
     ## Update the `num_classes` and update wandb config
    config.dataset_config.num_classes = len(set(train_dataset.classes))
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
