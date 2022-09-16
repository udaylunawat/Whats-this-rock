#!/usr/bin/env python
"""
Trains a model on rocks dataset
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import subprocess
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report

# speed improvements
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

import wandb

# from absl import app  # app.run(main)
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.data.preprocess import process_data
from src.models.models import get_model
from src.data.utils import get_tfds_from_dir, prepare
from src.models.utils import get_optimizer, get_model_weights_ds
from src.data.download import get_data
from src.callbacks.callbacks import get_callbacks
from src.visualization import plot


def seed_everything(seed):
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def unfreeze_model(cfg, model):
    model.layers[0].trainable = True
    # model.trainable = True
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[0].layers[-20:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    print("\nFinetuning model with BatchNorm layers freezed.\n")
    for layer in model.layers[0].layers[-20:]:
        print(layer.name, layer.trainable)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # model.compile(optimizer=optimizer,
    #               loss="categorical_crossentropy",
    #               metrics=["accuracy"])
    print('\n')
    model.summary()

    return model

def train(cfg, train_dataset, val_dataset, class_weights):

    tf.keras.backend.clear_session()

    model = get_model(cfg)
    model.summary()

    print(f"\nModel loaded: {cfg.backbone}.\n\n")

    optimizer = get_optimizer(cfg)
    optimizer = mixed_precision.LossScaleOptimizer(
        optimizer)  # speed improvements

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=cfg.loss,
        metrics=["accuracy",],
    )

    callbacks = get_callbacks(cfg)

    verbose = 1

    history = model.fit(
        train_dataset,
        epochs=cfg.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=-1,
        verbose=verbose,
    )

    return model, history


def evaluate(cfg, model, history, test_dataset, labels):
    # Scores
    test_dataset = prepare(test_dataset, cfg)
    scores = model.evaluate(test_dataset, return_dict=True)
    print("Scores: ", scores)

    # Predict

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

    wandb.log({"Confusion Matrix": cm})
    wandb.log({
        "Classification Report Image:":
        wandb.Image("cr.png", caption="Classification Report")
    })


@hydra.main(config_path="../../configs/",
            config_name="config.yaml",
            version_base='1.2')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.data_path)
    seed_everything(cfg.seed)

    if cfg.use_wandb:
        run = wandb.init(project=cfg.project,
                         notes=cfg.notes,
                         config=OmegaConf.to_container(cfg,
                                                       resolve=True,
                                                       throw_on_missing=True))

    artifact = wandb.Artifact('rocks', type='files')
    artifact.add_dir('src/')
    wandb.log_artifact(artifact)

    print(f"\nDatasets used for Training:- {cfg.dataset_id}")

    subprocess.run(['sh', 'src/scripts/clean_dir.sh'],
                   stdout=subprocess.PIPE).stdout.decode('utf-8')
    for dataset_id in cfg.dataset_id:
        get_data(dataset_id)

    process_data(cfg)

    train_dataset, val_dataset, test_dataset = get_tfds_from_dir(cfg)
    labels = train_dataset.class_names
    cfg.num_classes = len(labels)

    class_weights = None
    if cfg.class_weights:
        class_weights = get_model_weights_ds(train_dataset)

    train_dataset = prepare(train_dataset,
                            cfg,
                            shuffle=True,
                            augment=cfg.augmentation)
    val_dataset = prepare(val_dataset, cfg)
    model, history = train(cfg, train_dataset, val_dataset, class_weights)

    if cfg.trainable == False:
        model = unfreeze_model(cfg, model)
        epochs = cfg.epochs + 10
        callbacks = get_callbacks(cfg)
        history = model.fit(train_dataset,
                         epochs=epochs,
                         validation_data=val_dataset,
                         callbacks=callbacks,
                         initial_epoch=len(history.history['loss']),
                         verbose=2)

    evaluate(cfg, model, history, test_dataset, labels)
    run.finish()


if __name__ == "__main__":
    main()
