#!/usr/bin/env python

"""
Trains a model on the dataset.

Designed to show how to do a simple wandb integration with keras.
"""

from data_utilities import get_generators
from model_utilities import get_model, get_optimizer, get_best_checkpoint, get_model_weights, LRA
import plot

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

import wandb
from wandb.keras import WandbCallback

import os
import random
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt

# *IMPORANT*: Have to do this line *before* importing tensorflow
# os.environ['PYTHONHASHSEED'] = str(1)


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


class custom_callback(Callback):
    """log lr and clear checkpoints."""

    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))  # get the current learning rate
        wandb.log({'lr': lr})
        max = 0
        for file_name in os.listdir('checkpoints'):
            val_acc = int(os.path.basename(file_name).split('.')[-2])
            if val_acc > max:
                max = val_acc
            if val_acc < max:
                os.remove(os.path.join('checkpoints', file_name))


# read config file
with open('config.json') as config_file:
    default = json.load(config_file)


if __name__ == "__main__":

    reset_random_seeds()
    print(f"Default config:- {json.dumps(default, indent=2)}\n P.S - Not used in sweeps.\n\n")
    run = wandb.init(project="Whats-this-rock",
                     entity="rock-classifiers",
                     config=default, allow_val_change=True)
    config = wandb.config
    IMAGE_SIZE = (config["image_size"], config["image_size"])

    train_dataset, val_dataset, test_dataset = get_generators(config)
    labels = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']
    class_weights = get_model_weights(train_dataset)

    # build model
    K.clear_session()
    model = get_model(config)

    # model = load_model('checkpoints/visionary-sweep-10-efficientnet-epoch-2-val_f1_score-0.65.hdf5')

    api = wandb.Api()
    run = api.run("rock-classifiers/Whats-this-rock/3hvgnqas")
    run.file("model-best.h5").download()
    model = load_model('model-best.h5')

    print(f"Model loaded: {model.name}")
    opt = get_optimizer(config)

    config['metrics'].append(tfa.metrics.F1Score(
        num_classes=config['num_classes'],
        average='macro',
        threshold=0.5))

    model.compile(loss=config['loss_fn'],
                  optimizer=opt,
                  metrics=config["metrics"])
    model_checkpoint = ModelCheckpoint("checkpoints/"+f"{wandb.run.name}-"+config["model_name"]+
                                       "-epoch-{epoch}-val_f1_score-{val_f1_score:.2f}.hdf5", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_f1_score", factor=config['lr_reduce_factor'], patience=config['lr_reduce_patience'], verbose=1, min_lr=0.0000001)
    earlystopper = EarlyStopping(
        monitor='val_f1_score', patience=config['earlystopping_patience'], verbose=1, mode='auto', min_delta=config['earlystopping_min_delta'],
        restore_best_weights=True
    )
    # Define WandbCallback for experiment tracking
    wandbcallback = WandbCallback(monitor="val_f1_score",
                                  save_model=(True),
                                  save_graph=(False)
                                  )
    # callbacks = [wandbcallback, earlystopper, model_checkpoint, reduce_lr, custom_callback()]
    callbacks = [LRA(wandb=wandb, model=model, patience=config['lr_reduce_patience'], stop_patience=config['earlystopping_patience'], threshold=.75,
                     factor=config['lr_reduce_factor'], dwell=False, model_name=config['model_name'], freeze=config['freeze'], initial_epoch=0),
                     model_checkpoint, wandbcallback, custom_callback()]
    LRA.tepochs = config['max_epochs']  # used to determine value of last epoch for printing

    history = model.fit(
        train_dataset,
        epochs=config["max_epochs"],
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=-1,
        verbose=0,
    )

    # Scores
    scores = model.evaluate(test_dataset, return_dict=True)
    print('Accuracy: ', scores)
    wandb.log({'Test Accuracy':scores['accuracy']})
    wandb.log({'Test F1 Score':scores['f1_score']})

    # Predict
    pred = model.predict(test_dataset, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    # Confusion Matrix
    cm = plot.confusion_matrix(labels, test_dataset.classes, predicted_class_indices)
    wandb.log({"Confusion Matrix":cm})

    cm = wandb.plot.confusion_matrix(
        y_true=test_dataset.classes,
        preds=predicted_class_indices,
        class_names=labels)
    wandb.log({"conf_mat": cm})

    # Classification Report
    cl_report = classification_report(test_dataset.classes,
                                    predicted_class_indices,
                                    labels=[0,1,2,3,4,5,6],
                                    target_names=labels,
                                    output_dict=True)

    df = pd.DataFrame(cl_report)
    wandb.Table(dataframe=df)
    wandb.log({"Classification Report: cl_report"})

    cl_bar_plot = df.iloc[:3, :3].T.plot(kind='bar')
    wandb.log({"CR Bar Plot": wandb.Plotly(cl_bar_plot)})

    cr = sns.heatmap(pd.DataFrame(cl_report).iloc[:-1, :].T, annot=True)
    wandb.log({"Classification Report": wandb.Plotly(cr)})
    plt.savefig('cr.png', dpi=400)
    run.finish()