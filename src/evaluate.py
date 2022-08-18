
from model_utilities import get_model, get_optimizer
from data_utilities import get_generators

import os
import plot
import json
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

with open('config.json') as config_file:
    config = json.load(config_file)

# os.system('wget -O model-best.h5 https://www.dropbox.com/s/x91k9u765urnlai/model-best.h5')
run = wandb.init(project="Whats-this-rock",
                 entity="rock-classifiers",
                 config=config)

if not os.path.exists('model-best.h5'):

    api = wandb.Api()
    run = api.run("rock-classifiers/Whats-this-rock/3krh6juu")
    run.file("model-best.h5").download()

train_dataset, val_dataset, test_dataset = get_generators(config)
labels = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']

model = get_model(config)
model.load_weights('model-best.h5')

opt = get_optimizer(config)

config['metrics'].append(tfa.metrics.F1Score(
    num_classes=config['num_classes'],
    average='macro',
    threshold=0.5))

model.compile(loss=config['loss_fn'],
              optimizer=opt,
              metrics=config["metrics"])

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
