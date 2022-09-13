from model.utils import get_model, get_optimizer
from src.data.utils import get_generators

import plot
import json

# import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_addons as tfa
from sklearn.metrics import classification_report

# run = wandb.init(project="Whats-this-rock-inceptionresnetv2",
#                  entity="rock-classifiers",
#                  cfg=cfg)

# os.system('wget -O model-best.h5 https://www.dropbox.com/s/x91k9u765urnlai/model-best.h5')
# if not os.path.exists('model-best.h5'):
#     api = wandb.Api()
#     run = api.run("rock-classifiers/Whats-this-rock/3hvgnqas")
#     run.file("model-best.h5").download()

train_dataset, val_dataset, test_dataset = get_generators(cfg)
labels = [
    "Basalt", "Coal", "Granite", "Limestone", "Marble", "Quartzite",
    "Sandstone"
]

model = get_model(cfg)
try:
    model.load_weights("model-best.h5")
except:
    print("model-best.h5 should be present in the checkpoint dir.")

opt = get_optimizer(cfg)

cfg.metrics.append(
    tfa.metrics.F1Score(num_classes=cfg.num_classes,
                        average="macro",
                        threshold=0.5))

model.compile(loss=cfg.loss,
              optimizer=cfg.optimizer,
              metrics=cfg.metrics)

# Scores
scores = model.evaluate(test_dataset, return_dict=True)
print("Scores: ", scores)

# Predict
pred = model.predict(test_dataset, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

# Confusion Matrix
cm = plot.confusion_matrix(labels, test_dataset.classes,
                           predicted_class_indices)

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
plt.savefig('imgs/cr.png', dpi=400)
