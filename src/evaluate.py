
from model_utilities import get_model, get_optimizer, get_best_checkpoint, get_model_weights, delete_checkpoints, LRA
from data_utilities import get_generators

import plot
import json
import numpy as np
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, confusion_matrix

os.system('!wget -O best-model.h5 https://www.dropbox.com/s/t9cj6s8tg850cbn/copper-sound-262-inceptionresnetv2-epoch-1_val_accuracy-0.76.h5')
with open('config.json') as config_file:
    config = json.load(config_file)

train_dataset, val_dataset, test_dataset = get_generators(config)
labels = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']

model = get_model(config)
model.load_weights('best-model.h5')

opt = get_optimizer(config)

config['metrics'].append(tfa.metrics.F1Score(
    num_classes=config['num_classes'],
    average='macro',
    threshold=0.5))

model.compile(loss=config['loss_fn'],
              optimizer=opt,
              metrics=config["metrics"])

scores = model.evaluate(test_dataset)
print('Accuracy: ', scores)

pred = model.predict(test_dataset, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)
result = confusion_matrix(test_dataset.classes, predicted_class_indices)
print(result)

cl_report = classification_report(test_dataset.classes, predicted_class_indices)
print('Classification Report')
print(cl_report)
# wandb.log({"test_accuracy": test_acc, "Classification Report": cl_report})


# filenames = test_dataset.filenames
# nb_samples = len(filenames)
# pred = model.predict(test_dataset, steps=nb_samples, verbose=1)

# test_acc = sum([predicted_class_indices[i] == test_dataset.classes[i] for i in range(len(test_dataset))]) / len(test_dataset)
# # Confution Matrix and Classification Report
# print('Confusion Matrix')
# print(confusion_matrix(test_dataset.classes, predicted_class_indices))

# cm = plot.confusion_matrix(labels, test_dataset.classes, predicted_class_indices)
# wandb.log({"test_accuracy": test_acc, "Confusion Matrix": cm})
