import json
import os
import cv2
import wandb
import numpy as np
from io import BytesIO

from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config", "configs/baseline.py")
# print("Downloading model...")
# file_name = "model-best.h5"
# api = wandb.Api()
# run = api.run(
#     config["pretrained_model_link"]
# )
# run.file(file_name).download(replace=True)
# model = models.load_model(file_name)
# pml = config["pretrained_model_link"]
# print(f"Downloaded Trained model: {pml}.")

model = models.load_model('model-best.h5')

# os.system(
#     "wget -O model.h5 https://www.dropbox.com/s/urflwaj6fllr13d/model-best-efficientnet-val-acc-0.74.h5"
# )

# when importing download classification report
for f in run.files():
    if f.name.endswith('png'):
        print(os.path.basename(f.name))
        run.file(f.name).download(replace=True)

normalization_layer = layers.Rescaling(1.0 / 255)

IMAGE_SIZE = (config.dataset_config.image_width, config.dataset_config.image_width)
batch_size = config.dataset_config.batch_size

class_names = [
    "Basalt",
    "Coal",
    "Granite",
    "Limestone",
    "Marble",
    "Quartzite",
    "Sandstone",
]
num_classes = len(class_names)

model = models.load_model(file_name)
optimizer = optimizers.Adam()
f1_score = tfa.metrics.F1Score(num_classes=num_classes, average="macro", threshold=0.5)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f1_score]
)

print("Model loaded!")


def preprocess_image(file):
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(
        img, IMAGE_SIZE, interpolation=cv2.INTER_AREA
    )
    return img


def get_prediction(file):
    img = preprocess_image(file)
    prediction = model.predict(np.array([img / 255]), batch_size=1)
    return f"In this image I see {class_names[np.argmax(prediction)]} (with {(max(prediction[0]))*100:.3f}% confidence!)"


if __name__ == "__main__":
    pass
