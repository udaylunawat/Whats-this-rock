import json
import os
import cv2
import wandb
import numpy as np
from io import BytesIO

from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa

# read config file
with open("config.json") as config_file:
    config = json.load(config_file)

print("Downloading model...")
file_name = "model-best.h5"
api = wandb.Api()
run = api.run(
    config["pretrained_model_link"]
)
run.file(file_name).download(replace=True)
model = models.load_model(file_name)
pml = config["pretrained_model_link"]
print(f"Downloaded Trained model: {pml}.")

# os.system(
#     "wget -O model.h5 https://www.dropbox.com/s/urflwaj6fllr13d/model-best-efficientnet-val-acc-0.74.h5"
# )

normalization_layer = layers.Rescaling(1.0 / 255)
AUTOTUNE = AUTOTUNE

img_height, img_width = (config["image_size"], config["image_size"])
batch_size = config["batch_size"]

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
        img, (config["image_size"], config["image_size"]), interpolation=cv2.INTER_AREA
    )
    return img


def get_prediction(file):
    img = preprocess_image(file)
    prediction = model.predict(np.array([img / 255]), batch_size=1)
    return f"In this image I see {class_names[np.argmax(prediction)]} (with {(max(prediction[0]))*100:.3f}% confidence!)"


if __name__ == "__main__":
    pass
