import os
from io import BytesIO

import cv2
import numpy as np
import tensorflow_addons as tfa
import wandb
from hydra import compose, initialize
from tensorflow.keras import models, optimizers


# normalization_layer = layers.Rescaling(1.0 / 255)
initialize(config_path="../../configs/", version_base="1.2")
cfg = compose(config_name="config")
batch_size = cfg.batch_size

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


def get_run_data():
    """Get data for a wandb sweep."""
    api = wandb.Api()
    entity = "rock-classifiers"
    project = "Whats-this-rockv7"
    sweep_id = "snemzvnp"
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_accuracy", 0), reverse=True)

    model_found = False
    for run in runs:
        ext_list = list(map(lambda x: x.name.split(".")[-1], list(run.files())))
        if "png" and "h5" in ext_list:
            val_acc = run.summary.get("val_accuracy")
            print(f"Best run {run.name} with {val_acc}% validation accuracy")
            for f in run.files():
                file_name = os.path.basename(f.name)
                # print(os.path.basename(f.name))
                if file_name.endswith("png") and file_name.startswith("Classification"):
                    # Downloading Classification Report
                    run.file(file_name).download(replace=True)
                    print("Classification report donwloaded!")

            # Downloading model
            run.file("model.h5").download(replace=True)
            print("Best model saved to model-best.h5")
            model_found = True
            break

    if not model_found:
        print("No model found in wandb sweep, downloading fallback model!")
        os.system("wget -O model.h5 https://www.dropbox.com/s/urflwaj6fllr13d/model-best-efficientnet-val-acc-0.74.h5")


def preprocess_image(file, image_size):
    """Decode and resize image.

    Parameters
    ----------
    file : _type_
        _description_
    image_size : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return img


def load_model():
    file_name = "model.h5"
    model = models.load_model(file_name)
    optimizer = optimizers.Adam()
    f1_score = tfa.metrics.F1Score(num_classes=num_classes, average="macro", threshold=0.5)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", f1_score],
    )

    print("Model loaded!")
    return model


def get_prediction(file):
    """Get prediction for image.

    Parameters
    ----------
    file : File
        Image file

    Returns
    -------
    str
        Prediction with class name and confidence %.
    """
    model = load_model()
    img = preprocess_image(file, image_size=(cfg.image_size, cfg.image_size))
    prediction = model.predict(np.array([img / 255]), batch_size=1)
    return (
        f"In this image I see {class_names[np.argmax(prediction)]} (with {(max(prediction[0]))*100:.3f}% confidence!)"
    )


if __name__ == "__main__":
    get_run_data()
