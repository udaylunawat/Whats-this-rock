import json
import os
import cv2
import wandb
import numpy as np
from io import BytesIO

from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


def get_run_data():
    api = wandb.Api()
    entity = "rock-classifiers"
    project = "Whats-this-rockv7"
    sweep_id = "snemzvnp"
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sorted(sweep.runs,
    key=lambda run: run.summary.get("val_accuracy", 0), reverse=True)

    model_found = False
    for run in runs:
        ext_list = list(map(lambda x:x.name.split('.')[-1], list(run.files())))
        if 'png' and 'h5' in ext_list:
            val_acc = run.summary.get("val_accuracy")
            print(f"Best run {run.name} with {val_acc}% validation accuracy")
            for f in run.files():
                file_name = os.path.basename(f.name)
                # print(os.path.basename(f.name))
                if file_name.endswith('png') and file_name.startswith('Classification'):
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
        os.system(
            "wget -O model.h5 https://www.dropbox.com/s/urflwaj6fllr13d/model-best-efficientnet-val-acc-0.74.h5"
        )


@hydra.main(config_path="../../configs/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    file_name = "model.h5"
    get_run_data()
    model = models.load_model(file_name)
    normalization_layer = layers.Rescaling(1.0 / 255)

    IMAGE_SIZE = (cfg.image_size, cfg.image_size)
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

    model = models.load_model(file_name)
    optimizer = optimizers.Adam()
    f1_score = tfa.metrics.F1Score(num_classes=num_classes, average="macro", threshold=0.5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f1_score]
    )

    print("Model loaded!")


def preprocess_image(file, image_size):
    # TODO: Get image size from cfg
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return img


def get_prediction(file):
    img = preprocess_image(file, image_size=224)
    prediction = model.predict(np.array([img / 255]), batch_size=1)
    return f"In this image I see {class_names[np.argmax(prediction)]} (with {(max(prediction[0]))*100:.3f}% confidence!)"


if __name__ == "__main__":
    main()
