import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras import regularizers, initializers


def get_backbone(cfg) -> tf.keras.models:
    """Get backbone for the model.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration

    Returns
    -------
    tensorflow.keras.model
        Tensroflow Model

    Raises
    ------
    NotImplementedError
        raise error if wrong backbone name passed
    """
    weights = None
    models_dict = {
        # "convnexttiny":applications.ConvNeXtTiny,
        "vgg16": applications.VGG16,
        "resnet": applications.ResNet50,
        "inceptionresnetv2": applications.InceptionResNetV2,
        "mobilenetv2": applications.MobileNetV2,
        "efficientnetv2": applications.EfficientNetV2B0,
        "efficientnetv2m": applications.EfficientNetV2M,
        "xception": applications.Xception,
    }

    if cfg.use_pretrained_weights:
        weights = "imagenet"

    try:
        base_model = models_dict[cfg.backbone](
            include_top=False,
            weights=weights,
            input_shape=(cfg.image_size, cfg.image_size, cfg.image_channels),
        )
    except:
        raise NotImplementedError("Not implemented for this backbone.")

    base_model.trainable = cfg.trainable

    return base_model


def get_model(cfg):
    """Get an image classifier with a CNN based backbone.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig)
        Hydra Configuration

    Returns
    -------
    tensorflow.keras.Model
        Model object
    """
    # Backbone
    base_model = get_backbone(cfg)

    model = tf.keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(cfg.dropout_rate),
            layers.Dense(256, activation="relu"),
            layers.Dropout(cfg.dropout_rate),
            layers.Dense(64, activation="relu"),
            layers.Dropout(cfg.dropout_rate),
            layers.Dense(cfg.num_classes, activation="softmax"),
        ]
    )

    return model
