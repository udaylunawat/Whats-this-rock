import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras import regularizers, initializers


def get_backbone(cfg):
    """Get backbone for the model.

    cfg:
        cfg (omegaconf.DictConfig): Configuration.
    """
    weights = None
    models_dict = {
        'vgg16': applications.VGG16,
        'resnet': applications.ResNet50,
        'inceptionresnetv2': applications.InceptionResNetV2,
        'mobilenetv2': applications.MobileNetV2,
        'efficientnetv2': applications.EfficientNetV2B0,
        'efficientnetv2m': applications.EfficientNetV2M
    }

    if cfg.model.use_pretrained_weights:
        weights = "imagenet"

    try:
        base_model = models_dict[cfg.model.backbone](
            include_top=False,
            weights=weights,
            input_shape=(cfg.dataset.image.size, cfg.dataset.image.size,
                         cfg.dataset.image.channels))
    except:
        raise NotImplementedError("Not implemented for this backbone.")

    base_model.trainable = cfg.model.trainable

    return base_model


def get_model(cfg):
    """Get an image classifier with a CNN based backbone.

    cfg:
        cfg (omegaconf.DictConfig): Configuration.
    """

    # Backbone
    base_model = get_backbone(cfg)

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(cfg.model.dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(cfg.model.dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(cfg.model.dropout_rate),
        layers.Dense(cfg.num_classes, activation="softmax"),
    ])

    return model
