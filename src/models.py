import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers, initializers


def get_backbone(args):
    """Get backbone for the model.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    weights = None
    if args.model_config.use_pretrained_weights:
        weights = "imagenet"

    if args.model_config.backbone == 'vgg16':
        base_model = tf.keras.applications.VGG16(include_top=False,
                                                 weights=weights)

    elif args.model_config.backbone == 'resnet':
        base_model = tf.keras.applications.ResNet50(include_top=False,
                                                    weights=weights)

    elif args.model_config.backbone == 'inceptionresnetv2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                             weights=weights)

    elif args.model_config.backbone == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                       weights=weights)

    elif args.model_config.backbone == 'efficientnet':
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False,
                                                            weights=weights)

    elif args.model_config.backbone == 'EfficientNetV2M':
        base_model = tf.keras.applications.EfficientNetV2M(include_top=False,
                                                           weights=weights)

    else:
        raise NotImplementedError("Not implemented for this backbone.")
    base_model.trainable = args.model_config.trainable
    return base_model


def get_model(args):
    """Get an image classifier with a CNN based backbone.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """

    # Backbone
    base_model = get_backbone(args)

    model = tf.keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024),
            layers.Dropout(args.model_config.dropout_rate),
            layers.Dense(256),
            layers.Dropout(args.model_config.dropout_rate),
            layers.Dense(64),
            layers.Dropout(args.model_config.dropout_rate),
            layers.Dense(args.dataset_config.num_classes, activation="softmax"),
        ]
    )

    return model
