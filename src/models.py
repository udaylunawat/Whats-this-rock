import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras import regularizers, initializers


def get_backbone(args):
    """Get backbone for the model.

    Args:
        args (ml_collections.ConfigDict): Configuration.
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

    if args.model_config.use_pretrained_weights:
        weights = "imagenet"

    try:
        base_model = models_dict[args.model_config.backbone]
    except:
        raise NotImplementedError("Not implemented for this backbone.")

    base_model = base_model(include_top=False, weights=weights)
    base_model.trainable = args.model_config.trainable

    return base_model


def get_model(args):
    """Get an image classifier with a CNN based backbone.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """

    # Backbone
    base_model = get_backbone(args)

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024),
        layers.Dropout(args.model_config.dropout_rate),
        layers.Dense(256),
        layers.Dropout(args.model_config.dropout_rate),
        layers.Dense(64),
        layers.Dropout(args.model_config.dropout_rate),
        layers.Dense(args.dataset_config.num_classes, activation="softmax"),
    ])

    return model
