import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


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
        base_model.trainable = True
    elif args.model_config.backbone == 'resnet':
        base_model = tf.keras.applications.ResNet50(include_top=False,
                                                    weights=weights)
        base_model.trainable = True
    elif args.model_config.backbone == 'inceptionresnetv2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                             weights=weights)
        base_model.trainable = True
    elif args.model_config.backbone == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                       weights=weights)
        base_model.trainable = True
    elif args.model_config.backbone == 'efficientnet':
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False,
                                                            weights=weights)
        base_model.trainable = True
    elif args.model_config.backbone == 'EfficientNetV2M':
        base_model = tf.keras.applications.EfficientNetV2M(include_top=False,
                                                           weights=weights)
        base_model.trainable = True
    else:
        raise NotImplementedError("Not implemented for this backbone.")

    return base_model


def get_model(args):
    """Get an image classifier with a CNN based backbone.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    # Backbone
    base_model = get_backbone(args)

    # Stack layers
    inputs = layers.Input(shape=(args.model_config.model_img_height,
                                 args.model_config.model_img_width,
                                 args.model_config.model_img_channels))

    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    if args.model_config.post_gap_dropout:
        x = layers.Dropout(args.model_config.dropout_rate)(x)
    outputs = layers.Dense(args.dataset_config.num_classes,
                           activation='softmax')(x)

    return models.Model(inputs, outputs)
