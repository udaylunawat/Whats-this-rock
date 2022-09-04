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


def get_preprocess(args):

    if args.model_config.backbone == 'vgg16':
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    elif args.model_config.backbone == 'resnet':
        preprocess_input = tf.keras.applications.resnet.preprocess_input

    elif args.model_config.backbone == 'inceptionresnetv2':
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

    elif args.model_config.backbone == 'mobilenetv2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    elif args.model_config.backbone == 'efficientnet':
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

    elif args.model_config.backbone == 'EfficientNetV2M':
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
    return preprocess_input


def get_model(args):
    """Get an image classifier with a CNN based backbone.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(args.model_config.model_img_height,
                                     args.model_config.model_img_width,
                                     args.model_config.model_img_channels)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
    )
    # Backbone
    base_model = get_backbone(args)
    preprocess_input = get_preprocess(args)
    # Stack layers
    inputs = layers.Input(shape=(args.model_config.model_img_height,
                                 args.model_config.model_img_width,
                                 args.model_config.model_img_channels))

    if args.train_config.use_augmentations:
        x = data_augmentation(inputs)
    if args.model_config.preprocess:
        x = preprocess_input(x)

    x = base_model(inputs, training=args.model_config.trainable)
    x = layers.GlobalAveragePooling2D()(x)
    if args.model_config.post_gap_dropout:
        x = layers.Dropout(args.model_config.dropout_rate)(x)

    outputs = layers.Dense(args.dataset_config.num_classes,
                           activation='softmax', dtype='float32')(x)

    return models.Model(inputs, outputs)
