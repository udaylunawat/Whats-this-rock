from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0, ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, GlobalAveragePooling2D, \
    BatchNormalization, LeakyReLU, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.image import resize


def get_small_cnn(img_height, img_width, num_classes):
    inputs = Input(shape=(img_height, img_width, 3))
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid',
               activation='relu')(inputs)

    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='valid',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=2)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def get_large_cnn(img_height, img_width, num_classes):
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        LeakyReLU(),

        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling2D(5, 5),

        Conv2D(64, kernel_size=(3, 3)),
        BatchNormalization(),
        LeakyReLU(),

        Conv2D(128, kernel_size=(3, 3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling2D(5, 5),

        Flatten(),

        Dense(64),
        Dropout(rate=0.2),
        BatchNormalization(),
        LeakyReLU(),

        Dense(32),
        Dropout(rate=0.2),
        BatchNormalization(),
        LeakyReLU(),

        Dense(16),
        Dropout(rate=0.2),
        BatchNormalization(),
        LeakyReLU(1),

        Dense(4, activation='softmax'),
        Dense(num_classes)
    ])
    return model


def get_mobilenet(config, num_classes):
    mobilenet_pretrained = MobileNetV2(
        input_shape=(
            config["image_size"],
            config["image_size"],
            3),
        weights="imagenet",
        include_top=False)

    # Freeze layers
    mobilenet_pretrained.trainable = True

    # Add untrained final layers
    model = Sequential(
        [
            mobilenet_pretrained,
            GlobalAveragePooling2D(),
            Dense(1024),
            Dropout(0.3),
            Dense(256),
            Dropout(0.3),
            Dense(64),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def get_baseline_model(IMG_SIZE, num_classes, CHANNELS=3):
    input = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    flt_1 = Flatten(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))(input)
    Dense_1 = Dense(128, activation="relu")(flt_1)
    D_out_1 = Dropout(0.1)(Dense_1)
    Dense_2 = Dense(64, activation="relu")(D_out_1)
    output = Dense(num_classes, activation="softmax")(Dense_2)
    model = Model(input, output)

    return model


def get_efficientnet(config, num_classes):
    """Construct a simple categorical CNN following the Keras tutorial"""
    if K.image_data_format() == 'channels_first':
        input_shape = (3, config["image_size"], config["image_size"])
    else:
        input_shape = (config["image_size"], config["image_size"], 3)

    feature_extractor = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        classifier_activation="softmax",
        include_preprocessing=False,
    )

    # Freeze layers
    feature_extractor.trainable = False
    # Add untrained final layers
    model = Sequential(
        [
            feature_extractor,
            GlobalAveragePooling2D(),
            Dense(1024),
            Dropout(0.3),
            Dense(256),
            Dropout(0.3),
            Dense(64),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def get_resnet_model(config):
    IMAGE_SIZE = (config["image_size"], config["image_size"])
    input_t = Input(shape=(config["image_size"], config["image_size"], 3))
    res_model = ResNet50(include_top=False,
                         weights="imagenet",
                         input_tensor=input_t)

    to_res = IMAGE_SIZE

    model = Sequential()
    model.add(Lambda(lambda image: resize(image, to_res)))
    model.add(res_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(config["num_classes"], activation='softmax'))
    return model


def finetune(config, model, num_classes, history):

    pass
