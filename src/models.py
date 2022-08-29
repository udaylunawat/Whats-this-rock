from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    BatchNormalization,
    LeakyReLU,
    Input,
)
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.models import Sequential, Model


def get_baseline(config):
    input = Input(shape=(config["image_size"], config["image_size"], 3))
    flt_1 = Flatten(input_shape=(config["image_size"], config["image_size"], 3))(input)
    Dense_1 = Dense(128, activation="relu")(flt_1)
    D_out_1 = Dropout(0.1)(Dense_1)
    Dense_2 = Dense(64, activation="relu")(D_out_1)
    output = Dense(config["num_classes"], activation="softmax")(Dense_2)
    model = Model(input, output)

    return model


def get_small_cnn(config):
    inputs = Input(shape=(config["image_size"], config["image_size"], 3))
    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
    )(inputs)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
    )(x)

    x = MaxPooling2D(pool_size=2)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    outputs = Dense(config["num_classes"], activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)


def get_large_cnn(config):
    model = Sequential(
        [
            Conv2D(
                16,
                kernel_size=(3, 3),
                input_shape=(config["image_size"], config["image_size"], 3),
            ),
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
            Dense(4, activation="softmax"),
            Dense(config["num_classes"]),
        ]
    )
    return model


def get_mobilenet(config):
    model = Sequential()
    base_model = applications.MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(config["image_size"], config["image_size"], 3),
    )
    base_model.trainable = not config["freeze"]
    model.add(base_model)

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(
        Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(0.01),
            bias_regularizer=regularizers.l1_l2(0.01),
        )
    )

    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(config["num_classes"], activation="softmax"))
    return model


def get_mobilenetv2(config):
    base_model = applications.MobileNetV2(
        input_shape=(config["image_size"], config["image_size"], 3),
        weights="imagenet",
        include_top=False,
    )

    # freeze layers
    base_model.trainable = not config["freeze"]

    # Add untrained final layers
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024),
            Dropout(0.3),
            Dense(256),
            Dropout(0.3),
            Dense(64),
            Dropout(0.3),
            Dense(config["num_classes"], activation="softmax"),
        ]
    )
    return model


def get_efficientnet(config):
    """Construct a simple categorical CNN following the Keras tutorial."""
    base_model = applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(config["image_size"], config["image_size"], 3),
        classifier_activation="softmax",
        include_preprocessing=False,
    )

    # freeze layers
    base_model.trainable = not config["freeze"]
    # Add untrained final layers
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024),
            Dropout(0.3),
            Dense(256),
            Dropout(0.3),
            Dense(64),
            Dropout(0.3),
            Dense(config["num_classes"], activation="softmax"),
        ]
    )

    return model


def get_resnet(config):
    base_model = applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=(config["image_size"], config["image_size"], 3),
    )
    base_model.trainable = not config["freeze"]

    model = Sequential(
        [
            base_model,
            Flatten(),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            BatchNormalization(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            BatchNormalization(),
            Dense(config["num_classes"], activation="softmax"),
        ]
    )

    return model


def get_efficientnetv2m(config):
    inputs = Input(shape=(config["image_size"], config["image_size"], 3))
    model = applications.EfficientNetV2M(
        include_top=False, input_tensor=inputs, weights="imagenet"
    )

    # Freeze the pretrained weights
    model.trainable = not config["freeze"]

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(config["num_classes"], activation="softmax", name="pred")(x)

    # Compile
    model = Model(inputs, outputs, name="EfficientNet")

    return model


def get_inceptionresnetv2(config):
    base_model = applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(config["image_size"], config["image_size"], 3),
        pooling="max",
    )

    base_model.trainable = not config["freeze"]

    model = Sequential(
        [
            base_model,
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            Dense(
                256,
                kernel_regularizer=regularizers.l2(l=0.016),
                activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006),
                activation="relu",
                kernel_initializer=initializers.GlorotUniform(), # seed=123
            ),
            Dropout(rate=0.45, seed=123),
            Dense(
                config["num_classes"],
                activation="softmax",
                kernel_initializer=initializers.GlorotUniform(), # seed=123
            ),
        ]
    )

    return model
