from tensorflow.keras.applications import MobileNet, MobileNetV2, EfficientNetV2B0, ResNet50, InceptionResNetV2, vgg16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, GlobalAveragePooling2D, \
    BatchNormalization, LeakyReLU, Input, Lambda
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.image import resize


def get_baseline(config):
    input = Input(shape=(config['image_size'], config['image_size'], 3))
    flt_1 = Flatten(input_shape=(config['image_size'], config['image_size'], 3))(input)
    Dense_1 = Dense(128, activation="relu")(flt_1)
    D_out_1 = Dropout(0.1)(Dense_1)
    Dense_2 = Dense(64, activation="relu")(D_out_1)
    output = Dense(config['num_classes'], activation="softmax")(Dense_2)
    model = Model(input, output)

    return model


def get_small_cnn(config):
    inputs = Input(shape=(config['image_size'], config['image_size'], 3))
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

    outputs = Dense(config['num_classes'], activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def get_large_cnn(config):
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), input_shape=(config['image_size'], config['image_size'], 3)),
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
        Dense(config['num_classes'])
    ])
    return model


def get_mobilenet(config):
    model = Sequential()
    BaseModel = MobileNet(weights='imagenet', include_top=False, input_shape=(config['image_size'], config['image_size'], 3))
    model.add(BaseModel)

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(0.01),
                    bias_regularizer=regularizers.l1_l2(0.01)))

    model.add(BatchNormalization())
    model.add(Dropout(.3))
    model.add(Dense(config['num_classes'], activation='softmax'))
    return model


def get_mobilenetv2(config):
    mobilenet_pretrained = MobileNetV2(
        input_shape=(
            config['image_size'],
            config['image_size'],
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
            Dense(config['num_classes'], activation="softmax"),
        ]
    )
    return model


def get_efficientnet(config):
    """Construct a simple categorical CNN following the Keras tutorial"""
    if K.image_data_format() == 'channels_first':
        input_shape = (3, config['image_size'], config['image_size'])
    else:
        input_shape = (config['image_size'], config['image_size'], 3)

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
            Dense(config['num_classes'], activation="softmax"),
        ]
    )

    return model


def get_resnet(config):
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


def get_inceptionresnetv2(config):
    base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(config["image_size"], config["image_size"], 3), pooling='max')
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu', kernel_initializer=initializers.GlorotUniform(seed=123))(x)
    x = Dropout(rate=.45, seed=123)(x)
    output = Dense(config["num_classes"], activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=123))(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model


def get_vgg16(config):
    # Loading VGG16 model
    base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(config['image_size'], config['image_size'], 3))
    base_model.trainable = False  # Not trainable weights

    flatten_layer = Flatten()
    dense_layer_1 = Dense(50, activation='relu')
    dense_layer_2 = Dense(20, activation='relu')
    prediction_layer = Dense(config['num_classes'], activation='softmax')
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    dropout = Dropout(0.3)

    model = Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dropout,
        dense_layer_2,
        dropout,
        prediction_layer
    ])
    return model
