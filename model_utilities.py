import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import class_weight
from models import get_efficientnet, get_mobilenet, get_baseline_model, get_small_cnn, get_resnet_model

import tensorflow as tf


def get_optimizer(config):
    if config.optimizer == 'adam':
        opt = optimizers.Adam(config.init_learning_rate)
    elif config.optimizer == 'rms':
        opt = optimizers.RMSprop(learning_rate=config.init_learning_rate,
                                 rho=0.9, epsilon=1e-08, decay=0.0)
    elif config.optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=config.init_learning_rate)
    elif config.optimizer == 'adamax':
        opt = optimizers.Adamax(learning_rate=config.init_learning_rate)

    return opt


def get_model(config, num_classes):
    if config.model_name == "efficientnet":
        model = get_efficientnet(config, num_classes)
    elif config.model_name == "baseline":
        model = get_baseline_model(config.image_size, num_classes)
    elif config.model_name == "baseline_cnn":
        model = get_small_cnn(
            config.image_size,
            config.image_size,
            num_classes)
    elif config.model_name == "mobilenet":
        model = get_mobilenet(config, num_classes)
    elif config.model_name == "resnet":
        model = get_resnet_model(config)

    return model


# https://datamonje.com/image-data-augmentation/#cutout
# https://colab.research.google.com/drive/1on9rvQdr0s8CfqYeZBTbgk0K5I_Dha-X#scrollTo=py9wqsM05kio
def custom_augmentation(np_tensor):

    def random_contrast(np_tensor):
        return np.array(tf.image.random_contrast(np_tensor, 0.5, 2))

    def random_hue(np_tensor):
        return np.array(tf.image.random_hue(np_tensor, 0.5))

    def random_saturation(np_tensor):
        return np.array(tf.image.random_saturation(np_tensor, 0.2, 3))

    def random_crop(np_tensor):
        # cropped height between 70% to 130% of an original height
        new_height = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[0])
        # cropped width between 70% to 130% of an original width
        new_width = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[1])
        # resize to new height and width
        cropped = tf.image.resize_with_crop_or_pad(
            np_tensor, new_height, new_width)
        return np.array(tf.image.resize(cropped, np_tensor.shape[:2]))

    def gaussian_noise(np_tensor):
        mean = 0
        # variance: randomly between 1 to 25
        var = np.random.randint(1, 26)
        # sigma is square root of the variance value
        noise = np.random.normal(mean, var**0.5, np_tensor.shape)
        return np.clip(np_tensor + noise, 0, 255).astype('int')

    def cutout(np_tensor):
        cutout_height = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[0])
        cutout_width = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[1])
        cutout_height_point = np.random.randint(
            np_tensor.shape[0] - cutout_height)
        cutout_width_point = np.random.randint(
            np_tensor.shape[1] - cutout_width)
        np_tensor[cutout_height_point:cutout_height_point + cutout_height,
                  cutout_width_point:cutout_width_point + cutout_width, :] = 127
        return np_tensor

    if (np.random.uniform() < 0.1):
        np_tensor = random_contrast(np_tensor)
    if (np.random.uniform() < 0.1):
        np_tensor = random_hue(np_tensor)
    if (np.random.uniform() < 0.1):
        np_tensor = random_saturation(np_tensor)
    if (np.random.uniform() < 0.2):
        np_tensor = random_crop(np_tensor)

    # Gaussian noise giving error hence removed
    # if (np.random.uniform() < 0.2):
    #     np_tensor = gaussian_noise(np_tensor)
    if (np.random.uniform() < 0.3):
        np_tensor = cutout(np_tensor)
    return np.array(np_tensor)


def get_generators(config, train_df, test_df):

    if config.augment == "True":
        datagen = ImageDataGenerator(validation_split=0.2,
                                     horizontal_flip=True,
                                     fill_mode="nearest",
                                     zoom_range=[0.8, 1.25],
                                     brightness_range=[0.5, 1.2],
                                     width_shift_range=0.1,  # config.width_shift_range,
                                     height_shift_range=0.1,  # config.height_shift_range,
                                     rotation_range=30,  # config.rotation_range,
                                     shear_range=30,
                                     preprocessing_function=custom_augmentation,
                                     rescale=1. / 255.)
    elif config.augment == "False":
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="classes",
        subset="training",
        batch_size=config.batch_size,
        seed=42,
        color_mode='rgb',
        shuffle=True,
        class_mode="categorical",
        preprocessing_function=custom_augmentation,
        target_size=(
            config.image_size,
            config.image_size))

    val_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="classes",
        subset="validation",
        batch_size=config.batch_size,
        seed=42,
        color_mode='rgb',
        shuffle=True,
        class_mode="categorical",
        target_size=(
            config.image_size,
            config.image_size))

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_path",
        y_col='classes',
        batch_size=config.batch_size,
        validation_split=None,
        seed=42,
        shuffle=False,
        color_mode='rgb',
        class_mode=None,
        target_size=(
            config.image_size,
            config.image_size))

    train_generator = train_generator.prefetch(buffer_size=32)
    val_generator = val_generator.prefetch(buffer_size=32)
    return train_generator, val_generator, test_generator


def get_model_weights(train_generator):
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes)

    train_class_weights = dict(enumerate(class_weights))
    return train_class_weights
