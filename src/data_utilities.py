import pandas as pd
import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def find_filepaths(root_folder):
    filepaths = []
    for dirname, _, filenames in os.walk(root_folder):
        for filename in filenames:
            filepaths.append(os.path.join(dirname, filename))
    total_images_before_deletion = len(filepaths)
    print(f"Total images before deletion = {total_images_before_deletion}")
    return filepaths


def remove_corrupted_images(root_folder):
    print("\n\nRemoving corrupted images...")

    filepaths = find_filepaths(root_folder)
    del_count = 0
    for filepath in filepaths:
        try:
            fobj = open(filepath, "rb")
            is_JFIF = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()
        if not is_JFIF:
            del_count += 1
            shutil.move(
                filepath,
                os.path.join("data", "corrupted_images",
                             os.path.basename(filepath)),
            )
    print(
        f"Total {del_count} corrupted image moved to 'corrupted_images' folder\n"
    )
    return None


def get_df(root="data/2_processed"):
    """
    root: a folder present inside data dir, which contains classes containing images
    """
    classes = os.listdir(root)

    class_names = []
    images_paths = []
    file_names = []

    for class_name in classes:
        for dirname, _, filenames in os.walk(os.path.join(root, class_name)):
            for file_name in filenames:
                images_paths.append(os.path.join(root, class_name, file_name))
                class_names.append(class_name)
                file_names.append(file_name)

    df = pd.DataFrame(
        list(zip(file_names, class_names, images_paths)),
        columns=["file_name", "class", "file_path"],
    )

    return df


####################################### ImageDataGenerator Utilities ###################################


def scalar(img):
    return img / 127.5 - 1  # scale pixel between -1 and +1


def get_generators(config):
    IMAGE_SIZE = (config.dataset_config.image_width,
                  config.dataset_config.image_width)
    if config.train_config.use_augmentations:
        print("\n\nAugmentation is True! rescale=1./255")
        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.5, 1.5],
            rescale=1.0 / 255,
        )  # preprocessing_function=scalar
    elif not config.train_config.use_augmentations:
        print("No Augmentation!")
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    else:
        print("Error in config.augment. Stop Training!")

    train_dataset = train_datagen.flow_from_directory(
        "data/4_tfds_dataset/train",
        target_size=IMAGE_SIZE,
        batch_size=config.dataset_config.batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode="categorical",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 /
                                      255)  # preprocessing_function=scalar
    val_dataset = test_datagen.flow_from_directory(
        "data/4_tfds_dataset/val",
        shuffle=True,
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        batch_size=config.dataset_config.batch_size,
        class_mode="categorical",
    )

    test_generator = test_datagen.flow_from_directory(
        "data/4_tfds_dataset/test",
        batch_size=config.dataset_config.batch_size,
        seed=config.seed,
        color_mode="rgb",
        shuffle=False,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
    )

    return train_dataset, val_dataset, test_generator


def get_tfds_from_dir(config):
    IMAGE_SIZE = (config.dataset_config.image_width,
                  config.dataset_config.image_width)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/4_tfds_dataset/train",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=config.dataset_config.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=config.seed,
        # subset='training'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/4_tfds_dataset/val",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=config.dataset_config.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=config.seed,
        # subset='validation'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/4_tfds_dataset/test",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=config.dataset_config.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=config.seed,
        # subset='validation'
    )

    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
