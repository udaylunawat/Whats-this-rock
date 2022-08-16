import pandas as pd
import numpy as np
import os
import shutil
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from augment_utilities import apply_rand_augment, cut_mix_and_mix_up, preprocess_for_model, visualize_dataset

# load config from config.json file
with open('config.json') as config_file:
    config = json.load(config_file)

IMAGE_SIZE = (config["image_size"], config["image_size"])


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
            fobj = open(filepath, 'rb')
            is_JFIF = b'JFIF' in fobj.peek(10)
        finally:
            fobj.close()
        if not is_JFIF:
            del_count += 1
            shutil.move(filepath, os.path.join('data', 'corrupted_images', os.path.basename(filepath)))
    print(f"Total {del_count} corrupted image moved to 'corrupted_images' folder\n")
    return None


# https://towardsdatascience.com/stratified-sampling-you-may-have-been-splitting-your-dataset-all-wrong-8cfdd0d32502
def get_stratified_dataset_partitions_pd(
        df,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        target_variable=None):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Shuffle
    df_sample = df.sample(frac=1, random_state=12)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:
        grouped_df = df_sample.groupby(target_variable)
        arr_list = [np.split(g,
                             [int(train_split * len(g)),
                              int((1 - val_split) * len(g))]) for i,
                    g in grouped_df]

        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([t[1] for t in arr_list])
        test_ds = pd.concat([v[2] for v in arr_list])

    else:
        indices_or_sections = [
            int(train_split * len(df)), int((1 - val_split) * len(df))]
        train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


def get_data(sample_size):
    data = pd.read_csv(os.path.join("data/3_consume/", "image_paths.csv"), index_col=0)
    data = data.sample(frac=sample_size).reset_index(drop=True)
    # Splitting data into train, val and test samples using stratified splits
    train_df, val_df, test_df = get_stratified_dataset_partitions_pd(
        data, 0.8, 0.1, 0.1)
    train_df = pd.concat([train_df, val_df])
    return train_df, test_df


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

    import pandas as pd
    df = pd.DataFrame(list(zip(file_names, class_names, images_paths)), columns=['file_name', 'class', 'file_path'])

    return df


def undersample_df(data, class_name):
    merged_df = pd.DataFrame()
    for rock_type in data[class_name].unique():
        temp = data[data[class_name] == rock_type].sample(n=min(data[class_name].value_counts()))
        merged_df = pd.concat([merged_df, temp])

    return merged_df


def limit_data(data_dir,n=100):
    # https://stackoverflow.com/a/65966877/9292995
    a = []
    for i in os.listdir(data_dir):
        for k, j in enumerate(os.listdir(data_dir + '/' + i)):
            if k > n:continue
            a.append((f'{data_dir}/{i}/{j}',i))
    return pd.DataFrame(a,columns=['filename','class'])

####################################### ImageDataGenerator Utilities ###################################


def scalar(img):
    return img / 127.5 - 1  # scale pixel between -1 and +1


def get_generators(config):
    if config['augment']:
        print("Augmentation is True! rescale=1./255")
        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255) # preprocessing_function=scalar
    elif not config['augment']:
        print("No Augmentation!")
        train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
    else:
        print("Error in config.augment. Stop Training!")

    test_datagen = ImageDataGenerator(rescale=1./255) # preprocessing_function=scalar
    train_dataset = train_datagen.flow_from_directory(
        'data/4_tfds_dataset/train',
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        shuffle=True,
        color_mode='rgb',
        class_mode='categorical')
    val_dataset = test_datagen.flow_from_directory(
        'data/4_tfds_dataset/val',
        shuffle=False,
        color_mode='rgb',
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['image_size'],
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        'data/4_tfds_dataset/test',
        batch_size=config['batch_size'],
        seed=42,
        color_mode='rgb',
        shuffle=False,
        class_mode="categorical",
        target_size=(
            config['image_size'],
            config['image_size']))

    return train_dataset, val_dataset, test_generator


######################################## TFDS Dataset Utilities ########################################
def get_data_tfds():
    # build the tfds dataset from ImageFolder
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/image/ImageFolder

    builder = tfds.ImageFolder("data/4_tfds_dataset")
    print(builder.info)  # number of images, number of classes, etc.
    data = builder.as_dataset(split=None, as_supervised=True)

    data, builder = get_data_tfds()

    num_classes = builder.info.features['label'].num_classes
    config["num_classes"] = num_classes

    def load_dataset(split="train"):
        dataset = data[split]
        return prepare_dataset(dataset, split)

    if config['augment']:
        train_dataset = (
            load_dataset()
            .map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
            .map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)
        )
    else:
        train_dataset = load_dataset()

    train_dataset = train_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

    val_dataset = load_dataset(split="val")
    val_dataset = val_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

    # test_dataset = load_dataset(split="test")
    # test_dataset = test_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

    labels = builder.info.features['label'].names

    return train_dataset, val_dataset


# # https://stackoverflow.com/a/37343690/9292995
# # https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/


def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, config["num_classes"])
    return {"images": image, "labels": label}


def prepare_dataset(dataset, split):
    AUTOTUNE = tf.data.AUTOTUNE
    if split == "train":
        return (
            dataset.shuffle(10 * config["batch_size"])
            .map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(config["batch_size"])
        )
    elif split == "val" or split == "test":
        return (
            dataset.map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(config["batch_size"])
        )


# make some random data
# reset_random_seeds()

# default config/hyperparameter values
# you can modify these below or via command line
# https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/cnn_train.py
