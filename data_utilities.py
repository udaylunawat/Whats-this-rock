import pandas as pd
import numpy as np
import os
import shutil
import json
from box import Box
import tensorflow as tf
import tensorflow_datasets as tfds

# load config from config.json file
with open('config.json', 'r') as f:
    config = Box(json.load(f))


def remove_corrupted_images(root_dir):
    import os
    os.makedirs('corrupted_images', exist_ok=True)
    import cv2

    def check_images(s_dir, ext_list):
        bad_images = []
        bad_ext = []
        s_list = os.listdir(s_dir)
        for klass in s_list:
            klass_path = os.path.join(s_dir, klass)
            print('processing class directory ', klass)
            if os.path.isdir(klass_path):
                file_list = os.listdir(klass_path)
                for f in file_list:
                    f_path = os.path.join(klass_path, f)
                    index = f.rfind('.')
                    ext = f[index + 1:].lower()
                    if ext not in ext_list:
                        print('file ', f_path, ' has an invalid extension ', ext)
                        bad_ext.append(f_path)
                    if os.path.isfile(f_path):
                        try:
                            img = cv2.imread(f_path)
                            shape = img.shape
                        except:
                            print('file ', f_path, ' is not a valid image file')
                            bad_images.append(f_path)
                    else:
                        print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
            else:
                print('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
        return bad_images, bad_ext

    source_dir = root_dir
    good_exts = ['jpg', 'png', 'jpeg', 'gif', 'bmp']  # list of acceptable extensions
    bad_file_list, bad_ext_list = check_images(source_dir, good_exts)
    if len(bad_file_list) != 0:
        print('improper image files are listed below')
        for i in range(len(bad_file_list)):
            print(bad_file_list[i])
    else:
        print(' no improper image files were found')

    for bad_file in bad_file_list:
        shutil.move(bad_file, os.path.join('corrupted_images/', os.path.basename(bad_file)))

    from pathlib import Path
    import imghdr

    # https://stackoverflow.com/a/68192520/9292995
    for class_name in os.listdir(root_dir):
        data_dir = os.path.join(root_dir, class_name)
        image_extensions = [".png", ".jpg"]  # add there all your images file extensions

        img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
        for filepath in Path(data_dir).rglob("*"):
            if filepath.suffix.lower() in image_extensions:
                img_type = imghdr.what(filepath)
                if img_type is None:
                    print(f"{filepath} is not an image")
                    shutil.move(filepath, os.path.join('corrupted_images/', os.path.basename(filepath)))
                elif img_type not in img_type_accepted_by_tf:
                    print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                    shutil.move(filepath, os.path.join('corrupted_images/', os.path.basename(filepath)))


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


def undersample_df(data, class_name):
    merged_df = pd.DataFrame()
    for rock_type in data[class_name].unique():
        temp = data[data[class_name] == rock_type].sample(n=min(data[class_name].value_counts()))
        merged_df = pd.concat([merged_df, temp])

    return merged_df


def get_all_filePaths(folderPath):
    result = []
    for dirpath, dirnames, filenames in os.walk(folderPath):
        result.extend([os.path.join(dirpath, filename) for filename in filenames if os.path.splitext(filename)[1] in ['.JPEG' ,'.jpg', '.jpeg', '.png', '.PNG']])
    return result


######################################## TFDS Dataset Utilities ########################################
def get_data_tfds():
    # build the tfds dataset from ImageFolder
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/image/ImageFolder

    builder = tfds.ImageFolder("data/4_tfds_dataset")
    print(builder.info)  # number of images, number of classes, etc.
    data = builder.as_dataset(split=None, as_supervised=True)

    return data, builder


# # https://stackoverflow.com/a/37343690/9292995
# # https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/
IMAGE_SIZE = (config["image_size"], config["image_size"])

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, config.num_classes)
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
