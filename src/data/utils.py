import os
import cv2
import shutil
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, applications

from os import listdir
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import imghdr
from time import time


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def find_filepaths(root_folder):
    filepaths = []
    for dirname, _, filenames in os.walk(root_folder):
        for filename in filenames:
            filepaths.append(os.path.join(dirname, filename))
    total_images_before_deletion = len(filepaths)
    print(f"Total images before deletion = {total_images_before_deletion}")
    return filepaths


def remove_unsupported_images(root_folder):
    print("\n\nRemoving unsupported images...")
    count = 1
    filepaths = find_filepaths(root_folder)
    for filepath in filepaths:
        if filepath.endswith(('JFIF', 'webp', 'jfif')):
            shutil.move(
                filepath,
                os.path.join("data", "corrupted_images",
                             os.path.basename(filepath)),
            )
            count += 1
    print(f"Removed {count} unsupported files.")


@timer_func
def remove_corrupted_images(s_dir,
                            ext_list=[
                                'jpg', 'png', 'jpeg', 'gif', 'bmp', 'JPEG'
                            ]):
    print("\n\nRemoving corrupted images...")
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
                tip = imghdr.what(f_path)
                if ext_list.count(tip) == 0:
                    bad_images.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img = cv2.imread(f_path)
                        shape = img.shape
                    except:
                        print('file ', f_path, ' is not a valid image file')
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f,
                          ' in class directory ', klass)
        else:
            print('*** WARNING*** you have files in ', s_dir,
                  ' it should only contain sub directories')

    for f_path in bad_images:
        shutil.move(
            f_path,
            os.path.join("data", "corrupted_images", os.path.basename(f_path)),
        )
    print(f"removed {len(bad_images)} bad images.\n")


def get_dims(file):
    '''Returns dimenstions for an RBG image'''
    im = cv2.imread(file)
    if im is not None:
        arr = np.array(im)
        h, w = arr.shape[0], arr.shape[1]
        return h, w
    elif im is None:
        return None


def get_df(root: str = "data/2_processed"):
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


def get_value_counts(dataset_path):
    data = get_df(dataset_path)
    vc = data['file_name'].apply(lambda x: x.split('.')[-1]).value_counts()
    print(vc)


####################################### tf.data Utilities ###################################


def scalar(img):
    return img / 127.5 - 1  # scale pixel between -1 and +1


def get_preprocess(cfg):
    preprocess_dict = {
        'vgg16': applications.vgg16.preprocess_input,
        'resnet': applications.resnet.preprocess_input,
        'inceptionresnetv2': applications.inception_resnet_v2.preprocess_input,
        'mobilenetv2': applications.mobilenet_v2.preprocess_input,
        'efficientnetv2': applications.efficientnet_v2.preprocess_input,
        'efficientnetv2m': applications.efficientnet_v2.preprocess_input
    }

    return preprocess_dict[cfg.model.backbone]


def prepare(ds, cfg, shuffle=False, augment=False):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal",
                          input_shape=(cfg.dataset.image.size,
                                       cfg.dataset.image.size,
                                       cfg.dataset.image.channels)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    if augment:
        # Use data augmentation only on the training set.
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    if cfg.model.preprocess:
        preprocess_input = get_preprocess(cfg)
        ds = ds.map(lambda x, y: (preprocess_input(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    # # Batch all datasets.
    # ds = ds.batch(cfg.batch_size)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def get_tfds_from_dir(cfg):
    IMAGE_SIZE = (cfg.dataset.image.size, cfg.dataset.image.size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/4_tfds_dataset/train",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=cfg.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=cfg.seed,
        # subset='training'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/4_tfds_dataset/val",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=cfg.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=cfg.seed,
        # subset='validation'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/4_tfds_dataset/test",
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=cfg.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=cfg.seed,
        # subset='validation'
    )

    return train_ds, val_ds, test_ds


def rename_files(source_dir: str = "data/2_processed/tmp"):
    '''Renames files in classes and moves to 2_processed


    '''
    class_names = os.listdir(source_dir)
    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)
        dest_path = os.path.join('data/2_processed', class_name)
        count = len(os.listdir(dest_path)) + 1
        for filename in os.listdir(class_path):
            old_path = os.path.join(source_dir, class_name, filename)
            _, extension = os.path.splitext(filename)
            new_name = f"{class_name}_{count}{extension}"
            new_path = os.path.join(dest_path, new_name)
            shutil.copy(old_path, new_path)
            count += 1


def move_files(src_dir: str, dest_dir: str = 'data/2_processed/tmp'):
    '''Moves files to tmp directory in 2_processed

    src_dir: directory of rock subclass with files [Basalt, Marble, Coal, ...]
    '''
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    src_dir_name = os.path.basename(src_dir)
    dest_dir_path = os.path.join(dest_dir, src_dir_name.capitalize())
    os.makedirs(dest_dir_path, exist_ok=True)

    files = os.listdir(src_dir)
    total = len(files)
    for index, filename in tqdm(enumerate(files),
                                desc=f"Moving {src_dir_name}",
                                total=total):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir_path, filename)
        # print("Copying", src_path, dest_path)
        shutil.copy(src_path, dest_path)
        # print(f"Moved {index+1} files from {src_dir} to {dest_dir_path}")


def move_and_rename(class_dir: str):
    '''Moves files from class_dir to tmp, renames them there based on count, and moves back to 2_processed

    class_dir: A class dir of supporting classes (Marble, Coal, ...), which contains image files
    '''
    target_classes = os.listdir('data/2_processed/')
    if 'tmp' in target_classes:
        target_classes.remove('tmp')
    target_classes_lower = list(map(lambda x: x.lower(), target_classes))

    for subclass_dir in os.listdir(class_dir):
        if subclass_dir.lower() in target_classes_lower:
            subclass_dir_path = os.path.join(class_dir, subclass_dir)
            move_files(subclass_dir_path)
            rename_files()
            shutil.rmtree('data/2_processed/tmp')
