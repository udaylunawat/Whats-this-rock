import imghdr
import os
import shutil
import splitfolders
from time import time
from typing import Optional

import cv2
import logging
import keras_cv
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import applications, layers
from tqdm import tqdm


def get_new_name(dir_list: list) -> dict:
    """Return dict with old name and new name of files in multiple directories.

    {'data/1_extracted/dataset1/Basalt/14.jpg': 'data/2_processed/Basalt/dataset1_01_Basalt_14.jpg'}

    Parameters
    ----------
    dir_list : list
        list of dir paths

    Returns
    -------
    dict
        {old_name: new_name}
    """
    file_list = []
    for dir in dir_list:
        paths, _ = find_filepaths(dir)
        file_list.extend(paths)

    count = 1
    file_dict = {}
    for file_path in file_list:
        dataset = file_path.split("/")[-3]
        class_name = file_path.split("/")[-2]
        basename = os.path.basename(file_path)
        file_name = os.path.splitext(basename)[0]
        extension = os.path.splitext(basename)[1]
        new_file_name = os.path.join(
            "data",
            "2_processed",
            class_name,
            f"{dataset}_{class_name}_{str(count).zfill(3)}_{file_name}{extension}",
        )
        file_dict[file_path] = new_file_name
        count += 1

    return file_dict


def move_to_processed():
    dir1 = "data/1_extracted/dataset1"
    dir2 = "data/1_extracted/dataset2"
    for d1, d2 in zip(os.listdir(dir1), os.listdir(dir2)):
        assert d1 == d2
        path_dict = get_new_name([os.path.join(dir1, d1), os.path.join(dir2, d2)])

        for old_path, new_path in path_dict.items():
            shutil.copy(old_path, new_path)


def move_bad_files(txt_file, dest, text):
    """Moves files in txt_file to dest.

    Parameters
    ----------
    txt_file : file
        text file with path of bad images
    dest : _type_
        target destination
    """
    print(text)
    f = open(txt_file, "r")
    cleaned = list(map(lambda x:x.replace('\n', ''), f.readlines()))
    assert len(list(set([x for i,x in enumerate(cleaned) if cleaned.count(x) > 1]))) == 0

    count = 0
    for line in cleaned:
        if len(line) > 0 and not line.startswith('#') and not line == "":
            basename = os.path.basename(line)
            file_name = os.path.splitext(basename)[0]
            try:
                shutil.move(line.strip(), os.path.join(dest, file_name))
                count +=1
            except FileNotFoundError:
                continue
    print(f"\nMoved {count} images to {dest}.\n")


def sampling(cfg):
    """Oversamples/Undersample/No Sampling data into train, val, test.

    Parameters
    ----------
    cfg : _type_
        _description_
    """
    print(
        "\nSplitting files in Train, Validation and Test and saving to data/3_tfds_dataset/"
    )
    scc = min(get_df()["class"].value_counts())
    val_split = test_split = (1 - cfg.train_split) / 2
    print(
        f"Data Split:- Training {cfg.train_split:.2f}, Validation {val_split:.2f}, Test {test_split:.2f}"
    )
    if cfg.sampling == "oversample":
        print("\nSampling type:- Oversampling...")
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        print("Finding smallest class for oversampling fixed parameter.")
        print(f"Smallest class count is {scc}\n")
        splitfolders.fixed(
            "data/2_processed",
            output="data/3_tfds_dataset",
            oversample=True,
            fixed=(((scc // 2) - 1, (scc // 2) - 1)),
            seed=cfg.seed,
            move=False,
        )
    elif cfg.sampling == "undersample":
        print(f"Sampling type:- Undersampling to {cfg.sampling} samples.")
        splitfolders.fixed(
            "data/2_processed",
            output="data/3_tfds_dataset",
            fixed=(
                int(scc * cfg.train_split),
                int(scc * val_split),
                int(scc * test_split),
            ),
            oversample=False,
            seed=cfg.seed,
            move=False,
        )
    else:
        print("Sampling type:- No Sampling.")
        splitfolders.ratio(
            "data/2_processed",
            output="data/3_tfds_dataset",
            ratio=(cfg.train_split, val_split, test_split),
            seed=cfg.seed,
            move=False,
        )
    print("\n\n")


def timer_func(func):
    """Show the execution time of the function object passed.

    Parameters
    ----------
    func : _type_
        _description_
    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def find_filepaths(root_folder: str):
    """Recursively finds all files.

    Parameters
    ----------
    root_folder : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    filepaths = []
    for dirname, _, filenames in os.walk(root_folder):
        for filename in filenames:
            filepaths.append(os.path.join(dirname, filename))
    return sorted(filepaths), len(filepaths)


def remove_unsupported_images(root_folder: str):
    """Remove unsupported images.

    Parameters
    ----------
    root_folder : str
        Root Folder.
    """
    print("\n\nRemoving unsupported images...")
    count = 1
    filepaths, _ = find_filepaths(root_folder)
    for filepath in filepaths:
        if filepath.endswith(("JFIF", "webp", "jfif")):
            shutil.move(
                filepath,
                os.path.join("data", "corrupted_images", os.path.basename(filepath)),
            )
            count += 1
    print(f"Removed {count} unsupported files.\n")


@timer_func
def remove_corrupted_images(
    s_dir: str, ext_list: list = ["jpg", "png", "jpeg", "gif", "bmp", "JPEG"]
):
    """Remove corrupted images.

    Parameters
    ----------
    s_dir : str
        Source directory.
    ext_list : list, optional
        Extensions list, by default ["jpg", "png", "jpeg", "gif", "bmp", "JPEG"]
    """
    print("\nRemoving corrupted images...")
    classes = os.listdir(s_dir)

    def remove_corrupted_from_dir(rock_class):
        # remove corrupted images from single directory
        bad_images = []
        class_path = os.path.join(s_dir, rock_class)
        if os.path.isdir(class_path):
            file_list = os.listdir(class_path)
            for f in file_list:
                f_path = os.path.join(class_path, f)
                tip = imghdr.what(f_path)
                if ext_list.count(tip) == 0:
                    bad_images.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        cv2.imread(f_path)
                        # shape = img.shape
                    except Exception:
                        print("file ", f_path, " is not a valid image file")
                        bad_images.append(f_path)
                else:
                    print(
                        "*** fatal error, you a sub directory ",
                        f,
                        " in class directory ",
                        rock_class,
                    )
        else:
            print(
                "*** WARNING*** you have files in ",
                s_dir,
                " it should only contain sub directories",
            )

        for f_path in bad_images:
            shutil.move(
                f_path,
                os.path.join("data", "corrupted_images", os.path.basename(f_path)),
            )
        print(f"Removed {len(bad_images)} bad images from {rock_class}.")

    def remove_corrupted_images_multicore():
        # Multiprocessing
        # create all tasks
        from multiprocessing import Process

        processes = [
            Process(target=remove_corrupted_from_dir, args=(i,)) for i in classes
        ]
        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()
        # report that all tasks are completed
        print("Removed all corrupted images.", flush=True)

    for rock_class in classes:
        remove_corrupted_from_dir(rock_class)


def get_dims(file: str) -> Optional[tuple]:
    """Return dimenstions for an RBG image.

    Parameters
    ----------
    file : str
        file path for image

    Returns
    -------
    Optional[tuple, None]
        returns a tuple of heights and width of image or None
    """
    im = cv2.imread(file)
    if im is not None:
        arr = np.array(im)
        h, w = arr.shape[0], arr.shape[1]
        return h, w
    elif im is None:
        return None


def get_df(root: str = "data/2_processed") -> pd.DataFrame:
    """Return df with classes, image paths and file names.

    Parameters
    ----------
    root : str, optional
        directory to scan for image files, by default "data/2_processed"

    Returns
    -------
    pd.DataFrame
        with columns file_name, class and file_path
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


def get_value_counts(dataset_path: str) -> None:
    """Get class counts of all classes in the dataset.

    Parameters
    ----------
    dataset_path : str
        directory with subclasses
    """
    data = get_df(dataset_path)
    vc = data["file_name"].apply(lambda x: x.split(".")[-1]).value_counts()
    print(vc)


####################################### tf.data Utilities ###################################


def scalar(img: Image) -> Image:
    """Scale pixel between -1 and +1.

    Parameters
    ----------
    img : Image
        PIL Image

    Returns
    -------
    Image
        imagew with pixel values scaled between -1 and 1
    """
    return img / 127.5 - 1


def get_preprocess(cfg):
    """Return preprocess function for particular model.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig)
        Hydra Configuration

    Returns
    -------
    _type_
        _description_
    """
    preprocess_dict = {
        # "convnexttiny":applications.convnext,
        "vgg16": applications.vgg16,
        "resnet": applications.resnet,
        "inceptionresnetv2": applications.inception_resnet_v2,
        "mobilenetv2": applications.mobilenet_v2,
        "efficientnetv2": applications.efficientnet_v2,
        "efficientnetv2m": applications.efficientnet_v2,
        "xception": applications.xception,
    }

    return preprocess_dict[cfg.backbone].preprocess_input


def prepare(ds, cfg, shuffle=False, augment=False):
    """Prepare dataset using augment, preprocess, cache, shuffle and prefetch.

    Parameters
    ----------
    ds : _type_
        _description_
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration
    shuffle : bool, optional
        _description_, by default False
    augment : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # keras_cv
    def to_dict(image, label):
        image = tf.image.resize(image, (cfg.image_size, cfg.image_size))
        image = tf.cast(image, tf.float32)
        # label = tf.one_hot(label, cfg.num_classes)
        return {"images": image, "labels": label}

    def preprocess_for_model(inputs):
        images, labels = inputs["images"], inputs["labels"]
        images = tf.cast(images, tf.float32)
        return images, labels

    def cut_mix_and_mix_up(samples):
        samples = cut_mix(samples, training=True)
        samples = mix_up(samples, training=True)
        return samples

    def apply_rand_augment(inputs):
        inputs["images"] = rand_augment(inputs["images"])
        return inputs

    AUTOTUNE = tf.data.AUTOTUNE

    if cfg.preprocess:
        preprocess_input = get_preprocess(cfg)
        ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    if augment:
        # normal augmentation
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip(
                    "horizontal",
                    input_shape=(cfg.image_size, cfg.image_size, cfg.image_channels),
                ),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )
        # Use data augmentation only on the training set.
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    elif augment == "kerascv":
        # using keras_cv
        ds = ds.map(to_dict, num_parallel_calls=AUTOTUNE)
        rand_augment = keras_cv.layers.RandAugment(
            value_range=(0, 255),
            augmentations_per_image=3,
            magnitude=0.3,
            magnitude_stddev=0.2,
            rate=0.5,
        )
        cut_mix = keras_cv.layers.CutMix()
        mix_up = keras_cv.layers.MixUp()

        ds = ds.map(apply_rand_augment, num_parallel_calls=AUTOTUNE).map(
            cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE
        )

        ds = ds.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    # # Batch all datasets.
    # ds = ds.batch(cfg.batch_size)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


def get_tfds_from_dir(cfg):
    """Convert directory of images to tfds dataset.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration

    Returns
    -------
    _type_
        _description_
    """
    IMAGE_SIZE = (cfg.image_size, cfg.image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/3_tfds_dataset/train",
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=cfg.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=cfg.seed,
        # subset='training'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/3_tfds_dataset/val",
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=cfg.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=cfg.seed,
        # subset='validation'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/3_tfds_dataset/test",
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=cfg.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=cfg.seed,
        # subset='validation'
    )

    return train_ds, val_ds, test_ds


def rename_files(source_dir: str = "data/2_processed/tmp"):
    """Rename files in classes and moves to 2_processed.

    Parameters
    ----------
    source_dir : str, optional
        _description_, by default "data/2_processed/tmp"
    """
    class_names = os.listdir(source_dir)
    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)
        dest_path = os.path.join("data/2_processed", class_name)
        count = len(os.listdir(dest_path)) + 1
        for filename in os.listdir(class_path):
            old_path = os.path.join(source_dir, class_name, filename)
            _, extension = os.path.splitext(filename)
            new_name = f"{class_name}_{count}{extension}"
            new_path = os.path.join(dest_path, new_name)
            shutil.copy(old_path, new_path)
            count += 1


def move_files(src_dir: str, dest_dir: str = "data/2_processed/tmp"):
    """Move files to tmp directory in 2_processed.

    src_dir: directory of rock subclass with files [Basalt, Marble, Coal, ...]

    Parameters
    ----------
    src_dir : str
        _description_
    dest_dir : str, optional
        _description_, by default "data/2_processed/tmp"
    """
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    src_dir_name = os.path.basename(src_dir)
    dest_dir_path = os.path.join(dest_dir, src_dir_name.capitalize())
    os.makedirs(dest_dir_path, exist_ok=True)

    files = os.listdir(src_dir)
    total = len(files)
    for index, filename in tqdm(
        enumerate(files), desc=f"Moving {src_dir_name}", total=total
    ):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir_path, filename)
        # print("Copying", src_path, dest_path)
        shutil.copy(src_path, dest_path)
        # print(f"Moved {index+1} files from {src_dir} to {dest_dir_path}")


def move_and_rename(class_dir: str):
    """Move files from class_dir to tmp, renames them there based on count, and moves back to 2_processed class_dir: A class dir of supporting classes (Marble, Coal, ...), which contains image files.

    Parameters
    ----------
    class_dir : str
        _description_
    """
    target_classes = os.listdir("data/2_processed/")
    if "tmp" in target_classes:
        target_classes.remove("tmp")
    target_classes_lower = list(map(lambda x: x.lower(), target_classes))

    for subclass_dir in os.listdir(class_dir):
        if subclass_dir.lower() in target_classes_lower:
            subclass_dir_path = os.path.join(class_dir, subclass_dir)
            move_files(subclass_dir_path)
            rename_files()
            shutil.rmtree("data/2_processed/tmp")
