import os
import shutil
import argparse
import pandas as pd
# https://stackoverflow.com/a/64006242/9292995
import splitfolders
from data_utilities import get_all_filePaths, remove_corrupted_images


def setup_dirs_and_preprocess(args):
    if args.remove_class:
        shutil.rmtree(os.path.join(args.root, args.remove_class))

    all_paths = []
    all_classes = []

    for dataset in os.listdir(args.root):
        class_dirs = os.listdir(os.path.join(args.root ,dataset))
        for class_name in class_dirs:
            sub_classes = os.listdir(os.path.join(args.root, dataset, class_name))
            for subclass in sub_classes:
                shutil.move(os.path.join(args.root, dataset, class_name, subclass), 'data/2_processed')

    all_paths = get_all_filePaths(os.path.join('data/2_processed'))
    all_classes = list(map(lambda x:x.split('/')[-2], all_paths))

    shutil.rmtree(args.root)
    data = pd.DataFrame(list(zip(all_paths, all_classes)),
                        columns=['image_path', 'classes'])

    data.to_csv(os.path.join("data/3_consume/", "all_image_paths.csv"))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default='data/1_extracted/',
        help="Root Folder")
    parser.add_argument(
        "-usample",
        "--undersample",
        type=int,
        help="Undersample Data")
    parser.add_argument(
        "-osample",
        "--oversample",
        action='store_true',
        help="Oversample Data")
    parser.add_argument(
        "-remove",
        "--remove_class",
        type=str,
        default=None,
        help="Remove class")

    args = parser.parse_args()

    root_dir = 'data/1_extracted/'

    remove_corrupted_images(root_dir)
    data = setup_dirs_and_preprocess(args)
    print("Splitting files in Train, Validation and Test and saving to data/4_tfds_dataset/")
    if args.oversample:
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        splitfolders.fixed('data/2_processed', output="data/4_tfds_dataset", oversample=True, fixed=(30),
                           seed=1)
    elif args.undersample:
        splitfolders.fixed('data/2_processed', output="data/4_tfds_dataset",
                           fixed=(int(args.undersample * 0.75), int(args.undersample * 0.125), int(args.undersample * 0.125)),
                           oversample=False,
                           seed=1)
    else:
        splitfolders.ratio('data/2_processed', output="data/4_tfds_dataset",
                           ratio=(0.75, 0.125, 0.125),
                           seed=1)
