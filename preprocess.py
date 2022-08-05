import os
import shutil
import argparse
import pandas as pd
# https://stackoverflow.com/a/64006242/9292995
import splitfolders
from data_utilities import remove_corrupted_images


def remove_class(args):
    if args.remove_class in os.listdir(args.root):
        shutil.rmtree(os.path.join(args.root, args.remove_class))

def create_classes_dir(args):
    for dataset in os.listdir(args.root):
        class_dirs = os.listdir(os.path.join(args.root ,dataset))
        for class_name in class_dirs:
            sub_classes = os.listdir(os.path.join(args.root, dataset, class_name))
            for subclass in sub_classes:
                shutil.move(os.path.join(args.root, dataset, class_name, subclass), 'data/2_processed')

    shutil.rmtree(args.root)


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

    remove_class(args)
    create_classes_dir(args)
    remove_corrupted_images('data/2_processed')
    print("Splitting files in Train, Validation and Test and saving to data/4_tfds_dataset/")
    if args.oversample:
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        splitfolders.fixed('data/2_processed', output="data/4_tfds_dataset", oversample=True, fixed=(65),
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

