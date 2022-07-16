import os
import shutil
import argparse
import pandas as pd
# https://stackoverflow.com/a/64006242/9292995
import splitfolders
from data_utilities import get_all_filePaths, undersample_df, remove_corrupted_images


def setup_dirs_and_preprocess(args):
    if args.remove_class:
        shutil.rmtree(os.path.join(args.root, args.remove_class))

    all_paths = []
    all_classes = []

    class_dirs = os.listdir(args.root)
    for class_name in class_dirs:
        os.makedirs(os.path.join('data/2_processed', class_name))
        paths_list = get_all_filePaths(os.path.join(args.root, class_name))

        for image_path in paths_list:
            source = image_path

            target = os.path.join('data/2_processed', class_name)
            shutil.move(source, target)

            all_paths.append(os.path.join(target, os.path.basename(source)))
            all_classes.append(class_name)

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
        default='data/1_extracted/Rock_Dataset/',
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
    if args.oversample:
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        splitfolders.fixed('data/2_processed', output="data/4_tfds_dataset", oversample=True,
                           seed=1337)
    elif args.undersample:
        splitfolders.fixed('data/2_processed', output="data/4_tfds_dataset",
                           fixed=(int(args.undersample * 0.8), int(args.undersample * 0.2), int(args.undersample * 0.2)),
                           oversample=False,
                           seed=1337)
    else:
        splitfolders.ratio('data/2_processed', output="data/4_tfds_dataset",
                           ratio=(0.8, 0.1, 0.1),
                           seed=1337)

