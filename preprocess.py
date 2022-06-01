import os
import shutil
import argparse
import pandas as pd
from data_utilities import get_all_filePaths, undersample_df


def setup_dirs_and_preprocess(args):

    all_paths = []
    all_classes = []
    root_path = args.root
    class_dirs = os.listdir(root_path)
    for class_name in class_dirs:
        os.makedirs(os.path.join('data/2_processed', class_name))
        paths_list = get_all_filePaths(os.path.join(root_path, class_name))

        for image_path in paths_list:
            source = image_path

            target = os.path.join('data/2_processed', class_name)
            shutil.move(source, target)

            all_paths.append(os.path.join(target, os.path.basename(source)))
            all_classes.append(class_name)

    shutil.rmtree(root_path)
    data = pd.DataFrame(list(zip(all_paths, all_classes)),
                        columns=['image_path', 'classes'])
    if args.undersample:
        data = undersample_df(data, 'classes')

    data.to_csv("training_data.csv")


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
        action='store_true',
        help="Undersample Data")
    parser.add_argument(
        "-nousample",
        "--no_undersample",
        action='store_false',
        help="Don't Undersample Data")

    args = parser.parse_args()

    setup_dirs_and_preprocess(args)
