import os
import shutil
import argparse

# https://stackoverflow.com/a/64006242/9292995
import splitfolders
from data_utilities import remove_corrupted_images, get_df

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from ml_collections import config_dict
from ml_collections import config_flags

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config", "configs/baseline.py")

def create_classes_dir(args):
    for dataset in os.listdir(args.root):
        class_dirs = os.listdir(os.path.join(args.root, dataset))
        for class_name in class_dirs:
            sub_classes = os.listdir(os.path.join(args.root, dataset, class_name))
            for subclass in sub_classes:
                shutil.move(
                    os.path.join(args.root, dataset, class_name, subclass),
                    "data/2_processed",
                )

    shutil.rmtree(args.root)

def main(_):

    # Get configs from the config file.
    args = CONFIG.value

    # create_classes_dir(args)
    # remove_corrupted_images('data/2_processed')
    print("\n", get_df().info(), "\n")
    print(get_df()["class"].value_counts())
    print(
        "\nSplitting files in Train, Validation and Test and saving to data/4_tfds_dataset/"
    )
    if args.dataset_config.sampling == 'oversample':
        print("Oversampling...")
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        print("Finding smallest class for oversampling fixed parameter.")
        scc = min(get_df()["class"].value_counts())
        print(f"Smallest class count is {scc}\n")
        splitfolders.fixed(
            "data/2_processed",
            output="data/4_tfds_dataset",
            oversample=True,
            fixed=(((scc // 2) - 1, (scc // 2) - 1)),
            seed=args.seed,
        )
    elif isinstance(args.dataset_config.sampling, (int, float)):
        print(f"Undersampling to {args.dataset_config.sampling} samples.")
        splitfolders.fixed(
            "data/2_processed",
            output="data/4_tfds_dataset",
            fixed=(
                int(args.dataset_config.sampling * 0.75),
                int(args.dataset_config.sampling * 0.125),
                int(args.dataset_config.sampling * 0.125),
            ),
            oversample=False,
            seed=args.seed,
        )
    elif not args.dataset_config.sampling:
        print("No Sampling.")
        splitfolders.ratio(
            "data/2_processed",
            output="data/4_tfds_dataset",
            ratio=(0.75, 0.125, 0.125),
            seed=args.seed,
        )
if __name__ == "__main__":
    app.run(main)