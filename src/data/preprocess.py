# https://stackoverflow.com/a/64006242/9292995
import os
import subprocess
import logging


from src.data.utils import (
    get_df,
    get_value_counts,
    move_and_rename,
    remove_corrupted_images,
    remove_unsupported_images,
    move_to_processed,
    sampling
)


def process_data(cfg):
    """Download dataset, removes unsupported and corrupted images, and splits data into train, val and test.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration
    """
    move_to_processed()

    print("\n\nFiles other than jpg and png.\n")
    result = subprocess.run(
        ["ls", "data/2_processed", "-I", "*.jpg", "-I", "*.png", "-R"],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    print(result)

    print("\nFile types before cleaning:")
    get_value_counts("data/2_processed")

    remove_unsupported_images("data/2_processed")
    remove_corrupted_images("data/2_processed")

    print("\nFile types after cleaning:")
    get_value_counts("data/2_processed")

    print("\nCounts of classes:\n")
    print(get_df()["class"].value_counts())
    sampling(cfg)
