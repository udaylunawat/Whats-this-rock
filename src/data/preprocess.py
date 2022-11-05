# https://stackoverflow.com/a/64006242/9292995
import os
import subprocess
import logging
import hydra

from src.data.utils import (
    get_df,
    get_value_counts,
    move_and_rename,
    remove_corrupted_images,
    remove_unsupported_images,
    move_to_processed,
    sampling,
    move_bad_files,
)


@hydra.main(config_path="../../configs", config_name="config", version_base='1.2')
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

    move_bad_files("configs/bad_images selected by gemini.txt", "data/bad_images", "Moving bad images...")
    move_bad_files("configs/misclassified selected by gemini.txt", "data/misclassified_images", "Moving misclassified images...")
    move_bad_files("configs/duplicates selected by gemini.txt", "data/duplicate_images", "Moving duplicate images...")

    remove_unsupported_images("data/2_processed")
    remove_corrupted_images("data/2_processed")

    print("\nFile types after cleaning:")
    get_value_counts("data/2_processed")

    print("\nCounts of classes:\n")
    print(get_df()["class"].value_counts())
    sampling(cfg)


if __name__ == "__main__":
    process_data()