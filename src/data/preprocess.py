import os
import subprocess
import logging
import hydra

from src.data.download import download_datasets
from src.data.utils import (
    find_filepaths,
    get_df,
    get_value_counts,
    move_to_processed,
    sampling,
    clean_images,
)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def process_data(cfg):
    """Download dataset, removes unsupported and corrupted images, and splits data into train, val and test.

    Parameters
    ----------
    cfg : cfg (omegaconf.DictConfig):
        Hydra Configuration
    """
    os.system('sh rocks_classifier/scripts/clean_dir.sh')
    download_datasets()
    move_to_processed()

    print("\n\nFiles other than jpg and png.\n")
    files, _ = find_filepaths('data/2_processed/')
    print('\n'.join(list(filter(lambda x: not x.endswith('jpg') and not x.endswith('png'), files))))

    print("\nFile types before cleaning:")
    get_value_counts("data/2_processed")

    clean_images(cfg)

    print("\nFile types after cleaning:")
    get_value_counts("data/2_processed")

    print("\nCounts of classes:\n")
    get_value_counts("data/2_processed", column="class")

    sampling(cfg)


if __name__ == "__main__":
    process_data()
