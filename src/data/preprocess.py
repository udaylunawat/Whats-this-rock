# https://stackoverflow.com/a/64006242/9292995
import splitfolders
import os
import subprocess

from src.data.utils import *


def process_data(cfg):

    datasets = os.listdir('data/1_extracted')
    for dataset in datasets:
        print(f"\nProcessing {dataset}")
        for main_class in os.listdir(os.path.join('data/1_extracted',
                                                  dataset)):
            main_class_path = os.path.join('data/1_extracted/', dataset,
                                           main_class)
            move_and_rename(main_class_path)

    print("\nFiles other than jpg and png.\n\n")
    result = subprocess.run(
        ['ls', 'data/2_processed', '-I', '*.jpg', '-I', '*.png', '-R'],
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(result)

    print("\nFile types before cleaning:")
    get_value_counts('data/2_processed')

    remove_unsupported_images('data/2_processed')
    remove_corrupted_images('data/2_processed')

    print("\nFile types after cleaning:")
    get_value_counts('data/2_processed')

    print("\n", get_df().info(), "\n")
    print(get_df()["class"].value_counts())
    print(
        "\nSplitting files in Train, Validation and Test and saving to data/4_tfds_dataset/"
    )
    scc = min(get_df()["class"].value_counts())
    val_split = test_split = (1-cfg.train_split) / 2
    print(f"Data Split:- Training {cfg.train_split}, Validation {val_split}, Test {test_split}")
    if cfg.sampling == 'oversample':
        print("\nOversampling...")
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        print("Finding smallest class for oversampling fixed parameter.")
        print(f"Smallest class count is {scc}\n")
        splitfolders.fixed("data/2_processed",
                           output="data/4_tfds_dataset",
                           oversample=True,
                           fixed=(((scc // 2) - 1, (scc // 2) - 1)),
                           seed=cfg.seed,
                           move=False)
    elif cfg.sampling == 'undersample':
        print(f"Undersampling to {cfg.sampling} samples.")
        splitfolders.fixed("data/2_processed",
                           output="data/4_tfds_dataset",
                           fixed=(
                               int(scc * cfg.train_split),
                               int(scc * val_split),
                               int(scc * test_split),
                           ),
                           oversample=False,
                           seed=cfg.seed,
                           move=False)
    else:
        print("No Sampling.")
        splitfolders.ratio("data/2_processed",
                           output="data/4_tfds_dataset",
                           ratio=(cfg.train_split, val_split, test_split),
                           seed=cfg.seed,
                           move=False)
    print('\n\n')
