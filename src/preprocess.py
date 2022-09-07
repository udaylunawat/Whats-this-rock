# https://stackoverflow.com/a/64006242/9292995
import splitfolders
import os
import subprocess

from src.data_utilities import *



def process_data(config):

    # Get configs from the config file.
    args = config

    print("\nFiles other than jpg and png.")
    print(subprocess.run(["ls", "data/2_processed/", "-I", "*.jpg", "-I", "*.png"  "-R"], capture_output=True).stdout.decode())

    datasets = os.listdir('data/1_extracted')
    for dataset in datasets:
        for main_class in os.listdir(os.path.join('data/1_extracted', dataset)):
            main_class_path = os.path.join('data/1_extracted/',dataset, main_class)
            print(f"Processing {dataset}")
            move_and_rename(main_class_path)

    print("\nFile types before cleaning:")
    print(get_df('data/2_processed')['file_name'].apply(lambda x: x.split('.')[-1]).value_counts())
    remove_unsupported_images('data/2_processed')
    remove_corrupted_images('data/2_processed')
    print("\nFile types after cleaning:")
    print(get_df('data/2_processed')['file_name'].apply(lambda x: x.split('.')[-1]).value_counts())
    print("\n", get_df().info(), "\n")
    print(get_df()["class"].value_counts())
    print(
        "\nSplitting files in Train, Validation and Test and saving to data/4_tfds_dataset/"
    )
    scc = min(get_df()["class"].value_counts())
    if args.dataset_config.sampling == 'oversample':
        print("\nOversampling...")
        # If your datasets is balanced (each class has the same number of samples), choose ratio otherwise fixed.
        print("Finding smallest class for oversampling fixed parameter.")
        print(f"Smallest class count is {scc}\n")
        splitfolders.fixed("data/2_processed",
                           output="data/4_tfds_dataset",
                           oversample=True,
                           fixed=(((scc // 2) - 1, (scc // 2) - 1)),
                           seed=args.seed,
                           move=False)
    elif args.dataset_config.sampling == 'undersample':
        print(f"Undersampling to {args.dataset_config.sampling} samples.")
        splitfolders.fixed("data/2_processed",
                           output="data/4_tfds_dataset",
                           fixed=(
                               int(scc * 0.75),
                               int(scc * 0.125),
                               int(scc * 0.125),
                           ),
                           oversample=False,
                           seed=args.seed,
                           move=False)
    elif not args.dataset_config.sampling:
        print("No Sampling.")
        splitfolders.ratio("data/2_processed",
                           output="data/4_tfds_dataset",
                           ratio=(0.75, 0.125, 0.125),
                           seed=args.seed,
                           move=False)
    print('\n\n')
