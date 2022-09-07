# https://stackoverflow.com/a/64006242/9292995
import splitfolders
import os
from src.data_utilities import remove_unsupported_images, get_df


def process_data(config):

    # Get configs from the config file.
    args = config

    print("\nFiles other than .jpg.")
    os.system("ls data/2_processed/ -I "*.jpg"  -R"))

    remove_unsupported_images('data/2_processed')
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
