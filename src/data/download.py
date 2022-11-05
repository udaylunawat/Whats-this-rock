import os
import logging
from src.data.utils import timer_func, find_filepaths


@timer_func
def download_datasets():
    """Download the dataset with dataset_id.

    Parameters
    ----------
    dataset_id : int
        Dataset number
    """
    data_dict = {
        1: {"script": "src/scripts/dataset1.sh", "filecount": 2083},
        2: {"script": "src/scripts/dataset2.sh", "filecount": 4553},
    }
    for dataset_id in data_dict:
        if not os.path.exists(
            os.path.join("data", "1_extracted", f"dataset{dataset_id}")
        ):
            print(f"Downloading dataset {dataset_id}...")
            os.system(f"sh {data_dict[dataset_id]['script']}")
        else:
            _, count = find_filepaths(
                os.path.join("data", "1_extracted", f"dataset{dataset_id}")
            )
            assert count == data_dict[dataset_id]["filecount"]
            print(f"dataset{dataset_id} already exists.")
            print(f"Total Files in dataset{dataset_id}:- {count}.\n")


if __name__ == "__main__":
    download_datasets()
