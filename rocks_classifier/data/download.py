"""Dowload datasets and move them to the right folder.

`run_scripts` is the main function for doing the whole process.
"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../notebooks/01_b_download.ipynb.

# %% auto 0
__all__ = ['download_and_move', 'download_and_move_datasets']

# %% ../../notebooks/01_b_download.ipynb 1
_doc_ = """Dowload datasets and move them to the right folder.

`run_scripts` is the main function for doing the whole process.
"""

# %% ../../notebooks/01_b_download.ipynb 11
import os
import shutil
import requests
from .utils import timer_func, find_filepaths
from .utils import copy_configs_tocwd

# %% ../../notebooks/01_b_download.ipynb 12
class download_and_move:
    """Downloads datasets(zip files), extracts them to the correct folders, and rearranges them."""

    data_dict = {
        1: {
            "url": "https://huggingface.co/datasets/udayl/rocks/resolve/main/rock-classification.zip",
            "file_name": "rock-classification.zip",
            "folder_name": "Dataset",
            "filecount": 2083,
        },
        2: {
            "url": "https://huggingface.co/datasets/udayl/rocks/resolve/main/igneous-metamorphic-sedimentary-rocks-and-minerals.zip",
            "file_name": "igneous-metamorphic-sedimentary-rocks-and-minerals.zip",
            "folder_name": "Rock_Dataset",
            "filecount": 546,
        },
    }
    classes = [
        "Coal",
        "Basalt",
        "Granite",
        "Marble",
        "Quartzite",
        "Limestone",
        "Sandstone",
    ]

    @timer_func  # | hide_line
    def run_scripts(self):
        """
        Download the datasets using scripts.

        Uses `find_filepaths` to recursively find paths for all files in a directory.
        """
        copy_configs_tocwd()
        self.clean_data_dir()
        for dataset_id in self.data_dict:
            if self.archives_exist(dataset_id) and self.files_exists(dataset_id):
                # if both zip files exist and are extracted
                print(f"Dataset{dataset_id} already exists.")
                count = self.verify_files(dataset_id)
                print(f"Total Files in dataset{dataset_id}:- {count}.\n")
            if not self.archives_exist(dataset_id):
                # if zip files do not exist
                print(f"Downloading dataset {dataset_id}...")
                self.download_file(dataset_id)
            if not self.files_exists(dataset_id):
                # if zip files exists but they're not extracted
                print(f"Extracting dataset {dataset_id}...")
                self.extract_archive(f"data/0_raw/dataset{dataset_id}.zip")
                shutil.move(
                    f'data/1_extracted/{self.data_dict[dataset_id]["folder_name"]}',
                    f"data/1_extracted/dataset{dataset_id}",
                )
            self.move_subclasses_to_root_dir(dataset_id)

    def clean_data_dir(self):
        """Clean all data directories except 0_raw."""
        dir_0 = "0_raw"
        dir_1 = "1_extracted"
        dir_2 = "2_processed"
        dir_3 = "3_tfds_dataset"

        dir_list = [
            dir_1,
            dir_2,
            dir_3,
            "corrupted_images",
            "duplicate_images",
            "misclassified_images",
            "bad_images",
        ]

        classes = [
            "Coal",
            "Basalt",
            "Granite",
            "Marble",
            "Quartzite",
            "Limestone",
            "Sandstone",
        ]
        os.makedirs(os.path.join("data", dir_0), exist_ok=True)
        print("Cleaning data dir...")
        for dir_name in dir_list:
            for root, dirs, files in os.walk(os.path.join("data", dir_name)):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            os.makedirs(os.path.join("data", dir_name), exist_ok=True)

        for class_name in classes:
            os.makedirs(os.path.join("data", dir_2, class_name))

    def download_file(self, dataset_id, dest_dir="data/0_raw/"):
        """Download and write file to destination directory."""
        r = requests.get(self.data_dict[dataset_id]["url"], allow_redirects=True)
        open(os.path.join(dest_dir, f"dataset{dataset_id}.zip"), "wb").write(r.content)

    def extract_archive(self, file_path, dest_dir="data/1_extracted"):
        """Extract zip file to dest_dir."""
        shutil.unpack_archive(file_path, dest_dir, "zip")

    def archives_exist(self, dataset_id) -> bool:
        """Check if zip files already exist.

        Parameters
        ----------
        dataset_id : int
            dataset number

        Returns
        -------
        boolean
            returns True if zip files exist
        """
        if os.path.exists(os.path.join("data/0_raw", f"dataset{dataset_id}.zip")):
            return True

    def files_exists(self, dataset_id):
        """Check whether extracted files exist."""
        if os.path.exists(os.path.join("data", "1_extracted", f"dataset{dataset_id}")):
            count = self.verify_files(dataset_id)
            return count
        else:
            return False

    def move_subclasses_to_root_dir(self, dataset_id):
        """Move subclasses to data/2_processed."""
        root_path = f"data/1_extracted/dataset{dataset_id}"
        for class_name in os.listdir(f"data/1_extracted/dataset{dataset_id}"):
            class_path = os.path.join(root_path, class_name)
            for subclass in os.listdir(class_path):
                if subclass.capitalize() in self.classes:
                    folder_path = os.path.join(class_path, subclass)
                    shutil.move(folder_path, f"data/1_extracted/dataset{dataset_id}/")
                    shutil.move(
                        f"{root_path}/{subclass}",
                        f"{root_path}/{subclass.capitalize()}",
                    )
            shutil.rmtree(class_path, ignore_errors=False)

    def verify_files(self, dataset_id):
        """Verify the image counts."""
        _, count = find_filepaths(
            os.path.join("data", "1_extracted", f"dataset{dataset_id}")
        )
        assert count == self.data_dict[dataset_id]["filecount"]
        return count

# %% ../../notebooks/01_b_download.ipynb 13
def download_and_move_datasets():
    """Run the download and move datasets script."""
    download_and_move().run_scripts()
    print("Download and move process finished!")
