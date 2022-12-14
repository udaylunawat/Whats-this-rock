import pytest
import os

from rocks_classifier.data.download import download_and_move
from rocks_classifier.data.utils import find_filepaths

@pytest.fixture
def downloader():
    return download_and_move()

def test_download_and_move(downloader):
    """Test the download_and_move function."""
    downloader.run_scripts()

    assert os.path.exists("data/0_raw/dataset1.zip")
    assert os.path.exists("data/0_raw/dataset2.zip")
    assert os.path.exists("data/1_extracted/dataset1")
    assert os.path.exists("data/1_extracted/dataset2")

    files, count = find_filepaths("data/1_extracted/dataset1")
    assert count == downloader.data_dict[1]["filecount"]

    files, count = find_filepaths("data/1_extracted/dataset2")
    assert count == downloader.data_dict[2]["filecount"]
