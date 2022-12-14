import pytest
import os

from rocks_classifier.data.preprocess import process_data

def test_process_data():
    """Test the process_data function."""
    # test the process_data function with default configuration
    process_data()

    # test the process_data function with custom configuration
    cfg = {
        "train_split": 0.8,
        "validation_split": 0.1,
        "sampling": "undersampling",
    }
    process_data(cfg)
