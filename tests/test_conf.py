import pytest
import omegaconf
import logging
from omegaconf import DictConfig


# path to the config file
path = "configs/config.yaml"

# load the config file
config = omegaconf.OmegaConf.load(path)


def get_logger():
    """Create and configure logger."""
    # configure logger
    logging.basicConfig(filename="logs.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # creating an object
    logger = logging.getLogger()

    # setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    return logger


def test_cfg_type_check():
    """Test configuration files."""
    logger = get_logger()
    # dictionary with key as data type and value as list of keys with that data type in the config
    data_types = {
        int: ['seed', 'lr_decay_steps', 'epochs', 'batch_size', 'num_classes', 'last_layers'],
        float: ['lr'],
        bool: ['class_weights', 'custom_callback', 'use_pretrained_weights', 'preprocess', 'save_model', 'trainable', 'remove_bad', 'remove_misclassified', 'remove_duplicates', 'remove_corrupted', 'remove_unsupported'],
        str: ['optimizer', 'loss', 'monitor', 'data_path', 'sampling', 'augmentation', 'backbone'],
        # DictConfig: ['callback', 'wandb']
    }


    # loop through the data types
    for data_type, keys in data_types.items():
        # loop through the keys
        for key in keys:
            # check the data type of each key
            assert isinstance(config[key], data_type)

    # log the success message
    logger.info(f'All configs have passed type check!')


def test_config_yml():
    """Test the config.yaml file."""
    logger = get_logger()

    # check the data type of each config value
    # assert type(config["metrics"]) == list
    assert type(config["train_split"]) == float
    assert type(config["image_size"]) == int
    assert type(config["image_channels"]) == int
    assert type(config["dropout_rate"]) == float
    assert type(config["earlystopping"]["use"]) == bool
    assert type(config["earlystopping"]["patience"]) == int
    assert type(config["reduce_lr"]["use"]) == bool
    assert type(config["reduce_lr"]["factor"]) == float
    assert type(config["reduce_lr"]["min_lr"]) == float
    assert type(config["reduce_lr"]["patience"]) == int
    assert type(config["wandb"]["project"]) == str
    assert type(config["wandb"]["mode"]) == str

    # check the range and value of train_split in the config
    assert 0 <= config["train_split"] <= 1

    # check the range of image_size in the config
    assert config["image_size"] in [224, 256, 384, 512]

    # check the range of image_channels in the config
    assert config["image_channels"] in [1, 3]

    # check the range of dropout_rate in the config
    assert 0 <= config["dropout_rate"] <= 1

    # log the success message
    logger.info("All configs have passed the test!")

    # log the success message
    logger.info("All configs have passed the test!")

