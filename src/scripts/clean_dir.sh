#!/bin/bash

# setting up data dir
rm -rf data/2_processed data/3_tfds_dataset data/corrupted_images data/bad_images
mkdir -p data/0_raw data/1_extracted data/2_processed data/3_tfds_dataset data/corrupted_images checkpoints
mkdir -p data/2_processed/Coal/
mkdir -p data/2_processed/Basalt/
mkdir -p data/2_processed/Granite/
mkdir -p data/2_processed/Marble/
mkdir -p data/2_processed/Quartzite/
mkdir -p data/2_processed/Limestone/
mkdir -p data/2_processed/Sandstone/