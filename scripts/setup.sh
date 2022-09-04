#!/bin/bash

# installing
apt install -qq --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
pip install -r requirements-dev.txt --q

# setting up kaggle
wget -qq https://www.dropbox.com/s/ltp4ly8ilvxlgas/kaggle.json
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 /root/.kaggle/kaggle.json

# setting up data dir
rm -rf data/1_extracted data/2_processed data/3_consume data/4_tfds_dataset data/corrupted_images
mkdir -p data/0_raw data/1_extracted data/2_processed data/3_consume data/4_tfds_dataset data/corrupted_images checkpoints
mkdir -p data/2_processed/Coal/
mkdir -p data/2_processed/Basalt/
mkdir -p data/2_processed/Granite/
mkdir -p data/2_processed/Marble/
mkdir -p data/2_processed/Quartzite/
mkdir -p data/2_processed/Limestone/
mkdir -p data/2_processed/Sandstone/