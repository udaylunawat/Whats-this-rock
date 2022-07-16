#!/bin/bash

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle

rm -rf data
mkdir -p data/0_raw data/1_extracted data/2_processed data/3_consume data/4_tfds_dataset
kaggle datasets download mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals --path data/0_raw/
kaggle datasets download salmaneunus/rock-classification --path data/0_raw/
unzip -q data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -d data/1_extracted/
unzip -q data/0_raw/rock-classification.zip -d data/1_extracted/
chmod 600 /root/.kaggle/kaggle.json