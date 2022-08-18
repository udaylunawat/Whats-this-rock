#!/bin/bash

# setting up kaggle
wget https://www.dropbox.com/s/9mhxs1p4mgb8gyh/kaggle.json
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 /root/.kaggle/kaggle.json

pip install -r requirements-dev.txt

# setting up data dir
rm -rf data/1_extracted data/2_processed data/3_consume data/4_tfds_dataset data/corrupted_images
mkdir -p data/0_raw data/1_extracted data/2_processed data/3_consume data/4_tfds_dataset data/corrupted_images checkpoints

# dataset 1 processing
kaggle datasets download salmaneunus/rock-classification --path data/0_raw/
unzip -q data/0_raw/rock-classification.zip -d data/1_extracted/
mv data/1_extracted/Dataset/Igneous/* data/2_processed
mv data/1_extracted/Dataset/Metamorphic/* data/2_processed
mv data/1_extracted/Dataset/Sedimentary/* data/2_processed

# dataset 2 processing
kaggle datasets download mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals --path data/0_raw/
unzip -q data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -d data/1_extracted/
# https://serverfault.com/a/267266/979238
mv --backup=t data/1_extracted/Rock_Dataset/igneous\ rocks/Basalt/* data/2_processed/Basalt/
mv --backup=t data/1_extracted/Rock_Dataset/igneous\ rocks/granite/* data/2_processed/Granite/
mv --backup=t data/1_extracted/Rock_Dataset/metamorphic\ rocks/marble/* data/2_processed/Marble/
mv --backup=t data/1_extracted/Rock_Dataset/metamorphic\ rocks/quartzite/* data/2_processed/Quartzite/
mv --backup=t data/1_extracted/Rock_Dataset/sedimentary\ rocks/coal/* data/2_processed/Coal/
mv --backup=t data/1_extracted/Rock_Dataset/sedimentary\ rocks/Limestone/* data/2_processed/Limestone/
mv --backup=t data/1_extracted/Rock_Dataset/sedimentary\ rocks/Sandstone/* data/2_processed/Sandstone/
rm -rf data/1_extracted/Rock_Dataset/minerals

# dataset 3 processing
wget --quiet -O data/0_raw/dataset3.zip https://github.com/SmartPracticeschool/llSPS-INT-3797-Rock-identification-using-deep-convolution-neural-network/raw/master/dataset.zip
unzip -q data/0_raw/dataset3.zip -d data/1_extracted/
mv --backup=t data/1_extracted/dataset/*/Basalt/* data/2_processed/Basalt/
mv --backup=t data/1_extracted/dataset/*/Granite/* data/2_processed/Granite/
mv --backup=t data/1_extracted/dataset/*/Marble/* data/2_processed/Marble/
mv --backup=t data/1_extracted/dataset/*/Quartzite/* data/2_processed/Quartzite/
mv --backup=t data/1_extracted/dataset/*/Limestone/* data/2_processed/Limestone/
mv --backup=t data/1_extracted/dataset/*/Sandstone/* data/2_processed/Sandstone/

# dataset 4 processing
kaggle datasets download neelgajare/rocks-dataset --path data/0_raw/
unzip -q data/0_raw/rocks-dataset.zip -d data/1_extracted/
cp -r --backup=t data/1_extracted/Rocks/Basalt/* data/2_processed/Basalt/
cp -r --backup=t data/1_extracted/Rocks/Granite/* data/2_processed/Granite/
cp -r --backup=t data/1_extracted/Rocks/Marble/* data/2_processed/Marble/
cp -r --backup=t data/1_extracted/Rocks/Quartzite/* data/2_processed/Quartzite/
cp -r --backup=t data/1_extracted/Rocks/Coal/* data/2_processed/Coal/
cp -r --backup=t data/1_extracted/Rocks/Limestone/* data/2_processed/Limestone/
cp -r --backup=t data/1_extracted/Rocks/Sandstone/* data/2_processed/Sandstone/