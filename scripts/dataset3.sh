#!/bin/bash

# dataset 3 processing
wget --quiet -O data/0_raw/dataset3.zip https://github.com/SmartPracticeschool/llSPS-INT-3797-Rock-identification-using-deep-convolution-neural-network/raw/master/dataset.zip
unzip -q data/0_raw/dataset3.zip -d data/1_extracted/
cp -r --backup=t data/1_extracted/dataset/*/Basalt/* data/2_processed/Basalt/
cp -r --backup=t data/1_extracted/dataset/*/Granite/* data/2_processed/Granite/
cp -r --backup=t data/1_extracted/dataset/*/Marble/* data/2_processed/Marble/
cp -r --backup=t data/1_extracted/dataset/*/Quartzite/* data/2_processed/Quartzite/
cp -r --backup=t data/1_extracted/dataset/*/Limestone/* data/2_processed/Limestone/
cp -r --backup=t data/1_extracted/dataset/*/Sandstone/* data/2_processed/Sandstone/