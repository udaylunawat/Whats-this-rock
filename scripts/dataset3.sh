#!/bin/bash

# dataset 3 processing
wget --quiet -O data/0_raw/dataset3.zip https://github.com/SmartPracticeschool/llSPS-INT-3797-Rock-identification-using-deep-convolution-neural-network/raw/master/dataset.zip
unzip -qn data/0_raw/dataset3.zip -d data/1_extracted/
mv data/1_extracted/dataset data/1_extracted/dataset3

cp -r --backup=numbered data/1_extracted/dataset3/*/Basalt/* data/2_processed/Basalt/
cp -r --backup=numbered data/1_extracted/dataset3/*/Granite/* data/2_processed/Granite/
cp -r --backup=numbered data/1_extracted/dataset3/*/Marble/* data/2_processed/Marble/
cp -r --backup=numbered data/1_extracted/dataset3/*/Quartzite/* data/2_processed/Quartzite/
cp -r --backup=numbered data/1_extracted/dataset3/*/Limestone/* data/2_processed/Limestone/
cp -r --backup=numbered data/1_extracted/dataset3/*/Sandstone/* data/2_processed/Sandstone/