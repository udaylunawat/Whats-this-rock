#!/bin/bash

# dataset 1 processing
kaggle datasets download salmaneunus/rock-classification --path data/0_raw/
unzip -qn data/0_raw/rock-classification.zip -d data/1_extracted/
cp -r data/1_extracted/Dataset/Igneous/* data/2_processed
cp -r data/1_extracted/Dataset/Metamorphic/* data/2_processed
cp -r data/1_extracted/Dataset/Sedimentary/* data/2_processed