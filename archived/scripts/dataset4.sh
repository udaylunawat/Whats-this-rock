#!/bin/bash

# dataset 4 processing
kaggle datasets download neelgajare/rocks-dataset --path data/0_raw/
unzip -qn data/0_raw/rocks-dataset.zip -d data/1_extracted/
mkdir -p data/1_extracted/dataset4/rock_classes/
mv -n data/1_extracted/Rocks/* data/1_extracted/dataset4/rock_classes
rm -rf data/1_extracted/Rocks