#!/bin/bash

# dataset 4 processing
kaggle datasets download neelgajare/rocks-dataset --path data/0_raw/
unzip -qn data/0_raw/rocks-dataset.zip -d data/1_extracted/
mv data/1_extracted/Rocks data/1_extracted/dataset4

cp -r --backup=numbered data/1_extracted/dataset4/Basalt/* data/2_processed/Basalt/
cp -r --backup=numbered data/1_extracted/dataset4/Granite/* data/2_processed/Granite/
cp -r --backup=numbered data/1_extracted/dataset4/Marble/* data/2_processed/Marble/
cp -r --backup=numbered data/1_extracted/dataset4/Quartzite/* data/2_processed/Quartzite/
cp -r --backup=numbered data/1_extracted/dataset4/Coal/* data/2_processed/Coal/
cp -r --backup=numbered data/1_extracted/dataset4/Limestone/* data/2_processed/Limestone/
cp -r --backup=numbered data/1_extracted/dataset4/Sandstone/* data/2_processed/Sandstone/