#!/bin/bash

# dataset 4 processing
kaggle datasets download neelgajare/rocks-dataset --path data/0_raw/
unzip -qn data/0_raw/rocks-dataset.zip -d data/1_extracted/
cp -r --backup=t data/1_extracted/Rocks/Basalt/* data/2_processed/Basalt/
cp -r --backup=t data/1_extracted/Rocks/Granite/* data/2_processed/Granite/
cp -r --backup=t data/1_extracted/Rocks/Marble/* data/2_processed/Marble/
cp -r --backup=t data/1_extracted/Rocks/Quartzite/* data/2_processed/Quartzite/
cp -r --backup=t data/1_extracted/Rocks/Coal/* data/2_processed/Coal/
cp -r --backup=t data/1_extracted/Rocks/Limestone/* data/2_processed/Limestone/
cp -r --backup=t data/1_extracted/Rocks/Sandstone/* data/2_processed/Sandstone/