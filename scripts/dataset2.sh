#!/bin/bash

# dataset 2 processing
kaggle datasets download mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals --path data/0_raw/
unzip -q data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -d data/1_extracted/
# https://serverfault.com/a/267266/979238
cp -r --backup=t data/1_extracted/Rock_Dataset/igneous\ rocks/Basalt/* data/2_processed/Basalt/
cp -r --backup=t data/1_extracted/Rock_Dataset/igneous\ rocks/granite/* data/2_processed/Granite/
cp -r --backup=t data/1_extracted/Rock_Dataset/metamorphic\ rocks/marble/* data/2_processed/Marble/
cp -r --backup=t data/1_extracted/Rock_Dataset/metamorphic\ rocks/quartzite/* data/2_processed/Quartzite/
cp -r --backup=t data/1_extracted/Rock_Dataset/sedimentary\ rocks/coal/* data/2_processed/Coal/
cp -r --backup=t data/1_extracted/Rock_Dataset/sedimentary\ rocks/Limestone/* data/2_processed/Limestone/
cp -r --backup=t data/1_extracted/Rock_Dataset/sedimentary\ rocks/Sandstone/* data/2_processed/Sandstone/
rm -rf data/1_extracted/Rock_Dataset/minerals