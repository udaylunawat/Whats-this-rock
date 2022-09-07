#!/bin/bash

# dataset 2 processing
kaggle datasets download mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals --path data/0_raw/
unzip -qn data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -d data/1_extracted/
mv data/1_extracted/Rock_Dataset data/1_extracted/dataset2
# https://serverfault.com/a/267266/979238
cp -r --backup=numbered data/1_extracted/dataset2/igneous\ rocks/Basalt/* data/2_processed/Basalt/
cp -r --backup=numbered data/1_extracted/dataset2/igneous\ rocks/granite/* data/2_processed/Granite/
cp -r --backup=numbered data/1_extracted/dataset2/metamorphic\ rocks/marble/* data/2_processed/Marble/
cp -r --backup=numbered data/1_extracted/dataset2/metamorphic\ rocks/quartzite/* data/2_processed/Quartzite/
cp -r --backup=numbered data/1_extracted/dataset2/sedimentary\ rocks/coal/* data/2_processed/Coal/
cp -r --backup=numbered data/1_extracted/dataset2/sedimentary\ rocks/Limestone/* data/2_processed/Limestone/
cp -r --backup=numbered data/1_extracted/dataset2/sedimentary\ rocks/Sandstone/* data/2_processed/Sandstone/
rm -rf data/1_extracted/dataset2/minerals