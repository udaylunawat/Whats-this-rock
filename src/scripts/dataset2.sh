#!/bin/bash

# dataset 2 processing
kaggle datasets download mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals --path data/0_raw/
unzip -qn data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -d data/1_extracted/
mv data/1_extracted/Rock_Dataset data/1_extracted/dataset2

rm -rf data/1_extracted/dataset2/minerals