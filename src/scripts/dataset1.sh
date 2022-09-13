#!/bin/bash

# dataset 1 processing
kaggle datasets download salmaneunus/rock-classification --path data/0_raw/
unzip -qn data/0_raw/rock-classification.zip -d data/1_extracted/
mv -vn data/1_extracted/Dataset data/1_extracted/dataset1