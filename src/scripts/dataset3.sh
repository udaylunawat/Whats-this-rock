#!/bin/bash

# dataset 3 processing
wget --quiet -O data/0_raw/dataset3.zip https://github.com/SmartPracticeschool/llSPS-INT-3797-Rock-identification-using-deep-convolution-neural-network/raw/master/dataset.zip
unzip -qn data/0_raw/dataset3.zip -d data/1_extracted/
mv -vn data/1_extracted/dataset data/1_extracted/dataset3
