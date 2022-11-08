#!/bin/bash

# dataset 2 processing
wget --quiet -O data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -nc https://huggingface.co/datasets/udayl/rocks/resolve/main/igneous-metamorphic-sedimentary-rocks-and-minerals.zip
unzip -qn data/0_raw/igneous-metamorphic-sedimentary-rocks-and-minerals.zip -d data/1_extracted/
mv data/1_extracted/Rock_Dataset data/1_extracted/dataset2

rm -rf data/1_extracted/dataset2/minerals

mv data/1_extracted/dataset2/igneous\ rocks/Basalt data/1_extracted/dataset2/
mv data/1_extracted/dataset2/igneous\ rocks/granite data/1_extracted/dataset2/
mv data/1_extracted/dataset2/metamorphic\ rocks/marble data/1_extracted/dataset2/
mv data/1_extracted/dataset2/metamorphic\ rocks/quartzite data/1_extracted/dataset2/
mv data/1_extracted/dataset2/sedimentary\ rocks/Limestone data/1_extracted/dataset2/
mv data/1_extracted/dataset2/sedimentary\ rocks/Sandstone data/1_extracted/dataset2/
mv data/1_extracted/dataset2/sedimentary\ rocks/coal data/1_extracted/dataset2/

mv data/1_extracted/dataset2/granite data/1_extracted/dataset2/Granite
mv data/1_extracted/dataset2/marble data/1_extracted/dataset2/Marble
mv data/1_extracted/dataset2/quartzite data/1_extracted/dataset2/Quartzite
mv data/1_extracted/dataset2/coal data/1_extracted/dataset2/Coal

rm -rf data/1_extracted/dataset2/igneous\ rocks
rm -rf data/1_extracted/dataset2/metamorphic\ rocks
rm -rf data/1_extracted/dataset2/sedimentary\ rocks
