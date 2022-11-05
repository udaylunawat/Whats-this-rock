#!/bin/bash

# installing
apt install -qq --allow-change-held-packages libcudnn8=8.1.1.33-1+cuda11.2
pip install -r requirements-dev.txt

# setting up kaggle
wget -qq https://www.dropbox.com/s/ltp4ly8ilvxlgas/kaggle.json
mkdir -p ~/.kaggle
mv -n kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

sh src/scripts/clean_dir.sh
python src/data/preprocess.py