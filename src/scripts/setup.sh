#!/bin/bash

# installing
apt install -qq --allow-change-held-packages libcudnn8=8.1.1.33-1+cuda11.2
pip install -r requirements-dev.txt

sh src/scripts/clean_dir.sh
python src/data/download.py
python src/data/preprocess.py