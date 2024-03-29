#!/bin/bash

# installing
apt install -qq --allow-change-held-packages libcudnn8=8.1.1.33-1+cuda11.2
pip install -r requirements-dev.txt

sh rock_classifier/scripts/clean_dir.sh
python rock_classifier/data/download.py
python rock_classifier/data/preprocess.py