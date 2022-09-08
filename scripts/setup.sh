#!/bin/bash

# installing
apt install -qq --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
pip install -r requirements-dev.txt --q

# setting up kaggle
wget -qq https://www.dropbox.com/s/ltp4ly8ilvxlgas/kaggle.json
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 /root/.kaggle/kaggle.json
