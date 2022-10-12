# Whats-this-rock

## Rock Classification Telegram Bot

![code-size](https://img.shields.io/github/languages/code-size/udaylunawat/Whats-this-rock) ![repo-size](https://img.shields.io/github/repo-size/udaylunawat/Whats-this-rock) ![top-language](https://img.shields.io/github/languages/top/udaylunawat/Whats-this-rock)
[![Deploy to heroku.](https://github.com/udaylunawat/Whats-this-rock/actions/workflows/main.yaml/badge.svg)](https://github.com/udaylunawat/Whats-this-rock/actions/workflows/main.yaml)
<p align="left">
    <img src="imgs/marie.jpg " alt="What's my name?" width="200"/>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Table of Contents](#table-of-contents)
<!-- - [Motivation](#motivation) -->
- [Installation Steps](#🛠️-installation-steps)
  - [Use the Telegram Bot](#use-the-telegram-bot)
  - [Deploy the Telegram Bot](#deploy-telegram-bot)
  - [Train Model](#train-model)
- [Demo](#👨‍💻-demo)
- [Features I'd like to add](#features-id-like-to-add)
- [Technologies Used](#technologies-used)
- [Directory Tree](#directory-tree)
- [Contributing](#contributing)
- [License](#license)
<!-- - [Contact](#contact) -->

## 🛠️ Installation Steps

- ## Use the Telegram Bot

  You can try the bot [here](https://t.me/test7385_bot) on Telegram.

  Type `/help` to get instructions.

- ## Deploy Telegram Bot

  ```bash
  pip install -r requirements-prod.txt
  python src/bot.py
  ```

- ## Train Model

  > Paste your kaggle.json file in the root directory

### Run these commands

  ```bash
  pip install -r requirements-dev.txt
  sh src/scripts/setup.sh
  sh src/scripts/clean_dir.sh
  python src/models/train.py
  ```

  You can try different models and parameters by editing `config.json`.

  By using Hydra it's now much more easier to override parameters like this

  ```bash
  python src/models/train.py  callback.reduce_lr.factor=0.5 \
                              callback.reduce_lr.patience=4 \
                              callback.earlystopping.patience=20
  ```

## 👨‍💻 Demo

|  |  |  |
|---|---|---|
| ![ alt colab](https://www.tensorflow.org/images/colab_logo_32px.png)[Run in Colab](https://colab.research.google.com/drive/1N1CIqdOKlJSJla5PU53Yn9KWSao47eMv?usp=sharing) | ![ alt Source](https://www.tensorflow.org/images/GitHub-Mark-32px.png)[View Source on Github](https://github.com/udaylunawat/Whats-this-rock) | ![ alt noteboook](https://www.tensorflow.org/images/download_logo_32px.png)[Download Notebook]() |

## Features I'd like to add

- [x] Wandb
  - [x] Config
  - [x] Sweeps
- [x] Datasets
  - [x] 4 Datasets
- [x] Augmentation
- [x] Sampling
  - [x] Oversampling
  - [x] Undersampling
  - [x] class weights
- [x] Remove Corrupted Images
- [x] Try Multiple Optimizers
- [x] TFDS datasets
- [x] ImageDataGenerator
- [x] Models
  - [x] ConvNextTiny
  - [x] BaselineCNN
  - [x] Efficientnet
  - [x] Resnet101
  - [x] MobileNetv1
  - [x] MobileNetv2
  - [x] Xception
- [ ] LRScheduleer, LRDecay
  - [ ] Baseline without scheduler
  - [ ] Step decay
  - [ ] Cosine annealing
  - [ ] Classic cosine annealing with bathc steps w/o restart
- [x] Model Checkpoint, Resume Training
- [x] Evaluation
  - [x] Confusion Matrix
  - [x] Classification Report
- [x] Deploy Telegram Bot
  - [x] Heroku - Deprecated
  - [x] Railway
  - [x] Show CM and CL in bot
- [ ] Deploy to Huggingface spaces
- [ ] Accessing the model through FastAPI (Backend)
- [ ] Streamlit (Frontend)
- [x] Docker
- [x] GitHub Actions
  - [x] Deploy Bot when bot.py is updated.
  - [x] Lint code
- [ ] convert models.py to Classes and more OOP
- [ ] nbdev
- [ ] Group Runs
  - [ ] kfold cross validation
- [ ] [WandB Tables](https://twitter.com/ayushthakur0/status/1508962184357113856?s=21&t=VRL-ZXzznXV_Hg2h7QnjuA)
  - [ ] find the long tail examples or hard examples,
  - [ ] find the classes that the model is performing terribly on,
- [x] Configuration Management
  - [x] ml-collections
  - [x] Hydra
- [x] Performance improvement
  - [x] Convert to tf.data.Dataset
- [ ] Add Badges
  - [ ] Linting
  - [ ] Railway

## Technologies Used

- [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
- [Python](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Weights and Biases](https://wandb.ai/site)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [Heroku](https://heroku.com) - Deprecated as Heroku will no longer provide free-to-use service
- [Railway](https://railway.app)
- [Hydra - by Facebook](https://hydra.cc)

## Directory Tree

------------

    ├── imgs                              <- Images for skill banner, project banner and other images
    │
    ├── configs                           <- Configuration files
    │   ├── configs.yaml                  <- config for single run
    │   └── sweeps.yaml                   <- confguration file for sweeps hyperparameter tuning
    │
    ├── data
    │   ├── corrupted_images              <- corrupted images will be moved to this directory
    │   ├── sample_images                 <- Sample images for inference
    │   ├── 0_raw                         <- The original, immutable data dump.
    │   ├── 1_external                    <- Data from third party sources.
    │   ├── 2_interim                     <- Intermediate data that has been transformed.
    │   └── 3_processed                   <- The final, canonical data sets for modeling.
    │
    ├── notebooks                         <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                        the creator's initials, and a short `-` delimited description, e.g.
    │                                        1.0-jqp-initial-data-exploration`.
    │
    │
    ├── src                               <- Source code for use in this project.
    │   │
    │   ├── data                          <- Scripts to download or generate data
    │   │   ├── download.py
    │   │   ├── preprocess.py
    │   │   └── utils.py
    │   │
    │   ├── callbacks                     <- functions that are executed during training at given stages of the training procedure
    │   │   ├── custom_callbacks.py
    │   │   └── callbacks.py
    │   │
    │   ├── models                        <- Scripts to train models and then use trained models to make
    │   │   │                                predictions
    │   │   ├── evaluate.py
    │   │   ├── models.py
    │   │   ├── predict.py
    │   │   ├── train.py
    │   │   └── utils.py
    │   │
    │   └── scripts                       <- Scripts to setup dir structure and download datasets
    │   │   ├── clean_dir.sh
    │   │   ├── dataset1.sh
    │   │   ├── dataset2.sh
    │   │   ├── dataset3.sh
    │   │   ├── dataset4.sh
    │   │   └── setup.sh
    │.  │
    │   └── visualization                 <- Scripts for visualizations
    │
    ├── .dockerignore                     <- Docker ignore
    ├── .gitignore                        <- GitHub's excellent Python .gitignore customized for this project
    ├── LICENSE                           <- Your project's license.
    ├── Makefile                          <- Makefile with commands like `make data` or `make train`
    ├── README.md                         <- The top-level README for developers using this project.
    ├── requirements.txt                  <- The requirements file for reproducing the analysis environment, e.g.
    │                                        generated with `pip freeze > requirements.txt`
    └── setup.py                          <- makes project pip installable (pip install -e .) so src can be imported

## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/udaylunawat/Whats-this-rock/issues) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/udaylunawat/Whats-this-rock/issues). Please include sample queries and their corresponding results.

<!-- CONTRIBUTING -->
## 👨‍💻 Contributing

- Contributions make the open source community such an amazing place to learn, inspire, and create.
- Any contributions you make are **greatly appreciated**.
- Check out our [contribution guidelines](/CONTRIBUTING.md) for more information.

## 🛡️ License

LinkFree is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- [Dataset - by Mahmoud Alforawi](https://www.kaggle.com/datasets/mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals)

## 🙏 Support

This project needs a ⭐️ from you. Don't forget to leave a star ⭐️

<br>
<p align="center"> Walt might be the one who knocks <br> but Hank is the one who rocks. </br> </p>
