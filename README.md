# Whats-this-rock

## Rock Classification Telegram Bot

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/udaylunawat/Whats-this-rock/Lint%20Code%20Base)
![code-size](https://img.shields.io/github/languages/code-size/udaylunawat/Whats-this-rock)
![repo-size](https://img.shields.io/github/repo-size/udaylunawat/Whats-this-rock)
![top-language](https://img.shields.io/github/languages/top/udaylunawat/Whats-this-rock)

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
![GitHub issues](https://img.shields.io/github/issues-raw/udaylunawat/Whats-this-rock)
[![HitCount](https://hits.dwyl.com/udaylunawat/Whats-this-rock.svg?style=flat)](http://hits.dwyl.com/udaylunawat/Whats-this-rock)

![Twitter Follow](https://img.shields.io/twitter/follow/udaylunawat?style=social)

This project deploys a telegram bot that classifies rock images into 1 of 7 types.

<p >
    <img src="imgs/marie.jpg " alt="What's my name?" width="240" align="right"/>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Table of Contents](#table-of-contents)
<!-- - [Motivation](#motivation) -->
- [Installation Steps](#🛠️-installation-steps)
  - [Use the Telegram Bot](#use-the-telegram-bot)
  - [Deploy the Telegram Bot](#deploy-telegram-bot)
  - [Train Model](#train-model)
  - [Hyperparameter Tuning](#wandb-sweeps-hyperparameter-tuning)
- [Demo](#👨‍💻-demo)
- [Features I'd like to add](#features-id-like-to-add)
- [Technologies Used](#technologies-used)
- [Directory Tree](#directory-tree)
- [Contributing](#contributing)
- [License](#license)
<!-- - [Contact](#contact) -->

## 🛠️ Installation Steps

  > ## Use the Telegram Bot

  You can try the bot [here](https://t.me/test7385_bot) on Telegram.

  Type `/help` to get instructions.

  > ## Deploy Telegram Bot

  ```bash
  pip install -r requirements-prod.txt
  python src/bot.py
  ```

  >## Train Model

- Paste your kaggle.json file in the root directory

    Run these commands

    ```bash
    pip install -r requirements-dev.txt
    sh src/scripts/setup.sh
    python src/models/train.py
    ```

    You can try different models and parameters by editing `config.json`.

    By using Hydra it's now much more easier to override parameters like this

    ```bash
    python src/models/train.py  wandb.project=Whats-this-rockv \
                                dataset_id=[1,2,3,4] \
                                epochs=50 \
                                backbone=resnet
    ```

    <p align="left">
      <img src="imgs/result.png " alt="result" width="1500"/>
    </p>

  > ## Wandb Sweeps (Hyperparameter Tuning)

- Edit configs/sweeps.yaml

    ```bash
    wandb sweep \
    --project Whats-this-rock \
    --entity udaylunawat \
    configs/sweep.yaml
    ```

      This will return a command with $sweepid

    ```bash
    wandb agent udaylunawat/Whats-this-rock/$sweepid
    ```

## 👨‍💻 Demo

|  |  |  |
|---|---|---|
| ![ alt colab](https://www.tensorflow.org/images/colab_logo_32px.png)[Run in Colab](https://colab.research.google.com/drive/1N1CIqdOKlJSJla5PU53Yn9KWSao47eMv?usp=sharing) | ![ alt Source](https://www.tensorflow.org/images/GitHub-Mark-32px.png)[View Source on Github](https://github.com/udaylunawat/Whats-this-rock) | ![ alt noteboook](https://www.tensorflow.org/images/download_logo_32px.png)[Download Notebook]() |

## Features

<table border="0" class="left">
 <tr>
    <td><b><style="font-size:37px">Features added</b></td>
    <td><b><style="font-size:37px">Features planned</b></td>
 </tr>
 <tr>
    <td>

- Wandb
- Datasets
  - 4 Datasets
- Augmentation
  - keras-cv
  - Regular Augmentation
- Sampling
  - Oversampling
  - Undersampling
  - Class weights
- Remove Corrupted Images
- Try Multiple Optimizers (Adam, RMSProp, AdamW, SGD)
- Generators
  - TFDS datasets
  - ImageDataGenerator
- Models
  - ConvNextTiny
  - BaselineCNN
  - Efficientnet
  - Resnet101
  - MobileNetv1
  - MobileNetv2
  - Xception
- LRScheduleer, LRDecay
  - Baseline without scheduler
  - Step decay
  - Cosine annealing
  - Classic cosine annealing with bathc steps w/o restart
- Model Checkpoint, Resume Training
- Evaluation
  - Confusion Matrix
  - Classification Report
- Deploy Telegram Bot
  - Heroku - Deprecated
  - Railway
  - Show CM and CL in bot
- Docker
- GitHub Actions
  - Deploy Bot when bot.py is updated.
  - Lint code using GitHub super-linter
- Configuration Management
  - ml-collections
  - Hydra
- Performance improvement
  - Convert to tf.data.Dataset
- Linting & Formatting
  - Black
  - Flake8
  - isort
  - pydocstyle
    </td>
    <td>
- [ ] Deploy to Huggingface spaces
- [ ] Accessing the model through FastAPI (Backend)
- [ ] Streamlit (Frontend)
- [ ] convert models.py to Classes and more OOP style
- [ ] nbdev
- [ ] Group Runs
  - [ ] kfold cross validation
- [ ] [WandB Tables](https://twitter.com/ayushthakur0/status/1508962184357113856?s=21&t=VRL-ZXzznXV_Hg2h7QnjuA)
- [ ] find the long tail examples or hard examples,
- [ ] find the classes that the model is performing terribly on,
- [ ] Add Badges
  - [ ] Linting
  - [ ] Railway
  </td>

 </tr>
</table>

## Technologies Used
|  |  |  |
|--|--|--|
[![Google Colab][colab-shield]][googlecolab]|[![python-telegram-bot][telegram-shield]][python-telegram-bot]|[![Railway][railway-shield]][Railway]
[![Jupyter Notebook][jupyter-shield]][Jupyter]|[![Python][python-shield]][Python]|[![GitHub Actions][githubactions-shield]][GithubActions]
[![Weights & Biases][wandb-shield]][wandb]|[![TensorFlow][tensorflow-shield]][TensorFlow]|[![macOS][mac-shield]](Macos)
[![Docker][docker-shield]][Docker]|[![Git][git-shield]]()|[![Hydra][hydra-shield]][Hydra]
[![Black][black-shield]][black]|  |

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


[googlecolab]: https://colab.research.google.com/drive/1N1CIqdOKlJSJla5PU53Yn9KWSao47eMv?usp=sharing
[python-telegram-bot]: https://github.com/python-telegram-bot/python-telegram-bot
[railway]: https://railway.app
[jupyter]: https://jupyter.org
[python]: https://www.python.org/
[gitHubActions]: https://github.com/features/actions
[wandb]: http://wandb.ai
[TensorFlow]: https://www.tensorflow.org/
[docker]: http://docker.com
[git]: https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=black
[hydra]: http://hydra.cc
[black]: http://github.com/psf/black
[macos]: https://apple.com/macos

[colab-shield]: https://img.shields.io/badge/Compute-Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff&style=for-the-badge
[telegram-shield]: https://img.shields.io/badge/ChatBot-Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=black
[railway-shield]: https://img.shields.io/badge/Deployment-Railway-131415?style=for-the-badge&logo=railway&logoColor=black
[python-shield]: https://img.shields.io/badge/Language-python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[jupyter-shield]: https://img.shields.io/badge/Coding-jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=black
[githubactions-shield]: https://img.shields.io/badge/CI-github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=black
[wandb-shield]: https://img.shields.io/badge/MLOps-Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=000&style=for-the-badge
[tensorflow-shield]: https://img.shields.io/badge/ML_Framework-TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=black
[mac-shield]: https://img.shields.io/badge/OS-mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0
[docker-shield]: https://img.shields.io/badge/Container-docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=black
[git-shield]: https://img.shields.io/badge/Version_Control-git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=black
[hydra-shield]: https://img.shields.io/badge/config-hydra1.1-89b8cd?style=for-the-badge&labelColor=gray
[black-shield]: https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray

