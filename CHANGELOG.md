# Release notes

<!-- do not remove -->

## 0.0.9

- Changed Branch from nbdev to main
- nbdev Branch deleted

### Bugs Squashed

- Hydra works using --config-dir argument
	- Copied config to cwd (user accessible)
	- Removed `download_configs`
- get_confusion_matrix and get_classification report refactored from train.py to visualization.py


## 0.0.8

### New Features

- Hydra now works using decorators and takes CLI arguments

### Bugs Squashed

- Remove Bad images directory
- Hydra relative path
- Hydra ignoring passed arguments


## 0.0.7

### New Features

- Logging duplicate lines
- Downloading configs from Github (nbdev repo)

### Docs

- Inspiration
- Directory Tree Updated
- Added console_script commands

## 0.0.6

### New Features

- Added console_scripts using `settings.ini`
    - rocks_clean_data
    - rocks_download_data
    - rocks_process_data
    - rocks_train_model
	- rocks_deploy_bot

- Converted shell scripts to python scripts.
    > No need to call those messy shell scripts anymore!
- Removed `src` directory, primarily writing packages using nbdev now.
- download_datasets is now a `class` called `download_and_move`.
- Downloading datasets directly from huggingface datasets instead of using kaggle cli.

### Bugs Squashed

- Skipping downloading the zip files if they already exist.
