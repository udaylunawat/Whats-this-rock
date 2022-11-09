# Release notes

<!-- do not remove -->

## 0.0.6




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