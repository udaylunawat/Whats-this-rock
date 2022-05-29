import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# https://towardsdatascience.com/stratified-sampling-you-may-have-been-splitting-your-dataset-all-wrong-8cfdd0d32502
def get_stratified_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Shuffle
    df_sample = df.sample(frac=1, random_state=12)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:
      grouped_df = df_sample.groupby(target_variable)
      arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

      train_ds = pd.concat([t[0] for t in arr_list])
      val_ds = pd.concat([t[1] for t in arr_list])
      test_ds = pd.concat([v[2] for v in arr_list])

    else:
      indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
      train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds