import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from torch.utils.data import Dataset


class DfDataset(Dataset):
    def __init__(self, df, open_fn, dict_transform=None, cache_prob=-1):

        self.df = df
        self.open_fn = open_fn
        self.dict_transform = dict_transform
        self.cache_prob = cache_prob
        self.cache = dict()
    
    def prepare_new_item(self, index):
        row = self.df[index]
        dict_ = self.open_fn(row)

        if self.dict_transform is not None:
            dict_ = self.dict_transform(dict_)

        return dict_

    def prepare_item_from_cache(self, index):
        return self.cache.get(index, None)

    def __getitem__(self, index):
        dict_ = None

        if random.random() < self.cache_prob:
            dict_ = self.prepare_item_from_cache(index)

        if dict_ is None:
            dict_ = self.prepare_new_item(index)
            if self.cache_prob > 0:
                self.cache[index] = dict_

        return dict_

    def __len__(self):
        return len(self.df)


def default_fold_split(df, folds_seed, n_folds):
    df = shuffle(df, random_state=folds_seed)

    df_tmp = []
    for i, df_el in enumerate(np.array_split(df, n_folds)):
        df_el["fold"] = i
        df_tmp.append(df_el)
    df = pd.concat(df_tmp)
    return df


def column_fold_split(df, column, folds_seed, n_folds):
    df_tmp = []
    labels = shuffle(sorted(df[column].unique()), random_state=folds_seed)
    for i, fold_labels in enumerate(np.array_split(labels, n_folds)):
        df_label = df[df[column].isin(fold_labels)]
        df_label["fold"] = i
        df_tmp.append(df_label)
    df = pd.concat(df_tmp)
    return df
