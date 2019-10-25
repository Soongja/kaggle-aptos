import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('train.csv')

x = train_df['id_code'].values
y = train_df['diagnosis'].values

num_folds = 6
fold = 5

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2019)
skf.get_n_splits(x, y)

train_df['split'] = 'split'
for fold_idx, (train_index, val_index) in enumerate(skf.split(x, y)):
    if fold_idx == fold:
        print(fold_idx, len(train_index), len(val_index))
        train_df['split'].iloc[train_index] = 'train'
        train_df['split'].iloc[val_index] = 'val'
# print(non_missing_mask_count_df)

train_df.to_csv('folds/train_%sfold_%s.csv' % (num_folds, fold), index=False)
