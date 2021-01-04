import numpy as np


def make_binary_target(
    data,
    target_col,
    positive_ratio=0.5,
    sep = ','
):
    all_target_col = 'all_' + target_col
    data[all_target_col] = sep.join(list(data[target_col].unique()))
    data[all_target_col] = data[all_target_col].str.split(sep)
    data = data.explode(all_target_col)
    data = data.reset_index(drop=True)

    data.loc[data[target_col] == data[all_target_col], 'target'] = 1
    data.loc[data[target_col] != data[all_target_col], 'target'] = 0
    data = data.drop(target_col, axis=1)
    data = data.rename({all_target_col: target_col}, axis=1)
    data = data.drop_duplicates()

    n_positive = int(data['target'].sum())
    if positive_ratio > 0:
        n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)
    else:
        n_negative = 0
    negative_index = np.random.choice(data[data['target'] == 0].index, n_negative)
    positive_index = data[data['target'] == 1].index
    data = data.loc[np.concatenate([positive_index, negative_index])]

    return data.sort_index().reset_index(drop=True)
