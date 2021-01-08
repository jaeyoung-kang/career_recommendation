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

    data.loc[data[target_col] == data[all_target_col], 'label'] = 1
    data.loc[data[target_col] != data[all_target_col], 'label'] = 0
    data = data.drop(target_col, axis=1)
    data = data.rename({all_target_col: target_col}, axis=1)

    data = data.sort_values('label', ascending=False)
    data = data.drop_duplicates(
        subset=set(data.columns) - set(['label']),
    ).sort_index()

    n_positive = int(data['label'].sum())
    if positive_ratio > 0: # 0보다 작은 경우 그대로 아웃풋
        n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)
        negative_index = np.random.choice(data[data['label'] == 0].index, n_negative)
        positive_index = data[data['label'] == 1].index
        data = data.loc[np.concatenate([positive_index, negative_index])]

    return data.sort_index().reset_index(drop=True)
