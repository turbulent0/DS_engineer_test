import pandas as pd
import numpy as np
import warnings
from scipy import sparse

def drop_corr_func_with_col(dataset, y, nan_friendly=False):
    # функция удаляет один столбец из пары столбцов, между которыми корреляция >= 0.95. остается тот столбец, у которого корреляция с таргетом выше
    # dataset - датафрейм вместе со столбцом таргета, y - название столбца таргета
    # nan_friendly - если False - в датафрейме не должно быть Nan-полей, если True - Nan могут быть в датафрейме, при расчете значений корреляций будут учитываться только не-Nan поля в соответствующих столбцах
    dataset.insert(0, y, dataset.pop(y))

    if not nan_friendly:
        ck1 = pd.DataFrame(abs(np.corrcoef(dataset.values, rowvar=False)), index=dataset.columns, columns=dataset.columns).fillna(0)  #формируем матрицу корреляций - в абсолютных значениях
    else:
        #ck1 = pd.DataFrame(abs(ma.corrcoef(ma.masked_invalid(dataset.values), rowvar=False)), index=dataset.columns, columns=dataset.columns).fillna(0)
        ck1 = dataset.corr().abs().fillna(0)  # формируем матрицу корреляций - в абсолютных значениях

    upper_tri = ck1.where((np.triu(np.ones(ck1.shape), k=1).astype(np.bool)) | (ck1.columns == y)).drop(y, axis=0)  # оставляем только верхний треугольник без диагонали и корреляцию фичей с таргетом

    upper_tri_cx = upper_tri.copy()
    upper_tri_cx[(upper_tri_cx < 0.95) | (upper_tri_cx.columns == y)] = 0
    upper_tri_cx.fillna(0, inplace=True)

    cx = sparse.coo_matrix(upper_tri_cx)

    drop_col = []
    for row, col in zip(cx.row, cx.col):
        if upper_tri.columns[col] in drop_col or upper_tri.columns[row + 1] in drop_col:
            continue
        else:
            if (upper_tri.iloc[row, 0] < upper_tri.iloc[col - 1, 0]):
                drop_col.append(upper_tri.columns[row + 1])
            else:
                drop_col.append(upper_tri.columns[col])

    dataset.drop(drop_col, axis='columns', inplace=True)  # дропаем фичи из списка из датасета
    return dataset, drop_col