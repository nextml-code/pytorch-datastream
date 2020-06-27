from __future__ import annotations
from typing import Tuple, Union, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from datastream.tools import json


def split_dataframe(
    dataframe: pd.DataFrame,
    key_column: str,
    proportions: Dict[str, float],
    filepath: Optional[Path] = None,
    stratify_column: Optional[str] = None,
):
    '''
    Split and save result. Add new examples and continue from the old split.

    As new examples come in it can handle:
    - Changing test size
    - Adapt after removing examples from dataset
    - Adapt to new stratification
    '''
    if abs(sum(proportions.values()) - 1.0) >= 1e-6:
        raise AssertionError('Expected sum of proportions to be 1')
    
    if filepath is not None and filepath.exists():
        split = json.read(filepath)

        if set(proportions.keys()) != set(split.keys()):
            raise AssertionError(
                'Expected split names in split file to be the same as the keys'
                'of proportions'
            )
    else:
        split = {
            split_name: list()
            for split_name in proportions.keys()
        }

    split_proportions = tuple(proportions.items())

    for split_name, proportion in split_proportions[:-1]:
        if stratify_column is None:
            split = split_proportion(
                dataframe[key_column],
                proportion,
                split_name,
                split,
            )
        else:
            for strata in stratas(dataframe, stratify_column):
                split = split_proportion(
                    strata[key_column],
                    proportion,
                    split_name,
                    split,
                )

    last_split_name, _ = split_proportions[-1]
    split[last_split_name] += unassigned(dataframe[key_column], split)

    if filepath is not None:
        json.write(split, filepath)

    return {
        split_name: (
            dataframe[lambda df: df[key_column].isin(split[split_name])]
        )
        for split_name in proportions.keys()
    }


def stratas(dataframe, stratify_column):
    return [
        dataframe[lambda df: df[stratify_column] == strata_value]
        for strata_value in dataframe[stratify_column].unique()
    ]


def split_proportion(
    keys: pd.Series,
    proportion: float,
    split_name: str,
    previous_split: Dict[str, Tuple],
) -> Dict[str, Tuple]:
    unassigned_ = unassigned(keys, previous_split)

    if len(unassigned_) == 0:
        return previous_split
    else:
        n_previous_split = keys.isin(previous_split[split_name]).sum()

        n_target_split_ = n_target_split(keys, proportion)
        if n_target_split_ <= n_previous_split:
            return previous_split
        else:
            split = previous_split
            split[split_name] += selected(
                n_target_split_ - n_previous_split,
                unassigned_,
            )
            return split


def assigned(split):
    return sum(split.values(), list())


def unassigned(keys, split):
    return keys[~keys.isin(assigned(split))].tolist()


def n_target_split(keys, proportion):
    float_target_split = len(keys) * proportion

    probability = float_target_split - int(float_target_split)
    if probability >= 1e-6 and np.random.rand() <= probability:
        return int(float_target_split) + 1
    else:
        return int(float_target_split)


def selected(k, unassigned):
    return np.random.choice(
        unassigned, size=k, replace=False
    ).tolist()  


def test_split_dataframe():
    dataframe = pd.DataFrame(dict(
        index=np.arange(100),
        number=np.random.randn(100),
        stratify=np.concatenate([np.ones(70), np.zeros(30)]),
    ))

    split_file = Path('test_split_dataframe.json')

    split_dataframes = split_dataframe(
        dataframe,
        key_column='index',
        proportions=dict(
            gradient=0.8,
            early_stopping=0.1,
            compare=0.1,
        ),
        filepath=split_file,
        stratify_column='stratify',
    )

    split_file.unlink()

    assert tuple(map(len, split_dataframes.values())) == (80, 10, 10)
