from __future__ import annotations
from typing import Tuple, Dict, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd


def split_dataframes(
    dataframe: pd.DataFrame,
    key_column: str,
    proportions: Dict[str, float],
    stratify_column: Optional[str] = None,
    filepath: Optional[Path] = None,
    frozen: Optional[bool] = False,
):
    '''
    Split and save result. Add new examples and continue from the old split.

    As new examples come in it can handle:
    - Changing test size
    - Adapt after removing examples from dataset
    - Adapt to new stratification
    '''
    if abs(sum(proportions.values()) - 1.0) >= 1e-5:
        raise ValueError(' '.join([
            'Expected sum of proportions to be 1.',
            f'Proportions were {tuple(proportions.values())}',
        ]))

    if filepath is not None and filepath.exists():
        split = json.loads(filepath.read_text())

        if set(proportions.keys()) != set(split.keys()):
            raise ValueError(' '.join([
                'Expected split names in split file to be the same as the',
                'keys in proportions',
            ]))
    else:
        split = {
            split_name: list()
            for split_name in proportions.keys()
        }

    if dataframe[key_column].nunique() != len(dataframe):
        raise ValueError(f'key_column {key_column} contains duplicate values')

    if frozen:
        if sum(map(len, split.values())) == 0:
            raise ValueError('Frozen split is empty')
    else:
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
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(json.dumps(split, indent=4))

    return {
        split_name: (
            dataframe[dataframe[key_column].isin(split[split_name])]
        )
        for split_name in proportions.keys()
    }


def group_split_dataframes(
    dataframe: pd.DataFrame,
    split_column: str,
    proportions: Dict[str, float],
    filepath: Optional[Path] = None,
    frozen: Optional[bool] = False,
):
    key_dataframe = pd.DataFrame(dict(key=dataframe[split_column].unique()))
    splits = split_dataframes(
        key_dataframe, 'key', proportions, filepath=filepath, frozen=frozen
    )
    return {
        split_name: (
            dataframe[dataframe[split_column].isin(split['key'])]
        )
        for split_name, split in splits.items()
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


def mock_dataframe():
    return pd.DataFrame(dict(
        index=np.arange(100),
        number=np.random.randn(100),
        stratify=np.concatenate([np.ones(70), np.zeros(30)]),
    ))


def test_standard():
    split_file = Path('test_standard.json')
    split_dataframes_ = split_dataframes(
        mock_dataframe(),
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

    assert tuple(map(len, split_dataframes_.values())) == (80, 10, 10)


def test_group_split_dataframe():
    dataframe = mock_dataframe().assign(group=lambda df: df['index'] // 4)
    split_dataframes_ = group_split_dataframes(
        dataframe,
        split_column='group',
        proportions=dict(
            train=0.8,
            compare=0.2,
        ),
    )
    group_overlap = (
        set(split_dataframes_['train'].group)
        .intersection(split_dataframes_['compare'].group)
    )
    assert len(group_overlap) == 0
    assert tuple(map(len, split_dataframes_.values())) == (80, 20)


def test_validate_proportions():
    from pytest import raises

    split_file = Path('test_validate_proportions.json')
    with raises(ValueError):
        split_dataframes(
            mock_dataframe(),
            key_column='index',
            proportions=dict(train=0.4, test=0.4),
            filepath=split_file,
            stratify_column='stratify',
        )


def test_missing_key_column():
    from pytest import raises

    split_file = Path('test_missing_key_column.json')
    with raises(KeyError):
        split_dataframes(
            mock_dataframe(),
            key_column='should_fail',
            proportions=dict(train=0.8, test=0.2),
            filepath=split_file,
        )

    with raises(FileNotFoundError):
        split_file.unlink()


def test_missing_stratify_column():
    from pytest import raises

    with raises(KeyError):
        split_dataframes(
            mock_dataframe(),
            key_column='index',
            proportions=dict(train=0.8, test=0.2),
            stratify_column='should_fail'
        )


def test_no_split():
    '''we do not need to support this'''
    split_dataframes(
        mock_dataframe(),
        key_column='index',
        proportions=dict(all=1.0),
        stratify_column='stratify'
    )


def test_split_empty():
    split_dataframes_ = split_dataframes(
        mock_dataframe().iloc[:0],
        key_column='index',
        proportions=dict(train=0.8, test=0.2),
        stratify_column='stratify'
    )
    for df in split_dataframes_.values():
        assert len(df) == 0


def test_split_single_row():
    split_dataframes_ = split_dataframes(
        mock_dataframe().iloc[:1],
        key_column='index',
        proportions=dict(train=0.9999, test=0.0001),
        stratify_column='stratify'
    )
    assert len(split_dataframes_['train']) == 1
    assert len(split_dataframes_['test']) == 0


def test_changed_split_names():
    from pytest import raises

    split_file = Path('test_changed_split_names.json')
    split_dataframes(
        mock_dataframe(),
        key_column='index',
        proportions=dict(train=0.8, test=0.2),
        filepath=split_file,
        stratify_column='stratify',
    )

    with raises(ValueError):
        split_dataframes(
            mock_dataframe(),
            key_column='index',
            proportions=dict(should_fail=0.8, test=0.2),
            filepath=split_file,
            stratify_column='stratify',
        )
    split_file.unlink()


def test_frozen():
    from pytest import raises

    dataframe = mock_dataframe()

    with raises(ValueError):
        split_dataframes(
            dataframe,
            key_column='index',
            proportions=dict(train=0.8, test=0.2),
            frozen=True,
        )

    split_file = Path('test_frozen.json')
    split_dataframes(
        dataframe,
        key_column='index',
        proportions=dict(train=0.8, test=0.2),
        filepath=split_file,
        stratify_column='stratify',
    )
    split_file.unlink()
