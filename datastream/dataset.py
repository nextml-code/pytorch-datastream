from pydantic import BaseModel
from typing import Tuple, Callable, Any
from functools import partial
from itertools import repeat, chain
import numpy as np
import pandas as pd
import torch
from datastream import starcompose, star


class Dataset(BaseModel, torch.utils.data.Dataset):
    source: pd.DataFrame
    length: int
    functions: Tuple[Callable[..., Any], ...]
    composed_fn: Callable[..., Any]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(
        self,
        source: pd.DataFrame,
        length: int,
        functions: Tuple[Callable[..., Any], ...],
    ):
        BaseModel.__init__(
            self,
            source=source,
            length=length,
            functions=functions,
            composed_fn=starcompose(*functions),
        )

    @staticmethod
    def from_subscriptable(subscriptable):
        return (
            Dataset.from_dataframe(
                pd.DataFrame(dict(
                    example=subscriptable
                ))
            )
            .map(lambda row: row['example'])
        )

    @staticmethod
    def from_dataframe(dataframe):
        return Dataset(
            source=dataframe,
            length=len(dataframe),
            functions=tuple([lambda df, index: df.iloc[index]]),
        )

    def __getitem__(self, index):
        return self.composed_fn(self.source, index)

    def __len__(self):
        return self.length

    def __str__(self):
        return str('\n'.join(
            [str(self[index]) for index in range(min(3, len(self)))]
            + (['...'] if len(self) > 3 else [])
        ))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Dataset.concat([self, other])

    def map(self, function):
        return Dataset(
            source=self.source,
            length=self.length,
            functions=self.functions + tuple([function]),
        )

    def zip_index(self):
        composed_fn = self.composed_fn
        return Dataset(
            source=self.source,
            length=self.length,
            functions=tuple([lambda source, index: (
                composed_fn(source, index),
                index,
            )]),
        )

    def subset(self, indices):
        if type(indices) is pd.Series:
            indices = np.argwhere(indices.values).squeeze(1)

        return Dataset(
            source=self.source.iloc[indices],
            length=len(indices),
            functions=self.functions,
        )

    @staticmethod
    def create_from_concat_mapping(datasets):
        cumulative_lengths = np.cumsum(list(map(len, datasets)))

        def from_concat(index):
            dataset_index = np.sum(index >= cumulative_lengths)
            if dataset_index == 0:
                inner_index = index
            else:
                inner_index = index - cumulative_lengths[dataset_index - 1]

            return dataset_index, inner_index
        return from_concat

    @staticmethod
    def create_to_concat_mapping(datasets):
        cumulative_lengths = np.cumsum(list(map(len, datasets)))

        def to_concat(dataset_index, inner_index):
            if dataset_index == 0:
                index = inner_index
            else:
                index = inner_index + cumulative_lengths[dataset_index - 1]

            return index
        return to_concat

    @staticmethod
    def concat(datasets):
        from_concat_mapping = Dataset.create_from_concat_mapping(datasets)

        return Dataset(
            source=pd.DataFrame(dict(dataset=datasets)),
            length=sum(map(len, datasets)),
            functions=(
                lambda datasets, index: (
                    datasets,
                    *from_concat_mapping(index),
                ),
                lambda datasets, dataset_index, inner_index: (
                    datasets.iloc[dataset_index]['dataset'][inner_index]
                ),
            ),
        )

    @staticmethod
    def create_from_combine_mapping(datasets):
        dataset_lengths = list(map(len, datasets))
        cumprod_lengths = np.cumprod(dataset_lengths)

        def from_combine(index):
            return tuple(
                [index % cumprod_lengths[0]]
                + [
                    (index // cumprod_length) % dataset_length
                    for cumprod_length, dataset_length in zip(
                        cumprod_lengths[:-1], dataset_lengths[1:]
                    )
                ]
            )
        return from_combine

    @staticmethod
    def create_to_combine_mapping(datasets):
        cumprod_lengths = np.cumprod(list(map(len, datasets)))
        def to_concat(inner_indices):
            return inner_indices[0] + sum(
                [inner_index * cumprod_lengths[i]
                for i, inner_index in enumerate(inner_indices[1:])]
            )
        return to_concat

    @staticmethod
    def combine(datasets):
        from_combine_mapping = Dataset.create_from_combine_mapping(datasets)

        return Dataset(
            source=pd.DataFrame(dict(dataset=datasets)),
            length=np.prod(list(map(len, datasets))),
            functions=(
                lambda datasets, index: (
                    datasets['dataset'],
                    from_combine_mapping(index),
                ),
                lambda datasets, indices: tuple([
                    dataset[index] for dataset, index in zip(datasets, indices)
                ]),
            )
        )

    @staticmethod
    def zip(datasets):
        return Dataset(
            source=pd.DataFrame(dict(dataset=datasets)),
            length=min(map(len, datasets)),
            functions=tuple([lambda datasets, index: tuple(
                dataset[index] for dataset in datasets['dataset']
            )]),
        )


def test_subscript():
    number_list = [4, 7, 12]
    number_df = pd.DataFrame(dict(number=number_list))

    for dataset in (
        Dataset.from_subscriptable(number_list),
        Dataset.from_dataframe(number_df).map(lambda row: row['number'])
    ):

        if dataset[-1] != number_list[-1]:
            raise AssertionError('Unexpected result from dataset subscript')

        mapped_dataset = dataset.map(lambda number: number * 2)

        if mapped_dataset[-1] != number_list[-1] * 2:
            raise AssertionError(
                'Unexpected result from dataset subscript after map'
            )


def test_subset():
    dataset = Dataset.from_subscriptable([4, 7, 12]).subset([1, 2])

    if dataset[0] != 7:
        raise AssertionError('Unexpected result from subset dataset')


def test_concat_dataset():
    dataset = Dataset.concat([
        Dataset.from_subscriptable(list(range(5))),
        Dataset.from_subscriptable(list(range(4))),
    ])

    if dataset[6] != 1:
        raise AssertionError('Unexpected result from Dataset.concat')


def test_zip_dataset():
    dataset = Dataset.zip([
        Dataset.from_subscriptable(list(range(5))),
        Dataset.from_subscriptable(list(range(4))),
    ])

    if dataset[3] != (3, 3):
        raise AssertionError('Unexpected result from Dataset.zip')


def test_combine_dataset():
    from itertools import product

    datasets = [
        Dataset.from_subscriptable([1, 2]),
        Dataset.from_subscriptable([3, 4, 5]),
        Dataset.from_subscriptable([6, 7]),
    ]
    combined = Dataset.combine(datasets)
    to_combine_map = Dataset.create_to_combine_mapping(datasets)

    indices = list(product(*(range(len(ds)) for ds in datasets)))
    assert all(
        (
            combined[to_combine_map(inner_indices)]
            == tuple(ds[i] for ds, i in zip(datasets, inner_indices))
        )
        for index, inner_indices in enumerate(indices)
    )
