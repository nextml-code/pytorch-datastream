from functools import partial
from itertools import repeat, chain
import numpy as np
import pandas as pd
import torch
from datastream import starcompose, star


class Dataset(torch.utils.data.Dataset):
    def __init__(self, source, length, function_list):
        super().__init__()
        self.source = source
        self.length = length
        self.function_list = function_list
        self.composed_fn = starcompose(*function_list)

    @staticmethod
    def from_subscriptable(subscriptable):
        return Dataset(
            subscriptable,
            len(subscriptable),
            [lambda ds, index: ds[index]],
        )

    @staticmethod
    def from_dataframe(dataframe):
        return Dataset(
            dataframe,
            len(dataframe),
            [lambda df, index: df.iloc[index]],
        )

    def __getitem__(self, index):
        return self.composed_fn(self.source, index)

    def __len__(self):
        return self.length

    def __str__(self):
        return str('\n'.join(
            [str(self[index]) for index in range(min(3, len(self)))]
            + ['...'] if len(self) > 3 else []
        ))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Dataset.concat([self, other])

    def map(self, function):
        return Dataset(
            self.source,
            self.length,
            self.function_list + [function],
        )

    def zip_index(self):
        composed_fn = self.composed_fn
        return Dataset(
            self.source,
            self.length,
            [lambda source, index: (
                composed_fn(source, index),
                index,
            )],
        )

    def subset(self, indices):
        if type(indices) is pd.Series:
            indices = np.argwhere(indices.values).squeeze(1)

        if type(self.source) is pd.DataFrame:
            return Dataset(
                self.source.iloc[indices],
                len(indices),
                self.function_list,
            )
        else:
            return Dataset(
                indices,
                len(indices),
                [lambda indices, outer_index: (
                    self.source, indices[outer_index]
                )] + self.function_list,
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
            datasets,
            sum(map(len, datasets)),
            [
                lambda datasets, index: (
                    datasets,
                    *from_concat_mapping(index),
                ),
                lambda datasets, dataset_index, inner_index: (
                    datasets[dataset_index][inner_index]
                ),
            ],
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
            datasets,
            np.prod(list(map(len, datasets))),
            [
                lambda datasets, index: (
                    datasets,
                    from_combine_mapping(index),
                ),
                lambda datasets, indices: tuple([
                    dataset[index] for dataset, index in zip(datasets, indices)
                ]),
            ]
        )

    @staticmethod
    def zip(datasets):
        return Dataset(
            datasets,
            min(map(len, datasets)),
            [lambda datasets, index: tuple(
                dataset[index] for dataset in datasets
            )],
        )


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
