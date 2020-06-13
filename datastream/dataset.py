from __future__ import annotations
from pydantic import BaseModel
from typing import Tuple, Callable, Any, Union, List
from functools import partial
from itertools import repeat, chain
import numpy as np
import pandas as pd
import torch
from datastream import starcompose, star


class Dataset(BaseModel, torch.utils.data.Dataset):
    '''
    A ``Dataset`` is a mapping that allows pipelining of functions in a 
    readable syntax.

        >>> from datastream import Dataset
        >>> fruit_and_cost = (
        ...     ('apple', 5),
        ...     ('pear', 7),
        ...     ('banana', 14),
        ...     ('kiwi', 100),
        ... )
        >>> dataset = (
        ...     Dataset.from_subscriptable(fruit_and_cost)
        ...     .map(lambda fruit, cost: (
        ...         fruit,
        ...         cost * 2,
        ...     ))
        ... )
        >>> print(dataset[2])
        ('banana', 28)
    '''

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
    def from_subscriptable(subscriptable) -> Dataset:
        '''
        Create ``Dataset`` based on subscriptable i.e. implements
        ``__get_item__``. Mostly useful for simple examples.
        '''
        return (
            Dataset.from_dataframe(
                pd.DataFrame(dict(
                    example=subscriptable
                ))
            )
            .map(lambda row: row['example'])
        )

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame) -> Dataset:
        '''Create ``Dataset`` based on ``pandas.DataFrame``.'''
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

    def __add__(self, other: Dataset) -> Dataset:
        return Dataset.concat([self, other])

    def map(self, function: Callable) -> Dataset:
        '''Append a function to the dataset pipeline.'''
        return Dataset(
            source=self.source,
            length=self.length,
            functions=self.functions + tuple([function]),
        )

    def subset(
        self, mask: Union[pd.Series, np.array, List[bool]]
    ) -> Dataset:
        '''
        Select a subset of the dataset using a boolean mask.
        '''
        if isinstance(mask, list):
            mask = np.array(mask)
        elif isinstance(mask, pd.Series):
            mask = mask.values

        if len(mask.shape) != 1:
            raise AssertionError('Expected single dimension in mask')

        if len(mask) != len(self):
            raise AssertionError(
                'Expected mask to have the same length as the dataset'
            )

        indices = np.argwhere(mask).squeeze(1)
        return Dataset(
            source=self.source.iloc[indices],
            length=len(indices),
            functions=self.functions,
        )

    def zip_index(self) -> Dataset:
        '''
        Zip the output with its index. The output of the pipeline will be
        a tuple ``(output, index)``.
        '''
        composed_fn = self.composed_fn
        return Dataset(
            source=self.source,
            length=self.length,
            functions=tuple([lambda source, index: (
                composed_fn(source, index),
                index,
            )]),
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
    def concat(datasets: List[Dataset]) -> Dataset:
        '''
        Concatenate multiple datasets together so that they behave like a
        single dataset. Consider using ``Datastream.merge`` if you have multiple
        data sources instead.
        '''
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
    def combine(datasets: List[Dataset]) -> Dataset:
        '''
        Zip multiple datasets together so that all combinations of examples
        are possible creating tuples like ``(example1, example2, ...)``.
        '''
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
    def zip(datasets: List[Dataset]) -> Dataset:
        '''
        Zip multiple datasets together so that examples with matching indices
        create tuples like ``(example1, example2, ...)``.
        '''
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

        assert dataset[-1] == number_list[-1]

        mapped_dataset = dataset.map(lambda number: number * 2)

        assert mapped_dataset[-1] == number_list[-1] * 2


def test_subset():
    numbers = [4, 7, 12]
    dataset = Dataset.from_subscriptable(numbers).subset([False, True, True])

    assert dataset[0] == numbers[1]

    dataframe = pd.DataFrame(dict(number=numbers))
    dataset = (
        Dataset.from_dataframe(dataframe)
        .subset(dataframe['number'] >= 12)
    )
    assert dataset[0]['number'] == numbers[2]


def test_concat_dataset():
    dataset = Dataset.concat([
        Dataset.from_subscriptable(list(range(5))),
        Dataset.from_subscriptable(list(range(4))),
    ])

    assert dataset[6] == 1


def test_zip_dataset():
    dataset = Dataset.zip([
        Dataset.from_subscriptable(list(range(5))),
        Dataset.from_subscriptable(list(range(4))),
    ])

    assert dataset[3] == (3, 3)


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
