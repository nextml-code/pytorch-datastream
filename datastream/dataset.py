from __future__ import annotations
from pydantic import BaseModel
from typing import (
    Tuple, Callable, Any, Union, List, TypeVar, Generic, Dict, Optional
)
from pathlib import Path
from functools import partial
from itertools import repeat, chain
import numpy as np
import pandas as pd
import torch
from datastream import tools


T = TypeVar('T')
R = TypeVar('R')

class Dataset(BaseModel, torch.utils.data.Dataset, Generic[T]):
    '''
    A ``Dataset[T]`` is a mapping that allows pipelining of functions in a 
    readable syntax returning an item of type ``T``.

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

    dataframe: pd.DataFrame
    length: int
    functions: Tuple[Callable[..., Any], ...]
    composed_fn: Callable[[pd.DataFrame, int], T]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(
        self,
        dataframe: pd.DataFrame,
        length: int,
        functions: Tuple[Callable[..., Any], ...],
    ):
        BaseModel.__init__(
            self,
            dataframe=dataframe,
            length=length,
            functions=functions,
            composed_fn=tools.starcompose(*functions),
        )

    @staticmethod
    def from_subscriptable(subscriptable) -> Dataset:
        '''
        Create ``Dataset`` based on subscriptable i.e. implements
        ``__getitem__`` and ``__len__``. Should only be used for simple
        examples as a ``Dataset`` created with this method does not support
        methods that require a source dataframe (i.e. :func:``Dataset.split`` 
        and :func:``Dataset.subset``)
        '''

        return (
            Dataset.from_dataframe(
                pd.DataFrame(dict(index=range(len(subscriptable))))
            )
            .map(lambda row: subscriptable[row['index']])
        )

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame) -> Dataset[pd.Series]:
        '''Create ``Dataset`` based on ``pandas.DataFrame``.'''
        return Dataset(
            dataframe=dataframe,
            length=len(dataframe),
            functions=tuple([lambda df, index: df.iloc[index]]),
        )

    def __getitem__(self: Dataset[T], index: int) -> T:
        return self.composed_fn(self.dataframe, index)

    def __len__(self):
        return self.length

    def __str__(self):
        return str('\n'.join(
            [str(self[index]) for index in range(min(3, len(self)))]
            + (['...'] if len(self) > 3 else [])
        ))

    def __repr__(self):
        return str(self)

    def __add__(self: Dataset[T], other: Dataset[R]) -> Dataset[Union[T, R]]:
        return Dataset.concat([self, other])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __eq__(self: Dataset[T], other: Dataset[R]) -> bool:
        for item1, item2 in zip(self, other):
            if item1 != item2:
                return False
        return True

    def map(
        self: Dataset[T], function: Callable[Union[[T], [...]], R]
    ) -> Dataset[R]:
        '''
        Creates a new dataset with the function added to the dataset pipeline.
        Returned tuples are expanded as \\*args for the next mapped function.

        >>> (
        ...     Dataset.from_subscriptable([1, 2, 3])
        ...     .map(lambda number: (number, number + 1))
        ...     .map(lambda number, plus_one: number + plus_one)
        ... )[-1]
        7
        '''
        return Dataset(
            dataframe=self.dataframe,
            length=self.length,
            functions=self.functions + tuple([function]),
        )

    def subset(
        self, mask_fn: Callable[
            [pd.DataFrame], Union[pd.Series, np.array, List[bool]]
        ]  
    ) -> Dataset[T]:
        '''
        Select a subset of the dataset using a function that receives the
        source dataframe as input and is expected to return a boolean mask.

        >>> (
        ...     Dataset.from_dataframe(pd.DataFrame(dict(
        ...        number=[1, 2, 3]
        ...     )))
        ...     .subset(lambda df: df['number'] <= 2)
        ...     .map(lambda row: row['number'])
        ... )[-1]
        2
        '''

        mask = mask_fn(self.dataframe)
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
            dataframe=self.dataframe.iloc[indices],
            length=len(indices),
            functions=self.functions,
        )

    def split(
        self,
        key_column: str,
        proportions: Dict[str, float],
        stratify_column: Optional[str] = None,
        filepath: Optional[Union[str, Path]] = None,
        frozen: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Dataset[T]]:
        '''
        Split dataset into multiple parts. Optionally you can chose to stratify
        on a column in the source dataframe or save the split to a json file.
        If you are sure that the split strategy will not change then you can
        safely use a seed instead of a filepath.

        Saved splits can continue from the old split and handles:
        
        * New examples
        * Changing test size
        * Adapt after removing examples from dataset
        * Adapt to new stratification

        >>> split_file = Path('doctest_split_dataset.json')
        >>> split_datasets = (
        ...     Dataset.from_dataframe(pd.DataFrame(dict(
        ...         index=np.arange(100),
        ...         number=np.random.randn(100),
        ...     )))
        ...     .split(
        ...         key_column='index',
        ...         proportions=dict(train=0.8, test=0.2),
        ...         filepath=split_file,
        ...     )
        ... )
        >>> len(split_datasets['train'])
        80
        >>> split_file.unlink()  # clean up after doctest
        '''
        if filepath is not None:
            filepath = Path(filepath)

        if seed is None:
            split_dataframes = tools.split_dataframes
        else:
            split_dataframes = tools.numpy_seed(seed)(tools.split_dataframes)

        return {
            split_name: Dataset(
                dataframe=dataframe,
                length=len(dataframe),
                functions=self.functions,
            )
            for split_name, dataframe in split_dataframes(
                self.dataframe,
                key_column,
                proportions,
                stratify_column,
                filepath,
                frozen,
            ).items()
        }

    def zip_index(self: Dataset[T]) -> Dataset[Tuple[T, int]]:
        '''
        Zip the output with its index. The output of the pipeline will be
        a tuple ``(output, index)``.
        '''
        composed_fn = self.composed_fn
        return Dataset(
            dataframe=self.dataframe,
            length=self.length,
            functions=tuple([lambda dataframe, index: (
                composed_fn(dataframe, index),
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
    def concat(datasets: List[Dataset]) -> Dataset[R]:
        '''
        Concatenate multiple datasets together so that they behave like a
        single dataset. Consider using :func:`Datastream.merge` if you have
        multiple data sources.
        '''
        from_concat_mapping = Dataset.create_from_concat_mapping(datasets)

        return Dataset(
            dataframe=pd.DataFrame(dict(dataset_index=range(len(datasets)))),
            length=sum(map(len, datasets)),
            functions=(
                lambda index_dataframe, index: from_concat_mapping(index),
                lambda dataset_index, inner_index: (
                    datasets[dataset_index][inner_index]
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
    def combine(datasets: List[Dataset]) -> Dataset[Tuple]:
        '''
        Zip multiple datasets together so that all combinations of examples
        are possible (i.e. the product) creating tuples like
        ``(example1, example2, ...)``.
        '''
        from_combine_mapping = Dataset.create_from_combine_mapping(datasets)

        return Dataset(
            dataframe=pd.DataFrame(dict(dataset_index=range(len(datasets)))),
            length=np.prod(list(map(len, datasets))),
            functions=(
                lambda index_dataframe, index: from_combine_mapping(index),
                lambda *indices: tuple([
                    dataset[index] for dataset, index in zip(datasets, indices)
                ]),
            ),
        )

    @staticmethod
    def zip(datasets: List[Dataset]) -> Dataset[Tuple]:
        '''
        Zip multiple datasets together so that examples with matching indices
        create tuples like ``(example1, example2, ...)``.

        The length of the created dataset is the minimum length of the zipped
        datasets.
        '''
        return Dataset(
            dataframe=pd.DataFrame(dict(dataset_index=range(len(datasets)))),
            length=min(map(len, datasets)),
            functions=tuple([lambda index_dataframe, index: tuple(
                dataset[index] for dataset in datasets
            )]),
        )


def test_equal():
    dataset1 = Dataset.from_subscriptable([4, 7, 12])
    assert dataset1 == dataset1

    dataset2 = Dataset.from_subscriptable([4, 7, 13])
    assert dataset1 != dataset2


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
    dataset = Dataset.from_subscriptable(numbers).subset(
        lambda numbers_df: [False, True, True]
    )

    assert dataset[0] == numbers[1]

    dataframe = pd.DataFrame(dict(number=numbers))
    dataset = (
        Dataset.from_dataframe(dataframe)
        .subset(lambda df: df['number'] >= 12)
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
   

def test_split_dataset():
    dataset = Dataset.from_dataframe(pd.DataFrame(dict(
        index=np.arange(100),
        number=np.random.randn(100),
        stratify=np.concatenate([np.ones(50), np.zeros(50)]),
    ))).map(tuple)

    split_file = Path('test_split_dataset.json')
    proportions = dict(
        gradient=0.7,
        early_stopping=0.15,
        compare=0.15,
    )

    kwargs = dict(
        key_column='index',
        proportions=proportions,
        filepath=split_file,
        stratify_column='stratify',
    )

    split_datasets1 = dataset.split(**kwargs)
    split_datasets2 = dataset.split(**kwargs)
    split_datasets3 = dataset.split(
        key_column='index',
        proportions=proportions,
        stratify_column='stratify',
        seed=100,
    )
    split_datasets4 = dataset.split(
        key_column='index',
        proportions=proportions,
        stratify_column='stratify',
        seed=100,
    )
    split_datasets5 = dataset.split(
        key_column='index',
        proportions=proportions,
        stratify_column='stratify',
        seed=800,
    )

    split_file.unlink()

    assert split_datasets1 == split_datasets2
    assert split_datasets1 != split_datasets3
    assert split_datasets3 == split_datasets4
    assert split_datasets3 != split_datasets5
