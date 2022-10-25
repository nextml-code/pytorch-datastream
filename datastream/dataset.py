from __future__ import annotations
from pydantic import BaseModel
from typing import (
    Tuple,
    Callable,
    Union,
    List,
    TypeVar,
    Generic,
    Dict,
    Optional,
    Iterable,
)
from pathlib import Path
from functools import lru_cache
import string
import random
import textwrap
import inspect
import numpy as np
import pandas as pd
from datastream import tools


T = TypeVar("T")
R = TypeVar("R")


class Dataset(BaseModel, Generic[T]):
    """
    A ``Dataset[T]`` is a mapping that allows pipelining of functions in a
    readable syntax returning an example of type ``T``.

        >>> from datastream import Dataset
        >>> fruit_and_cost = (
        ...     ('apple', 5),
        ...     ('pear', 7),
        ...     ('banana', 14),
        ...     ('kiwi', 100),
        ... )
        >>> dataset = (
        ...     Dataset.from_subscriptable(fruit_and_cost)
        ...     .starmap(lambda fruit, cost: (
        ...         fruit,
        ...         cost * 2,
        ...     ))
        ... )
        >>> dataset[2]
        ('banana', 28)
    """

    dataframe: Optional[pd.DataFrame]
    length: int
    get_item: Callable[[pd.DataFrame, int], T]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @staticmethod
    def from_subscriptable(subscriptable) -> Dataset:
        """
        Create ``Dataset`` based on subscriptable i.e. implements
        ``__getitem__`` and ``__len__``.

        Should only be used for simple examples as a ``Dataset`` created with
        this method does not support methods that require a source dataframe
        like :func:`Dataset.split` and :func:`Dataset.subset`.
        """

        return Dataset.from_dataframe(
            pd.DataFrame(dict(index=range(len(subscriptable))))
        ).map(lambda row: subscriptable[row["index"]])

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame) -> Dataset[pd.Series]:
        """
        Create ``Dataset`` based on ``pandas.DataFrame``.
        :func:`Dataset.__getitem__` will return a row from the dataframe and
        :func:`Dataset.map` should be given a function that takes a row from
        the dataframe as input.

        >>> (
        ...     Dataset.from_dataframe(pd.DataFrame(dict(
        ...        number=[1, 2, 3]
        ...     )))
        ...     .map(lambda row: row['number'] + 1)
        ... )[-1]
        4
        """
        return Dataset(
            dataframe=dataframe,
            length=len(dataframe),
            get_item=lambda df, index: df.iloc[index],
        )

    @staticmethod
    def from_paths(paths: Iterable[str, Path], pattern: str) -> Dataset[pd.Series]:
        """
        Create ``Dataset`` from paths using regex pattern that extracts information
        from the path itself.
        :func:`Dataset.__getitem__` will return a row from the dataframe and
        :func:`Dataset.map` should be given a function that takes a row from
        the dataframe as input.

        >>> image_paths = ["dataset/damage/1.png"]
        >>> (
        ...     Dataset.from_paths(image_paths, pattern=r".*/(?P<class_name>\w+)/(?P<index>\d+).png")
        ...     .map(lambda row: row["class_name"])
        ... )[-1]
        'damage'
        """
        paths = list(paths)
        return Dataset.from_dataframe(
            pd.Series(paths).astype(str).str.extract(pattern).assign(path=paths)
        )

    def __getitem__(
        self: Dataset[T],
        select: Union[int, slice, Iterable, Callable[[pd.DataFrame], Iterable[int]]],
    ) -> Union[T, Dataset[T]]:
        """Get selection from the ``Dataset[T]``"""
        if np.issubdtype(type(select), np.integer):
            return self.get_item(self.dataframe, select)
        else:
            dataframe = self.dataframe.iloc[select]
            return self.replace(dataframe=dataframe, length=len(dataframe))

    def __len__(self):
        return self.length

    def __str__(self):
        return str(
            "\n".join(
                [str(self[index]) for index in range(min(3, len(self)))]
                + (["..."] if len(self) > 3 else [])
            )
        )

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

    def replace(self, **kwargs):
        new_dict = self.dict()
        new_dict.update(**kwargs)
        return type(self)(**new_dict)

    def map(self: Dataset[T], function: Callable[[T], R]) -> Dataset[R]:
        """
        Creates a new dataset with the function added to the dataset pipeline.

        >>> (
        ...     Dataset.from_subscriptable([1, 2, 3])
        ...     .map(lambda number: number + 1)
        ... )[-1]
        4
        """

        def composed_fn(dataframe, index):
            item = self.get_item(dataframe, index)
            try:
                return function(item)
            except Exception as e:
                item_text = textwrap.shorten(str(item), width=79)

                raise Exception(
                    "\n".join(
                        [
                            repr(e),
                            "",
                            "Above exception originated from",
                            f"module: {inspect.getmodule(function)}",
                            "from mapped function:",
                            inspect.getsource(function),
                            "for item:",
                            item_text,
                        ]
                    )
                ).with_traceback(e.__traceback__)

        return Dataset(
            dataframe=self.dataframe,
            length=self.length,
            get_item=composed_fn,
        )

    def starmap(self: Dataset[T], function: Callable[Union[..., R]]) -> Dataset[R]:
        """
        Creates a new dataset with the function added to the dataset pipeline.
        The dataset's pipeline should return an iterable that will be
        expanded as \\*args to the mapped function.

        >>> (
        ...     Dataset.from_subscriptable([1, 2, 3])
        ...     .map(lambda number: (number, number + 1))
        ...     .starmap(lambda number, plus_one: number + plus_one)
        ... )[-1]
        7
        """
        return self.map(tools.star(function))

    def subset(
        self, mask_fn: Callable[[pd.DataFrame], Union[pd.Series, np.array, List[bool]]]
    ) -> Dataset[T]:
        """
        Select a subset of the dataset using a function that receives the
        source dataframe as input and is expected to return a boolean mask.

        Note that this function can still be called after multiple operations
        such as mapping functions as it uses the source dataframe.

        >>> (
        ...     Dataset.from_dataframe(pd.DataFrame(dict(
        ...        number=[1, 2, 3]
        ...     )))
        ...     .map(lambda row: row['number'])
        ...     .subset(lambda dataframe: dataframe['number'] <= 2)
        ... )[-1]
        2
        """
        dataframe = self.dataframe[mask_fn(self.dataframe)]
        return self.replace(dataframe=dataframe, length=len(dataframe))

    def split(
        self,
        key_column: str,
        proportions: Dict[str, float],
        stratify_column: Optional[str] = None,
        filepath: Optional[Union[str, Path]] = None,
        frozen: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Dataset[T]]:
        """
        Split dataset into multiple parts. Optionally you can chose to stratify
        on a column in the source dataframe or save the split to a json file.
        If you are sure that the split strategy will not change then you can
        safely use a seed instead of a filepath.

        Saved splits can continue from the old split and handles:

        * New examples
        * Changing test size
        * Adapt after removing examples from dataset
        * Adapt to new stratification

        >>> split_datasets = (
        ...     Dataset.from_dataframe(pd.DataFrame(dict(
        ...         index=np.arange(100),
        ...         number=np.arange(100),
        ...     )))
        ...     .map(lambda row: row['number'])
        ...     .split(
        ...         key_column='index',
        ...         proportions=dict(train=0.8, test=0.2),
        ...         seed=700,
        ...     )
        ... )
        >>> len(split_datasets['train'])
        80
        >>> split_datasets['test'][0]
        3
        """
        if filepath is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        if stratify_column is not None:
            return tools.stratified_split(
                self,
                key_column=key_column,
                proportions=proportions,
                stratify_column=stratify_column,
                filepath=filepath,
                seed=seed,
                frozen=frozen,
            )
        else:
            return tools.unstratified_split(
                self,
                key_column=key_column,
                proportions=proportions,
                filepath=filepath,
                seed=seed,
                frozen=frozen,
            )

    def with_columns(
        self: Dataset[T], **kwargs: Callable[pd.Dataframe, pd.Series]
    ) -> Dataset[T]:
        """
        Append new column(s) to the :attr:`.Dataset.dataframe` by passing the
        new column names as keywords with functions that take the
        :attr:`.Dataset.dataframe` as input and return :func:`pandas.Series`.

        >>> (
        ...     Dataset.from_dataframe(pd.DataFrame(dict(number=[1, 2, 3])))
        ...     .with_columns(twice=lambda df: df['number'] * 2)
        ...     .map(lambda row: row['twice'])
        ... )[-1]
        6
        """
        if len(set(kwargs.keys()) & set(self.dataframe.columns)) >= 1:
            raise ValueError("Should not replace existing columns")

        dataframe = self.dataframe.assign(**kwargs)
        return Dataset(
            dataframe=dataframe,
            length=len(dataframe),
            get_item=self.get_item,
        )

    def zip_index(self: Dataset[T]) -> Dataset[Tuple[T, int]]:
        """
        Zip the output with its index. The output of the pipeline will be
        a tuple ``(output, index)``.

        >>> (
        ...     Dataset.from_subscriptable([4, 5, 6])
        ...     .zip_index()
        ... )[0]
        (4, 0)
        """
        return Dataset(
            dataframe=self.dataframe,
            length=self.length,
            get_item=lambda dataframe, index: (
                self.get_item(dataframe, index),
                index,
            ),
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
        """
        Concatenate multiple datasets together so that they behave like a
        single dataset.

        Consider using :func:`Datastream.merge` if you have
        multiple data sources instead as it allows you to control the number
        of samples from each source in the training batches.
        """
        from_concat_mapping = Dataset.create_from_concat_mapping(datasets)

        if any([dataset.dataframe is None for dataset in datasets]):

            def get_item(dataframe, index):
                dataset_index, inner_index = from_concat_mapping(index)
                return datasets[dataset_index][inner_index]

            return Dataset(
                dataframe=None,
                length=sum(map(len, datasets)),
                get_item=get_item,
            )
        else:
            dataset_column = "__concat__" + "".join(
                [random.choice(string.ascii_lowercase) for _ in range(8)]
            )

            dataframes = [dataset.dataframe for dataset in datasets]
            for dataframe in dataframes:
                for col in dataframe.columns:
                    if dataframe[col].dtype == int and any(
                        [col not in other.columns for other in dataframes]
                    ):
                        dataframe[col] = dataframe[col].astype(object)

            new_dataframe = pd.concat(dataframes)
            new_dataframe[dataset_column] = [
                from_concat_mapping(index)[0] for index in range(len(new_dataframe))
            ]

            def get_item(dataframe, index):
                dataset_index = int(dataframe.iloc[index][dataset_column])
                return datasets[dataset_index].get_item(dataframe, index)

            return Dataset(
                dataframe=new_dataframe,
                length=sum(map(len, datasets)),
                get_item=get_item,
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
                [
                    inner_index * cumprod_lengths[i]
                    for i, inner_index in enumerate(inner_indices[1:])
                ]
            )

        return to_concat

    @staticmethod
    def combine(datasets: List[Dataset]) -> Dataset[Tuple]:
        """
        Zip multiple datasets together so that all combinations of examples
        are possible (i.e. the product) creating tuples like
        ``(example1, example2, ...)``.

        The created dataset will not have a dataframe because combined
        datasets are often very long and it is expensive to enumerate them.
        """
        from_combine_mapping = Dataset.create_from_combine_mapping(datasets)

        def get_item(dataframe, index):
            indices = from_combine_mapping(index)
            return tuple([dataset[index] for dataset, index in zip(datasets, indices)])

        return Dataset(
            dataframe=None,
            length=np.prod(list(map(len, datasets))),
            get_item=get_item,
        )

    @staticmethod
    def zip(datasets: List[Dataset]) -> Dataset[Tuple]:
        """
        Zip multiple datasets together so that examples with matching indices
        create tuples like ``(example1, example2, ...)``.

        The length of the created dataset is the minimum length of the zipped
        datasets.

        The created dataset's dataframe is a the concatenation of the input
        datasets' dataframes. It is concatenated over columns with an added
        multiindex column like this:
        ``pd.concat(dataframes, axis=1, keys=['dataset0', 'dataset1', ...])``

        >>> Dataset.zip([
        ...     Dataset.from_subscriptable([1, 2, 3]),
        ...     Dataset.from_subscriptable([4, 5, 6, 7]),
        ... ])[-1]
        (3, 6)
        """
        length = min(map(len, datasets))
        return (
            Dataset.from_dataframe(
                pd.concat(
                    [
                        dataset.dataframe.iloc[:length].reset_index()
                        for dataset in datasets
                    ],
                    axis=1,
                    keys=[
                        f"dataset{dataset_index}"
                        for dataset_index in range(len(datasets))
                    ],
                ).assign(_index=list(range(length)))
            )
            .map(lambda row: row["_index"].iloc[0])
            .map(lambda index: tuple(dataset[index] for dataset in datasets))
        )

    def cache(
        self,
        key_column: str,
    ):
        """Cache intermediate step in-memory based on key column."""
        key_mapping = dict(
            zip(
                self.dataframe[key_column],
                range(len(self)),
            )
        )

        @lru_cache(maxsize=None)
        def only_key(key):
            return self.get_item(self.dataframe, key_mapping[key])

        return Dataset(
            dataframe=self.dataframe,
            length=self.length,
            get_item=lambda dataframe, index: only_key(
                dataframe.iloc[index][key_column]
            ),
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
        Dataset.from_dataframe(number_df).map(lambda row: row["number"]),
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
    dataset = Dataset.from_dataframe(dataframe).subset(lambda df: df["number"] >= 12)
    assert dataset[0]["number"] == numbers[2]


def test_with_columns():
    from pytest import raises

    with raises(ValueError):
        dataset = Dataset.from_dataframe(
            pd.DataFrame(
                dict(
                    key=np.arange(100),
                )
            )
        ).with_columns(key=lambda df: df["key"] * 2)


def test_concat_dataset():
    dataset = Dataset.concat(
        [
            Dataset.from_subscriptable(list(range(5))),
            Dataset.from_subscriptable(list(range(4))),
        ]
    )

    assert dataset[6] == 1


def test_concat_heterogenous_datasets():
    dataset1 = Dataset.from_dataframe(
        pd.DataFrame(dict(a=[1], b=["a"])).set_index("a"),
    )
    dataset2 = Dataset.from_dataframe(
        pd.DataFrame(dict(a=[1], b=[1], c=[2])).set_index("a"),
    )
    dataset = Dataset.concat([dataset1, dataset2]).map(lambda row: row["b"])

    assert list(dataset) == ["a", 1]

    dataset_other_functions = Dataset.concat(
        [
            dataset1.map(lambda row: row["b"]),
            dataset2.map(lambda row: row["c"]),
        ]
    )

    assert list(dataset_other_functions) == ["a", 2]


def test_zip_dataset():
    dataset = Dataset.zip(
        [
            Dataset.from_subscriptable(list(range(5))),
            Dataset.from_subscriptable(list(range(4))),
        ]
    )

    assert dataset[3] == (3, 3)

    for x, y in zip(
        dataset.subset(lambda df: np.arange(len(df)) <= 2),
        dataset,
    ):
        assert x == y


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
    dataset = Dataset.from_dataframe(
        pd.DataFrame(
            dict(
                index=np.arange(100),
                number=np.random.randn(100),
                stratify=np.concatenate([np.ones(50), np.zeros(50)]),
            )
        )
    ).map(tuple)

    filepath = Path("test_split_dataset.json")
    proportions = dict(
        gradient=0.7,
        early_stopping=0.15,
        compare=0.15,
    )

    kwargs = dict(
        key_column="index",
        proportions=proportions,
        filepath=filepath,
        stratify_column="stratify",
    )

    split_datasets1 = dataset.split(**kwargs)
    split_datasets2 = dataset.split(**kwargs)
    split_datasets3 = dataset.split(
        key_column="index",
        proportions=proportions,
        stratify_column="stratify",
        seed=100,
    )
    split_datasets4 = dataset.split(
        key_column="index",
        proportions=proportions,
        stratify_column="stratify",
        seed=100,
    )
    split_datasets5 = dataset.split(
        key_column="index",
        proportions=proportions,
        stratify_column="stratify",
        seed=800,
    )
    filepath.unlink()

    assert split_datasets1 == split_datasets2
    assert split_datasets1 != split_datasets3
    assert split_datasets3 == split_datasets4
    assert split_datasets3 != split_datasets5


def test_group_split_dataset():
    dataset = Dataset.from_dataframe(
        pd.DataFrame(
            dict(
                group=np.arange(100) // 4,
                number=np.random.randn(100),
            )
        )
    ).map(tuple)

    filepath = Path("test_split_dataset.json")
    proportions = dict(
        gradient=0.7,
        early_stopping=0.15,
        compare=0.15,
    )

    kwargs = dict(
        key_column="group",
        proportions=proportions,
        filepath=filepath,
    )

    split_datasets1 = dataset.split(**kwargs)
    split_datasets2 = dataset.split(**kwargs)
    split_datasets3 = dataset.split(
        key_column="group",
        proportions=proportions,
        seed=100,
    )
    split_datasets4 = dataset.split(
        key_column="group",
        proportions=proportions,
        seed=100,
    )
    split_datasets5 = dataset.split(
        key_column="group",
        proportions=proportions,
        seed=800,
    )

    filepath.unlink()

    assert split_datasets1 == split_datasets2
    assert split_datasets1 != split_datasets3
    assert split_datasets3 == split_datasets4
    assert split_datasets3 != split_datasets5


def test_missing_stratify_column():
    from pytest import raises

    dataset = Dataset.from_dataframe(
        pd.DataFrame(
            dict(
                index=np.arange(100),
                number=np.random.randn(100),
            )
        )
    ).map(tuple)

    with raises(KeyError):
        dataset.split(
            key_column="index",
            proportions=dict(train=0.8, test=0.2),
            stratify_column="should_fail",
        )


def test_split_proportions():
    dataset = Dataset.from_dataframe(
        pd.DataFrame(
            dict(
                index=np.arange(100),
                number=np.random.randn(100),
                stratify=np.arange(100) // 10,
            )
        )
    ).map(tuple)

    splits = dataset.split(
        key_column="index",
        proportions=dict(train=0.8, test=0.2),
        stratify_column="stratify",
    )

    assert len(splits["train"]) == 80
    assert len(splits["test"]) == 20


def test_with_columns_split():
    dataset = (
        Dataset.from_dataframe(
            pd.DataFrame(
                dict(
                    index=np.arange(100),
                    number=np.arange(100),
                )
            )
        )
        .map(tuple)
        .with_columns(split=lambda df: df["index"] * 2)
    )

    splits = dataset.split(
        key_column="index",
        proportions=dict(train=0.8, test=0.2),
    )

    assert splits["train"][0][0] * 2 == splits["train"][0][2]


def test_split_filepath():

    dataset = Dataset.from_dataframe(
        pd.DataFrame(
            dict(
                index=np.arange(100),
                number=np.random.randn(100),
                stratify=np.arange(100) // 10,
            )
        )
    ).map(tuple)

    filepath = Path("tmp_test_split.json")
    splits1 = dataset.split(
        key_column="index",
        proportions=dict(train=0.8, test=0.2),
        filepath=filepath,
    )

    splits2 = dataset.split(
        key_column="index",
        proportions=dict(train=0.8, test=0.2),
        filepath=filepath,
    )

    assert splits1["train"][0] == splits2["train"][0]
    assert splits1["test"][0] == splits2["test"][0]

    filepath.unlink()


def test_update_stratified_split():

    for _ in range(5):

        dataset = Dataset.from_dataframe(
            pd.DataFrame(
                dict(
                    index=np.arange(100),
                    number=np.random.randn(100),
                    stratify1=np.random.randint(0, 10, 100),
                    stratify2=np.random.randint(0, 10, 100),
                )
            )
        ).map(tuple)

        filepath = Path("tmp_test_split.json")

        splits1 = dataset.subset(lambda df: df["index"] < 50).split(
            key_column="index",
            proportions=dict(train=0.8, test=0.2),
            filepath=filepath,
            stratify_column="stratify1",
        )

        splits2 = dataset.split(
            key_column="index",
            proportions=dict(train=0.8, test=0.2),
            filepath=filepath,
            stratify_column="stratify2",
        )

        assert (
            splits1["train"]
            .dataframe["index"]
            .isin(splits2["train"].dataframe["index"])
            .all()
        )

        assert (
            splits1["test"]
            .dataframe["index"]
            .isin(splits2["test"].dataframe["index"])
            .all()
        )

        filepath.unlink()


def test_concat_missing_columns():
    dataset1 = Dataset.from_dataframe(
        pd.DataFrame(dict(a=[1, 2, 3], b=["a", "b", "c"]))
    )
    dataset2 = Dataset.from_dataframe(
        pd.DataFrame(dict(c=[True, False], d=[[1, 2], [3, 4]]))
    )
    concatenated = Dataset.concat([dataset1, dataset2])

    assert type(concatenated[0]["a"]) == int
    assert type(concatenated[-1]["a"]) == float
    assert type(concatenated[0]["b"]) == str
    assert type(concatenated[-1]["c"]) == bool
    assert type(concatenated[-1]["d"]) == list
