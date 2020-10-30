from __future__ import annotations
from pydantic import BaseModel, PositiveInt
from typing import (
    Tuple,
    Dict,
    List,
    Callable,
    Optional,
    TypeVar,
    Generic,
    Union,
)
import numpy as np
import torch
from pathlib import Path

from datastream import Dataset
from datastream.samplers import (
    StandardSampler,
    MergeSampler,
    ZipSampler,
    MultiSampler,
    RepeatSampler,
)


T = TypeVar('T')
R = TypeVar('R')


class Datastream(BaseModel, Generic[T]):
    '''
    ``Datastream[T]`` combines a ``Dataset[T]`` and a sampler into a stream of
    examples. By default the samples are drawn without replacement until the
    full dataset is exhausted. The proportion of the dataset that should be
    drawn before allowing replacement can be changed with
    :func:`Datastream.sample_proportion`.

    >>> data_loader = (
    ...     Datastream(Dataset.from_subscriptable([1, 2, 3]))
    ...     .data_loader(batch_size=16, n_batches_per_epoch=100)
    ... )
    >>> len(next(iter(data_loader)))
    16
    '''

    dataset: Dataset[T]
    sampler: Optional[torch.utils.data.Sampler]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(
        self,
        dataset: Dataset[T],
        sampler: torch.utils.data.Sampler = None
    ):
        if len(dataset) == 0:
            raise ValueError('Cannot create datastream from empty dataset')

        super().__init__(
            dataset=dataset,
            sampler=(
                StandardSampler(len(dataset))
                if sampler is None
                else sampler
            )
        )

    def __len__(self):
        return len(self.sampler)
    
    def __iter__(self):
        return map(self.dataset.__getitem__, iter(self.sampler))

    @staticmethod
    def merge(datastreams_and_ns: Tuple[Union[
        Datastream[T],
        Tuple[Datastream[T], int]
    ], ...]) -> Datastream[T]:
        '''
        Merge multiple datastreams by interleaving them. Optionally you can
        define different lengths per ``Datastream``.

        .. highlight:: python
        .. code-block:: python

            Datastream.merge([
                (datastream1, 2),
                (datastream2, 1),
                (datastream3, 1),
            ])
        '''
        datastreams_and_ns = [
            x if type(x) is tuple else (x, 1)
            for x in datastreams_and_ns
        ]

        return Datastream(
            Dataset.concat([
                datastream.dataset for datastream, n in datastreams_and_ns
            ]),
            MergeSampler(*zip(*[
                (datastream.sampler, datastream.dataset, n)
                for (datastream, n) in datastreams_and_ns
            ])),
        )

    @staticmethod
    def zip(datastreams: List[Datastream]) -> Datastream[Tuple]:
        '''
        Zip multiple datastreams together so that all combinations of examples
        are possible (i.e. the product) creating tuples like
        ``(example1, example2, ...)``. The samples are drawn independently
        from each underlying datastream.
        '''
        return Datastream(
            Dataset.combine([
                datastream.dataset for datastream in datastreams
            ]),
            ZipSampler(*zip(*[
                (datastream.sampler, datastream.dataset)
                for datastream in datastreams
            ])),
        )

    def map(
        self: Datastream[T], function: Callable[[T], R]
    ) -> Datastream[R]:
        '''
        Creates a new Datastream with a new mapped dataset. See
        :func:`Dataset.map` for details.
        '''
        return Datastream(
            self.dataset.map(function),
            self.sampler,
        )

    def starmap(
        self: Datastream[T], function: Callable[[...], R]
    ) -> Datastream[R]:
        ''' 
        Creates a new Datastream with a new starmapped dataset. See
        :func:`Dataset.starmap` for details.
        '''
        return Datastream(
            self.dataset.starmap(function),
            self.sampler,
        )

    def data_loader(
        self,
        n_batches_per_epoch: int = None,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        '''
        Get ``torch.utils.data.DataLoader`` for use in pytorch pipeline.

        The argument ``n_batches_per_epoch`` overrides the underlying length
        of the dataset. If the epoch ends before the full dataset has been
        processed then it will continue from the same point the next epoch.

        >>> data_loader = (
        ...     Datastream(Dataset.from_subscriptable([5, 5, 5]))
        ...     .data_loader(batch_size=5, n_batches_per_epoch=10)
        ... )
        >>> list(data_loader)[0]
        tensor([5, 5, 5, 5, 5])
        '''
        if n_batches_per_epoch is None:
            sampler = self.sampler
        else:
            sampler = RepeatSampler(
                self.sampler,
                n_batches_per_epoch * kwargs['batch_size'],
            )

        return torch.utils.data.DataLoader(
            self.dataset, sampler=sampler, **kwargs
        )

    def zip_index(self: Datastream[T]) -> Datastream[Tuple[T, int]]:
        '''
        Zip the output with its underlying `Dataset` index. The output of the
        pipeline will be a tuple ``(output, index)``

        This method is useful when you want modify your sample weights during
        training since that requires the index of the example.

        See :func:`Dataset.zip_index` for more details.
        '''
        return Datastream(
            self.dataset.zip_index(),
            self.sampler,
        )

    def weight(self, index: int) -> float:
        '''Get sample weight for specific example.'''
        return self.sampler.weight(index)

    def update_weights_(self, function: Callable[[np.array], np.array]):
        '''Update all sample weights by function **in-place**.'''
        self.sampler.update_weights_(function)

    def update_example_weight_(self, weight: Union[List, float], index: int):
        '''Update sample weight for specific example **in-place**.'''
        self.sampler.update_example_weight_(weight, index)

    def sample_proportion(
        self: Datastream[T],
        proportion: float,
    ) -> Datastream[T]:
        '''
        Create new ``Datastream[T]`` with changed proportion. This changes the
        numbers of drawn samples before restarting sampling with new weights
        and allowing sample replacement.

        It is important to set this if you are using sample weights because the
        default is to sample without replacement with proportion 1.0 which will
        cause the weighting scheme to only affect the order in which the
        samples are drawn.
        '''
        return Datastream(
            self.dataset,
            self.sampler.sample_proportion(proportion),
        )

    def take(
        self: Datastream[T],
        n_samples: PositiveInt,
    ) ->  Datastream[T]:
        '''
        Like :func:`Datastream.sample_proportion` but specify the number of
        samples instead of a proportion.
        '''
        if n_samples < 1:
            raise ValueError('n_samples must be greater than or equal to 1')
        return self.sample_proportion(n_samples / len(self))

    def state_dict(self) -> Dict:
        '''Get state of datastream. Useful for checkpointing sample weights.'''
        return dict(sampler=self.sampler.state_dict())

    def load_state_dict(self, state_dict: Dict):
        '''Load saved state from :func:`Datastream.state_dict`.'''
        return self.sampler.load_state_dict(state_dict['sampler'])

    def multi_sample(self: Datastream[T], n: int) -> Datastream[T]:
        '''
        Split datastream into clones with different sample weights and then
        merge them. The weights when accessed will be a sequence of multiple
        weights.

        This allows sample strategies where you for example stratify based on
        the model's predictions as shown below.

        .. highlight:: python
        .. code-block:: python

            datastream = (
                Datastream(dataset)
                .zip_index()
                .multi_sample(n_classes)
                .sample_proportion(0.01)
            )

            data_loader = datastream.data_loader(...)

            for indices, batch in data_loader:
                ...

                for index in indices:
                    datastream.update_weight(index, predicted_classes)

        '''
        return Datastream(
            self.dataset,
            MultiSampler.from_number(n, self.dataset),
        )

    def cache(
        self,
        key_column: str,
    ):
        '''Cache dataset in-memory. See :func:`Dataset.cache` for details.'''
        return Datastream(
            self.dataset.cache(key_column),
            self.sampler,
        )


def test_infinite():

    datastream = Datastream(Dataset.from_subscriptable(list('abc')))
    it = iter(datastream.data_loader(batch_size=8, n_batches_per_epoch=10))
    for _ in range(10):
        batch = next(it)


def test_iter():

    datastream = Datastream(Dataset.from_subscriptable(list('abc')))
    assert len(list(datastream)) == 3


def test_empty():

    import pytest

    with pytest.raises(ValueError):
        Datastream(Dataset.from_subscriptable(list()))


def test_datastream_merge():

    datastream = Datastream.merge([
        Datastream(Dataset.from_subscriptable(list('abc'))),
        Datastream(Dataset.from_subscriptable(list('def'))),
    ])

    it = iter(datastream.sampler)
    for _ in range(2):
        index = next(it)

    it = iter(datastream.data_loader(batch_size=8, n_batches_per_epoch=10))
    for _ in range(10):
        batch = next(it)

    assert (
        len(list(
            datastream.data_loader(batch_size=1)
        )) == len(datastream)
    )


def test_datastream_zip():

    datasets = [
        Dataset.from_subscriptable([1, 2]),
        Dataset.from_subscriptable([3, 4, 5]),
        Dataset.from_subscriptable([6, 7]),
    ]

    datastreams = [
        Datastream(ds, sampler=torch.utils.data.SequentialSampler(ds))
        for ds in datasets
    ]
    zipped_datastream = Datastream.zip(datastreams)

    batch = next(iter(zipped_datastream.data_loader(batch_size=3)))
    assert len(batch) == 3 and len(batch[0]) == 3
    assert batch[0][0] == 1 and batch[0][1] == 2 and batch[0][2] == 1
    assert batch[1][0] == 3 and batch[1][1] == 4 and batch[1][2] == 5
    assert batch[2][0] == 6 and batch[2][1] == 7 and batch[2][2] == 6

    assert (
        len(list(
            zipped_datastream.data_loader(batch_size=1)
        )) == len(zipped_datastream)
    )


def test_datastream_merge_zip_merge():
    '''
    Repeating because it only sometimes recreated an error that occured
    when using mixup/mixmatch
    '''

    def RandomDatastream():
        return Datastream(Dataset.from_subscriptable(
            list(range(np.random.randint(1, 10)))
        ))

    def MergedDatastream():
        return Datastream.merge([RandomDatastream(), RandomDatastream()])

    def ZippedMergedDatastream():
        return Datastream.zip([MergedDatastream(), MergedDatastream()])

    for attempt in range(10):
        print('attempt:', attempt)
        datastream = Datastream.merge([
            (ZippedMergedDatastream(), 1),
            (ZippedMergedDatastream(), 5),
        ])

        it = iter(datastream.data_loader(
            batch_size=16, n_batches_per_epoch=10
        ))
        for _ in range(10):
            print(next(it))


def test_datastream_simple_weights():

    dataset = Dataset.from_subscriptable([1, 2, 3, 4])
    datastream = (
        Datastream(dataset)
        .zip_index()
        .starmap(lambda integer, index: dict(
            integer=integer,
            index=index,
        ))
        .sample_proportion(0.5)
    )

    removed_indices = [0, 3]
    for index in removed_indices:
        datastream.update_example_weight_(0.0, removed_indices)

    samples = list(datastream.data_loader(batch_size=1))

    assert len(samples) == 2

    for sample in samples:
        if sample['index'] in removed_indices:
            raise AssertionError(
                'Samples with 0 weight were drawn from the dataset'
            )


def test_merge_datastream_weights():

    datasets = [
        Dataset.from_subscriptable([1, 2]),
        Dataset.from_subscriptable([3, 4, 5]),
        Dataset.from_subscriptable([6, 7]),
    ]

    datastream = (
        Datastream.merge([
            Datastream(dataset)
            for dataset in datasets
        ])
        .zip_index()
        .starmap(lambda integer, index: dict(
            integer=integer,
            index=index,
        ))
        .sample_proportion(0.5)
    )

    removed_indices = [0, 3]
    for index in removed_indices:
        datastream.update_example_weight_(0.0, index)

    samples = list(datastream.data_loader(batch_size=4, n_batches_per_epoch=4))

    datastream.update_weights_(lambda weights: weights * 0.9 + 1 * 0.1)


def test_multi_sample():

    data = [1, 2, 4]
    n_multi_sample = 2

    datastream = (
        Datastream(
            Dataset.from_subscriptable(data)
        )
        .map(lambda number: number ** 2)
        .multi_sample(n_multi_sample)
        .sample_proportion(0.5)
        .zip_index()
        .starmap(lambda number, index: (number ** 0.5, index))
    )

    output = [
        (number, index)
        for number, index in datastream.data_loader(batch_size=1)
    ]
    assert len(output) == len(data) * n_multi_sample
    print(output)

    state = datastream.state_dict()
    datastream.load_state_dict(state)

    for index, number in zip(output, range(2)):
        datastream.update_example_weight_(index, 0)

    output2 = [
        (number, index)
        for number, index in datastream.data_loader(batch_size=1)
    ]
    assert len(output2) == len(data) * n_multi_sample

    zero_indices = set([index for _, index in output[:2]])
    for number, index in output2:
        assert index not in zero_indices


def test_take():

    import pytest

    datastream = Datastream(Dataset.from_subscriptable(list('abc'))).take(2)
    assert len(list(datastream.data_loader(batch_size=1))) == 2

    with pytest.raises(ValueError):
        Datastream(Dataset.from_subscriptable(list('abc'))).take(0)

    datastream = Datastream.merge([
        Datastream(Dataset.from_subscriptable(list('abc'))),
        Datastream(Dataset.from_subscriptable(list('d'))),
    ])
    assert len(list(datastream.take(2).data_loader(batch_size=1))) == 2


def test_sequential_sampler():

    from datastream.samplers import SequentialSampler

    dataset = Dataset.from_subscriptable(list('abc'))
    datastream = Datastream(dataset, SequentialSampler(len(dataset))).take(2)
    assert len(list(datastream.data_loader(batch_size=1))) == 2

    datastream = Datastream(dataset, SequentialSampler(len(dataset)))
    it = iter(datastream.data_loader(batch_size=6, n_batches_per_epoch=10))
    assert next(it) == ['a', 'b', 'c', 'a', 'b', 'c']
