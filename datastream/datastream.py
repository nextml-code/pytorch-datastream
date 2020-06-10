from functools import partial
from itertools import repeat, chain, islice
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
from datastream import starcompose, star, repeat_map_chain, Dataset


class StandardSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, length, proportion=1.0, replacement=False):
        super().__init__(
            torch.ones(length).double(),
            num_samples=int(length * proportion),
            replacement=replacement,
        )

    def weight(self, index):
        return self.weights[index].item()

    def update_weights_(self, function):
        self.weights[:] = function(self.weights)
        
    def update_example_weight_(self, weight, index):
        if hasattr(weight, 'item'):
            weight = weight.item()

        self.weights[index] = weight

    def sample_proportion(self, proportion):
        sampler = StandardSampler(
            len(self),
            proportion,
            self.replacement,
        )
        sampler.weights = self.weights
        return sampler

    def state_dict(self):
        return dict(weights=self.weights)

    def load_state_dict(self, state_dict):
        self.weights[:] = state_dict['weights']


class MergeSampler(torch.utils.data.Sampler):
    def __init__(self, samplers, datasets, ns):
        self.samplers = samplers
        self.datasets = datasets
        self.ns = ns
        self.from_mapping = Dataset.create_from_concat_mapping(datasets)
        self.merged_samplers = MergeSampler.merge_samplers(
            samplers, datasets, ns
        )
        self.length = MergeSampler.merged_samplers_length(samplers)

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.merged_samplers)

    @staticmethod
    def merged_samplers_length(samplers):
        return (
            max([len(sampler) for sampler in samplers])
            * len(samplers)
        )

    @staticmethod
    def merge_samplers(samplers, datasets, ns):
        to_mapping = Dataset.create_to_concat_mapping(datasets)

        def batch(iterable, n):
            while True:
                yield [next(iterable) for _ in range(n)]

        index_batch = zip(*[
            batch(map(
                partial(to_mapping, dataset_index),
                repeat_map_chain(iter, sampler),
            ), n)
            for dataset_index, (sampler, n) in enumerate(zip(samplers, ns))
        ])

        return chain.from_iterable(chain.from_iterable(index_batch))

    def weight(self, index):
        dataset_index, inner_index = self.from_mapping(index)
        return self.samplers[dataset_index].weight(inner_index)

    def update_weights_(self, function):
        for sampler in self.samplers:
            sampler.update_weights_(function)

    def update_example_weight_(self, weight, index):
        dataset_index, inner_index = self.from_mapping(index)
        self.samplers[dataset_index].update_example_weight_(
            weight, inner_index
        )

    def sample_proportion(self, proportion):
        return MergeSampler(
            [
                sampler.sample_proportion(proportion)
                for sampler in self.samplers
            ],
            self.datasets,
            self.ns,
        )

    def state_dict(self):
        return dict(
            samplers=[sampler.state_dict() for sampler in self.samplers]
        )

    def load_state_dict(self, state_dict):
        for sampler, state_dict in zip(self.samplers, state_dict['samplers']):
            sampler.load_state_dict(state_dict)


class ZipSampler(torch.utils.data.Sampler):
    def __init__(self, samplers, datasets):
        self.samplers = samplers
        self.datasets = datasets
        self.from_mapping = Dataset.create_from_combine_mapping(datasets)
        self.zipped_samplers = ZipSampler.zip_samplers(samplers, datasets)
        self.length = max(map(len, samplers))

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.zipped_samplers)

    @staticmethod
    def zip_samplers(samplers, datasets):
        to_mapping = Dataset.create_to_combine_mapping(datasets)

        create_sampler = starcompose(
            partial(map, partial(repeat_map_chain, iter)),
            tuple,
            zip,
            partial(map, to_mapping),
        )
        return create_sampler(samplers)

    def weight(self, index):
        return [
            sampler.weight(inner_index)
            for sampler, inner_index in zip(
                self.samplers, self.from_mapping(index)
            )
        ]

    def update_weights_(self, function):
        for sampler in self.samplers:
            sampler.update_weights_(function)

    def update_example_weight_(self, weights, index):
        inner_indices = self.from_mapping(index)
        for sampler, weight, inner_index in zip(self.samplers, weights, inner_indices):
            sampler.update_example_weight_(
                weight, inner_index
            )

    def sample_proportion(self, proportion):
        return ZipSampler([
            sampler.sample_proportion(proportion)
            for sampler in self.samplers
        ])

    def state_dict(self):
        return dict(
            samplers=[sampler.state_dict() for sampler in self.samplers]
        )

    def load_state_dict(self, state_dict):
        for sampler, state_dict in zip(self.samplers, state_dict['samplers']):
            sampler.load_state_dict(state_dict)



# TODO: write custom sampler that avoid replacement between samplers
class MultiSampler(torch.utils.data.Sampler):
    def __init__(self, samplers, dataset):
        self.samplers = samplers
        self.dataset = dataset
        self.length = len(dataset)
        self.merged_samplers = MultiSampler.merge_samplers(
            samplers,
            [1 for _ in samplers],
        )

    @staticmethod
    def from_number(n, dataset):
        return MultiSampler(
            [StandardSampler(len(dataset)) for _ in range(n)],
            dataset,
        )

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.merged_samplers)

    @staticmethod
    def merge_samplers(samplers, ns):
        def batch(iterable, n):
            while True:
                yield [next(iterable) for _ in range(n)]

        index_batch = zip(*[
            batch(repeat_map_chain(iter, sampler), n)
            for sampler, n in zip(samplers, ns)
        ])

        return chain.from_iterable(chain.from_iterable(index_batch))

    def weight(self, index):
        return [sampler.weight(index) for sampler in self.samplers]

    def update_weights_(self, function):
        for sampler in self.samplers:
            sampler.update_weights_(function)

    def update_example_weight_(self, weights, index):
        for sampler, weight in zip(self.samplers, weights):
            sampler.update_example_weight_(
                weight, index
            )

    def sample_proportion(self, proportion):
        return MultiSampler(
            [
                sampler.sample_proportion(proportion)
                for sampler in self.samplers
            ],
            self.dataset
        )

    def state_dict(self):
        return dict(
            samplers=[sampler.state_dict() for sampler in self.samplers]
        )

    def load_state_dict(self, state_dict):
        for sampler, state_dict in zip(self.samplers, state_dict['samplers']):
            sampler.load_state_dict(state_dict)



class RepeatSampler(torch.utils.data.Sampler):
    def __init__(self, sampler, length, epoch_bound=False):
        '''
        Wrapper that repeats and limits length of sampling based on
        epoch length and batch size
        '''
        super().__init__(range(length))
        self.sampler = sampler
        self.length = length
        self.epoch_bound = epoch_bound
        self.queue = iter(self.sampler)

    def __iter__(self):
        if self.epoch_bound:
            self.queue = iter(self.sampler)

        for _ in range(self.length):
            try:
                yield next(self.queue)
            except StopIteration:
                self.queue = iter(self.sampler)
                yield next(self.queue)

    def __len__(self):
        return self.length

    def weight(self, index):
        return self.sampler.weight(index)

    def update_weights_(self, function):
        self.sampler.update_weights_(function)

    def update_example_weight_(self, weights, index):
        self.sampler.update_example_weight_(weights, index)

    def sample_proportion(self, proportion):
        return RepeatSampler(
            sampler.sample_proportion(proportion),
            self.length,
            self.epoch_bound,
        )

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        return self.sampler.load_state_dict(state_dict)


class Datastream:
    def __init__(self, dataset, sampler=None):
        super().__init__()
        self.dataset = dataset

        if sampler is None:
            sampler = StandardSampler(len(self.dataset))
        self.sampler = sampler

    def data_loader(self, n_batches_per_epoch=None, **kwargs):
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

    def weight(self, index):
        return self.sampler.weight(index)

    def update_weights_(self, function):
        self.sampler.update_weights_(function)

    def update_example_weight_(self, weights, index):
        self.sampler.update_example_weight_(weights, index)

    def sample_proportion(self, proportion):
        return Datastream(
            self.dataset,
            self.sampler.sample_proportion(proportion),
        )

    @staticmethod
    def merge(datastreams_and_ns):
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
    def zip(datastreams):
        return Datastream(
            Dataset.combine([
                datastream.dataset for datastream in datastreams
            ]),
            ZipSampler(*zip(*[
                (datastream.sampler, datastream.dataset)
                for datastream in datastreams
            ])),
        )

    def multi_sample(self, n):
        return Datastream(
            self.dataset,
            MultiSampler.from_number(n, self.dataset),
        )


    @staticmethod
    def shared_dataset(datastreams):
        # same as multi sample? possibly allow different pipelines?
        pass


    def map(self, fn):
        return Datastream(
            self.dataset.map(fn),
            self.sampler,
        )

    def zip_index(self):
        return Datastream(
            self.dataset.zip_index(),
            self.sampler,
        )

    def state_dict(self):
        return dict(sampler=self.sampler.state_dict())

    def load_state_dict(self, state_dict):
        return self.sampler.load_state_dict(state_dict['sampler'])


def test_datastream_merge():

    datastream = Datastream.merge([
        Datastream(Dataset.from_subscriptable(list('abc'))),
        Datastream(Dataset.from_subscriptable(list('def'))),
    ])

    it = iter(datastream.sampler)
    for _ in range(2):
        index = next(it)

    it = iter(datastream.data_loader(batch_size=8))
    for _ in range(10):
        batch = next(it)


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


def test_datastream_merge_zip_merge():
    '''
    repeating because it only sometimes recreated an error that occured
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

        it = iter(datastream.data_loader(batch_size=16, n_batches_per_epoch=10))
        for _ in range(10):
            print(next(it))


def test_datastream_simple_weights():

    dataset = Dataset.from_subscriptable([1, 2, 3, 4])
    datastream = (
        Datastream(dataset)
        .zip_index()
        .map(lambda integer, index: dict(
            integer=integer,
            index=index,
        ))
        .sample_proportion(0.5)
    )

    removed_indices = [0, 3]
    for index in removed_indices:
        datastream.update_example_weight_(0.0, removed_indices)

    samples = list(datastream.data_loader(batch_size=1))

    if len(samples) != 2:
        raise AssertionError(
            'Expected 2 samples due to proportion 0.5 and dataset length 4'
        )

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
        .map(lambda integer, index: dict(
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
