from __future__ import annotations
from pydantic import BaseModel
from typing import Tuple, Callable, Iterable
from functools import partial
from itertools import chain, islice
import torch
from datastream.tools import starcompose, repeat_map_chain
from datastream import Dataset


class StandardSampler(BaseModel, torch.utils.data.Sampler):
    proportion: float
    replacement: bool
    sampler: torch.utils.data.WeightedRandomSampler

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(self, length, proportion=1.0, replacement=False):
        BaseModel.__init__(
            self,
            proportion=proportion,
            replacement=replacement,
            sampler=torch.utils.data.WeightedRandomSampler(
                torch.ones(length).double(),
                num_samples=max(int(length * proportion), 1),
                replacement=replacement,
            )
        )

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return iter(self.sampler)

    @property
    def weights(self):
        return self.sampler.weights

    def weight(self, index):
        return self.sampler.weights[index].item()

    def update_weights_(self, function):
        self.sampler.weights[:] = function(self.sampler.weights)

    def update_example_weight_(self, weight, index):
        if hasattr(weight, 'item'):
            weight = weight.item()

        self.sampler.weights[index] = weight

    def sample_proportion(self, proportion):
        sampler = StandardSampler(
            len(self),
            proportion,
            self.replacement,
        )
        sampler.sampler.weights = self.sampler.weights
        return sampler

    def state_dict(self):
        return dict(weights=self.sampler.weights)

    def load_state_dict(self, state_dict):
        self.sampler.weights[:] = state_dict['weights']


class MergeSampler(BaseModel, torch.utils.data.Sampler):
    samplers: Tuple[torch.utils.data.Sampler, ...]
    datasets: Tuple[Dataset, ...]
    ns: Tuple[int, ...]
    length: int
    from_mapping: Callable[[int], Tuple[int, int]]
    merged_samplers: Iterable

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(self, samplers, datasets, ns):
        BaseModel.__init__(
            self,
            samplers=samplers,
            datasets=datasets,
            ns=ns,
            length=MergeSampler.merged_samplers_length(samplers),
            from_mapping=Dataset.create_from_concat_mapping(datasets),
            merged_samplers=MergeSampler.merge_samplers(
                samplers, datasets, ns
            ),
        )

    def __len__(self):
        return self.length

    def __iter__(self):
        return islice(self.merged_samplers, self.length)

    @staticmethod
    def merged_samplers_length(samplers):
        return max([len(sampler) for sampler in samplers])

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


class ZipSampler(BaseModel, torch.utils.data.Sampler):
    samplers: Tuple[torch.utils.data.Sampler, ...]
    datasets: Tuple[Dataset, ...]
    length: int
    from_mapping: Callable[[int], Tuple[int, ...]]
    zipped_samplers: Iterable

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(self, samplers, datasets):
        BaseModel.__init__(
            self,
            samplers=samplers,
            datasets=datasets,
            length=max(map(len, samplers)),
            from_mapping=Dataset.create_from_combine_mapping(datasets),
            zipped_samplers=ZipSampler.zip_samplers(samplers, datasets),
        )

    def __len__(self):
        return self.length

    def __iter__(self):
        return islice(self.zipped_samplers, self.length)

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
        for sampler, weight, inner_index in zip(
            self.samplers, weights, inner_indices
        ):
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
class MultiSampler(BaseModel, torch.utils.data.Sampler):
    samplers: Tuple[torch.utils.data.Sampler, ...]
    dataset: Dataset
    length: int
    merged_samplers: Iterable

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(self, samplers, dataset):
        BaseModel.__init__(
            self,
            samplers=samplers,
            dataset=dataset,
            length=len(dataset) * len(samplers),
            merged_samplers=MultiSampler.merge_samplers(
                samplers,
                [1 for _ in samplers],
            )
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
        return islice(self.merged_samplers, self.length)

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


class RepeatSampler(BaseModel, torch.utils.data.Sampler):
    sampler: torch.utils.data.Sampler
    length: int
    epoch_bound: bool = False
    queue: Iterable

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, sampler, length, epoch_bound=False):
        '''
        Wrapper that repeats and limits length of sampling based on
        epoch length and batch size
        '''
        BaseModel.__init__(
            self,
            sampler=sampler,
            length=length,
            epoch_bound=epoch_bound,
            queue=iter(sampler)
        )

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
            self.sampler.sample_proportion(proportion),
            self.length,
            self.epoch_bound,
        )

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        return self.sampler.load_state_dict(state_dict)
