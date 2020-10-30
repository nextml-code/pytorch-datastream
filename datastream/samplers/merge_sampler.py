from __future__ import annotations
from pydantic import BaseModel
from typing import Tuple, Callable, Iterable
from functools import partial
from itertools import chain, islice
import torch
from datastream.tools import repeat_map_chain
from datastream import Dataset


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
