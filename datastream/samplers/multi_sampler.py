from __future__ import annotations
from pydantic import BaseModel
from typing import Tuple, Iterable
from itertools import chain, islice
import torch
from datastream.tools import repeat_map_chain
from datastream.samplers import StandardSampler
from datastream import Dataset


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
