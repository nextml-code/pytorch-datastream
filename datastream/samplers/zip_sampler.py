from __future__ import annotations

from functools import partial
from itertools import islice
from typing import Callable, Iterable, Tuple

import torch
import torch.utils.data
from pydantic import BaseModel, ConfigDict

from datastream import Dataset
from datastream.tools import repeat_map_chain, starcompose


class ZipSampler(BaseModel, torch.utils.data.Sampler):
    samplers: Tuple[torch.utils.data.Sampler, ...]
    datasets: Tuple[Dataset, ...]
    length: int
    from_mapping: Callable[[int], Tuple[int, ...]]
    zipped_samplers: Iterable

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

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
            for sampler, inner_index in zip(self.samplers, self.from_mapping(index))
        ]

    def update_weights_(self, function):
        for sampler in self.samplers:
            sampler.update_weights_(function)

    def update_example_weight_(self, weights, index):
        inner_indices = self.from_mapping(index)
        for sampler, weight, inner_index in zip(self.samplers, weights, inner_indices):
            sampler.update_example_weight_(weight, inner_index)

    def sample_proportion(self, proportion):
        return ZipSampler(
            [sampler.sample_proportion(proportion) for sampler in self.samplers]
        )

    def state_dict(self):
        return dict(samplers=[sampler.state_dict() for sampler in self.samplers])

    def load_state_dict(self, state_dict):
        for sampler, state_dict in zip(self.samplers, state_dict["samplers"]):
            sampler.load_state_dict(state_dict)
