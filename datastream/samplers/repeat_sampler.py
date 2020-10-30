from __future__ import annotations
from pydantic import BaseModel
from typing import Iterable
import torch


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
