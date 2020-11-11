from __future__ import annotations
from pydantic import BaseModel
import torch


class SequentialSampler(BaseModel, torch.utils.data.Sampler):
    sampler: torch.utils.data.SequentialSampler

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __init__(self, length):
        BaseModel.__init__(
            self,
            sampler=torch.utils.data.SequentialSampler(torch.ones(length))
        )

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return iter(self.sampler)

    def sample_proportion(self, proportion):
        return SequentialSampler(min(
            len(self),
            int(len(self) * proportion)
        ))
