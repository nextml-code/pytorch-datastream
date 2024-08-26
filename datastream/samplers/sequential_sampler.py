from __future__ import annotations

import torch
import torch.utils.data
from pydantic import BaseModel, ConfigDict


class SequentialSampler(BaseModel, torch.utils.data.Sampler):
    sampler: torch.utils.data.SequentialSampler

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    def __init__(self, length):
        BaseModel.__init__(
            self, sampler=torch.utils.data.SequentialSampler(torch.ones(length))
        )

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return iter(self.sampler)

    def sample_proportion(self, proportion):
        return SequentialSampler(min(len(self), int(len(self) * proportion)))
