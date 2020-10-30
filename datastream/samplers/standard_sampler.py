from __future__ import annotations
from pydantic import BaseModel
import torch


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
                num_samples=int(max(1, min(length, length * proportion))),
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
