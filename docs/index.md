# pytorch-datastream

Simple dataset to dataloader library for pytorch.

## Quick Example

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

dataset = (
    Dataset.from_subscriptable([1, 2, 3])
    .map(lambda number: number + 1)
)

assert dataset[-1] == 4

data_loader = (
    Datastream(dataset)
    .data_loader(batch_size=16, n_batches_per_epoch=100)
)

assert len(next(iter(data_loader))) == 16
```

## Features

- Simple, readable dataset pipeline creation
- Built-in support for:
  - Imbalanced datasets
  - Oversampling / stratification
  - Weighted sampling
  - Easy conversion to PyTorch DataLoader
- Testable examples in documentation
- Type hints and Pydantic validation
- Clean, maintainable codebase

## Installation

Install with poetry:

```text
poetry add pytorch-datastream
```

Or with pip:

```text
pip install pytorch-datastream
```

## Next Steps

- Check out the [Getting Started](getting-started.md) guide
- See the [Dataset](dataset.md) and [Datastream](datastream.md) API references
