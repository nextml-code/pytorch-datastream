# Getting Started

## Installation

```bash
pip install pytorch-datastream
```

## Usage

### Dataset

A `Dataset[T]` is a mapping that allows pipelining of functions in a readable syntax returning an example of type `T`.

```python
from datastream import Dataset

fruits_and_cost = (
    ('apple', 5),
    ('pear', 7),
    ('banana', 14),
    ('kiwi', 100),
)

dataset = (
    Dataset.from_subscriptable(fruits_and_cost)
    .starmap(lambda fruit, cost: (
        fruit,
        cost * 2,
    ))
)

assert dataset[2] == ('banana', 28)
```

### Datastream

A `Datastream[T]` is an iterable that yields batches of type `T` from one or more datasets.

```python
import numpy as np
from datastream import Dataset, Datastream

dataset = Dataset.from_subscriptable([1, 2, 3, 4])
datastream = Datastream(dataset)

for batch in datastream.data_loader(batch_size=2):
    assert len(batch) == 2
```

### Merge

Merge multiple datasets into a single datastream. The proportion of samples from each dataset in a batch can be controlled by passing tuples of `(datastream, proportion)`.

```python
import numpy as np
from datastream import Dataset, Datastream

dataset1 = Dataset.from_subscriptable([1, 2, 3, 4])
dataset2 = Dataset.from_subscriptable([5, 6, 7, 8])

datastream = Datastream.merge([
    (Datastream(dataset1), 1),
    (Datastream(dataset2), 1),
])

for batch in datastream.data_loader(batch_size=2):
    assert len(batch) == 2
```
