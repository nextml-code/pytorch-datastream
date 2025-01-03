# Datastream

A `Datastream[T]` combines a `Dataset[T]` and a sampler into a stream of examples.

By default, samples are drawn without replacement until the dataset is exhausted. The sampling behavior can be modified using `sample_proportion`.

## Basic Usage

### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

# Create a simple dataset
dataset = Dataset.from_subscriptable([1, 2, 3])

# Create a datastream with batching
data_loader = (
    Datastream(dataset)
    .data_loader(batch_size=2)
)

# First batch should have 2 items
batch = next(iter(data_loader))
assert len(batch) == 2
```

## Constructor

### `Datastream`

```python
Datastream(dataset: Dataset[T], sampler: Optional[torch.utils.data.Sampler] = None) -> Datastream[T]
```

Create a new datastream from a dataset and optional sampler.

#### Parameters

- `dataset`: The source dataset to stream from
- `sampler`: Optional sampler to use. If None, a StandardSampler will be used

#### Raises

- `ValueError`: If dataset is empty

## Data Loading Methods

### `data_loader`

```python
data_loader(self, n_batches_per_epoch: Optional[int] = None, **kwargs) -> torch.utils.data.DataLoader
```

Get a PyTorch DataLoader for use in training pipeline.

#### Parameters

- `n_batches_per_epoch`: Optional number of batches per epoch. If provided, overrides the underlying length of the dataset
- `**kwargs`: Additional arguments passed to PyTorch DataLoader

#### Returns

- A PyTorch DataLoader instance

#### Notes

If `n_batches_per_epoch` is set and the epoch ends before the full dataset has been processed, it will continue from the same point in the next epoch.

This is particularly useful when:

- Training on very large datasets where you want fixed-size epochs
- Using weighted sampling where you want to ensure all classes are seen equally
- Doing curriculum learning where you want to control exactly how many samples are seen

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

data_loader = (
    Datastream(Dataset.from_subscriptable([5, 5, 5]))
    .data_loader(batch_size=2, n_batches_per_epoch=3)
)
batches = list(data_loader)
assert len(batches) == 3  # Always get exactly 3 batches
assert len(batches[0]) == 2  # Each batch has size 2
```

## Sampling Methods

### `sample_proportion`

```python
sample_proportion(self, proportion: float) -> Datastream[T]
```

Create new Datastream with changed sampling proportion.

#### Parameters

- `proportion`: The proportion of the dataset to sample before allowing replacement

#### Returns

- A new Datastream with modified sampling behavior

#### Notes

This changes the number of drawn samples before restarting sampling with new weights and allowing sample replacement.

It is important to set this if you are using sample weights because the default is to sample without replacement with proportion 1.0, which will
cause the weighting scheme to only affect the order in which the samples are drawn.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

# Create a datastream that will draw half the dataset before allowing replacement
datastream = (
    Datastream(Dataset.from_subscriptable([1, 2, 3, 4]))
    .sample_proportion(0.5)  # Draw 2 samples before replacement
)

# Sample size is still the full dataset length
assert len(list(datastream)) == len(datastream)

# But after 2 samples, items can be repeated
samples = []
for _ in range(4):
    samples.extend(list(datastream))
assert len(set(samples)) < len(samples)  # Some samples are repeated
```

### `take`

```python
take(self, n_samples: PositiveInt) -> Datastream[T]
```

Create new Datastream that draws a fixed number of samples.

#### Parameters

- `n_samples`: Number of samples to draw before allowing replacement

#### Returns

- A new Datastream with modified sampling behavior

#### Notes

Like `sample_proportion` but specify the number of samples instead of a proportion.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

datastream = (
    Datastream(Dataset.from_subscriptable([1, 2, 3, 4, 5]))
    .take(2)  # Draw exactly 2 samples before allowing replacement
)
assert len(list(datastream)) == 2
```

## Weight Management Methods

### `weight`

```python
weight(self, index: int) -> float
```

Get sample weight for specific example.

#### Parameters

- `index`: Index of the example to get weight for

#### Returns

- The weight of the example at the given index

#### Notes

Weights affect the probability of sampling each example.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

datastream = Datastream(Dataset.from_subscriptable([1, 2, 3]))
assert datastream.weight(0) == 1.0  # Default weight is 1.0
```

### `update_weights_`

```python
update_weights_(self, function: Callable[[np.array], np.array]) -> None
```

Update all sample weights by function **in-place**.

#### Parameters

- `function`: Function that takes array of weights and returns modified weights

#### Notes

This is useful for implementing importance sampling or curriculum learning strategies.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
import numpy as np
from datastream import Dataset, Datastream

# Create a datastream where we'll downweight all samples
datastream = Datastream(Dataset.from_subscriptable([1, 2, 3]))
datastream.update_weights_(lambda weights: weights * 0.5)
assert datastream.weight(0) == 0.5
```

### `update_example_weight_`

```python
update_example_weight_(self, weight: Union[List, float], index: int) -> None
```

Update sample weight for specific example **in-place**.

#### Parameters

- `weight`: New weight value(s) for the example
- `index`: Index of the example to update

#### Notes

This is useful when you want to adjust the sampling probability of individual examples, for instance based on model performance.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

datastream = Datastream(Dataset.from_subscriptable([1, 2, 3]))
datastream.update_example_weight_(0.5, index=0)  # Make first example half as likely
assert datastream.weight(0) == 0.5
```

### `multi_sample`

```python
multi_sample(self, n: int) -> Datastream[T]
```

Split datastream into clones with different sample weights and merge them.

#### Parameters

- `n`: Number of weight clones to create

#### Returns

- A new Datastream with multiple weight sets

#### Notes

The weights when accessed will be a sequence of multiple weights. This allows sample strategies where you for example stratify based on the model's predictions.
A common use case is handling multi-label classification where you want to ensure good coverage of all classes.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

n_classes = 3
datastream = (
    Datastream(Dataset.from_subscriptable([1, 2, 3]))
    .zip_index()
    .multi_sample(n_classes)
    .sample_proportion(0.5)
)

# Each example now has n_classes weights that can be adjusted independently
weights = [datastream.weight(0) for _ in range(n_classes)]
assert len(weights) == n_classes
```

## Static Methods

### `merge`

```python
merge(datastreams_and_ns: Tuple[Union[Datastream[T], Tuple[Datastream[T], int]], ...]) -> Datastream[T]
```

Creates a merged datastream where samples are drawn one at a time from each underlying datastream.

#### Parameters

- `datastreams_and_ns`: List of datastreams or tuples of (datastream, n_samples)

#### Returns

- A new merged Datastream

#### Notes

Also known as "interleave". Optionally you can define the number of drawn samples per Datastream.

This is useful when you want to:

- Combine multiple data sources with different sampling rates
- Implement curriculum learning by controlling how often each type of example is seen
- Balance between different tasks in multi-task learning

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

datastream1 = Datastream(Dataset.from_subscriptable([1, 1]))  # Task 1
datastream2 = Datastream(Dataset.from_subscriptable([2, 2]))  # Task 2
datastream3 = Datastream(Dataset.from_subscriptable([3, 3, 3, 3]))  # Task 3

# Draw more samples from task 3 (might be harder to learn)
merged = Datastream.merge([
    (datastream1, 1),  # Draw 1 sample at a time from task 1
    (datastream2, 1),  # Draw 1 sample at a time from task 2
    (datastream3, 2),  # Draw 2 samples at a time from task 3
])

samples = list(merged)
assert samples == [1, 2, 3, 3, 1, 2, 3, 3]  # Task 3 appears twice as often
```

### `zip`

```python
zip(datastreams: List[Datastream]) -> Datastream[Tuple]
```

Zip multiple datastreams together so that samples are drawn independently.

#### Parameters

- `datastreams`: List of datastreams to zip together

#### Returns

- A new zipped Datastream that yields tuples

#### Notes

Samples are drawn independently from each underlying datastream, creating tuples like `(example1, example2, ...)`.
This is different from `Dataset.combine`, which creates all possible combinations (cartesian product) of examples.

This is particularly useful for:

- Creating paired samples for contrastive learning
- Implementing data augmentation strategies
- Combining different types of inputs

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset, Datastream

# Create two streams: one for images, one for labels
datastream1 = Datastream(Dataset.from_subscriptable([1, 2]))  # e.g., image IDs
datastream2 = Datastream(Dataset.from_subscriptable([3, 4]))  # e.g., augmentation params

# Get samples drawn independently from each datastream
zipped = Datastream.zip([datastream1, datastream2])
samples = list(zipped)
print("Samples:", samples)  # Debug output
print("Length:", len(samples))  # Debug output
print("Expected length:", max(len(datastream1.dataset), len(datastream2.dataset)))  # Debug output
assert len(samples) == 2  # Independent samples: (1,3), (2,4)

# For comparison, Dataset.combine creates all possible combinations
combined = Dataset.combine([datastream1.dataset, datastream2.dataset])
combined_samples = list(combined)
print("Combined samples:", combined_samples)  # Debug output
print("Combined length:", len(combined_samples))  # Debug output
print("Expected combined length:", len(datastream1.dataset) * len(datastream2.dataset))  # Debug output
assert len(combined_samples) == 4  # All combinations: (1,3), (1,4), (2,3), (2,4)
```
