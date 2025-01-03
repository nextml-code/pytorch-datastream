# Datastream

A `Datastream[T]` combines a `Dataset[T]` and a sampler into a stream of examples.

By default, samples are drawn without replacement until the dataset is exhausted. The sampling behavior can be modified using `sample_proportion`.

```python test
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

## Methods

### data_loader

Get a PyTorch DataLoader for use in training pipeline. The argument `n_batches_per_epoch` overrides the underlying length of the dataset.
If the epoch ends before the full dataset has been processed then it will continue from the same point the next epoch.

This is particularly useful when:

- Training on very large datasets where you want fixed-size epochs
- Using weighted sampling where you want to ensure all classes are seen equally
- Doing curriculum learning where you want to control exactly how many samples are seen

```python test
data_loader = (
    Datastream(Dataset.from_subscriptable([5, 5, 5]))
    .data_loader(batch_size=2, n_batches_per_epoch=3)
)
batches = list(data_loader)
assert len(batches) == 3  # Always get exactly 3 batches
assert len(batches[0]) == 2  # Each batch has size 2
```

### sample_proportion

Create new Datastream with changed proportion. This changes the numbers of drawn samples before restarting sampling with new weights
and allowing sample replacement.

It is important to set this if you are using sample weights because the default is to sample without replacement with proportion 1.0 which will
cause the weighting scheme to only affect the order in which the samples are drawn.

```python test
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

### take

Like `sample_proportion` but specify the number of samples instead of a proportion.

```python test
datastream = (
    Datastream(Dataset.from_subscriptable([1, 2, 3, 4, 5]))
    .take(2)  # Draw exactly 2 samples before allowing replacement
)
assert len(list(datastream)) == 2
```

### weight

Get sample weight for specific example. Weights affect the probability of sampling each example.

```python test
datastream = Datastream(Dataset.from_subscriptable([1, 2, 3]))
assert datastream.weight(0) == 1.0  # Default weight is 1.0
```

### update*weights*

Update all sample weights by function **in-place**. This is useful for implementing importance sampling
or curriculum learning strategies.

```python test
import numpy as np

# Create a datastream where we'll downweight all samples
datastream = Datastream(Dataset.from_subscriptable([1, 2, 3]))
datastream.update_weights_(lambda weights: weights * 0.5)
assert datastream.weight(0) == 0.5
```

### update*example_weight*

Update sample weight for specific example **in-place**. This is useful when you want to adjust
the sampling probability of individual examples, for instance based on model performance.

```python test
datastream = Datastream(Dataset.from_subscriptable([1, 2, 3]))
datastream.update_example_weight_(0.5, index=0)  # Make first example half as likely
assert datastream.weight(0) == 0.5
```

### multi_sample

Split datastream into clones with different sample weights and then merge them. The weights when accessed will be a sequence of multiple weights.

This allows sample strategies where you for example stratify based on the model's predictions. A common use case is handling
multi-label classification where you want to ensure good coverage of all classes.

```python test
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

### merge

Creates a merged datastream where samples are drawn one at a time from each underlying datastream (also known as "interleave").
Optionally you can define the number of drawn samples per Datastream.

This is useful when you want to:

- Combine multiple data sources with different sampling rates
- Implement curriculum learning by controlling how often each type of example is seen
- Balance between different tasks in multi-task learning

```python test
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

### zip

Zip multiple datastreams together so that all combinations of examples are possible (i.e. the product) creating tuples like `(example1, example2, ...)`.
The samples are drawn independently from each underlying datastream.

This is particularly useful for:

- Creating paired samples for contrastive learning
- Implementing data augmentation strategies
- Combining different types of inputs

```python test
# Create two streams: one for images, one for labels
datastream1 = Datastream(Dataset.from_subscriptable([1, 2]))  # e.g., image IDs
datastream2 = Datastream(Dataset.from_subscriptable([3, 4]))  # e.g., augmentation params

# Get all combinations of images and augmentations
zipped = Datastream.zip([datastream1, datastream2])
samples = list(zipped)
assert len(samples) == 4  # All combinations: (1,3), (1,4), (2,3), (2,4)
```
