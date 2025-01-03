# Dataset

A `Dataset[T]` is a mapping that allows pipelining of functions in a readable syntax returning an example of type `T`.

<!--pytest-codeblocks:importorskip(datastream)-->

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

## Class Methods

### `from_subscriptable`

```python
from_subscriptable(data: Subscriptable[T]) -> Dataset[T]
```

Create `Dataset` based on subscriptable i.e. implements `__getitem__` and `__len__`.

#### Parameters

- `data`: Any object that implements `__getitem__` and `__len__`

#### Returns

- A new Dataset instance

#### Notes

Should only be used for simple examples as a `Dataset` created with this method does not support methods that require a source dataframe like `Dataset.split` and `Dataset.subset`.

### `from_dataframe`

```python
from_dataframe(df: pd.DataFrame) -> Dataset[pd.Series]
```

Create `Dataset` based on `pandas.DataFrame`.

#### Parameters

- `df`: Source pandas DataFrame

#### Returns

- A new Dataset instance where `__getitem__` returns a row from the dataframe

#### Notes

`Dataset.map` should be given a function that takes a row from the dataframe as input.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
import pandas as pd
from datastream import Dataset

dataset = (
    Dataset.from_dataframe(pd.DataFrame(dict(
        number=[1, 2, 3]
    )))
    .map(lambda row: row['number'] + 1)
)

assert dataset[-1] == 4
```

### `from_paths`

```python
from_paths(paths: List[str], pattern: str) -> Dataset[pd.Series]
```

Create `Dataset` from paths using regex pattern that extracts information from the path itself.

#### Parameters

- `paths`: List of file paths
- `pattern`: Regex pattern with named groups to extract information from paths

#### Returns

- A new Dataset instance where `__getitem__` returns a row from the generated dataframe

#### Notes

`Dataset.map` should be given a function that takes a row from the dataframe as input.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset

image_paths = ["dataset/damage/1.png"]
dataset = (
    Dataset.from_paths(image_paths, pattern=r".*/(?P<class_name>\w+)/(?P<index>\d+).png")
    .map(lambda row: row["class_name"])
)

assert dataset[-1] == 'damage'
```

## Instance Methods

### `map`

```python
map(self, function: Callable[[T], U]) -> Dataset[U]
```

Creates a new dataset with the function added to the dataset pipeline.

#### Parameters

- `function`: Function to apply to each example

#### Returns

- A new Dataset with the mapping function added to the pipeline

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset

dataset = (
    Dataset.from_subscriptable([1, 2, 3])
    .map(lambda number: number + 1)
)

assert dataset[-1] == 4
```

### `starmap`

```python
starmap(self, function: Callable[..., U]) -> Dataset[U]
```

Creates a new dataset with the function added to the dataset pipeline.

#### Parameters

- `function`: Function that accepts multiple arguments unpacked from the pipeline output

#### Returns

- A new Dataset with the mapping function added to the pipeline

#### Notes

The dataset's pipeline should return an iterable that will be expanded as arguments to the mapped function.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset

dataset = (
    Dataset.from_subscriptable([1, 2, 3])
    .map(lambda number: (number, number + 1))
    .starmap(lambda number, plus_one: number + plus_one)
)

assert dataset[-1] == 7
```

### `subset`

```python
subset(self, function: Callable[[pd.DataFrame], pd.Series]) -> Dataset[T]
```

Select a subset of the dataset using a function that receives the source dataframe as input.

#### Parameters

- `function`: Function that takes a DataFrame and returns a boolean mask

#### Returns

- A new Dataset containing only the selected examples

#### Notes

This function can still be called after multiple operations such as mapping functions as it uses the source dataframe.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
import pandas as pd
from datastream import Dataset

dataset = (
    Dataset.from_dataframe(pd.DataFrame(dict(
        number=[1, 2, 3]
    )))
    .map(lambda row: row['number'])
    .subset(lambda dataframe: dataframe['number'] <= 2)
)

assert dataset[-1] == 2
```

### `split`

```python
split(
    self,
    key_column: str,
    proportions: Dict[str, float],
    stratify_column: Optional[str] = None,
    filepath: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dataset[T]]
```

Split dataset into multiple parts.

#### Parameters

- `key_column`: Column to use as unique identifier for examples
- `proportions`: Dictionary mapping split names to proportions
- `stratify_column`: Optional column to use for stratification
- `filepath`: Optional path to save/load split configuration
- `seed`: Optional random seed for reproducibility

#### Returns

- Dictionary mapping split names to Dataset instances

#### Notes

Optionally you can stratify on a column in the source dataframe or save the split to a json file.
If you are sure that the split strategy will not change then you can safely use a seed instead of a filepath.

Saved splits can continue from the old split and handle:

- New examples
- Changing test size
- Adapt after removing examples from dataset
- Adapt to new stratification

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
import numpy as np
import pandas as pd
from datastream import Dataset

split_datasets = (
    Dataset.from_dataframe(pd.DataFrame(dict(
        index=np.arange(100),
        number=np.arange(100),
    )))
    .map(lambda row: row['number'])
    .split(
        key_column='index',
        proportions=dict(train=0.8, test=0.2),
        seed=700,
    )
)
assert len(split_datasets['train']) == 80
assert split_datasets['test'][0] == 3
```

### `zip_index`

```python
zip_index(self) -> Dataset[Tuple[T, int]]
```

Zip the output with its underlying Dataset index.

#### Returns

- A new Dataset where each example is a tuple of `(output, index)`

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset

dataset = Dataset.from_subscriptable([4, 5, 6]).zip_index()
assert dataset[0] == (4, 0)
```

### `cache`

```python
cache(self, key_column: str) -> Dataset[T]
```

Cache intermediate step in-memory based on key column.

#### Parameters

- `key_column`: Column to use as cache key

#### Returns

- A new Dataset with caching enabled

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
import pandas as pd
from datastream import Dataset

df = pd.DataFrame({'key': ['a', 'b'], 'value': [1, 2]})
dataset = Dataset.from_dataframe(df).cache('key')
assert dataset[0]['value'] == 1
```

### `concat`

```python
concat(datasets: List[Dataset[T]]) -> Dataset[T]
```

Concatenate multiple datasets together.

#### Parameters

- `datasets`: List of datasets to concatenate

#### Returns

- A new Dataset combining all input datasets

#### Notes

Consider using `Datastream.merge` if you have multiple data sources instead as it allows you to control the number of samples from each source in the training batches.

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset

dataset1 = Dataset.from_subscriptable([1, 2])
dataset2 = Dataset.from_subscriptable([3, 4])
combined = Dataset.concat([dataset1, dataset2])
assert len(combined) == 4
assert combined[2] == 3
```

### `combine`

```python
combine(datasets: List[Dataset]) -> Dataset[Tuple]
```

Zip multiple datasets together so that all combinations of examples are possible.

#### Parameters

- `datasets`: List of datasets to combine

#### Returns

- A new Dataset yielding tuples of all possible combinations

#### Notes

Creates tuples like `(example1, example2, ...)` for all possible combinations (i.e. the cartesian product).

#### Examples

<!--pytest-codeblocks:importorskip(datastream)-->

```python
from datastream import Dataset

dataset1 = Dataset.from_subscriptable([1, 2])
dataset2 = Dataset.from_subscriptable([3, 4])
combined = Dataset.combine([dataset1, dataset2])
assert len(combined) == 4  # 2 * 2 = 4 combinations
assert combined[0] == (1, 3)  # First combination
```
