# Dataset

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

## Class Methods

### from_subscriptable

Create `Dataset` based on subscriptable i.e. implements `__getitem__` and `__len__`.

Should only be used for simple examples as a `Dataset` created with this method does not support methods that require a source dataframe like `Dataset.split` and `Dataset.subset`.

### from_dataframe

Create `Dataset` based on `pandas.DataFrame`. `Dataset.__getitem__` will return a row from the dataframe and `Dataset.map` should be given a function that takes a row from the dataframe as input.

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

### from_paths

Create `Dataset` from paths using regex pattern that extracts information from the path itself.
`Dataset.__getitem__` will return a row from the dataframe and `Dataset.map` should be given a function that takes a row from the dataframe as input.

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

### map

Creates a new dataset with the function added to the dataset pipeline.

```python
from datastream import Dataset

dataset = (
    Dataset.from_subscriptable([1, 2, 3])
    .map(lambda number: number + 1)
)

assert dataset[-1] == 4
```

### starmap

Creates a new dataset with the function added to the dataset pipeline.
The dataset's pipeline should return an iterable that will be expanded as arguments to the mapped function.

```python
from datastream import Dataset

dataset = (
    Dataset.from_subscriptable([1, 2, 3])
    .map(lambda number: (number, number + 1))
    .starmap(lambda number, plus_one: number + plus_one)
)

assert dataset[-1] == 7
```

### subset

Select a subset of the dataset using a function that receives the source dataframe as input and is expected to return a boolean mask.

Note that this function can still be called after multiple operations such as mapping functions as it uses the source dataframe.

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

### split

Split dataset into multiple parts. Optionally you can stratify on a column in the source dataframe or save the split to a json file.
If you are sure that the split strategy will not change then you can safely use a seed instead of a filepath.

Saved splits can continue from the old split and handle:

- New examples
- Changing test size
- Adapt after removing examples from dataset
- Adapt to new stratification

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

### zip_index

Zip the output with its underlying Dataset index. The output of the pipeline will be a tuple `(output, index)`.

```python
from datastream import Dataset

dataset = Dataset.from_subscriptable([4, 5, 6]).zip_index()
assert dataset[0] == (4, 0)
```

### cache

Cache intermediate step in-memory based on key column.

```python
import pandas as pd
from datastream import Dataset

df = pd.DataFrame({'key': ['a', 'b'], 'value': [1, 2]})
dataset = Dataset.from_dataframe(df).cache('key')
assert dataset[0]['value'] == 1
```

### concat

Concatenate multiple datasets together so that they behave like a single dataset.

Consider using `Datastream.merge` if you have multiple data sources instead as it allows you to control the number of samples from each source in the training batches.

```python
from datastream import Dataset

dataset1 = Dataset.from_subscriptable([1, 2])
dataset2 = Dataset.from_subscriptable([3, 4])
combined = Dataset.concat([dataset1, dataset2])
assert len(combined) == 4
assert combined[2] == 3
```

### combine

Zip multiple datasets together so that all combinations of examples are possible (i.e. the product) creating tuples like `(example1, example2, ...)`.

```python
from datastream import Dataset

dataset1 = Dataset.from_subscriptable([1, 2])
dataset2 = Dataset.from_subscriptable([3, 4])
combined = Dataset.combine([dataset1, dataset2])
assert len(combined) == 4  # 2 * 2 = 4 combinations
assert combined[0] == (1, 3)  # First combination
```
