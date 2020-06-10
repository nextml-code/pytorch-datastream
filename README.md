# Pytorch Datastream

The main components of this library are:
- Datastream
- Dataset

A `Dataset` is a mapping that allows pipelining of functions in a readable syntax and `Datastream` combines a dataset and a sampler into a stream of examples.        

## Install

    pip install pytorch-datastream

## Usage

The list below is meant to showcase functions that are useful in most standard and non-standard cases. It is not meant to be an exhaustive list.

    Dataset.from_subscriptable
    Dataset.from_dataframe
    Dataset
        .map
        .subset

    Datastream.merge
    Datastream.zip
    Datastream
        .map
        .data_loader
        .zip_index
        .update_weights_
        .update_example_weight_
        .weight
        .state_dict
        .load_state_dict
        .multi_sample
        .sample_proportion

### Dataset from subscriptable

    from datastream import Dataset

    fruits_and_cost = (
        ('apple', 5),
        ('pear', 7),
        ('banana', 14),
        ('kiwi', 100),
    )

    dataset = (
        Dataset.from_subscriptable(fruits_and_cost)
        .map(lambda fruit, cost: (
            fruit,
            cost * 2,
        ))
    )

    print(dataset[2]) # ('banana', 28)

### Dataset from pandas dataframe

    from PIL import Image
    from imgaug import augmenters as iaa
    from datastream import Dataset

    augmenter = iaa.Sequential([...])

    def preprocess(image, class_names):
        ...

    dataset = (
        Dataset.from_dataframe(df)
        .map(lambda row: (
            row['image_path'],
            row['class_names'],
        ))
        .map(lambda image_path, class_names: (
            Image.open(image_path),
            class_names,
        ))
        .map(lambda image, class_names: (
            augmenter.augment(image=image),
            class_names,
        ))
        .map(preprocess)
    )

### Datastream to pytorch data loader

    data_loader = (
        Datastream(dataset)
        .data_loader(
            batch_size=32,
            num_workers=8,
            n_batches_per_epoch=100,
        )
    )

### Datastream to pytorch data loader for evaluation

    evaluate_data_loader = (
        Datastream(dataset, torch.utils.data.SequentialSampler())
        .data_loader(
            batch_size=32,
            num_workers=8,
        )
    )

### Merge / stratified / oversampled datastreams

    datastream = Datastream.merge([
        (datastream1, 2),
        (datastream2, 1),
        (datastream3, 1),
    ])

### Weighted datastreams

    datastream = (
        Datastream(dataset)
        .zip_index()
    )

    data_loader = datastream.data_loader(...)

    for indices, batch in data_loader:
        ...

        for index in indices:
            datastream.update_weight(index, example_loss.exp())

### Unsupervised weighted datastreams

    datastream = (
        Datastream(dataset)
        .zip_index()
        .multi_sample(N_CLASSES)
        .sample_proportion(0.01)
    )

    data_loader = datastream.data_loader(...)

    for indices, batch in data_loader:
        ...

        for index in indices:
            datastream.update_weight(index, predicted_classes)

### Decaying datastream weights

    DECAY_FACTOR = 0.999

    datastream.update_weights(lambda weights: (
        weights * DECAY_FACTOR
        + weights.mean() * (1 - DECAY_FACTOR)
    ))
