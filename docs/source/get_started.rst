===========
Get started
===========

Installation
============
To download and install the library from pypi simply execute:

``pip install pytorch-datastream``

Usage
=====

Dataset from subscriptable
--------------------------
Simple usage with ``Dataset.from_subscriptable``. This is mostly useful for
simple examples. It is often preferable to use ``Dataset.from_dataframe``.

.. highlight:: python
.. code-block:: python

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

Dataset from pandas dataframe
-----------------------------
This example tries to show a simple data pipeline in pseudo-code where a
dataset is is created from a dataframe, then images are read from disk,
augmented, and preprocessed before training.

.. highlight:: python
.. code-block:: python

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

Datastream to pytorch data loader
---------------------------------
The final step of converting the datastream to a ``torch.data.util.DataLoader``
before using it in your training / evaluation loop. You can specify an
alternative epoch length if you do not want it to be defined by the dataset.
This is very useful when oversampling or weighting because epoch length quickly
loses its meaning then.

.. highlight:: python
.. code-block:: python

    data_loader = (
        Datastream(dataset)
        .data_loader(
            batch_size=32,
            num_workers=8,
            n_batches_per_epoch=100,
        )
    )

Datastream to pytorch data loader for evaluation
------------------------------------------------
You can optionally specify your own sampler when creating a datastream.
In this case we specify ``torch.utils.data.SequentialSampler`` which will give
us a very minor boost in speed when evaluating but we lose the ability to
sample by weight.

.. highlight:: python
.. code-block:: python

    evaluate_data_loader = (
        Datastream(dataset, torch.utils.data.SequentialSampler())
        .data_loader(
            batch_size=32,
            num_workers=8,
        )
    )

Merge / stratify / oversample datastreams
-----------------------------------------
It is very common to have imbalanced datasets or multiple data sources of very
different length and dissimilar characteristics. ``Datastream.merge`` provides
a simple intuitive way to construct batches that give a good training signal
in these cases.

.. highlight:: python
.. code-block:: python

    datastream = Datastream.merge([
        (datastream1, 2),
        (datastream2, 1),
        (datastream3, 1),
    ])

Weighted datastreams
--------------------
You can change the weights of different examples if you e.g. want to focus
more on learning to handle the difficult examples rather than the easy ones
that might give near zero loss.

.. highlight:: python
.. code-block:: python

    datastream = (
        Datastream(dataset)
        .sample_proportion(0.5)
        .zip_index()
    )

    data_loader = datastream.data_loader(...)

    for indices, batch in data_loader:
        ...

        for index in indices:
            datastream.update_weight_(index, example_loss.exp())

Unsupervised weighted datastreams
---------------------------------
Weighting can be applied dynamically based on model guessing which makes it a
good candidate for unsupervised stratification. We can for example try to
create batches with an equal number of examples from each class based on
the model's predictions as shown below:

.. highlight:: python
.. code-block:: python

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
            datastream.update_weight_(index, predicted_classes)

Decaying datastream weights
---------------------------
It can be useful to modify all the sample weights at the same time. In this
case we are letting the sample weights decay to the mean during training
as the prediction grows older.

.. highlight:: python
.. code-block:: python

    DECAY_FACTOR = 0.999

    datastream.update_weights_(lambda weights: (
        weights * DECAY_FACTOR
        + weights.mean() * (1 - DECAY_FACTOR)
    ))
