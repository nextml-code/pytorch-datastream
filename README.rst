==================
Pytorch Datastream
==================

This is a simple library for creating very readable dataset pipelines and
reusing best practices for issues such as imbalanced datasets. There are
just two components to keep track of: ``Dataset`` and ``Datastream``.

``Dataset`` is a simple mapping between an index and an example. It provides 
pipelining of functions in a very readable syntax originally adapted from
tensorflow 2's ``tf.data.Dataset``.

``Datastream`` combines a ``Dataset`` and a sampler into a stream of examples.
It provides a simple solution to oversampling / stratification, weighted
sampling, and finally converting to a ``torch.utils.data.DataLoader``.

Install
=======

.. code-block::

    pip install pytorch-datastream

Usage
=====

The list below is meant to showcase functions that are useful in most standard
and non-standard cases. It is not meant to be an exhaustive list. See the 
`documentation <https://pytorch-datastream.readthedocs.io/en/latest/>`_ for 
a more extensive list on API and usage.

.. code-block:: python

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

Dataset from subscriptable
--------------------------

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

.. code-block:: python

    data_loader = (
        Datastream(dataset)
        .data_loader(
            batch_size=32,
            num_workers=8,
            n_batches_per_epoch=100,
        )
    )

More usage examples
-------------------

See the `documentation <https://pytorch-datastream.readthedocs.io/en/latest/>`_
for examples with oversampling / stratification and weighted sampling.
