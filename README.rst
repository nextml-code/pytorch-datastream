==================
Pytorch Datastream
==================

.. image:: https://badge.fury.io/py/pytorch-datastream.svg
       :target: https://badge.fury.io/py/pytorch-datastream

.. image:: https://img.shields.io/pypi/pyversions/pytorch-datastream.svg
       :target: https://pypi.python.org/pypi/pytorch-datastream

.. image:: https://readthedocs.org/projects/pytorch-datastream/badge/?version=latest
       :target: https://pytorch-datastream.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/l/pytorch-datastream.svg
       :target: https://pypi.python.org/pypi/pytorch-datastream



This is a simple library for creating readable dataset pipelines and
reusing best practices for issues such as imbalanced datasets. There are
just two components to keep track of: ``Dataset`` and ``Datastream``.

``Dataset`` is a simple mapping between an index and an example. It provides 
pipelining of functions in a readable syntax originally adapted from
tensorflow 2's ``tf.data.Dataset``.

``Datastream`` combines a ``Dataset`` and a sampler into a stream of examples.
It provides a simple solution to oversampling / stratification, weighted
sampling, and finally converting to a ``torch.utils.data.DataLoader``.


Install
=======

.. code-block::

    poetry add pytorch-datastream

Or, for the old-timers:

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
        .split
        .cache
        .with_columns

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


Simple image dataset example
----------------------------
Here's a basic example of loading images from a directory:

.. code-block:: python

    from datastream import Dataset
    from pathlib import Path
    from PIL import Image

    # Assuming images are in a directory structure like:
    # images/
    #   class1/
    #     image1.jpg
    #     image2.jpg
    #   class2/
    #     image3.jpg
    #     image4.jpg

    image_dir = Path("images")
    image_paths = list(image_dir.glob("**/*.jpg"))

    dataset = (
        Dataset.from_paths(image_paths, pattern=r".*/(?P<class_name>\w+)/(?P<image_name>\w+).jpg")
        .map(lambda row: dict(
            image=Image.open(row["path"]),
            class_name=row["class_name"],
            image_name=row["image_name"],
        ))
    )

    # Access an item from the dataset
    first_item = dataset[0]
    print(f"Class: {first_item['class_name']}, Image name: {first_item['image_name']}")


Merge / stratify / oversample datastreams
-----------------------------------------
The fruit datastreams given below repeatedly yields the string of its fruit
type.

.. code-block:: python

    >>> datastream = Datastream.merge([
    ...     (apple_datastream, 2),
    ...     (pear_datastream, 1),
    ...     (banana_datastream, 1),
    ... ])
    >>> next(iter(datastream.data_loader(batch_size=8)))
    ['apple', 'apple', 'pear', 'banana', 'apple', 'apple', 'pear', 'banana']


Zip independently sampled datastreams
-------------------------------------
The fruit datastreams given below repeatedly yields the string of its fruit
type.

.. code-block:: python

    >>> datastream = Datastream.zip([
    ...     apple_datastream,
    ...     Datastream.merge([pear_datastream, banana_datastream]),
    ... ])
    >>> next(iter(datastream.data_loader(batch_size=4)))
    [('apple', 'pear'), ('apple', 'banana'), ('apple', 'pear'), ('apple', 'banana')]


More usage examples
-------------------
See the `documentation <https://pytorch-datastream.readthedocs.io/en/latest/>`_
for more usage examples.
