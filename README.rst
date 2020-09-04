==================
Pytorch Datastream
==================

.. image:: https://badge.fury.io/py/pytorch-datastream.svg
       :target: https://badge.fury.io/py/pytorch-datastream

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
