Welcome to pytorch-datastream's documentation!
==============================================

This is a simple library for creating readable dataset pipelines and reusing
best practices for issues such as imbalanced datasets. There are just two 
components to keep track of: ``Dataset`` and ``Datastream``.

``Dataset`` is a simple mapping between an index and an example. It provides 
pipelining of functions in a readable syntax originally adapted from
tensorflow 2's ``tf.data.Dataset``.

``Datastream`` combines a ``Dataset`` and a sampler into a stream of examples.
It provides a simple solution to oversampling / stratification, weighted
sampling, and finally converting to a ``torch.utils.data.DataLoader``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   get_started
   dataset
   datastream

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
