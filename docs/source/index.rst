.. burstH2MM documentation master file, created by
   sphinx-quickstart on Sun Jul 17 08:00:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

burstH2MM documentation
=======================

burstH2MM is designed to make analyzing single molecule burst experiments with |H2MM| easy.
burstH2MM is built to work with FRETBursts, on which it depends for burst search and selection.
burstH2MM serves to organize and process the results of |H2MM| optimizations and access the results of *Viterbi* analysis, which assigns each photon a state, and further processing segments the data into dwells.

Installation
------------

To install `burstH2MM` simply type

.. code-block:: bash

    pip install burstH2MM

into your terminal window.
Python 3.8 or greater is required.

Please Cite
-----------

|H2MM| was originally introduced by `Pirchi and Tsukanov et. al. 2016 <https://doi.org/10.1021/acs.jpcb.6b10726>`_ and this implementation is built on the extension introduced in `Harris et. al. 2022 <https://doi.org/10.1038/s41467-022-28632-x>`_. If you use this library in a publication, please make sure to cite both these papers.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Tutorial
   HowTo
   Discussion
   Documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Thank you for using |H2MM|

.. |H2MM| replace:: H\ :sup:`2`\ MM
