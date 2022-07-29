Access the results of an optimization
-------------------------------------

|H2MM| is at its core an optimization algorithm.
The underlying algorithm finds the most likely model for a predetermined number of states.
It is the user who is responsible for comparing optimized models with different numbers of states and selecting the ideal model.

.. note::
    For this tutorial, we will assume the following code has been executed prior to all given code snippets::

        # import statements
        import numpy as np
        from matplotlib import pyplot as plt
        import fretbursts as frb
        import burstH2MM as hmm
        sns = frb.init_notebook()
        # path to your file
        filename = 'your_file.hdf5'
        # load data into fretbursts
        # load the data into the data object frbdata
        frbdata = frb.loader.photon_hdf5(filename)
        frb.loader.alex_apply_period(frbdata)
        # calculate background counts
        frbdata.calc_bg(frb.bg.exp_fit, F_bg=1.7)
        # now perform burst search
        frbdata.burst_search(m=10, F=6)
        # make sure to set the appropriate thresholds of ALL size
        # parameters to the particulars of your experiment
        frbdata_sel = frbdata.select_bursts(frb.select_bursts.size, th1=50)
        # now make the BurstData object
        bdata = hmm.BurstData(frbdata_sel)

|calc_models| is designed to streamline this process, which optimizes models until the ideal model is found (it actually calculates one more than necessary because it must see that there is at least one model with too many models).

So before |calc_models| is called on an |H2MM_list| object, it has no optimizations associated with it.
After |calc_models| is called, then several different state models are stored in the |opts| attribute, but are also accesible by indexing |H2MM_list|.

>>> bdata.models[0]
AssertionError: No optimizations run, must run calc_models or optimize for H2MM_result objects to exist

So, we first need to run |calc_models| and we can now access the indices.

>>> bdata.models.calc_models()
The model converged after 2 iterations
The model converged after 36 iterations
The model converged after 125 iterations
The model converged after 405 iterations
>>> bdata.models[0]
<burstH2MM.BurstSort.H2MM_result at 0x7f0ebed12eb0>

The elements within |H2MM_list| are, as you see above, |H2MM_result| objects.
These serve to organize both the optimized models, and various results we will discuss later in this section.
But the first thing to examine are the statistical discriminators.
These take the loglikelihood of the model given the data, and add penalties for the number of states.
There are 2 primary discriminators:

#. BIC: the Bayes Information Criterion
    - Based on the likelihood of the model over all possible paths through the data, usually found to always improve with more states, and therefore less usefull
#. ICL: The integrated Complete Likelihood
    - Based on the likelihood of the most likely path through the data for the model given the data, usually found to be minimized for the ideal model

In both cases, the smaller the better.
Since these are computed for each optimized model, each |H2MM_result| object (index of |H2MM_list|), has an attribute to get this value.
However, since these values are generally only useful in relation to other models optimized against the same data, the |H2MM_result object has its own attribute to return an array with the values of all the optimized models:

#. BIC:
    - |result_bic|

        >>> bdata.models[0].bic
        323390.7888996974

    - |list_bic|

        >>> bdata.models.BIC
        array([323390.7888997 , 274041.04580956, 266584.73431314, 265800.67189548])

#. ICL:
    - |result_icl|

        >>> bdata.models[0].icl
        323390.78889956

    - |list_icl|

        >>> bdata.models.ICL
        array([323390.78889956, 275340.11281888, 269885.74672888, 271100.67269877])

burstH2MM has an easy way to compare these::

    # give ICL_plot a H2MM_list object
    hmm.ICL_plot(bdata.models)

.. image:: images/iclplot.png

Note that we do not index `bdata.models` because this is comparing the different state models, not looking at a single model.

So now that we know how to select the model, what actually composes a |H2MM| model?
There are three key components:

#. Initial probability matrix (the *.prior* matrix)
    - The likelihood of a burst beginning in a given state
#. Observation probability matrix (the *.obs* matrix)
    - Contains probability a given state will emit a photon in a given index, from this the E and S values can be calculated.
#. Transition probability matrix (the *.trans* matrix)
    - The rate at which each state transitions to the others. This indicates the rate of transitions, characterizing the thermodynamic stability of each state (assuming you are using a model that has an appropriate number of states, over and underfit models will obviously have transition rates not reflective of the actual dynamics).

.. note::

    Note that the attribute names for the statistical discriminators from |H2MM_list| object use captital letters, while |H2MM_result| objects are use lowercase letters.

The initial probability matrix does not have a clear physical meaning, but the observation probability and transition probability matrices contain very valuable information.
burstH2MM automatically converts the values in these from the abstract units of the core algorithm into more human-friendly units (E/S values and transition rates in seconds).

E and S can be accessed with the attributes |model_E| and |model_S|

>>> bdata.models[2].E
array([0.66031034, 0.15955158, 0.06730048])

>>> bdata.models[2].E
array([0.43073408, 0.55348988, 0.9708039 ])

The above values are the raw values, if you want to have them corrected for leakage, direct excitation, and the beta and gamma values, you can access them by adding `_corr` to the attribute name, to get |model_E_corr| and |model_S_corr|

The transition rates are accessed through the |model_trans| attributes.

>>> bdata.models[2].trans
array([[1.99994147e+07, 5.31727465e+02, 5.35447960e+01],
       [2.05278839e+02, 1.99996914e+07, 1.03279378e+02],
       [7.90898846e+00, 1.16271335e+02, 1.99998758e+07]])

These are in s\ :sup:`-1`\  and teh organization is [from state, to state]. Notice that the diagonal is all very large values, this is because the diagonal represents the probability that the system remains in the same state from one time step to the next, as the time steps are in the clock rate of the acquisiation (typically 20 mHz, meaning 50 ns from one time step to the next) this is a very large number.

Now |H2MM| also contains the *Viterbi* algorithm, which takes the data and optimized model, and finds the most likely state of each photon.
burstH2MM continues to perform analysis on this state path to produce a number of usefull parameters to help understand the data.


Bellow is a list and desciption of the different possible parameters and their descriptions.

+----------------------+----------------------------------------------------------------+--------------+
| Attribute            | Description                                                    | Type         |
+----------------------+----------------------------------------------------------------+--------------+
| |nanohist|           | Number of photons in each state and TCSPC bin                  | state stream |
|                      |                                                                | nanotime     |
|                      |                                                                | array        |
+----------------------+----------------------------------------------------------------+--------------+
| |trans_locs|         | The location of transitions with bursts                        | burst list   |
+----------------------+----------------------------------------------------------------+--------------+
| |burst_dwell_num|    | Duration of each dwell (in ms)                                 | dwell array  |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_state|        | The state of each dwell                                        | dwell array  |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_pos|          | Numerical indicator of location within the                     | dwell array  |
|                      | burst of each dwell | dwell array                              |              |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_ph_counts|    | Number of photons in each stream and dwell                     | state dwell  |
|                      |                                                                | array        |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_ph_counts_bg| | Background corrected number                                    | dwell state  |
|                      | of photons in each stream and dwell                            | array        |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_E|            | Raw FRET efficiency (E\ :sup:`raw`\ ) of each dwell            | dwell array  |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_E_corr|       | Fully corrected FRET efficiency of each dwell (E)              | dwell array  |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_S|            | Raw stoichiometry (S\ :sup:`raw`\ ) of each dwell              | dwell array  |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_S_corr|       | Fully corrected stoichiometry of each dwell (S)                | dwell array  |
+----------------------+----------------------------------------------------------------+--------------+
| |dwell_nano_mean|    | Mean nanotime of each stream in each dwell                     | stream dwell |
|                      |                                                                | array        |
+----------------------+----------------------------------------------------------------+--------------+

.. |H2MM| replace:: H\ :sup:`2`\ MM
.. |DD| replace:: D\ :sub:`ex`\ D\ :sub:`em`
.. |DA| replace:: D\ :sub:`ex`\ A\ :sub:`em`
.. |AA| replace:: A\ :sub:`ex`\ A\ :sub:`em`
.. |BurstData| replace:: :class:`BurstData <burstH2MM.BurstSort.BurstData>`
.. |div_models| replace:: :attr:`BurstData.div_models <burstH2MM.BurstSort.BurstData.div_models>`
.. |auto_div| replace:: :meth:`BurstData.auto_div() <burstH2MM.BurstSort.BurstData.auto_div>`
.. |new_div| replace:: :meth:`BurstData.new_div() <burstH2MM.BurstSort.BurstData.new_div>`
.. |H2MM_list| replace:: :class:`H2MM_list <burstH2MM.BurstSort.H2MM_list>`
.. |list_bic| replace:: :attr:`H2MM_list.BIC <burstH2MM.BurstSort.H2MM_list.BIC>`
.. |list_bicp| replace:: :attr:`H2MM_list.BICp <burstH2MM.BurstSort.H2MM_list.BICp>`
.. |list_icl| replace:: :attr:`H2MM_list.ICL <burstH2MM.BurstSort.H2MM_list.ICL>`
.. |calc_models| replace:: :meth:`H2MM_list <burstH2MM.BurstSort.H2MM_list.calc_models>`
.. |opts| replace:: :attr:`H2MM_list.opts <burstH2MM.BurstSort.H2MM_list.opts>`
.. |H2MM_result| replace:: :class:`H2MM_result <burstH2MM.BurstSort.H2MM_result>`
.. |model_E| replace:: :attr:`H2MM_result.E <burstH2MM.BurstSort.H2MM_result.E>`
.. |model_E_corr| replace:: :attr:`H2MM_result.E_corr <burstH2MM.BurstSort.H2MM_result.E_corr>`
.. |model_S| replace:: :attr:`H2MM_result.S <burstH2MM.BurstSort.H2MM_result.S>`
.. |model_S_corr| replace:: :attr:`H2MM_result.S_corr <burstH2MM.BurstSort.H2MM_result.S_corr>`
.. |model_trans| replace:: :attr:`H2MM_result.trans <burstH2MM.BurstSort.H2MM_result.trans>`
.. |nanohist| replace:: :attr:`H2MM_result.nanohist <burstH2MM.BurstSort.H2MM_result.nanohist>`
.. |dwell_pos| replace:: :attr:`H2MM_result.dwell_pos <burstH2MM.BurstSort.H2MM_result.dwell_pos>`
.. |dwell_dur| replace:: :attr:`H2MM_result.dwell_dur <burstH2MM.BurstSort.H2MM_result.dwell_dur>`
.. |dwell_state| replace:: :attr:`H2MM_result.dwell_state <burstH2MM.BurstSort.H2MM_result.dwell_state>`
.. |dwell_ph_counts| replace:: :attr:`H2MM_result.dwell_ph_counts <burstH2MM.BurstSort.H2MM_result.dwell_ph_counts>`
.. |dwell_ph_counts_bg| replace:: :attr:`H2MM_result.dwell_ph_counts_bg <burstH2MM.BurstSort.H2MM_result.dwell_ph_counts_bg>`
.. |dwell_E| replace:: :attr:`H2MM_result.dwell_E <burstH2MM.BurstSort.H2MM_result.dwell_E>`
.. |dwell_E_corr| replace:: :attr:`H2MM_result.dwell_E_corr <burstH2MM.BurstSort.H2MM_result.dwell_E_corr>`
.. |dwell_S| replace:: :attr:`H2MM_result.dwell_S <burstH2MM.BurstSort.H2MM_result.dwell_S>`
.. |dwell_S_corr| replace:: :attr:`H2MM_result.dwell_S_corr <burstH2MM.BurstSort.H2MM_result.dwell_S_corr>`
.. |burst_dwell_num| replace:: :attr:`H2MM_result.burst_dwell_num <burstH2MM.BurstSort.H2MM_result.burst_dwell_num>`
.. |dwell_nano_mean| replace:: :attr:`H2MM_result.dwell_nano_mean <burstH2MM.BurstSort.H2MM_result.dwell_nano_mean>`
.. |trans_locs| replace:: :attr:`H2MM_result.trans_locs <burstH2MM.BurstSort.H2MM_result.trans_locs>`
.. |result_bic| replace:: :attr:`H2MM_result.bic <burstH2MM.BurstSort.H2MM_result.bic>`
.. |result_bicp| replace:: :attr:`H2MM_result.bicp <burstH2MM.BurstSort.H2MM_result.bicp>`
.. |result_icl| replace:: :attr:`H2MM_result.icl <burstH2MM.BurstSort.H2MM_result.icl>`
.. |dwell_ES_scatter| replace:: :func:`dwell_ES_scatter() <burstH2MM.Plotting.dwell_ES_scatter>`
.. |dwell_tau_hist| replace:: :func:`dwell_tau_hist() <burstH2MM.Plotting.dwell_tau_hist>`
.. |dwell_E_hist| replace:: :func:`dwell_E_hist() <burstH2MM.Plotting.dwell_E_hist>`