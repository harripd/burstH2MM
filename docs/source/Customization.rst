|H2MM| customization
====================

Customizing divisors
--------------------

There are two methods for defining new :ref:`divisor schemes <divisor>`.

#. |auto_div| - high level method
#. |new_div| - low level method

.. note::

    |auto_div| acutally calls |new_div|, but simplifies the specification of divisors to the end user.

So let's see it in action:

    name = bdata.auto_div(2)

So where are the divisors in this system?
In all streams, there are 2 divisors (3 indices per stream), and they equally divide the nanotimes into these indices.

|auto_div| offers the option to add one more level of granularity::

    name = bdata.auto_div([2,1,1])

So what will this do?
Now, the number of divisors is specified *per stream*, meaning the |DD| stream will have 2 divisors, while the |DA| and |AA| streams will have only 1 divisor.
The even distribution of nanotimes between the divisors will however be maintained.

Finally, |new_div| offers the greatest granularity, but also requires the most work by the user.
When using |new_div|, you must specify the nanotime divisors themselves.

The function call looks like this::

    divs = [np.array([2500]), np.array([2300]), np.array([800])]
    name = bdata.new_div(divs)

Customizing optimizations
-------------------------

Blah blah blah

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