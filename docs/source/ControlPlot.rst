Control plotting functions
==========================

Customizing state plotting
--------------------------

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
        bdata.models.calc_models()

The central plotting functions of burstH2MM are all highly customizable.

The first and simplest form of customization is using the `ax` keyword argument, which is universal to all plotting functions.
This lets you make a `matplotlib axes <mpl_ax>`_ (usually made with `plt.subplots() <plt_subplots>`_ or related functions), and then plot all elements within that axes.
This also lets overlaping differnt plots into one axes.

So here's an example, and we'll set the title of the axes afterward::

    # make the axes
    fig, ax = plt.subplots(figsize=(5,5))
    hmm.dwell_ES_scatter(bdata.models[2], ax=ax)
    ax.set_title("Dwell Based Results")

.. image:: images/dwellES55.png

.. _by_state:

Customizing plots by state
--------------------------

Additional customizations focus on how states are individually plotted.
This is done by passing lists of keyword argument dictionaries to specific keyword argumnets.

The first of these keyword arguments we will explore is `state_kwargs`.
This is universall to all plotting functions begining with `dwell_`.
For it you pass a list of dictionaries of keyword arguments for the underlying matploplib function, one dictionary for each state.

Confusing, here's a simple example, where we assign a color to each state in the |dwell_ES_scatter| plot::

    # set up list, same length as number of states in the model
    state_color = [{'color':'m'}, {'color':'yellow'}, {'color':'c'}]
    hmm.dwell_ES_scatter(bdata.models[2], state_kwargs=state_color)

.. image:: images/cmyESscatter.png

So what happened here?
Since models[2] has 3 states, the input `state_kwargs` keyword argument needs to be a list or tuple of length 3.
States in a model have an order, established in the model itself.
Each element of the list is passed, *per state* to the maptloib `scatter() <plt_scatter>`_ function as \*\*kwargs.
So the first state gets the keyword arguemtn `color='m'`, the second state `color='yellow'` and the third `color='m'`.

.. note::

    The different plotting functions use different matplotlib and seaborn functions.
    So plotting fucntions that create histograms use `plt.hist() <plt_hist>`_, while scatter functions use `plt.scatter() <plt_scatter>`_, and kde plot functions use `sns.kdeplot() <sns_kdeplot>`_

Only displaying certain states
------------------------------

What if you want to only look at a few states?
You can select, and control the order of the plotting of different states with the `states` keyword argument.

Let's say we want to only look at the FRET states (which are the 0th and 1st states in sample data set, but might be different when you are using other datasets).
To do this, we make an array of just the indices of those states, and then pass that array to the `states` keyword argument::

    # make the axes
    fig, ax = plt.subplots(figsize=(5,5))

    # specify the states we want
    states = np.array([0, 1])

    # now we plot
    hmm.dwell_ES_scatter(bdata.models[2], ax=ax, states=states)
    ax.set_title("FRET states")

.. image:: images/dwellES55fret.png

Selecting states and controlling their plotting
***********************************************

So how do we combine the `states` and `state_kwargs`?
It's pretty simple, `states` serves like a "master", and so each state specified in `states` is matched with an element of `state_kwargs`, assuming they come *in the same order*.
So, basicaly specify `state_kwargs` dictionaries in the same order as the states you specify in `states`, and obviously, they need to be the same length, otherwise you will get an error.

So here's an example where we re-plot the FRET states, but in reverse order, and see how the `state_kwargs` are also reorderd::

    # make the axes
    fig, ax = plt.subplots(figsize=(5,5))

    # specify the states we want, now with 1 before 0
    states = np.array([1, 0])
    # make the state_kwargs, we'll add labels this time
    state_kwargs = [{'color':'yellow', 'label':'FRET state 1'}, {'color':'m', 'label':'FRET state 2}]

    # now we plot
    hmm.dwell_ES_scatter(bdata.models[2], ax=ax, states=states, state_kwargs=state_kwargs)

    # add title, and legend to the plot
    ax.set_title("FRET states")
    ax.legend()

.. image:: images/dwellES55fretcm.png

Selecting photon streams
------------------------

But what about the |dwell_nano_mean| parameter?
It has not only information per state, but also information per stream.
Some other dwell parameters are similar.
To select and/or specify a stream, we have the `streams` keyword argument, and the `stream_kwargs` keyword argument to customize those plotting for those functions as well.
For this we will use the |dwell_tau_hist| function.

So let's see the default appearance first::

    fig, ax = plt.subplots(figsize=(3, 5))
    hmm.dwell_tau_hist(bdata.models[2], ax=ax)

.. image:: images/dwellnthist.png

By default, |dwell_tau_hist| only shows the mean nanotimes for the |DD| photon stream.
But what if we wanted to look at a different stream?
To do this we use the `streams` keyword argument.
It functions like the :ref:`states <by_state>` keyword argument before.

So, let's look at the |DD| and |DA| streams::

    fig, ax = plt.subplots(figsize=(5, 3))
    streams = [frb.Ph_sel(Dex="Dem"), frb.Ph_sel(Dex="Aem")]
    hmm.dwell_tau_hist(models[2], ax=ax, streams=streams)

.. image: images/dwellnanomeanmulti.png

Or just the |DA| stream::

    fig, ax = plt.subplots(figsize=(5, 3))
    streams = [frb.Ph_sel(Dex="Aem")]
    hmm.dwell_tau_hist(models[2], ax=ax, streams=streams)

.. image: images/dwellnanomeanAA.png


Customizing plotting of photon streams
--------------------------------------

For plots where there are specific selections per stream in addition to per state, the `stream_kwargs` keyword argument extists.
It functions much like the `state_kwargs` argument, matching the order of `streams` and needing to be the same length.

Also, `state_kwargs` and `stream_kwargs` merge dictionaries, so you can specify both, and not have a problem.

So let's see an example::

    fig, ax = plt.subplots(figsize=(5, 3))
    streams = [frb.Ph_sel(Dex="Dem"), frb.Ph_sel(Dex="Aem")]
    stream_kw = [{'color':'b'}, {'color':'r'}]
    hmm.dwell_tau_hist(models[2], ax=ax, streams=streams, stream_kwargs=stream_kw)

.. image:: images/dwellnanomeancbystream.png

But now, the problem is we have no idea which state goes with what, so let's use the `states` keyword argument to specify only the 0th state::

    fig, ax = plt.subplots(figsize=(5, 3))
    streams = [frb.Ph_sel(Dex="Dem"), frb.Ph_sel(Dex="Aem")]
    stream_kw = [{'color':'b'}, {'color':'r'}]
    state = np.array([0])
    hmm.dwell_tau_hist(models[2], ax=ax, streams=streams, stream_kwargs=stream_kw, states=state)

.. image:: images/dwellnanomean1scbstream.png

Finally, `stream_kwargs` and `state_kwargs` work together, the two dictionaries for a particular stream and state combination are merged::

    fig, ax = plt.subplots(figsize=(5, 3))
    streams = [frb.Ph_sel(Dex="Dem"), frb.Ph_sel(Dex="Aem")]
    stream_kw = [{'color':'b'}, {'color':'r'}]
    state_kw = [{'edgecolor':'darkblue'}, {'edgecolor':'darkorange'}, {'edgecolor':'olive'}]
    hmm.dwell_tau_hist(models[2], ax=ax, streams=streams, stream_kwargs=stream_kw, state_kwargs=state_kw)


.. note::

    In the dictionary merging process, if the same key is present in both dictionaries, then the value in `stream_kwargs` will be used, and the values in `state_kwargs` over-written.

.. image:: images/dwellnanomeansskw.png

Plotting state and stream specific plotting in one array
--------------------------------------------------------

Now, sometimes you need even more control, because the two keyworkd argument arrays clash.
For this there is the `kwarg_arr` keyword argument.
In `kwarg_arr`, you provide an array of dictionaries that will be the keyword arguments for `scatter() <plt_scatter>`_, the outer dimention indicates which state, the inner, the stream.

.. note::

    `kwarg_arr` is mean to take the place of the combination of `state_kwargs` and `stream_kwargs`
    As such, if `kwarg_arr` and `state_kwargs` cannot be specified at the same time.
    If `stream_kwargs` is specified at the same time as `kwarg_arr`, then burstH2MM will make a check.
    If `kwarg_arr` is formated like `state_kwargs`, then it will be treated like `state_kwargs`.
    On the other hand, if it is formated as demosntrated bellow, `stream_kwargs` will be ignored, and a warning will be presented.

::

    fig, ax = plt.subplots(figsize=(6, 4))
    kwarr = [[{'color':'g', 'label':'State 0, DexDem'}, 
              {'color':'darkgreen', 'label':'State 0, DexDem'}], 
             [{'color':'r', 'label':'State 1, DexDem'}, 
              {'color':'darkred', 'label':'State1, DexAem'}], 
             [{'color':'b', 'label':'State 2, DexDem'}, 
              {'color':'darkblue', 'label':'State2, DexAem'}]]
    hmm.dwell_tau_hist(models[2], ax=ax, kwarg_arr=kwarr, streams=[frb.Ph_sel(Dex="Dem"), frb.Ph_sel(Aex="Aem")])
    ax.legend()


So `kwarg_arr` allows the most customization, but is also the longest to define.

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

.. _plt_scatter: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
.. _mpl_ax: https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes
.. _plt_subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=subplots#matplotlib.pyplot.subplots
.. _plt_hist: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
.. _sns_kdeplot: https://seaborn.pydata.org/generated/seaborn.kdeplot.html