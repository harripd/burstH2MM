#########
burstH2MM
#########

.. contents::



*************
Introduction
*************

burstH2MM is a package designed to make processing FRETBursts data with photon-by-photon hidden Markov Modeling (H2MM) easy to use, and to calculate things such as E, S, transition rates, and other dwell and model parameters, removing the need to manually recode such details each time.

The basic unit of burstH2MM is:

    :BurstData: The container of a  s set of photon streams.

    :H2MM_list: The container for a divisor scheme

    :H2MM_model: A H2MM optimization.

This is contains most analysis parameters. Things such as dwell E, dwell S, and dwell mean nanotime

While each class can be assigned to a new variable or list, all three classes keep a record of the object they create, and the object that created them. Therefore it is generally encouraged to simply work with the originating BurstData object and refer to all subfield through it following the appropirate referencing of attributes.

=================
Tutorial
=================

To make burst data, first perform analysis with FRETBursts as you normally would. (Background analysis -> burst selection -> burst filtration, with appropriate checks along the way to ensure that the data is of appropriate quality.

**Burst filtration** is generally necessary because H2MM should have for each burst a decent number of photons (20 to 30 minimum, but 50 is ideal)

First we do the pre-H2MM analysis with FRETBursts::

    import burstH2MM as hmm
    import fretbursts as frb
    import burstH2MM as hmm
    # load data
    d = frb.loader.photon_hdf5('HP3_300NA_SPC630.hdf5')
    frb.loader.alex_apply_period(d)
    # Burst search
    d.calc_bg(frb.bg.exp_fit, t=30, F=1.7)
    d.burst_search()
    d = d.fuse_bursts(ms=0)
    # burst selection
    d_sel = d.select_bursts(frb.select_bursts.size, th1=30)

Now that bursts have been selected and refined, it is now possible to analyze the data with H2MM, so we take the FRETBursts data object and use it to make BurstData object in burstH2MM. This BurstData object will be the basis for all subsequent analysis::

    # transfer data to burstH2MM
    hmm_d = hmm.BurstData(d_sel)

Now we can perform optimizations on this data::

    # run optimization
    hmm_d.models.calc_models()
    ideal = hmm_d.models.ideal # get best modeld of the optimization

Finally, let's plot the results::

    # now plot the result of the best model
    hmm.scatter_ES(hmm_d.models[ideal])

This is the test case. For all the additional work we need to do.

========
Plotting
========

.. note::
    This is a note on what is going on
    How well will this work?

This is outside the note