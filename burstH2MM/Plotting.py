#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Module: Plotting
# Author: Paul David Harris
# Created; 29 Jun 2022
# Modified: 21 July 2022
# Purpose: plotting functions for burstH2MM
"""
.. _plotting:

Plotting
========

This section provides all the plotting functions for burstH2MM.
Most fuctions take a H2MM_result object as input, and customization is provided
through various keyword arguments.
"""

from itertools import cycle, repeat
import functools
import warnings
import numpy as np
import matplotlib.pyplot as plt
import fretbursts as frb
import seaborn as sns
from . import BurstSort


def _useideal(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        if isinstance(args[0], BurstSort.H2MM_list):
            assert hasattr(args[0], 'ideal'), ValueError("Ideal model not set, set with H2MM_list.ideal = ")
            args = list(args)
            args[0] = args[0][args[0].ideal]
        return func(*args, **kwargs)
    return wrap


def _check_ax(ax):
    """
    Hidden function to return a new axis if no axis is specified in the ax kwarg

    Parameters
    ----------
    ax : matplotlib.pyplot.axes or None
        Axis where the plot will be placed

    Returns
    -------
    nax : matplotlib.pyplot.axes
        The current axis if ax was None

    """
    if ax is None:
        nax = plt.gca()
    else:
        nax = ax
    return nax

def _update_ret(dicta, dictb):
    """
    Return dictionary resulting from updating dicta with dictb

    Parameters
    ----------
    dicta : dict
        Original dictionary
    dictb : dict
        dictionary with values to update/add to dicta

    Returns
    -------
    dct : dict
        Updated dictionary

    """
    dct = dicta.copy()
    dct.update(dictb)
    return dct
    

def _check_states(model, states):
    """
    Check that states kwarg is valid

    Parameters
    ----------
    model : H2MM_result
        H2MM_result of data.
    states : numpy.ndarray
        The states to be used.

    Returns
    -------
    states : numpy.ndarray
        States to be used.

    """
    if states is None:
        states = np.arange(model.model.nstate)
    elif isinstance(states, (list, int, tuple)):
        states = np.atleast_1d(states)
    if not np.all(states < model.model.nstate):
        ValueError("Cannot plot state {np.max(states)}, but model only has {model.model.nstate} states")
    return states

def _check_streams(model, streams):
    """
    Check that states kwarg is valid

    Parameters
    ----------
    model : H2MM_result
        H2MM_result of data.
    streams : list of fretbursts.Ph_sel
        The streams to be used.

    Returns
    -------
    streams : list of fretbursts.Ph_sel
        Streams to be used.

    """
    if streams is None:
        streams = model.parent.parent.ph_streams
    elif isinstance(streams, frb.Ph_sel):
        streams = [streams]
    in_stream = np.array([stream in model.parent.parent.ph_streams for stream in streams])
    if not np.all(in_stream):
        ValueError(f"Stream(s) {[stream for stream, in_s in zip(streams, in_stream) if not in_s]} not in BurstData")
    return streams


def _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr):
    """
    Check input kwargs of dwell_param function 

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    states : TYPE
        DESCRIPTION.
    streams : TYPE
        DESCRIPTION.
    state_kwargs : TYPE
        DESCRIPTION.
    stream_kwargs : TYPE
        DESCRIPTION.
    kwarg_arr : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        Two incompatible keyword arguments specified.
    ValueError
        Invalid value in keyword argument.

    Returns
    -------
    states : numpy.ndarray
        The states to be used.
    streams : list of fretbursts.Ph_sel
        The streams to be used.
    kwarg_arr : numpy.ndarray of dicts
        Keyword arguments for each state/stream combination.

    """
    states = _check_states(model, states)
    streams = _check_streams(model, streams)
    if state_kwargs is not None and kwarg_arr is not None:
        raise TypeError("Cannot specify both state_kwargs and kwarg_arr arguments at the same time")
    if kwarg_arr is None:
        if state_kwargs is None:
            state_kwargs = np.array([dict() for _ in states])
        if len(state_kwargs) != len(states):
            raise ValueError(f"Incompattible dimensions of states and state_kwargs, got {len(states)} and {len(state_kwargs)}")
        if stream_kwargs is None:
            stream_kwargs = np.array([dict() for _ in streams])
        if len(stream_kwargs) != len(streams):
            raise ValueError(f"Incompattible dimensions of streams and stream_kwargs, got {len(streams)} and {len(stream_kwargs)}")
        kwarg_arr = np.array([[_update_ret(skwarg, dkwarg) for dkwarg in stream_kwargs] for skwarg in state_kwargs])
    else:
        if len(kwarg_arr) != len(states): 
            ValueError(f"Incompatible dimensions of kwarg_arr and states, got {len(kwarg_arr)} and {len(states)}")
        if stream_kwargs is None:
            stream_kwargs = np.array([dict() for _ in streams])
            stream_kwargs_set = False
        else:
            stream_kwargs_set = True
        for i in range(len(kwarg_arr)):
            if isinstance(kwarg_arr[i], dict):
                kwarg_arr[i] = np.array([_update_ret(kwarg_arr[i], dkwarg) for dkwarg in stream_kwargs])
            elif stream_kwargs_set:
                warnings.warn("kwarg_arr specifies streams, stream_kwargs will be ignored")
            if len(kwarg_arr[i]) != len(streams):
                ValueError(f"Incompatible dimensions fo kwarg_arr and streams, got {len(kwarg_arr[i])} and {len(streams)}")
            
    return states, streams, kwarg_arr


def _single_sort(param, mask, stream, ph_streams):
    """
    Iterator for non-stream based dwell parameters

    Parameters
    ----------
    param : numpy.ndarray
        array for given parameter.
    mask : bool numpy.ndarray
        Mask of dwells to include.
    stream : list of fretbursts.Ph_sel
        Ignored, the streams to include.
    ph_streams : list of fretbursts.Ph_sel
        The ph_streams of BurstData object.

    Yields
    ------
    values: numpy.ndarray
         Masked parameter values.

    """
    yield param[mask]


def _stream_sort(param, mask, stream, ph_streams):
    """
    Iterator for stream based dwell parameters, iterates over all requested streams

    Parameters
    ----------
    param : numpy.ndarray
        array for given parameter.
    mask : bool numpy.ndarray
        Mask of dwells to include.
    stream : list of fretbursts.Ph_sel
        The streams to iterate include.
    ph_streams : list of fretbursts.Ph_sel
        The ph_streams of BurstData object.

    Yields
    ------
    values: numpy.ndarray
         Masked parameter values.

    """
    param_out = param[:,mask]
    for st in stream:
        i = np.argwhere([ph_s == st for ph_s in ph_streams]).reshape(-1)
        yield param_out[i,:].sum(axis=0)

def _make_dwell_pos(model, dwell_pos):
    """
    Make mask of dwells in the requested position

    Parameters
    ----------
    model : H2MM_result
        H2MM_result for which mask to make the mask.
    dwell_pos : int, list, tuple, numpy array or mask_generating callable
        The position(s) desired.

    Returns
    -------
    pos_mask : bool numpy.ndarray
        Mask of dwells at requested position(s).

    """
    if dwell_pos is None:
        pos_mask = np.ones(model.dwell_pos.shape, dtype=bool)
    elif type(dwell_pos) in (np.ndarray, list, tuple):
        pos_mask = np.sum([model.dwell_pos == i for i in dwell_pos], axis=0) > 0
    elif type(dwell_pos) == int:
        pos_mask = model.dwell_pos == dwell_pos
    else:
        pos_mask = dwell_pos(model)
    return pos_mask

# global mapping param type to iterator generating function    
__param_func = {"bar":_single_sort, "ratio":_single_sort, "stream":_stream_sort}

@_useideal
def scatter_ES(model, ax=None, add_corrections=False, state_kwargs=None, **kwargs):
    """
    Plot the position of all states in E and S
        
    .. note::
        
        If the ax kwarg is used, it is assumed to be used in conjunction with 
        other plots, and thus the xlim and ylim values will not be set

    Parameters
    ----------
    model : H2MM_result
        Model to plot values of E/S.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw scatterplot in. The default is None.
    add_corrections : bool, optional
        Use the corrected (True) or raw (False) E/S values. The default is False.
    **kwargs : keyword arguments
        Keyword arguments passed to ax.scatter to control the plotting

    Returns
    -------
    collection : matplotlib.collections.PathCollection
        The path collection the scatter plot method returns

    """
    ax = _check_ax(ax)
    E_name, S_name = ("E_corr", "S_corr") if add_corrections else ("E", "S")
    E, S = getattr(model, E_name), getattr(model, S_name)
    collection = ax.scatter(E, S, **kwargs)
    return collection
    

@_useideal
def axline_E(model, ax=None, add_corrections=False, horizontal=False, state_kwargs=None,
            **kwargs):
    """
    Add bars to plot indicating the FRET efficiency of states
    
    .. note::
    
        If the ax kwarg is used, it is assumed to be used in conjunction with 
        other plots, and thus the xlim and ylim values will not be set

    Parameters
    ----------
    model : H2MM_result
        Model to plot values of E
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw scatterplot in. The default is None.
    add_corrections : bool, optional
        Use the corrected (True) or raw (False) E values. The default is False.
    horizontal : bool, optional
        Whether to plot the bars horizontally (True) or vertically (False)
        The default is False.
    state_kwargs : list[dict], optional
        Keyword arguments per state passed to ax.axvline. The default is None.
    **kwargs : dict
        passed to ax.axvline as kwargs

    Returns
    -------
    lines : list[matplotlib.lines.Line2D]
        List of Lines returned by ax.axvline

    """
    ax = _check_ax(ax)
    
    axline = ax.axhline if horizontal else ax.axvline
    E  = model.E_corr if add_corrections else model.E
    if state_kwargs is None:
        state_kwargs = repeat({})
    elif len(state_kwargs) != E.size:
        raise ValueError(f"state_kwargs must have the same number of elements as input models has staes, got {len(state_kwargs)} and {E.size}")
    lines = [axline(e, **kw) for e, kw in zip(E, state_kwargs)]
    return lines

@_useideal
def axline_S(model, ax=None, add_corrections=False, horizontal=False, state_kwargs=None, 
            **kwargs):
    """
    Add bars to plot indicating the Stoichiometry of states
    
    .. note::
    
        If the ax kwarg is used, it is assumed to be used in conjunction with 
        other plots, and thus the xlim and ylim values will not be set

    Parameters
    ----------
    model : H2MM_result
        Model to plot values of S
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw scatterplot in. The default is None.
    add_corrections : bool, optional
        Use the corrected (True) or raw (False) S values. The default is False.
    horizontal : bool, optional
        Whether to plot the bars horizontally (True) or vertically (False)
        The default is False.
    state_kwargs : list[dict], optional
        Keyword arguments per state passed to ax.axvline. The default is None.
    **kwargs : dict
        passed to ax.axvline as kwargs

    Returns
    -------
    lines : list[matplotlib.lines.Line2D]
        List of Lines returned by ax.axvline

    """
    ax = _check_ax(ax)
    axline = ax.axhline if horizontal else ax.axvline
    S  = model.S_corr if add_corrections else model.S
    if state_kwargs is None:
        state_kwargs = repeat({})
    elif len(state_kwargs) != S.size:
        raise ValueError(f"state_kwargs must have the same number of elements as input models has staes, got {len(state_kwargs)} and {S.size}")
    lines = [axline(s, **kw) for s, kw in zip(S, state_kwargs)]
    return lines

    

@_useideal
def dwell_param_hist(model, param, streams=None, dwell_pos=None, states=None, 
                     state_kwargs=None, stream_kwargs=None, label_kwargs=None, 
                     kwarg_arr=None, ax=None, **kwargs):
    """
    Generate histograms of specified parameter of given model for states and
    streams of model

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    param : str
        Name of parameter to be histogramed.
    streams : list of frebursts.Ph_sel, optional
        The streams to use, ignored if param is not stream based. If None, take
        all streams in BurstData
        The default is None.
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
        The default is None.
    states : numpy.ndarray, optional
        The states to include, if None, all states are used. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        List of kwargs of same length as states, specifies specific additional
        kwargs passed to hist for each state. 
        The default is None.
    stream_kwargs : list of kwarg dicts, optional
        List of kwargs of same length as streams, specifies specific additional
        kwargs passed to hist for each stream. 
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    kwarg_arr : array of kwargs dicts, optional
        Array of dicts to use as kwargs for specific combinations of states/streams
        in data. Cannot be specified at same time as state_kwargs. If 2D, then will
        overwrite stream_kwargs, 2nd dimension, if exists specifies stream kwargs
        The default is None.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    **kwargs : keyword arguments
        Universal kwargs handed to ax.hist.

    Raises
    ------
    ValueError
        Unnacceptle set of kwargs specified.

    Returns
    -------
    None.

    """
    if param not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{param}', must be one of {[key for key in model.dwell_params.keys()]}")
    states, streams, kwarg_arr = _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr)
    if label_kwargs is None:
        label_kwargs = {}
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keword arguments, got {type(label_kwargs)}")
    ax = _check_ax(ax)
    pos_mask = _make_dwell_pos(model, dwell_pos)
    param_func = __param_func[model.dwell_params[param]]
    param_n = getattr(model, param)
    bin_style = {"bar":np.arange(np.nanmax(param_n)+1), "ratio":np.arange(0,1.05, 0.05)}
    state = model.dwell_state
    in_kwargs = kwargs.copy()
    if model.dwell_params[param] in bin_style:
        in_kwargs.update({'bins':bin_style[model.dwell_params[param]]})
    for i, skwargs in zip(states, kwarg_arr):
        mask = (state==i) * pos_mask
        for j, param_s in enumerate(param_func(param_n, mask, streams, model.parent.parent.ph_streams)):
            new_kwargs = in_kwargs.copy()
            new_kwargs.update(skwargs[j])
            ax.hist(param_s, **new_kwargs)
    ax.set_xlabel(model.param_labels[param], **label_kwargs)
    ax.set_ylabel("counts", **label_kwargs)


@_useideal
def dwell_params_scatter(model, paramx, paramy, states=None, state_kwargs=None, dwell_pos=None, 
                         streams=None, stream_kwargs=None, label_kwargs=None, kwarg_arr=None,
                         ax=None,  **kwargs):
    """
    Generate a plot of one parameter against another of dwells in a H2MM_result

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    paramx : str
        Name of parameter to be plotted along x-axis.
    paramy : str
        Name of parameter to be plotted along y-axis
    states : numpy.ndarray, optional
        The states to include, if None, all states are used. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        List of kwargs of same length as states, specifies specific additional
        kwargs passed to hist for each state. 
        The default is None.
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
        The default is None.
    streams : list of frebursts.Ph_sel, optional
        The streams to use, ignored if param is not stream based. If None, take
        all streams in BurstData
        The default is None.
    stream_kwargs : list of kwarg dicts, optional
        List of kwargs of same length as streams, specifies specific additional
        kwargs passed to hist for each stream. 
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    kwarg_arr : array of kwargs dicts, optional
        Array of dicts to use as kwargs for specific combinations of states/streams
        in data. Cannot be specified at same time as state_kwargs. If 2D, then will
        overwrite stream_kwargs, 2nd dimension, if exists specifies stream kwargs
        The default is None.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    **kwargs : TYPE
        Universal kwargs handed to ax.hist.

    Raises
    ------
    ValueError
        Unnacceptle set of kwargs specified..

    Returns
    -------
    None.

    """
    if paramx not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{paramx}', must be one of {[key for key in model.dwell_params.keys()]}")
    if paramy not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{paramy}', must be one of {[key for key in model.dwell_params.keys()]}")
    if label_kwargs is None:
        label_kwargs = {}
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keword arguments, got {type(label_kwargs)}")
    states, streams, kwarg_arr = _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr)
    pos_mask = _make_dwell_pos(model, dwell_pos)
    paramx_n = getattr(model, paramx)
    paramy_n = getattr(model, paramy)
    ax = _check_ax(ax)
    xtype = model.dwell_params[paramx]
    ytype = model.dwell_params[paramy]
    paramx_func = __param_func[xtype]
    paramy_func = __param_func[ytype]
    rpt = (xtype!="stream", ytype!="stream") if (ytype=="stream") != (xtype=="stream") else (False, False)
    state = model.dwell_state
    in_kwargs = dict(s=10, alpha=0.8)
    in_kwargs.update(kwargs)
    for i, skwargs in zip(states, kwarg_arr):
        mask = (state==i) * pos_mask
        paramx_sort = paramx_func(paramx_n, mask, streams, model.parent.parent.ph_streams)
        paramx_sort = cycle(paramx_sort) if rpt[0] else paramx_sort
        paramy_sort = paramy_func(paramy_n, mask, streams, model.parent.parent.ph_streams)
        paramy_sort = cycle(paramy_sort) if rpt[1] else paramy_sort
        for j, (paramx_s, paramy_s) in enumerate(zip(paramx_sort, paramy_sort)):
            new_kwargs = in_kwargs.copy()
            new_kwargs.update(skwargs[j])
            ax.scatter(paramx_s, paramy_s, **new_kwargs)
    ax.set_xlabel(model.param_labels[paramx], **label_kwargs)
    ax.set_ylabel(model.param_labels[paramy], **label_kwargs)
    

@_useideal
def dwell_param_transition_kde_plot(model, param, include_edge=True, ax=None, 
                                    stream=frb.Ph_sel(Dex="Dem"), states=None, 
                                    label_kwargs=None, **kwargs):
    """
    Make kdeplot of transitions, without separating diffent types of transitions

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    param : str
        Name of parameter to be plotted .
    include_edge : TYPE, optional
        Whether or not to include tranistions at the edges of bursts in dwells. 
        The default is True.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    stream : fretbursts.Ph_sel, optional
        Which stream to plot for stream based parameters, ignored for non-stream
        parameters. 
        The default is fretbursts.Ph_sel(Dex="Dem").
    states : bool numpy.ndarray, square, optional
        Which from-to transitions to include. If None, all transitions plotted.
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    **kwargs : TYPE
        kwargs passed to kdeplot.

    Raises
    ------
    ValueError
        Incompattible kwargs passed.

    Returns
    -------
    None.

    """
    if param not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{param}', must be one of {[key for key in model.dwell_params.keys()]}")
    if states is not None:
        assert type(states) == np.ndarray and states.dtype==bool, ValueError("States must be square mask")
        if not (states.ndim == 2 and states.shape[0] == states.shape[1] and states.shape[0] == model.model.nstate):
            raise ValueError(f"states must be square mask with shape ({model.model.nstate}, {model.model.nstate}), got {states.shape}")
    if param in ("dwell_state", "dwell_pos"):
        raise ValueError(f"Cannot plot '{param}': Transition plot meaningless for parameter '{param}'")
    if label_kwargs is None:
        label_kwargs = {}
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keword arguments, got {type(label_kwargs)}")
    param_n = getattr(model, param)
    if model.dwell_params[param] == "stream":
        st = np.argwhere([stream == ph_sel for ph_sel in model.parent.parent.ph_streams])[0,0]
        param_n = param_n[st,:]
    ax = _check_ax(ax)
    pos = model.dwell_pos[1:]
    e = 2 if include_edge else 1
    paramx = param_n[:-1][pos<e]
    paramy = param_n[1:][pos<e]
    if states is not None:
        state = model.dwell_state
        statemask = states[state[:-1],state[1:]]
        statemask = statemask[pos<e]
        paramx = paramx[statemask]
        paramy = paramy[statemask]
    sns.kdeplot(x=paramx, y=paramy, ax=ax, **kwargs)
    ax.set_xlabel(model.param_labels[param], **label_kwargs)
    ax.set_ylabel(model.param_labels[param], **label_kwargs)


@_useideal
def dwell_param_transition(model, param, include_edge=True, plt_type="scatter", ax=None,
                                  from_state=None, to_state=None, trans_mask=None,
                                  state_kwargs=None, streams=None, stream_kwargs=None,
                                  label_kwargs=None, kwarg_arr=None, **kwargs):
    """
    Plot transition map, separating differnt state to state transitions, either as
    scatter or kdeplot

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    param : str
        Name of parameter to be plotted .
    include_edge : TYPE, optional
        Whether or not to include tranistions at the edges of bursts in dwells. 
        The default is True.
    plt_type : str, optional
        "scatter" or "kde", specify whether to plot as scatter plot or kde-plot
        The default is scatter.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    from_state : numpy.ndarray, optional
        States of origin dwell to include. If None, all states included.
        The default is None.
    to_state : numpy.ndarray, optional
        States of desination dwell to include. If None, all states included.
        The default is None.
    trans_mask : bool np.ndarray, optional
        Mask of particular transitions to include. If None, all transitions
        included (Note: transitions from a state to the same state do not exist
        so diagonal values automatically False)
        The default is None.
    state_kwargs : 2D array of kwarg dicts, optional
        2D array in shape of from_state X to_state of kwargs to pass to each 
        combination of transitions scatter or kde-plots. 
        The default is None.
    streams : frebrursts.Ph_sel or list thereof, optional
        Which streams to include, if None, all streams included. Ignored if param
        is not stream-based. Generally recomended to select a single stream to
        prevent confusion between state transitions and different streams. 
        The default is None.
    stream_kwargs : list of kwarg dict, optional
        List of per-stream kwargs to pass to scatter or kdeplot. 
        The default is None.
    kwarg_arr : array of kwarg dicts, optional
        Array (2 or 3D) of arguments to pass per individual plot organized as
        from_state X to_state (optional X stream). Cannot be specified at same
        time as state_kwargs, if 3D, stream_kwargs ignored.
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    **kwargs : TYPE
        Universal kwargs handed to ax.hist.

    Raises
    ------
    ValueError
        Incompattible keyword arguments specified.

    Returns
    -------
    None.

    """
    ex_par = {"dwell_pos", "dwell_state"}
    assert param not in ex_par, ValueError(f"Redundant dwell parameter {param}, cannot make transition plot for this parameter")
    assert param in model.dwell_params, ValueError(f"{param} is not a dwell-based parameter, cannot generate transition plot")
    if state_kwargs is not None and kwarg_arr is not None:
        raise ValueError("Cannot specify both kwarg_arr and state_kwargs or stream_kwargs")
    from_state = _check_states(model, from_state)
    to_state = _check_states(model, to_state)
    if trans_mask is None:
        trans_mask = np.array([[ts != fs for ts in to_state] for fs in from_state])
    if streams is None:
        streams = model.parent.parent.ph_streams
    elif type(streams) == frb.Ph_sel:
        streams = [streams, ]
    if stream_kwargs is not None and len(stream_kwargs) != len(streams):
        raise ValueError(f"Incompatible lengths of streams and stream_kwargs arguments, got {len(streams)} and {len(stream_kwargs)}")
    elif stream_kwargs is None:
        stream_kwargs = np.array([dict() for _ in streams])
        stream_kwargs_set = False
    else:
        stream_kwargs_set = True
    # if else for state_kwargs and/or kwarg_arr, in the end creates kwarg_arr for final use
    if state_kwargs is None and kwarg_arr is None:
        kwarg_arr = np.array([[[skwarg for skwarg in stream_kwargs] for j in to_state] for i in from_state])
    # check state_kwargs if state_kwargs are given
    elif state_kwargs is not None:
        if len(state_kwargs) != len(from_state):
            raise ValueError(f"Incompatible state_kwargs and from_streams arguments, must have equal number of elements, got {len(state_kwargs)} and {len(from_state)}")
        for skwarg in state_kwargs:
            if len(skwarg) != len(to_state):
                raise ValueError(f"Incompatible inner state_kwargs and to_streams arguments, must have equal number of elements, got {len(skwarg)} and {len(to_state)}")
            for j in range(len(skwarg)):
                if skwarg[j] is None:
                    skwarg[j] = dict()
        kwarg_arr = np.array([[[_update_ret(skwarg, dkwarg) for dkwarg in stream_kwargs] for skwarg in stkwarg] for stkwarg in state_kwargs])
    # check if the kwarg_arr arguments all work
    elif kwarg_arr is not None:
        if len(kwarg_arr) != len(from_state):
            raise ValueError(f"Incompatiblae kwarg_arr outer dimension size, got {len(kwarg_arr)}, expected {len(from_state)}")
        for i in range(len(kwarg_arr)):
            if len(kwarg_arr[i]) != len(to_state):
                raise ValueError(f"Incompatiblae kwarg_arr[{i}] size, got {len(kwarg_arr[i])}, expected {len(to_state)}")
            for j in range(len(kwarg_arr[i])):
                if not trans_mask[i][j]:
                    continue
                if kwarg_arr[i][j] is None:
                    kwarg_arr[i][j] = stream_kwargs
                elif type(kwarg_arr[i][j]) == dict:
                    kwarg_arr[i][j] = np.array([_update_ret(kwarg_arr[i][j], dkwarg) for dkwarg in stream_kwargs])
                elif len(kwarg_arr[i][j]) != len(streams):
                    raise ValueError(f"Incompatiblae kwarg_arr[{i}][{j}] size, got {len(kwarg_arr[i][j])}, expected {len(stream_kwargs)}")
                elif stream_kwargs_set:
                    warnings.warn("kwarg_arr specifies photon streams, stream_kwargs will be ignored")
    if label_kwargs is None:
        label_kwargs = {}
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keword arguments, got {type(label_kwargs)}")
    assert plt_type in ("scatter", "kde"), ValueError(f"plt_type must be 'scatter' or 'kde', got {plt_type}")
    is_scatter = plt_type == "scatter"
    ax = _check_ax(ax)
    param_n = getattr(model, param)
    state_arr = model.dwell_state
    pfunc = __param_func[model.dwell_params[param]]
    e = 2 if include_edge else 1
    pos = model.dwell_pos[1:] < e
    dummy_mask = np.ones(state_arr.size, dtype=bool)
    new_kwargs = dict(s=10, alpha=0.8)
    new_kwargs.update(kwargs)
    for i, stf in enumerate(from_state):
        for j, stt in enumerate(to_state):
            if not trans_mask[i,j]:
                continue
            for k, param_s in enumerate(pfunc(param_n, dummy_mask, streams, model.parent.parent.ph_streams)):
                in_kwargs = new_kwargs.copy()
                in_kwargs.update(kwarg_arr[i][j][k])
                mask = (state_arr[:-1] == stf) * (state_arr[1:] == stt) * pos
                param_x = param_s[:-1][mask]
                param_y = param_s[1:][mask]
                if is_scatter:
                    ax.scatter(param_x, param_y, **in_kwargs)
                else:
                    sns.kdeplot(x=param_x, y=param_y, ax=ax,**in_kwargs)
    ax.set_xlabel(model.param_labels[param])
    ax.set_ylabel(model.param_labels[param])


def dwell_E_hist(model, ax=None, add_corrections=False, states=None, state_kwargs=None, 
                 label_kwargs=None, **kwargs):
    """
    Plot histogram of dwell FRET efficiency per state

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    add_corrections : bool, optional
        Use corrected or raw E values. 
        The default is False.
    states : numpy.ndarray, optional
        Which states to plot, if None, all states plotted. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        Kwargs passed per state. 
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    None.

    """
    in_kwargs = kwargs.copy()
    in_kwargs.update({'alpha':0.5})
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    dwell_param_hist(model, E, states=states, state_kwargs=state_kwargs, label_kwargs=label_kwargs, **in_kwargs)

def dwell_S_hist(model, ax=None, states=None, state_kwargs=None,add_corrections=False, 
                 label_kwargs=None, **kwargs):
    """
    Plot histogram of dwell stoichiometry per state

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    add_corrections : bool, optional
        Use corrected or raw E values. 
        The default is False.
    states : numpy.ndarray, optional
        Which states to plot, if None, all states plotted. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        Kwargs passed per state. 
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    None.

    """
    in_kwargs = kwargs.copy()
    in_kwargs.update({'alpha':0.5})
    S = "dwell_S_corr" if add_corrections else "dwell_S"
    dwell_param_hist(model, S, state_kwargs=state_kwargs, label_kwargs=label_kwargs,**in_kwargs)

def dwell_tau_hist(model, ax=None, streams=[frb.Ph_sel(Dex="Dem"), ], states=None, state_kwargs=None, 
                  stream_kwargs=None, label_kwargs=None, kwarg_arr=None, **kwargs):
    """
    Plot histograms of mean nanotimes of each state. Default is to plot only 
    D\ :sub:`ex`\ D\ :sub:`em`\  stream.

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    streams : list of fretbursts.Ph_sel, optional
        The stream(s) to inlcude. The default is [frb.Ph_sel(Dex="Dem"), ].
    states : numpy.ndarray, optional
        Which states to plot, if None, all states plotted. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        Kwargs passed per state. 
        The default is None.
    stream_kwargs : list of kwarg dict, optional
        List of per-stream kwargs to pass to scatter or kdeplot. 
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    kwarg_arr : array of kwargs dicts, optional
        Array of dicts to use as kwargs for specific combinations of states/streams
        in data. Cannot be specified at same time as state_kwargs. If 2D, then will
        overwrite stream_kwargs, 2nd dimension, if exists specifies stream kwargs
        The default is None.
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    None.

    """
    in_kwargs = kwargs.copy()
    in_kwargs.update({'alpha':0.5})
    dwell_param_hist(model, "dwell_nano_mean", streams=streams, states=states, state_kwargs=state_kwargs, 
                     stream_kwargs=stream_kwargs, label_kwargs=label_kwargs, kwarg_arr=kwarg_arr, **in_kwargs)

def dwell_ES_scatter(model, ax=None, states=None, state_kwargs=None, add_corrections=False, 
                     label_kwargs=None, **kwargs):
    """
    Dwell based ES scatter plot

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    states : numpy.ndarray, optional
        Which states to plot, if None, all states plotted. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        Kwargs passed per state. 
        The default is None.
    add_corrections : bool, optional
        Use corrected or raw E values. 
        The default is False.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    None.

    """
    in_kwargs = dict(s=10, alpha=0.7)
    in_kwargs.update(kwargs)
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    S = "dwell_S_corr" if add_corrections else "dwell_S"
    dwell_params_scatter(model, E, S, ax=ax, states=states, state_kwargs=state_kwargs, 
                         label_kwargs=label_kwargs,**in_kwargs)
    

def dwell_E_tau_scatter(model, ax=None, add_corrections=False, streams=[frb.Ph_sel(Dex="Dem"), ], 
                        states=None, state_kwargs=None, stream_kwargs=None, label_kwargs=None, 
                        kwarg_arr=None, **kwargs):
    """
    E-tau_D scatter plot

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    add_corrections : bool, optional
        Use corrected or raw E values. 
        The default is False.
    streams : list of fretbursts.Ph_sel, optional
        The stream(s) to inlcude. The default is [frb.Ph_sel(Dex="Dem"), ].
    states : numpy.ndarray, optional
        Which states to plot, if None, all states plotted. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        Kwargs passed per state. 
        The default is None.
    stream_kwargs : list of kwarg dict, optional
        List of per-stream kwargs to pass to scatter or kdeplot. 
        The default is None.
    label_kwargs : dict, optional
        Keyword arguments to pass to ax.label. The default is None
    kwarg_arr : array of kwargs dicts, optional
        Array of dicts to use as kwargs for specific combinations of states/streams
        in data. Cannot be specified at same time as state_kwargs. If 2D, then will
        overwrite stream_kwargs, 2nd dimension, if exists specifies stream kwargs
        The default is None.
    **kwargs : TYPE
        Universal kwargs handed to ax.hist.

    Returns
    -------
    None.

    """
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    dwell_params_scatter(model, "dwell_nano_mean", E, ax=ax, streams=streams, 
                         state_kwargs=state_kwargs, stream_kwargs=stream_kwargs, 
                         label_kwargs=label_kwargs, kwarg_arr=kwarg_arr, **kwargs)


def _stat_disc_plot(model_list, param, highlight_ideal=False, ideal_kwargs=None, ax=None,**kwargs):
    """
    Plot statistical discriminator

    Parameters
    ----------
    model_list : H2MM_list
        Set of optimizations to plot.
    param : str
        name of statistical parameter to plot.
    highlight_ideal : bool, optional
        Whether or not to plot the ideal/selected model separately. 
        The default is False.
    ideal_kwargs : dict or None, optional
        The kwargs to be passed specifically to the ideal model point.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        kwargs to be passed to ax.scatter

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        list of collections produced by the scatter plot

    """
    if ideal_kwargs is None:
        ideal_kwargs = {'c':'r'}
    if len(kwargs) == 0:
        kwargs = {'c':'b'}
    ax = _check_ax(ax)
    mask = np.array([opt is not None for opt in model_list.opts])
    states = np.arange(1, len(model_list.opts)+1)
    vals = getattr(model_list, param)
    collections = list()
    if highlight_ideal:
        ideal_state = states[model_list.ideal]
        ideal_val = vals[model_list.ideal]
        id_kwargs = kwargs.copy()
        id_kwargs.update(ideal_kwargs)
        collection = ax.scatter(ideal_state, ideal_val, **id_kwargs)
        collections.append(collection)
        mask[model_list.ideal] = False
    collection = ax.scatter(states[mask], vals[mask])
    collections.append(collection)
    ax.set_xlabel("states")
    ax.set_ylabel(model_list.stat_disc_labels[param])
    return collections
    

def ICL_plot(model_list, highlight_ideal=False, ideal_kwargs=None, ax=None,**kwargs):
    """
    Plot the ICL of each state

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compoared, a H2MM_list object (a divisor scheme).
    highlight_ideal : bool, optional
        Whether or not to plot the ideal/selected model separately. 
        The default is False.
    ideal_kwargs : dict or None, optional
        The kwargs to be passed specifically to the ideal model point.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        kwargs to be passed to ax.scatter

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        list of collections produced by the scatter plot
        
    """
    collections = _stat_disc_plot(model_list, 'ICL',highlight_ideal=highlight_ideal, ideal_kwargs=ideal_kwargs, ax=ax,**kwargs)
    return collections

def BIC_plot(model_list, highlight_ideal=False, ideal_kwargs=None, ax=None,**kwargs):
    """
    Plot the Bayes Information Criterion of each state

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compoared, a H2MM_list object (a divisor scheme).
    highlight_ideal : bool, optional
        Whether or not to plot the ideal/selected model separately. 
        The default is False.
    ideal_kwargs : dict or None, optional
        The kwargs to be passed specifically to the ideal model point.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        kwargs to be passed to ax.scatter

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        list of collections produced by the scatter plot

    """
    collections = _stat_disc_plot(model_list, 'BIC',highlight_ideal=highlight_ideal, ideal_kwargs=ideal_kwargs, ax=ax,**kwargs)
    return collections

def BICp_plot(model_list, highlight_ideal=False, ideal_kwargs=None, ax=None,**kwargs):
    """
    Plot the modified Bayes Information Criterion of each state

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compoared, a H2MM_list object (a divisor scheme).
    highlight_ideal : bool, optional
        Whether or not to plot the ideal/selected model separately. 
        The default is False.
    ideal_kwargs : dict or None, optional
        The kwargs to be passed specifically to the ideal model point.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        kwargs to be passed to ax.scatter

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        list of collections produced by the scatter plot

    """
    collections = _stat_disc_plot(model_list, 'BICp',highlight_ideal=highlight_ideal, ideal_kwargs=ideal_kwargs, ax=ax,**kwargs)
    return collections


def raw_nanotime_hist(data, streams=None, stream_kwargs=None, ax=None, **kwargs):
    """
    Plot the histogram of nanotimes of photons (in bursts) per stream.
    Usefull for visualizing the fluoresence decays, and deciding where to place
    the IRF thresh

    Parameters
    ----------
    data : BurstData
        The BurstData object for which the nanotime histogram will be plotted.
    streams : fretbursts.Ph_sel or list[fretbursts.Ph_sel], optional
        The stream(s) for which the nanotime is to be plotted, must be with 
        the photon selection of the BurstData object. If None, will plot all
        streams in data.
        The default is None.
    stream_kwargs : dict or list[dict], optional
        Per stream kwargs, passed to ax.plot, must match streams. If None, no 
        stream specific kwargs passed to ax.plot.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Kwargs passed to all plots.

    Raises
    ------
    ValueError
        Mismatched streams and stream_kwargs lengths

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        List of path collections, per stream, from ax.plot of each nanotime deacy.
    leg : matplotlib.legend.Legend
        Legend object

    """
    ax = _check_ax(ax)
    # check main kwargs are all compatible/correct
    if streams is None:
        streams = data.ph_streams
    elif isinstance(streams, frb.Ph_sel):
        streams = [streams]
    if stream_kwargs is None:
        stream_kwargs = [dict() for _ in streams]
    elif isinstance(stream_kwargs, dict):
        stream_kwargs = [stream_kwargs]
    if len(streams) != len(stream_kwargs):
        raise ValueError(f'streams and stream_kwargs must have same length, got {len(streams)} and {len(stream_kwargs)}')
    # get locations of selected streams
    stream_id = [np.argwhere([stream == psel for psel in data.ph_streams])[0,0] for stream in streams]
    index = np.concatenate(data.models.index)
    nanos = np.concatenate(data.nanos)
    # calcualte the decay histogram
    hists = [np.bincount(nanos[index==idx], minlength=data.data.nanotimes_params[0]['tcspc_num_bins']) for idx in stream_id]
    nanotime_bin = np.arange(data.data.nanotimes_params[0]['tcspc_num_bins'])
    collections = list()
    for hist, stream, s_kwargs in zip(hists, streams,stream_kwargs):
        in_kwargs = kwargs.copy()
        # add label name to kwarg dictionary
        in_kwargs.update({'label':stream.__str__()})
        in_kwargs.update(s_kwargs)
        collection = ax.plot(nanotime_bin, hist, **in_kwargs)
        collections.append(collection)
    leg = ax.legend()
    ax.set_xlabel("nanotime bin")
    ax.set_ylabel("counts")
    return collections, leg