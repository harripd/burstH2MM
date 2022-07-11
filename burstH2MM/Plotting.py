#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:41:39 2022

@author: paul
"""

from itertools import cycle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import fretbursts as frb
import seaborn as sns
# import BurstSort



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
    
    pass

def mid_dwell(model):
    """
    Return mask of all middle dwells in model (not whole, beginning or end)

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    mid_dwells: bool np.ndarray
        Mask of mid-dwells.

    """
    return model.dwell_pos == 0

def end_dwell(model):
    """
    Return mask of all dwells at the end of bursts

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    end_dwells: bool np.ndarray
        Mask of end-dwells.

    """
    return model.dwell_pos == 1

def begin_dwell(model):
    """
    Return mask of all dwells at the beginning of bursts

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    end_dwells: bool np.ndarray
        Mask of beginning-dwells.

    """
    return model.dwell_pos == 2

def burst_dwell(model):
    """
    Return mask of all dwells the span the entire bursts

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    burst_dwells: bool np.ndarray
        Mask of burst-dwells.

    """
    return model.dwell_pos == 3

def edge_dwell(model):
    """
    Return mask of all dwells at the beginning and end of bursts

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    edge_dwells: bool np.ndarray
        Mask of dwells at beginning and end of bursts

    """
    return (model.dwell_pos == 2) + (model.dwell_pos == 3)

def not_mid_dwell(model):
    """
    Return mask of all dwells that are not in the middle of bursts

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    not_mid_dwells: bool np.ndarray
        Mask of all dwells not in the middle of bursts.

    """
    return model.dwell_pos != 0

def burst_init_dwell(model):
    """
    Return mask of all dwells that start at the beginning of a burst

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    init_dwells: bool np.ndarray
        Mask of begin and whole-bursts-dwells.

    """
    return model.dwell_pos > 1

def burst_end_dwell(model):
    """
    Return mask of all dwells that end at the end of bursts (whole-burst and end dwells)

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.

    Returns
    -------
    burst_end_dwells: bool np.ndarray
        Mask of end-dwells and whole burst dwells.

    """
    return (model.dwell_pos == 1) + (model.dwell_pos == 4)
    

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
    assert np.all(states < model.model.nstate), ValueError("Cannot plot state {np.max(states)}, but model only has {model.model.nstate} states")
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
    in_stream = np.array([stream in model.parent.parent.ph_streams for stream in streams])
    assert np.all(in_stream), ValueError(f"Stream(s) {[stream for stream, in_s in zip(streams, in_stream) if not in_s]} not in BurstData")
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
        assert len(kwarg_arr) == len(states), ValueError(f"Incompatible dimensions of kwarg_arr and states, got {len(kwarg_arr)} and {len(states)}")
        if stream_kwargs is None:
            stream_kwargs = np.array([dict() for _ in streams])
            stream_kwargs_set = False
        else:
            stream_kwargs_set = True
        for i in range(len(kwarg_arr)):
            if type(kwarg_arr[i]) == dict:
                kwarg_arr[i] = np.array([_update_ret(kwarg_arr[i], dkwarg) for dkwarg in stream_kwargs])
            else:
                assert len(kwarg_arr[i]) == len(streams), ValueError(f"Incompatible dimensions fo kwarg_arr and streams, got {len(kwarg_arr[i])} and {len(streams)}")
                if not stream_kwargs_set:
                    warnings.warn("kwarg_arr specifies streams, stream_kwargs will be ignored")
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


def dwell_param_hist(model, param, streams=None, dwell_pos=None, states=None, 
                     state_kwargs=None, stream_kwargs=None, kwarg_arr=None, 
                     ax=None, **kwargs):
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
        Unnacceptle set of kwargs specified.

    Returns
    -------
    None.

    """
    if param not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{param}', must be one of {[key for key in model.dwell_params.keys()]}")
    states, streams, kwarg_arr = _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr)
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


def dwell_params_scatter(model, paramx, paramy, states=None, state_kwargs=None, dwell_pos=None, 
                         streams=None, stream_kwargs=None, kwarg_arr=None, ax=None, **kwargs):
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
    

def dwell_param_transition_kde_plot(model, param, include_edge=True, ax=None, 
                                    stream=frb.Ph_sel(Dex="Dem"), states=None, 
                                    **kwargs):
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



def dwell_param_transition(model, param, include_edge=True, plt_type="scatter", ax=None,
                                  from_state=None, to_state=None, trans_mask=None,
                                  state_kwargs=None, streams=None, stream_kwargs=None,
                                  kwarg_arr=None, **kwargs):
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
                

def dwell_E_hist(model, ax=None, add_corrections=False, states=None, state_kwargs=None, **kwargs):
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
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    None.

    """
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    dwell_param_hist(model, E, states=states, state_kwargs=state_kwargs, **kwargs)

def dwell_S_hist(model, ax=None, states=None, state_kwargs=None,add_corrections=False, **kwargs):
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
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    None.

    """
    S = "dwell_S_corr" if add_corrections else "dwell_S"
    dwell_param_hist(model, S, state_kwargs=state_kwargs, **kwargs)
        

def dwell_ES_scatter(model, ax=None, states=None, state_kwargs=None, add_corrections=False, **kwargs):
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
    dwell_params_scatter(model, E, S, ax=ax, state_kwargs=state_kwargs, **in_kwargs)
    

def dwell_E_tau_scatter(model, ax=None, add_corrections=False, streams=[frb.Ph_sel(Dex="Dem"), ], states=None, state_kwargs=None, 
                  stream_kwargs=None, kwarg_arr=None, **kwargs):
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
                         kwarg_arr=kwarg_arr, **kwargs)