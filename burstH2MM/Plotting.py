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
Most functions take a H2MM_result object as input, and customization is provided
through various keyword arguments.
"""

from collections.abc import Iterable
from itertools import chain, cycle, repeat, permutations
import functools
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

import fretbursts as frb


from . import BurstSort
from . import Masking

#: defaults for colors of streams
_color_dict = {frb.Ph_sel(Dex="Dem"):'g', frb.Ph_sel(Dex="Aem"):'r', frb.Ph_sel(Aex="Aem"):'purple'}

def _useideal(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        args = list(args)
        if isinstance(args[0], BurstSort.BurstData):
            args[0] = args[0].models
        if isinstance(args[0], BurstSort.H2MM_list):
            if not hasattr(args[0], 'ideal'):
                raise ValueError("Ideal model not set, set with H2MM_list.ideal = ")
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
    elif not isinstance(states, np.ndarray) and isinstance(states, Iterable):
        states = np.array(tuple(s for i, s in enumerate(states) if i <= model.nstate))
    elif isinstance(states, int):
        states = np.atleast_1d(states)
    elif not isinstance(states, np.ndarray):
        raise TypeError(f"Incompatible type for states, must be finite iterable or int, but got type: {type(states)}")
    if states.dtype == bool:
        states = np.argwhere(states).reshape(-1)
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
        streams = [streams, ]
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
    elif isinstance(dwell_pos, (list, tuple)):
        pos_mask = np.sum([model.dwell_pos == i for i in dwell_pos], axis=0) > 0
    elif isinstance(dwell_pos, int):
        pos_mask = model.dwell_pos == dwell_pos
    elif isinstance(dwell_pos, np.ndarray):
        if dwell_pos.dtype == bool:
            if model.dwell_pos.size == dwell_pos.size:
                pos_mask = dwell_pos
            else:
                raise ValueError(f'dwell_pos inorrect size, got {dwell_pos.size}, expected {model.dwell_pos.size}')
        elif np.issubdtype(dwell_pos.dtype, np.integer):
            if dwell_pos.max() > model.model.nstate:
                raise ValueError(f'dwell_pos includes non-exitstent states, model has only {model.model.nstate} states')
            elif dwell_pos.min() < 0:
                raise ValueError('dwell_pos incldues negative values, only non-negative values allowed')
            pos_mask = np.sum([model.dwell_pos == i for i in dwell_pos], axis=0) > 0
    elif callable(dwell_pos):
        pos_mask = dwell_pos(model)
    else:
        raise TypeError('Invalid dwell_pos type')
    return pos_mask

# global mapping param type to iterator generating function    
__param_func = {"bar":_single_sort, "ratio":_single_sort, "stream":_stream_sort}

@_useideal
def burst_ES_scatter(model, add_corrections=False, flatten_dynamics=False, 
                     type_kwargs=None, label_kwargs=None, ax=None, **kwargs):
    """
    Plot E-S scatter plot of bursts, colored based on states present in dwell.

    Parameters
    ----------
    model : H2MM_result
        Model of data to be plotted .
    add_corrections : bool, optional
        Whether to use the corrected E/S values of bursts, or to include background,
        leakage, direct excitation, gamma and beta value corrections. 
        The default is False.
    flatten_dynamics : bool, optional
        If True, bursts with multiple states are all plotted together.
        Useful for models with many numbers of states
        If False, every unique combination of states in burst are plotted separately.
        The default is False.
    type_kwargs : list[dict], optional
        List or tuple of dictionaries with burst-type specific arguments handed
        to ax.scatter. 
        
            .. note::
            
                The order is based on the order in which the burst-types are plotted.
                This will depend on whether `flatten_dynamics` is `True` or `False`
                If `True` the order is the states, and finally dynamics (thus the length
                will be # of state + 1).
                If `False`, then the order is that of `burst_type`, i.e. bitmap representation.
                Thus, the order will mix single and multiple states. Be careful your order.
            
        The default is None.
    label_kwargs : dict, optional
        Dictionary of keyword arguments to pass to ax.label
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw scatterplot in. The default is None.
    **kwargs : dict
        Dictionary of keyword arguments handed to ax.scatter.

    Raises
    ------
    ValueError
        Incorrect length of type_kwargs.

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        List of PathCollections returned by ax.scatter.

    """
    ax = _check_ax(ax)
    if label_kwargs is None:
        label_kwargs = dict()
    elif not isinstance(label_kwargs, dict):
        raise ValueError("label_kwargs must be dict of keys for ax.label")
    if add_corrections:
        E = np.concatenate(model.parent.parent.data.E)
        S = np.concatenate(model.parent.parent.data.S)
        xlabel, ylabel = "E", "S"
    else:
        Dr = np.argwhere([ph_stream == frb.Ph_sel(Dex='Dem') 
                          for ph_stream in model.parent.parent.ph_streams])[0,0]
        Ar = np.argwhere([ph_stream == frb.Ph_sel(Dex='Aem') 
                          for ph_stream in model.parent.parent.ph_streams])[0,0]
        Cr = np.argwhere([ph_stream == frb.Ph_sel(Aex='Aem') 
                          for ph_stream in model.parent.parent.ph_streams])[0,0]
        A = np.array([(idx == Ar).sum() for idx in model.parent.parent.models.index])
        D = np.array([(idx == Dr).sum() for idx in model.parent.parent.models.index])
        C = np.array([(idx == Cr).sum() for idx in model.parent.parent.models.index])
        DA = (A+D)
        E = np.empty(A.shape)
        nan_mask = DA == 0
        E[~nan_mask] = A[~nan_mask] / DA[~nan_mask]
        E[nan_mask] = np.nan
        S = DA / (DA + C)
        xlabel, ylabel = r"E$\rm_{raw}$", r"S$\rm_{raw}$"
    burst_color = model.burst_type
    if flatten_dynamics:
        burst_color_new = -1 * np.ones(burst_color.size, dtype=int)
        for i in range(model.model.nstate):
            burst_color_new[burst_color == 2**i] = i
        burst_color_new[burst_color_new == -1] = model.model.nstate
        burst_color = burst_color_new
    uni = np.unique(burst_color)
    if type_kwargs is None:
        type_kwargs = repeat({})
    elif len(type_kwargs) != uni.size:
        raise ValueError(f"type_kwargs must be of same length as number of burst types, got {len(type_kwargs)} for type_kwargs, but {uni.size} needed")
    collections = list()
    for un, skwargs in zip(uni, type_kwargs):
        E_sub = E[burst_color == un]
        S_sub = S[burst_color == un]
        in_kwargs = kwargs.copy()
        in_kwargs.update(skwargs)
        collection = ax.scatter(E_sub, S_sub, **in_kwargs)
        collections.append(collection)
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)
    return collections

@_useideal
def scatter_ES(model, ax=None, add_corrections=False, states=None, **kwargs):
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
    states : array-like, optional
        A mask for which states to use. May be a 1-D integer array of states, or
        boolean mask of which states to plot, in the latter case, must be of the
        same size as the number of states in the model. The default is None
    **kwargs : keyword arguments
        Keyword arguments passed to ax.scatter to control the plotting

    Returns
    -------
    collection : matplotlib.collections.PathCollection
        The path collection the scatter plot method returns

    """
    ax = _check_ax(ax)
    if states is None:
        states = np.arange(model.nstate)
    elif isinstance(states, (list, tuple)):
        states = np.array(states)
    E_name, S_name = ("E_corr", "S_corr") if add_corrections else ("E", "S")
    E, S = getattr(model, E_name), getattr(model, S_name)
    collection = ax.scatter(E[states], S[states], **kwargs)
    return collection
    

@_useideal
def trans_arrow_ES(model, ax=None, add_corrections=False, min_rate=1e1,  
                   states=None, positions=0.5, rotate=True, sep=2e-2, 
                   fstring='3.0f', unit=False, 
                   from_arrow='-', from_props=None, to_arrow='-|>', to_props=None, 
                   arrowprops=None, state_kwargs=None, **kwargs):
    """
    Generate arrows between states in E-S plot indicating the transition rate.

    Parameters
    ----------
    model : H2MM_result
        H2MM_result to plot the transition rate arrows.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw scatterplot in. The default is None.
    add_corrections : bool, optional
        Use the corrected (True) or raw (False) E values. The default is False.
    min_rate : float, optional
        Minimum transition rate (in seconds) to plot, ignored if states is specified.
        The default is 1e1.
    states : tuple[tuple[int, int]], optional
        Manually specify which transition rate to plot. Specify as tuple of 2 tuples,
        each 2 tuple as (from_state, to_state). If None, automatically generate
        transition rates. The default is None.
    positions : float or numpy.ndarray, optional
        The position of the transition rate label, specified as fraction of distance
        between states of transition rate. If specified as float, same position is
        used for all rates, if specified as numpy.ndarray, then each rate
        is specified as matrix [from_state][to_state] (diagonals ignored). 
        The default is 0.5.
    rotate : bool, optional
        Whether or not to rotate the transition rate label. The default is True.
    sep : float, optional
        When transition rates [i,j] and [j,i] are specified, the offset in data 
        points, to add to the positions (to prevent overlap between foroward and 
        backward transition rates). 
        The default is 2e-2.
    fstring : str, optional
        The format string defining how to format the transition rate. 
        The default is '3.0f'
    unit : bool or str, optional
        The unit to display for transition rates. If False, no unit is displayed.
        If True, show s^-1, if a string (not recomended) this string will be appended
        to the end of all transition rates. The default is False
    from_arrow : str, optional
        Format string for arrow pointing to from_state (value passed to 'arrowstyle'). 
        The default is '-'.
    from_props : 
        The default is None
    to_arrow : str, optional
        Format string for arrow pointing to to_state (value passed to 'arrowstyle'). 
        The default is '-\|>'.
    to_props : 
        The default is None
    state_kwargs : tuple[tuple[dict]], optional
        Transition specific keyword arguments to be passed to ax.annotate(). 
        The default is None.
    **kwargs : dict
        keyword arguments passed to ax.annotate().

    Raises
    ------
    ValueError
        One or more keyword arguments contains invalid types or is of invalid length.

    Returns
    -------
    annos : list[list[matplotlib.text.Annotation]]
        List of list of annotations added to the plot.

    """
    ax = _check_ax(ax)
    if states is None:
        states = tuple((i, j) for i, j in permutations(range(model.nstate), 2) if model.trans[i,j] >= min_rate)
    else:
        try:
            trans_num = len(states)
            states = tuple(tuple(st) for st in states if len(st)==2)
        except TypeError as e:
            raise TypeError("Incorrect format for states, must be ((from_state, to state), ...)") from e
        else:
            if trans_num != len(states):
                raise ValueError("Each state must be defined by a [from_state, to_state] like array")
    if state_kwargs is None:
        state_kwargs = [[kwargs for j in range(model.nstate)] for i in range(model.nstate)]
    if isinstance(positions,  (float, int)):
        positions = positions * np.ones((model.nstate, model.nstate))
    if arrowprops is None:
        arrowprops = {'color':'k'}
    if not isinstance(arrowprops, dict):
        raise TypeError(f"arrowprops must be dict, got {type(arrowprops)}")
    else:
        arrowprops = _update_ret({'color':'k'}, arrowprops)
    if to_props is None:
        to_props = dict()
    if not isinstance(to_props, (dict, map)):
        raise TypeError(f"to_props must be mapping, got {type(to_props)}")
    if from_props is None:
        from_props = dict()
    if not isinstance(from_props, (dict, map)):
        raise ValueError(f"from_props must be mapping, got {type(from_props)}")
    if isinstance(unit, bool):
        unit = r'$s^{-1}$' if unit else ''
    elif not isinstance(unit, str):
        raise ValueError(f"unit must be bool or str, got {type(unit)}")
    base_kwargs = dict(xycoords='data', textcoords='data', horizontalalignment='center', 
                       verticalalignment='center', rotation_mode='anchor', 
                       transform_rotates_text=True)
    to_kwargs = base_kwargs.copy()
    from_kwargs = base_kwargs.copy()
    to_kwargs['arrowprops'] = _update_ret(arrowprops, {'arrowstyle':to_arrow, **to_props})
    if 'facecolor' in to_kwargs['arrowprops'] or 'edgecolor' in to_kwargs['arrowprops']:
        to_kwargs['arrowprops'].pop('color', None)
    from_kwargs['arrowprops'] = _update_ret(arrowprops, {'arrowstyle':from_arrow, **from_props})
    if 'facecolor' in from_kwargs['arrowprops'] or 'edgecolor' in from_kwargs['arrowprops']:
        from_kwargs['arrowprops'].pop('color', None)
    E = model.E_corr if add_corrections else model.E
    S = model.S_corr if add_corrections else model.S
    annos = list()
    for i, j in states:
        tstr = ('%%%s' % fstring) % model.trans[i,j] + ' ' + unit
        try:
            st_kw_to = _update_ret(to_kwargs, state_kwargs[i][j])
            st_kw_from = _update_ret(from_kwargs, state_kwargs[i][j])
        except IndexError:
            raise IndexError(f"state_kwargs too short, state_kwargs[{i}][{j}] out of range")
        except Exception as e:
            raise TypeError("state_kwargs or element therof of incompatible type/shape") from e
        orig = np.array([E[i], S[i]])
        dest = np.array([E[j], S[j]])
        try:
            text = (dest - orig)*positions[i][j] + orig 
        except IndexError:
            raise IndexError(f"positions too short, positions[{i}][{j}] out of range")
        except Exception as e:
            raise ValueError("positions of incompatible argument") from e
        if (j, i) in states:
            dist = np.sqrt((E[j]-E[i])**2+(S[j]-S[i])**2)
            shift = ([-(S[j]-S[i]), (E[j]-E[i])]) / dist * sep
            orig += shift
            dest += shift
            text += shift
        if rotate:
            if E[j] - E[i] != 0.0:
                angle = np.arctan((S[j]-S[i])/(E[j]-E[i])) * 180 / np.pi
            else:
                angle = 90.0 if S[j] > S[i] else -90.0
        else:
            angle = 0.0
        anno = [None for _ in range(2)]
        try:
            anno[0] = ax.annotate(tstr, dest, xytext=text, rotation=angle, **st_kw_to)
        except Exception as e:
            raise e
        try:
            anno[1] = ax.annotate(tstr, orig, xytext=text, rotation=angle, **st_kw_from)
        except Exception as e:
            raise e
        annos.append(anno)
    return annos



def _check_line_kwargs(model, states, state_kwargs):
    states = _check_states(model, states)
    if state_kwargs is None:
        state_kwargs = tuple(dict() for _ in states)
    if len(state_kwargs) != states.size:
        raise ValueError("state_kwargs must be same size as states, got {len(state_kwargs)} and {states.size}")
    return states, state_kwargs


@_useideal
def axline_E(model, ax=None, add_corrections=False, horizontal=False, states=None, 
             state_kwargs=None, **kwargs):
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
    states : array-like, optional
        Which states to plot, identified as array of states, boolean mask or int.
        The default is None
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
    states, state_kwargs = _check_line_kwargs(model, states, state_kwargs)
    axline = ax.axhline if horizontal else ax.axvline
    E  = model.E_corr if add_corrections else model.E
    E = E[states]
    lines = [axline(e, **kw) for e, kw in zip(E, state_kwargs)]
    return lines

@_useideal
def axline_S(model, ax=None, add_corrections=False, horizontal=False, states=None, 
             state_kwargs=None, **kwargs):
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
    states : array-like, optional
        Which states to plot, identified as array of states, boolean mask or int.
        The default is None
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
    states, state_kwargs = _check_line_kwargs(model, states, state_kwargs)
    axline = ax.axhline if horizontal else ax.axvline
    S  = model.S_corr if add_corrections else model.S
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
        Unacceptable set of kwargs specified.

    Returns
    -------
    collections : list[list[matplotlib.container.BarContainer]]
        list of lists of bar containers produced by the ax.hist 
        organized as [states][streams]

    """
    if param not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{param}', must be one of {[key for key in model.dwell_params.keys()]}")
    states, streams, kwarg_arr = _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr)
    if label_kwargs is None:
        label_kwargs = dict()
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keword arguments, got {type(label_kwargs)}")
    ax = _check_ax(ax)
    pos_mask = _make_dwell_pos(model, dwell_pos)
    param_func = __param_func[model.dwell_params[param]]
    param_n = getattr(model, param)
    bin_style = {"bar":np.arange(np.nanmax(param_n)+1), "ratio":np.arange(0,1.05, 0.05)}
    state = model.dwell_state
    in_kwargs = kwargs.copy()
    if model.dwell_params[param] in bin_style and 'bins' not in kwargs:
        in_kwargs.update({'bins':bin_style[model.dwell_params[param]]})
    collections = list()
    for i, skwargs in zip(states, kwarg_arr):
        mask = (state==i) * pos_mask
        collections.append(list())
        for j, param_s in enumerate(param_func(param_n, mask, streams, model.parent.parent.ph_streams)):
            new_kwargs = in_kwargs.copy()
            new_kwargs.update(skwargs[j])
            collections[-1].append(ax.hist(param_s, **new_kwargs)[2])
    ax.set_xlabel(model.param_labels[param], **label_kwargs)
    ax.set_ylabel("counts", **label_kwargs)
    return collections


@_useideal
def dwell_params_scatter(model, paramx, paramy, states=None, state_kwargs=None, dwell_pos=None, 
                         streams=None, stream_kwargs=None, label_kwargs=None, kwarg_arr=None,
                         ax=None,  plot_type='scatter', **kwargs):
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
    plot_type : str, optional
        'scatter' or 'kde' whether to plot with ax.scatter or sns.kdeplot.
        The default is 'scatter'
    **kwargs : TYPE
        Universal kwargs handed to ax.hist.

    Raises
    ------
    ValueError
        Unacceptable set of kwargs specified..

    Returns
    -------
    collections : list[list[matplotlib.collections.PathCollection]]
        List of lists of matplotlib PathCollections return by ax.scatter
        Organized as [state][streams]

    """
    if paramx not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{paramx}', must be one of {[key for key in model.dwell_params.keys()]}")
    if paramy not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{paramy}', must be one of {[key for key in model.dwell_params.keys()]}")
    if label_kwargs is None:
        label_kwargs = dict()
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keword arguments, got {type(label_kwargs)}")
    states, streams, kwarg_arr = _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr)
    pos_mask = _make_dwell_pos(model, dwell_pos)
    paramx_n = getattr(model, paramx)
    paramy_n = getattr(model, paramy)
    ax = _check_ax(ax)
    in_kwargs = dict()
    if plot_type == 'scatter':
        plot_func = ax.scatter
        in_kwargs.update(s=10, alpha=0.8)
    elif plot_type == 'kde':
        plot_func = lambda px, py, **kwargs: sns.kdeplot(x=px, y=py, ax=ax, **kwargs)
    xtype = model.dwell_params[paramx]
    ytype = model.dwell_params[paramy]
    paramx_func = __param_func[xtype]
    paramy_func = __param_func[ytype]
    rpt = (xtype!="stream", ytype!="stream") if (ytype=="stream") != (xtype=="stream") else (False, False)
    state = model.dwell_state
    in_kwargs.update(kwargs)
    collections = list()
    for i, skwargs in zip(states, kwarg_arr):
        mask = (state==i) * pos_mask
        paramx_sort = paramx_func(paramx_n, mask, streams, model.parent.parent.ph_streams)
        paramx_sort = cycle(paramx_sort) if rpt[0] else paramx_sort
        paramy_sort = paramy_func(paramy_n, mask, streams, model.parent.parent.ph_streams)
        paramy_sort = cycle(paramy_sort) if rpt[1] else paramy_sort
        collections.append(list())
        for j, (paramx_s, paramy_s) in enumerate(zip(paramx_sort, paramy_sort)):
            new_kwargs = in_kwargs.copy()
            new_kwargs.update(skwargs[j])
            collection = plot_func(paramx_s, paramy_s, **new_kwargs)
            collections[-1].append(collection)
    ax.set_xlabel(model.param_labels[paramx], **label_kwargs)
    ax.set_ylabel(model.param_labels[paramy], **label_kwargs)
    return collections


@_useideal
def dwell_param_transition_kde_plot(model, param, include_edge=True, ax=None, 
                                    stream=frb.Ph_sel(Dex="Dem"), states=None, 
                                    label_kwargs=None, **kwargs):
    """
    Make kdeplot of transitions, without separating different types of transitions

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    param : str
        Name of parameter to be plotted .
    include_edge : TYPE, optional
        Whether or not to include transitions at the edges of bursts in dwells. 
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
        Incompatible kwargs passed.

    Returns
    -------
    collection : matplotlib.axes._subplots.AxesSubplot
        Returned from sns.kdeplot

    """
    if param not in model.dwell_params:
        raise ValueError(f"Invalid parameter, param: '{param}', must be one of {[key for key in model.dwell_params.keys()]}")
    if states is not None:
        assert isinstance(states, np.ndarray) and states.dtype==bool, ValueError("States must be square mask")
        if not (states.ndim == 2 and states.shape[0] == states.shape[1] and states.shape[0] == model.model.nstate):
            raise ValueError(f"states must be square mask with shape ({model.model.nstate}, {model.model.nstate}), got {states.shape}")
    if param in ("dwell_state", "dwell_pos"):
        raise ValueError(f"Cannot plot '{param}': Transition plot meaningless for parameter '{param}'")
    if label_kwargs is None:
        label_kwargs = {}
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keyword arguments, got {type(label_kwargs)}")
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
    collection = sns.kdeplot(x=paramx, y=paramy, ax=ax, **kwargs)
    ax.set_xlabel(model.param_labels[param], **label_kwargs)
    ax.set_ylabel(model.param_labels[param], **label_kwargs)
    return collection


@_useideal
def dwell_param_transition(model, param, include_edge=True, plt_type="scatter", ax=None,
                                  from_state=None, to_state=None, trans_mask=None,
                                  state_kwargs=None, streams=None, stream_kwargs=None,
                                  label_kwargs=None, kwarg_arr=None, **kwargs):
    """
    Plot transition map, separating different state to state transitions, either as
    scatter or kdeplot

    Parameters
    ----------
    model : H2MM_result or H2MM_list
        Source of data.
    param : str
        Name of parameter to be plotted .
    include_edge : TYPE, optional
        Whether or not to include transitions at the edges of bursts in dwells. 
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
        States of destination dwell to include. If None, all states included.
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
        is not stream-based. Generally recommended to select a single stream to
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
        Incompatible keyword arguments specified.

    Returns
    -------
    collections : list[list[list[matplotlib.collections.PathCollection or matplotlib.axes._subplots.AxesSubplot]]]
        The collections or axes returned by each call to ax.scatter or sns.kdeplot
        organized [from_state][to_state][stream]

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
                    raise ValueError(f"Incompatible kwarg_arr[{i}][{j}] size, got {len(kwarg_arr[i][j])}, expected {len(stream_kwargs)}")
                elif stream_kwargs_set:
                    warnings.warn("kwarg_arr specifies photon streams, stream_kwargs will be ignored")
    if label_kwargs is None:
        label_kwargs = {}
    elif not isinstance(label_kwargs, dict):
        raise ValueError(f"label_kwargs must be dictionary of keyword arguments, got {type(label_kwargs)}")
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
    collections = list()
    for i, stf in enumerate(from_state):
        collections.append(list())
        for j, stt in enumerate(to_state):
            if not trans_mask[i,j]:
                continue
            collections[-1].append(list())
            for k, param_s in enumerate(pfunc(param_n, dummy_mask, streams, model.parent.parent.ph_streams)):
                in_kwargs = new_kwargs.copy()
                in_kwargs.update(kwarg_arr[i][j][k])
                mask = (state_arr[:-1] == stf) * (state_arr[1:] == stt) * pos
                param_x = param_s[:-1][mask]
                param_y = param_s[1:][mask]
                if is_scatter:
                    collection = ax.scatter(param_x, param_y, **in_kwargs)
                else:
                    collection = sns.kdeplot(x=param_x, y=param_y, ax=ax,**in_kwargs)
                collections[-1][-1].append(collection)
    ax.set_xlabel(model.param_labels[param])
    ax.set_ylabel(model.param_labels[param])
    return collections


def dwell_E_hist(model, ax=None, add_corrections=False, states=None, state_kwargs=None, 
                 label_kwargs=None, dwell_pos=None,**kwargs):
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
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
        The default is None.
    
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    collections : list[matplotlib.container.BarContainer]
        list of bar containers produced by the ax.hist 

    """
    in_kwargs = {'alpha':0.5}
    in_kwargs.update(**kwargs)
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    collections = dwell_param_hist(model, E, ax=ax, states=states, state_kwargs=state_kwargs,
                                   label_kwargs=label_kwargs, dwell_pos=dwell_pos, **in_kwargs)
    # move lists of streams (not applicable to E)
    collections = [collection[0] for collection in collections]
    return collections

def dwell_S_hist(model, ax=None, states=None, state_kwargs=None,add_corrections=False, 
                 label_kwargs=None, dwell_pos=None, **kwargs):
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
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
        The default is None.
    **kwargs : 
        Universal kwargs for ax.hist.

    Returns
    -------
    collections : list[matplotlib.container.BarContainer]
        list of lists of bar containers produced by the ax.hist 
        organized as [states][streams]

    """
    in_kwargs = {'alpha':0.5, 'bins':20}
    in_kwargs.update(**kwargs)
    S = "dwell_S_corr" if add_corrections else "dwell_S"
    collections = dwell_param_hist(model, S, states=states, state_kwargs=state_kwargs, 
                                   label_kwargs=label_kwargs, dwell_pos=dwell_pos, 
                                   **in_kwargs)
    collections = [collection[0] for collection in collections]
    return collections

@_useideal
def dwell_trans_dur_hist(model, to_state=None, from_state=None, include_beg=True, 
                          to_state_kwargs=None, from_state_kwargs=None, 
                          kwarg_arr=None, ax=None, **kwargs):
    to_state = _check_states(model, to_state)
    from_state = _check_states(model, from_state)
    if (from_state_kwargs is not None or to_state_kwargs is not None) and kwarg_arr is not None:
        warnings.warn("Specifying to_state_kwargs or from_state_kwargs at same time as kwarg_arr, will result in dictionary smashing")
    # build kwargs array, first check kwarg_arr
    kwarg_mat = [[dict(alpha=0.5, bins=20) for j in to_state] for i in from_state]
    if kwarg_arr is not None:
        if not isinstance(kwarg_arr, Iterable):
            raise ValueError("kwarg_arr is not iterable type")
        elif len(kwarg_arr) != len(from_state):
            raise ValueError("kwarg_arr must be same length as from_state, got {len(from_state)}, and {len(kwarg_arr)}")
        for i, kwi in enumerate(kwarg_arr):
            if not isinstance(kwi, Iterable):
                raise ValueError("kwarg_arr[{i}] is not iterable type")
            elif len(kwi) != len(to_state):
                raise ValueError("kwarg_arr[{i}] must be same length as to_state, got {len(to_state)}, and {len(kwi)}")
            for j, kwj in enumerate(kwi):
                kwarg_mat[i][j] = _update_ret(kwarg_mat[i][j], kwj)
    # update from_state_kwargs
    if from_state_kwargs is not None:
        if not isinstance(from_state_kwargs, Iterable):
            raise ValueError("")
        elif len(from_state_kwargs) != len(from_state):
            raise ValueError("")
        for i, fsk in enumerate(from_state_kwargs):
            for j in range(len(to_state)):
                kwarg_mat[i][j] = _update_ret(kwarg_mat[i][j], fsk)
    # update from_state_kwargs
    if to_state_kwargs is not None:
        if not isinstance(to_state_kwargs, Iterable):
            raise ValueError("")
        elif len(to_state_kwargs) != len(to_state):
            raise ValueError("")
        for i in range(len(from_state)):
            for j, tsk in enumerate(to_state_kwargs):
                kwarg_mat[i][j] = _update_ret(kwarg_mat[i][j], tsk)
    # plotting
    ax = _check_ax(ax)
    collections = [[None for tmpj in to_state] for tmpi in from_state]
    for f, t in permutations(from_state, to_state):
        if f == t:
            continue
        msk = BurstSort._trans_mask(model, f, t, include_beg=include_beg)
        collections[i][j] = ax.hist(model.dwell_dur[msk], **kwarg_mat[f][t])
    return collections
        

def dwell_dur_hist(model, ax=None, states=None, state_kwargs=None, label_kwargs=None, 
                   dwell_pos=None, **kwargs):
    in_kwargs = {'alpha':0.5, 'bins':20}
    in_kwargs.update(**kwargs)
    
    collections = dwell_param_hist(model, "dwell_dur", states=states, 
                                   state_kwargs=state_kwargs, 
                                   label_kwargs=label_kwargs, dwell_pos=dwell_pos, 
                                   **in_kwargs)
    collections = [collection[0] for collection in collections]
    return collections



def dwell_tau_hist(model, ax=None, streams=[frb.Ph_sel(Dex="Dem"), ], states=None, 
                   state_kwargs=None, stream_kwargs=None, label_kwargs=None, 
                   dwell_pos=None, kwarg_arr=None, **kwargs):
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
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
        The default is None.
    kwarg_arr : array of kwargs dicts, optional
        Array of dicts to use as kwargs for specific combinations of states/streams
        in data. Cannot be specified at same time as state_kwargs. If 2D, then will
        overwrite stream_kwargs, 2nd dimension, if exists specifies stream kwargs
        The default is None.
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    collections : list[list[matplotlib.container.BarContainer]]
        list of lists of bar containers produced by the ax.hist 
        organized as [states][streams]

    """
    in_kwargs = {'alpha':0.5}
    in_kwargs.update(**kwargs)
    collections = dwell_param_hist(model, "dwell_nano_mean", streams=streams, states=states, 
                                   state_kwargs=state_kwargs, stream_kwargs=stream_kwargs, 
                                   label_kwargs=label_kwargs, dwell_pos=dwell_pos, 
                                   kwarg_arr=kwarg_arr, **in_kwargs)
    return collections

def dwell_ES_scatter(model, ax=None, plot_type='scatter', states=None, state_kwargs=None, add_corrections=False, 
                     label_kwargs=None, dwell_pos=None, **kwargs):
    """
    Dwell based ES scatter plot

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    plot_type : str, optional
        'scatter' or 'kde' whether to plot with ax.scatter or sns.kdeplot.
        The default is 'scatter'
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
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
        The default is None.
    **kwargs : dict
        Universal kwargs for ax.hist.

    Returns
    -------
    collections : list[matplotlib.collections.PathCollection]
        List of matplotlib PathCollections return by ax.scatter

    """
    in_kwargs = dict(s=10, alpha=0.7) if plot_type == 'scatter' else dict()
    in_kwargs.update(kwargs)
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    S = "dwell_S_corr" if add_corrections else "dwell_S"
    collections = dwell_params_scatter(model, E, S, ax=ax, states=states, state_kwargs=state_kwargs, 
                                       label_kwargs=label_kwargs, dwell_pos=dwell_pos, 
                                       plot_type=plot_type, **in_kwargs)
    collections = [collection[0] for collection in collections]
    return collections
    

def dwell_E_tau_scatter(model, ax=None, plot_type='scatter', add_corrections=False, 
                        streams=[frb.Ph_sel(Dex="Dem"), ], states=None, state_kwargs=None, 
                        stream_kwargs=None, label_kwargs=None, dwell_pos=None, 
                        kwarg_arr=None, **kwargs):
    """
    E-tau_D scatter plot

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to draw histogram(s) in. The default is None.
    plot_type : str, optional
        'scatter' or 'kde' whether to plot with ax.scatter or sns.kdeplot.
        The default is 'scatter'
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
    dwell_pos : int, list, tuple, numpy array or mask_generating callable, optional
        Which dwell position(s) to include.If None, do not filter by burst position.
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
    collections : list[list[matplotlib.collections.PathCollection]]
        List of lists of matplotlib PathCollections return by ax.scatter
        Organized as [state][streams]

    """
    E = 'dwell_E_corr' if add_corrections else 'dwell_E'
    collections = dwell_params_scatter(model, "dwell_nano_mean", E, ax=ax, states=states, 
                                       state_kwargs=state_kwargs, streams=streams, 
                                       stream_kwargs=stream_kwargs, 
                                       label_kwargs=label_kwargs, dwell_pos=dwell_pos, 
                                       kwarg_arr=kwarg_arr, plot_type=plot_type, **kwargs)
    return collections


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
    if mask.size == 0:
        raise ValueError(f"No models calculated, cannot plot {param}")
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
    collection = ax.scatter(states[mask], vals[mask], **kwargs)
    collections.append(collection)
    ax.set_xlabel("states")
    ax.set_ylabel(model_list.stat_disc_labels[param])
    return collections
    

def ICL_plot(model_list, highlight_ideal=False, ideal_kwargs=None, ax=None,**kwargs):
    """
    Plot the ICL of each state model.

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compared, a H2MM_list object (a divisor scheme).
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
    Plot the Bayes Information Criterion of each state model.

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compared, a H2MM_list object (a divisor scheme).
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
    Plot the modified Bayes Information Criterion of each state model.

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compared, a H2MM_list object (a divisor scheme).
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


def path_BIC_plot(model_list, highlight_ideal=False, ideal_kwargs=None, ax=None,**kwargs):
    """
    Plot the Bayes Information Criterion of the most likely path of each state model.

    Parameters
    ----------
    model : H2MM_list
        The set of optimizations to be compared, a H2MM_list object (a divisor scheme).
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
    collections = _stat_disc_plot(model_list, 'path_BIC',highlight_ideal=highlight_ideal, ideal_kwargs=ideal_kwargs, ax=ax,**kwargs)
    return collections


def raw_nanotime_hist(data, streams=None, stream_kwargs=None, ax=None, yscale='linear',
                      normalize=False, **kwargs):
    """
    Plot the histogram of nanotimes of photons (in bursts) per stream.
    Usefull for visualizing the fluorescence decays, and deciding where to place
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
    yscale: str optional
        The argument passed to the ax.set_yscale function.
        Primary options are 'linear' (default) and 'log'.
        The default is 'linear'
    normalize : bool, optional
        Whether to plot normalize the number of counts to the maximum per stream. 
        The default is False.
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
        List of path collections, per stream, from ax.plot of each nanotime decay.
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
    for hist, stream, s_kwargs in zip(hists, streams, stream_kwargs):
        in_kwargs = kwargs.copy()
        # add label name to kwarg dictionary
        in_kwargs.update({'label':stream.__str__()})
        if 'c' not in s_kwargs or 'color' not in s_kwargs or 'c' not in kwargs or 'color' not in kwargs:
            in_kwargs.update({'color':_color_dict[stream]})
        in_kwargs.update(s_kwargs)
        y = hist/hist.max() if normalize else hist
        collection = ax.plot(nanotime_bin, y, **in_kwargs)
        collections.append(collection)
    leg = ax.legend()
    ax.set_yscale(yscale)
    ax.set_xlabel("nanotime bin")
    if normalize:
        ax.set_ylabel("normalized counts")
    else:
        ax.set_ylabel("counts")
    return collections, leg


@_useideal
def state_nanotime_hist(model, states=None, state_kwargs=None, 
                        streams=frb.Ph_sel(Dex="Dem"), stream_kwargs=None, 
                        kwarg_arr=None, yscale='log', raw_bin=False, 
                        normalize=False,ax=None, **kwargs):
    """
    Plot the nanotime decays per state per stream of a given model.

    Parameters
    ----------
    model : H2MM_result
        The model containing the data to be plotted
    states : numpy.ndarray, optional
        Which states to plot, if None, all states plotted. 
        The default is None.
    state_kwargs : list of kwarg dicts, optional
        Kwargs passed per state. 
        The default is None.
    streams : list of fretbursts.Ph_sel, optional
        The stream(s) to inlcude. The default is [frb.Ph_sel(Dex="Dem"), ].
    stream_kwargs : list of kwarg dict, optional
        List of per-stream kwargs to pass to scatter or kdeplot. 
        The default is None.
    kwarg_arr : array of kwargs dicts, optional
        Array of dicts to use as kwargs for specific combinations of states/streams
        in data. Cannot be specified at same time as state_kwargs. If 2D, then will
        overwrite stream_kwargs, 2nd dimension, if exists specifies stream kwargs
        The default is None.
    yscale : str, optional
        Agrument passed to ax.set_yscale, primary options are 'log' and 'linear.
        The default is 'log'.
    raw_bin : bool, optional
        Whether to plot the raw nanotime bin (True) or convert into units of ns
        (False, default). The default is False.
    normalize : bool, optional
        Whether to plot normalize the number of counts to the maximum per stream. 
        The default is False.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Additional keyword arguments, passed to ax.plot.

    Returns
    -------
    collections : list[list[matplotlib.collections.PathCollection]]
        A list of lists of path collections returned by ax.plot, per state per stream.

    """
    ax = _check_ax(ax)
    states, streams, kwarg_arr = _process_kwargs(model, states, streams, state_kwargs, stream_kwargs, kwarg_arr)
    collections = list()
    for state, state_kwarr in zip(states, kwarg_arr):
        collections.append(list())
        for stream, kw_arr in zip(streams, state_kwarr):
            strm = np.argwhere([stream == s for s in model.parent.parent.ph_streams])[0,0]
            in_kwargs = kwargs.copy()
            in_kwargs.update({'label': f"State: {state}, Stream: {stream.__str__()}"})
            in_kwargs.update(kw_arr)
            x = np.arange(0,model.nanohist.shape[2],1)
            if not raw_bin:
                x = x * model.parent.parent.data.nanotimes_params[0]['tcspc_unit']*1e3
            y = model.nanohist[state, strm, :]/model.nanohist[state, strm, :].max() if normalize else model.nanohist[state, strm, :]
            collection = ax.plot(x, y, **in_kwargs)
            collections[-1].append(collection)
    if normalize:
        ax.set_ylabel("relative counts")
    else:
        ax.set_ylabel("counts")
    ax.set_yscale(yscale)
    if raw_bin:
        ax.set_xlabel("nanotime (ns)")
    else:
        ax.set_xlabel("nanotime bin")
    return collections


def axline_irf_thresh(data, horizontal=False, stream_kwargs=None, ax=None, **kwargs):
    """
    Plot lines indicating the positions of the IRF thresholds

    Parameters
    ----------
    data : BurstData
        The BurstData for which to show the IRF thresholds.
    horizontal : bool, optional
        Plot the lines vertiaclly (False) or horizontally (True).
        The default is False.
    stream_kwargs : list[dict], optional
        List of keyword arguments to pass the axvline or axhline per stream.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Keyword arguments to pass to axvline or axhline.

    Raises
    ------
    ValueError
        Incorrect format for stream_kwargs, most likely due to length.

    Returns
    -------
    lines : list[matplotlib.lines.Line2D]
        List of Lines returned by ax.axvline

    """
    ax = _check_ax(ax)
    axline = ax.axhline if horizontal else ax.axvline
    if stream_kwargs is None:
        if 'c' not in kwargs or 'color' not in kwargs:
            stream_kwargs = (_update_ret(kwargs, {'color':_color_dict[sel]}) for sel in data.ph_streams)
        else:
            stream_kwargs = repeat(kwargs)
    elif len(stream_kwargs) != data.nstream:
        raise ValueError(f"stream_kwargs must have the same number of elements as photon streams in data, got {len(stream_kwargs)} and {len(data.ph_streams)}")
    else:
        stream_kwargs = (_update_ret(kwargs, kw) for kw in stream_kwargs)
    lines = [axline(irf, **kw) for irf, kw in zip(data.irf_thresh, stream_kwargs)]
    return lines


def axline_divs(model_list, horizontal=False, stream_kwargs=None, ax=None, **kwargs):
    """
    Plot lines indicating the positions of divisors for the given H2MM_list with
    a divisor scheme

    Parameters
    ----------
    model_list : H2MM_list
        The H2MM_list for which to show positions of the divisors
    horizontal : bool, optional
        Plot the lines vertically (False) or horizontally (True).
        The default is False.
    stream_kwargs : list[dict], optional
        List of keyword arguments to pass the axvline or axhline per stream.
        The default is None.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Keyword arguments to pass to axvline or axhline.

    Raises
    ------
    ValueError
        Incorrect format for stream_kwargs, most likely due to length.

    Returns
    -------
    lines : list[list[matplotlib.lines.Line2D]]
        List of lists of Lines returned by ax.axvline

    """
    ax = _check_ax(ax)
    axline = ax.axhline if horizontal else ax.axvline
    if stream_kwargs is None:
        if 'c' not in kwargs or 'color' not in kwargs:
            stream_kwargs = (_update_ret(kwargs, {'color':_color_dict[sel]}) for sel in model_list.parent.ph_streams)
        else:
            stream_kwargs = repeat(kwargs)
    elif len(stream_kwargs) != model_list.parent.nstream:
        raise ValueError(f"stream_kwargs must have the same number of elements as photon streams in data, got {len(stream_kwargs)} and {len(model_list.parent.ph_streams)}")
    else:
        stream_kwargs = (_update_ret(kwargs, kw) for kw in stream_kwargs)
    lines = [[axline(dv, **kw) for dv in div] for div, kw in zip(model_list.divisor_scheme, stream_kwargs)]
    return lines


__ll_scatter_params = {'E':('E_rng', 'E_ll_rng', '_E',
                            lambda err: getattr(err, 'E_eval'), 
                            lambda err: getattr(err, 'E_space')),
                       'S':('S_rng', 'S_ll_rng', '_S',
                            lambda err: getattr(err, 'S_eval'), 
                            lambda err: getattr(err, 'S_space')),
                       'trans':('t_rate_rng', 't_ll_rng', '_trans',
                                lambda err: getattr(err, 'trans_eval'),
                                lambda err: getattr(err, 'trans_space')),
                       't':('t_rate_rng', 't_ll_rng', '_trans',
                            lambda err: getattr(err, 'trans_eval'),
                            lambda err: getattr(err, 'trans_space'))}

@_useideal
def ll_param_scatter(err, param, loc, ax=None, flex=None, thresh=None, space_kwargs=None,
                     rng_only=True, logscale=False, label_kwargs=None, **kwargs):
    """
    Generic function for plotting 1D scatter of how loglikelihood varies along a
    given model parameter

    Parameters
    ----------
    err : H2MM_result or Loglik_Error
        Model or loglik_error object to plot variability of loglikelihood along
        specified parameter.
    param : str
        which parameter to plot
    loc : tuple[int] or tuple[int, int]
        the state or transition to plot
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. 
        If None use default.
        The default is None.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is None.
    space_kwargs : dict, optional
        Dictionary of keyword arguments passed to the respective `E/S/trans_space()`
        function. These are `rng` to specify the range of values to scan, and 
        `steps`, which specifies how many points to place evenly within the range.
        The default is None
    rng_only: bool, optional
        If `True` only plot the specified range around the optimal parameter
        (the result of the :meth:`burstH2MM.ModelError.Loglik_Error.E_space`, 
        :meth:`burstH2MM.ModelError.Loglik_Error.S_space` or 
        :meth:`burstH2MM.ModelError.Loglik_Error.trans_space` functions), if `False`
        plot all values evaluated ever for the logliklihood varried across the
        given parameter. The default is True.
    logscale : bool, optional
        Whether or not to plot the x axis in a log-scale. The default is False.
    label_kwargs : dict
        Keyword arguments passed to ax.set_xlabel and ax.set_ylabel
    **kwargs : dict
        Keyword arguments passed to ax.scatter.

    Returns
    -------
    ret : matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.

    """
    ax = _check_ax(ax)
    if isinstance(err, BurstSort.H2MM_result):
        err = err.loglik_err
    if param not in __ll_scatter_params:
        raise ValueError(f"param must be 'E', 'S', 't', or 'trans', got {param}")
    attr_rng, attr_ll, attr_err, attr_eval, attr_space = __ll_scatter_params[param]
    attr_eval, attr_space = attr_eval(err), attr_space(err)
    x = getattr(err, attr_rng)[loc]
    y = getattr(err, attr_ll)[loc]
    if space_kwargs is not None or rng_only:
        if np.any(getattr(err, attr_err).mask[loc]) and (not isinstance(space_kwargs['rng'], Iterable) or len(space_kwargs) <= 2):
            attr_eval(locs=(loc, ), thresh=thresh, flex=flex)
        x_t, y_t = attr_space(loc, **space_kwargs if space_kwargs is not None else dict())
        if rng_only:
            x, y = x_t, y_t
    if label_kwargs is None:
        label_kwargs = dict()
    elif not isinstance(label_kwargs, dict):
        raise TypeError(f"label_kwargs must be dict, got {type(label_kwargs)}")
    if x.size == 0:
        attr_eval(locs=(loc, ), thresh=thresh, flex=flex)
        attr_space(loc)
        x = getattr(err, attr_rng)[loc]
        y = getattr(err, attr_ll)[loc]
    if logscale:
        ax.set_xscale('log')
    ret = ax.scatter(x,y, **kwargs)
    xlabel = err.parent.param_labels[param]
    xlabel += f" state {loc[0]}" if len(loc)==1 else f" from state {loc[0]} to state {loc[1]}"
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel("LL", **label_kwargs)
    return ret

def ll_E_scatter(err, state, ax=None, rng=None, steps=20, **kwargs):
    """
    Plot how the loglikelihood decreases when varying the FRET efficiency of a
    specified state away from the optimal value.
    
    .. note::
        
        This method is a wrapper around :func:`ll_param_scatter`


    Parameters
    ----------
    err : H2MM_result or Loglik_Error
        Model or loglik_error object to plot variability of loglikelihood along
        specified parameter.
    state : int
        State for which to plot the variability of loglikelihood with varied E
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) S vals, or factor by which to multiply
        error range, or array of S values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. 
        If None use default.
        The default is None.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is None.
    space_kwargs : dict, optional
        Dictionary of keyword arguments passed to the respective `E/S/trans_space()`
        function. These are `rng` to specify the range of values to scan, and 
        `steps`, which specifies how many points to place evenly within the range.
        The default is None
    rng_only: bool, optional
        If `True` only plot the specified range around the optimal parameter
        (the result of the :meth:`burstH2MM.ModelError.Loglik_Error.E_space`, 
        :meth:`burstH2MM.ModelError.Loglik_Error.S_space` or 
        :meth:`burstH2MM.ModelError.Loglik_Error.trans_space` functions), if `False`
        plot all values evaluated ever for the logliklihood varried across the
        given parameter. The default is True.
    label_kwargs : dict
        Keyword arguments passed to ax.set_xlabel and ax.set_ylabel
    **kwargs : dict
        Keyword arguments passed to ax.scatter.

    Raises
    ------
    ValueError
        Invalid input to state.

    Returns
    -------
    ret : matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    if not isinstance(state, int):
        raise ValueError("state must be an int")
    ret = ll_param_scatter(err, 'E', (state, ), ax=ax, 
                           space_kwargs={'rng':rng, 'steps':steps}, **kwargs)
    return ret

def ll_S_scatter(err, state, ax=None, rng=None, steps=20, **kwargs):
    """
    Plot how the loglikelihood decreases when varying the stoichiometry of a
    specified state away from the optimal value.
    
    .. note::
        
        This method is a wrapper around :func:`ll_param_scatter`

    Parameters
    ----------
    err : H2MM_result or Loglik_Error
        Model or loglik_error object to plot variability of loglikelihood along
        specified parameter.
    state : int
        State for which to plot the variability of loglikelihood with varied E
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) S vals, or factor by which to multiply
        error range, or array of S values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. 
        If None use default.
        The default is None.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is None.
    rng_only: bool, optional
        If `True` only plot the specified range around the optimal parameter
        (the result of the :meth:`burstH2MM.ModelError.Loglik_Error.E_space`, 
        :meth:`burstH2MM.ModelError.Loglik_Error.S_space` or 
        :meth:`burstH2MM.ModelError.Loglik_Error.trans_space` functions), if `False`
        plot all values evaluated ever for the logliklihood varried across the
        given parameter. The default is True.
    label_kwargs : dict
        Keyword arguments passed to ax.set_xlabel and ax.set_ylabel
    **kwargs : dict
        Keyword arguments passed to ax.scatter.

    Raises
    ------
    ValueError
        Invalid input to state.

    Returns
    -------
    ret : matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    if not isinstance(state, int):
        raise ValueError("state must be an int")
    ret = ll_param_scatter(err, 'S', (state, ), ax=ax,
                           space_kwargs={'rng':rng, 'steps':steps}, **kwargs)
    return ret

def ll_trans_scatter(err, from_state, to_state, ax=None, rng=None, steps=20, logscale=True, **kwargs):
    """
    Plot how the loglikelihood decreases when varying the transition rate of a
    specified (from_state to_state) pair away from the optimal value.
    
    .. note::
        
        This method is a wrapper around :func:`ll_param_scatter`


    Parameters
    ----------
    err : H2MM_result or Loglik_Error
        Model or loglik_error object to plot variability of loglikelihood along
        specified parameter.
    from_state : int
        The state the system transitions from.
    to_state : int
        The state the system transitions to.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) S vals, or factor by which to multiply
        error range, or array of S values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.
    logscale : bool, optional
        Whether or not to plot the x axis in a log-scale. The default is True.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. 
        If None use default.
        The default is None.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is None.
    rng_only: bool, optional
        If `True` only plot the specified range around the optimal parameter
        (the result of the :meth:`bursH2MM.ModelError.Loglik_Error.E_space`, 
        :meth:`burstH2MM.ModelError.Loglik_Error.S_space` or 
        :meth:`burstH2MM.ModelError.Loglik_Error.trans_space` functions), if `False`
        plot all values evaluated ever for the logliklihood varried across the
        given parameter. The default is True.
    label_kwargs : dict
        Keyword arguments passed to ax.set_xlabel and ax.set_ylabel
    **kwargs : dict
        Keyword arguments passed to ax.scatter.

    Raises
    ------
    ValueError
        Invalid input to either from_state or to_state.

    Returns
    -------
    matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    if not isinstance(from_state, int):
        raise ValueError("from_state must be an int")
    if not isinstance(to_state, int):
        raise ValueError("to_state must be an int")
    ret = ll_param_scatter(err, 'trans', (from_state, to_state), ax=ax, logscale=logscale, 
                           space_kwargs={'rng':rng, 'steps':steps}, **kwargs)
    return ret

@_useideal
def covar_param_ll_scatter(err, param, state, ax=None, label_kwargs=None,**kwargs):
    """
    Plot the loglikelihood of the covariance of a given parameter type and state/
    transition.

    Parameters
    ----------
    err : Loglik_Error
        Error object to plot.
    param : str
        parameter to plot, one of the parameters of :class:`ModelSet`.
    state : tuple[int] or tuple[int, int]
        State or transition to plot (the state/transition which was fixed for the
        given parameter type during optimizations).
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    label_kwargs : dict
        Keyword arguments passed to ax.set_xlabel and ax.set_ylabel
    **kwargs : dict
        Dictionary of keyword arguments passed to ax.scatter.

    Returns
    -------
    matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    if isinstance(err, BurstSort.H2MM_result):
        err = err.loglik_err
    if label_kwargs is None:
        label_kwargs = dict()
    elif not isinstance(label_kwargs, dict):
        raise TypeError(f"label_kwargs must be dict, got {type(label_kwargs)}")
    ax = _check_ax(ax)
    paramt = param.split("_corr")[0]
    modarray = getattr(err, paramt+"_covar")
    if modarray.mask[state]:
        covar_eval = getattr(err, "covar_"+paramt)
        covar_eval(state)
        modarray = getattr(err, paramt+"_covar")
    modset = modarray[state]
    x = getattr(modset, param)[(slice(None), ) + state]
    y = modset.loglik
    ret = ax.scatter(x, y, **kwargs)
    xlabel = err.parent.param_labels[param]
    xlabel += f" state {state[0]} (fixed)" if len(state) == 1 else f" from state {state[0]} to state {state[1]} (fixed)"
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel("LL", **label_kwargs)
    return ret


def covar_E_ll_scatter(err, state, ax=None, add_corrections=False,**kwargs):
    """
    Plot the loglikelihood of the covariance of E along a given state.

    Parameters
    ----------
    err : Loglik_Error
        Error object to plot.
    state : int
        Which state to plot with optimization holding E of that state fixed.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Dictionary of keyword arguments passed to ax.scatter.

    Returns
    -------
    matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    Eparam = "E_corr" if add_corrections else "E"
    ret = covar_param_ll_scatter(err, Eparam, (state, ), ax=ax, **kwargs)
    return ret


def covar_S_ll_scatter(err, state, ax=None, add_corrections=False,**kwargs):
    """
    Plot the loglikelihood of the covariance of S along a given state.

    Parameters
    ----------
    err : Loglik_Error
        Error object to plot.
    state : int
        Which state to plot with optimization holding S of that state fixed.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Dictionary of keyword arguments passed to ax.scatter.

    Returns
    -------
    matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    Sparam = "S_corr" if add_corrections else "S"
    ret = covar_param_ll_scatter(err, Sparam, (state, ), ax=ax, **kwargs)
    return ret
    
    
def covar_trans_ll_scatter(err, from_state, to_state, ax=None, **kwargs):
    """
    Plot the loglikelihood of the covariance of transition rate along a given
    transition.

    Parameters
    ----------
    err : Loglik_Error
        Error object to plot.
    from_state : int
        State from which the system transitions.
    to_state : int
        State to which the system transitions.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    **kwargs : dict
        Dictionary of keyword arguments passed to ax.scatter.

    Returns
    -------
    matplotlib.collections.PathCollection
        Path collection returned by ax.scatter.
    
    """
    state = (from_state, to_state)
    ret = covar_param_ll_scatter(err, "trans", state, ax=ax, **kwargs)
    return ret


def _find_burst(model, burst):
    """
    Find the best burst according to the specified criterion.

    Parameters
    ----------
    model : H2MM_result
        :class:`H2MM_result <burstH2MM.BurstData.H2MM_result>` object to find the best burst from.
    burst : int or str
        If int, the burst index, and drectly returned. If a string, then what
        criterion to look for the best burst.
        
        Options:
        
            1. 'states': find the burst with the most states present
            2. 'transitions': find the burst with the most transition present
            3. 'longest': longest burst in data
            4. 'photons': burst with the most photons
            
        For criterion based on maximizing a dwell-based parameter, will return
        the parameter that maximizes the given criterion first, and then among
        all of those bursts maximize other dwell-based criterion as well.

    Raises
    ------
    IndexError
        burst input is too large for the given data.
    TypeError
        burst was not either a string or an integer.

    Returns
    -------
    int
        Index 

    """
    if np.issubdtype(type(burst), np.integer) :
        if burst >= len(model.parent.index):
            raise IndexError(f'Burst out of range for model with {len(model.parent.index)} bursts, and given {burst}')
        return burst
    elif isinstance(burst, str):
        state_num = (model.burst_state_counts != 0).sum(axis=0)
        max_state = state_num.max() == state_num
        trans_num = model.burst_dwell_num
        max_trans = trans_num.max() == trans_num
        if burst.lower() == 'states':
            max_states = np.argwhere(max_state).reshape(-1)
            trans_max = np.argmax(trans_num[max_state])
            return max_states[trans_max]
        elif burst.lower() == 'transitions':
            max_transs = np.argwhere(max_trans).reshape(-1)
            state_max = np.argmax(state_num[max_trans])
            return max_transs[state_max]
        elif burst.lower() == 'longest':
            return np.argmax([t[-1]-t[1] for t in model.parent.parent.times])
        elif burst.lower() == 'photons':
            return np.argmax([idx.shape[0] for idx in model.parent.index])
        else:
            return 0
    else:
        raise TypeError(f"burst must be integer, 'states' or 'transitions', got {type(burst)}")


@_useideal
def plot_burst_path(model, burst, param='E', ax=None, state_color=None, linewidth=None, stream=None,**kwargs):
    """
    Plot the state trajectory of a burst in a model.

    Parameters
    ----------
    model : H2MM_result
        The :class:`H2MM_result <burstH2MM.BurstData.H2MM_result>` object from which 
        to plot the state path of the specified
        burst.
    burst : int
        Index of the burst.
    param : str, optional
        Name of the parameter to use as the parameter value to plot for each state. 
        The default is 'E'.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    state_color : 1-D array-like, optional
        Color of lines for each state. The default is None.
    linewidth : 1-D array-like, optional
        Width of each state. The default is None.
    **kwargs : dict
        kwargs passed to maptlotlib.collections.LineCollection.

    Returns
    -------
    matplotlib.collections.LineCollections
        The line collection added to the axes.
    l   
    """
    burst = _find_burst(model, burst)
    if state_color is None:
        state_color = ['b' for _ in range(model.nstate+1)]
    elif len(state_color) == model.nstate:
        state_color += ('grey',)
    ax = _check_ax(ax)
    times, path = model.parent.parent.times[burst], model.path[burst]
    times = (times - times[0]) * model.parent.parent.data.clk_p*1e3
    # get locations of transitions
    tloc = model.trans_locs[burst]
    tlocb, tloce = tloc[:-1], tloc[1:] - 1
    tms_b, tms_e, pth = times[tlocb], times[tloce], path[tlocb]
    pth_e = getattr(model, param)[pth]
    if stream is not None and pth_e.ndim == 2:
        if isinstance(stream, int):
            pth_e = pth_e[:,stream]
        elif isinstance(stream, frb.Ph_sel):
            stream = np.argwhere([stream == ph_sel for ph_sel in model.parent.parent.ph_streams])[0,0]
            pth_e = pth_e[:,stream]
        else:
            raise TypeError(f"stream must be int, or fretbursts.Ph_sel, got {type(stream)}")
    elif stream is not None:
        raise ValueError("stream must be None for non-stream base parameters")
    # build path array
    linepath = np.empty((2*tlocb.shape[0],2,2), dtype=float)
    with np.nditer([tms_b, tms_e, pth_e], flags=['f_index']) as idx:
        for xb, xe, y in idx:
            linepath[2*idx.index, 0, :] = xb, y
            linepath[2*idx.index, 1, :] = xe, y
            linepath[2*idx.index+1, 0, :] = xe, y
            linepath[2*idx.index-1, 1, :] = xb, y
    linepath = linepath[:-1]
    pthc = np.array([pth, np.ones(pth.shape, dtype=int)*model.nstate]).astype(int).T.reshape(-1)[:-1]
    clr = [state_color[i] for i in pthc]
    lc = LineCollection(linepath, color=clr, linewidth=linewidth, **kwargs)
    ax.add_artist(lc)
    ax.set_xlim((times[0], times[-1]))
    return lc


def _stream_color_map(stream_map, index_red, streams, idx_keep, index_keep, name, exclude, kwargs, nonefill):
    """
    Build colors input to LineCollection from inputs

    Parameters
    ----------
    stream_map : TYPE
        DESCRIPTION.
    index_red : TYPE
        DESCRIPTION.
    streams : TYPE
        DESCRIPTION.
    idx_keep : TYPE
        DESCRIPTION.
    index_keep : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    exclude : TYPE
        DESCRIPTION.
    kwargs : TYPE
        DESCRIPTION.
    nonefill : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.
    TypeError
        DESCRIPTION.

    Returns
    -------
    colors : list
        List of colors for LineCollection.

    """
    if stream_map is not None:
        if isinstance(stream_map, dict):
            if all(isinstance(key, frb.Ph_sel) for key in stream_map.keys()):
                sel = {i:stream_map[streams[np.argwhere([i in idx for idx in index_keep])[0,0]]] for i in idx_keep}
                colors = [sel[i] for i in index_red]
            elif all(np.issubdtype(type(key), np.integer) for key in stream_map.keys()):
                colors = [stream_map[i] for i in index_red]
            else:
                raise ValueError(f"Incompatible input to {name}, must be dict with keys of type frebtursts.Ph_sel or int")
        elif isinstance(stream_map, np.ndarray):
            if len(stream_map) != idx_keep.shape[0]:
                raise ValueError(f"Incompatible size for {name}")
            colors = [stream_map[i] for i in index_red]
        else:
            raise TypeError(f"{name} must be dict, numpy array or None, got {type(stream_map)}")
        if any(ex in kwargs for ex in exclude):
            warnings.warn(f"Specifying {name} will override any argumnets in {exclude} from being passed to ax.scatter")
    elif stream_map is None:
        if any(ex in kwargs for ex in exclude):
            for ex in exclude:
                if ex in kwargs:
                    colors = kwargs.pop(ex)
                    break
        else:
            colors = index_red if nonefill else None
            
    return colors

def plot_burst_index(data, burst, ax=None, colapse_div=False, streams=None, 
                     stream_pos=None, stream_color=None, rng=None, invert=False, 
                     stream_labels=None, stream_edge=None, **kwargs):
    """
    Plot indexes of a given burst

    Parameters
    ----------
    data : H2MM_resutl, H2MM_list or BurstData
        Parent data object, either :class:`H2MM_list <burstH2MM.BurstSort.H2MM_list>` 
        or :class:`BurstData <burstH2MM.BurstSort.BurstData>`, if the later will 
        use non-divisor models indexes, if the former, then will divide streams by 
        the divisor scheme used.
    burst : int
        Which burst to plot.
    ax : matplotlib.axes or None, optional
        The axes where the plot will be placed. The default is None.
    colapse_div : bool, optional
        If using a divisor scheme, whether or not to transform into the base 
        photon streams without divisors. The default is False.
    streams : list[fretbursts.Ph_sel], optional
        Which streams to plot, if None will use all streams in input data. 
        The default is None.
    stream_pos : dict or numpy.ndarray, optional
        Dictionary of {stream:position} identifying the y location for each photon
        index. The default is None.
    stream_color : dict or array-like, optional
        Colors for each index. The default is None.
    rng : array-like, optional
        If stream_pos not specified, the range over which the indexes will be
        distributed. The default is None.
    invert : bool, optional
        If True, then indexes plotted from top to bottom, if False, first indexes 
        plotted from bottom to top. The default is False.
    stream_labels : bool, dict, optional
        Whether or not to display tick labels on y-axis, provided as dictionary:
        {stream:label} where stream is either fretbursts.Ph_sel or int, and
        label is str. The default is None.
    stream_edge : dict or array-like, optional
        Edge-colors for each index. The default is None.
    **kwargs : dict
        Kwargs passed to ax.scatter() .

    Raises
    ------
    ValueError
        Invalid values in one or more kwargs
    TypeError
        Incorrect type for one or more kwargs.

    Returns
    -------
    ret : matplotlib.collections.PathCollection
        Result of ax.scatter for plotting photons.

    """
    # set the input data to H2MM_list
    if isinstance(data, BurstSort.BurstData):
        data = data.models
    elif isinstance(data, BurstSort.H2MM_result):
        burst = _find_burst(data, burst)
        data = data.parent
    if isinstance(burst, str):
        burst = _find_burst(data.opts[data.ideal], burst)
    if streams is None:
        streams = data.parent.ph_streams
    if stream_pos is not None and rng is not None:
        raise ValueError("Cannot specify both stream_pos and rng keyword arguments at the same time")
    elif isinstance(stream_pos, dict):
        # check that keys of stream_pos are correct
        if all(isinstance(key, frb.Ph_sel) for key in stream_pos.keys()):
            sel_dict = True
        elif all(np.issubdtype(type(key), np.integer) for key in stream_pos.keys()):
            sel_dict = False
        else:
            raise ValueError("Invalid values for stream_pos keys")
    elif isinstance(stream_pos, np.ndarray):
        pos = stream_pos
    elif rng is None and stream_pos is None:
        rng = (0.0, 1.0)

    ax = _check_ax(ax)
    times, index = data.parent.times[burst], data.index[burst]

    times = (times - times[0]) * data.parent.data.clk_p * 1e3
    # find which indices will be used
    index_all = np.arange(data.ndet)
    index_keep = [index_all[data._get_stream_slc(sel)] for sel in streams]
    idx_keep = np.concatenate(index_keep)
    # remove unused indexes and if colapse_div, reduce to base streams
    if colapse_div:
        sz = len(streams) + 1
        mask = np.array([i in idx_keep for i in index])
        dmap = dict()
        for i, div in enumerate(index_keep):
            dmap.update({j:i for j in div})
        times = times[mask]
        index = index[mask]
        index_red = np.array([dmap[i] for i in index])
    else:
        sz = idx_keep.size + 1
        mask = np.array([i in idx_keep for i in index])
        times = times[mask]
        index_red = index[mask]
    if stream_pos is None or isinstance(stream_pos, np.ndarray):
        if stream_pos is None:
            pos = np.array([i / sz for i in range(1,sz)])
            if invert:
                pos = 1 - pos
            pos += rng[0]
            pos *= rng[1] - rng[0]
        else:
            pos = stream_pos
        index_pos = pos[index_red]
    else:
        if sel_dict:
            pos_map = dict()
            for idx, stream in zip(index_keep, streams):
                pos_map.update({i:stream_pos[stream] for i in idx})
            index_pos = np.array([pos_map[i] for i in index_red])
        else:
            index_pos = np.array([stream_pos[i] for i in index_red])
    colors = _stream_color_map(stream_color, index_red, streams, idx_keep, index_keep, 
                               'stream_color', ('c', 'color'), kwargs, True)
    edges = _stream_color_map(stream_edge, index_red, streams, idx_keep, index_keep, 
                              'stream_edge', ('ec', 'edgecolor'), kwargs, False)
    ret = ax.scatter(times, index_pos, c=colors, ec=edges,**kwargs)
    if stream_labels is True:
        if colapse_div or data.div_map[-1] == len(data.parent.ph_streams):
            labels = [str(ph_sel) for ph_sel in streams]
        else:
            labels = list(chain.from_iterable((f'{sel} {i}' for i, _ in enumerate(idx)) for sel, idx in zip(streams, index_keep)))
    elif isinstance(stream_labels, dict):
        if all(isinstance(key, frb.Ph_sel) for key in stream_labels.keys()):
            if colapse_div:
                labels = [stream_labels[ph_sel] for ph_sel in streams]
            else:
                labels = list(chain.from_iterable((stream_labels[sel] for _ in idx) for sel, idx in zip(streams, index_keep)))
        elif all(np.issubdtype(type(key), np.integer) for key in stream_labels.keys()):
            labels = [stream_labels[i] for i in idx_keep]
    elif stream_labels not in (None, False):
        raise TypeError(f"tick_labels must be None, bool, or dictionary of fretbursts.Ph_sel or int, got {type(stream_labels)}")
    if stream_labels not in (None, False):
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
    return ret


__stream_color = {frb.Ph_sel(Dex='Dem'):'g', frb.Ph_sel(Dex='Aem'):'r', frb.Ph_sel(Aex='Aem'):'purple'}

@_useideal
def plot_burstjoin(model, burst, ax=None, add_corrections=False, state_color=None,
                   stream_color=None):
    """
    Wrapper function plots a burst with E, S paths and burst photons.

    Parameters
    ----------
    model : H2MM_result
        The :class:`H2MM_result <burstH2MM.BurstData.H2MM_result>` object from 
        which to plot the state path of the specified
        burst.
    burst : int or str
        Index of the burst, or one of {'states', 'transitions'} to find the burst
        with either the most states, or transitions automatically.
    add_corrections : bool, optional
        Plot correct E/S values or raw. The default is False.
    state_color : 1-D array-like, optional
        Color of lines for each state. The default is None.
    stream_color : dict, optional
        Color for each stream, given as dict {stream:color} where stream is 
        fretbursts.Ph_sel and color is a matplotlib color specification. 
        The default is None.

    Returns
    -------
    None.

    """
    ax = _check_ax(ax)
    burst = _find_burst(model, burst)
    E, S = ('E_corr', 'S_corr') if add_corrections else ('E', 'S')
    if model._hasE:
        axE = ax.twinx()
        plot_burst_path(model, burst, ax=axE, param=E, state_color=state_color)
        Elim = [-1.0, 1.0] if model._hasS else [0.0, 1.0]
        axE.set_ylim(Elim)
        axE.yaxis.set_ticks_position('left')
        ticks = axE.get_yticks()
        ticks = ticks[(ticks > 0.0)* (ticks < 1.0)]
        axE.set_yticks(ticks)
        axE.set_ylabel('E')
        axE.yaxis.set_label_coords(-0.15, 0.5, transform=axE.get_yaxis_transform())
    if model._hasS:
        axS = ax.twinx()
        plot_burst_path(model, burst, ax=axS, param=S, state_color=state_color)
        Slim = [0.0, 2.0] if model._hasE else [0.0, 1.0]
        axS.set_ylim(Slim)
        axS.yaxis.set_ticks_position('left')
        ticks = axS.get_yticks()
        ticks = ticks[(ticks > 0.0)* (ticks < 1.0)]
        axS.set_yticks(ticks)
        axS.set_ylabel('S')
        axS.yaxis.set_label_coords(-0.15, 0.5, transform=axS.get_yaxis_transform())
    if stream_color is None:
        stream_color = {sel:__stream_color[sel] if sel in __stream_color else 'b' for sel in model.parent.parent.ph_streams}
    plot_burst_index(model.parent, burst, ax=ax, colapse_div=True, 
                     stream_color=stream_color, invert=True, stream_labels=True)
    ax.yaxis.set_ticks_position('right')
    ax.set_ylim([0,1])