#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Module: BurstSort
# Author: Paul David Harris
# Created: 1 Jun 2022
# Modified 17 Jul 2022
# Purpose: Core classes for burstH2MM
"""
.. _burstsort:
    
BurstSort
=========

The core of burstH2MM, this module provides the base classes for performing and
anlysizing H2MM resutls of burst data provided by FRETBursts
"""

import numpy as np
import warnings
import fretbursts as frb
import H2MM_C as h2
from itertools import chain


def _single_div(data, ph_stream):
    """
    Get the excitation window of a given stream

    Parameters
    ----------
    data : FRETBursts Data
        data object with burst search and selection performed
    ph_stream : FRETBursts Ph_sel
        photon selection

    Returns
    -------
    ret : 2 element numpy array
        The excitation window of the given stream

    """
    if data.alternated:
        if ph_stream.Dex is not None and ph_stream.Aex is None:
            ret = np.array([data.D_ON[0], data.D_ON[1] + 1])
        elif ph_stream.Dex is None and ph_stream.Aex is not None:
            ret = np.array([data.A_ON[0], data.A_ON[1] + 1])
        else:
            ret = np.array([np.min([data.D_ON[0], data.A_ON[0]]), np.max([data.D_ON[1], data.A_ON[1]])+1])
    else:
        ret = np.array([0, data.nanotimes_params[0]['tcspc_num_bins']])
    return ret


def _make_divs(data, ph_streams):
    """
    Get limits of divisors for each stream (the excitation window of each stream)

    Parameters
    ----------
    data : FRETBursts Data
        data object with burst search and selection performed
    ph_streams : list[fretbursts.Ph_sel]
        photon streams of each desired index

    Returns
    -------
    list
        2 element numpy arrays of limits of divisors for a each stream

    """
    return [_single_div(data, stream) for stream in ph_streams]



def _sort_base(data, ph_streams):
    """
    Creates lists of numpy arrays of bursts in data object, uses ph_streams to
    identify the index mapping

    Parameters
    ----------
    data : FRETBursts Data
        data object with burst search and selection performed
    ph_streams : list[fretbursts.Ph_sel]
        photon streams of each desired index

    Returns
    -------
    index : list[numpy.ndarray]
        indexes of photons in bursts
    time : list[numpy.ndarray]
        Times of photons in bursts
    masks_ch : list[numpy.ndarray]
        masks of photons in ph_streams for each channel, used to make sorting
        nanotimes, and particles more efficient later

    """
    index_ch = [np.array([(j+1)*data.get_ph_mask(ich=i, ph_sel=sel) 
                      for j, sel in enumerate(ph_streams)], dtype=np.uint32).sum(axis=0).astype(np.uint32) 
            for i in range(data.nch)]
    masks_ch = [index != 0 for index in index_ch]
    index_ch = [index - 1 for index in index_ch]
    times_ch = data.ph_times_m
    time, index = list(), list()
    for i, mburst in enumerate(data.mburst):
        times, indexes, masks = times_ch[i], index_ch[i], masks_ch[i]
        for istart, istop in zip(mburst.istart, mburst.istop+1):
            time.append(times[istart:istop][masks[istart:istop]])
            index.append(indexes[istart:istop][masks[istart:istop]])
            if time[-1].size < 4:
                raise TypeError("One or bursts has too few photons for photons streams given apply higher threshholds during burst selection")
    return index, time, masks_ch

def _sort_nanos(data, masks_ch):
    """
    Generate list of nanotimes in bursts from data and masks_ch

    Parameters
    ----------
    data : FRETBursts Data object
        DESCRIPTION.
    masks_ch : list[numpy.ndarray]
        Which photons belong to photons streams used in ph_streams

    Returns
    -------
    nano : list[numpy.ndarray]
        Nanotimes of used photons within bursts

    """
    nano = list()
    for i, mburst in enumerate(data.mburst):
        nanos, mask = data.nanotimes[i], masks_ch[i]
        for istart, istop in zip(mburst.istart, mburst.istop+1):
            nano.append(nanos[istart:istop][mask[istart:istop]])
    return nano

def _sort_pars(data, masks_ch):
    """
    Generate list of particles of photons in bursts from data and masks_ch

    Parameters
    ----------
    data : FRETBursts Data object
        Data with bursts identified
    masks_ch : list of numpy arrays
        Which photons belong to photons streams used in ph_streams

    Returns
    -------
    particles : list of numpy arrays
        particles of used photons within bursts

    """
    particles = list()
    for i, mburst in enumerate(data.mburst):
        pars, mask = data.particles_m[i], masks_ch[i]
        for istart, istop in zip(mburst.istart, mburst.istop+1):
            particles.append(pars[istart:istop][mask[istart:istop]])
    return particles

def _sort_by_time(times, *args):
    """
    Sort photons by times, used to make sure all photons are in order for usALEX
    experiments

    Parameters
    ----------
    times : list[numpy.ndarray]
        Times of photons in bursts
    *args : list[numpy.ndarray]
        Other in of photons in bursts, which will also be reordered

    Returns
    -------
    times : list[numpy.ndarray]
        Times of photons in bursts, sorted
    *args: list of numpy arrays
        Other parameters of photons in bursts, sorted

    """
    for i in range(len(times)):
        sort = np.argsort(times[i])
        times[i] = times[i][sort]
        for j in len(args):
            args[j][i] = args[j][i][sort]
    return times, *args

def _sort_nsALEX(data, ph_streams):
    """
    Generate burst arrays from data object given index assigments in ph_streams,
    for measurements with pulsed excitation

    Parameters
    ----------
    data : FRETBursts.Data
        data object with burst search and selection performed
    ph_streams : list[fretbursts.Ph_sel]
        photon streams of each desired index

    Returns
    -------
    indexs : list[numpy.ndarray]
        The indeces of photosn in bursts
    times : list[numpy.ndarray]
        the macrotimes of photons in bursts
    nanos : list[numpy.ndarray]
        The nanotimes of photons in bursts
    particles : list[numpy.ndarray]
        If the data object has particles (data was simulated), the particle ids
        of photons within bursts, if data is real data, None

    """
    indexs, times, masks_ch = _sort_base(data, ph_streams)
    nanos = _sort_nanos(data, masks_ch)
    if hasattr(data, 'particles_m'):
        particles = _sort_pars(data, masks_ch)
    else:
        particles = None
    return indexs, times, nanos, particles

def _sort_usALEX(data, ph_streams, Aex_stream, Aex_shift):
    """
    Generate burst arrays from data object given index assigments in ph_streams,
    for measurements with continuous wave excitation

    Parameters
    ----------
    data : fretbursts.Data
        data object with burst search and selection performed
    ph_streams : list[fretbursts.Ph_sel]
        photon streams of each desired index
    Aex_stream : fretbursts.Ph_sel or list[fretbursts.Ph_sel]
        The stream(s) of Aex photons
    Aex_shift : str or None
        'shift', 'rand' or 'even' if shifting of Aex photosn to take place, None
        otherwise

    Returns
    -------
    indexs : list[numpy.ndarray]
        The indeces of photosn in bursts
    times : list[numpy.ndarray]
        the macrotimes of photons in bursts
    particles : list[numpy.ndarray] or None
        If the data object has particles (data was simulated), the particle ids
        of photons within bursts, if data is real data, None

    """
    indexes, times, masks_ch = _sort_base(data, ph_streams)
    if hasattr(data, 'particles_m'):
        particles = _sort_pars(data, masks_ch)
    else:
        particles = None
    if type(Aex_shift) == frb.Ph_sel:
        Aex_shift = [Aex_shift, ]
    if Aex_shift:
        Aex = list()
        for i in range(len(ph_streams)):
            if ph_streams[i] in Aex_stream:
                Aex.append(i)
        Aex_mask = [np.array([index == A for A in Aex]).sum(axis=0) != 0 for index in indexes]
    if Aex_shift =='shfit':
        alex_shift = data.D_ON[0] - data.A_ON[0]
        for i in range(len(times)):
            times[i][Aex_mask[i]] += alex_shift
    elif Aex_shift == 'rand':
        D_ON, D_OFF = data.D_ON[0], data.D_OFF[1]
        for i in range(len(times)):
            aex_rnd = np.random.randint(D_ON, D_OFF, size=Aex_mask[i].sum())
            times[i][Aex_mask[i]] = aex_rnd + (times[Aex_mask[i]] // data.alex_period)*data.alex_period
    elif Aex_shift == 'even':
        D_ON, D_OFF = data.D_ON[0], data.D_OFF[1]
        D_dur = D_OFF - D_ON
        for i in range(len(times)):
            tms, inv, cnts = np.unique(times[i][Aex_mask[i]]//data.alex_period, 
                                       return_inverse=True, return_counts=True)
            mask = Aex_mask[i]
            time = times[i]
            Aex_times = np.empty(inv.shape, dtype=np.uint64)
            for j, (tm, cnt) in enumerate(zip(tms, inv)):
                t_beg = tm*data.alex_period + D_ON + D_dur/(cnt+1)
                t_end = tm*data.alex_period + D_OFF
                Aex_times[j==inv] = np.linspace(t_beg, t_end, cnt).astype(np.uint64)
            time[mask] = Aex_times
    if Aex_shift:
        if particles:
            times, indexes, particles = _sort_by_time(times, indexes, particles)
        else:
            times, indexes = _sort_by_time(times, indexes)
    return indexes, times, particles

def _complete_div(data, divs, streams):
    """
    Add beginning and ending of excitation window to divisors

    Parameters
    ----------
    data : fretbursts.Data
        data with bursts selected
    divs : list[numpy.ndarray]
        Divisors to be completed
    streams : list[numpy.ndarray]
        streams of each set of divisors

    Returns
    -------
    n_divs : list[numpy.ndarray]
        Completed divisor arrays

    """
    n_divs = [np.concatenate([[stream[0]], div, [stream[1]]]) for stream, div in zip(_make_divs(data, streams), divs)]
    if np.any(np.concatenate([np.diff(div) <= 0 for div in n_divs])):
        raise ValueError("Divs must be in asscending order and contained within excitation window")
    return n_divs
        


def _divisor_sort(indexs, nanos, divs):
    """
    Generate new indexes based on a set of divisors from nanotimes and indexes

    Parameters
    ----------
    indexs : list[numpy.ndarray]
        indexes of photons in bursts based on photon stream, with no divisors applied
    nanos : list[numpy.ndarray]
        nanotimes of photons in bursts
    divs : list[numpy.ndarray]
        Divisors (including start and end of excitation window) for each stream.
        This defines a new divisor scheme

    Returns
    -------
    new_index : list[numpy.ndarray]
        Indexes of photons in bursts in new divisor scheme
    
    """
    div_map = np.cumsum([0] + [div.size - 1 for div in divs])
    new_index = list()
    for index, nano in zip(indexs, nanos):
        index_n = np.empty(index.size, dtype=np.uint32)
        for i, (shift, div) in enumerate(zip(div_map[:-1], divs)):
            mask = index == i
            for d, (div_b, div_e) in enumerate(zip(div[:-1], div[1:])):
                index_n[mask*(div_b <= nano)*(div_e > nano)] = shift + d
        new_index.append(index_n)
    return new_index

def _colapse_index(index,nstream,colapse):
    """
    Check that data contains examples of all possible types of photons specified
    by the number of photon streams. Returns a set of indeces removing unused streams
    and a numpy array of the unused indeces. If colapse is false, raises an error
    if streams are missing, and if none are missing returns the unaltered indeces

    Parameters
    ----------
    index : list[numpy.ndarray]
        indexes of photons in bursts based on photon stream, with no divisors applied
    nstream : int
        Number of photon streams in the data
    colapse : Bool
        Whether or not to compress unused indeces, if false, raises error if 
        there are missing indeces

    Raises
    ------
    ValueError
        Indeces are missing from the set of indeces given to index based on nstream

    Returns
    -------
    index_n : list[numpy.ndarray]
        Colapsed indeces of phtons in bursts
    uni : numpy.ndarray
        The unique indexes from the indeces handed to the _colapse_index

    """
    if colapse:
        sep = np.cumsum(np.array([0] + [idx.size for idx in index]))
        uni, inv = np.unique(np.concatenate(index), return_inverse=True)
        index_n = [inv[sep_b:sep_e] for sep_b, sep_e in zip(sep[:-1], sep[1:])]
    else:
        uni = np.unique(np.concatenate(index))
        if uni.size != nstream: 
            raise ValueError("No photons from one or more streams defined by divisor scheme")
        index_n = index
    return index_n, uni

def _calc_trans_locs(path):
    """
    Determine the positions of transitions within a list of state path arrays

    Parameters
    ----------
    path : list[numpy.ndarray]
        The states of photons within burst determined by viterbi

    Returns
    -------
    trans_pos : list[numpy.ndarray]
        The index of each photon in a new state, each burst always begins with
        the 0th photon (1st state in the array, "assumed transition")

    """
    return [np.concatenate([[0], np.argwhere(np.diff(state)!=0)[:,0] + 1, [state.size]]) for state in path]

def _calc_burst_dwell_num(trans_locs):
    """
    Calculate the number of dwells in each burst

    Parameters
    ----------
    trans_locs : list[numpy.ndarray]
        Locations of transitions (changes in state) with bursts

    Returns
    -------
    burst_dwell_num: numpy.ndarray
        number of dwells in each burst

    """
    return np.array([tl.size-1 for tl in trans_locs])


def _calc_dwell_pos(tloc):
    """
    Make array identifying whether dwells in other concatenated arrays are from
    the middle (0), beginning (2), or end (1) of a burst.
    Beginning and end dwells are assigned 2 and 3 respectively, so that a mask 
    can be constructed where the start of bursts is identified by taking
    dwell_pos > 1

    Parameters
    ----------
    tloc : list[numpy.ndarray]
        The trans_locs lists, identifiying where transitions are in bursts
        including the 0 index and size of the burst at the beginning and end
        of a burst

    Returns
    -------
    dwell_pos : numpy.ndarray
        An array which identifies bursts based on their position within a burst:
            0: dwells in the middle of a burst
            
            1: dwells at the end of a burst
            
            2: dwells at the beginning of a burst
            
            3: dwells which span the entire burst (the burst and dwell are synonymous)

    """
    dwell_type = [None for _ in range(len(tloc))]
    for i, trans in enumerate(tloc):
        tsz = trans.shape[0] - 1
        if tsz == 1:
            d_type = np.array([3], dtype=np.uint8)
        else:
            d_type = np.zeros(tsz, dtype=np.uint8)
            d_type[0], d_type[tsz-1] = 2, 1
        dwell_type[i] = d_type
    return np.concatenate(dwell_type)


def _calc_dwell_dur(tloc, times):
    """
    Calculate the duration of dwells based on times and locations of transitions
    Durations in units of data acquisition (clk_p)

    Parameters
    ----------
    tloc : list[numpy.ndarray]
        The location of transitions in bursts, including first and last photons
    times : list[numpy.ndarray]
        Tha macrotimes of photons within bursts

    Returns
    -------
    dwell_dur : numpy.ndarray
        concatenated array of the duration of dwells

    """
    dwell_dur = list()
    for loc, time in zip(tloc, times):
        dur = np.empty(loc.size -1)
        tstart = time[0]
        for i, l in enumerate(loc[1:-1]):
            tstop = (time[l] + time[l-1]) // 2
            dur[i] = tstop - tstart
            tstart = tstop
        dur[-1] = time[-1] - tstart
        dwell_dur.append(dur)
    return np.concatenate(dwell_dur)


def _calc_ph_counts(tloc, index, ndet):
    """
    Identify the number of photons in each photon stream in dwells in bursts
    Note: returns concatenated array

    Parameters
    ----------
    tloc : list[numpy.ndarray]
        The index of each photon in a new state, each burst always begins with
        the 0th photon (1st state in the array, "assumed transition")
    index : list[numpy.ndarray]
        The indeces of photons in bursts
    ndet : int
        The number of photon streams in data

    Returns
    -------
    ph_counts : numpy.ndarray
        The counts of photons in dwells, columns are for counts in each index

    """
    ph_counts = list()
    for idx, tlc in zip(index, tloc):
        pcnt = np.array([np.bincount(idx[b:e].astype(np.uint32), minlength=ndet) 
                         for b, e in zip(tlc[:-1], tlc[1:])]).T
        ph_counts.append(pcnt)
    ph_counts = np.concatenate(ph_counts, axis=1)
    return ph_counts

def _calc_dwell_bg(num_dwells, dwell_dur, bg_rt, brst_pos, streams):
    """
    Calculate the background adjusted counts of photons in dwells per stream.
    Organized [stream, dwell].

    Parameters
    ----------
    num_dwells : numpy.ndarray
        Number of dwells in each burst.
    dwell_dur : numpy.ndarray
        dwell_dur of H2MM_result, duration in clk_p of each dwell.
    bg_rt : dict of list[numpy.ndarray]
        The bg of fretbursts.data used to create BurstData.
    brst_pos : list[numpy.ndarray]
        The bp of fretbursts.data used to create BurstData..
    streams : list[fretbursts.Ph_sel]
        The ph_streams of BurstData.

    Returns
    -------
    dwell_bg: numpy.ndarray
        Background correction per stream per dwell.

    """
    # make the bg per stream 
    bg_stream =((np.concatenate([bg[pos] for bg, pos in zip(bg_rt[stream], brst_pos)])) for stream in streams)
    # resize bg_s for dwells
    bg_stream = (np.concatenate([np.repeat(b, bdw) for b, bdw in zip(bg, num_dwells)]) for bg in bg_stream)
    # apply background correction to each stream, note: 1e-3 because dwell_dur in ms
    dwell_bg = np.array([bg_s * dwell_dur*1e-3 for bg_s in bg_stream])
    return dwell_bg


def _calc_ph_counts_bg(num_dwells, ph_counts, dwell_dur, bg_rt, brst_pos, streams):
    """
    Calculate the background adjusted counts of photons in dwells per stream.
    Organized [stream, dwell].

    Parameters
    ----------
    num_dwells : numpy.ndarray
        Number of dwells in each burst.
    ph_counts : numpy.ndarray
        dwell_ph_counts of H2MM_result, the number of photons in each stream in
        each dwell.
    dwell_dur : numpy.ndarray
        dwell_dur of H2MM_result, duration in clk_p of each dwell.
    bg_rt : dict of list[numpy.ndarray]
        The bg of fretbursts.data used to create BurstData.
    brst_pos : list[numpy.ndarray]
        The bp of fretbursts.data used to create BurstData..
    streams : list[fretbursts.Ph_sel]
        The ph_streams of BurstData.

    Returns
    -------
    ph_counts_bg: numpy.ndarray
        Background corrected number of photons per stream per dwell.

    """
    dwell_bg = _calc_dwell_bg(num_dwells, dwell_dur, bg_rt, brst_pos, streams)
    return ph_counts - dwell_bg
    

def _auto_irf_thresh(index, nanos, nstream):
    """
    Autmatlically determine the threshholds for IRF, using maximum of histogram.
    Not recomended for general use.

    Parameters
    ----------
    index : list[numpy.ndarray]
        Indexes of photons in bursts.
    nanos : list[numpy.ndarray]
        Nanotimes of photons in bursts.
    nstream : int
        Number of streams.

    Returns
    -------
    irf_thresh: numpy.ndarray
        Maximum of histogram of nanotime for each stream.

    """
    nano_concat = np.concatenate(nanos)
    index_concat = np.concatenate(index)
    thresh = [np.argmax(np.bincount(nano_concat[index_concat==i])) for i in range(nstream)]
    return np.array(thresh)

def make_divisors(data, ndiv, include_irf_thresh=False):
    """
    Automatically make a set of divisors evenly dividing the nanotime by ndiv divisors

    Parameters
    ----------
    data : BurstData
        Data object of bursts with defined streams
    ndiv : int or numpy array
        Either an int splitting all photon streams by ndiv divisors, or a numpy array
        where each element defines the number of divisors for each stream
    include_irf_thresh : bool, optional
        Whether or not to add the irf_threshold of the stream as an additional divisor.
        The default is False.

    Raises
    ------
    ValueError
        When incorrect number of streams specified for the BurstData object.

    Returns
    -------
    divs : list[numpy.ndarray]
        The new set of divisors.

    """
    if type(ndiv) == int:
        ndiv = np.array([ndiv for _ in range(data.nstream)])
    if len(ndiv) != data.nstream:
        raise ValueError(f"Divisors must be either in or define a number for each photon stream, there are {data.nstream} streams, but got {len(ndiv)} divisor sets")
    # sort nanotimes per stream
    nanos = [np.sort(np.concatenate([nano[index==i] for nano, index in zip(data.nanos, data.models.index)]))
                     for i in range(data.nstream)]
    # remove nanotimes less than the IRF threshold
    if include_irf_thresh:
        nanos = [nano[nano>irf] for nano, irf in zip(nanos, data.irf_thresh)]
    # define the faction of all nanotimes that should be used to make the divisor
    frac = [np.array([(n+1) * nano.size/(divs+1) for n in range(divs)]).astype(np.uint32) for nano, divs in zip(nanos, ndiv)]
    # make the divisors
    divs = [nano[div] for nano, div in zip(nanos, frac)]
    if include_irf_thresh:
        divs = [np.concatenate([[irf], div]) for irf, div in zip(data.irf_thresh, divs)]
    return divs
    
    


def calc_nanohist(model, conf_thresh=None):
    """
    Generate the fluorescence decay per state per stream from the model/data

    Parameters
    ----------
    model : H2MM_result
        A h2mm state-model and Viterbi analysis
    conf_thresh : float, optional
        The threshold of the :attr:`H2MM_result.scale` for a photon to be included
        in the nanotime histogram. If None, use :attr:`H2MM_result.conf_thresh`
        of model.
        The default is None

    Returns
    -------
    nanohist : numpy.ndarray
        A 3-D numpy array, containing fluoresence decays per state per stream
        array indexed as follows: [state, stream, nanotime_bin]

    """
    if conf_thresh is None:
        conf_thresh = model.conf_thresh
    mask = np.concatenate([scale for scale in model.scale]) >= conf_thresh
    nanos = np.concatenate(model.parent.parent.nanos)[mask]
    stream = np.concatenate(model.parent.parent.models.index)[mask]
    state = np.concatenate(model.path)[mask]
    bins = np.arange(0, model.parent.parent.data.nanotimes_params[0]['tcspc_num_bins'],1)
    nanohist = np.array([[np.histogram(nanos[(stream == i)*(state==j)], bins=bins)[0] 
                          for i in range(len(model.parent.parent.ph_streams))] 
                         for j in range(model.model.nstate)])
    return nanohist

def calc_dwell_nanomean(model, ph_streams, irf_thresh, conf_thresh=None):
    """
    Calcualate the mean nanotimes of dwells for a given (amalgamated) set of 
    photon streams, must define the irf_thresh to exlcude the IRF

    Parameters
    ----------
    model : H2MM_result
        A h2mm state-model and Viterbi analysis
    ph_streams : fretbursts.Ph_sel or list[fretbursts.Ph_sel]
        Either a Ph_sel object of the stream desired, or a list of Ph_sel objects
        defining all streams to take the aggregate mean over (generally not recomended)
    irf_thresh : int, or iterable of ints
        The threshold of the IRF, all photons with nanotime before this threshold
        excluded from analysis
    conf_thresh : float, optional
        The threshold of the :attr:`H2MM_result.scale` for a photon to be included
        in the nanotime histogram. If None, use :attr:`H2MM_result.conf_thresh`
        of model.
        The default is None

    Returns
    -------
    dwell_nanomean : numpy.ndarray
        Mean nanotime of each dwell for the given photon streams, and IRF threshhold

    """
    # check and reshape initial inputs
    if conf_thresh is None:
        conf_thresh = model.conf_thresh
    if type(ph_streams) == frb.Ph_sel:
        ph_streams = [ph_streams,]
    irf_thresh = np.atleast_1d(irf_thresh)
    idx = np.empty(len(ph_streams), dtype=int)
    # get indexes of photon streams
    for i, ph_stream in enumerate(ph_streams):
        idx[i] = np.argwhere([ph_stream == p_stream for p_stream in model.parent.parent.ph_streams])[0,0]
    # loop over bursts to calculate mean, i keeps track of dwell number
    i, dwell_nanomean = 0, np.zeros(model.dwell_state.size)
    for nano, trans_loc, index, scale in zip(model.parent.parent.nanos, model.trans_locs, model.parent.parent.models.index, model.scale):
        b_mask = np.zeros(nano.size, dtype=bool)
        adj_nano = nano.copy()
        scale_thresh = scale >= conf_thresh
        for ix, thresh in zip(idx, irf_thresh):
            s_mask = (index == ix)*(nano > thresh)*scale_thresh
            b_mask += s_mask
            adj_nano[s_mask] -= thresh
        for start, stop in zip(trans_loc[:-1], trans_loc[1:]):
            nanod = adj_nano[start:stop][b_mask[start:stop]]
            dwell_nanomean[i] = np.mean(nanod) if nanod.size > 0 else np.nan
            i += 1
    return dwell_nanomean * model.parent.parent.data.nanotimes_params[0]['tcspc_unit'] * 1e9
    

# Object of data for anlyasis by H2MM, of a set of divisor streams
class BurstData:
    """
    Class to organize a photon selection, functions to manage H2MM_list objects 
    with different divisor schemes of a data set
    
    """    
    def __init__(self, data, ph_streams=None, Aex_stream=None, Aex_shift=None,
                 irf_thresh=None, conserve_memory=False):
        if not hasattr(data, 'mburst'):
            raise ValueError("Bursts not selected yet")
        if Aex_shift and data.lifetime:
            raise NotImplementedError("Aex_shift only for usALEX")
        if not data.lifetime and irf_thresh is not None:
            raise NotImplementedError("IRF threshold only for pulsed excitation")
        if ph_streams is None:
            if data.alternated:
                ph_streams = (frb.Ph_sel(Dex="Dem"), 
                              frb.Ph_sel(Dex="Aem"), 
                              frb.Ph_sel(Aex="Aem"))
                if Aex_stream is None:
                    Aex_stream = frb.Ph_sel(Aex='Aem')
            else:
                ph_streams = (frb.Ph_sel(Dex="Dem"), frb.Ph_sel(Dex="Aem"))
        self.__data = data
        self.__ph_streams = ph_streams
        self.__Aex_stream = Aex_stream
        self.__Aex_shift = Aex_shift
        self.__nstream = len(ph_streams)
        self.conserve_memory = conserve_memory
        if data.lifetime:    
            index, times, nanos, particles = _sort_nsALEX(data, ph_streams)
            #: list of burst photon times
            self.times = times
            #: list of burst photon nanotimes
            self.nanos = nanos
            #: list of burst photon particle ids if particles exist in source data
            self.particles = particles
            #: The primary H2MM_list object, with no divisor scheme.
            self.models = H2MM_list(self, index)
            self._irf_thresh = _auto_irf_thresh(index, self.nanos, self.__nstream) if irf_thresh is None else irf_thresh
            self._irf_thresh_set = irf_thresh is not None
            #: dictiary of H2MM_list objects for each divisor
            self.div_models = dict()
        else:
            index, times, particles = _sort_usALEX(data, ph_streams, Aex_stream, Aex_shift)
            self.models = H2MM_list(self, index, times, particles=particles)
    
    def __getitem__(self, key):
        return self.div_models[key]
  
    @property
    def parent(self):
        """Returns self, so that [H2MM_list].parent.parent will return BurstData instead of error"""
        return self
    
    @property
    def irf_thresh(self):
        """The theshholds to exclude the IRF per stream"""
        return self._irf_thresh
    
    @irf_thresh.setter
    def irf_thresh(self, thresh):
        if type(thresh) == int:
            if frb.Ph_sel(Aex="Aem") in self.ph_streams:
                warnings.warn("Setting single threshold for donor and acceptor excitation will result in aberant mean nanotimes in the AexAem channel")
            irf_thresh = np.array([thresh for _ in range(self.__nstream)])
        elif type(thresh) in {list, tuple} and len(thresh) == len(self.ph_streams):
            irf_thresh = np.array(thresh)
        elif type(thresh) == np.ndarray and thresh.ndim == 1 and thresh.shape[0] == self.__nstream:
            irf_thresh = thresh
        else:
            raise ValueError(f"irf threshold must be {len(self.ph_streams)} length list, tuple or numpy array, got {type(thresh)}")
        self._irf_thresh, self._irf_thresh_set = irf_thresh, True
        # loop over all instances where the dwell nano mean has been calculated, and recalculate with new threshold
        for h2_list in chain([self.models], self.div_models.values()):
            for opt in h2_list.opts:
                if hasattr(opt, "_dwell_nano_mean"):
                    opt._dwell_nano_mean = opt._full_dwell_nano_mean()
    
    @property
    def data(self):
        """The Data object used to create the data"""
        return self.__data
    
    @property
    def ph_streams(self):
        """The photon streams used in analysis"""
        return self.__ph_streams
    
    @property
    def Aex_stream(self):
        """Which stream(s) used as acceptor excitation"""
        return self.__Aex_stream
    
    @property
    def Aex_shift(self):
        """Boolean for whether acceptor excitation photons shifted into donor excitation window"""
        return self.__Aex_shift
    
    @property
    def nstream(self):
        """Number of photon streams in the bursts"""
        return self.__nstream
    
    def new_div(self, divs, name=None, colapse=False, fix_divs=True):
        """
        Set up a new set of divisors **pusled laser excitation only**

        Parameters
        ----------
        divs : list[numpy.ndarray]
            List of divisors, use 1 array per photon stream. Divisors must be
            in ascending order and within the excitation window of their given
            photon stream.
        name : str, optional
            The name of the key identifying the divisor scheme. If none specified,
            a name will be assigned automatically, as div{n} where n is an integer.
            The default is None.
        colapse : bool, optional
            Whether to automatically remove unused indeces, if false, raises an
            error if there are missing indeces.
            
            .. note:: 
                
                if fix_divs is False, this check will not be performed.
            
            The default is False.
        fix_divs : bool, optional
            Whether to check for unused indeces. 
            The default is True.

        Returns
        -------
        name : str
            The key of the new divisor scheme

        """
        # check if all inputs are compatible/correct
        if not self.__data.lifetime:
            raise NotImplementedError("Cannot use divisors with continuous wave excitation")
        if len(divs) != self.__nstream:
            raise ValueError("Incorrect number of streams for BurstData streams")
        # generate new divisors
        if name is None:
            name = f"div{len(self.div_models)}"
        nstream = np.sum([d.size + 1 for d in divs])
        c_divs = _complete_div(self.__data, divs, self.__ph_streams)
        index = _divisor_sort(self.models.index, self.nanos, c_divs)
        if fix_divs:
            index, uni = _colapse_index(index, nstream, colapse)
            if colapse:
                idx_map = -1*np.ones(uni.max()+1, dtype=np.int32)
                for i, n in enumerate(uni):
                    idx_map[n] = i
                if np.any(idx_map == -1):
                    drop_div = idx_map != -1
                    pos = 0
                    ndiv = list()
                    for div in c_divs:
                        n_div = drop_div[pos:pos+div.size-1]
                        n_div = div[:-1][n_div]
                        ndiv.append(n_div[1:])
                        pos += div.size - 1
                c_divs = _complete_div(self.__data, ndiv, self.__ph_streams)
        self.div_models[name] = H2MM_list(self, index, divisor_scheme=c_divs)
        return name
    
    
    def auto_div(self, ndivs, name=None, include_irf_thresh=False):
        """
        Create a new divisor scheme with the given number of divisors per stream

        Parameters
        ----------
        ndivs : int or array like
            The number of divisors in each stream if an int, or per stream if list
            of numpy arrays.
            If an int :
                
                Each stream will be divided by that many divisors, such that
                there is an equal probability of a photon arriving in each divisor
                for the entire data set.
                
            If a array like:
                
                The number of divisors per stream
                
        name : str, optional
            The name of the key identifying the divisor scheme. If none specified,
            a name will be assigned automatically, as div{n} where n is an integer.
            The default is None.
        include_irf_thresh : bool, optional
            Whether or not to include the IRF threshhold as a separate divisor.
            The default is False.

        Returns
        -------
        name: str
            The key of the new divisor scheme.

        """
        divs = make_divisors(self, ndivs, include_irf_thresh=include_irf_thresh)
        return self.new_div(divs, name=name)


def _conv_crit(model_list, attr, thresh):
    """
    Whether a model has converged

    Parameters
    ----------
    model_list : H2MM_list
        Object organizing a given divisor scheeme
    attr : str
        Name of attribute used to identify the ideal model
    thresh : float
        Theshhold to consider a model converged

    Returns
    -------
    conv : int
        Index of ideal model.

    """
    discrim = getattr(model_list, attr)
    if thresh is None:
        conv = np.argmin(discrim) if model_list.num_opt > 1 else 1
    else:
        conv = np.argwhere(conv < 0.05)[0,0] if  model_list.num_opt > 1 else 1
    return conv

def _calc_burst_state_counts(path, trans_locs, nstate):
    """
    Find the nubmer of times a given state occurs in each burst

    Parameters
    ----------
    path : list[numpy.ndarray]
        The paths list from Viterbi.
    nstate : int
        Number of states in model.

    Returns
    -------
    state_counts : np.ndarray
        Array of dwells in each state by burst, organized [state, burst]

    """
    state_counts = np.array([np.bincount(p[tl[:-1]].astype(np.int64), minlength=nstate) 
                             for p, tl in zip(path, trans_locs)]).T
    return state_counts

def _calc_burst_code(state_counts):
    """
    Make bitmasks per burst of which states present

    Parameters
    ----------
    dwell_counts : list[numpy.ndarray]
        state_counts array.

    Returns
    -------
    burst_code : numpy.ndarray
        Array of ints with binary representation of which states pressent.

    """
    burst_code = np.array([sum([2**i for i, b in enumerate(burst) if b != 0]) for burst in state_counts.T])
    return burst_code

def find_ideal(model_list, conv_crit, thresh=None, auto_set=False):
    """
    Identify the ideal state model

    Parameters
    ----------
    model_list : H2MM_list
        The set of optimizations for which the ideal models is to be determined
    conv_crit : str
        Which convergence criterion to use.
    thresh : float, optional
        The threshold diffence between statistical discriminators to determine
        if the convergence criterion have been met. 
        The default is None.

    Returns
    -------
    ideal : int
        Index of ideal state model. Note: i+1 is the number of states
        (because python indexes from 0)

    """
    if len(model_list.opts) < 1:
        warnings.warn("Only one model calculated, no sources of comparison")
    ideal = _conv_crit(model_list, conv_crit, thresh)
    if ideal == 1 and len(model_list.opts):
        ideal = 0
    return ideal


# class for a given set of divisors, aggregates different state models
class H2MM_list:
    """
    Class for organizing optimized models under a divisor scheme.
    
    .. note::
        
        These objects are rarely created by the user, rather, they are created
        by :class:`BurstData` upon initiation (stored in the :attr:`BurstData.models`
        attribute) or by :meth:`BurstData.new_div` or :meth:`BurstData.auto_div`
        for making the new divisor schemes.
    
    
    """
    #: dictionary of statistical discriminators as keys and labels for matplotlib axis as values
    stat_disc_labels = {'ICL':"ICL", 'BIC':"BIC", 'BICp':"BIC'"}
    def __init__(self, parent, index, divisor_scheme=None, conserve_memory=False):
        if divisor_scheme is None:
            divisor_scheme = _make_divs(parent.data, parent.ph_streams)
        self.parent= parent
        #: list of burst photon indeces, specific to divisor schem
        self.index = index
        #: list storing H2MM_result objects, len is always maximum number of states, filled with None for state models not calculated
        self.opts = list()
        self.__divisors = divisor_scheme
        self.__div_map = np.cumsum([0]+[d.size - 1 for d in divisor_scheme])
        self.__ndet = self.__div_map[-1]
        #: whether or not to trim data of non-ideal H2MM_result
        self.conserve_memory = conserve_memory
    
    def __getitem__(self, key):
        if len(self.opts) == 0:
            raise ValueError("No optimizations run, must run calc_models or optimize for H2MM_result objects to exist")
        return self.opts[key]
    
    def __iter__(self):
        for opt in self.opts:
            if opt is not None:
                yield opt
    
    @property
    def num_opt(self):
        """Number of optimizations conducted"""
        return np.array([opt is not None for opt in self.opts]).sum()

    @property
    def BIC(self):
        """The BIC of the models, returns inf for non-existent models"""
        bic = np.empty(len(self.opts))
        for i, m in enumerate(self.opts):
            if m is not None:
                bic[i] = m.bic
            else:
                bic[i] = np.inf
        return bic
    
    @property
    def BICp(self):
        """The BIC' of the models, returns inf for non-existent models"""
        BIC = np.empty(len(self.opts))
        kp = np.empty(len(self.opts), dtype=int)
        for i, m in enumerate(self.opts):
            if m is not None:
                BIC[i] = m.bic
                kp[i] = m.nphot - m.k
            else:
                BIC[i] = np.inf
                kp[i] = 1
        BIC = BIC - BIC.min()
        return BIC / kp
    
    @property
    def ICL(self):
        """The ICL of the models, returns inf for non-existent models"""
        ICL = np.empty(len(self.opts))
        for i, m in enumerate(self.opts):
            if m is not None:
                ICL[i] = m.icl
            else:
                ICL[i] = np.inf
        return ICL
    
    @property
    def ndet(self):
        """Number of photon streams in the divisor scheme"""
        return self.__ndet
    
    @property
    def divisor_scheme(self):
        """The divisor scheme used to make the current H2MM_list"""
        return [div[1:-1] for div in self.__divisors]
    
    @property
    def div_map(self):
        """The indicies in the model where indices change stream"""
        return self.__div_map
    
    @property
    def dwell_params(self):
        """List of attributes that define dwells"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell_params")
        return self.opts[self.ideal].dwell_params
    
    @property
    def E(self):
        """The FRET efficiency of each state in the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return E")
        return self.opts[self.ideal].E
    
    @property
    def E_corr(self):
        """The gamma corrected FRET efficiency of each state in the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return E_corr")
        return self.opts[self.ideal].E_corr

    @property
    def S(self):
        """The stoichiometry of each state in the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return S")
        return self.opts[self.ideal].S
    
    @property
    def S_corr(self):
        """The gamma/beta corrected stoichiometry of each state in the ideal model"""
        if not hasattr(self, "ideal"):
            raise  ValueError("Ideal model not set, cannot return S_corr")
        return self.opts[self.ideal].S_corr
    
    @property
    def trans(self):
        """The Transition rate matrix of the ideal model, rates in s^{-1}"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return trans")
        return self.opts[self.ideal].trans

    @property
    def trans_locs(self):
        """List of numpy arrays, identifying the beginning of each new dwell in a burst, for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return trans_locs")
        return self.opts[self.ideal].trans_locs
    
    @property
    def burst_dwell_num(self):
        """Number of dwells in each burst for the ideal model"""
        if not hasattr(self, "ideal"):
            ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].burst_dwell_num
    
    @property
    def dwell_state(self):
        """State of each dewll for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_state
    
    @property
    def dwell_dur(self):
        """The duration of each dwell, in seconds, for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_dur
    
    @property
    def dwell_pos(self):
        """
        The position within a burst of each dwell for the ideal model:
            0: middle dwells
            1: ending dwells
            2: beginning dwells
            3: whole burst dwells
        """
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_pos
    
    @property
    def dwell_ph_counts(self):
        """The number of photons in a dwell, per stream, organized [stream, dwell], for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_ph_counts
    
    @property
    def dwell_ph_counts_bg(self):
        """The background corrected number of photons in a dwell, per stream, organized [stream, dwell], for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_ph_counts_bg
    
    @property
    def dwell_E(self):
        """The raw FRET efficiency of each dwell"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params, for the ideal model")
        return self.opts[self.ideal].dwell_E
    
    @property
    def dwell_E_corr(self):
        """The FRET efficiency of each dwell, with corrections for background, leakage, direct excitation, and gamma applied, for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_E_corr
    
    @property
    def dwell_S(self):
        """The raw stoichiometry of each dwell, for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_S
    
    @property
    def dwell_S_corr(self):
        """The stoichiometry of each dwell, with corrections for background, leakage, direct excitation, and gamma and beta applied, for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_S_corr
    
    @property
    def nanohist(self):
        """Histogram of nanotimes, sorted by state, organized [state, stream, nanotime], for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].nanohist
    
    @property
    def dwell_nano_mean(self):
        """Mean nanotime of each stream and dwell, organized [stream, dwell], for the ideal model"""
        if not hasattr(self, "ideal"):
            raise ValueError("Ideal model not set, cannot return dwell params")
        return self.opts[self.ideal].dwell_nano_mean
    
    def optimize(self, model, replace=False, **kwargs):
        """
        Optimize a model against data, and add optimized data to .opts list

        Parameters
        ----------
        model : H2MM_C.h2mm_model
            Initial model to be optimized.
        replace : bool, optional
            If an identical state model has been optimized, whether to replace 
            with newly optimized model. If false, raises error indicating model
            already exists.
            The default is False.
        **kwargs : dict
            Keyword arguments passed to optimization function, controls num_cores
            etc.

        Raises
        ------
        Exception
            Optmized model of given number of states already exists.

        Returns
        -------
        None.

        """
        if self.ndet != model.ndet:
            raise ValueError("Model streams inconsistent with data")
        nstate = model.nstate
        stid = nstate - 1
        while len(self.opts) < nstate:
            self.opts.append(None)
        if not replace and self.opts[stid] is not None:
            raise Exception(f"Already Optimized model for {nstate} states")
        model.optimize(self.index, self.parent.times, **kwargs)
        vkwargs = {"num_cores":kwargs["num_cores"]} if "num_cores" in kwargs else {}
        self.opts[stid] = H2MM_result(self, model, **vkwargs)
    
    def calc_models(self, min_state=1, to_state=4, max_state=8, models=None, 
                    conv_crit="ICL", thresh=None,**kwargs):
        """
        Optimize h2mm models against data for increasing numbers of states
        until either maximum number reached, or convergence criterion met.

        Parameters
        ----------
        min_state : int, optional
            Number of states in model that is first optimized. 
            The default is 1.
        to_state : int, optional
            The minimum number of states in the model with the most states in the
            models optimized. Ensures even if convergence criterion already met
            that this state model will exist. 
            The default is 4.
        max_state : int, optional
            Maximum number of states in model that will be optmized, whether or not
            the convergence criterion have been met- sets upper bound on duration
            of while loop. 
            The default is 8.
        models : list[H2MM_C.h2mm_model], optional
            List of initial models to use. If models for given number of states
            do not exist in this list, the factory_h2mm_function will be used.
            The default is None.
        conv_crit : str, optional
            Which of the build-in convergence criterion to use to evaluate if 
            the ideal state model has been found. 
            The default is 'ICL'.
        thresh : float, optional
            The threshold diffence between statistical discriminators to determine
            if the convergence criterion have been met. The default is None.
        **kwargs : dict
            Keyword arguments passed to optimization function, controls num_cores
            etc.

        Returns
        -------
        ideal : int
            The index of the ideal model based on the threshold

        """
        if models is None:
            models = []
        else:
            if np.any([m.ndet != self.ndet for m in models]):
                raise ValueError("Model stream inconsistent with data")
            if np.unique([m.nstate for m in models]).size != len(models):
                raise ValueError("Multiple models with same number of states")
        i = min_state
        model_states = [m.nstate for m in models]
        self.ideal_crit = conv_crit
        while i <= max_state and (i <= to_state or _conv_crit(self, conv_crit, thresh) == i-2):
            if i in model_states:
                model = [m for m in models if m.nstate==i][0]
                new_kwargs = {'replace':True}
                new_kwargs.update(kwargs)
                self.optimize(model, **new_kwargs)
            elif len(self.opts) < i or self.opts[i-1] is None:
                model = h2.factory_h2mm_model(i, self.ndet, bounds=kwargs.get('bounds', None))
                self.optimize(model, **kwargs)
            i += 1
        ideal = _conv_crit(self, conv_crit, thresh)
        return ideal
    
    def find_ideal(self, conv_crit, thresh=None, auto_set=False):
        """
        Identify the ideal state model

        Parameters
        ----------
        conv_crit : str
            Which convergence criterion to use.
        thresh : float, optional
            The threshold diffence between statistical discriminators to determine
            if the convergence criterion have been met. 
            The default is None.
        auto_set : bool, optional
            Whether to set the ideal 

        Returns
        -------
        ideal : int
            Index of ideal state model. Note: i+1 is the number of states
            (because python indexes from 0)

        """
        ideal = find_ideal(self, conv_crit, thresh=thresh)
        if auto_set:
            self.ideal = ideal
        return ideal
    
    def free_data(self, save_ideal=True):
        """
        Clear large data arrays from models

        Parameters
        ----------
        save_ideal : bool, optional
            Whether or not to keep the large data arrays for the ideal state model.
            The default is True.

        Returns
        -------
        None.

        """
        for i, opt in enumerate(self.opts):
            if opt is not None and (not save_ideal or i != self.ideal):
                opt.trim_data()


# class to store individual h2mm models and their associated parameters
class H2MM_result:
    """ 
    Class to represent the results of analysis with H2MM and Viterbi
    
    .. note::
        
        This class is rarely created direclty be the user, rather usually created
        by :class:`H2MM_list` running :meth:`H2MM_list.optimize` or 
        :meth:`H2MM_list.calc_models` and stored in the :attr:`H2MM_list.opts` 
        list attribute.
    
    
    """
    #: tuple of parameters that include dwell or photon information- cleared by trim data
    large_params = ("path", "scale", "_trans_locs", "_burst_dwell_num", "_dwell_pos", 
                    "_dwell_state", "_dwell_dur", "_dwell_ph_counts", "_dwell_ph_counts_bg",
                    "_dwell_E", "_dwell_S", "_dwell_E_corr", "_dwell_S_corr",
                    "_dwell_nano_mean", "_dwell_ph_counts_bg", "_nanohist", "_burst_state_counts")
    #: dictionary of dwell parameters as keys, and values the type of plot to use in scatter/histogram plotting
    dwell_params = {"dwell_pos":"bar" , "dwell_state":"bar", "dwell_dur":"ratio",
                    "dwell_E":"ratio", "dwell_S":"ratio", 
                    "dwell_E_corr":"ratio", "dwell_S_corr":"ratio", 
                    "dwell_nano_mean":"stream", 
                    "dwell_ph_counts":"stream", "dwell_ph_counts_bg":"stream"}
    #: dictionary of parameters as keys and axis labels as values
    param_labels = {"dwell_pos":"dwell position", "dwell_state":"state", "dwell_dur":"ms",
                   "dwell_E":"E$^{raw}$", "dwell_S":"S$^{raw}$",
                   "dwell_E_corr":"E", "dwell_S_corr":"S", 
                   "dwell_nano_mean":"ns", 
                   "dwell_ph_counts":"counts", "dwell_ph_counts_bg":"counts"}
    
    def __init__(self, parent, model, **kwargs):
        path, scale, ll, icl = h2.viterbi_path(model, parent.index, parent.parent.times, 
                                                                   **kwargs)
        #: most likely state of each photon in each burst
        self.path = path
        #: posterior probability of each photon in each burst
        self.scale = scale
        # loglikelihood of each photon in each burst
        self.ll = ll
        #: Integrated Complete Likelihood (ICL) for the model
        self.icl = icl
        self.parent = parent
        #: The optimized H2MM_C.h2mm_model representing the data
        self.model = model
        if self.parent.parent.nanos is not None:
            self._conf_thresh = 0.0
    
    def trim_data(self):
        """Remove photon-level arrays, conserves memory for less important models"""
        for attr in H2MM_result.large_params:
            if hasattr(self, attr):
                delattr(self, attr)
                
    
    @property
    def nstate(self):
        """Number of states in model"""
        return self.model.nstate
            
    @property
    def bic(self):
        """The Bayes Information Criterion of the model"""
        return self.model.bic
    
    @property
    def nphot(self):
        """Total number of photons in the burst data"""
        return self.model.nphot
    
    @property
    def ndet(self):
        """Number of photon streams in divisor scheme"""
        return self.model.ndet
    
    @property
    def k(self):
        """Number of free parameters in H2MM model"""
        return self.model.k
    
    @property
    def _DexDem(self):
        """The probability of DexDem from emmision probability matrix (sum of all DexDem streams)"""
        Ds = np.argwhere([ph_stream == frb.Ph_sel(Dex='Dem') for ph_stream in self.parent.parent.ph_streams])[0,0]
        Dr = slice(self.parent.div_map[Ds],self.parent.div_map[Ds+1])
        return self.model.obs[:,Dr].sum(axis=1)
    
    @property 
    def _DexAem(self):
        """The probability of DexAem from emmision probability matrix (sum of all DexAem streams)"""
        As = np.argwhere([ph_stream == frb.Ph_sel(Dex='Aem') for ph_stream in self.parent.parent.ph_streams])[0,0]
        Ar = slice(self.parent.div_map[As],self.parent.div_map[As+1])
        return self.model.obs[:,Ar].sum(axis=1)
    @property 
    def _AexAem(self):
        """The probability of AexAem from emmision probability matrix (sum of all AexAem streams)"""
        As = np.argwhere([ph_stream == frb.Ph_sel(Aex='Aem') for ph_stream in self.parent.parent.ph_streams])[0,0]
        Ar = slice(self.parent.div_map[As],self.parent.div_map[As+1])
        return self.model.obs[:,Ar].sum(axis=1)
    
    @property
    def E(self):
        """The FRET efficiency of each state in the model"""
        return self._DexAem / (self._DexDem + self._DexAem)
    
    @property 
    def E_corr(self):
        """The gamma corrected FRET efficiency of each state in the model"""
        F_fret = self._DexAem - (self.parent.parent.data.leakage * self._DexDem)
        if frb.Ph_sel(Aex="Aem") in self.parent.parent.ph_streams:
            F_fret -= self.parent.parent.data.dir_ex * self._AexAem
        return F_fret / (F_fret + (self.parent.parent.data.gamma * self._DexDem))
    
    @property
    def S(self):
        """The stoichiometry of each state in the model"""
        return (self._DexDem + self._DexAem) / (self._DexDem + self._DexAem + self._AexAem)
    
    @property 
    def S_corr(self):
        """The gamma/beta corrected stoichiometry of each state in the model"""
        F_fret = self._DexAem - (self.parent.parent.data.leakage * self._DexDem)- self.parent.parent.data.dir_ex * self._AexAem
        F_dex = F_fret + (self.parent.parent.data.gamma * self._DexDem)
        return F_dex / ((self._AexAem / self.parent.parent.data.beta) + F_dex)
    
    @property
    def trans(self):
        """The Transition rate matrix, rates in s^{-1}"""
        return self.model.trans / self.parent.parent.data.clk_p
    
    @property
    def trans_locs(self):
        """List of numpy arrays, identifying the beginning of each new dwell in a burst"""
        if not hasattr(self, "path"):
            self.path, self.scale, _, _ = h2.viterbi_path(self.model, self.parent.index, self.parent.parent.times)
        if not hasattr(self, "_trans_locs"):
            self._trans_locs = _calc_trans_locs(self.path)
        return self._trans_locs
    
    @property
    def burst_dwell_num(self):
        """Number of dwells in each burst"""
        if not hasattr(self, "_burst_dwell_num"):
            self._burst_dwell_num = _calc_burst_dwell_num(self.trans_locs)
        return self._burst_dwell_num
    
    @property
    def burst_state_counts(self):
        """Counts of number of dwells in each state per burst"""
        if not hasattr(self, "_burst_state_counts"):
            self._burst_state_counts = _calc_burst_state_counts(self.path, self.trans_locs,self.model.nstate)
        return self._burst_state_counts
    
    @property
    def burst_type(self):
        """Binary code indicating which states present in a burst. Functions as binary mask"""
        if not hasattr(self, "_burst_type"):
            if not hasattr(self, "_burst_state_counts"):
                burst_state_counts = _calc_burst_state_counts(self.path, self.trans_locs, self.model.nstate)
            else:
                burst_state_counts = self._burst_state_counts
            self._burst_type = _calc_burst_code(burst_state_counts)
        return self._burst_type
    
    @property
    def dwell_pos(self):
        """
        The position within a burst of each dwell:
            0: middle dwells
            1: ending dwells
            2: beginning dwells
            3: whole burst dwells
        """
        if not hasattr(self, "_dwell_pos"):
            self._dwell_pos = _calc_dwell_pos(self.trans_locs)
        return self._dwell_pos
    
    @property
    def dwell_state(self):
        """The state of each dwell"""
        if not hasattr(self, "_dwell_state"):
            self._dwell_state = np.concatenate([pt[sp[:-1]] for pt, sp in zip(self.path, self.trans_locs)])
        return self._dwell_state
    
    @property
    def dwell_dur(self):
        """The duration of each dwell, in miliseconds"""
        if not hasattr(self, '_dwell_dur'):
            self._dwell_dur = _calc_dwell_dur(self.trans_locs, self.parent.parent.times) * self.parent.parent.data.clk_p * 1e3
        return self._dwell_dur
    
    @property
    def dwell_ph_counts(self):
        """The number of photons in a dwell, per stream, organized [stream, dwell]"""
        if not hasattr(self, '_dwell_ph_counts'):
            self._dwell_ph_counts = \
                _calc_ph_counts(self.trans_locs, self.parent.parent.models.index, self.parent.parent.nstream)
        return self._dwell_ph_counts
    
    @property
    def dwell_ph_counts_bg(self):
        """The background corrected number of photons in a dwell, per stream, organized [stream, dwell]"""
        if not hasattr(self, "_dwell_ph_counts_bg"):
            self._dwell_ph_counts_bg = \
                _calc_ph_counts_bg(self.burst_dwell_num, self.dwell_ph_counts,self.dwell_dur, 
                                   self.parent.parent.data.bg, self.parent.parent.data.bp, 
                                   self.parent.parent.ph_streams)
        return self._dwell_ph_counts_bg
   
    @property 
    def dwell_E(self):
        """The raw FRET efficiency of each dwell"""
        if not hasattr(self, '_dwell_E'):
            Dr = np.argwhere([ph_stream == frb.Ph_sel(Dex='Dem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            Ar = np.argwhere([ph_stream == frb.Ph_sel(Dex='Aem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            D = self.dwell_ph_counts[Dr,:]
            A = self.dwell_ph_counts[Ar,:]
            self._dwell_E = np.divide(A, D + A, out=np.array([np.nan for _ in range(A.size)]), where=(D+A)!=0)
        return self._dwell_E
    
    @property
    def dwell_E_corr(self):
        """The FRET efficiency of each dwell, with corrections for background, leakage, direct excitation, and gamma applied"""
        if not hasattr(self, '_dwell_E_corr'):
            Dr = np.argwhere([ph_stream == frb.Ph_sel(Dex='Dem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            Ar = np.argwhere([ph_stream == frb.Ph_sel(Dex='Aem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            D = self.dwell_ph_counts_bg[Dr,:]
            A = self.dwell_ph_counts_bg[Ar,:]
            if frb.Ph_sel(Aex="Aem") in self.parent.parent.ph_streams:
                Cr = np.argwhere([ph_stream == frb.Ph_sel(Aex='Aem') 
                                  for ph_stream in self.parent.parent.ph_streams])[0,0]
                C = self.dwell_ph_counts_bg[Cr,:]
                F_fret = A - (self.parent.parent.data.leakage * D) - (C * self.parent.parent.data.dir_ex)
            else:
                F_fret = A - (self.parent.parent.data.leakage * D)
            F_tot = F_fret + (self.parent.parent.data.gamma * D)
            self._dwell_E_corr = np.divide(F_fret, F_tot, out=np.array([np.nan for _ in range(A.size)]), where= F_tot != 0.0)
        return self._dwell_E_corr
    
    @property
    def dwell_S(self):
        """The raw stoichiometry of each dwell"""
        if frb.Ph_sel(Aex="Aem") not in self.parent.parent.ph_streams:
            raise AttributeError("Parent BurstData must include AexAem stream ")
        if not hasattr(self, '_dwell_S'):
            Dr = np.argwhere([ph_stream == frb.Ph_sel(Dex='Dem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            Ar = np.argwhere([ph_stream == frb.Ph_sel(Dex='Aem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            Cr = np.argwhere([ph_stream == frb.Ph_sel(Aex='Aem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            D = self.dwell_ph_counts[Dr,:] + self.dwell_ph_counts[Ar,:]
            C = self.dwell_ph_counts[Cr,:]
            self._dwell_S = np.divide(D, D + C, out=np.array([np.nan for _ in range(D.size)]), where=(D+C)!=0)
        return self._dwell_S
    
    @property
    def dwell_S_corr(self):
        """The stoichiometry of each dwell, with corrections for background, leakage, direct excitation, and gamma and beta applied"""
        if frb.Ph_sel(Aex="Aem") not in self.parent.parent.ph_streams:
            raise AttributeError("Parent BurstData must include AexAem stream ")
        """The stoichiometry of each dwell"""
        if not hasattr(self, '_dwell_S_corr'):
            data = self.parent.parent.data
            Dr = np.argwhere([ph_stream == frb.Ph_sel(Dex='Dem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            Ar = np.argwhere([ph_stream == frb.Ph_sel(Dex='Aem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            Cr = np.argwhere([ph_stream == frb.Ph_sel(Aex='Aem') 
                              for ph_stream in self.parent.parent.ph_streams])[0,0]
            print(Dr, Ar, Cr)
            D = self.dwell_ph_counts_bg[Dr,:]
            C = self.dwell_ph_counts_bg[Cr,:]
            A = self.dwell_ph_counts_bg[Ar,:]
            F_fret = A - (self.parent.parent.data.leakage * D) - (C * self.parent.parent.data.dir_ex)
            F_tot = F_fret + (data.gamma * D)
            S_tot = F_tot + (C/data.beta)
            S_corr = np.divide(F_tot, S_tot, 
                               out=np.array([np.nan for _ in range(C.size)]), 
                                             where= S_tot != 0)
            self._dwell_S_corr = S_corr
        return self._dwell_S_corr

    @property
    def conf_thresh(self):
        """Threshhold for considering a photon nantotime in calculating dwell_nano_mean and nanohist"""
        if not hasattr(self, "_conf_thresh"):
            raise AttributeError("Parent fretbursts.Data object must include nanotimes")
        return self._conf_thresh
    
    @conf_thresh.setter
    def conf_thresh(self, conf_thresh):
        if not isinstance(conf_thresh, (float, int)):
            raise TypeError("Input must be single number")
        elif conf_thresh < 0 or conf_thresh >= 1:
            raise ValueError("conf_thresh must be within [0, 1)")
        self._conf_thresh = float(conf_thresh)
        # recalculate parameters with new threshhold
        if hasattr(self, "_nanohist"):
            self._nanohist = calc_nanohist(self)
        if hasattr(self, "_dwell_nano_mean"):
            self._dwell_nano_mean = self._full_dwell_nano_mean()
    
    @property
    def nanohist(self):
        """Histogram of nanotimes, sorted by state, organized [state, stream, nanotime]"""
        if not hasattr(self, '_nanohist'):
            self._nanohist = calc_nanohist(self)
        return self._nanohist
    
    @property
    def dwell_nano_mean(self):
        """Mean nanotime of each stream and dwell, organized [stream, dwell]"""
        if not hasattr(self, "_dwell_nano_mean"):
            self._dwell_nano_mean = self._full_dwell_nano_mean()
        if not self.parent.parent._irf_thresh_set:
            warnings.warn("IRF threshold was set automatically, recommend manually setting threshold with irf_thresh")
        return self._dwell_nano_mean
    
    def _full_dwell_nano_mean(self):
        """
        (Re)Calcualte the mean nanotime of each stream and dwell

        Returns
        -------
        nanomean: 2D numpy.ndarray
            The mean nanotime of each stream and dwell, organized [stream, dwell]

        """
        return np.array([calc_dwell_nanomean(self, ph_sel, irf_thresh) 
                         for ph_sel, irf_thresh in zip(self.parent.parent.ph_streams, self.parent.parent.irf_thresh)])
    
    def calc_dwell_nanomean(self, ph_streams, irf_thresh):
        """
        Generate the mean nanotimes of dwells, for given stream and IRF threshhold

        Parameters
        ----------
        ph_streams : fretbursts.Ph_sel, or list[fretbursts.Ph_sel]
            The photon stream(s) for which the mean dwell nanotime is to be calculated
        irf_thresh : int or iterable of int
            The threshold at which to include photons for caclulation of mean nanotime
            Should be set to be at the end of the IRF

        Returns
        -------
        dwell_nanomean : numpy.ndarray
            Mean nanotime of the given photon stream(s) in aggregate
            NOTE: different photon streams are not separated in calculation,
            must be called separetly per stream to calculate each mean nanotime

        """
        return calc_dwell_nanomean(self, ph_streams, irf_thresh)
    