#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Module: Masking
# Author: Paul David Harris
# Created; 21 Jun 2022
# Modified: 21 July 2022
# Purpose: selecting particular sets of bursts
"""
Selections
==========

Functions for selecting types of bursts and photons
"""

import numpy as np

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

def dwell_size(model, ph_min, ph_max=np.inf, streams=None):
    """
    Return a mask of dwells with a minimum number of photons, can also specify
    a maximum value through ph_max, and isolate from a set stream with stream.
    
    Uses the uncorected photon counts.
    

    Parameters
    ----------
    model : H2MM_result
        Model to make mask.
    ph_min : int
        Minimum nubmer of photons in a dwell.
    ph_max : int, optional
        Maximum number of photons in a dwell. 
        
        .. note::
       
            Default is np.inf to establish no maximum value. float values are
            allowed, but discouraged.
        
        The default is np.inf.
    stream : fretbursts.Ph_sel, or list[fretbursts.Ph_sel] optional
        The desired photon stream(s), if None, takes sum in all photon streams.
        The default is None.

    Raises
    ------
    TypeError
        Stream is not frb.Ph_sel.

    Returns
    -------
    mask : numpy.ndarray dtype=bool
        Mask of dwells fitting the input criterion.

    """
    ph_counts = model.dwell_ph_counts
    if streams is None:
        ph_counts = ph_counts.sum(axis=0)
    elif streams in model.parent.parent.ph_streams:
        i = np.argwhere([streams == ph for ph in model.parent.parent.ph_streams])[0,0]
        ph_counts = ph_counts[i,:]
    elif np.all([s in model.parent.parent.ph_streams for s in streams]):
        i = np.argwhere([streams == ph for ph in model.parent.parent.ph_streams]).reshape(-1)
        ph_counts = ph_counts[i,:].sum(axis=0)
    else:
        raise TypeError(f"strem must be a fretbursts.Ph_sel, or array_like[fretbursts.Ph_sel], got {type(streams)}")
    mask = (ph_counts >= ph_min) * (ph_counts <= ph_max)
    return mask
        