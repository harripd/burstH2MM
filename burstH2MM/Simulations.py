#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Module: Simulation
# Author: Paul David Harris
# Created: 25 Sept 2022
# Modified 02 Oct 2022
# Purpose: Module for creating simulated data
"""
Simulations
===========

Module for generating simulated data sets, allowing comparison of simulated data
to real data.

"""

import numpy as np
import warnings
import H2MM_C as h2
from .BurstSort import H2MM_result, H2MM_list, BurstData


class _model_dummy():
    pass
    
class _Sim_Data(BurstData):
    def __init__(self, base, times, index):
        for name, val in base.__dict__.items():
            if name not in ('times', 'models'):
                setattr(self, name, val)
        self.times = times
        self.models = _model_dummy()
        self.models.index = index

class _Sim_List(H2MM_list):
    def __init__(self, base, index, times):
        for name, val in base.__dict__.items():
            if name not in  ('index', 'parent'):
                setattr(self, name, val)
        self.index = index
        self.__div_map = base.div_map
        index_reduce = list()
        for idx in index:
            nidx = np.zeros(idx.shape, dtype=np.uint32)
            for i, (b, e) in enumerate(zip(base.div_map[:-1], base.div_map[1:])):
                nidx[(idx >= b) & (idx < e)] = i
            index_reduce.append(nidx)
        self.parent = _Sim_Data(base.parent, times, index_reduce)

class Sim_Result(H2MM_result):
    """
    Base class for storing and calculating parameters of simulated data.
    
    .. note::
        
        This is usually created using the :func:`simulate` function, and not with
        the default constructor
    
    Parameters
    ----------
    parent: H2MM_model
        The :class:`H2MM_model` object containing the model from which simulated
        data was created
    sim_times: list[numpy.ndarray]
        The list of burst times of simulated data
    sim_states: list[numpy.ndarray]
        The list of simulated photon states, stored in the .path attribute
    sim_dets: list[numpy.ndarray]
        The list of simulated photon indices, stored in the .parent.index attribute
    """
    def __init__(self, base, sim_times, sim_states, sim_dets):
        self.model = base.model.evaluate(sim_dets, sim_times, inplace=False)
        self.path = sim_states
        self.scale = [np.ones(p.shape) for p in sim_states]
        self.parent = _Sim_List(base.parent, sim_dets, sim_times)
    
    @property
    def dwell_nano_mean(self):
        raise AttributeError("Simulations do not include nanotimes")
    @property
    def nanohist(self):
        raise AttributeError("Simulations do not include nanotimes")
    @property
    def conf_thresh(self):
        raise AttributeError("conf_thresh meaningless for simulated data")
    
    def trim_data(self):
        """Remove photon-level arrays, conserves memory for less important models"""
        for attr in self.large_params:
            if hasattr(self, attr) and attr not in ('path', 'scale'):
                delattr(self, attr)
        

    

def simulate(model, times=None):
    """
    Genearte a Monete-Carlo simulation based on an H2MM_model object.
    By defualt takes the photon arrival times of the data used in the optimization.
    However, a custom set of arrival times may be inputed through the times
    keyword argument.

    Parameters
    ----------
    model : H2MM_model
        A :class:`H2MM_model <BurstSort.H2MM_model>` object, from which a simulated
        dataset will be generated.
    times : list[np.ndarray], optional
        If given, simulation will use these times instead of the arrival times
        of the original data as a starting point. The default is None.

    Returns
    -------
    sim_result : Sim_Result
        A data object containing the simulaetd data, behaves essentially identically
        to a :class:`H2MM_model <BurstSort.H2MM_model>` except without photon
        nanotimes.

    """
    if times is None:
        times = model.parent.parent.times
    sim_state, sim_index = list(), list()
    for tm in times:
        state, index = h2.sim_phtraj_from_times(model.model, tm)
        sim_state.append(state)
        sim_index.append(index)
    sim_result = Sim_Result(model, times, sim_state, sim_index)
    return sim_result
