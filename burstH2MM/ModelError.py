#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: 10/01/2023
# 
# Author: paul

"""
Uncertainties
-------------

.. note::
    
    These classes are intended to be accessed primarily through the H2MM_list
    object and should rarely be interacted with directly by the user
"""

from collections.abc import Iterable
from collections import namedtuple
import numpy as np
from itertools import permutations

import fretbursts as frb
import H2MM_C as h2


def _getindexes(shape, *args):
    """
    Generate numpy array that can be iterated over to get individual indexes of
    each 

    Parameters
    ----------
    shape : tuple[int ...]
        Tuple describing shape of array for which to generate indexes to iterate
        over.
    *args : int, slice
        Numpy compatible set of indexes.

    Returns
    -------
    idx : 2-D numpy.ndarray
        Numpy array of indexes, outer axis=0 is to be iterated over, inner convert
        to tuple to get index for each iteration.

    """
    index = np.indices(shape)
    idx = index[(slice(None),)+args]
    idx = idx.swapaxes(0,-1)
    idx = idx.reshape(-1,index.shape[0])
    return idx

def _unique_by_first(*args):
    """
    Return arrays sorted by the first array, removing repeated values in first
    array.

    Parameters
    ----------
    *args : numpy.ndarray
        Arrays to sort, must be same size/shape.

    Returns
    -------
    numpy.ndarray
        Arrays sorted by first array, removing elements of first array.

    """
    uni, sort = np.unique(args[0], return_index=True)
    return tuple(val[sort] for val in args)


class ModelSet:
    """
    Internal class used to organize multiple H\ :sup:`2`\ MM models, which are
    connected to the same data and divisor scheme, and with the same number of
    states.
    Used for comparing models with slightly different values.
    
    Parameters
    ----------
    modlist : H2MM_list
        The list object defining the divisor scheme
    models : sequence [H2MM_C.h2mm_model]
        The models to compare/organize
        
    """
    def __init__(self, modlist, models):
        self._parent = modlist
        if len(models) != 0:
            if not all(isinstance(m, h2.h2mm_model) and m.nstate == models[0].nstate and m.ndet == models[0].ndet for m in models):
                raise ValueError("Models must be H2MM_C.h2mm_model, and of same number of states/streams")
            if modlist.ndet != models[0].ndet:
                raise ValueError("Models and H2MM_list divisor scheme detectors do not match")
            self._models = np.array(models) if not isinstance(models, np.ndarray) else models
        else:
            self._models = np.empty((0,), dtype=object)
        
        
    
    def add_model(self, model):
        """
        Add a model to the set of models.

        Parameters
        ----------
        model : H2MM_C.h2mm_model
            Model to add to the set.

        Raises
        ------
        TypeError
            Incorrect type.
        ValueError
            Model not of same dimensions as other input.
        
        """
        if not isinstance(model, h2.h2mm_model):
            raise TypeError(f"model must be H2MM_C.h2mm_model, got {type(model)}")
        if model.ndet != self._parent.ndet:
            raise ValueError("Model and H2MM_list divisor scheme detectors do not match")
        if model.nstate != self._models[0].nstate:
            raise ValueError("model must have same number of states as existing models")
        self._models.append(model)
    
    def __len__(self):
        return self._models.size
    
    def __getitem__(self, val):
        return ModelSet(self._parent, self._models[val])
    
    @property
    def _hasE(self):
        """Whether the object supports E"""
        return self._parent._hasE
    
    @property
    def _hasS(self):
        """Whether the object supports S"""
        return self._parent._hasS
    @property
    def nstate(self):
        """Number of states of all models in set"""
        if len(self._models) == 0:
            return None
        return self._models[0].nstate
    
    @property
    def ndet(self):
        """
        Number of photon streams in parent :class:`burstH2MM.BurstSort.BurstData` 
        object.
        """
        if len(self._models) == 0:
            return None
        return self._models[0].ndet
    
    @property
    def E(self):
        """FRET efficiencies of models in Set, organized ``[model, state]``"""
        if not self._hasE:
            raise AttributeError("Parent BurstData must include DexDem and DexAem streams")
        return np.asarray([self._parent._E_from_model(m) for m in self._models])
    
    @property
    def S(self):
        """Stoichiometries of models in Set, organized ``[model, state]``"""
        if not self._hasS:
            raise AttributeError("Parent BurstData must include AexAem stream")
        return np.asarray([self._parent._S_from_model(m) for m in self._models])
    
    @property
    def E_corr(self):
        """Corrected FRET efficiencies of models in Set, organized ``[model, state]``"""
        if not self._hasE:
            raise AttributeError("Parent BurstData must include DexDem and DexAem streams")
        return np.asarray([self._parent._E_from_model_corr(m) for m in self._models])
    
    @property
    def S_corr(self):
        """Corrected stoichiometries of models in Set, organized ``[model, state]``"""
        if not self._hasS:
            raise AttributeError("Parent BurstData must include AexAem stream")
        return np.asarray([self._parent._S_from_model_corr(m) for m in self._models])
    
    @property
    def trans(self):
        """Transition rate of models in Set, organized ``[model, from_state, to_state]``"""
        return np.asarray([m.trans for m in self._models]) / self._parent.parent.data.clk_p
    
    @property
    def prior(self):
        """Initial proabilities of models in Set, organized ``[model, state]``"""
        return np.array([m.prior for m in self._models])
    
    @property
    def obs(self):
        """Emission proabilities of models in Set, organized ``[model, state, stream]``"""
        return np.asarray([m.obs for m in self._models])
    
    @property
    def loglik(self):
        """Loglikelihoods of models in Set"""
        return np.array([m.loglik for m in self._models])
    
    @property
    def models(self):
        """tuple of H2MM_C.h2mm_model objects in set"""
        return tuple(m.copy() for m in self._models)


class Bootstrap_Error:
    """
    Internal class for storing bootstrap error calculations.
    
    This class is usually created when the :meth:`burstH2MM.BurstSort.H2MM_result.bootstrap_eval`
    method is called, and stored under :attr:`burstH2MM.BurstSort.H2MM_result.bootstrap_err`
    
    :class:`burstH2MM.BurstSort.H2MM_result` has several propery aliases for the different
    error values stored by this class.
    
    .. note::
        
        The standard init method is rarely used, rather, this is usually instantiated
        with :meth:`Bootstrap_Error.model_eval` instead.
        
    
    Parameters
    ----------
    parent : H2MM_model
        H2MM_model object for which the error was generated
    models : list[H2MM_C.h2mm_model]
        The H2MM_C.h2mm_model objects of each subset
    
    """
    def __init__(self, parent, models):
        #: parent :class:`burstH2MM.BurstSort.H2MM_result` object
        self.parent = parent
        #: list of optimized models of subsets
        self.models = models
    
    @classmethod
    def model_eval(cls, model, subsets=10):
        """
        Standard way of creating :class:`Bootstrap_Error` objects, takes a model
        and subdivides the data into ``subsets`` number of subsets, and runs
        separate optimizations on each subset, with the original model as an initial
        model, then takes the standard deviation of all model values across the
        separate optimizations, to evaluate the model error.

        Parameters
        ----------
        model : H2MM_result
            The model for which to calculate the bootstrap error.
        subsets : int, optional
            Number of subsets to divide the data into to calculate the bootstrap
            error. The default is 10.

        Returns
        -------
        Bootstrap_Error
            Object organizing the error calculation.

        """
        models = _bootstrap_eval(model, subsets=subsets)
        return cls(model, models)
    
    @property
    def trans_std(self):
        """Standard deviation of transition rates"""
        return self.models.trans.std(axis=0)
    
    @property
    def trans_err(self):
        """Standard error of transition rates"""
        return self.trans_std / np.sqrt(len(self.models))
    
    @property
    def E_std(self):
        """Standard deviation of FRET efficiencies"""
        return self.models.E.std(axis=0)
    
    @property
    def E_err(self):
        """Standard error of FRET efficiencies"""
        return self.E_std / np.sqrt(len(self.models))

    @property
    def E_std_corr(self):
        """Standard deviation of corrected FRET efficiencies"""
        return self.models.E_corr.std(axis=0)
    
    @property
    def E_err_corr(self):
        """Standard error of corrected FRET efficiencies"""
        return self.E_std_corr / np.sqrt(len(self.models))

    @property
    def S_std(self):
        """Standard deviation of stoichiometries"""
        return self.models.S.std(axis=0)
    
    @property
    def S_err(self):
        """Standard error of stoichiometries"""
        return self.S_std / np.sqrt(len(self.models))

    @property
    def S_std_corr(self):
        """Standard deviation of corrected stoichiometries"""
        return self.models.S_corr.std(axis=0)
    
    @property
    def S_err_corr(self):
        """Standard error of corrected stoichiometries"""
        return self.S_std_corr / np.sqrt(len(self.models))
    
    @property
    def prior_std(self):
        """Standard deviation of initial probabilities"""
        return self.models.prior.std(axis=0)
    
    @property
    def prior_err(self):
        """Standard error of initial probabilities"""
        return self.prior_std / np.sqrt(len(self.models))
    
    @property
    def obs_std(self):
        """Standard deviation of emission probabilities"""
        return self.models.obs.std(axis=0)
    
    @property
    def obs_err(self):
        """Standard error of emission probabilities"""
        return self.obs_std / np.sqrt(len(self.models))
        
    
def _bootstrap_eval(model, subsets=10, **kwargs):
    """
    Evaluate the error using subsets of the data (boostrap method).
    
    .. note::
        
        This method takes time, it runs many H2MM optimizations on different
        subsets of you data

    Parameters
    ----------
    model : H2MM_model
        A H2MM_model object for which the error will be calculated.
    subsets : int, optional
        Number of subsets to divide the data-set into. The default is 10.
    **kwargs : dict
        Kwargs handed to H2MM_C.EM_H2MM_C.

    Returns
    -------
    model_array : ModelVar
        The models optimized against subsets
    
    """
    mod_arr = [model.model.optimize(model.parent.index[i::subsets], model.parent.parent.times[i::subsets], 
                                    inplace=False, **kwargs) for i in range(subsets)]
    model_array = ModelSet(model.parent, mod_arr)
    return model_array


def _trans_adjust(mtrans, vals, locs):
    """
    Adjust trans matrix by values stored in vals at locations in locs.
    Normailzation achieved against values along diagonal

    Parameters
    ----------
    mtrans : numpy.ndaray
        Transition probability matrix.
    vals : tuple[float]
        Values to place in transtion prbability matrix.
    locs : tuple[tuple[int, int]]
        Locations to be adjusted.

    Returns
    -------
    mtrans : numpy.ndarray
        Adjusted transition probability matrix.

    """
    mtrans = mtrans.copy()
    for i in range(mtrans.shape[0]):
        adj = sum([mtrans[loc] - val for loc, val in zip(locs, vals) if loc[0] == i])
        mtrans[i,i] += adj
    for loc, val in zip(locs, vals):
        mtrans[loc] = val
    return mtrans

def _obs_adjust(mobs, vals, locs):
    """
    Adjust obs matrix by values stored in vls at locations in locs.
    Normalizatio applied evenly across un-adjusted values

    Parameters
    ----------
    mobs : numpy.ndarray
        Emission probabiity matrix.
    vals : tuple[float]
        Values to place in emission probability matrix.
    locs : tuple[tuple[int, int]]
        Location to be adjusted.

    Returns
    -------
    mobs : numpy.ndarray
        Adjusted emission probability matrix.

    """
    mobs[locs] = vals
    mask = np.ones(mobs.shape, dtype=bool)
    mask[locs] = False
    for i in range(mobs.shape[0]):
        tot = 1 - mobs[i,~mask[i]].sum()
        mobs[i,mask[i,:]] = mobs[i,mask[i]] * tot /  mobs[i,mask[i]].sum()
    return mobs

def _Eobs_adjust(obs, vals, states, DDslc, DAslc):
    """
    Adjust E values of model to values in vals of states in states.

    Parameters
    ----------
    model : numpy.ndarray
        Emission proability matrix
    vals : tuple[float ...]
        list/tuple of floating point from (0.0, 1.0) to adjust states to.
    states : tuple[int ...]
        tuple/list of states to adjust E values.
    DDslc : slice
        Slice of obs matrix for DexDem
    DAslc : slice
        Slice of obs matrix for DexAem
    
    Returns
    -------
    obs : numpy.ndarray
        New obs array with adjusted values.

    """
    DDsum = obs[states,DDslc].sum(axis=1)
    DAsum = obs[states,DAslc].sum(axis=1)
    tot = DDsum + DAsum
    obs[states,DAslc] = np.array([obs[s,DAslc] * t * v / a for s, t, v, a in zip(states, tot, vals, DAsum)])
    obs[states,DDslc] = np.array([obs[s,DDslc] * t * (1-v) / d for s, t, v, d in zip(states, tot, vals, DDsum)])
    return obs

def _E_adjust(model, vals, states):
    """
    Adjust E values of model to values in vals of states in states.

    Parameters
    ----------
    model : H2MM_result
        H2MM_result object to produce adjusted model.
    vals : tuple[float ...]
        list/tuple of floating point from (0.0, 1.0) to adjust states to.
    states : tuple[int ...]
        tuple/list of states to adjust E values.

    Returns
    -------
    obs : numpy.ndarray
        New obs array with adjusted values.

    """
    obs = model.model.obs
    DDslc = model._DexDem_slc
    DAslc = model._DexAem_slc
    _Eobs_adjust(obs, vals, states, DDslc, DAslc)
    return obs

def _Sobs_adjust(obs, vals, states, DDslc, DAslc, AAslc):
    """
    Adjust S values of model to values in vals of states in states.

    Parameters
    ----------
    model : numpy.ndarray
        Emission proability matrix
    vals : tuple[float ...]
        list/tuple of floating point from (0.0, 1.0) to adjust states to.
    states : tuple[int ...]
        tuple/list of states to adjust S values.
    DDslc : slice
        Slice of obs matrix for DexDem
    DAslc : slice
        Slice of obs matrix for DexAem
    AAslc : slice
        Slice of obs matrix for AexAem
    
    Returns
    -------
    obs : numpy.ndarray
        New obs array with adjusted values.

    """
    Dsum = obs[states,DDslc].sum(axis=1) + obs[states,DAslc].sum(axis=1)
    AAsum = obs[states,AAslc].sum(axis=1)
    tot = Dsum + AAsum
    obs[states,AAslc] = np.array([obs[s,AAslc] * (1-v) * t / a for s, t, v, a in zip(states, tot, vals, AAsum)])
    obs[states,DDslc] = np.array([obs[s,DDslc] * v * t / d for s, t, v, d in zip(states, tot, vals, Dsum)])
    obs[states,DAslc] = np.array([obs[s,DAslc] * v * t / d for s, t, v, d in zip(states, tot, vals, Dsum)])
    return obs


def _S_adjust(model, vals, states):
    """
    Adjust S values of model to values in vals of states in states.

    Parameters
    ----------
    model : H2MM_result
        H2MM_result object to produce adjusted model.
    vals : tuple[float ...]
        list/tuple of floating point from (0.0, 1.0) to adjust states to.
    states : tuple[int ...]
        tuple/list of states to adjust S values.

    Returns
    -------
    obs : numpy.ndarray
        New obs array with adjusted values.

    """
    obs = model.model.obs
    DDslc = model._DexDem_slc
    DAslc = model._DexAem_slc
    AAslc = model._AexAem_slc
    obs = _Sobs_adjust(obs, vals, states, DDslc, DAslc, AAslc)
    return obs


def _param_space(err, state, rng, steps, param):
    pgroup = err._param_groups[param]
    if isinstance(state, Iterable) and len(state) == 1:
        state = state[0]
    try:
        no_param = np.any(getattr(err, param).mask[state])
    except:
        raise ValueError(f"err does not have {param}")
    need_param = isinstance(rng, (float, int, type(None)))
    if no_param and need_param:
        raise ValueError(f"{param} error not calculated, cannot auto-generate range")
    if rng is None:
        factor = 2
    elif isinstance(rng, (int, float)):
        factor = rng
    if isinstance(rng, (int, float, type(None))):
        err_param = getattr(err, pgroup.err)[state]
        opt_param = getattr(err.parent, param)[state]
        low = factor*(err_param[0] - opt_param) + opt_param
        high = factor*(err_param[1] - opt_param) + opt_param
        if low < 0.0:
            low = 0.0
        if high > 1.0:
            high = 1.0
    elif len(rng) == 2:
        low, high = rng
    else:
        rng_arr = rng
    if isinstance(rng, (int, float, type(None))) or len(rng) == 2:
        rng_arr = np.linspace(low, high, steps, endpoint=True)
    return rng_arr

def _param_space_ll(err, state, rng, steps, param):
    pgroup = err._param_groups[param]
    rng_arr = _param_space(err, state, rng, steps, param)
    # check for values that already exist
    mask = np.ones(rng_arr.shape, dtype=bool)
    ll_arr = np.empty(rng_arr.shape)
    # See which have been calculated
    rng  = getattr(err, pgroup.rng)[state]
    rng_ll = getattr(err, pgroup.rng_ll)[state]
    _, keep, drop = np.intersect1d(rng, rng_arr, return_indices=True)
    mask[drop] = False
    fill_arr = rng_ll[keep]
    rng_arr_m = rng_arr[mask]
    # perform calculation
    prior, trans = err.model.prior, err.model.trans
    obs = (pgroup.afunc(err.parent, (S, ), (state, )) for S in rng_arr_m)
    models = [h2.h2mm_model(prior, trans, ob) for ob in obs]
    models = h2.H2MM_arr(models, err.parent.parent.index, err.parent.parent.parent.times)
    # infill already calculated numbers
    ll_arr[mask] = np.array([mod.loglik for mod in models])
    ll_arr[~mask] = fill_arr
    return rng_arr, ll_arr


def err_trans_search(model, loc, flex=5e-2, factor=1e-1, max_iter=100, thresh=0.5):
    """
    Search for the deviation above and bellow the optimizal

    Parameters
    ----------
    model : H2MM_result
        Optimization result for which to calculate the error in transition rates.
    loc : tuple[int, int]
        Transition rate (from_state, to_state) for which to calculate the
        error.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. The default is 5e-2.
    factor : float, optional
        Factor by which to offset the transition rate for initial search. The default is 1e-1.
    max_iter : int, optional
        maximum number of attempts to try in finding target transition rate.
        The default is 100.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is 0.5.

    Returns
    -------
    ll_rt : numpy.ndarray
        low/high transition rates with target loglikelihoods.
    ll_lh : numpy.ndarray
        Actual loglikelihoods of low/high transition rates.
    rate_array : numpy.ndarray
        All rates attempted in search.
    ll_array : TYPE
        All loglikelihoods attempted in search.

    """
    # define two lambda functions for the high and low transition rate transformations
    base = np.log(model.model.trans[loc])
    upmult = lambda f: np.exp(base+f)
    downmult = lambda f: np.exp(base-f)
    m_iter = max_iter
    # pull out variables to make code less verbose
    prior, trans, obs = model.model.prior, model.model.trans, model.model.obs
    index, times = model.parent.index, model.parent.parent.times
    it, mit = 0, 0
    # set minimuma nd maximum values to accept as "converged"
    targl, targh = model.loglik - thresh - flex, model.loglik - thresh + flex
    # pre-allocate arrays, instead of using list
    rate_array, ll_array = np.empty(10), np.empty(10)
    sz = ll_array.size
    ll_lh, ll_rt = np.zeros(2), np.zeros(2)
    # iterate first low values, then high values
    for i, mult in enumerate((downmult, upmult)):
        low, high, cont = 0.0, np.inf, True
        while cont:
            rate = mult(factor)
            tr = _trans_adjust(trans, (rate,), (loc,))
            # check adjusted array still valid trans array
            if np.all(tr > 0.0):
                mod = h2.h2mm_model(prior, tr, obs)
                llm = h2.H2MM_arr(mod,index, times)
                ll = llm.loglik
                rate_array[it] = rate
                ll_array[it] = ll
                it += 1
                mit = 0
            else:
                mit += 1
                if mit > 10:
                    break
                high = factor
                factor = factor / 2
                continue
                
            # cases of bellow/above target range
            if ll < targl:
                high = factor
            elif ll > targh:
                low = factor
            else:
                ll_lh[i] = ll
                ll_rt[i] = rate
                cont = False
            factor = (high+low)/2 if high != np.inf else factor * 2
            # final cleanup- break if 
            if it >= max_iter:
                cont = False
            elif it >= sz - 1:
                ll_array = np.concatenate([ll_array, np.empty(10)])
                rate_array = np.concatenate([rate_array, np.empty(10)])
                sz = ll_array.size
        max_iter = max_iter + m_iter
    # cut unused parts of array
    rate_array, ll_array = rate_array[:it], ll_array[:it]
    sort = np.argsort(rate_array)
    rate_array = rate_array[sort]
    ll_array = ll_array[sort]
    ll_rt = ll_rt / model.parent.parent.data.clk_p
    rate_array = rate_array / model.parent.parent.data.clk_p
    return ll_rt, ll_lh, rate_array, ll_array

def _trans_space(err, trans, rng=None, steps=20):
    """
    Calculate the loglikelihoods of models with a range trans values of state.
    By default, takes +/- 10 times the optimal transition rate.
    into 20 steps.

    Parameters
    ----------
    err : Loglik_Error
        Error object to use as input.
    trans : [int, int]
        Which state to evaluate.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) trans vals, or factor by which to multiply
        error range, or array of trans values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.

    Raises
    ------
    ValueError
        Impropper rng input.

    Returns
    -------
    rng_arr : numpy.ndarray
        trans values of evaluated models.
    ll : numpy.ndarray
        loglikelihoods of evaluated models.

    """
    s0, s1 = trans
    logclk_p = np.log10(err.parent.parent.parent.data.clk_p)
    if np.any(getattr(err, '_trans').mask[trans]) and isinstance(rng, (float, int, type(None))):
        raise ValueError("trans error not calculated, cannot auto-generate range")
    elif rng is None:
        factor = 0.5
    elif isinstance(rng, (int, float)):
        factor = rng
    if isinstance(rng, (int, float, type(None))):
        Tlow = np.log10(err._trans[s0,s1,0]) - factor + logclk_p
        Thigh = np.log10(err._trans[s0,s1,1]) + factor + logclk_p
    elif len(rng) == 2:
        Tlow, Thigh = rng
        Tlow, Thigh = np.log10(Tlow) + logclk_p, np.log10(Thigh) + logclk_p
    else:
        rng_arr = np.array(rng) * err.parent.parent.parent.data.clk_p
    if isinstance(rng, (int, float, type(None))) or len(rng) == 2:
        rng_arr = np.logspace(Tlow, Thigh, steps, endpoint=True)
    return rng_arr
    
def trans_space_ll(err, trans, rng=None, steps=20):
    """
    Calculate the loglikelihoods of models with a range trans values of state.
    By default, takes +/- 10 times the optimal transition rate.
    into 20 steps.

    Parameters
    ----------
    err : Loglik_Error
        Error object to use as input.
    trans : [int, int]
        Which state to evaluate.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) trans vals, or factor by which to multiply
        error range, or array of trans values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.

    Raises
    ------
    ValueError
        Impropper rng input.

    Returns
    -------
    rng_arr : numpy.ndarray
        trans values of evaluated models.
    ll : numpy.ndarray
        loglikelihoods of evaluated models.

    """
    rng_arr = _trans_space(err, trans, rng=rng, steps=steps)
    # check for values that already exist
    mask = np.ones(rng_arr.shape, dtype=bool)
    ll_arr = np.empty(rng_arr.shape)
    # See which have been calculated
    rng  = err.t_rate_rng[trans]
    rng_ll = err.t_ll_rng[trans]
    _, keep, drop = np.intersect1d(rng, rng_arr, return_indices=True)
    mask[drop] = False
    fill_arr = rng_ll[keep]
    rng_arr_m = rng_arr[mask]
    # perform calculation
    prior, mtrans, obs = err.model.prior, err.model.trans, err.model.obs
    trans_arr = (_trans_adjust(mtrans, (t, ), (trans, )) for t in rng_arr_m)
    models = [h2.h2mm_model(prior, t, obs) for t in trans_arr]
    models = h2.H2MM_arr(models, err.parent.parent.index, err.parent.parent.parent.times)
    # infill already calculated numbers
    ll_arr[mask] = np.array([mod.loglik for mod in models])
    ll_arr[~mask] = fill_arr
    rng_arr = rng_arr / err.parent.parent.parent.data.clk_p
    return rng_arr, ll_arr


def err_param_search(err, state, flex, step, max_iter, thresh, param):
    """
    Base function for calculating the loglikelihood error of the given param of a particular
    state.

    Parameters
    ----------
    err : Loglik_Error
        Loglik_Error for which to calculate the loglikelihood error.
    state : int
        state to evaluate loglikelihood error of param.
    flex : float
        Allowed variability in target decrease in loglikelihood.
    step : float
        Amount to shift param in each iteration before low/high bounds have been found.
    max_iter : int
        maximum number of attempts to try in finding target transition rate.
    thresh : float
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error.

    Returns
    -------
    ll_rt : numpy.ndarray
        low/high param values with target loglikelihoods.
    ll_lh : numpy.ndarray
        Actual loglikelihoods of low/high param values.
    rate_array : numpy.ndarray
        All rates attempted in search.
    ll_array : numpy.ndarray
        All loglikelihoods attempted in search.

    """
    model = err.parent
    pgroup =err._param_groups[param]
    # define two lambda functions for the high and low transition rate transformations
    base = getattr(model, param)[state]
    downstep = lambda v: v - step if high == 0.0 else (high + low) / 2
    upstep = lambda v: v + step if high == 1.0 else (high + low) / 2
    m_iter = max_iter
    # pull out variables to make code less verbose
    prior, trans = model.model.prior, model.model.trans
    index, times = model.parent.index, model.parent.parent.times
    it = 0
    # set minimuma nd maximum values to accept as "converged"
    targl, targh = model.loglik - thresh - flex, model.loglik - thresh + flex
    # pre-allocate arrays, instead of using list
    param_arr, ll_array = np.empty(10), np.empty(10)
    sz = ll_array.size
    ll_lh, ll_rt = np.empty(2), np.empty(2)
    # iterate first low values, then high values
    for i, fstep in enumerate((downstep, upstep)):
        low = base
        high = 0.0 if fstep is downstep else 1.0
        cont = True
        P = base
        while cont:
            Pc = fstep(P)
            if Pc > 0.0 and Pc < 1.0:
                obs = pgroup.afunc(model, (Pc,), (state,))
                mod = h2.h2mm_model(prior, trans, obs)
                llm = h2.H2MM_arr(mod, index, times)
                ll = llm.loglik
                param_arr[it] = Pc
                ll_array[it] = ll
                P = Pc
                it += 1
            else:
                Pc = (P + Pc) / 2
                continue
            # cases of bellow/above target range
            if ll < targl:
                high = P
            elif ll > targh:
                low = P
            else:
                ll_lh[i] = ll
                ll_rt[i] = P
                cont = False
            if it >= max_iter:
                cont = False
            elif it >= sz:
                ll_array = np.concatenate([ll_array, np.empty(10)])
                param_arr = np.concatenate([param_arr, np.empty(10)])
                sz = ll_array.size
        max_iter = max_iter + m_iter
    # cut unused parts of array
    param_arr, ll_array = param_arr[:it], ll_array[:it]
    sort = np.argsort(param_arr)
    param_arr = param_arr[sort]
    ll_array = ll_array[sort]
    return ll_rt, ll_lh, param_arr, ll_array


def err_E_search(err, state, flex=5e-2, step=1e-2, max_iter=100, thresh=0.5):
    """
    Base function for calculating the loglikelihood error of the E of a particular
    state.

    Parameters
    ----------
    model : Loglik_Error
        Loglik_Error for which to calculate the loglikelihood error.
    state : int
        state to evaluate loglikelihood error of E.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. The default is 5e-2.
    step : float, optional
        Amount to shift E in each iteration before low/high bounds have been found. 
        The default is 1e-2.
    max_iter : int, optional
        maximum number of attempts to try in finding target transition rate.
        The default is 100.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is 0.5.

    Returns
    -------
    ll_rt : numpy.ndarray
        low/high E values with target loglikelihoods.
    ll_lh : numpy.ndarray
        Actual loglikelihoods of low/high E values.
    rate_array : numpy.ndarray
        All rates attempted in search.
    ll_array : numpy.ndarray
        All loglikelihoods attempted in search.

    """
    return err_param_search(err, state, flex, step, max_iter, thresh, 'E')


def E_space(err, state, rng=None, steps=20):
    """
    Calculate the loglikelihoods of models with a range E values of state.
    By default, takes a range twice as large as the calculated error, and divides
    into 20 steps.

    Parameters
    ----------
    err : Loglik_Error
        Error object to use as input.
    state : int
        Which state to evaluate.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) S vals, or factor by which to multiply
        error range, or array of S values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.

    Raises
    ------
    ValueError
        Impropper rng input.

    Returns
    -------
    rng_arr : numpy.ndarray
        E values of evaluated models.
    ll_arr : numpy.ndarray
        loglikelihoods of evaluated models.

    """
    return _param_space_ll(err, state, rng, steps, 'E')


def err_S_search(err, state, flex=5e-2, step=1e-2, max_iter=100, thresh=0.5):
    """
    Base function for calculating the loglikelihood error of the S of a particular
    state.

    Parameters
    ----------
    err : Loglik_Error
        Loglik_Error for which to calculate the loglikelihood error.
    state : int
        state to evaluate loglikelihood error of S.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. The default is 5e-2.
    step : float, optional
        Amount to shift E in each iteration before low/high bounds have been found. 
        The default is 1e-2.
    max_iter : int, optional
        maximum number of attempts to try in finding target transition rate.
        The default is 100.
    thresh : float, optional
        Decrease in loglikelihood of adjusted model to target as characterizing
        the error. The default is 0.5.

    Returns
    -------
    ll_rt : numpy.ndarray
        low/high S values with target loglikelihoods.
    ll_lh : numpy.ndarray
        Actual loglikelihoods of low/high S values.
    rate_array : numpy.ndarray
        All rates attempted in search.
    ll_array : TYPE
        All loglikelihoods attempted in search.

    """
    return err_param_search(err, state, flex, step, max_iter, thresh, 'S')


def S_space(err, state, rng=None, steps=20):
    """
    Calculate the loglikelihoods of models with a range S values of state.
    By default, takes a range twice as large as the calculated error, and divides
    into 20 steps.

    Parameters
    ----------
    err : Loglik_Error
        Error object to produce parameter range.
    state : int
        State to evaluate.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) S vals, or factor by which to multiply
        error range, or array of S values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 20.

    Raises
    ------
    ValueError
        Impropper rng input.

    Returns
    -------
    rng_arr : numpy.ndarray
        S values of evaluated models.
    ll_arr : numpy.ndarray
        loglikelihoods of evaluated models.

    """
    return _param_space_ll(err, state, rng, steps, 'S')

def _min_change(val):
    """ 
    Return floats that are the smallest possible decriment and increment of the input.
    Used to fix a value.
    """
    num, den = val.as_integer_ratio()
    return (num-1)/den, (num+1)/den

def _fix_h2mm_trans(trans, state):
    """
    Generate limits object to fix the transition rate value of a given state
    
    Parameters
    ----------
    trans : numpy.ndarray
        Transition probability matrix.
    state : tuple[int, int]
        State to fix.

    Returns
    -------
    H2MM_C.h2mm_limits
        Object to fix the value of a given state.

    """
    min_trans = np.zeros(trans.shape)
    max_trans = np.ones(trans.shape)
    min_trans[state], max_trans[state] = _min_change(trans[state])
    return h2.h2mm_limits(min_trans=min_trans, max_trans=max_trans)

    
def covar_trans(err, trans, rng=None, steps=10, **kwargs):
    """
    Calculate the covariance of a given transition.
    Runs optimizations with the given transition parameter fixed to the value returned
    by :meth:`burstH2MM.ModelError.Loglik_Error.trans_space`, and stores models in 
    cooresponding element of :attr:`burstH2MM.ModelError.Loglik_Error.trans_covar` 

    Parameters
    ----------
    err : Loglik_Error
        Logklik_Error to calculate the covariance of.
    trans : tuple(int, int) 
        Transition along which to run optimizations with fixed values of transition rate.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) E vals, or factor by which to multiply
        error range, or array of E values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 10.
    **kwargs : dict
        Arguments passed to ``H2MM_C.h2mm_model.optimize``.
    
    Returns
    -------
    ModelSet
        ModelSet of optimized models.
    
    """
    # extract useful values
    prior, tmat, obs = err.model.prior, err.model.trans, err.model.obs
    index, times = err.parent.parent.index, err.parent.parent.parent.times
    # setup limits/other kwargs for h2mm optimizations
    kw = dict(max_iter=400, converged_min=1e-6*err.flex)
    kw.update(kwargs)
    # get transition rate array
    rng_arr = _trans_space(err, trans, rng=rng, steps=steps)
    # convert transition rates to generators fo h2mm models and bounds
    trans_arr = (_trans_adjust(tmat, (t, ), (trans, )) for t in rng_arr)
    model_arr = (h2.h2mm_model(prior, t, obs) for t in trans_arr)
    opt_arr = [model.optimize(index, times, bounds=_fix_h2mm_trans(model.trans, trans),
                              bounds_func='revert', **kw) for model in model_arr]
    return ModelSet(err.parent.parent, opt_arr)


def _fix_h2mm_E(new, current, old, bound):
    """
    bounds_func function fixing the given E value
    

    Parameters
    ----------
    new : H2MM_C.h2mm_model
        new model.
    current : H2MM_C.h2mm_model
        Current model.
    old : H2MM_C.h2mm_model
        Old model.
    bound : ((float, ), (int, ), slice, slice)
        vals, states, DDslc, DAslc to define the value to fix E to, the state to
        fix, and slices for DexDem and DexAem

    Returns
    -------
    H2MM_C.h2mm_model
        Updated model with fixed E value.

    """
    vals, states, DDslc, DAslc = bound
    obs = _Eobs_adjust(new.obs, vals, states, DDslc, DAslc)
    return h2.h2mm_model(new.prior, new.trans, obs)

def _fix_h2mm_S(new, current, old, bound):
    """
    bounds_func function fixing the given S value
    

    Parameters
    ----------
    new : H2MM_C.h2mm_model
        new model.
    current : H2MM_C.h2mm_model
        Current model.
    old : H2MM_C.h2mm_model
        Old model.
    bound : ((float, ), (int, ), slice, slice, slice)
        vals, states, DDslc, DAslc, AAslc to define the value to fix S to, the 
        state to fix, and slices for DexDem, DexAem, and AexAem

    Returns
    -------
    H2MM_C.h2mm_model
        Updated model with fixed S value.

    """
    vals, states, DDslc, DAslc, AAslc = bound
    obs = _Sobs_adjust(new.obs, vals, states, DDslc, DAslc, AAslc)
    return h2.h2mm_model(new.prior, new.trans, obs)


def covar_E(err, state, rng=None, steps=10, **kwargs):
    """
    Calculate the covariance of a given state along E.
    Runs optimizations with the given E parameter fixed to the value returned
    by :meth:`burstH2MM.ModelError.Loglik_Error.E_space`, and stores models in 
    cooresponding element of :attr:`burstH2MM.ModelError.Loglik_Error.E_covar` 
    
    Parameters
    ----------
    err : Loglik_Error
        Logklik_Error to calculate the covariance of.
    state : int 
        State along which to run optimizations with fixed values of E.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) E values, or factor by which to multiply
        error range, or array of E values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 10.
    **kwargs : dict
        Arguments passed to ``H2MM_C.h2mm_model.optimize``.
    
    Returns
    -------
    ModelSet
        ModelSet of optimized models.
    
    """
    index, times = err.parent.parent.index, err.parent.parent.parent.times
    prior, trans = err.model.prior, err.model.trans
    rng_arr = _param_space(err, state, rng, steps, 'E')
    obs_arr = (_E_adjust(err.parent, (o, ), (state, )) for o in rng_arr)
    model_arr = (h2.h2mm_model(prior, trans, o) for o in obs_arr)
    # setup limits/other kwargs for h2mm optimizations
    kw = dict(max_iter=400, converged_min=1e-6*err.flex)
    kw.update(kwargs)
    # run calculation
    model_arr = [model.optimize(index, times, bounds_func=_fix_h2mm_E, 
                                bounds=((rg,),(state,), 
                                        err.parent._DexDem_slc, 
                                        err.parent._DexAem_slc ), **kw) for model, rg in zip(model_arr, rng_arr)]
    return ModelSet(err.parent.parent, model_arr)


def covar_S(err, state, rng=None, steps=10, **kwargs):
    """
    Calculate the covariance of a given state along S.
    Runs optimizations with the given S parameter fixed to the value returned
    by :meth:`burstH2MM.ModelError.Loglik_Error.S_space`, and stores models in 
    cooresponding element of :attr:`burstH2MM.ModelError.Loglik_Error.S_covar` 
    
    Parameters
    ----------
    err : Loglik_Error
        Logklik_Error to calculate the covariance of.
    state : int 
        State along which to run optimizations with fixed values of S.
    rng : tuple[int, int], int, float, numpy.ndarray, optional
        Custom range specified by (low, high) S values, or factor by which to multiply
        error range, or array of S values. The default is None.
    steps : int, optional
        Number of models to evaluate. The default is 10.
    **kwargs : dict
        Arguments passed to ``H2MM_C.h2mm_model.optimize``.
    
    Returns
    -------
    ModelSet
        ModelSet of optimized models.
    
    """
    index, times = err.parent.parent.index, err.parent.parent.parent.times
    prior, trans = err.model.prior, err.model.trans
    rng_arr = _param_space(err, state, rng, steps, 'S')
    obs_arr = (_S_adjust(err.parent, (o, ), (state, )) for o in rng_arr)
    model_arr = (h2.h2mm_model(prior, trans, o) for o in obs_arr)
    # setup limits/other kwargs for h2mm optimizations
    kw = dict(max_iter=400, converged_min=1e-6*err.flex)
    kw.update(kwargs)
    # run calculation
    model_arr = [model.optimize(index, times, bounds_func=_fix_h2mm_S, 
                                bounds=((rg,),(state,), 
                                        err.parent._DexDem_slc, 
                                        err.parent._DexAem_slc, 
                                        err.parent._AexAem_slc), **kw) for model, rg in zip(model_arr, rng_arr)]
    return ModelSet(err.parent.parent, model_arr)
    

LL_Set = namedtuple('LL_Set', ['err', 'err_ll', 'rng', 'rng_ll', 'afunc', 'sfunc'])


class Loglik_Error:
    """
    Object for evaluation of error by comparing loglikelihoods of models with
    parameters offset and the optimial model.
    
    Created from :class:`burstH2MM.BurstData.H2MM_result` object, with empty arrays allocated,
    but values are only calcualted when various methods are called.
    
    """
    #: default decrease in loglikelihood to consider as the bounds of error
    _thresh = 0.5
    #: precision of loglikelihood needed to consider error bound to be found
    _flex = 5e-3
    #: dictionary of parameter names to map to different types
    _param_groups = {'E':LL_Set('_E', 'E_ll', 'E_rng', 'E_ll_rng', _E_adjust, E_space),
                    'S':LL_Set('_S', 'S_ll', 'S_rng', 'S_ll_rng', _S_adjust, S_space),
                    'trans':LL_Set('_trans', 'trans_ll', 't_rate_rng', 't_ll_rng',
                                   _trans_adjust, _trans_space)}
    
    def __init__(self, parent):
        self.parent = parent
        #: all transition rates evalated so far, oranized as numpy.ndarray of numpy.ndarrays
        self.t_rate_rng = np.empty((self.nstate, self.nstate), dtype=object)
        #: loglik of all transition rates evalated so far, oranized as numpy.ndarray of numpy.ndarrays
        self.t_ll_rng = np.empty((self.nstate, self.nstate), dtype=object)
        self._trans = np.ma.empty((self.nstate, self.nstate, 2))
        self._trans[...] = np.ma.masked
        self._trans.fill_value = np.inf
        #: loglikelihoods of the low/high transision rate error range
        self.trans_ll = np.ma.empty((self.nstate, self.nstate, 2))
        self.trans_ll[...] = np.ma.masked
        self.trans_ll.fill_value = np.nan
        #: Covariant optimizations of FRET efficiencies
        self.trans_covar = np.ma.empty((self.nstate, self.nstate), dtype=object)
        self.trans_covar[...] = np.ma.masked
        for i, j in permutations(range(self.nstate), 2):
            self.t_rate_rng[i,j] = np.empty((0,))
            self.t_ll_rng[i,j] = np.empty((0,))
        if parent._hasE:
            #: all FRET efficiencies evalated so far, oranized as numpy.ndarray of numpy.ndarrays
            self.E_rng = np.empty((self.nstate), dtype=object)
            #: loglikelihoods of all FRET efficiencies evalated so far, oranized as numpy.ndarray of numpy.ndarrays
            self.E_ll_rng = np.empty((self.nstate), dtype=object)
            self._E = np.ma.empty((self.nstate,2)) 
            self._E[...] = np.ma.masked
            self._E.fill_value = np.nan
            #: loglikelihoods of low/high values of evaluated E error
            self.E_ll =  np.ma.empty((self.nstate,2))
            self.E_ll[...] = np.ma.masked
            self.E_ll.fill_value = -np.inf
            for i in range(self.nstate):
                self.E_rng[i] = np.empty((0,))
                self.E_ll_rng[i] = np.empty((0,))
            #: Array of covarience calculations of FRET efficiency
            self.E_covar = np.ma.empty((self.nstate, ), dtype=object)
            self.E_covar[...] = np.ma.masked
        if parent._hasS:
            #: all stoichiometries evalated so far, oranized as numpy.ndarray of numpy.ndarrays
            self.S_rng = np.empty((self.nstate), dtype=object)
            #: loglikelihoods of all stoichiometries evalated so far, oranized as numpy.ndarray of numpy.ndarrays
            self.S_ll_rng = np.empty((self.nstate), dtype=object)
            self._S = np.ma.empty((self.nstate,2)) 
            self._S[...] = np.ma.masked
            self._S.fill_value = np.nan
            #: loglikelihoods of low/high values S error
            self.S_ll =  np.ma.empty((self.nstate,2))
            self.S_ll[...] = np.ma.masked
            self.S_ll.fill_value = -np.inf
            for i in range(self.nstate):
                self.S_rng[i] = np.empty((0,))
                self.S_ll_rng[i] = np.empty((0,))
            #: Covariant optimizations of stoichiometries
            self.S_covar = np.ma.empty((self.nstate, ), dtype=object)
            self.S_covar[...] = np.ma.masked
    
    @property
    def model(self):
        """Parent :class:`burstH2MM.BurstSort.H2MM_result` object"""
        return self.parent.model
    
    @property
    def nstate(self):
        """Nubmer of states in model"""
        return self.parent.model.nstate
    
    @property
    def thresh(self):
        """The threshold in loglikelihood to consider the uncertainty range"""
        return self._thresh
    
    @thresh.setter
    def thresh(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError(f"thresh must be a number, got type: {type(val)}")
        elif val <= 0:
            raise ValueError("thresh must be positive")
        self._thresh = val
    
    @property
    def flex(self):
        """Accuracy to which the threshold must be found"""
        return self._flex
    
    @flex.setter
    def flex(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError(f"thresh must be a number, got type: {type(val)}")
        elif val <= 0:
            raise ValueError("thresh must be positive")
        self._flex = val
    
    @property
    def trans(self):
        """[nstate, nstate, 2] matrix of low/high error transition rates"""
        return self._trans
    
    @property
    def trans_low(self):
        """
        Transition rates lower than the optimal that have loglikelihood thresh
        lower than the optimal
        """
        return self.trans[:,:,0]

    @property
    def trans_high(self):
        """
        Transition rates higher than the optimal that have loglikelihood thresh
        lower than the optimal
        """
        return self.trans[:,:,1]
    
    @property
    def E_lh(self):
        """Uncertainty in E given as low/high values, shape [state, 2]"""
        return self._E
    
    @property
    def S_lh(self):
        """Uncertainty in S given as low/high values, shape [state, 2]"""
        return self._S
    
    @property
    def E(self):
        """
        Estimated error based on the loglikelihood range of low/high E values
        of calculated error
        """
        return (self._E[:,1] - self._E[:,0])/2
    
    @property
    def S(self):
        """
        Estimated error based on the loglikelihood range of low/high S values
        of calculated error
        """
        return (self._S[:,1] - self._S[:,0])/2
    
    def clear_trans(self):
        """Clear all transition rate related uncertainty values
        
        .. note::
            
            Does not clear range arrays.
        
        """
        self._trans = np.ma.empty((self.nstate, self.nstate, 2))
        self._trans[...] = np.ma.masked
        self._trans.fill_value = np.inf
        self.trans_ll = np.ma.empty((self.nstate, self.nstate, 2))
        self.trans_ll[...] = np.ma.masked
        self.trans_ll.fill_value = -np.inf
        
    
    def clear_E(self):
        """Clear all FRET efficiency related uncertainty values
        
        .. note::
            
            Does not clear range arrays.
        
        """
        if not self.parent._hasE:
            raise AttributeError("Parent data object does not have E")
        self._E = np.ma.empty((self.nstate,2)) 
        self._E[...] = np.ma.masked
        self._E.fill_value = np.nan
        self.E_ll =  np.ma.empty((self.nstate,2))
        self.E_ll[...] = np.ma.masked
        self.E_ll.fill_value = -np.inf
        
    def clear_S(self):
        """Clear all stoichiometry related uncertainty values
        
        .. note::
            
            Does not clear range arrays.
        
        """
        if not self.parent._hasS:
            raise AttributeError("Parent data object does not have S")
        self._S = np.ma.empty((self.nstate,2)) 
        self._S[...] = np.ma.masked
        self._S.fill_value = -np.inf
        self.S_ll =  np.ma.empty((self.nstate,2))
        self.S_ll[...] = np.ma.masked
        self.S_ll.fill_value = -np.inf
        
    def clear_all(self):
        """Clear all uncertainty values.
        
        .. note::
            
            Does not clear range arrays.
        
        """
        self.clear_trans()
        if self.parent._hasE:
            self.clear_E()
        if self.parent._hasS:
            self.clear_S()
        
    def get_trans_err(self, *args, flex=None, factor=1e-1, max_iter=100, thresh=None):
        """
        Retrieve or calculate the loglikelihood error of specific transition rate(s).

        Parameters
        ----------
        *args : location specifier
            Transition from_state to_state as integers or slices of transition
            rates for which the error should be retrieved/calculated.
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        factor : float, optional
            Factor by which to offset the transition rate for initial search. The default is 1e-1.
        max_iter : int, optional
            maximum number of attempts to try in finding target transition rate.
            The default is 100.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.

        Returns
        -------
        numpy.ndarray
            Array of low/high bounds of the loglikelihood error of the requested
            transition rates.

        """
        locs = _getindexes((self.nstate, self.nstate), *args)
        for loc in locs:
            if np.any(self._trans.mask[tuple(loc)]):
                self.trans_eval(locs=(loc, ), flex=flex, factor=factor, max_iter=max_iter, thresh=thresh)
        return self.trans[args+(slice(None),)]
    
    def get_E_err(self, *args, flex=None, step=1e-2, max_iter=100, thresh=None, simple=True):
        """
        Retrieve or calculate the loglikelihood error of specific FRET efficiency(s).

        Parameters
        ----------
        *args : location specifier
            State(s) as integer or slice of FRET efficiency(s) for which
            the error should be retrieved/calculated.
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        factor : float, optional
            Factor by which to offset the transition rate for initial search. The default is 1e-1.
        max_iter : int, optional
            maximum number of attempts to try in finding target transition rate.
            The default is 100.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.
        simple : bool, optional
            Whether to return 1 (True) or 2 (False) values showing half the difference,
            or the low and high values of the requested states

        Returns
        -------
        numpy.ndarray
            Array of loglikelihood errors of requested FRET efficiency(s).

        """
        locs = _getindexes((self.nstate, ), *args)
        for loc in locs:
            loc = tuple(loc)
            if np.any(self._E.mask[loc]):
                self.E_eval(locs=loc, flex=flex, step=step, max_iter=max_iter, thresh=thresh)
        return self.E[args] if simple else self.E_lh[args+(slice(None),)]
    
    def get_S_err(self, *args, flex=None, step=1e-2, max_iter=100, thresh=None, simple=True):
        """
        Retrieve or calculate the loglikelihood error of specific stoichiometry(s).

        Parameters
        ----------
        *args : location specifier
            State(s) as integer or slice of stoichiometries(s) for which the error
            should be retrieved/calculated.
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        step : float, optional
            Shift in S for initial search. The default is 1e-2.
        max_iter : int, optional
            maximum number of attempts to try in finding target transition rate.
            The default is 100.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.
        simple : bool, optional
            Whether to return 1 (True) or 2 (False) values showing half the difference,
            or the low and high values of the requested states

        Returns
        -------
        numpy.ndarray
            Array of loglikelihood errors of requested stoichiometry(s).

        """
        locs = _getindexes((self.nstate, ), *args)
        for loc in locs:
            loc = tuple(loc)
            if np.any(self._S.mask[loc]):        
                self.S_eval(locs=loc, flex=flex, step=step, max_iter=max_iter, thresh=thresh)
        return self.S[args] if simple else self.S_lh[args+(slice(None),)]
    
    def trans_eval(self, locs=None, flex=None, factor=1e-1, max_iter=100, thresh=None):
        """
        Evaluate the loglikelihood based error for transition rates.

        Parameters
        ----------
        locs : tuple[tuple[int, int] ...], optional
            Locations of trans errors to evaluate, if None, will evaluate all 
            transition rates. The default is None.
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        factor : float, optional
            Factor by which to offset the transition rate for initial search. The default is 1e-1.
        max_iter : int, optional
            maximum number of attempts to try in finding target transition rate.
            The default is 100.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.

        """
        if locs is None:
            locs = permutations(range(self.nstate), 2)
        if thresh is None:
            thresh = self.thresh
        if flex is None:
            flex = self.flex
        for i, j in locs:
            if i == j:
                continue
            res = err_trans_search(self.parent, (i,j), flex=flex, factor=factor, max_iter=max_iter, thresh=thresh)
            self._trans[i,j,:], self.trans_ll[i,j,:], self.t_rate_rng[i,j], self.t_ll_rng[i,j] = res
    
    def trans_space(self, trans, rng=None, steps=20):
        """
        Calculate the loglikelihoods of models with a range transition rate values 
        of a given transition. By default, takes a range twice as large as the 
        calculated error, and divides into 20 steps.

        Parameters
        ----------
        err : Loglik_Error
            Error object to produce parameter range.
        trans : [int, int]
            Transition (from_state, to_state) to evaluate
        rng : tuple[int, int], int, float, numpy.ndarray, optional
            Custom range specified by (low, high) transition rate values, or factor
            by which to multiply error range, or array of transition rate values. 
            The default is None.
        steps : int, optional
            Number of models to evaluate. The default is 20.
        
        Returns
        -------
        rng : numpy.ndarray
            1D array of the transition rate evaluated
        ll : numpy.ndarray
            1D array of the loglikelihoods of the resultant models
        
        """
        rng, ll = trans_space_ll(self, trans, rng=rng, steps=steps)
        t_rng = np.concatenate([self.t_rate_rng[trans], rng])
        t_ll_rng = np.concatenate([self.t_ll_rng[trans], ll])
        self.t_rate_rng[trans], self.t_ll_rng[trans] = _unique_by_first(t_rng, t_ll_rng)
        return rng, ll
        
    def E_eval(self, locs=None, flex=None, step=1e-2, max_iter=100, thresh=None):
        """
        Evaluate the loglike based error for FRET efficiency

        Parameters
        ----------
        locs : iterable[float]
            The states for which to evaluate the loglik error. If None, will
            calculate all states.
            The default is None.
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        step : float, optional
            Shift in E for initial search. The default is 1e-2.
        max_iter : int, optional
            maximum number of attempts to try in finding target transition rate.
            The default is 100.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.

        
        """
        if not self.parent._hasE:
            raise AttributeError("Parent BurstData object does not contain DexDem and DexAem streams")
        if locs is None:
            locs = range(self.nstate)
        if thresh is None:
            thresh = self.thresh
        if flex is None:
            flex = self.flex
        for i in locs:
            E, E_ll, E_rng, E_ll_rng =  \
                err_E_search(self, i,flex=flex, step=step, max_iter=max_iter, thresh=thresh)
            self._E[i,:], self.E_ll[i,:] = E, E_ll
            E_rng = np.concatenate([self.E_rng[i], E_rng])
            E_ll_rng = np.concatenate([self.E_ll_rng[i], E_ll_rng])
            sort = np.argsort(E_rng)
            self.E_rng[i] = E_rng[sort]
            self.E_ll_rng[i] = E_ll_rng[sort]
            
    def E_space(self, state, rng=None, steps=20):
        """
        Calculate the loglikelihoods of models with a range E values of state.
        By default, takes a range twice as large as the calculated error, and divides
        into 20 steps.

        Parameters
        ----------
        err : Loglik_Error
            Error object to produce parameter range.
        state : int
            State to evaluate.
        rng : tuple[int, int], int, float, numpy.ndarray, optional
            Custom range specified by (low, high) S vals, or factor by which to multiply
            error range, or array of S values. The default is None.
        steps : int, optional
            Number of models to evaluate. The default is 20.
            
        Returns
        -------
        rng : numpy.ndarray
            1D array of the transition rate evaluated
        ll : numpy.ndarray
            1D array of the loglikelihoods of the resultant models
        
        """
        if not self.parent._hasE:
            raise AttributeError("Parent BurstData object does not contain DexDem and DexAem streams")
        rng, ll = E_space(self, state, rng=rng, steps=steps)
        E_rng = np.concatenate([self.E_rng[state], rng])
        E_ll_rng = np.concatenate([self.E_ll_rng[state], ll])
        self.E_rng[state], self.E_ll_rng[state] = _unique_by_first(E_rng, E_ll_rng)
        return rng, ll
        
    def S_eval(self, locs=None, flex=None, step=1e-2, max_iter=100, thresh=None):
        """
        Evaluate the loglike based error for Stoichiometry

        Parameters
        ----------
        locs : iterable[float]
            The states for which to evaluate the loglik error. If None, will
            calculate all states.
            The default is None.
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        step : float, optional
            Shift in E for initial search. The default is 1e-2.
        max_iter : int, optional
            maximum number of attempts to try in finding target transition rate.
            The default is 100.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.

        
        """
        if not self.parent._hasE:
            raise AttributeError("Parent BurstData object does not contain DexDem and DexAem streams")
        if locs is None:
            locs = range(self.nstate)
        if thresh is None:
            thresh = self.thresh
        if flex is None:
            flex = self.flex
        for i in locs:
            S, S_ll, S_rng, S_ll_rng =  \
                err_S_search(self, i,flex=flex, step=step, max_iter=max_iter, thresh=thresh)
            self._S[i,:], self.S_ll[i,:] = S, S_ll
            S_rng = np.concatenate([self.S_rng[i], S_rng])
            S_ll_rng = np.concatenate([self.S_ll_rng[i], S_ll_rng])
            sort = np.argsort(S_rng)
            self.S_rng[i] = S_rng[sort]
            self.S_ll_rng[i] = S_ll_rng[sort]
    
    def S_space(self, state, rng=None, steps=20):
        """
        Calculate the loglikelihoods of models with a range S values of state.
        By default, takes a range twice as large as the calculated error, and divides
        into 20 steps.

        Parameters
        ----------
        err : Loglik_Error
            Error object to produce parameter range.
        state : int
            State to evaluate.
        rng : tuple[int, int], int, float, numpy.ndarray, optional
            Custom range specified by (low, high) S vals, or factor by which to multiply
            error range, or array of S values. The default is None.
        steps : int, optional
            Number of models to evaluate. The default is 20.
        
        Returns
        -------
        rng : numpy.ndarray
            1D array of the transition rate evaluated
        ll : numpy.ndarray
            1D array of the loglikelihoods of the resultant models
        
        """
        if not self.parent._hasE:
            raise AttributeError("Parent BurstData object does not contain DexDem and DexAem streams")
        rng, ll = S_space(self, state, rng=rng, steps=steps)
        S_rng = np.concatenate([self.S_rng[state], rng])
        S_ll_rng = np.concatenate([self.S_ll_rng[state], ll])
        self.S_rng[state], self.S_ll_rng[state] = _unique_by_first(S_rng, S_ll_rng)
        return rng, ll
        
    
    def all_eval(self, flex=None, thresh=None, trans_kw=None, E_kw=None, S_kw=None):
        """
        Evaluate loglik error for all available parameters

        Parameters
        ----------
        flex : float, optional
            Allowed variability in target decrease in loglikelihood. 
            If None use default.
            The default is None.
        thresh : float, optional
            Decrease in loglikelihood of adjusted model to target as characterizing
            the error. The default is None.

        trans_kw : dict, optional
            Keyword argument dictionary, passed to :meth:`burstH2MM.ModelError.Loglik_Error.get_trans_err`.
            The default is None.
        E_kw : dict, optional
            Keyword argument dictionary, passed to :meth:`burstH2MM.ModelError.Loglik_Error.get_E_err`.
            The default is None.
        S_kw : dict, optional
            Keyword argument dictionary, passed to :meth:`burstH2MM.ModelError.Loglik_Error.get_S_err`.
            The default is None.
        
        """
        if trans_kw is None:
            trans_kw = dict()
        if E_kw is None:
            E_kw = dict()
        if S_kw is None:
            S_kw = dict()
        
        if thresh is None:
            thresh = self.thresh
        if flex is None:
            flex = self.flex
        
        trans_k = dict(thresh=thresh, flex=flex)
        trans_k.update(trans_kw)
        E_k = dict(thresh=thresh, flex=flex)
        E_k.update(E_kw)
        S_k = dict(thresh=thresh, flex=flex)
        S_k.update(S_kw)
        
        self.get_trans_err(slice(None), slice(None), **trans_k)
        if self.parent._hasE:
            self.get_E_err(slice(None), **E_k)
        if self.parent._hasS:
            self.get_S_err(slice(None), **S_k)
    
    def covar_E(self, state, rng=None, steps=10, **kwargs):
        """
        Calculate the covariance of a given state along E.
        Runs optimizations with the given E parameter fixed to the value returned
        by :meth:`burstH2MM.ModelError.Loglik_Error.E_space`, and stores models 
        in cooresponding element of :attr:`burstH2MM.ModelError.Loglik_Error.E_covar` 
        
        Parameters
        ----------
        state : int 
            State along which to run optimizations with fixed values of E.
        rng : tuple[int, int], int, float, numpy.ndarray, optional
            Custom range specified by (low, high) E values, or factor by which to multiply
            error range, or array of E values. The default is None.
        steps : int, optional
            Number of models to evaluate. The default is 10.
        **kwargs : dict
            Arguments passed to ``H2MM_C.h2mm_model.optimize``.
        
        """
        locs = _getindexes((self.nstate, ), state)
        for loc in locs:
            loc = tuple(loc)
            self.E_covar[loc]  = covar_E(self, loc, rng=rng, steps=steps, **kwargs)
    
    def covar_S(self, state, rng=None, steps=10, **kwargs):
        """
        Calculate the covariance of a given state along S.
        Runs optimizations with the given S parameter fixed to the value returned
        by :meth:`burstH2MM.ModelError.Loglik_Error.S_space`, and stores models 
        in cooresponding element of :attr:`burstH2MM.ModelError.Loglik_Error.S_covar` 
        
        Parameters
        ----------
        state : int 
            State along which to run optimizations with fixed values of S.
        rng : tuple[int, int], int, float, numpy.ndarray, optional
            Custom range specified by (low, high) S values, or factor by which 
            to multiply error range, or array of S values. 
            The default is None.
        steps : int, optional
            Number of models to evaluate. The default is 10.
        **kwargs : dict
            Arguments passed to `H2MM_C.h2mm_model.optimize`.
        
        """
        locs = _getindexes((self.nstate, ), state)
        for loc in locs:
            loc = tuple(loc)
            self.S_covar[loc]  = covar_S(self, loc, rng=rng, steps=steps, **kwargs)
    
    def covar_trans(self, *args, rng=None, steps=10, **kwargs):
        """
        Calculate the covariance of a given transition rate.
        Runs optimizations with the given transition rate fixed to the value returned
        by :meth:`burstH2MM.ModelError.Loglik_Error.trans_space`, and stores models 
        in cooresponding element of :attr:`burstH2MM.ModelError.Loglik_Error.trans_covar` 
        
        Parameters
        ----------
        from_state : int
            State from which the system is transitioning.
        to_state : int 
            State to which the system is transitioning.
        rng : tuple[int, int], int, float, numpy.ndarray, optional
            Custom range specified by (low, high) transition rates, or factor by
            which to multiply error range, or array of transition rates. 
            The default is None.
        steps : int, optional
            Number of models to evaluate. The default is 10.
        **kwargs : dict
            Arguments passed to `H2MM_C.h2mm_model.optimize`.
            
            .. note::
                
                To reduce the time that these evaluations, if not over-ridden in
                keyword arguments, ``max_iter=400`` and ``converged_min=1e-6*self.flex``
        
        """
        locs = _getindexes((self.nstate, self.nstate), *args)
        for loc in locs:
            if loc[0] == loc[1]:
                continue
            loc = tuple(loc)
            self.trans_covar[loc] = covar_trans(self, loc, rng=rng, steps=steps, **kwargs)