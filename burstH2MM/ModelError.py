#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: 10/01/2023
# 
# Author: paul

"""
Error Evaluation
----------------

.. note::
    
    These classes are intended to be accessed primarily through the H2MM_list
    object and should rarely be interacted with directly by the user
"""

from collections.abc import Iterable
import numpy as np
from itertools import permutations

import fretbursts as frb
import H2MM_C as h2


def _getindexes(shape, *args):
    index = np.indices(shape)
    idx = index[(slice(None),)+args]
    idx = idx.swapaxes(0,-1)
    idx = idx.reshape(-1,index.shape[0])
    return idx

class Bootstrap_Error:
    """
    Internal class for storing bootstrap error calculations.
    
    This class is usually created when the :meth:`burstH2MM.BurstSort.H2MM_result.bootstrap_eval'
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
    prior : numpy.ndarray
        Standard deviation of each element in the initial probability matrix
    trans : numpy.ndarray
        Standard deviation of each element in the transition probability matrix
    obs : numpy.ndarray
        Standard deviation of each element in the emmision probability matrix
    E : numpy.ndarray
        Standard deviation in FRET efficiency per state
    S : numpy.ndarray
        Variance in stoichiometry per state
    
    """
    def __init__(self, parent, models, prior, trans, obs, *ES):
        #: parent :class:`burstH2MM.BurstSort.H2MM_result` object
        self.parent = parent
        #: list of optimized models of subsets
        self.models = models
        #: Standard deviation of model prior matricies
        self.prior = prior
        # Standard deviation of model transition probability matrices
        self.trans = trans
        #: Standard deviation of the emission probability matrix
        self.obs = obs
        for attr, arg in zip(['E', 'S'], ES):
            setattr(self, attr, arg)
        self.subsets = len(self.models)
    
    @classmethod
    def model_eval(cls, model, subsets=10):
        """
        Standard way of creating :class:`Bootstrap_Error` objects, takes a model
        and subdivides the data into `subsets` number of subsets, and runs
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
        models, prior, trans, obs, *ES = _bootstrap_eval(model, subsets=subsets)
        return cls(model, models, prior, trans, obs, *ES)
        
    
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
    mod_arr : list[H2MM_C.h2mm_model]
        The models optimized against subsets
    prior : numpy.ndarray
        Standard deviation of the obs values of the different subsets.
    trans : numpy.ndarray
        Standard deviation of the obs values of the different subsets.
    obs : numpy.ndarray
        Standard deviation of the obs values of the different subsets.
    E : numpy.ndarray
        Standard deviation of the E values of the different subsets.
    S : numpy.ndarray
        Standard deviation of the S values of the different subsets.

    """
    mod_arr = [model.model.optimize(model.parent.index[i::subsets], model.parent.parent.times[i::subsets], 
                                    inplace=False, **kwargs) for i in range(subsets)]
    prior = np.std([m.prior for m in mod_arr], axis=0)
    trans = np.std([m.trans for m in mod_arr], axis=0) / model.parent.parent.data.clk_p
    obs = np.std([m.obs for m in mod_arr], axis=0)
    ES = []
    if model._hasE:
        E = np.std([m.obs[:,model._DexAem_slc] / (m.obs[:,model._DexAem_slc] + m.obs[:,model._DexDem_slc]) for m in mod_arr], axis=0)
        ES.append(E)
        if model._hasS:
            Dex = np.array([m.obs[:,model._DexDem_slc] + m.obs[:,model._DexAem_slc] for m in mod_arr])
            DAex = Dex + np.array([m.obs[:,model._AexAem_slc] for m in mod_arr])
            S = np.std(Dex / DAex, axis=0)
            ES.append(S)
    return mod_arr, prior, trans, obs, *ES


def _trans_space(trans, lim, nstep):
    """
    Space number logrithmically arround trans, +/- log(lim) with nstep steps.

    Parameters
    ----------
    trans : float
        DESCRIPTION.
    lim : float
        Logarithmic .
    nstep : int
        number of elements in output array.

    Returns
    -------
    space : np.ndarray
        Array of numbers logarithmically spaced around trans.

    """
    trans = np.log(trans)/np.log(10)
    return np.logspace(trans-lim, trans+lim, nstep)


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
    DDsum = obs[states,DDslc].sum(axis=1)
    DAsum = obs[states,DAslc].sum(axis=1)
    tot = DDsum + DAsum
    obs[states,DAslc] = np.array([obs[s,DAslc] * t * v / a for s, t, v, a in zip(states, tot, vals, DAsum)])
    obs[states,DDslc] = np.array([obs[s,DDslc] * t * (1-v) / d for s, t, v, d in zip(states, tot, vals, DDsum)])
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
    Dsum = obs[states,DDslc].sum(axis=1) + obs[states,DAslc].sum(axis=1)
    AAsum = obs[states,AAslc].sum(axis=1)
    tot = Dsum + AAsum
    obs[states,AAslc] = np.array([obs[s,AAslc] * (1-v) * t / a for s, t, v, a in zip(states, tot, vals, AAsum)])
    obs[states,DDslc] = np.array([obs[s,DDslc] * v * t / d for s, t, v, d in zip(states, tot, vals, Dsum)])
    obs[states,DAslc] = np.array([obs[s,DAslc] * v * t / d for s, t, v, d in zip(states, tot, vals, Dsum)])
    return obs

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
    return ll_rt, ll_lh, rate_array, ll_array

def trans_space(err, state, rng=None, steps=20):
    """
    Calculate the loglikelihoods of models with a range trans values of state.
    By default, takes +/- 10 times the optimal transition rate.
    into 20 steps.

    Parameters
    ----------
    err : Loglik_Error
        Error object to use as input.
    state : int
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
    s0, s1 = state
    if np.any(getattr(err, '_trans').mask[state]) and isinstance(rng, (float, int, type(None))):
        raise ValueError("trans error not calculated, cannot auto-generate range")
    elif rng is None:
        factor = 0.5
    elif isinstance(rng, (int, float)):
        factor = rng
    if isinstance(rng, (int, float, type(None))):
        Tlow = np.log(err._trans[s0,s1,0])/np.log(10) - factor
        Thigh = np.log(err._trans[s0,s1,1])/np.log(10) + factor
    elif len(rng) == 2:
        Tlow, Thigh = rng
    else:
        rng_arr = rng
    if isinstance(rng, (int, float, type(None))) or len(rng) == 2:
        rng_arr = np.logspace(Tlow, Thigh, steps, endpoint=True)
    prior, mtrans, obs = err.model.prior, err.model.trans, err.model.obs
    trans = (_trans_adjust(mtrans, (t, ), (state, )) for t in rng_arr)
    models = [h2.h2mm_model(prior, t, obs) for t in trans]
    models = h2.H2MM_arr(models, err.parent.parent.index, err.parent.parent.parent.times)
    ll = np.array([m.loglik for m in models])
    return rng_arr, ll

### The following functions are currently unused, inteaded for future addition 
### to code allowing creating covariant loglik spaces


# def ll_array_trans(model, loc, lim=1, nstep=20):
#     trange = _trans_space(model.model.trans[loc], lim, nstep)
#     trans = [_trans_adjust(model.model.trans, np.array([t]), (loc,)) for t in trange]
#     prior = model.model.prior
#     obs = model.model.obs
#     index = model.parent.index
#     times = model.parent.parent.times
#     ll = [h2.h2mm_model(prior, tr, obs) for tr in trans]
#     ll = h2.H2MM_arr(ll, index, times)
#     ll = np.array([l.loglik for l in ll])
#     return trange, ll 


# def ll_error_trans(model, lim=1, nstep=20):
#     ll = [[None for j in range(model.nstate)] for i in range(model.nstate)]
#     t_array = [[None for j in range(model.nstate)] for i in range(model.nstate)]
#     for i, j in permutations(range(model.nstate), 2):
#         t_array[i][j], ll[i][j] = ll_array_trans(model, (i,j), lim=lim, nstep=nstep)
#     return t_array, ll

# def _ES_space(ES, lim, step):
#     low, high = ES - lim, ES + lim
#     if low < 0.0:
#         low, high = step, 2 * lim + step
#     elif high > 1.0:
#         low, high = 1 - 2*lim, 1.0
#     return np.arange(low, high, step)

# def ll_error_E(model, lim=0.1, step=0.001):
#     ll = [None for _ in range(model.nstate)]
#     prior = model.model.prior
#     trans = model.model.trans
#     index = model.parent.index
#     times = model.parent.parent.times
#     for i in range(model.nstate):
#         obs_step = _ES_space(model.E[i], lim, step)
#         obs_arr = [_E_adjust(model, np.array([o]), (i,)) for o in obs_step]
#         obs_arr = [h2.h2mm_model(prior, trans, o) for o in obs_arr]
#         obs_arr = h2.H2MM_arr(obs_arr, index, times)
#         ll[i] = (obs_step, np.array([mod.loglik for mod in obs_arr]))
#     return ll

# def ll_error_S(model, lim=0.1, step=0.001):
#     ll = [None for _ in range(model.nstate)]
#     prior = model.model.prior
#     trans = model.model.trans
#     index = model.parent.index
#     times = model.parent.parent.times
#     for i in range(model.nstate):
#         obs_step = _ES_space(model.S[i], lim, step)
#         obs_arr = [_S_adjust(model, np.array([o]), (i,)) for o in obs_step]
#         obs_arr = [h2.h2mm_model(prior, trans, o) for o in obs_arr]
#         obs_arr = h2.H2MM_arr(obs_arr, index, times)
#         ll[i] = (obs_step, np.array([mod.loglik for mod in obs_arr]))
#     return ll


# def ll_error_covar(model, locs, lim=1, nstep=20):
#     mod = model.model
#     tvar = (_trans_space(mod.trans[loc], lm, ns) for loc, lm, ns in zip(locs, lim, nstep))
#     grid = np.meshgrid(*tvar)
#     covar = np.meshgrid(*grid)
#     mod_arr = np.empty(covar[0].shape, dtype=object)
#     with np.nditer(covar, flags=['multi_index']) as it:
#         for cv in it:
#             mod_arr[it.multi_index] = h2.h2mm_model(mod.prior, _trans_adjust(mod.trans, cv, locs))
#     mod_arr = h2.H2MM_arr(mod_arr, model.parent.index, model.parent.parent.times)
#     ll_arr = np.empty(covar[0].shape, dtype=float)
#     with np.nditer(mod_arr, flags=['multi_index']) as it:
#         for ll in it:
#             ll_arr[it.multi_index] = ll
#     return covar, ll_arr

def err_E_search(model, state, flex=5e-2, step=1e-2, max_iter=100, thresh=0.5):
    """
    Base function for calculating the loglikelihood error of the E of a particular
    state.

    Parameters
    ----------
    model : H2MM_result
        Model for which to calculate the loglikelihood error.
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
        Actual loglikelihoods of low/high S values.
    rate_array : numpy.ndarray
        All rates attempted in search.
    ll_array : TYPE
        All loglikelihoods attempted in search.

    """
    # define two lambda functions for the high and low transition rate transformations
    base = model.E[state]
    downstep = lambda e: e - step if high == 0.0 else (high + low) / 2
    upstep = lambda e: e + step if high == 1.0 else (high + low) / 2
    m_iter = max_iter
    # pull out variables to make code less verbose
    prior, trans = model.model.prior, model.model.trans
    index, times = model.parent.index, model.parent.parent.times
    it = 0
    # set minimuma nd maximum values to accept as "converged"
    targl, targh = model.loglik - thresh - flex, model.loglik - thresh + flex
    # pre-allocate arrays, instead of using list
    E_array, ll_array = np.empty(10), np.empty(10)
    sz = ll_array.size
    ll_lh, ll_rt = np.empty(2), np.empty(2)
    # iterate first low values, then high values
    for i, fstep in enumerate((downstep, upstep)):
        low = base
        high = 0.0 if fstep is downstep else 1.0
        cont = True
        E = base
        while cont:
            Ec = fstep(E)
            if Ec > 0.0 and Ec < 1.0:
                obs = _E_adjust(model, (Ec,), (state,))
                mod = h2.h2mm_model(prior, trans, obs)
                llm = h2.H2MM_arr(mod, index, times)
                ll = llm.loglik
                E_array[it] = Ec
                ll_array[it] = ll
                E = Ec
                it += 1
            else:
                Ec = (E + Ec) / 2
                continue
            # cases of bellow/above target range
            if ll < targl:
                high = E
            elif ll > targh:
                low = E
            else:
                ll_lh[i] = ll
                ll_rt[i] = E
                cont = False
            if it >= max_iter:
                cont = False
            elif it >= sz:
                ll_array = np.concatenate([ll_array, np.empty(10)])
                E_array = np.concatenate([E_array, np.empty(10)])
                sz = ll_array.size
        max_iter = max_iter + m_iter
    # cut unused parts of array
    E_array, ll_array = E_array[:it], ll_array[:it]
    sort = np.argsort(E_array)
    E_array = E_array[sort]
    ll_array = ll_array[sort]
    return ll_rt, ll_lh, E_array, ll_array


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
    if isinstance(state, Iterable) and len(state) == 1:
        state = state[0]
    no_E = np.any(getattr(err, '_E').mask[state])
    need_E = isinstance(rng, (float, int, type(None)))
    if no_E and need_E:
        raise ValueError("E error not calculated, cannot auto-generate range")
    if rng is None:
        factor = 2
    elif isinstance(rng, (int, float)):
        factor = rng
    if isinstance(rng, (int, float, type(None))):
        Elow = factor*(err._E[state,0] - err.parent.E[state]) + err.parent.E[state]
        Ehigh = factor*(err._E[state,1] - err.parent.E[state]) + err.parent.E[state]
        if Elow < 0.0:
            Elow = 0.0
        if Ehigh > 1.0:
            Ehigh = 1.0
    elif len(rng) == 2:
        Elow, Ehigh = rng
    else:
        rng_arr = rng
    if isinstance(rng, (int, float, type(None))) or len(rng) == 2:
        rng_arr = np.linspace(Elow, Ehigh, steps, endpoint=True)
    # check for values that already exist
    mask = np.ones(rng_arr.shape, dtype=bool)
    ll_arr = np.empty(rng_arr.shape)
    if hasattr(err, 'E_rng'):
        _, keep, drop = np.intersect1d(err.E_rng[state], rng_arr, return_indices=True)
        mask[drop] = False
        fill_arr = err.E_ll_rng[state][keep]
        rng_arr_m = rng_arr[mask]
    else:
        rng_arr_m = rng_arr
        fill_arr = np.zeros(0)
    # perform calculation
    prior, trans = err.model.prior, err.model.trans
    obs = (_E_adjust(err.parent, (E, ), (state, )) for E in rng_arr_m)
    models = [h2.h2mm_model(prior, trans, ob) for ob in obs]
    models = h2.H2MM_arr(models, err.parent.parent.index, err.parent.parent.parent.times)
    # infill already calculated numbers
    ll_arr[mask] = np.array([mod.loglik for mod in models])
    ll_arr[~mask] = fill_arr
    return rng_arr, ll_arr


def err_S_search(model, state, flex=5e-2, step=1e-2, max_iter=100, thresh=0.5):
    """
    Base function for calculating the loglikelihood error of the S of a particular
    state.

    Parameters
    ----------
    model : H2MM_result
        Model for which to calculate the loglikelihood error.
    state : int
        state to evaluate loglikelihood error of S.
    flex : float, optional
        Allowed variability in target decrease in loglikelihood. The default is 5e-2.
    step : TYPE, optional
        DESCRIPTION. The default is 1e-2.
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
    # define two lambda functions for the high and low transition rate transformations
    base = model.S[state]
    downstep = lambda s: s - step if high == 0.0 else (high + low) / 2
    upstep = lambda s: s + step if high == 1.0 else (high + low) / 2
    m_iter = max_iter
    # pull out variables to make code less verbose
    prior, trans = model.model.prior, model.model.trans
    index, times = model.parent.index, model.parent.parent.times
    it = 0
    # set minimuma nd maximum values to accept as "converged"
    targl, targh = model.loglik - thresh - flex, model.loglik - thresh + flex
    # pre-allocate arrays, instead of using list
    S_array, ll_array = np.empty(10), np.empty(10)
    sz = ll_array.size
    ll_lh, ll_rt = np.empty(2), np.empty(2)
    # iterate first low values, then high values
    for i, fstep in enumerate((downstep, upstep)):
        low = base
        high = 0.0 if fstep is downstep else 1.0
        cont = True
        S = base
        while cont:
            Sc = fstep(S)
            if Sc > 0.0 and Sc < 1.0:
                obs = _S_adjust(model, (Sc,), (state,))
                mod = h2.h2mm_model(prior, trans, obs)
                llm = h2.H2MM_arr(mod, index, times)
                ll = llm.loglik
                S_array[it] = Sc
                ll_array[it] = ll
                S = Sc
                it += 1
            else:
                Sc = (S + Sc) / 2
                continue
            # cases of bellow/above target range
            if ll < targl:
                high = S
            elif ll > targh:
                low = S
            else:
                ll_lh[i] = ll
                ll_rt[i] = S
                cont = False
            if it >= max_iter:
                cont = False
            elif it >= sz:
                ll_array = np.concatenate([ll_array, np.empty(10)])
                S_array = np.concatenate([S_array, np.empty(10)])
                sz = ll_array.size
        max_iter = max_iter + m_iter
    # cut unused parts of array
    S_array, ll_array = S_array[:it], ll_array[:it]
    sort = np.argsort(S_array)
    S_array = S_array[sort]
    ll_array = ll_array[sort]
    return ll_rt, ll_lh, S_array, ll_array


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
    if isinstance(state, Iterable) and len(state) == 1:
        state = state[0]
    no_S = np.any(getattr(err, '_S').mask[state])
    need_S = isinstance(rng, (float, int, type(None)))
    if no_S and need_S:
        raise ValueError("S error not calculated, cannot auto-generate range")
    if rng is None:
        factor = 2
    elif isinstance(rng, (int, float)):
        factor = rng
    if isinstance(rng, (int, float, type(None))):
        Slow = factor*(err._S[state,0] - err.parent.S[state]) + err.parent.S[state]
        Shigh = factor*(err._S[state,1] - err.parent.S[state]) + err.parent.S[state]
        if Slow < 0.0:
            Slow = 0.0
        if Shigh > 1.0:
            Shigh = 1.0
    elif len(rng) == 2:
        Slow, Shigh = rng
    else:
        rng_arr = rng
    if isinstance(rng, (int, float, type(None))) or len(rng) == 2:
        rng_arr = np.linspace(Slow, Shigh, steps, endpoint=True)
    # check for values that already exist
    mask = np.ones(rng_arr.shape, dtype=bool)
    ll_arr = np.empty(rng_arr.shape)
    if hasattr(err, 'S_rng'):
        _, keep, drop = np.intersect1d(err.S_rng[state], rng_arr, return_indices=True)
        mask[drop] = False
        fill_arr = err.S_ll_rng[state][keep]
        rng_arr_m = rng_arr[mask]
    else:
        rng_arr_m = rng_arr
        fill_arr = np.zeros(0)
    # perform calculation
    prior, trans = err.model.prior, err.model.trans
    obs = (_S_adjust(err.parent, (S, ), (state, )) for S in rng_arr_m)
    models = [h2.h2mm_model(prior, trans, ob) for ob in obs]
    models = h2.H2MM_arr(models, err.parent.parent.index, err.parent.parent.parent.times)
    # infill already calculated numbers
    ll_arr[mask] = np.array([mod.loglik for mod in models])
    ll_arr[~mask] = fill_arr
    return rng_arr, ll_arr


class Loglik_Error:
    """
    Object for evaluation of error by comparing loglikelihoods of models with
    parameters offset and the optimial model.
    
    """
    #: default decrease in loglikelihood to consider as the bounds of error
    _thresh = 0.5
    #: precision of loglikelihood needed to consider error bound to be found
    _flex = 5e-2
    
    def __init__(self, parent):
        self.parent = parent
        #: all transition rates evalated so far, oranized as numpy.ndarray of numpy.ndarrays
        self.t_rate_rng = np.empty((self.nstate, self.nstate), dtype=object)
        #: loglik of all transition rates evalated so far, oranized as numpy.ndarray of numpy.ndarrays
        self.t_ll_rng = np.empty((self.nstate, self.nstate), dtype=object)
        self._trans = np.ma.empty((self.nstate, self.nstate, 2))
        self._trans[...] = np.ma.masked
        self.trans_ll = np.ma.empty((self.nstate, self.nstate, 2))
        self.trans_ll[...] = np.ma.masked
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
            self.E_ll =  np.ma.empty((self.nstate,2))
            self.E_ll[...] = np.ma.masked
            for i in range(self.nstate):
                self.E_rng[i] = np.empty((0,))
                self.E_ll_rng[i] = np.empty((0,))
        if parent._hasS:
            #: all stoichiometries evalated so far, oranized as numpy.ndarray of numpy.ndarrays
            self.S_rng = np.empty((self.nstate), dtype=object)
            #: loglikelihoods of all stoichiometries evalated so far, oranized as numpy.ndarray of numpy.ndarrays
            self.S_ll_rng = np.empty((self.nstate), dtype=object)
            self._S = np.ma.empty((self.nstate,2)) 
            self._S[...] = np.ma.masked
            self.S_ll =  np.ma.empty((self.nstate,2))
            self.S_ll[...] = np.ma.masked
            for i in range(self.nstate):
                self.S_rng[i] = np.empty((0,))
                self.S_ll_rng[i] = np.empty((0,))
    

        # for use later in code, when covariate is fully implemented
        # self._locstup = self._gen_locs_list
    
    ### not used yet
    # def _gen_locs_list(self):
    #     streams = self.parent.parent.parent.ph_streams
    #     loc_enum = list()
    #     if frb.Ph_sel(Dex='Dem') in streams and frb.Ph_sel(Dex='Aem') in streams:
    #         loc_enum += [('E', i) for i in range(self.parent.nstate)]
    #     if frb.Ph_sel(Aex='Aem') in streams:
    #         loc_enum += [('S', i) for i in range(self.parent.nstate)]
    #     loc_enum += [('t', i, j) for i, j in permutations(range(self.parent.nstate), 2)]
    #     return tuple(loc_enum)
    
    @property
    def model(self):
        """Parent :class:`burstH2MM.BurstSort.H2MM_result` object"""
        return self.parent.model
    
    @property
    def nstate(self):
        """Nubmer of states in model"""
        return self.parent.model.nstate
    
    @property
    def trans(self):
        """[nstate, nstate, 2] matrix of low/high error transition rates"""
        return self._trans / self.parent.parent.parent.data.clk_p
    
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
        return self._E
    
    def S_lh(self):
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
        return self._trans[args+(slice(None),)]
    
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
            thresh = self._thresh
        if flex is None:
            flex = self._flex
        for i, j in locs:
            if i == j:
                continue
            res = err_trans_search(self.parent, (i,j), flex=flex, factor=factor, max_iter=max_iter, thresh=thresh)
            self._trans[i,j,:], self.trans_ll[i,j,:], self.t_rate_rng[i,j], self.t_ll_rng[i,j] = res
    
    def trans_space(self, state, rng=None, steps=20):
        rng, ll = trans_space(self, state, rng=rng, steps=steps)
        t_rng = np.concatenate([self.t_rate_rng[state], rng])
        t_ll_rng = np.concatenate([self.t_ll_rng[state], ll])
        sort = np.argsort(t_rng)
        self.t_rate_rng[state] = t_rng[sort]
        self.t_ll_rng[state] = t_ll_rng[sort]
        
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
            thresh = self._thresh
        if flex is None:
            flex = self._flex
        for i in locs:
            E, E_ll, E_rng, E_ll_rng =  \
                err_E_search(self.parent, i,flex=flex, step=step, max_iter=max_iter, thresh=thresh)
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
        """
        if not self.parent._hasE:
            raise AttributeError("Parent BurstData object does not contain DexDem and DexAem streams")
        rng, ll = E_space(self, state, rng=rng, steps=steps)
        E_rng = np.concatenate([self.E_rng[state], rng])
        E_ll_rng = np.concatenate([self.E_ll_rng[state], ll])
        _, sort = np.unique(E_rng, return_index=True)
        self.E_rng[state] = E_rng[sort]
        self.E_ll_rng[state] = E_ll_rng[sort]
        
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
            thresh = self._thresh
        if flex is None:
            flex = self._flex
        for i in locs:
            S, S_ll, S_rng, S_ll_rng =  \
                err_S_search(self.parent, i,flex=flex, step=step, max_iter=max_iter, thresh=thresh)
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
        """
        if not self.parent._hasE:
            raise AttributeError("Parent BurstData object does not contain DexDem and DexAem streams")
        rng, ll = S_space(self, state, rng=rng, steps=steps)
        S_rng = np.concatenate([self.S_rng[state], rng])
        S_ll_rng = np.concatenate([self.S_ll_rng[state], ll])
        _, sort = np.unique(S_rng, return_index=True)
        self.S_rng[state] = S_rng[sort]
        self.S_ll_rng[state] = S_ll_rng[sort]
        
    
    def univariate_eval(self, flex=None, thresh=None, trans_kw=None, E_kw=None, S_kw=None):
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
            Keyword argument dictionary, passed to :meth:`Loglik_Error.trans_eval`.
            The default is None.
        E_kw : dict, optional
            Keyword argument dictionary, passed to :meth:`Loglik_Error.E_eval`.
            The default is None.
        S_kw : dict, optional
            Keyword argument dictionary, passed to :meth:`Loglik_Error.S_eval`.
            The default is None.

        Returns
        -------
        None.

        """
        if trans_kw is None:
            trans_kw = dict()
        if E_kw is None:
            E_kw = dict()
        if S_kw is None:
            S_kw = dict()
        
        if thresh is None:
            thresh = self._thresh
        if flex is None:
            flex = self._flex
        
        trans_k = dict(thresh=thresh, flex=flex)
        trans_k.update(trans_kw)
        E_k = dict(thresh=thresh, flex=flex)
        E_k.update(E_kw)
        S_k = dict(thresh=thresh, flex=flex)
        S_k.update(S_kw)
        
        self.trans_eval(**trans_k)
        if self.parent._hasE:
            self.E_eval(**E_k)
            if self.parent._hasS:
                self.S_eval(**S_k)
    
    
    ### not implemented yet
    # def _covar_eval(locs, lim=None, step=None):
    #     if lim is None:
    #         lim = [dict() for _ in locs]
    #     elif isinstance(lim, (int, float)):
    #         lim = [lim for _ in locs]
    #     if step is None:
    #         step = [dict() for _ in locs]
    #     elif isinstance(step, (float, int)):
    #         step = [step for _ in locs]
    #     if len(locs) != len(lim) != len(step):
    #         raise ValueError("locs, lim and step must be the same length")
        