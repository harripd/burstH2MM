import pytest
# from urllib.request import urlretrieve

from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt

import fretbursts as frb
import burstH2MM as bhm
import H2MM_C as h2


def new_plot(*args, **kwargs):
    try:
        plt.close('all')
        fig, ax = plt.subplots(*args, **kwargs)
    except:
        fig, ax = plt.gcf(), plt.gca()
    return fig, ax


def all_less(a, m):
    if a is None:
        return True
    elif np.issubdtype(type(a), np.number):
        return a < m
    else:
        return max(a) < m


def nsalex_data():
    data = frb.loader.photon_hdf5("HP3_TE300_SPC630.hdf5")
    frb.loader.alex_apply_period(data)
    data.calc_bg(fun=frb.bg.exp_fit, time_s=30, tail_min_us='auto', F_bg=1.7)
    data.burst_search(m=10, F=6, ph_sel=frb.Ph_sel(Dex='DAem', Aex='Aem'))
    data_da = data.select_bursts(frb.select_bursts.size, th1=30)
    data_da.leakage = 0.01
    data_da.dir_ex = 0.01
    data_da.gamma = 0.9
    data_da.beta = 1.1
    return data_da


def usalex_data():
    data = frb.loader.photon_hdf5("HairPin3_RT_400mM_NaCl_A_31_TA.hdf5")
    frb.loader.alex_apply_period(data)
    data.calc_bg(fun=frb.bg.exp_fit, time_s=30, tail_min_us='auto', F_bg=1.7)
    data.burst_search(m=10, F=6, ph_sel=frb.Ph_sel(Dex='DAem', Aex='Aem'))
    data_da = data.select_bursts(frb.select_bursts.size, th1=100)
    return data_da


@pytest.fixture(scope='module', params=[nsalex_data, usalex_data])
def alex_data(request):
    d = request.param()
    return d


### Tests for BurstData, before optimizations are conducted

def test_raises(alex_data):
    """Test that H2MM_list does not return values for H2MM_results when ideal model not set"""
    bdata = bhm.BurstData(alex_data)
    assert bdata.models.num_opt == 0
    with pytest.raises(ValueError):
        bdata.models.E
        bdata.models.E_corr
        bdata.models.S
        bdata.models.S_corr
        bdata.models.trans_locs
        bdata.models.dwell_ph_counts
        bdata.models.dwell_ph_counts_bg
        bdata.models.dwell_E
        bdata.models.dwell_E_corr
        bdata.models.dwell_S
        bdata.models.dwell_S_corr
        bdata.models.burst_dwell_num
        bdata.models.nanohist
        bdata.models.dwell_nano_mean
        

def test_optimization(alex_data):
    """Smoke test for optimization"""
    bdata = bhm.BurstData(alex_data)
    model = h2.factory_h2mm_model(2,3)
    bdata.models.optimize(model)
    assert bdata.models.ICL.size > 0


def test_sph2mm(alex_data):
    """Smoke test for single parameter optimization"""
    bdata = bhm.BurstData(alex_data, ph_streams=[frb.Ph_sel(Dex='Dem'), frb.Ph_sel(Dex='Aem')])
    model = h2.factory_h2mm_model(2,2)
    bdata.models.optimize(model)    
    assert bdata.models.ICL.size > 0

def test_shift(alex_data):
    """smoke test for shift methods"""
    if not alex_data.lifetime:
        model = h2.factory_h2mm_model(2,3)
        bdata = bhm.BurstData(alex_data, Aex_shift='rand')
        bdata.models.optimize(model)
        bdata = bhm.BurstData(alex_data, Aex_shift='shift')
        bdata.models.optimize(model)
        bdata = bhm.BurstData(alex_data, Aex_shift=False)
        bdata.models.optimize(model)
    else:
        for shift in ('even', 'rand', 'shift', True):
            with pytest.raises(NotImplementedError):
                bdata = bhm.BurstData(alex_data, Aex_shift=shift)    



### Fixtures computed only after BurstData has been verified

@pytest.fixture(scope='module')
def alex_hmm(alex_data):
    bdata = bhm.BurstData(alex_data)
    bdata.models.calc_models()
    if alex_data.lifetime:
        bdata.irf_thresh = np.array([2355, 2305, 220])
        bdata.auto_div(2,name='twodiv')
        bdata.div_models['twodiv'].calc_models()
    return bdata




### Tests that optimizations etc. have worked

def test_bg_correction(alex_hmm):
    """Test calcualtion of bg correction"""
    if alex_hmm.data.lifetime: # only applies to non-usALEX data, since time shift changes burst duration
        assert np.allclose(alex_hmm.models[0].dwell_E_corr, np.concatenate(alex_hmm.data.E))
        assert np.allclose(alex_hmm.models[0].dwell_S_corr, np.concatenate(alex_hmm.data.S))
        assert np.allclose(alex_hmm.div_models['twodiv'][0].dwell_E_corr, np.concatenate(alex_hmm.data.E))
        assert np.allclose(alex_hmm.div_models['twodiv'][0].dwell_S_corr, np.concatenate(alex_hmm.data.S))



def test_ideal(alex_hmm):
    alex_hmm.models.ideal = 2
    assert hasattr(alex_hmm.models, "E")
    assert hasattr(alex_hmm.models, "E_corr")
    assert hasattr(alex_hmm.models, "S")
    assert hasattr(alex_hmm.models, "S_corr")
    assert hasattr(alex_hmm.models, "trans_locs")
    assert hasattr(alex_hmm.models, "dwell_E")
    assert hasattr(alex_hmm.models, "dwell_E_corr")
    assert hasattr(alex_hmm.models, "dwell_S")
    assert hasattr(alex_hmm.models, "dwell_S_corr")
    if alex_hmm.data.lifetime:
        assert hasattr(alex_hmm.models, 'dwell_nano_mean')
        assert hasattr(alex_hmm.models, 'nanohist')
        assert hasattr(alex_hmm.models, 'state_nano_mean')
        assert hasattr(alex_hmm.models, 'state_nano_mean_std')
        assert hasattr(alex_hmm.models, 'state_nano_mean_err')
    else:
        with pytest.raises(AttributeError):
            alex_hmm.models.dwell_nano_mean
        with pytest.raises(AttributeError):
            alex_hmm.models.nanohist
        with pytest.raises(AttributeError):
            alex_hmm.models.state_nano_mean
        with pytest.raises(AttributeError):
            alex_hmm.models.err_nano_mean_std
        with pytest.raises(AttributeError):
            alex_hmm.models.state_nano_mean_err


def test_nanowarnings():
    data = nsalex_data()
    frb.loader.alex_apply_period(data)
    data.calc_bg(fun=frb.bg.exp_fit, time_s=30, tail_min_us='auto', F_bg=1.7)
    data.burst_search(m=10, F=6, ph_sel=frb.Ph_sel(Dex='DAem', Aex='Aem'))
    data_da = data.select_bursts(frb.select_bursts.size, th1=100)
    bdata = bhm.BurstData(data_da)
    bdata.models.calc_models(to_state=2, max_state=3)
    with pytest.warns(UserWarning):
        bdata.models[1].dwell_nano_mean
    bdata.auto_div(2,name='twodiv')
    bdata.div_models['twodiv'].calc_models()
    with pytest.warns(UserWarning):
        bdata.div_models['twodiv'][1].state_nano_mean
            
def test_nanohist(alex_hmm):
    if alex_hmm.data.lifetime:
        phot = alex_hmm.models[1].nanohist.sum()
        divphot = alex_hmm.div_models['twodiv'][1].nanohist.sum()
        assert divphot == phot == sum(times.shape[0] for times in alex_hmm.times)

def test_gamma(alex_hmm):
    if alex_hmm.data.lifetime:
        alex_hmm.models[1].conf_thresh = None
        rphot = sum(times.shape[0] for times in alex_hmm.times)
        alex_hmm.models[1].gamma_thresh = 0.95
        phot = alex_hmm.models[1].nanohist.sum()
        assert rphot > phot
        alex_hmm.div_models['twodiv'][1].gamma_thresh = 0.95
        dphot = alex_hmm.div_models['twodiv'][1].nanohist.sum()
        assert rphot > dphot
        

def test_conf_thresh(alex_hmm):
    if alex_hmm.data.lifetime:
        alex_hmm.models[1].gamma_thresh = None
        rphot = sum(times.shape[0] for times in alex_hmm.times)
        alex_hmm.models[1].conf_thresh = 0.95
        phot = alex_hmm.models[1].nanohist.sum()
        assert rphot > phot
        alex_hmm.div_models['twodiv'][1].conf_thresh = 0.95
        dphot = alex_hmm.div_models['twodiv'][1].nanohist.sum()
        assert rphot > dphot

###
# ModelError testing
###

def test_ModelSet(alex_hmm):
    models = [h2.factory_h2mm_model(2,alex_hmm.models.ndet) for _ in range(4)]
    modset = bhm.ModelError.ModelSet(alex_hmm.models, models)
    assert len(modset) == 4
    for param in ('E', 'S', 'E_corr', 'S_corr'):
        parammat = getattr(modset, param)
        assert np.all(parammat == parammat[0,...])
    if alex_hmm.data.lifetime:
        models = [h2.factory_h2mm_model(2,alex_hmm.div_models['twodiv'].ndet) for _ in range(4)]
        modset = bhm.ModelError.ModelSet(alex_hmm.div_models['twodiv'], models)
        assert len(modset) == 4
        for param in ('E', 'S', 'E_corr', 'S_corr'):
            parammat = getattr(modset, param)
            assert np.all(parammat == parammat[0,...])
        

def test_bootstrap(alex_hmm):
    alex_hmm.models[1].bootstrap_eval()
    hasattr(alex_hmm.models[1], 'E_std_bs')
    hasattr(alex_hmm.models[1], 'S_std_bs')
    hasattr(alex_hmm.models[1], 'trans_std_bs')


def test_ll_error(alex_hmm):
    assert np.all(alex_hmm.models[1].E_err_ll.mask), "Unset elements not masked"
    assert np.all(alex_hmm.models[1].S_err_ll.mask), "Unset elements not masked"
    assert np.all(alex_hmm.models[1].loglik_err.trans.mask), "Unset elements not masked"
    alex_hmm.models[1].loglik_err.get_E_err(0)
    assert np.sum(alex_hmm.models[1].E_err_ll.mask) == 1, "Unset elements not masked"
    assert (alex_hmm.models[1].loglik_err.get_S_err(slice(None))).size == alex_hmm.models[1].S.size, "Bad return value"
    assert not np.any(alex_hmm.models[1].S_err_ll.mask), "Unset elements not masked"
    alex_hmm.models[1].loglik_err.get_trans_err(0,1)
    # check that only 1 value in mask is False (array is 4 elements, 4-1=3)
    assert np.sum(alex_hmm.models[1].trans_err_high_ll.mask) == 3, "Unset elements not masked" 
    alex_hmm.models[1].loglik_err.get_trans_err(1,slice(None))
    alex_hmm.models[1].loglik_err.get_trans_err(slice(None), slice(None))


def test_ll_covar_E(alex_hmm):
    ESarray = np.array([0.1,0.11,0.12,0.13,0.14])
    alex_hmm.models[1].loglik_err.get_E_err(0)
    alex_hmm.models[1].loglik_err.get_E_err(1)
    alex_hmm.models[1].loglik_err.covar_E(1, converged_min=1e-10)
    alex_hmm.models[1].loglik_err.covar_E(0, rng=(0.1,0.4), steps=4)
    assert len(alex_hmm.models[1].loglik_err.E_covar[0]) == 4
    alex_hmm.models[1].loglik_err.covar_E(0, rng=ESarray)
    assert len(alex_hmm.models[1].loglik_err.E_covar[0]) == ESarray.size
    assert np.allclose(alex_hmm.models[1].loglik_err.E_covar[0].E[:,0], ESarray)
    
    
def test_ll_covar_S(alex_hmm):
    ESarray = np.array([0.1,0.11,0.12,0.13,0.14])
    alex_hmm.models[1].loglik_err.get_S_err(0)
    alex_hmm.models[1].loglik_err.get_S_err(1)
    alex_hmm.models[1].loglik_err.covar_S(1, converged_min=1e-10, max_iter=10)
    alex_hmm.models[1].loglik_err.covar_S(0, rng=(0.1,0.4), steps=4, max_iter=10)
    assert len(alex_hmm.models[1].loglik_err.S_covar[0]) == 4
    alex_hmm.models[1].loglik_err.covar_S(0, rng=ESarray, max_iter=10)
    assert len(alex_hmm.models[1].loglik_err.S_covar[0]) == ESarray.size
    assert np.allclose(alex_hmm.models[1].loglik_err.S_covar[0].S[:,0], ESarray)
    
    
def teest_ll_covar_trans(alex_hmm):
    trans_array = np.array([100,120,140,160,180])
    alex_hmm.models[1].loglik_err.get_trans_err(0,1)
    alex_hmm.models[1].loglik_err.get_trans_err(1,0)
    alex_hmm.models[1].loglik_err.covar_trans(0,1, converged_min=1e-10, max_iter=10)
    alex_hmm.models[1].loglik_err.covar_trans(0,1, rng=(100,200), steps=4, converged_min=1e-10, max_iter=10)
    assert len(alex_hmm.models[1].loglik_err.trans_covar[1,0]) == 4
    alex_hmm.models[1].loglik_err.covar_trans(1,0, rng=trans_array, max_iter=10)
    assert len(alex_hmm.models[1].loglik_err.trans_covar[1,0]) == trans_array.size
    assert np.all(alex_hmm.models[1].loglik_err.trans_covar[1,0].trans[:,1,0] == trans_array)
    
    
    
### Smoke tests of burst plotting functions

@pytest.mark.parametrize("func", (bhm.ICL_plot, bhm.BIC_plot, bhm.BICp_plot, bhm.path_BIC_plot))
def test_stat_disc_plot(alex_hmm, func):
    func(alex_hmm.models)
    fig, ax = new_plot()
    func(alex_hmm.models, ax=ax, highlight_ideal=True)
    plt.close('all')


@pytest.mark.parametrize("func", (bhm.dwell_E_hist, bhm.dwell_S_hist, bhm.dwell_tau_hist))
def test_dwell_hist(alex_hmm, func):
    # skip test for tau on usALEX data
    if func is not bhm.dwell_tau_hist or alex_hmm.data.lifetime:
        func(alex_hmm.models)
        alex_hmm.models.ideal = 2
        fig, ax = new_plot()
        func(alex_hmm.models, ax=ax, bins=np.arange(0,1,0.2), states=np.array([1,0]))
        fig, ax = new_plot()
        func(alex_hmm.models, ax=ax, bins=np.arange(0,1,0.2))
        if func is bhm.dwell_tau_hist:
            fig, ax = new_plot()
            func(alex_hmm.models, ax=ax, bins=np.arange(0,1,0.2), 
                 streams=[frb.Ph_sel(Dex='Aem'), frb.Ph_sel(Aex='Aem')])
    plt.close('all')


@pytest.mark.parametrize('func', (bhm.dwell_ES_scatter, bhm.dwell_E_tau_scatter))
def test_dwell_scatter(alex_hmm, func):
    if alex_hmm.data.lifetime or func is not bhm.dwell_E_tau_scatter:
        func(alex_hmm.models[2])
        alex_hmm.models.ideal = 2
        fig, ax = new_plot()
        func(alex_hmm.models, ax=ax, s=10, states=np.array([1,0]))
        fig, ax = new_plot()
        func(alex_hmm.models, ax=ax, plot_type='kde', add_corrections=True)
        if alex_hmm.data.lifetime:
            func(alex_hmm.div_models['twodiv'][2])
            alex_hmm.div_models['twodiv'].ideal = 2
            fig, ax = new_plot()
            func(alex_hmm.div_models['twodiv'], ax=ax, s=10, states=np.array([1,0]))
            fig, ax = plt.subplots()
            func(alex_hmm.div_models['twodiv'], ax=ax, plot_type='kde', add_corrections=True)
    else:
        with pytest.raises(Exception):
            func(alex_hmm.models[2])
    plt.close('all')


def test_burst_ES_scatter(alex_hmm):
    fig, ax = new_plot(figsize=(6,6))
    bhm.burst_ES_scatter(alex_hmm.models[2], ax=ax)
    fig, ax = new_plot(figsize=(6,6))
    bhm.burst_ES_scatter(alex_hmm.models[2], flatten_dynamics=True, ax=ax)
    plt.close('all')
    
def test_arrow(alex_hmm):
    bhm.trans_arrow_ES(alex_hmm.models[2])
    alex_hmm.ideal = 2
    fig, ax = new_plot(figsize=(6,6))
    bhm.dwell_ES_scatter(alex_hmm.models, ax=ax)
    bhm.trans_arrow_ES(alex_hmm.models, states=((0,1),(1,2)), fstring='3.1f')
    plt.close('all')
    fig, ax = new_plot(figsize=(6,6))
    bhm.dwell_ES_scatter(alex_hmm.models, ax=ax)
    bhm.trans_arrow_ES(alex_hmm.models, positions=np.array([[0.5, 0.2, 0.4],[0.1, 0.5, 0.6],[0.7,0.5,0.6]]))
    plt.close('all')

def test_scatter_ES(alex_hmm):
    bhm.scatter_ES(alex_hmm.models[3])
    alex_hmm.ideal = 3
    fig, ax = new_plot(figsize=(6,6))
    bhm.dwell_ES_scatter(alex_hmm.models, ax=ax)
    bhm.scatter_ES(alex_hmm.models, states=np.array([1,2]))
    plt.close('all')
    

@pytest.mark.parametrize('from_state, to_state', [(a, b) for a, b in 
                                                  permutations((None, 0,1, 5, [1],[1,2], np.array([2]),
                                                                np.array([2,1]), 5, [2,5]),2)])
def test_dwell_trans_durs(alex_hmm, from_state, to_state):
    if all_less(from_state, 3) and all_less(to_state, 3):        
        bhm.dwell_trans_dur_hist(alex_hmm.models[2], from_state=from_state, to_state=to_state)
        fig, ax = new_plot()
        bhm.dwell_trans_dur_hist(alex_hmm.models[2], from_state=from_state, to_state=to_state, ax=ax)
        
    else:
        with pytest.raises(Exception):
            bhm.dwell_trans_dur_hist(alex_hmm.models[2], from_state=from_state, to_state=to_state)
    plt.close('all')



@pytest.mark.parametrize('func', (bhm.mid_dwell, bhm.end_dwell, bhm.begin_dwell, bhm.burst_dwell,
                                  bhm.edge_dwell, bhm.not_mid_dwell, bhm.burst_begin_dwell,
                                  bhm.burst_end_dwell, 0, 5, [0, 1], np.array([0, 1]),
                                  [0,5], np.array([0,5])))
def test_dwell_pos(alex_hmm, func):
    if callable(func) or all_less(func, 3):
        bhm.dwell_dur_hist(alex_hmm.models[2], dwell_pos=func)
    else:
        with pytest.raises(Exception):
            bhm.dwell_dur_hist(alex_hmm.models[2], dwell_pos=func)
    plt.close('all')

@pytest.mark.parametrize('func', (bhm.dwell_trans, bhm.dwell_trans_from))
def test_dwell_trans(alex_hmm, func):
    bhm.dwell_dur_hist(alex_hmm.models[2], dwell_pos=lambda model: func(model, 1), states=[0])
    bhm.dwell_dur_hist(alex_hmm.models[2], dwell_pos=lambda model: func(model, 1, True), states=[0])
    bhm.dwell_dur_hist(alex_hmm.models[2], dwell_pos=lambda model: func(model, 1, False), states=[0])
    with pytest.raises(ValueError):
        bhm.dwell_dur_hist(alex_hmm.models[2], dwell_pos=lambda model: func(model, 3), states=[0])
    plt.close('all')

def test_dwell_size(alex_hmm):
    bhm.dwell_E_hist(alex_hmm.models[2], dwell_pos=lambda model: bhm.dwell_size(model, 5))
    bhm.dwell_E_hist(alex_hmm.models[2], dwell_pos=lambda model: bhm.dwell_size(model, 5, ph_max=100))
    bhm.dwell_E_hist(alex_hmm.models[2], dwell_pos=lambda model: bhm.dwell_size(model, 5, streams=[frb.Ph_sel(Dex='Dem')]))
    plt.close('all')

@pytest.mark.parametrize('func', (bhm.axline_E, bhm.axline_S))
def test_axlines(alex_hmm, func):
    fig, ax = new_plot()
    bhm.dwell_E_hist(alex_hmm.models[2], ax=ax)
    func(alex_hmm.models[2], ax=ax, states=np.array([2,1]), state_kwargs=[{'alpha':0.9}, {'alpha':0.8}])
    plt.close('all')


def test_raw_nanohist(alex_hmm):
    fig, ax = new_plot()
    if alex_hmm.data.lifetime:
        bhm.raw_nanotime_hist(alex_hmm, ax=ax)
        bhm.axline_irf_thresh(alex_hmm, ax=ax)
    else:
        with pytest.raises(Exception):
            bhm.axline_irf_thresh(alex_hmm.models[2], ax=ax)
    plt.close('all')
        

def test_state_nanohist(alex_hmm):
    fig, ax = new_plot()
    if alex_hmm.data.lifetime:
        bhm.state_nanotime_hist(alex_hmm.models[2], ax=ax)
    else:
        with pytest.raises(Exception):
            bhm.state_nanotime_hist(alex_hmm.models[2], ax=ax)
    plt.close('all')

@pytest.mark.parametrize('func', (bhm.ll_E_scatter, bhm.ll_S_scatter))
def test_ll_scatter_scatter(alex_hmm, func):
    fig, ax = new_plot()
    func(alex_hmm.models[2], 0, ax=ax)
    fig, ax = new_plot()
    func(alex_hmm.models[2], 1, ax=ax, rng=3)
    fig, ax = new_plot()
    func(alex_hmm.models[2], 2, ax=ax, rng=(0.2,0.3))
    fig, ax = new_plot()
    func(alex_hmm.models[2], 1, ax=ax, rng=np.arange(0.1,0.5,0.01))
    plt.close('all')

def test_ll_trans_scatter(alex_hmm):
    fig, ax = new_plot()
    bhm.ll_trans_scatter(alex_hmm.models[2], 0,1, ax=ax)
    fig, ax = new_plot()
    bhm.ll_trans_scatter(alex_hmm.models[2], 0,1, ax=ax, rng=np.logspace(-8, -6, 10))
    plt.close('all')


def test_plot_burst_index(alex_hmm):
    for tp in ('states','transitions','longest','photons', 0):    
        fig, ax = new_plot()
        bhm.plot_burst_index(alex_hmm.models[2], tp, ax=ax)
        plt.close('all')
    fig, ax = new_plot()
    bhm.plot_burst_index(alex_hmm.models[2], 'states', ax=ax, streams=[frb.Ph_sel(Dex='Dem'),frb.Ph_sel(Dex='Aem')],
                         stream_pos={frb.Ph_sel(Dex='Dem'):0.2, frb.Ph_sel(Dex='Aem'):0.7}, 
                         stream_color={frb.Ph_sel(Dex='Dem'):'c', frb.Ph_sel(Dex='Aem'):'r'},
                         stream_edge={frb.Ph_sel(Dex='Dem'):'k', frb.Ph_sel(Dex='Aem'):'orange'},
                         stream_labels=False,
                         s=20)
    plt.close('all')
    fig, ax = new_plot()
    bhm.plot_burst_index(alex_hmm.models[2], 'states', invert=True, rng=(1,3))
    plt.close('all')
    if alex_hmm.data.lifetime:
        fig, ax = new_plot()
        bhm.plot_burst_index(alex_hmm.div_models['twodiv'], 'states', colapse_div=True)
        plt.close('all')
        fig, ax = new_plot()
        bhm.plot_burst_index(alex_hmm.div_models['twodiv'], 'states', 
                             stream_pos={i:0.1*(i+1) for i in range(alex_hmm.div_models['twodiv'].ndet)})
        plt.close('all')

def test_plot_burst_path(alex_hmm):
    for tp in ('states','transitions','longest','photons', 0):    
        fig, ax = new_plot()
        bhm.plot_burst_path(alex_hmm.models[2], 'states', param='E', ax=ax, 
                            state_color=['r','g','b','k'], linewidth=[1,2,3,4])
        plt.close('all')
        fig, ax = new_plot()
        bhm.plot_burst_path(alex_hmm.models[2], 'states', param='S', ax=ax, 
                            state_color=['r','g','b'], linewidth=[1,2,3,4])
        plt.close('all')
        if alex_hmm.data.lifetime:
            fig, ax = new_plot()
            bhm.plot_burst_path(alex_hmm.div_models['twodiv'][2], 'states',param='E', ax=ax, 
                                state_color=['r','g','b'], linewidth=[1,2,3,4])
            plt.close('all')
            fig, ax = new_plot()
            bhm.plot_burst_path(alex_hmm.div_models['twodiv'][2], 'states',param='state_nano_mean', ax=ax, 
                                state_color=['r','g','b'], linewidth=[1,2,3,4], stream=0)
            plt.close('all')
            fig, ax = new_plot()
            bhm.plot_burst_path(alex_hmm.div_models['twodiv'][2], 'states',param='state_nano_mean', ax=ax, 
                                state_color=['r','g','b'], linewidth=[1,2,3,4], stream=frb.Ph_sel(Dex='Aem'))
            plt.close('all')
            


if __name__ == '__main__':
    pytest.main("-x -v tests/test_burstH2MM.py")
