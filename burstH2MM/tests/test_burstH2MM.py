import pytest
from urllib.request import urlretrieve

import numpy as np
import fretbursts as frb
import burstH2MM as bhm
import H2MM_C as hm




@pytest.fixture
def nsalex_data():
    no_success = True
    while no_success:
        try:
            urlretrieve('https://zenodo.org/record/5902313/files/HP3_TE300_SPC630.hdf5', filename='HP3_TE300_SPC630.hdf5')
        except:
            no_success = True
        else:
            no_success = False    
    data = frb.loader.photon_hdf5("HP3_TE300_SPC630.hdf5")
    frb.loader.alex_apply_period(data)
    data.calc_bg(fun=frb.bg.exp_fit, time_s=30, tail_min_us='auto', F_bg=1.7)
    data.burst_search(m=10, F=6, ph_sel=frb.Ph_sel(Dex='DAem', Aex='Aem'))
    data_all = data.select_bursts(frb.select_bursts.size, add_naa=True, th1=50)
    data_da = data_all.select_bursts(frb.select_bursts.size, th1=30)
    
    return data_da

def test_raises(nsalex_data):
    """Test that H2MM_list does not return values for H2MM_results when ideal model not set"""
    bdata = bhm.BurstData(nsalex_data)
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


def test_optimization(nsalex_data):
    """Smoke test for optimization"""
    bdata = bhm.BurstData(nsalex_data)
    model = hm.factory_h2mm_model(2,3)
    bdata.models.optimize(model)
    assert bdata.models.ICL.size > 0

@pytest.fixture
def nsalex_hmm(nsalex_data):
    bdata = bhm.BurstData(nsalex_data)
    bdata.models.calc_models()
    return bdata


def test_bg_correction(nsalex_hmm):
    """Test calcualtion of bg correction"""
    assert np.allclose(nsalex_hmm.models[0].dwell_E_corr, np.concatenate(nsalex_hmm.data.E))
    assert np.allclose(nsalex_hmm.models[0].dwell_S_corr, np.concatenate(nsalex_hmm.data.S))

def test_ideal(nsalex_hmm):
    nsalex_hmm.models.ideal = 3
    assert hasattr(nsalex_hmm.models, "E")
    assert hasattr(nsalex_hmm.models, "E_corr")
    assert hasattr(nsalex_hmm.models, "S")
    assert hasattr(nsalex_hmm.models, "S_corr")
    assert hasattr(nsalex_hmm.models, "trans_locs")
    assert hasattr(nsalex_hmm.models, "dwell_E")
    assert hasattr(nsalex_hmm.models, "dwell_E_corr")
    assert hasattr(nsalex_hmm.models, "dwell_S")
    assert hasattr(nsalex_hmm.models, "dwell_S_corr")

if __name__ == '__main__':
    pytest.main("-x -v tests/test_burstH2MM.py")
