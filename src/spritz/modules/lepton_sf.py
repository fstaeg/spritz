import awkward as ak
import numpy as np
import spritz.framework.variation as variation_module
from spritz.framework.framework import correctionlib_wrapper


def lepton_sf(events, variations, ceval_lepton_sf, cfg, ele_use_latinos_sf=False, mu_use_latinos_sf=False):
    minpt_mu = 10.0001 if mu_use_latinos_sf else 15.0001
    maxpt_mu = 199.9999 if mu_use_latinos_sf else float('inf')
    mineta_mu = -2.3999
    maxeta_mu = 2.3999
    minp_mu = 50.0001
    maxp_mu = float('inf')

    minpt_ele = 10.0001
    maxpt_ele = 499.9999 if ele_use_latinos_sf else float('inf')
    mineta_ele = -2.4999
    maxeta_ele = 2.4999
    ele_mask = abs(events.Lepton.pdgId) == 11
    mu_mask = abs(events.Lepton.pdgId) == 13

    run_period = ak.copy(events.run_period)
    pt = ak.copy(events.Lepton.pt)
    eta = ak.copy(events.Lepton.eta)
    p = ak.copy(events.Lepton.p)

    pt = ak.where(ele_mask & (pt < minpt_ele), minpt_ele, pt)
    pt = ak.where(ele_mask & (pt > maxpt_ele), maxpt_ele, pt)
    pt = ak.where(mu_mask & (pt < minpt_mu), minpt_mu, pt)
    pt = ak.where(mu_mask & (pt > maxpt_mu), maxpt_mu, pt)

    p = ak.where(mu_mask & (p < minp_mu), minp_mu, p)
    p = ak.where(mu_mask & (p > maxp_mu), maxp_mu, p)

    eta = ak.where(ele_mask & (eta < mineta_ele), mineta_ele, eta)
    eta = ak.where(ele_mask & (eta > maxeta_ele), maxeta_ele, eta)
    eta = ak.where(mu_mask & (eta < mineta_mu), mineta_mu, eta)
    eta = ak.where(mu_mask & (eta > maxeta_mu), maxeta_mu, eta)

    sfs_dict = {}
    sfs_dict["ele_reco_below"] = {
        "wrap": correctionlib_wrapper(ceval_lepton_sf["Electron_RecoSF_RecoBelow20"]),
        "mask": ele_mask & (pt >= 10.0) & (pt < 20.0),
        "output": "reco_sf",
    }

    sfs_dict["ele_reco_above"] = {
        "wrap": correctionlib_wrapper(ceval_lepton_sf["Electron_RecoSF_RecoAbove20"]),
        "mask": ele_mask & (pt >= 20.0),
        "output": "reco_sf",
    }

    sfs_dict["ele_wp"] = {
        "wrap": correctionlib_wrapper(ceval_lepton_sf["Electron_IdIsoSF_" + (
            "LatinosHWW" if ele_use_latinos_sf else cfg["leptonsWP"]["eleWP"].split('_')[-1])]),
        "mask": ele_mask & events.Lepton["isTightElectron_" + cfg["leptonsWP"]["eleWP"]],
        "output": "ele_id_iso_sf",
    }

    sfs_dict["muon_reco"] = {
        "wrap": correctionlib_wrapper(ceval_lepton_sf["Muon_RecoSF" + ("_highPtId" if cfg["leptonsWP"]["muWP"]=="cut_highPtId" else "")]),
        "mask": mu_mask,
        "output": "muon_reco_sf",
    }

    sfs_dict["muon_id"] = {
        "wrap": correctionlib_wrapper(ceval_lepton_sf["Muon_IdSF_" + ("LatinosHWW" if mu_use_latinos_sf else cfg["leptonsWP"]["muWP"].split('_')[-1])]),
        "mask": mu_mask & events.Lepton["isTightMuon_" + cfg["leptonsWP"]["muWP"]],
        "output": "muon_id_sf",
    }

    sfs_dict["muon_iso"] = {
        "wrap": correctionlib_wrapper(ceval_lepton_sf["Muon_IsoSF_" + ("LatinosHWW" if mu_use_latinos_sf else cfg["leptonsWP"]["muWP"].split('_')[-1])]),
        "mask": mu_mask & events.Lepton["isTightMuon_" + cfg["leptonsWP"]["muWP"]],
        "output": "muon_iso_sf",
    }

    # Lepton Reco SF
    lepton_reco_sf = {k: ak.ones_like(pt) for k in ['nominal','up','down']}
    
    # electrons (nominal, syst_up, syst_down)
    for reco_sf in ['ele_reco_below','ele_reco_above']:
        mask = sfs_dict[reco_sf]['mask']
        _eta = ak.mask(eta, mask)
        _pt = ak.mask(pt, mask)
        lepton_reco_sf['nominal'] = ak.where(
            mask,
            sfs_dict[reco_sf]['wrap']('nominal', _eta, _pt),
            lepton_reco_sf['nominal']
        )
        for variation in ['up','down']:
            lepton_reco_sf[variation] = ak.where(
                mask,
                sfs_dict[reco_sf]['wrap']('syst_'+variation, _eta, _pt),
                lepton_reco_sf[variation]
            )

    # muons (nominal, syst, stat)
    mask = sfs_dict['muon_reco']['mask']
    _eta = ak.mask(eta, mask)
    _pt = ak.mask(pt, mask)
    _p = ak.mask(p, mask)
    lepton_reco_sf['nominal'] = ak.where(
        mask,
        sfs_dict['muon_reco']['wrap'](_eta, _p if cfg["leptonsWP"]["muWP"]=="cut_highPtId" else _pt, 'nominal'),
        lepton_reco_sf['nominal']
    )
    for variation in ['stat','syst']:
        lepton_reco_sf[variation] = ak.where(
            mask,
            sfs_dict['muon_reco']['wrap'](_eta, _p if cfg["leptonsWP"]["muWP"]=="cut_highPtId" else _pt, variation),
            ak.zeros_like(pt)
        )
    lepton_reco_sf['err'] = np.sqrt(
        lepton_reco_sf['stat']**2 + lepton_reco_sf['syst']**2
    )
    for sign,variation in zip([+1,-1],['up','down']):
        lepton_reco_sf[variation] = ak.where(
            mask,
            lepton_reco_sf['nominal'] + sign*lepton_reco_sf['err'],
            lepton_reco_sf[variation]
        )

    # finalize lepton reco sf and variations
    events[('Lepton','RecoSF')] = lepton_reco_sf['nominal']

    for t, mask in zip(['ele','mu'], [ele_mask,mu_mask]):
        for tag in ['up','down']:
            var_name = f'{t}_reco_{tag}'
            varied_col = variation_module.Variation.format_varied_column(
                ('Lepton', 'RecoSF'), var_name
            )
            events[varied_col] = ak.where(
                mask, 
                lepton_reco_sf[tag], 
                lepton_reco_sf['nominal']
            )
            variations.register_variation([('Lepton', 'RecoSF')], var_name)

    # Lepton ID and Iso SF
    lepton_idiso_sf = {k: ak.ones_like(pt) for k in ['nominal','up','down']}

    # electrons (ID and Iso SF combined; nominal, stat, syst)
    mask = sfs_dict['ele_wp']['mask']
    _run_period = ak.mask(run_period, mask)
    _eta = ak.mask(eta, mask)
    _pt = ak.mask(pt, mask)
    lepton_idiso_sf['nominal'] = ak.where(
        mask,
        (sfs_dict['ele_wp']['wrap'](_run_period, 'nominal', _eta, _pt) if ele_use_latinos_sf
            else sfs_dict['ele_wp']['wrap']('nominal', _eta, _pt)),
        lepton_idiso_sf['nominal']
    )
    if ele_use_latinos_sf:
        ele_syst = ak.where(
            mask,
            sfs_dict['ele_wp']['wrap'](_run_period, 'syst', _eta, _pt),
            ak.zeros_like(pt)
        )
        ele_stat = ak.where(
            mask,
            sfs_dict['ele_wp']['wrap'](_run_period, 'stat', _eta, _pt),
            ak.zeros_like(pt)
        )
        ele_err = np.sqrt(ele_syst**2 + ele_stat**2)

    for sign,variation in zip([+1,-1],['up','down']):
        lepton_idiso_sf[variation] = ak.where(
            mask,
            (lepton_idiso_sf['nominal'] + sign*ele_err if ele_use_latinos_sf
                else sfs_dict['ele_wp']['wrap']('syst_'+variation, _eta, _pt)),
            lepton_idiso_sf[variation]
        )

    # muons (ID and Iso SF separate; nominal, syst)
    muon_idiso_sf = {
        idiso_sf: {
            'nominal': ak.ones_like(pt),
            'stat': ak.zeros_like(pt),
            'syst': ak.zeros_like(pt)
        } for idiso_sf in ['muon_id','muon_iso']
    }
    
    for idiso_sf in ['muon_id','muon_iso']:
        mask = sfs_dict[idiso_sf]['mask']
        _eta = ak.mask(eta, mask)
        _pt = ak.mask(pt, mask)
        _p = ak.mask(p, mask)
        muon_idiso_sf[idiso_sf]['nominal'] = ak.where(
            mask,
            (sfs_dict[idiso_sf]['wrap']('nominal', _eta, _pt) if mu_use_latinos_sf
                else sfs_dict[idiso_sf]['wrap'](_eta, _p if cfg["leptonsWP"]["muWP"]=="cut_highPtId" else _pt, 'nominal')),
            muon_idiso_sf[idiso_sf]['nominal']
        )
        muon_idiso_sf[idiso_sf]['syst'] = ak.where(
            mask,
            (sfs_dict[idiso_sf]['wrap']('syst', _eta, _pt) if mu_use_latinos_sf
                else sfs_dict[idiso_sf]['wrap'](_eta, _p if cfg["leptonsWP"]["muWP"]=="cut_highPtId" else _pt, 'syst')),
            muon_idiso_sf[idiso_sf]['syst']
        )
        muon_idiso_sf[idiso_sf]['stat'] = ak.where(
            mask,
            (ak.zeros_like(pt) if mu_use_latinos_sf
                else sfs_dict[idiso_sf]['wrap'](_eta, _p if cfg["leptonsWP"]["muWP"]=="cut_highPtId" else _pt, 'stat')),
            muon_idiso_sf[idiso_sf]['stat']
        )
        muon_idiso_sf[idiso_sf]['err'] = np.sqrt(
            muon_idiso_sf[idiso_sf]['stat']**2 + muon_idiso_sf[idiso_sf]['syst']**2
        )

    # combine ID and Iso SF
    muon_sf = muon_idiso_sf['muon_id']['nominal']*muon_idiso_sf['muon_iso']['nominal']
    muon_err = np.sqrt(
        muon_idiso_sf['muon_id']['nominal']**2 * muon_idiso_sf['muon_iso']['err']**2
        + muon_idiso_sf['muon_iso']['nominal']**2 * muon_idiso_sf['muon_id']['err']**2
    )

    lepton_idiso_sf['nominal'] = ak.where(
        mu_mask,
        muon_sf,
        lepton_idiso_sf['nominal']
    )
    for sign,variation in zip([+1,-1],['up','down']):
        lepton_idiso_sf[variation] = ak.where(
            mu_mask,
            lepton_idiso_sf['nominal'] + sign*muon_err,
            lepton_idiso_sf[variation]
        )


    # finalize lepton ID and Iso SF and variations
    events[('Lepton','TightSF')] = lepton_idiso_sf['nominal']

    for t, mask in zip(['ele','mu'], [ele_mask,mu_mask]):
        for tag in ['up','down']:
            var_name = f'{t}_idiso_{tag}'
            varied_col = variation_module.Variation.format_varied_column(
                ('Lepton', 'TightSF'), var_name
            )
            events[varied_col] = ak.where(
                mask, 
                lepton_idiso_sf[tag], 
                lepton_idiso_sf['nominal']
            )
            variations.register_variation([('Lepton', 'TightSF')], var_name)


    return events, variations
