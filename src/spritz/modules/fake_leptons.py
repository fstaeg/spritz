import awkward as ak
import numpy as np
import scipy as sc

def erf(x, a, b, c, d):
    return a-b*sc.special.erf((x-c)/d)

def logistic(x, a, b, c, d):
    return a+b/(1+np.exp((x-c)/d))

def exponential(x, a, b, c):
    return a+b*np.exp(-x/c)


_PARAMS = {
    "inc_mm": {
        "erf": {
            "param": np.array([1.714e+00, 6.261e-01, 8.519e+01, 8.993e+01]),
            "cov": np.array([
                [ 1.837e-02,  2.030e-02, -2.781e+00,  2.768e+00],
                [ 2.030e-02,  2.283e-02, -3.070e+00,  3.148e+00],
                [-2.781e+00, -3.070e+00,  4.238e+02, -4.205e+02],
                [ 2.768e+00,  3.148e+00, -4.205e+02,  4.464e+02],
            ]),
        },
        "logistic": {
            "param": np.array([1.078e+00, 1.189e+00, 9.155e+01, 3.680e+01]),
            "cov": np.array([
                [ 6.753e-04, -3.746e-03,  1.938e-01, -1.421e-01],
                [-3.746e-03,  3.921e-02, -2.592e+00,  1.368e+00],
                [ 1.938e-01, -2.592e+00,  1.833e+02, -8.936e+01],
                [-1.421e-01,  1.368e+00, -8.936e+01,  4.997e+01],
            ]),
        },
        "exponential": {
            "param": np.array([9.873e-01, 1.814e+00, 8.398e+01]),
            "cov": np.array([
                [ 1.058e-03,  6.921e-04, -1.921e-01],
                [ 6.921e-04,  3.417e-03, -2.429e-01],
                [-1.921e-01, -2.429e-01,  4.097e+01],
            ]),
        },
    },
    
    "inc_mm_bveto": {
        "erf": {
            "param": np.array([2.4332327078194371, 0.18657857745195813, 95.47538517379735, 0.0041357773239211080]),
            "cov": np.array([
                [ 8.4715556528329418e-04,  6.7592407283748108e-05, -3.1687434495629068e-03, -9.4930031601572331e-07],
                [ 6.7592407283748108e-05,  8.5522918096314941e-04, -3.9318548551526243e-03, -1.1779151944493464e-06],
                [-3.1687434495629068e-03, -3.9318548551526243e-03,  6.7114255752041541e-01,  2.0049160358057085e-04],
                [-9.4930031601572331e-07, -1.1779151944493464e-06,  2.0049160358057085e-04,  8.1876433921955943e-08],
            ]),
        },
        "logistic": {
            "param": np.array([2.2466465278266718, 0.37315712596906037, 95.400624935381657, 0.016757148888925065]),
            "cov": np.array([
                [ 8.6898099333971435e-05, -9.0588521467064293e-06,  1.6931154824628921e-03, -1.2296821107141427e-04],
                [-9.0588521467064293e-06,  1.8467610378085245e-04,  1.1120192804015495e-03, -8.0764094870637945e-05],
                [ 1.6931154824628921e-03,  1.1120192804015495e-03,  2.0038200190824886e+04, -1.4497489017957648e+03],
                [-1.2296821107141427e-04, -8.0764094870637945e-05, -1.4497489017957648e+03,  1.0512005373783498e+02],
            ]),
        },
        "exponential": {
            "param": np.array([-163.59122957349595, 166.76351716435056, 20245.127817698572]),
            "cov": np.array([
                [ 1.4429161363646162e-01, -1.3628479695383369e-01, -2.2240003647768000e+02],
                [-1.3628479695383369e-01,  1.4444917806789037e-01, -1.9240216312618438e+02],
                [-2.2240003647768000e+02, -1.9240216312618438e+02,  1.1219025465718681e+07],
            ]),
        },
    },
}

def transferFactor(x, model='exponential', variation='nominal', region='inc_mm'):
    entry = _PARAMS.get(region, {}).get(model)
    if entry is None:
        return np.ones_like(x)

    param, cov = entry["param"], entry["cov"]
    tf = eval(model)(x, *param)

    if variation in ['up', 'down']:
        rng = np.random.default_rng(seed=0)
        param_b = rng.multivariate_normal(param, cov, size=100)
        tf_b = [eval(model)(x, *p) for p in param_b]
        tf_err = np.std(tf_b, axis=0)
        return tf + tf_err if variation == 'up' else tf - tf_err

    elif variation == 'nominal':
        return tf

    return np.ones_like(x)


# Maps variation name → (model, tf_variation) for compute_fake_lep_weight
_FAKE_VAR_MODEL = {
    'nom':                ('exponential', 'nominal'),
    'fakerw_param_up':    ('exponential', 'up'),
    'fakerw_param_down':  ('exponential', 'down'),
    'fakerw_model_exp':   ('exponential', 'nominal'),
    'fakerw_model_erf':   ('erf',         'nominal'),
}


def register_fake_lep_variations(variations):
    """Register fakerw variations without any field-swapping substitution.

    Use this when the weight is recomputed inside the variation loop
    (e.g. when it depends on bVeto, which is only available inside the loop).
    Call compute_fake_lep_weight inside the loop to set events['fakeLepWeight'].
    """
    for var_name in _FAKE_VAR_MODEL:
        if var_name == 'nom':
            continue
        variations.register_variation([], var_name)
    return variations


def compute_fake_lep_weight(events, region_masks, variation='nom'):
    """Compute fakeLepWeight for one variation using per-region TF parameters.

    region_masks: dict mapping region name → boolean array (mutually exclusive masks
    that partition events; unmatched events default to weight 1).
    variation: key from _FAKE_VAR_MODEL (e.g. 'nom', 'fakerw_param_up').
    """
    mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass
    model, tf_var = _FAKE_VAR_MODEL.get(variation, ('exponential', 'nominal'))

    result = ak.ones_like(mll)
    for region, mask in region_masks.items():
        tf = transferFactor(mll, model=model, variation=tf_var, region=region)
        result = ak.where(mask, tf, result)
    return result


def reweightFakeLep(events, variations, region='inc_mm'):
    """Single-region fake lep reweighting (used by runners without bVeto splitting)."""
    mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass

    events['fakeLepWeight'] = transferFactor(mll, model='exponential', variation='nominal', region=region)
    events['fakeLepWeight_fakerw_param_up'] = transferFactor(mll, model='exponential', variation='up', region=region)
    events['fakeLepWeight_fakerw_param_down'] = transferFactor(mll, model='exponential', variation='down', region=region)
    events['fakeLepWeight_fakerw_model_exp'] = transferFactor(mll, model='exponential', variation='nominal', region=region)
    events['fakeLepWeight_fakerw_model_erf'] = transferFactor(mll, model='erf', variation='nominal', region=region)

    variations.register_variation(['fakeLepWeight'], 'fakerw_param_up')
    variations.register_variation(['fakeLepWeight'], 'fakerw_param_down')
    variations.register_variation(['fakeLepWeight'], 'fakerw_model_exp')
    variations.register_variation(['fakeLepWeight'], 'fakerw_model_erf')

    return events, variations



