import awkward as ak
import numpy as np
import scipy as sc

def erf(x, a, b, c, d):
    return a-b*sc.special.erf((x-c)/d)

def logistic(x, a, b, c, d):
    return a+b/(1+np.exp((x-c)/d))

def exponential(x, a, b, c):
    return a+b*np.exp(-x/c)


def transferFactor(x, model='logistic', variation='nominal'):
    if model == 'erf':
        param = np.array([1.714e+00, 6.261e-01, 8.519e+01, 8.993e+01])
        cov = np.array([
            [ 1.837e-02,  2.030e-02, -2.781e+00,  2.768e+00],
            [ 2.030e-02,  2.283e-02, -3.070e+00,  3.148e+00],
            [-2.781e+00, -3.070e+00,  4.238e+02, -4.205e+02],
            [ 2.768e+00,  3.148e+00, -4.205e+02,  4.464e+02]
        ])
    
    elif model == 'logistic':
        param = np.array([1.078e+00, 1.189e+00, 9.155e+01, 3.680e+01])
        cov = np.array([
            [ 6.753e-04, -3.746e-03,  1.938e-01, -1.421e-01],
            [-3.746e-03,  3.921e-02, -2.592e+00,  1.368e+00],
            [ 1.938e-01, -2.592e+00,  1.833e+02, -8.936e+01],
            [-1.421e-01,  1.368e+00, -8.936e+01,  4.997e+01]
        ])
    
    elif model == 'exponential':
        param = np.array([9.873e-01, 1.814e+00, 8.398e+01])
        cov = np.array([
            [ 1.058e-03,  6.921e-04, -1.921e-01],
            [ 6.921e-04,  3.417e-03, -2.429e-01],
            [-1.921e-01, -2.429e-01,  4.097e+01]
        ])

    else:
        return np.ones_like(x)

    tf = eval(model)(x, *param)

    if variation in ['up','down']: # compute uncertainty
        rng = np.random.default_rng(seed=0)
        param_b = rng.multivariate_normal(param, cov, size=100)
        tf_b = [eval(model)(x, *p) for p in param_b]
        tf_err = np.std(tf_b, axis=0)
        
        if variation == 'up':
            return tf+tf_err
        elif variation == 'down':
            return tf-tf_err

    elif variation == 'nominal':
        return tf

    else:
        return np.ones_like(x)


def reweightFakeLep(events, variations):
    
    mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass
    
    events['fakeLepWeight'] = transferFactor(mll, model='logistic', variation='nominal')
    events['fakeLepWeight_fakerw_param_up'] = transferFactor(mll, model='logistic', variation='up')
    events['fakeLepWeight_fakerw_param_down'] = transferFactor(mll, model='logistic', variation='down')
    events['fakeLepWeight_fakerw_model_exp'] = transferFactor(mll, model='exponential', variation='nominal')
    events['fakeLepWeight_fakerw_model_erf'] = transferFactor(mll, model='erf', variation='nominal')

    variations.register_variation(['fakeLepWeight'], 'fakerw_param_up')
    variations.register_variation(['fakeLepWeight'], 'fakerw_param_down')
    variations.register_variation(['fakeLepWeight'], 'fakerw_model_exp')
    variations.register_variation(['fakeLepWeight'], 'fakerw_model_erf')
    
    return events, variations



