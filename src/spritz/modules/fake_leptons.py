import awkward as ak
import numpy as np
import scipy as sc
import json

def erf(x, a, b, c, d):
    return a-b*sc.special.erf((x-c)/d)

def logistic(x, a, b, c, d):
    return a+b/(1+np.exp((x-c)/d))

def exponential(x, a, b, c):
    return a+b*np.exp(-x/c)


def transferFactor(x, parameters, model="erf", variation="nominal"):
    if model not in parameters:
        return np.ones_like(x)

    param, cov = parameters[model]["parameters"], parameters[model]["covariance"]
    nominal = eval(model)(x, *param)

    if variation in ["up","down"]:
        # compute uncertainty
        rng = np.random.default_rng(seed=0)
        param_b = rng.multivariate_normal(param, cov, size=100)
        err = np.std([eval(model)(x, *p) for p in param_b], axis=0)
        
        if variation == "up":
            return nominal + err
        if variation == "down":
            return nominal - err
    
    elif variation == "nominal":
        return nominal
    
    else:
        return np.ones_like(x)


def reweightFakes(events, variation_name, parameters):
    if variation_name == "fakes_before":
        return ak.ones_like(events.weight)

    if variation_name == "fakes_model":
        model = "logistic"
    else:
        model = "erf"
    
    if variation_name == "fakes_param_up":
        variation = "up"
    elif variation_name == "fakes_param_down":
        variation = "down"
    else:
        variation = "nominal"

    mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass

    return transferFactor(mll, parameters, model, variation)


def getFakeRW(variations, cfg):
    with open(cfg["fakesRW"], "r") as f:
        parameters = json.load(f)

    variations.register_variation([], "fakes_param_up")
    variations.register_variation([], "fakes_param_down")
    variations.register_variation([], "fakes_model")
    variations.register_variation([], "fakes_before")

    return variations, parameters


