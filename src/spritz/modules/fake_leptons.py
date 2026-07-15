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

    # compute uncertainty
    rng = np.random.default_rng(seed=0)
    param_b = rng.multivariate_normal(param, cov, size=100)
    err = np.std([eval(model)(x, *p) for p in param_b], axis=0)

    if variation == "up":
        return nominal + err
    elif variation == "down":
        return nominal - err
    elif variation == "nominal":
        return nominal
    else:
        return np.ones_like(x)


def reweightFakes(events, variations, cfg):
    with open(cfg["fakesRW"], "r") as f:
        parameters = json.load(f)

    mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass

    events["fakesRW"] = transferFactor(mll, parameters, "erf", "nominal")
    events["fakesRW_fakes_param_up"] = transferFactor(mll, parameters, "erf", "up")
    events["fakesRW_fakes_param_down"] = transferFactor(mll, parameters, "erf", "down")
    events["fakesRW_fakes_model"] = transferFactor(mll, parameters, "logistic", "nominal")

    variations.register_variation(["fakesRW"], "fakes_param_up")
    variations.register_variation(["fakesRW"], "fakes_param_down")
    variations.register_variation(["fakesRW"], "fakes_model")
    
    return events, variations



