import awkward as ak
import numpy as np
import scipy as sc

def transferFactor(x,model='logistic'):
    if model=='erf':
        a = 1.762766860364751
        b = 0.6842345851501445
        c = 74.84135223473334
        d = 102.53340682913422
        return a-b*sc.special.erf((x-c)/d)
    elif model=='logistic':
        a = 1.068429573576053
        b = 1.2399710549676959
        c = 86.2515937898194
        d = 40.62815812489409
        return a+b/(1+np.exp((x-c)/d))
    elif model=='exponential':
        a = 0.991751462110116
        b = 1.709864672574629
        c = 85.64234416285053
        return a+b*np.exp(-x/c)
    else:
        return 1.

def reweightFakeLep(events,model='logistic'):
    mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass
    events['fakeLepWeight'] = ak.where(events.mm_ss, transferFactor(mll,model), ak.ones_like(mll))
    return events
