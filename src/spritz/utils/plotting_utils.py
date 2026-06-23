import numpy as np
import hist
import matplotlib as mpl
import math

def darker_color(color):
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    darker_factor = 4 / 5
    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)

def union(lists):
    res = list()
    for l in lists:
        res += [x for x in l if not x in res]
    return res

unc_colors = ["purple","red","green"] + list(mpl.colors.TABLEAU_COLORS.values())

def oom(number):
    if number==0: return 0
    else: return int(math.floor(math.log(number, 10)))

def get_yrange(histo_dict, denominator=None, ylog=False, divide=None, variations=None):
    if denominator is not None:
        denominator_nom = np.where(histo_dict[denominator].nominal >= 1e-6, histo_dict[denominator].nominal, 1e-6)
        ymin = min([h.min(divide=denominator_nom, variations=variations) for h in histo_dict.values()])
        ymax = max([h.max(divide=denominator_nom, variations=variations) for h in histo_dict.values()])
        ydiff = min(1, 1.2 * max(ymax-1, 1-ymin))
        yrange = (1-ydiff, 1+ydiff)

    else:
        ymin = min([h.min(divide=divide, variations=variations) for h in histo_dict.values()])
        ymax = max([h.max(divide=divide, variations=variations) for h in histo_dict.values()])
        if ylog:
            ymin = max(1e-2, 0.5*ymin)
            ymax = ymax * 10**((oom(ymax)-oom(ymin))/4)
        else:
            ymin = 0
            ymax = ymax + ymax/5
        yrange = (ymin, ymax)

    return yrange


class HistVariation(object):

    def __init__(self, variations_dict={}, kind=None):
        self.variations_dict = variations_dict
        self.kind = kind

    def __getitem__(self, key):
        if isinstance(key, slice):
            return HistVariation(
                variations_dict={k: self.variations_dict[k][key.start:key.stop] for k in self.keys()},
                kind=self.kind
            )
        elif key in self.keys():
            return self.variations_dict[key]
        else:
            return getattr(self, key)

    def keys(self): 
        return list(self.variations_dict.keys())

    def values(self): 
        return list(self.variations_dict.values())

    @classmethod
    def make_variation(cls, directory, nuisance, sample):
        
        h = directory[f"histo_{sample}"].to_hist()
        name, type, kind = nuisance.get("name"), nuisance.get("type"), nuisance.get("kind")
        
        if kind in ["envelope", "square", "stdev"]:
            read_tag = lambda v : v if not isinstance(v,tuple) else v[1]
            variation_tags = [read_tag(v) for v in nuisance["samples"][sample]]
            if sample=="Single Top" and name=="PDFWeight":
                variations_dict = {
                    variation: directory[f"histo_{sample}_{variation}"].values().copy() for variation in variation_tags
                }
            else:
                variations_dict = {
                    variation: directory[f"histo_{sample}_{variation}"].values()-h.values() for variation in variation_tags
                }
        else:
            if type == "lnN":
                scaling = float(nuisance["samples"][sample])
                up, down = (scaling-1)*h.values(), (1-1/scaling)*h.values()
            elif type == "stat":
                up, down = np.sqrt(h.variances()), np.sqrt(h.variances())
            else:
                up = abs(directory[f"histo_{sample}_{name}Up"].values()-h.values())
                down = abs(directory[f"histo_{sample}_{name}Down"].values()-h.values())

            variations_dict = { "up": up, "down": down }
        
        return cls(variations_dict, kind)

    @classmethod
    def add(cls, summands):
        
        kind = summands[0].kind
        variations = summands[0].keys()
        assert all([s.kind==kind for s in summands])

        variations_dict = {
            vari: sum([s[vari] for s in summands if vari in s.keys()]) for vari in variations
        }
        
        return cls(variations_dict, kind)

    def up(self):
        if self.kind == "envelope":
            up = np.max(np.array([variation for variation in self.values()]), axis=0)
            return np.max((up, np.zeros_like(up)), axis=0)
        elif self.kind == "square":
            varied = np.array([variation for variation in self.values()])
            variation_histo = sum(np.square(varied))
            return np.sqrt(variation_histo)
        elif self.kind == "stdev":
            varied = np.array([variation for variation in self.values()])
            return np.std(varied, axis=0)
        else:
            return self["up"]

    def down(self):
        if self.kind == "envelope":
            down = np.min(np.array([variation for variation in self.values()]), axis=0)
            return np.min((down, np.zeros_like(down)), axis=0)
        elif self.kind == "square":
            varied = np.array([variation for variation in self.values()])
            variation_histo = sum(np.square(varied))
            return np.sqrt(variation_histo)
        elif self.kind == "stdev":
            varied = np.array([variation for variation in self.values()])
            return np.std(varied, axis=0)
        else:
            return self["down"]


class Histogram(object):

    def __init__(self, name, nominal, varied={}, corrected={}, is_data=False, color="black", linestyle="solid", axis=None):
        self.name = name
        self.nominal = nominal
        self.varied = varied
        self.corrected = corrected
        self.is_data = is_data
        self.color = color
        self.linestyle = linestyle
        self.axis = axis

    def __getitem__(self, key):
        if isinstance(key, slice):
            edges = self.edges[key.start:key.stop+1]
            if self.variable_width:
                axis = hist.axis.Variable(edges, name=self.axis.name)
            else:
                axis = hist.axis.Regular(len(edges)-1, edges[0], edges[-1], name=self.axis.name)

            varied = {
                nuis: self.varied[nuis][key.start:key.stop] for nuis in self.variation_names
            }

            corrected = {
                corr: self.corrected[corr][key.start:key.stop] for corr in self.correction_names
            }

            return Histogram(
                name=self.name,
                nominal=self.nominal[key.start:key.stop],
                varied=varied,
                corrected=corrected,
                is_data=self.is_data,
                color=self.color,
                axis=axis
            )
        
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def make_correction(directory, correction, sample):
        corr_name = correction.get("name", correction)
        return directory[f"histo_{sample}_{corr_name}Before"].to_hist().values()

    @classmethod
    def make_hist(cls, directory, nuisances, corrections, sample, is_data=False, color="black"):
        
        nominal = directory[f"histo_{sample}"].to_hist()
        nuisances = { k:v for k,v in nuisances.items() if (sample in v["samples"] 
            and not v["type"] in ["rateParam","auto"]) }
        corrections = { k:v for k,v in corrections.items() if sample in v["samples"] }

        varied = {
            nuis: HistVariation.make_variation(directory, nuisances[nuis], sample) for nuis in nuisances
        }

        corrected = {
            corr: cls.make_correction(directory, corrections[corr], sample) for corr in corrections
        }
        
        return cls(name=sample, nominal=nominal.values(), varied=varied, corrected=corrected, is_data=is_data, color=color, axis=nominal.axes[0])

    @classmethod
    def sum_hist(cls, name, histos, is_data=False, color="black"):
        
        nominal = sum([h.nominal for h in histos])
        nuisances = union([h.variation_names for h in histos])
        corrections = union([h.correction_names for h in histos])
        
        varied = {
            nuis: HistVariation.add([h.varied[nuis] for h in histos if nuis in h.variation_names]
            ) for nuis in nuisances
        }

        corrected = {
            corr: sum([h.corrected[corr] for h in histos if corr in h.correction_names]
                + [h.nominal for h in histos if not corr in h.correction_names]
            ) for corr in corrections
        }
        
        return cls(name=name, nominal=nominal, varied=varied, corrected=corrected, is_data=is_data, color=color, axis=histos[0].axis if len(histos)>0 else None)

    @classmethod
    def empty_like(cls, histo, name=None, is_data=None, color=None):
        name = histo.name if name is None else name
        is_data = histo.is_data if is_data is None else is_data
        color = histo.color if color is None else color 

        return cls(name=name, nominal=np.zeros_like(histo.nominal), varied={}, corrected={}, is_data=is_data, color=color, axis=histo.axis)

    @property
    def centers(self): return self.axis.centers

    @property
    def edges(self): return self.axis.edges

    @property
    def widths(self): return self.axis.widths

    @property
    def variable_width(self): return isinstance(self.axis, hist.axis.Variable)

    @property
    def variation_names(self): return list(self.varied.keys())

    @property
    def correction_names(self): return list(self.corrected.keys())

    def up(self, variations=None):
        if variations is None: variations = self.variation_names
        else: variations = [v for v in variations if v in self.variation_names]
        return np.sqrt(sum([np.square(self.varied[v].up()) for v in variations]))

    def down(self, variations=None):
        if variations is None: variations = self.variation_names
        else: variations = [v for v in variations if v in self.variation_names]
        return np.sqrt(sum([np.square(self.varied[v].down()) for v in variations]))

    def rel_up(self, variations=None):
        if variations is None: variations = self.variation_names
        else: variations = [v for v in variations if v in self.variation_names]
        nominal = np.where(self.nominal >= 1e-6, self.nominal, 1e-6)
        return np.sqrt(sum([np.square(self.varied[v].up()/nominal) for v in variations]))

    def rel_down(self, variations=None):
        if variations is None: variations = self.variation_names
        else: variations = [v for v in variations if v in self.variation_names]
        nominal = np.where(self.nominal >= 1e-6, self.nominal, 1e-6)
        return np.sqrt(sum([np.square(self.varied[v].down()/nominal) for v in variations]))

    def set_axis(self, axis):
        assert len(axis.centers) == len(self.axis.centers)
        self.axis = axis

    def max(self, divide=None, variations=None, ignore_unc=False): 
        if divide is None:
            divide = self.widths if self.variable_width else np.ones_like(self.centers)
        if ignore_unc:
            return np.max((self.nominal)/divide)
        else:
            return np.max((self.nominal+self.up(variations))/divide)
        
    def min(self, divide=None, variations=None, ignore_unc=False): 
        if divide is None:
            divide = self.widths if self.variable_width else np.ones_like(self.centers)
        if ignore_unc: 
            return np.min((self.nominal)/divide)
        else:
            return np.min((self.nominal-self.down(variations))/divide)

    def plot_data(self, ax, divide=None, label=False, color=None, short_label=False):
        if divide is None:
            divide = self.widths if self.variable_width else np.ones_like(self.centers)

        color = self.color if color is None else color
        if short_label:
            labeltxt = self.name if label else None
        else:
            labeltxt = self.name + f" [{int(round(np.sum(self.nominal), 0))}]" if label else None
        
        ax.errorbar(
            y=self.nominal/divide, x=self.centers, 
            yerr=(self.down()/divide, self.up()/divide),
            label=labeltxt, color=color, fmt="o", markersize=4
        )

    def plot_mc_unc(self, ax, divide=None, label=None, color=None, uncertainties=None, highlight=[]):
        if divide is None:
            divide = self.widths if self.variable_width else np.ones_like(self.centers)
        
        unc_up = round(np.sum(self.up(uncertainties)) / np.sum(self.nominal) * 100, 2)
        unc_down = round(np.sum(self.down(uncertainties)) / np.sum(self.nominal) * 100, 2)

        color = darker_color(self.color) if color is None else color
        labeltxt = f"Syst [-{unc_down}, +{unc_up}]%" if label else None

        ax.stairs(
            values=(self.nominal + self.up(uncertainties)) / divide,
            baseline=(self.nominal - self.down(uncertainties)) / divide,
            edges=self.edges, label=labeltxt, fill=True, color=color, alpha=0.25
        )

        for i,unc in enumerate(highlight):
            if not unc in self.variation_names: continue
            unc_up = round(np.sum(self.up([unc])) / np.sum(self.nominal) * 100, 2)
            unc_down = round(np.sum(self.down([unc])) / np.sum(self.nominal) * 100, 2)
            labeltxt = f"{unc} [-{unc_down}, +{unc_up}]%" if label else None
            ax.stairs(
                values=(self.nominal + self.up([unc])) / divide,
                baseline=(self.nominal - self.down([unc])) / divide,
                edges=self.edges, label=labeltxt, fill=True, color=unc_colors[i], alpha=0.25
            )


    def plot_mc(self, ax, divide=None, baseline=None, label=False, color=None, short_label=True, zorder=1, fill=False, linestyle=None, alpha=1.):
        if divide is None:
            divide = self.widths if self.variable_width else np.ones_like(self.centers)
        if linestyle is None:
            linestyle = self.linestyle
        
        color = self.color if color is None else color
        if short_label:
            labeltxt = self.name if label else None
        else:
            labeltxt = self.name + f" [{int(round(np.sum(self.nominal), 0))}]" if label else None

        nominal = self.nominal if baseline is None else self.nominal+baseline
        
        ax.stairs(
            values=nominal/divide, 
            edges=self.edges, label=labeltxt,
            fill=fill, color=color, edgecolor=darker_color(color), alpha=alpha,
            linewidth=1, linestyle=linestyle, zorder=zorder
        )


class StackedHistogram(object):

    def __init__(self, histos):
        self.histos = []
        self.axis = histos[0].axis if len(histos)>0 else None
        
        for h in histos:
            assert h.axis == self.axis
            self.histos.append(h)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return StackedHistogram(
                histos=[h[key.start:key.stop] for h in self.histos]
            )
        elif isinstance(key, int):
            return self.histos[key]
        else:
            for h in self.histos:
                if h.name==key:
                    return h
            return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        for h in self.histos:
            yield h

    @property
    def centers(self): return self.axis.centers

    @property
    def edges(self): return self.axis.edges

    @property
    def widths(self): return self.axis.widths

    @property
    def variable_width(self): return isinstance(self.axis, hist.axis.Variable)

    def contains(self, key):
        return any([h.name==key for h in self.histos])

    def sum(self, name, color="black"):
        return Histogram.sum_hist(name=name, histos=self.histos, color=color)

    def add(self, histo, position=None):
        assert (self.axis is None) or (histo.axis == self.axis)
        if position is None:
            self.histos.append(histo)
        else:
            self.histos = self.histos[:position] + [histo] + self.histos[position:]

    def set_axis(self, axis):
        assert (self.axis is None) or (len(axis.centers) == len(self.axis.centers))
        for h in self.histos:
            h.set_axis(axis)
        self.axis = axis

    def max(self, divide=None, variations=None, ignore_unc=False, total=False):
        if total:
            return max([h.max(divide, variations, ignore_unc) for h in self.histos])
        elif len(self.histos)>0:
            return self.histos[0].max(divide, variations, ignore_unc)
        else: 
            return 0.

    def min(self, divide=None, variations=None, ignore_unc=False, total=False):
        if total:
            return min([h.min(divide, variations, ignore_unc) for h in self.histos])
        elif len(self.histos)>0:
            return self.histos[0].min(divide, variations, ignore_unc)
        else: 
            return 0.

    def plot_stack(self, ax, divide=None, label=False, short_label=False):
        for i,h in enumerate(self.histos):
            h.plot_mc(
                ax, divide=divide,
                baseline=sum([h.nominal for h in self.histos[:i]]),
                label=label, short_label=short_label, fill=True,
                zorder=1-i/(len(self.histos)+1)
            )

