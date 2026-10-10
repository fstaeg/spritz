import hist
import numpy as np
import pandas as pd
import matplotlib as mpl

unc_colors = ["purple","red","green"] + list(mpl.colors.TABLEAU_COLORS.values())

def darker_color(color):
    if color is None:
        return None
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    darker_factor = 4 / 5
    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)


class Histogram(object):

    def __init__(self, name, data, axis=None, nuisances={}, nuisance_groups={}, corrections={}, is_data=False, color="black", linestyle=None):
        self.name = name
        self.data = data
        self.axis = axis
        self.nuisances = nuisances
        self.nuisance_groups = nuisance_groups
        self.corrections = corrections

        self.is_data = is_data
        self.color = color
        self.linestyle = linestyle

        self.divide = self.widths if self.variable_width else np.ones_like(self.widths)

        self.cache = {}

    @classmethod
    def make_hist(cls, directory, sample, nuisances={}, corrections={}, label=None, is_data=False, color="black", linestyle=None):
        name = sample if label is None else label

        nominal = directory[f"histo_{sample}"]#.to_hist()
        axis = nominal.axes[0]
        values = nominal.values()
        variances = nominal.variances()

        columns, data = ["nominal"], [values]
        _nuisances, _nuisance_groups, _corrections = {}, {}, {}

        # statistical
        columns.append("variances")
        data.append(variances)
        _nuisances["stat"] = {"kind": "stat"}
        
        # nuisances
        for tag,nuisance in nuisances.items():
            _kind, _variations, _data = cls.load_nuisance(directory, sample, tag, nuisance, values)
            if _kind is not None and _kind=="group":
                _nuisance_groups[tag] = _variations
            elif _kind is not None:
                _nuisances[tag] = {"kind": _kind, "variations": _variations}
                columns.extend(_variations)
                data.extend(_data)

        # corrections
        for tag,correction in corrections.items():
            _data = cls.load_correction(directory, sample, correction, values)
            if _data is not None:
                _corrections[tag] = {"variation": f"{tag} Before"}
                columns.append(f"{tag} Before")
                data.append(_data)

        data = pd.DataFrame(data, columns)

        return cls(name, data, axis, _nuisances, _nuisance_groups, _corrections, is_data, color, linestyle)

    @staticmethod
    def load_nuisance(directory, sample, tag, nuisance, nominal):
        name, samples = nuisance.get("name"), nuisance.get("samples")
        type, kind = nuisance.get("type"), nuisance.get("kind")

        data, variations = list(), list()

        if kind is None:
            kind = type

        if kind in ["envelope", "square", "stdev"] and sample in samples:
            for i,var in enumerate(nuisance["variations"]):
                #variations.append(var["label"])
                variations.append(f"{name}_{i}")
                data.append(directory[f"histo_{sample}_{name}_{i}"].values()-nominal)

        elif kind == "weight" and sample in samples:
            variations.extend([f"{tag} Up", f"{tag} Down"])
            data.extend([
                directory[f"histo_{sample}_{name}Up"].values()-nominal,
                directory[f"histo_{sample}_{name}Down"].values()-nominal
            ])

        elif kind == "lnN" and sample in samples:
            scaling = float(nuisance["samples"][sample])
            variations.extend([f"{tag} Up", f"{tag} Down"])
            data.extend([(scaling-1)*nominal, (1/scaling-1)*nominal])

        elif kind == "group":
            return kind, nuisance["nuisances"], []

        else:
            return None, [], []

        return kind, variations, data

    @staticmethod
    def load_correction(directory, sample, correction, nominal):
        name, samples = correction.get("name"), correction.get("samples")
        if sample in samples:
            return directory[f"histo_{sample}_{name}Before"].values()-nominal
        else:
            return None

    @classmethod
    def empty_like(cls, histo, **kwargs):
        name = kwargs.get("name", histo.name)
        is_data = kwargs.get("is_data", histo.is_data)
        color = kwargs.get("color", histo.color)
        linestyle = kwargs.get("linestyle", histo.linestyle)
        
        axis = histo.axis
        columns = ["nominal", "variances"]
        data = pd.DataFrame(np.zeros_like(histo.data.loc[columns]), columns)
        nuisances = {"stat": {"kind": "stat"}}
        nuisance_groups = {}
        corrections = {}

        return cls(name, data, axis, nuisances, nuisance_groups, corrections, is_data, color, linestyle)

    @classmethod
    def new_like(cls, histo, **kwargs):
        name = kwargs.get("name", histo.name)
        is_data = kwargs.get("is_data", histo.is_data)
        color = kwargs.get("color", histo.color)
        linestyle = kwargs.get("linestyle", histo.linestyle)
        
        axis = histo.axis
        data = histo.data.copy()
        nuisances = histo.nuisances.copy()
        nuisance_groups = histo.nuisance_groups.copy()
        corrections = histo.corrections.copy()

        return cls(name, data, axis, nuisances, nuisance_groups, corrections, is_data, color, linestyle)

    def __getitem__(self, key):
        if isinstance(key, slice):
            edges = self.edges[key.start:key.stop+1]
            if self.variable_width:
                axis = hist.axis.Variable(edges, name=self.axis.name)
            else:
                axis = hist.axis.Regular(len(edges)-1, edges[0], edges[-1], name=self.axis.name)

            data = self.data.loc[:, key.start:key.stop-1]

            return Histogram(self.name, data, axis, self.nuisances, self.nuisance_groups,
                self.corrections, self.is_data, self.color, self.linestyle)
        
        elif key in self.data.index:
            return self.data.loc[key].to_numpy().copy()

        elif key == "stat Up":
            return np.sqrt(self.data.loc["variances"]).to_numpy().copy()

        elif key == "stat Down":
            return -np.sqrt(self.data.loc["variances"]).to_numpy().copy()

        elif key.replace(" Up", "").replace(" Down", "") in self.nuisance_groups:
            if "Up" in key:
                nuisances = self.nuisance_groups[key.replace(" Up", "")]
                return self.up(nuisances)
            elif "Down" in key:
                nuisances = self.nuisance_groups[key.replace(" Down", "")]
                return self.down(nuisances)
        
        else:
            return getattr(self, key)

    @property
    def nominal(self): return self.data.loc["nominal"].to_numpy().copy()

    @property
    def centers(self): return self.axis.centers

    @property
    def edges(self): return self.axis.edges

    @property
    def widths(self): return self.axis.widths

    @property
    def variable_width(self): return isinstance(self.axis, hist.axis.Variable)

    @property
    def integral(self):
        if "integral" not in self.cache:
            self.cache["integral"] = np.sum(self.nominal)
        return self.cache["integral"]

    def max(self, nuisances=None, divide=None): 
        if nuisances is None: nuisances = list(self.nuisances.keys())
        else: nuisances = [nuis for nuis in nuisances if nuis in self.nuisances]
        key = tuple(sorted(nuisances))
        
        if divide is None:
            divide = self.divide
            if ("max", key) not in self.cache:
                up = self.nominal+self.up(nuisances)
                self.cache[("max", key)] = np.max(up/divide)
            return self.cache[("max", key)]
        else:
            up = self.nominal+self.up(nuisances)
            return np.max(up/divide)

    def min(self, nuisances=None, divide=None): 
        if nuisances is None: nuisances = list(self.nuisances.keys())
        else: nuisances = [nuis for nuis in nuisances if nuis in self.nuisances]
        key = tuple(sorted(nuisances))
        
        if divide is None:
            divide = self.divide
            if ("min", key) not in self.cache:
                down = self.nominal+self.down(nuisances)
                self.cache[("min", key)] = np.min(down/divide)
            return self.cache[("min", key)]
        else:
            down = self.nominal+self.down(nuisances)
            return np.min(down/divide)

    def nuis_up(self, nuis):
        if ("nuis_up", nuis) not in self.cache:
            kind = self.nuisances[nuis].get("kind")
            variations = self.nuisances[nuis].get("variations", "variances")
            data = self.data.loc[variations]
            if kind == "stat":
                res = np.sqrt(data).to_numpy()
            elif kind == "envelope":
                up = np.max(data, axis=0)
                res = np.max((up, np.zeros_like(up)), axis=0)
            elif kind == "square":
                res = np.sqrt(np.sum(data**2, axis=0)).to_numpy()
            elif kind == "stdev":
                res = np.std(data, axis=0).to_numpy()
            else:
                res = data.loc[f"{nuis} Up"].to_numpy()
            self.cache[("nuis_up", nuis)] = res

        return self.cache[("nuis_up", nuis)]

    def nuis_down(self, nuis):
        if ("nuis_down", nuis) not in self.cache:
            kind = self.nuisances[nuis].get("kind")
            variations = self.nuisances[nuis].get("variations", "variances")
            data = self.data.loc[variations]
            if kind == "stat":
                res = -np.sqrt(data).to_numpy()
            elif kind == "envelope":
                down = np.min(data, axis=0)
                res = np.min((down, np.zeros_like(down)), axis=0)
            elif kind == "square":
                res = -np.sqrt(np.sum(data**2, axis=0)).to_numpy()
            elif kind == "stdev":
                res = -np.std(data, axis=0).to_numpy()
            else:
                res = data.loc[f"{nuis} Down"].to_numpy()
            self.cache[("nuis_down", nuis)] = res
        
        return self.cache[("nuis_down", nuis)]

    def up(self, nuisances=None):
        if nuisances is None: nuisances = list(self.nuisances.keys())
        else: nuisances = [nuis for nuis in nuisances if nuis in self.nuisances]
        key = tuple(sorted(nuisances))

        if ("up", key) not in self.cache:  
            up = np.zeros(self.data.shape[1])
            for nuis in nuisances:
                if self.nuisances[nuis].get("kind") == "weight":
                    nuis_up = np.max((self.nuis_up(nuis), self.nuis_down(nuis)), axis=0)
                else:
                    nuis_up = self.nuis_up(nuis)
                up += nuis_up**2
            self.cache[("up", key)] = np.sqrt(up)
        
        return self.cache[("up", key)]

    def down(self, nuisances=None):
        if nuisances is None: nuisances = list(self.nuisances.keys())
        else: nuisances = [nuis for nuis in nuisances if nuis in self.nuisances]
        key = tuple(sorted(nuisances))
        
        if ("down", key) not in self.cache:  
            down = np.zeros(self.data.shape[1])
            for nuis in nuisances:
                if self.nuisances[nuis].get("kind") == "weight":
                    nuis_down = np.min((self.nuis_up(nuis), self.nuis_down(nuis)), axis=0)
                else:
                    nuis_down = self.nuis_down(nuis)
                down += nuis_down**2
            self.cache[("down", key)] = -np.sqrt(down)
        
        return self.cache[("down", key)]

    def add(self, other):
        assert self.axis == other.axis

        nuisances = {**self.nuisances, **other.nuisances}
        nuisance_groups = {**self.nuisance_groups, **other.nuisance_groups}
        corrections = {**self.corrections, **other.corrections}

        self.data = self.data.add(other.data, fill_value=0.)
        self.nuisances = nuisances
        self.nuisance_groups = nuisance_groups
        self.corrections = corrections
        self.cache = {}

    def sub(self, other):
        assert self.axis == other.axis

        nuisances = {**self.nuisances, **other.nuisances}
        nuisance_groups = {**self.nuisance_groups, **other.nuisance_groups}
        corrections = {**self.corrections, **other.corrections}

        self.data = self.data.sub(other.data, fill_value=0.)
        variances = self.data.loc["variances"].add(other.data.loc["variances"])
        self.data.loc["variances"] = variances
        self.nuisances = nuisances
        self.nuisance_groups = nuisance_groups
        self.corrections = corrections
        self.cache = {}

    def set_axis(self, axis):
        assert len(axis.centers) == len(self.axis.centers)
        self.axis = axis
        self.divide = self.widths if self.variable_width else np.ones_like(self.widths)

    def plot_setup(self, **kwargs):
        show_label = kwargs.get("show_label", False)
        short_label = kwargs.get("short_label", False)
        if show_label:
            label = kwargs.get("label", self.name)
            if not short_label:
                label += f" [{round(self.integral, 0)}]"
        else:
            label = None

        linestyle = kwargs.get("linestyle")
        if linestyle is None:
            linestyle = self.linestyle if self.linestyle else "solid"

        return {
            "label": label,
            "linestyle": linestyle,
            "linewidth": kwargs.get("linewidth", 1),
            "color": kwargs.get("color", self.color),
            "fill": kwargs.get("fill", False),
            "alpha": kwargs.get("alpha", 1),
            "zorder": kwargs.get("zorder", 1),
        }

    def plot_data(self, ax, variation=None, divide=None, slice_bins=None, nuisances=None, **kwargs):
        x, y = self.centers, self.nominal
        up, down = self.up(nuisances), self.down(nuisances)
        
        if variation is not None:
            y += self[variation]

        if divide is None:
            _divide = self.divide
        else:
            _divide = divide

        if slice_bins is not None:
            new_axis, bins = slice_bins
            x = new_axis.centers
            y = y[bins.start:bins.stop]
            up = up[bins.start:bins.stop]
            down = down[bins.start:bins.stop]
            if divide is None:
                _divide = new_axis.widths
            else:
                _divide = _divide[bins.start:bins.stop]

        kwargs = self.plot_setup(**kwargs)
        label, color = kwargs["label"], kwargs["color"]

        ax.errorbar(
            y=y/_divide, yerr=(-down/_divide, up/_divide), x=x, 
            label=label, color=color, fmt="o", markersize=4
        )

    def plot_mc(self, ax, variation=None, baseline=None, divide=None, slice_bins=None, **kwargs):
        x, y = self.edges, self.nominal
        
        if variation is not None:
            y += self[variation]
        
        if baseline is not None:
            y += baseline
        
        if divide is None:
            _divide = self.divide
        else:
            _divide = divide
            y = np.where(y > 1e-6, y, 1e-6)

        if slice_bins is not None:
            new_axis, bins = slice_bins
            x = new_axis.edges
            y = y[bins.start:bins.stop]
            if divide is None:
                _divide = new_axis.widths
            else:
                _divide = _divide[bins.start:bins.stop]

        kwargs = self.plot_setup(**kwargs)
        label, color = kwargs["label"], kwargs["color"]
        alpha, fill, zorder = kwargs["alpha"], kwargs["fill"], kwargs["zorder"]
        linewidth, linestyle = kwargs["linewidth"], kwargs["linestyle"]
        baseline = 0 if fill else None
    
        ax.stairs(
            values=y/_divide, edges=x, baseline=baseline, 
            label=label, color=color, edgecolor=darker_color(color), alpha=alpha, 
            fill=fill, zorder=zorder, linewidth=linewidth, linestyle=linestyle
        )

    def plot_mc_unc(self, ax, variation=None, divide=None, slice_bins=None, nuisances=None, highlight_nuis=None, **kwargs):
        x, y = self.edges, self.nominal
        up, down = self.up(nuisances), self.down(nuisances)
        
        if variation is not None:
            y += self[variation]
        
        if divide is None:
            _divide = self.divide
        else:
            _divide = divide

        if slice_bins is not None:
            new_axis, bins = slice_bins
            x = new_axis.edges
            y = y[bins.start:bins.stop]
            up = up[bins.start:bins.stop]
            down = down[bins.start:bins.stop]
            if divide is None:
                _divide = new_axis.widths
            else:
                _divide = _divide[bins.start:bins.stop]

        show_label = kwargs.get("show_label_unc", False)
        color = kwargs.get("color", self.color)

        if show_label: 
            label = kwargs.get("label_unc", "Syst")
            unc_up = round(np.sum(up) / self.integral * 100, 1)
            unc_down = round(np.sum(down) / self.integral * 100, 1)
            label += f" [{unc_down}, +{unc_up}]%"
        else:
            label = None

        ax.stairs(
            values=(y+up)/_divide, baseline=(y+down)/_divide, edges=x, 
            label=label, color=color, alpha=0.25, fill=True
        )

        if highlight_nuis is not None:
            up, down = self.up(highlight_nuis), self.down(highlight_nuis)
            if slice_bins is not None:
                up = up[bins.start:bins.stop]
                down = down[bins.start:bins.stop]
            if show_label:
                label = kwargs.get("label_highlight", highlight_nuis[0])
                unc_up = round(np.sum(up) / self.integral * 100, 1)
                unc_down = round(np.sum(down) / self.integral * 100, 1)
                label += f" [{unc_down}, +{unc_up}]%"
            
            ax.stairs(
                values=(y+up)/_divide, baseline=(y+down)/_divide, edges=x, 
                label=label, color=unc_colors[0], alpha=0.25, fill=True
            )


class StackedHistogram(object):

    def __init__(self, histos):
        self.histos = histos
        self.axis = histos[0].axis
        
        for h in self.histos:
            assert h.axis == self.axis

        self.divide = self.widths if self.variable_width else np.ones_like(self.widths)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return StackedHistogram(
                histos=[h[key.start:key.stop] for h in self.histos]
            )
        elif isinstance(key, int):
            return self.histos[key]
        else:
            return getattr(self, key)

    @property
    def nominal(self): 
        nominal = self.histos[0].nominal
        for h in self.histos[1:]:
            nominal += h.nominal
        return nominal

    @property
    def centers(self): return self.axis.centers

    @property
    def edges(self): return self.axis.edges

    @property
    def widths(self): return self.axis.widths

    @property
    def variable_width(self): return isinstance(self.axis, hist.axis.Variable)

    def max(self, nuisances=[], divide=None):
        if divide is None:
            divide = self.divide
        return np.max(self.nominal/divide)

    def min(self, nuisances=[], divide=None, total=False):
        if total:
            if divide is None:
                divide = self.divide
            return np.min(self.nominal/divide)
        else:
            return self.histos[0].min(nuisances, divide)

    def sum(self, name=None, color="black", linestyle=None):
        sum_hist = Histogram.new_like(self.histos[0], name=name, color=color, linestyle=linestyle)
        for h in self.histos[1:]:
            sum_hist.add(h)
        return sum_hist

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
        self.divide = self.widths if self.variable_width else np.ones_like(self.widths)

    def plot_stack(self, ax, divide=None, slice_bins=None, **kwargs):
        baseline = np.zeros_like(self.axis.centers)
        for i,h in enumerate(self.histos):
            h.plot_mc(
                ax, divide=divide, baseline=baseline, slice_bins=slice_bins, fill=True,
                zorder=1-i/(len(self.histos)+1), **kwargs
            )
            baseline += h.nominal

