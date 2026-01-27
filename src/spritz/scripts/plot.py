import concurrent.futures
import json
import os
import subprocess
import sys
from copy import deepcopy

import matplotlib as mpl
import mplhep as hep
import numpy as np
import uproot
import hist
from spritz.framework.framework import get_analysis_dict, get_fw_path

mpl.use("Agg")
from matplotlib import pyplot as plt

d = deepcopy(hep.style.CMS)

d["font.size"] = 12
d["figure.figsize"] = (5, 5)

d_triplediff = deepcopy(d)
#d_triplediff["font.size"] = 10
d_triplediff["figure.figsize"] = (10, 10)

plt.style.use(d)


def darker_color(color):
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    darker_factor = 4 / 5
    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)


def make_hist(directory, nuisances, sample):
    
    h = directory[f"histo_{sample}"].to_hist()
    histo = {
        'nom': h.values(),
        'stat_up': h.values() + np.sqrt(h.variances()),
        'stat_down': h.values() - np.sqrt(h.variances()),
        'up': np.zeros_like(h.values()),
        'down': np.zeros_like(h.values()),
    }

    for nuisance in nuisances:
        if nuisances[nuisance]["type"] == "rateParam":
            continue
        if nuisances[nuisance]["type"] == "auto":
            continue
        if nuisances[nuisance]["type"] == "stat":
            continue
        name = nuisances[nuisance]["name"]

        if sample not in nuisances[nuisance]["samples"]:
            histo[f"{nuisance}_up"] = h.values().copy()
            histo[f"{nuisance}_down"] = h.values().copy()
            continue

        if nuisances[nuisance]["type"] == "lnN":
            scaling = float(nuisances[nuisance]["samples"][sample])
            histo[f"{nuisance}_up"] = scaling * h.values()
            histo[f"{nuisance}_down"] = 1.0 / scaling * h.values()
        else:
            histo[f"{nuisance}_up"] = directory[f"histo_{sample}_{name}Up"].values().copy()
            histo[f"{nuisance}_down"] = directory[f"histo_{sample}_{name}Down"].values().copy()

        histo["up"] += np.square(histo[f"{nuisance}_up"] - h.values())
        histo["down"] += np.square(histo[f"{nuisance}_down"] - h.values())

    histo["up"] = np.sqrt(histo["up"])
    histo["down"] = np.sqrt(histo["down"])
    
    return histo


def vary_systematics(histos, nuisances, samples):
    v_syst = {
        syst: {
            "up": np.zeros_like(histos[samples[0]]['nom']), 
            "down": np.zeros_like(histos[samples[0]]['nom']),
        }
        for syst in nuisances
    }

    for sample in samples:
        for nuisance in nuisances:
            if sample not in nuisances[nuisance]["samples"]:
                v_syst[nuisance]["up"] += histos[sample]["nom"].copy()
                v_syst[nuisance]["down"] += histos[sample]["nom"].copy()
            else:
                v_syst[nuisance]["up"] += histos[sample][nuisance + "_up"].copy()
                v_syst[nuisance]["down"] += histos[sample][nuisance + "_down"].copy()

    return v_syst


def sum_hist(histos, nuisances, samples):
    v_syst = vary_systematics(histos, nuisances, samples)
    
    nom = sum([histos[sample]['nom'] for sample in samples])
    up = np.sqrt(sum([np.square(v_syst[nuis]['up'] - nom) for nuis in nuisances]))
    down = np.sqrt(sum([np.square(v_syst[nuis]['down'] - nom) for nuis in nuisances]))
    
    return {'nom': nom, 'up': up, 'down': down}


def plot_data(ax, x, y, zorder, color='black', label='Data', divide=None):
    if divide is None:
        divide = np.ones_like(x)
    ax.errorbar(
        x,
        y['nom']/divide,
        yerr=(y['down']/divide, y['up']/divide),
        fmt="o",
        markersize=4,
        label=f"{label} [{int(round(np.sum(y['nom']), 0))}]",
        zorder=zorder,
        color=color
    )


def plot_mc_stack(ax, edges, histos, colors, divide=None):
    if divide is None:
        divide = np.ones_like(edges[1:])
    for i,sample in enumerate(histos.keys()):
        y = histos[sample]['nom'].copy()
        ax.stairs(
            (y+sum([histos[sample]['nom'] for sample in list(histos.keys())[:i]]))/divide,
            edges,
            label=sample + f" [{int(round(np.sum(y), 0))}]",
            fill=True,
            zorder=-i,
            color=colors[sample],
            edgecolor=darker_color(colors[sample]),
            linewidth=1.0,
        )


def plot_mc(ax, edges, y, zorder, label='Tot MC', unc=True, fill=False, color='darkgrey', divide=None):
    if divide is None:
        divide = np.ones_like(edges[1:])
    if unc:
        unc_up = round(np.sum(y['up']) / np.sum(y['nom']) * 100, 2)
        unc_down = round(np.sum(y['down']) / np.sum(y['nom']) * 100, 2)
        ax.stairs(
            (y['nom']+y['up'])/divide,
            edges,
            baseline=(y['nom']-y['down'])/divide,
            label=f"Syst [-{unc_down}, +{unc_up}]%",
            fill=True,
            hatch="///",
            color=darker_color(color),
            facecolor="none",
            zorder=zorder+1,
        )

    ax.stairs(
        y['nom']/divide, 
        edges, 
        label=f"{label} [{int(round(np.sum(y['nom']), 0))}]", 
        color=color,
        fill=fill,
        zorder=zorder
    )


def plot_ratio(ax, x, edges, numerator, denominator, yrange=None, printout=False, full_range=False, increased_range=False):
    denominator['nom'] = np.where(denominator['nom'] >= 1e-6, denominator['nom'], 1e-6)
    # plot denominator
    ax.stairs(
        1+denominator['up']/denominator['nom'],
        edges,
        baseline=1-denominator['down']/denominator['nom'],
        fill=True,
        color=denominator.get('color','black'),
        alpha=0.25,
        zorder=-2*(len(numerator)+1)
    )
    ax.plot(
        edges, 
        np.ones_like(edges), 
        color=denominator.get('color','black'), 
        linestyle="dashed",
        zorder=-2*len(numerator)-1
    )

    # plot ratio
    if len(numerator)>1:
        offsets = np.linspace(-0.3, 0.3, len(numerator))
    else:
        offsets = np.array([0.0])
    
    res_dict = {}

    for i in range(len(numerator)):
        edge, x_i = 0., x.copy()
        for j in range(len(x)):
            width = x[j]-edge
            edge = x[j]+width
            x_i[j] = x[j]+offsets[i]*width
        
        if numerator[i].get('is_data'):
            ax.errorbar(
                np.array(x_i),
                numerator[i]['nom']/denominator['nom'],
                yerr=(numerator[i]['down']/denominator['nom'], numerator[i]['up']/denominator['nom']),
                fmt="o",
                markersize=4,
                color=numerator[i].get('color','black'),
                zorder=-2*len(numerator)
            )
        else:
            # uncertainty
            ax.stairs(
                (numerator[i]['nom']+numerator[i]['up'])/denominator['nom'],
                edges,
                baseline=(numerator[i]['nom']-numerator[i]['down'])/denominator['nom'],
                fill=True, 
                color=numerator[i]['color'],
                alpha=0.25,
                zorder=-2*len(numerator)+i,
            )
            # histogram
            ax.stairs(
                numerator[i]['nom']/denominator['nom'], 
                edges, 
                fill=False,
                edgecolor=numerator[i]['color'],
                linewidth=1.0,
                zorder=-2*len(numerator)+1+i,
            )
        if printout:
            print(
                json.dumps({
                    'edges': list(edges),
                    'nom': list(numerator[i]['nom']/denominator['nom']),
                    'up': list(np.sqrt(np.square(numerator[i]['down'])+np.square(denominator['up']))/denominator['nom']),
                    'down': list(np.sqrt(np.square(numerator[i]['up'])+np.square(denominator['down']))/denominator['nom'])
                })
            )

    # get envelope
    envelope = {'up': np.ones_like(denominator['nom']), 'down': np.ones_like(denominator['nom'])}
    for i in range(len(denominator['nom'])):
        for j in range(len(numerator)):
            if numerator[j]['nom'][i]/denominator['nom'][i] > envelope['up'][i]:
                envelope['up'][i] = numerator[j]['nom'][i]/denominator['nom'][i]
            if 0 < numerator[j]['nom'][i]/denominator['nom'][i] < envelope['down'][i]:
                envelope['down'][i] = numerator[j]['nom'][i]/denominator['nom'][i]

    # set yrange
    ymin = min(envelope['down'])
    ymax = max(envelope['up'])
    if yrange is not None:
        ax.set_ylim(*yrange)
    elif full_range:
        ax.set_ylim(0.9*min(1,ymin), 1.1*max(1,ymax))
    elif increased_range:
        ax.set_ylim(0.9*min(1,ymin), 1.1*max(1,min(5,ymax)))
    else:
        ylim = min(1,1.15*max(abs(1-ymax),abs(1-ymin)))
        ax.set_ylim(min(1-ylim,0.94), max(1+ylim,1.06))

    ax.set_xlim(np.min(edges), np.max(edges))


def plot(
    input_file,
    region,
    variable,
    weights,
    samples,
    nuisances,
    lumi,
    colors,
    year_label,
    variable_label=None,
    plotFakes=False
):
    print("Doing ", region, variable)

    directory = input_file[f"{region}/{variable}/nominal"]
    mc_samples = [x for x in samples if not samples[x].get('is_data', False) and not x in ['W+Jets','QCD','TTToSemiLeptonic']]
    mcfakes_samples = [x for x in samples if x in ['W+Jets','QCD','TTToSemiLeptonic']]
    bkg_samples = [x for x in mc_samples if not samples[x].get('is_signal', False)]
    
    # get the histograms
    histos = {
        sample: make_hist(directory, nuisances, sample)
        for sample in samples
    }

    # prepare total MC histogram
    histo_mc = sum_hist(histos, nuisances, mc_samples)

    if plotFakes and not '_ss' in region:
        directory_ss = input_file[f"{region}_ss/{variable}/nominal"]
        histos_ss = {
            sample: make_hist(directory_ss, nuisances, sample)
            for sample in samples
        }
        histo_mc_ss = sum_hist(histos_ss, nuisances, mc_samples)
        
        if 'Data' in histos_ss:
            histo_data_ss = {
                    'nom': histos_ss['Data']['nom'],
                    'up': histos_ss['Data']['stat_up'] - histos_ss['Data']['nom'],
                    'down': histos_ss['Data']['nom'] - histos_ss['Data']['stat_down']
                }
        else:
            histo_data_ss = {
                'nom': np.zeros_like(histo_mc_ss['nom']),
                'up': np.zeros_like(histo_mc_ss['nom']),
                'down': np.zeros_like(histo_mc_ss['nom'])
            }

        # subtract MC from data to get fakes
        histo_fakes = {
            'nom': histo_data_ss['nom']-histo_mc_ss['nom'],
            'up': np.sqrt(np.square(histo_data_ss['up'])+np.square(histo_mc_ss['down'])),
            'down': np.sqrt(np.square(histo_data_ss['down'])+np.square(histo_mc_ss['up']))
        }

        histo_mc['nom'] = histo_mc['nom'] + histo_fakes['nom']
        histo_mc['up'] = np.sqrt(np.square(histo_mc['up'])+np.square(histo_fakes['up']))
        histo_mc['down'] = np.sqrt(np.square(histo_mc['down'])+np.square(histo_fakes['down']))

    # prepare data histogram
    if 'Data' in histos:
        histo_data = {
            'nom': histos['Data']['nom'],
            'up': histos['Data']['stat_up'] - histos['Data']['nom'],
            'down': histos['Data']['nom'] - histos['Data']['stat_down']
        }
    else:
        histo_data = {
            'nom': np.zeros_like(histo_mc['nom']),
            'up': np.zeros_like(histo_mc['nom']),
            'down': np.zeros_like(histo_mc['nom'])
        }

    # set up figure
    axis = directory[f"histo_{list(samples.keys())[0]}"].to_hist().axes[0]
    x = axis.centers
    edges = axis.edges
    widths = axis.widths
    variable_binwidth = isinstance(axis, hist.axis.Variable)

    if variable=='triple_diff':
        plt.style.use(d_triplediff)
        fig, ax = plt.figure(dpi=200), list()
        gs = mpl.gridspec.GridSpec(
            12, 1,
            figure=fig,
            height_ratios=[2,1,0.25,2,1,0.25,2,1,0.25,2,1,0.25]
        )
        for i in range(4):
            ax.append(fig.add_subplot(gs[3*i]))
            ax.append(fig.add_subplot(gs[3*i+1], sharex=ax[2*i]))
        fig.tight_layout(pad=-1.3)

    else:
        plt.style.use(d)
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,1]}, dpi=200)
        fig.tight_layout(pad=-0.5)
    
    hep.cms.label(region, data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)
    ry0,ry1 = 0.8,1.2
    y0 = max(1e-4 if variable_binwidth else 0.5, 0.5*np.min((histo_fakes['nom'] if plotFakes and not '_ss' in region else histos[mc_samples[0]]['nom'])/(widths if variable_binwidth else np.ones_like(histos[mc_samples[0]]['nom']))))
    y1 = max(np.max(histo_data['nom']/widths if variable_binwidth else histo_data['nom']), np.max(histo_mc['nom']/widths if variable_binwidth else histo_mc['nom'])) * 1e3

    def plots(axes,hdata,hmc,hmc_tot,x,edges,widths,ncols_legend=3):
        # plot data
        plot_data(
            axes[0],
            x, 
            hdata,
            zorder=2,
            divide=widths if variable_binwidth else None
        )

        # plot MC
        plot_mc_stack(
            axes[0], 
            edges, 
            hmc,
            colors,
            divide=widths if variable_binwidth else None
        )

        # plot total MC and uncertainty
        plot_mc(
            axes[0], 
            edges, 
            hmc_tot,
            zorder=1,
            divide=widths if variable_binwidth else None
        )

        # finalize upper panel
        axes[0].set_yscale("log")
        axes[0].legend(
            loc="upper center",
            frameon=True,
            ncols=ncols_legend,
            framealpha=0.8,
            fontsize=8,
        )

        axes[0].set_ylim(y0, y1)
        axes[0].set_ylabel("Events/(bin width)" if variable_binwidth else "Events")
        axes[0].tick_params(labelbottom=False)
        axes[0].set_xlabel("")

        # lower panel
        plot_ratio(
            axes[1], 
            x, 
            edges, 
            [hdata | {'is_data': True, 'color': 'black'}],
            hmc_tot,
            yrange=(ry0,ry1)
        )

        axes[1].set_ylabel("DATA / MC")
        axes[1].tick_params(labelbottom=False)
        axes[1].set_xlabel("")

    if variable=='triple_diff':
        ylls = [
                '$0 < |y_{\\ell\\ell}|$ / $|y_{\\ell\\ell}^{max}| < 0.2$',
                '$0.2 < |y_{\\ell\\ell}|$ / $|y_{\\ell\\ell}^{max}| < 0.4$',
                '$0.4 < |y_{\\ell\\ell}|$ / $|y_{\\ell\\ell}^{max}| < 0.6$',
                '$0.6 < |y_{\\ell\\ell}|$ / $|y_{\\ell\\ell}^{max}| < 1$',
            ]
        bbox = {'boxstyle':'round', 'alpha':0.8, 'fc':'white'} 
        for isubplot in range(4):
            ax[2*isubplot].set_title(ylls[isubplot], loc='center', fontsize=10)
            histo_data_ = {'nom': histo_data['nom'][50*isubplot:50*(isubplot+1)],
                'up': histo_data['up'][50*isubplot:50*(isubplot+1)],
                'down': histo_data['down'][50*isubplot:50*(isubplot+1)]}
            histo_mc_ = {'nom': histo_mc['nom'][50*isubplot:50*(isubplot+1)],
                'up': histo_mc['up'][50*isubplot:50*(isubplot+1)],
                'down': histo_mc['down'][50*isubplot:50*(isubplot+1)]}
            histo_mcs_ = ({'Fakes':{'nom':histo_fakes['nom'][50*isubplot:50*(isubplot+1)]}} if plotFakes and not '_ss' in region else {}) | {sample:{'nom':histos[sample]['nom'][50*isubplot:50*(isubplot+1)]} for sample in mc_samples}
            widths_ = widths[50*isubplot:50*(isubplot+1)]
            x_ = x[0:50]
            edges_ = edges[0:51]

            plots(
                (ax[2*isubplot],ax[2*isubplot+1]),
                histo_data_,
                histo_mcs_,
                histo_mc_,
                x_,
                edges_,
                widths_,
                ncols_legend=5
            )
            
            ax[2*isubplot].vlines(
                [0,10,20,30,40], 
                y0, 
                y1,
                colors='black')

            ax[2*isubplot+1].vlines(
                [0,10,20,30,40], 
                ry0, 
                ry1,
                colors='black')

            thetas = [
                '$-1 < \\cos \\theta^{\\ast} < -0.6$',
                '$-0.6 < \\cos \\theta^{\\ast} < -0.2$',
                '$-0.2 < \\cos \\theta^{\\ast} < 0.2$',
                '$0.2 < \\cos \\theta^{\\ast} < 0.6$',
                '$0.6 < \\cos \\theta^{\\ast} < 1$'

            ]

            ipanel, npanels = 0, len(thetas)

            for theta in thetas:
                ax[2*isubplot].text(ipanel/npanels+0.007, 0.06,
                    theta, 
                    fontsize=8,
                    transform=ax[2*isubplot].transAxes,
                    bbox=bbox
                )
                ipanel+=1

    else:
        plots(
            (ax[0],ax[1]),
            histo_data,
            ({'Fakes':histo_fakes} if plotFakes and not '_ss' in region else {}) | {sample:histos[sample] for sample in mc_samples},
            histo_mc,
            x,
            edges,
            widths
        )
        ax[-1].tick_params(labelbottom=True)

    if variable_label:
        ax[-1].set_xlabel(variable_label)
    else:
        ax[-1].set_xlabel(variable)
    
    fig.savefig(
        f"plots/{region}_{variable}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )

    plt.close()

    for weight in weights:
        directory = input_file[f"{region}/{variable}/{weight}"]
        histos_weight = {
            sample: make_hist(directory, nuisances, sample)
            for sample in mc_samples
        }

        # prepare total MC histogram
        histo_mc_weight = sum_hist(histos_weight, nuisances, mc_samples)

        # set up figure
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,2]}, dpi=200)
        fig.tight_layout(pad=-0.5)
        hep.cms.label(region, data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)

        # plot
        plot_data(
            ax[0], 
            x,
            histo_data,
            zorder=0,
            divide=widths if variable_binwidth else None
        )
        plot_mc(
            ax[0], 
            edges, 
            histo_mc_weight,
            zorder=-2, 
            label=f'Before {weight}', 
            unc=False, 
            fill=False, 
            color='blue',
            divide=widths if variable_binwidth else None
        )
        plot_mc(
            ax[0], 
            edges, 
            histo_mc,
            zorder=-1, 
            label=f'After {weight}', 
            unc=False, 
            fill=False, 
            color='red',
            divide=widths if variable_binwidth else None
        )

        # finalize upper panel
        ax[0].set_yscale("log")
        ax[0].legend(
            loc="upper center",
            frameon=True,
            ncols=2,
            framealpha=0.8,
            fontsize=8,
        )
        ax[0].set_ylim(
            max(0.5, np.min(histo_mc['nom']/widths if variable_binwidth else histo_mc['nom']) / 5), 
            np.max(histo_mc['nom']/widths if variable_binwidth else histo_mc['nom']) * 50
        )
        ax[0].set_ylabel("Events/(bin width)" if variable_binwidth else "Events")

        # lower panel
        plot_ratio(
            ax[1], 
            x, 
            edges, 
            [histo_mc_weight | {'color': 'blue'}, histo_mc | {'color': 'red'}], 
            histo_data
        )

        ax[1].set_ylabel("MC / DATA")
        if variable_label:
            ax[1].set_xlabel(variable_label)
        else:
            ax[1].set_xlabel(variable)

        fig.savefig(
            f"plots/corrections/{region}_{variable}_{weight}.png",
            facecolor="white",
            pad_inches=0.1,
            bbox_inches="tight",
        )
        
        plt.close()


def main():
    analysis_dict = get_analysis_dict()
    samples = analysis_dict["samples"]

    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]
    nuisances = analysis_dict["nuisances"]
    check_weights = list(analysis_dict["check_weights"].keys())

    colors = analysis_dict["colors"]
    plot_label = analysis_dict["plot_label"]
    year_label = analysis_dict.get("year_label", "Run-II")
    lumi = analysis_dict["lumi"]
    print("Doing plots")

    proc = subprocess.Popen(
        "mkdir -p plots && mkdir -p plots/corrections && " + f"cp {get_fw_path()}/data/common/index.php plots/",
        shell=True,
    )
    proc.wait()

    # FIXME add nuisance for stat
    nuisances["stat"] = {
        "name": "stat",
        "type": "stat",
        "samples": dict((skey, "1.00") for skey in samples),
    }

    plotFakes = len(sys.argv) > 1 and sys.argv[1] == "--fakes"

    cpus = 15

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = []

        input_file = uproot.open("histos.root")
        for region in regions:
            for variable in variables:
                if "axis" not in variables[variable]:
                    continue
                tasks.append(
                    executor.submit(
                        plot,
                        input_file,
                        region,
                        variable,
                        check_weights,
                        samples,
                        nuisances,
                        lumi,
                        colors,
                        year_label,
                        variables[variable].get("label"),
                        plotFakes
                    )
                )
        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()


if __name__ == "__main__":
    main()
