#!/usr/bin/env python3

"""
usage: put under source folder, required files: evolving_state.txt, calib_state.txt, state.txt
After first run, integration_states.txt, vio_states.txt are generated and figures are saved in current dir
You can move the figures and state.txt, integration_states.txt, vio_states.txt into a folder
Rerun to generate graphs more efficiently by specifying the folder names that has the above three files
"""
import inspect
import os
from os import path as osp
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.ticker as mtick

linestyle_algo = {
    "IVL+CNL": "-",
    "IVL": ":",
    "MSE": "--",
    "CNL": "-."
}

linemarks_algo = {
    'Plaintext-Transformer': 'o',
    'BFP-Transformer': '*',
    'Bug2Fix-Transformer': '+',
    'SeuenceR': 'x',
    'BFP-RNN': '^',
    'Bug2Fix': None
}

def read_run_parameters(folder):
    with open(folder + "/config.json", "r") as f:
        parameters = json.load(f)
        return parameters

def load_folder_to_dataframe(file):
    with open(file, "r") as f:
        import pandas as pd
        d = pd.read_csv(f)

        d["name_run"] = osp.basename(file).split("&")[-1][:-4]
        d = d[['name_run', 'dataset', 'seq_name', 'model', 'test_type',
               'traj_rmse', 'ATE', 'T-RTE', 'D-RTE', 'Drift_pos (m/m)',
               'mse_loss_x', 'mse_loss_y', 'mse_loss_avg']]
        return d


def load_folder_dict(ndict):
    l = []
    for file in ndict:
        try:
            l.append(load_folder_to_dataframe(file))
        except:
            print("Could not read from ", ndict)
    dataset_length = [len(el["dataset"].unique()) for el in l]
    nmax_dataset = max(dataset_length)
    dataset_sets = [set(el["dataset"].unique()) for el in l]
    dataset_sets = set.intersection(*dataset_sets)
    if len(dataset_sets) < nmax_dataset:
        print("Some dataset were removed because no result for some run were found.")
        print(
            f"At least one run had {nmax_dataset}. While overlapping dataset vector size is {len(dataset_sets)}"
        )
        input("Press Enter to continue...")
    if len(dataset_sets) == 0:
        print("Could not find any common dataset!")
    for i in range(len(l)):
        l[i] = l[i][l[i].dataset.isin(dataset_sets)]

    d = pd.concat(l)
    d = d.sort_values("name_run")
    return d


def plot_var_boxplot_per(data, var, per="algo"):
    if len(data["name_run"].unique()) == 1:
        sns.boxplot(x=per, y=var, data=data, whis=[0, 1])
        sns.swarmplot(
            x=per, y=var, data=data, color="black", edgecolor="black", dodge=True
        )
    else:
        ax = sns.boxplot(
            x=per,
            y=var,
            hue="name_run",
            data=data,
            palette="Set1",
            whis=1.5,
            showfliers=True,
            fliersize=2,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        # sns.swarmplot(x="algo", y=var, hue="name_run", data=data, palette="Set1", size=1, dodge=True)


def plot_var_cdf(data, var, linestyle=None):
    from cycler import cycler
    if linestyle is None:
        linestyle = linestyle_algo
    ax = plt.gca()
    percentile = np.arange(0, 100.0 / 100, 0.1 / 100.0)
    if len(data["name_run"].unique()) == 1:
        color_algo = {}
        for i, nr in enumerate(data["algo"].unique()):
            color_algo[nr] = "C" + str(i)

        for algo in data["algo"].unique():
            d = data[data.algo == algo][var].quantile(percentile)
            plt.plot(
                d,
                percentile,
                linestyle=linestyle[algo],
                color=color_algo[algo],
                label=f"{algo}",
            )
        plt.xlim(left=0)
        plt.ylim([0, 1])
        plt.ylabel("cdf")
        plt.xlabel(var)
        plt.grid()
    else:
        color_rn = {}
        for i, nr in enumerate(data["name_run"].unique()):
            color_rn[nr] = "C" + str(i)

        for nr in data["name_run"].unique():
            drun = data[data.name_run == nr]
            for algo in data["algo"].unique():
                d = drun[drun.algo == algo][var].quantile(percentile)
                plt.plot(
                    d,
                    percentile,
                    linestyle=linestyle[f"{nr}-{algo}"],
                    color=color_rn[nr],
                    label=f"{nr}-{algo}",
                )
        plt.ylim([0, 1])
        plt.xlim(left=0)
        plt.ylabel("cdf")
        plt.xlabel(var)
        plt.grid()

def plot_all_stats(d, per="algo"):
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["ate", "rpe_rmse_1000", "drift_ratio"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot_per(d, func, per)
        plt.gca().legend().set_visible(False)
    funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot_per(d, func, per)
        plt.gca().legend().set_visible(False)
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
    )
    # plt.savefig('./barplot.svg', bbox_inches='tight')
    plt.show()

    # Plot CDF
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["ate", "rpe_rmse_1000", "drift_ratio"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_cdf(d, func)

    funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_cdf(d, func)
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
    )
    # plt.savefig('./cdfplot.svg', bbox_inches='tight')
    plt.show()

def plot_var_boxplot(data, var):
    if len(data["name_run"].unique()) == 1:
        sns.boxplot(x="algo", y=var, data=data, whis=[0, 1])
        sns.swarmplot(
            x="algo", y=var, data=data, color="black", edgecolor="black", dodge=True
        )
    else:
        sns.boxplot(
            x="algo",
            y=var,
            hue="name_run",
            data=data,
            palette="Set1",
            whis=1.5,
            showfliers=False,
            fliersize=2,
        ).set(xlabel=None)
        # sns.swarmplot(x="algo", y=var, hue="name_run", data=data, palette="Set1", size=1, dodge=True)


def plot_net(data, dataset, model, prefix='net', outdir=None):
    # bar plot
    df = data.copy()
    df = df.loc[(df['dataset'] == dataset) & (df["model"] == model)].rename(
        columns={
            "model": "algo",
            "version": "name_run",
            "ATE": "ATE (m)",
            "T-RTE": "T-RTE (m)",
            "D-RTE": "D-RTE (m)",
            "Drift_pos (m/m)": "DR (%)",
            "traj_rmse": "RMSE of Traj.",
            "mse_loss_avg": "avg MSE loss",
        }
    )

    configs = read_run_parameters(osp.join(outdir, dataset, model))

    df['name_run'].replace(configs['run_config'], inplace=True)

    for test in set(df["test_type"].unique()):
        d = df.loc[df["test_type"] == test]
        if len(d) == 0:
            continue
        fig = plt.figure(figsize=(8, 3), dpi=90)
        funcs = ["ATE (m)", "DR (%)"]
        for i, func in enumerate(funcs):
            plt.subplot2grid([1, len(funcs)], [0, i], fig=fig)
            plot_var_boxplot(d, func)
            plt.legend([])
        fig.tight_layout()
        plt.legend(
            loc='center', bbox_to_anchor=(-0.1, 1.12), fancybox=True,
            shadow=True, ncol=5
        )
        plt.subplots_adjust(hspace=0.12, top=0.86, bottom=0.1, left=0.07, right=0.98)
        if outdir:
            plt.savefig(osp.join(outdir, f"{dataset}_{model}_{test}_net.png"), bbox_inches='tight')
        plt.show()
    del df


def getfunctions(module):
    l = []
    for key, value in module.__dict__.items():
        if inspect.isfunction(value):
            l.append(value)
    return l

def plot_cdf_ax(data, var, ax, fontsize=10, fontname="Adobe Arabic"):
    percentile = np.arange(0, 100.0 / 100, 0.1 / 100.0)
    color_rn = {}
    for i, nr in enumerate(data["name_run"].unique()):
        color_rn[nr] = "C" + str(i)
    for nr in data["name_run"].unique():
        drun = data[data.name_run == nr]
        for algo in drun["algo"].unique():
            d = drun[drun.algo == algo][var].quantile(percentile)
            ax.plot(
                d,
                percentile,
                linestyle=linestyle_algo[nr],
                color=color_rn[nr],
                label=f"{nr}",
            )
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1])
    ax.set_xlabel(var, fontsize=fontsize, fontname=fontname)
    ax.grid()

def plot_comparison_cdf(data, dataset, model, ticksize=16, fontsize=20,
                        tickfont="Crimson Text",
                        fontname="Crimson Text", prefix='cdf', outdir=None):
    df = data.copy()
    df = df.loc[(df['dataset'] == dataset) & (df["model"] == model)].rename(
        columns={
            "model": "algo",
            "version": "name_run",
            "ATE": "ATE (m)",
            "T-RTE": "T-RTE (m)",
            "D-RTE": "D-RTE (m)",
            "Drift_pos (m/m)": "DR (%)",
            "mse_loss_x": "X",
            "mse_loss_y": "Y",
            "mse_loss_avg": "avg MSE loss",
        }
    )

    configs = read_run_parameters(osp.join(outdir, dataset, model))

    df['name_run'].replace(configs['run_config'], inplace=True)


    for test in set(df["test_type"].unique()):
        d = df.loc[df["test_type"] == test]
        if len(d) == 0:
            continue
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), dpi=90)
        funcs = ["ATE (m)", "T-RTE (m)", "D-RTE (m)"]
        for i, func in enumerate(funcs):
            plot_cdf_ax(d, func, axs[i], fontsize=fontsize, fontname=fontname)
        for i in range(3):
            axs[0].set_ylabel("CDF", fontsize=fontsize, fontname=fontname)
            plt.setp(axs[i].get_xticklabels(), fontsize=ticksize, fontname=tickfont)
            plt.setp(axs[i].get_yticklabels(), fontsize=ticksize, fontname=tickfont)

        leg = plt.legend(
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            bbox_transform=fig.transFigure,
            fontsize=fontsize - 3,
        )
        plt.setp(leg.texts, family=fontname)
        plt.subplots_adjust(hspace=0.1, top=0.86, bottom=0.15, left=0.07, right=0.98)
        if outdir:
            plt.savefig(osp.join(outdir, f"{dataset}_{model}_{test}_cdf.png"), bbox_inches='tight')
        plt.show()
    del df


def get_all_files_recursive(data_path):
    condidates = []
    for path, subdirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("csv"):
                configs = path.split("/")[-2:]
                configs.append("CTIN")

                df = pd.read_csv(osp.join(path, file))
                if "seq_name" not in df:
                    df = df.rename(columns={"Unnamed: 0": "seq_name"})
                else:
                    df = df.drop(columns=["Unnamed: 0"])

                df['dataset'] = configs[0]
                df['test_type'] = configs[1]
                df['model'] = configs[2]
                df['version'] = re.split('&|_', file)[-1][:-4]
                condidates.append(df)
    return pd.concat(condidates).fillna(0)

def run_all(df):
    dataset_sets = set(df["dataset"].unique())
    model_sets = set(df["model"].unique())

    for dataset in dataset_sets:
        for model in model_sets:
            plot_net(df, dataset, model, outdir=project_dir, prefix="net")
            plot_comparison_cdf(df, dataset, model, outdir=project_dir, prefix="cdf")



def run_single_dataset(data, outdir, ticksize=16, fontsize=20,
                       tickfont="Crimson Text", fontname="Crimson Text"):
    df = data.copy()
    labels = ['Plaintext-Transformer', 'BFP-Transformer', 'Bug2Fix-Transformer',
              'SeuenceR', 'BFP-RNN', 'Bug2Fix']
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), dpi=90)
    for idx, test in enumerate(['small', 'median', 'big']):
        d = df.loc[df["dataset"] == test]
        if len(d) == 0:
            continue
        plot_ax(d, test, labels, axs[idx], fontsize=fontsize, fontname=fontname)

    for i in range(3):
        axs[0].set_ylabel("Accuracy (%)", fontsize=fontsize, fontname=fontname)
        plt.setp(axs[i].get_xticklabels(), fontsize=ticksize, fontname=tickfont)
        plt.setp(axs[i].get_yticklabels(), fontsize=ticksize, fontname=tickfont)

    leg = plt.legend(
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.523, 1),
        bbox_transform=fig.transFigure,
        fontsize=fontsize - 5,
    )
    plt.setp(leg.texts, family=fontname)
    plt.subplots_adjust(hspace=0.1, top=0.86, bottom=0.15, left=0.07, right=0.98)

    if outdir:
        plt.savefig(osp.join(outdir, f"perf_acc.png"), bbox_inches='tight')
    plt.show()
    del df


def plot_ax(data, var, funcs, ax, fontsize=10, fontname="Adobe Arabic"):
    color_rn = {}
    for i, nr in enumerate(funcs):
        color_rn[nr] = "C" + str(i)

    for fun in funcs:
        ax.plot(
            data['Beam_Size'],
            data[fun] * 100,
            # linestyle=linestyle_algo[nr],
            marker=linemarks_algo[fun],
            color=color_rn[fun],
            label=f"{fun}",
        )
    ax.set_xlim(left=0)
    # ax.set_ylim([0, 1])
    ax.set_xlabel(var, fontsize=fontsize, fontname=fontname)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,
    #                                                     decimals=None,
    #                                                     symbol='%',
    #                                                     is_latex=False))
    ax.grid()

if __name__ == "__main__":
    project_dir = os.getcwd()

    df = pd.read_csv(osp.join(project_dir, "2022-icst_rq1.csv"))

    run_single_dataset(df, project_dir)

