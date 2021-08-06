import glob
import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tc_study.visualization import logger
from tc_study.visualization.utils import get_variables_combinations, get_model_files

#sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def save_figure(out_fname, dpi=300):
    plt.savefig(out_fname, dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()


def draw_truncated_scores(in_fname, out_path, y_label="full.gaussian_total_correlation", ci=None):
    """ Generate and save a line plot of mean and sampled unsupervised score for a given model and dataset with one
    hyper-parameter of increasing value.

    :param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param y_label: label to use for y axis. Can be gaussian_total_correlation, or mutual_info_score along with any
    active/mixed/passive combination.
    Default is full gaussian_total_correlation
    :param ci: the error bar to use in graph. If None, no error bar will appear.
    :return: None
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    model = df.model.iloc[0]
    dataset = df.dataset.iloc[0]
    if y_label not in df.columns:
        return
    text_y_label = "_".join(y_label.split("."))
    if ci:
        out_fname = "{}/{}_{}_{}_sd_plot.pdf".format(out_path, text_y_label, model, dataset)
    else:
        out_fname = "{}/{}_{}_{}_plot.pdf".format(out_path, text_y_label, model, dataset)
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]

    g = sns.catplot(data=df, x=x_label, y=y_label, hue="representation", ci=ci, kind="point",
                    linestyles=["-", "--"], markers=["o", "x"], legend_out=False, dodge=False)
    g.set_axis_labels(x_label.replace("tc_", "").replace("_", " ").capitalize(), text_y_label.replace("_", " ").capitalize())
    plt.tight_layout()
    save_figure(out_fname)

def draw_all_truncated_scores(in_path, out_path, y_label, global_res_file="global_results.tsv"):
    in_path = in_path.rstrip("/")
    glob_file = "{}/{}".format(in_path, global_res_file)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        for comb in get_variables_combinations():
            draw_truncated_scores(file, out_path, y_label="{}.{}".format(comb, y_label))
            # draw_truncated_scores(file, out_path,  y_label="{}.{}".format(comb, y_label), ci="sd")


def draw_tc_vp(in_fname, out_path, y_label="full.gaussian_total_correlation", ci=None):
    """ Generate and save a line plot of an unsupervised metric score along with an histogram of passive variables for
    a given model and dataset with one hyper-parameter of increasing value.

    :param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param y_label: the metric to use, default is normalized gaussian TC
    :param ci: the error bar to use in graph. If None, no error bar will appear.
    :return: None
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    model = df.model.iloc[0]
    dataset = df.dataset.iloc[0]
    text_y_label = "_".join(y_label.split("."))
    if ci:
        out_fname = "{}/passive_variables_{}_{}_{}_sd_plot.pdf".format(out_path, text_y_label, model, dataset)
    else:
        out_fname = "{}/passive_variables_{}_{}_{}_plot.pdf".format(out_path, text_y_label, model, dataset)
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]
    fig, ax1 = plt.subplots()

    sns.barplot(data=df, x=x_label, y="num_passive_variables", palette="viridis", ax=ax1, alpha=0.5, ci=None)
    ax2 = ax1.twinx()
    sns.pointplot(data=df, x=x_label, y=y_label, ax=ax2, ci=ci, hue="representation", linestyles=["-", "--"],
                  markers=["o", "x"])

    ax1.set_xlabel(x_label.replace("tc_", "").replace("_", " ").capitalize())
    ax1.set_ylabel("Passive variables (averaged)")
    ax2.set_ylabel(y_label.split(".")[1].replace("_", " ").capitalize())

    fig.tight_layout()

    save_figure(out_fname)


def draw_all_tc_vp(in_path, out_path, y_label, global_res_file="global_results.tsv"):
    glob_file = "{}/{}".format(in_path, global_res_file)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        draw_tc_vp(file, out_path, y_label="full.{}".format(y_label))
        # draw_tc_vp(file, out_path, y_label, ci="sd")


def draw_stacked_variables_count(in_fname, out_path):
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    model = df.model.iloc[0]
    dataset = df.dataset.iloc[0]
    out_fname = "{}/variables_types_{}_{}_plot.pdf".format(out_path, model, dataset)
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]
    df["num_variables"] = 10
    df["passive_mixed"] = df[["num_passive_variables", "num_mixed_variables"]].sum(axis=1)

    fig, ax = plt.subplots()
    active, mixed, passive = sns.color_palette()[:3]
    total_bar = sns.barplot(x=x_label, y="num_variables", data=df, color=active, ci=None)
    mixed_passive_bar = sns.barplot(x=x_label, y="passive_mixed", data=df, color=mixed,  ci=None)
    passive_bar = sns.barplot(x=x_label, y="num_passive_variables", data=df, color=passive, ci=None)

    top_bar = mpatches.Patch(color=active, label="Active variables")
    middle_bar = mpatches.Patch(color=mixed, label="Mixed variables")
    bottom_bar = mpatches.Patch(color=passive, label="Passive variables")
    plt.legend(handles=[top_bar, middle_bar, bottom_bar], loc="lower right")

    ax.set_xlabel(x_label.replace("tc_", "").replace("_", " ").capitalize())
    ax.set_ylabel("Number of variables (averaged)")

    fig.tight_layout()
    save_figure(out_fname)


def draw_all_stacked_variable_counts(in_path, out_path, global_res_file="global_results.tsv"):
    glob_file = "{}/{}".format(in_path, global_res_file)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        draw_stacked_variables_count(file, out_path)
        # draw_tc_vp(file, out_path, y_label, ci="sd")


def draw_downstream_task_reg(in_fname, out_path, ds_task, representation="mean"):
    """ Generate and save a line plot of downstream task score for different variable types of a given model and
    dataset with one hyper-parameter of increasing value.

    :param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param ds_task: the downstream task to plot
    :param representation: the representation, either mean or sampled. default is mean
    :return:
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    df = df[df.representation == representation]
    out_fname = "{}/{}_{}_{}_{}_plot.pdf".format(out_path, ds_task, df.model.iloc[0], df.dataset.iloc[0], representation)
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]
    combs = ["{}.mean_test_accuracy".format(c) for c in ["random", *get_variables_combinations()]]
    combs = list(df.columns.intersection(combs))
    to_keep = {"model_index", "representation", x_label}
    to_drop = set(df.columns.values) - set(combs) - to_keep
    df = df.drop(to_drop, axis=1)
    df = df.melt(id_vars=list(to_keep), value_vars=combs, var_name="combination", value_name="accuracy")
    df["combination"] = df["combination"].apply(lambda x: x.split(".")[0])
    fig, ax = plt.subplots()

    sns.lineplot(data=df, x=x_label, y="accuracy", hue="combination", style="combination", dashes=False, markers=True, palette='Dark2', ci=None, alpha=0.6, ms=18)
    fig.tight_layout()

    save_figure(out_fname)


def draw_all_downstream_task_reg(in_path, out_path, ds_task="logistic_regression_10000", global_res_file="global_results.tsv"):
    glob_file = "{}/{}".format(in_path, global_res_file).replace("//", "/")
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        draw_downstream_task_reg(file, out_path, ds_task)
        draw_downstream_task_reg(file, out_path, ds_task, "sampled")


def draw_model_histograms(in_fname, out_path, representation="mean"):
    with open(in_fname, "r") as f:
        res = json.load(f)
    discretized = res["evaluation_results.hist"]
    for i in range(len(discretized)):
        fig, ax = plt.subplots()
        out_fname = "{}/histogram_{}_z{}.pdf".format(out_path, representation, i)
        counts, bins = discretized[i]
        ax.hist(bins[:-1], bins, weights=counts)
        fig.tight_layout()
        save_figure(out_fname)
        plt.close(fig)
        plt.cla()
        plt.clf()


def draw_all_histograms(in_path, out_path, representation="mean"):
    model_files, model_ids = get_model_files(in_path, representation)
    for model, i in zip(model_files, model_ids):
        out_model = "{}/{}".format(out_path, i)
        os.makedirs(out_model, exist_ok=True)
        logger.info("Computing {} histogram for model {}".format(representation, i))
        draw_model_histograms(model, out_model, representation)
