import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tc_study.visualization.utils import get_variables_combinations

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})


def save_figure(out_fname, dpi=300):
    plt.savefig(out_fname, dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()


def draw_truncated_scores(in_fname, out_path, y_label="gaussian_total_correlation", ci=None):
    """ Generate and save a line plot of mean and sampled unsupervised score for a given model and dataset with one
    hyper-parameter of increasing value.

    :param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param y_label: label to use for y axis. Can be gaussian_total_correlation, gaussian_wasserstein_correlation,
    gaussian_wasserstein_correlation_norm, or mutual_info_score. Default is gaussian_total_correlation
    :param ci: the error bar to use in graph. If None, no error bar will appear.
    :return: None
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    model = df.model.iloc[0]
    dataset = df.dataset.iloc[0]
    if ci:
        out_fname = "{}/truncated_{}_{}_{}_sd_plot.pdf".format(out_path, y_label, model, dataset)
    else:
        out_fname = "{}/truncated_{}_{}_{}_plot.pdf".format(out_path, y_label, model, dataset)
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]
    g = sns.catplot(data=df, x=x_label, y=y_label, col="combination", hue="representation", ci=ci, kind="point",
                    linestyles=["-", "--"], markers=["o", "x"], legend_out=False, dodge=False, col_wrap=2)
    g.set_axis_labels(x_label.replace("_", " ").capitalize(), y_label.replace("_", " ").capitalize())

    plt.tight_layout()
    save_figure(out_fname)


def draw_synthetic_scores(in_fname, out_path, y_label="gaussian_total_correlation", ci=None):
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    params = df.cov_min.unique()[0], df.cov_max.unique()[0]
    param_name = "cov"
    base_fname = "{}/truncated_factors_{}_{}_{}_{}_{}_noise_{}".format(out_path, y_label, df.num_factors.unique()[0],
                                                                       param_name, params[0], params[1],
                                                                       df.noise_strength.unique()[0])
    out_fname = "{}_sd_plot.pdf".format(base_fname) if ci else "{}_plot.pdf".format(base_fname)
    df["passive_variables"] = df.num_factors - df.active_variables

    g = sns.relplot(data=df, x="passive_variables", y=y_label, col="combination", hue="representation",
                    style="representation", kind="line", col_wrap=2)
    g.set_axis_labels("Passive variables", y_label.replace("_", " ").capitalize())

    plt.tight_layout()
    save_figure(out_fname)


def draw_all_truncated_scores(in_path, out_path, y_label, global_res_file="global_results.tsv"):
    in_path = in_path.rstrip("/")
    glob_file = "{}/{}".format(in_path, global_res_file)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        draw_truncated_scores(file, out_path, y_label)
        draw_truncated_scores(file, out_path, y_label, ci="sd")


def draw_all_synthetic_scores(in_path, out_path, y_label):
    in_path = in_path.rstrip("/")
    all_files = glob.glob("{}/*.tsv".format(in_path))
    for file in all_files:
        draw_synthetic_scores(file, out_path, y_label)
        draw_truncated_scores(file, out_path, y_label, ci="sd")


def draw_tc_vp(in_fname, out_path, y_label="gaussian_total_correlation", ci=None):
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

    if ci:
        out_fname = "{}/passive_variables_{}_{}_{}_sd_plot.pdf".format(out_path, y_label, model, dataset)
    else:
        out_fname = "{}/passive_variables_{}_{}_{}_plot.pdf".format(out_path, y_label, model, dataset)
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]
    fig, ax1 = plt.subplots()

    sns.barplot(data=df, x=x_label, y="num_passive_variables", palette="viridis", ax=ax1, alpha=0.5)
    ax2 = ax1.twinx()
    sns.pointplot(data=df, x=x_label, y=y_label, ax=ax2, ci=ci, hue="representation", linestyles=["-", "--"],
                  markers=["o", "x"])

    ax1.set_xlabel(x_label.replace("_", " ").capitalize())
    ax1.set_ylabel("Passive variables (averaged)")
    ax2.set_ylabel(y_label.split(".")[1].replace("_", " ").capitalize())

    fig.tight_layout()

    save_figure(out_fname)


def draw_all_tc_vp(in_path, out_path, y_label, global_res_file="global_results.tsv"):
    glob_file = "{}/{}".format(in_path, global_res_file)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        draw_tc_vp(file, out_path, y_label)
        # draw_tc_vp(file, out_path, y_label, ci="sd")


def draw_downstream_task_reg(in_fname, out_path, metric="full.gaussian_total_correlation",
                             ds_task="full.logistic_regression_1000", representation="mean"):
    """ Generate and save a line plot of unsupervised comparison score and downstream task score for a given model and
    dataset with one hyper-parameter of increasing value.

    :param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param metric: the metric to use, default is gaussian TC
    :param ds_task: the downstream task to plot
    :param representation: the representation, either mean or sampled. default is mean
    :return:
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    df = df[df.representation == representation]
    out_fname = "{}/{}_{}_{}_{}_{}_plot.pdf".format(out_path, ds_task, metric, df.model.iloc[0], df.dataset.iloc[0],
                                                    representation)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x_labels = ["beta", "c_max", "gamma", "lambda_od", "tc_beta"]
    x_label = list(df.columns.intersection(x_labels))[0]
    sns.pointplot(data=df, x=x_label, y=metric, ax=ax1)
    sns.pointplot(data=df, x=x_label, lines="---", y=ds_task, ax=ax2)
    x_label = x_label.replace("_", " ").capitalize()
    ax1.set(xlabel=x_label, ylabel=metric.replace("_", " ").capitalize())
    ax2.set(xlabel=x_label, ylabel=ds_task.replace("_", " ").capitalize())
    fig.tight_layout()

    save_figure(out_fname)


def draw_all_downstream_task_reg(in_path, out_path, metric="gaussian_total_correlation",
                                 ds_task="logistic_regression_1000", global_res_file="global_results.tsv"):
    glob_file = "{}/{}".format(in_path, global_res_file)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    all_files.remove(glob_file)
    for file in all_files:
        for comb in get_variables_combinations():
            draw_downstream_task_reg(file, out_path, metric="{}.{}".format(comb, metric),
                                     ds_task="{}.{}".format(comb, ds_task))

