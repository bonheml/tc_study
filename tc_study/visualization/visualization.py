import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_figure(out_fname, dpi=300):
    plt.savefig(out_fname, dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()


def draw_truncated_scores(in_fname, out_path, y_label="gaussian_total_correlation_norm"):
    """ Generate and save a line plot of mean and sampled unsupervised score for a given model and dataset with one
    hyper-parameter of increasing value.

    :param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param y_label: label to use for y axis. Can be gaussian_total_correlation, gaussian_wasserstein_correlation,
    gaussian_wasserstein_correlation_norm, or mutual_info_score. Default is gaussian_total_correlation_norm
    :return: None
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    out_fname = "{}/truncated_{}_{}_{}_plot.pdf".format(out_path, y_label, df.model.iloc[0], df.dataset.iloc[0])
    x_labels = ["beta", "c_max", "gamma", "lambda_od"]
    x_label = list(df.columns.intersection(x_labels))[0]
    g = sns.catplot(data=df, x=x_label, y=y_label, col="truncated", hue="representation", ci=None, kind="point",
                    linestyles=["-", "--"], markers=["o", "x"], dodge=True)
    g.set_axis_labels(x_label.replace("_", " ").capitalize(), y_label.replace("_", " ").capitalize())

    plt.tight_layout()
    save_figure(out_fname)


def draw_all_truncated_scores(in_path, out_path, y_label):
    all_files = glob.glob("{}/*.tsv".format(in_path))
    for file in all_files:
        draw_truncated_scores(file, out_path, y_label)


def draw_tc_vp(in_fname, out_path, y_label="gaussian_total_correlation"):
    """ Generate and save a line plot of an unsupervised metric score along with an histogram of passive variables for
    a given model and dataset with one hyper-parameter of increasing value.

    ::param in_fname: The tsv in_fname containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param y_label: the metric to use, default is normalized gaussian TC
    :return: None
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    out_fname = "{}/passive_variables_{}_{}_{}_plot.pdf".format(out_path, y_label, df.model.iloc[0], df.dataset.iloc[0])
    x_labels = ["beta", "c_max", "gamma", "lambda_od"]
    x_label = list(df.columns.intersection(x_labels))[0]
    fig, ax1 = plt.subplots()

    sns.barplot(data=df, x=x_label, y="num_passive_variables", palette="viridis", ax=ax1, ci=None, alpha=0.5)
    ax2 = ax1.twinx()
    sns.pointplot(data=df, x=x_label, y=y_label, ax=ax2, ci=None, hue="representation", linestyles=["-", "--"],
                  markers=["o", "x"])

    ax1.set_xlabel(x_label.replace("_", " ").capitalize())
    ax1.set_ylabel("Average number of passive variables")
    ax2.set_ylabel(y_label.replace("_", " ").capitalize())

    fig.tight_layout()

    save_figure(out_fname)


def draw_all_tc_vp(in_path, out_path, y_label):
    all_files = glob.glob("{}/*.tsv".format(in_path))
    for file in all_files:
        draw_tc_vp(file, out_path, y_label)


def draw_downstream_task_reg(in_fname, out_path, metric="gaussian_total_correlation",
                             ds_task="logistic_regression_1000", representation="mean"):
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
    df = df[df.truncated == False]
    df = df[df.representation == representation]
    out_fname = "{}/{}_{}_{}_{}_{}_plot.pdf".format(out_path, ds_task, metric, df.model.iloc[0], df.dataset.iloc[0],
                                                    representation)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x_labels = ["beta", "c_max", "gamma", "lambda_od"]
    x_label = list(df.columns.intersection(x_labels))[0]
    sns.pointplot(data=df, x=x_label, y=metric, ax=ax1)
    sns.pointplot(data=df, x=x_label, lines="---", y=ds_task, ax=ax2)
    x_label = x_label.replace("_", " ").capitalize()
    ax1.set(xlabel=x_label, ylabel=metric.replace("_", " ").capitalize())
    ax2.set(xlabel=x_label, ylabel=ds_task.replace("_", " ").capitalize())
    fig.tight_layout()

    save_figure(out_fname)


def draw_all_downstream_task_reg(in_path, out_path):
    all_files = glob.glob("{}/*.tsv".format(in_path))
    for file in all_files:
        draw_downstream_task_reg(file, out_path)
