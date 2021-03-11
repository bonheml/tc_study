import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_graph(in_fname, out_path, y_label="gaussian_total_correlation"):
    """ Generate and save a line plot of mean and sampled unsupervised score for a given model type with one
    hyper-parameter of increasing value.

    :param in_fname: The tsv file containing unsupervised scores of models along with the dataset, representation,
    model index, model type and hyper-parameter values.
    :param out_path: path where the pdf graph will be saved
    :param y_label: label to use for y axis. Can be gaussian_total_correlation, gaussian_wasserstein_correlation,
    gaussian_wasserstein_correlation_norm, or mutual_info_score. Default is gaussian_total_correlation
    :return: None
    """
    df = pd.read_csv(in_fname, index_col=None, header=0, sep="\t")
    out_fname = "{}/{}_{}_{}_plot.pdf".format(out_path, y_label, df.model.iloc[0], df.dataset.iloc[0])
    x_labels = ["beta", "c_max", "gamma", "lambda_od"]
    x_label = list(df.columns.intersection(x_labels))[0]
    sns.lineplot(data=df, x=x_label, y=y_label, hue="representation", style="representation")
    plt.savefig(out_fname, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


def draw_all(in_path, out_path, y_label="gaussian_total_correlation"):
    fname = "{}/{}_aggregated_plot.pdf".format(out_path, y_label)
    all_files = glob.glob("{}/*.tsv".format(in_path))
    fig, axes = plt.subplots(6, 6, figsize=(36, 36))

    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep="\t")
        df_list.append(df)
    df = pd.concat(df_list, axis=0, ignore_index=True)

    models = list(df.model.unique())
    datasets = list(df.dataset.unique())
    pad = 10
    for ax, col in zip(axes[0], models):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 5], datasets):
        ax2 = ax.twinx()
        ax2.set_ylabel(row, labelpad=pad, size='large')
        ax2.yaxis.set_ticks([])

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            draw_one(df[(df.dataset == dataset) & (df.model == model)], axes[i, j], i == 5, j == 0, y_label)
    fig.tight_layout()
    plt.savefig(fname, dpi=300)


def draw_one(df, ax, set_x_label, set_y_label, y_label):
    df = df.dropna(axis=1, how="all")
    x_labels = ["beta", "c_max", "gamma", "lambda_od"]
    x_label = list(df.columns.intersection(x_labels))[0]
    g = sns.lineplot(ax=ax, data=df, x=x_label, y=y_label, hue="representation", style="representation")
    if set_x_label is False:
        g.set(xlabel=None)
    if set_y_label is False:
        g.set(ylabel=None)

# Todo: Implement histogram plots showing number of passive variable along with TCs
# def test():
#   fig = plt.figure(figsize=(10, 5))
#   ax1 = fig.add_subplot(111)
#   ax2 = ax1.twinx()
#   sns.barplot(x='Bin', y='Frequency', data=myDF, color='blue', ax=ax1)
#
#   sns.lineplot(x=myDF.index, y='Cumulative', data=myDF, marker='s', color='orange', ax=ax2)
