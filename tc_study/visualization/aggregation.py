from disentanglement_lib.utils import aggregate_results
from tc_study.utils.models import get_model_info
from tc_study.utils.string_manipulation import remove_suffix, remove_prefix


def aggregate_scores(model_info, base_path, representation, metric="normalized_unsupervised"):
    """ Aggregate the unsupervised metrics scores of models matching model_info and returns a dataframe with the scores,
    dataset, representation, model type, and hyper-parameters used.

    :param model_info: A Series containing the dataset and model type to use along with model indexes matching the
    (model type, dataset) pair.
    :param base_path: The path containing the models to retrieve
    :param representation: The representation to use. Can be mean or sampled.
    :param metric: The metric to aggregate, can be unsupervised or truncated_unsupervised
    :return: A dataframe containing the TC scores of models matching <model_info> along with the dataset, representation,
    model index, model type and hyper-parameter values.
    """
    m = remove_suffix(remove_suffix(remove_prefix(metric, "downstream_task_"), "_sklearn"), "_metric")
    to_keep = {"evaluation_results.gaussian_total_correlation": "gaussian_total_correlation",
               "evaluation_results.gaussian_wasserstein_correlation": "gaussian_wasserstein_correlation",
               "evaluation_results.gaussian_wasserstein_correlation_norm": "gaussian_wasserstein_correlation_norm",
               "evaluation_results.mutual_info_score": "mutual_info_score",
               "evaluation_results.num_passive_variables": "num_passive_variables",
               "evaluation_results.10:mean_test_accuracy": "{}_10".format(m),
               "evaluation_results.100:mean_test_accuracy": "{}_100".format(m),
               "evaluation_results.1000:mean_test_accuracy": "{}_1000".format(m),
               "evaluation_results.10000:mean_test_accuracy": "{}_10000".format(m),
               "postprocess_config.dataset.name": "dataset",
               "postprocess_config.postprocess.name": "representation",
               "train_config.model.model_num": "model_index",
               "train_config.model.name": "model"}

    model_params = {"train_config.dip_vae.lambda_od": "lambda_od",
                    "train_config.beta_tc_vae.beta": "beta",
                    "train_config.factor_vae.gamma": "gamma",
                    "train_config.annealed_vae.c_max": "c_max",
                    "train_config.vae.beta": "beta"}

    all_cols = dict(to_keep, **model_params)
    result_file_pattern = ["{}/{}/metrics/{}/{}/results/aggregate/evaluation.json"
                           .format(base_path, i, representation, metric)
                           for i in range(model_info.start_idx, model_info.end_idx + 1)]
    df = aggregate_results._get(result_file_pattern)
    df = df[df.columns.intersection(all_cols.keys())]
    df = df.rename(columns=all_cols, errors="ignore")
    df[["representation", "model", "dataset"]] = df[["representation", "model", "dataset"]].replace({"'": ""},
                                                                                                    regex=True)
    return df


def aggregate_all_scores(base_path, out_path):
    """ Aggregate and save unsupervised scores per (model_type, dataset) pair from
    pre-trained models from "Challenging Common Assumptions in the Unsupervised Learning
    of Disentangled Representations" (http://proceedings.mlr.press/v97/locatello19a.html).

    :param base_path: path where the models will be saved. If models are already saved there, no download will be done.
    :param out_path: path where the aggregated TC scores will be saved along with their graph
    :return: None
    """
    models_info = get_model_info()

    # Models trained on shapes3d are not released so removing this part
    models_info = models_info[models_info.dataset != "shapes3d"]

    for i, model_info in models_info.iterrows():
        print("Aggregating unsupervised scores of {} on {}".format(model_info.model, model_info.dataset))
        df = aggregate_scores(model_info, base_path, "sampled")
        df = df.append(aggregate_scores(model_info, base_path, "mean"), ignore_index=True)
        pv = aggregate_scores(model_info, base_path, "mean", "passive_variables").drop(columns=["representation"])
        df = df.merge(pv)
        df["truncated"] = False
        df2 = aggregate_scores(model_info, base_path, "sampled", "truncated_unsupervised")
        df2 = df2.append(aggregate_scores(model_info, base_path, "mean", "truncated_unsupervised"), ignore_index=True)
        df2["truncated"] = True
        df = df.append(df2, ignore_index=True)
        df.to_csv("{}/{}_{}.tsv".format(out_path, model_info.model, model_info.dataset), sep="\t", index=False)
        del df, df2



