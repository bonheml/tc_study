import multiprocessing

import numpy as np
import pandas as pd
from disentanglement_lib.utils.aggregate_results import _load
from tensorflow._api.v1.compat.v1 import gfile

from tc_study.visualization import logger
from tc_study.utils.models import get_model_info
from tc_study.utils.string_manipulation import remove_prefix
from tc_study.visualization.utils import get_variables_combinations


def get_files(pattern):
    files = gfile.Glob(pattern)
    with multiprocessing.Pool() as pool:
        try:
            all_results = pool.map(_load, files)
        except Exception as e:
            pool.close()
            raise e
    return pd.DataFrame(all_results)


def aggregate_scores(model_info, base_path, representation, metric="truncated_unsupervised"):
    """ Aggregate the unsupervised metrics scores of models matching model_info and returns a dataframe with the scores,
    dataset, representation, model type, and hyper-parameters used.

    :param model_info: A Series containing the dataset and model type to use along with model indexes matching the
    (model type, dataset) pair.
    :param base_path: The path containing the models to retrieve
    :param representation: The representation to use. Can be mean or sampled.
    :param metric: The metric to aggregate, can be truncated_unsupervised (default), downstream_task_logistic_regression, or
    :return: A dataframe containing the TC scores of models matching <model_info> along with the dataset, representation,
    model index, model type and hyper-parameter values.
    """
    m = remove_prefix(metric, "truncated_downstream_task_")
    comb = get_variables_combinations()
    to_keep = {"postprocess_config.dataset.name": "dataset",
               "postprocess_config.postprocess.name": "representation",
               "train_config.model.model_num": "model_index",
               "train_config.model.name": "model",
               "evaluation_results.passive_variables_idx": "passive_variables_idx",
               "evaluation_results.mixed_variables_idx": "mixed_variables_idx",
               "evaluation_results.active_variables_idx": "active_variables_idx"}
    for c in comb:
        to_keep += {
            "evaluation_results.{}.gaussian_total_correlation".format(c): "{}.gaussian_total_correlation".format(c),
            "evaluation_results.{}.mutual_info_score".format(c): "{}.mutual_info_score".format(c),
            "evaluation_results.{}.10:mean_test_accuracy".format(c): "{}.{}_10".format(c, m),
            "evaluation_results.{}.100:mean_test_accuracy".format(c): "{}.{}_100".format(c, m),
            "evaluation_results.{}.1000:mean_test_accuracy".format(c): "{}.{}_1000".format(c, m),
            "evaluation_results.{}.10000:mean_test_accuracy".format(c): "{}.{}_10000".format(c, m)}

    model_params = {"train_config.dip_vae.lambda_od": "lambda_od",
                    "train_config.beta_tc_vae.beta": "tc_beta",
                    "train_config.factor_vae.gamma": "gamma",
                    "train_config.annealed_vae.c_max": "c_max",
                    "train_config.vae.beta": "beta"}

    all_cols = dict(to_keep, **model_params)
    result_file_pattern = ["{}/{}/metrics/{}/{}/results/aggregate/evaluation.json"
                           .format(base_path, i, representation, metric)
                           for i in range(model_info.start_idx, model_info.end_idx + 1)]
    df = get_files(result_file_pattern)
    existing_cols = df.columns.intersection(all_cols.keys())
    df = df[existing_cols]
    df = df.rename(columns=all_cols, errors="ignore")
    df[["representation", "model", "dataset"]] = df[["representation", "model", "dataset"]].replace({"'": ""},
                                                                                                    regex=True)
    col_ids = [c for c in existing_cols.tolist() if "." not in c]
    df = pd.melt(df, id_vars=col_ids, var_name="combination")
    df[['combination', 'metric']] = df['combination'].str.split('.', 1, expand=True)
    df = df.pivot_table(index=["model_index", "combination"], columns=["metric"])
    df = df.droplevel(0, axis=1).rename_axis(None, axis=1).reset_index()
    return df


def get_aggregated_version(df_list):
    model_params = ["lambda_od", "beta", "gamma", "c_max", "tc_beta"]
    norm_scores = np.linspace(0, 1, 6)
    df = pd.concat(df_list)
    df["regularization"] = np.nan
    for p in model_params:
        df[p] = df[p].replace(df[p].dropna().unique(), norm_scores)
        df["regularization"] = df["regularization"].fillna(df[p])
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

    # Removing unused dip-vae-i model
    models_info = models_info[models_info.model != "dip_vae_i"]
    df_list = []

    for i, model_info in models_info.iterrows():
        logger("Aggregating unsupervised scores of {} on {}".format(model_info.model, model_info.dataset))
        df = aggregate_scores(model_info, base_path, "sampled")
        df = df.append(aggregate_scores(model_info, base_path, "mean"), ignore_index=True)
        df2 = aggregate_scores(model_info, base_path, "mean", "truncated_downstream_task_logistic_regression")
        df2 = df2.append(aggregate_scores(model_info, base_path, "sampled", "truncated_downstream_task_logistic_regression"))
        df = df.merge(df2)
        df.to_csv("{}/{}_{}.tsv".format(out_path, model_info.model, model_info.dataset), sep="\t", index=False)
        df_list.append(df)
        del df2
    main_df = get_aggregated_version(df_list)
    main_df.to_csv("{}/global_results.tsv".format(out_path), sep="\t", index=False)
