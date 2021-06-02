import pathlib
from functools import partial
from itertools import combinations_with_replacement, combinations
from multiprocessing import Pool

import gin
import numpy as np
import tensorflow as tf
from disentanglement_lib.evaluation.metrics import unsupervised_metrics as dlib_um
from disentanglement_lib.evaluation.metrics import utils as dlib_utils

from tc_study.experiment import logger
from tc_study.experiment.utils import get_variables_idx, get_model_paths, gin_evaluation
from tc_study.utils.tf_config import set_cpu_option


def compute_unsupervised_metrics(idxs_to_keep, cov_mus, mus_discrete, prefix):
    scores = {}
    num_codes = len(idxs_to_keep)
    if num_codes < 2:
        return scores

    if len(idxs_to_keep) == cov_mus.shape[0]:
        cov = cov_mus
        discrete = mus_discrete
    else:
        # Create a covariance matrix and discrete representation containing only the latent factors of interest
        source_idxs = tuple(map(list, (zip(*combinations_with_replacement(idxs_to_keep, 2)))))
        cov = np.zeros((num_codes, num_codes))
        target_idxs = np.triu_indices(num_codes)
        cov[target_idxs] = cov_mus[source_idxs]
        cov = cov.T + cov - np.diag(np.diag(cov))
        discrete = mus_discrete[idxs_to_keep]

    scores["{}.gaussian_total_correlation".format(prefix)] = dlib_um.gaussian_total_correlation(cov)
    scores["{}.gaussian_wasserstein_correlation".format(prefix)] = dlib_um.gaussian_wasserstein_correlation(cov)
    mutual_info_matrix = dlib_utils.discrete_mutual_info(discrete, discrete)
    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes ** 2 - num_codes)
    scores["{}.mutual_info_score".format(prefix)] = mutual_info_score
    if prefix == "full":
        scores["mutual_info_matrix"] = mutual_info_matrix.tolist()

    return scores


@gin.configurable(
    "truncated_unsupervised_metrics",
    blacklist=["ground_truth_data", "representation_function", "random_state", "artifact_dir"])
def truncated_unsupervised_metrics(ground_truth_data, representation_function, random_state, active_variables_idx,
                                   mixed_variables_idx, passive_variables_idx, artifact_dir=None,
                                   num_train=gin.REQUIRED, batch_size=16):
    """Computes total correlation and mutual information scores based on the implementation of unsupervised_metrics
     function from disentanglement lib and evaluate on all possible combinations of passive, mixed, and active variables

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
    representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param active_variables_idx: The indexes of active variables
    :param mixed_variables_idx: The indexes of mixed variables
    :param passive_variables_idx: The indexes of passive variables
    :param artifact_dir: Optional path to directory where artifacts can be saved.
    :param num_train: Number of points used for training.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    mus_train, _ = dlib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                         random_state,
                                                         batch_size)
    num_codes = mus_train.shape[0]
    mus_discrete = dlib_utils.make_discretizer(mus_train)
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    # Compute full scores
    scores.update(compute_unsupervised_metrics(range(num_codes), cov_mus, mus_discrete, "full"))

    # Compute truncated scores combining all possible combinations of variable type
    variables = {"active": active_variables_idx, "passive": passive_variables_idx, "mixed": mixed_variables_idx}
    variables.update({"{}_{}".format(v1, v2): variables[v1] + variables[v2]
                      for v1, v2 in combinations(["active", "mixed", "passive"], 2)
                      if len(variables[v1]) > 0 and len(variables[v2]) > 0})

    for suffix, indexes in variables.items():
        scores.update(compute_unsupervised_metrics(indexes, cov_mus, mus_discrete, suffix))
    scores["num_passive_variables"] = len(passive_variables_idx)
    scores["num_mixed_variables"] = len(mixed_variables_idx)
    scores["num_active_variables"] = len(active_variables_idx)
    scores["correlation_matrix"] = np.corrcoef(cov_mus).tolist()
    scores["covariance_matrix"] = cov_mus.tolist()
    return scores


def compute_truncated_unsupervised_metrics(path, representation, overwrite=True, multiproc=False):
    """Compute unsupervised metrics scores for a given representation

    :param path: Path of the model
    :param representation: type of representation, can be mean or sampled
    :param overwrite: If true, overwrite previous results, otherwise raises an error if previous results exists
    :param multiproc: True if function is executed in multiprocess Pool, else False
    :return: None
    """
    if multiproc is True:
        tf.keras.backend.clear_session()
        set_cpu_option()

    gin_bindings = [
        "evaluation.evaluation_fn = @truncated_unsupervised_metrics",
        "truncated_unsupervised_metrics.num_train = 10000",
        "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20", "evaluation.name = 'truncated unsupervised metrics'",
    ]
    path = pathlib.Path(path)
    result_path = path.parent.parent / "metrics" / representation / "truncated_unsupervised"
    truncation_file = (path.parent.parent / "metrics" / "mean" / "filtered_variables" / "results" / "aggregate"
                       / "evaluation.json")
    idxs = ["truncated_unsupervised_metrics.{} = {}".format(k, v)
            for k, v in get_variables_idx(truncation_file).items()]
    bindings = gin_bindings + idxs
    logger.info("Computing unsupervised metrics of {} using {} representation"
                .format(path.parent.parent, representation))
    gin_evaluation(path, result_path, overwrite, bindings)


def compute_all_truncated_unsupervised_metrics(base_path, representation, model_ids=None, overwrite=True, nb_proc=None):
    model_paths = get_model_paths(base_path, representation, model_ids=model_ids)
    if nb_proc is not None:
        f = partial(compute_truncated_unsupervised_metrics, representation=representation, overwrite=overwrite,
                    multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, model_paths)
    else:
        for path in model_paths:
            compute_truncated_unsupervised_metrics(path, representation, overwrite)
