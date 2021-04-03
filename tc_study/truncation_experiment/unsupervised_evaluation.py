from tc_study.truncation_experiment import utils
import pathlib
import gin
import numpy as np
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils as dlib_utils
from disentanglement_lib.evaluation.metrics import unsupervised_metrics as dlib_unsupervised_metrics

from tc_study.truncation_experiment.utils import get_pv


def unsupervised_metrics(mus_train):
    """Computes unsupervised scores based on covariance and mutual information along with normalised versions.

    :param mus_train: the generated representations
    :return: Dictionary with scores.
    """
    scores = {}
    num_codes = mus_train.shape[0]
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    # Gaussian total correlation.
    scores["gaussian_total_correlation"] = dlib_unsupervised_metrics.gaussian_total_correlation(cov_mus)
    scores["gaussian_total_correlation_norm"] = (scores["gaussian_total_correlation"] / cov_mus)

    # Gaussian Wasserstein correlation.
    scores["gaussian_wasserstein_correlation"] = dlib_unsupervised_metrics.gaussian_wasserstein_correlation(cov_mus)
    scores["gaussian_wasserstein_correlation_norm"] = (
            scores["gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_mus)))

    # Compute average mutual information between different factors.
    mus_discrete = dlib_utils.make_discretizer(mus_train)
    mutual_info_matrix = dlib_utils.discrete_mutual_info(mus_discrete, mus_discrete)

    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes ** 2 - num_codes)
    scores["mutual_info_score"] = mutual_info_score
    return scores


@gin.configurable(
    "truncated_unsupervised_metrics",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def truncated_unsupervised_metrics(ground_truth_data,
                                   representation_function,
                                   random_state,
                                   artifact_dir=None,
                                   num_train=gin.REQUIRED,
                                   batch_size=16):
    """Computes unsupervised scores based on covariance and mutual information along with normalised versions. This
    follows the implementation of unsupervised_metrics function from disentanglement lib but passive variables are
    truncated and TC is normalised.

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
    representation for each observation.
    :param random_state: Numpy random state used for randomness.
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
    mus_train, num_vp = utils.make_truncation(mus_train)

    # If all variable are passive or only one factor is kept, return without calculation
    if mus_train.shape[0] > 1:
        scores = unsupervised_metrics(mus_train)

    scores["num_passive_variables"] = num_vp

    return scores


@gin.configurable(
    "normalized_unsupervised_metrics",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def normalized_unsupervised_metrics(ground_truth_data,
                                    representation_function,
                                    random_state,
                                    artifact_dir=None,
                                    num_train=gin.REQUIRED,
                                    batch_size=16):
    """Computes unsupervised scores based on covariance and mutual information along with normalised versions. This
    follows the implementation of unsupervised_metrics function from disentanglement lib but TC is also normalised.

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
    representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param artifact_dir: Optional path to directory where artifacts can be saved.
    :param num_train: Number of points used for training.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    del artifact_dir
    mus_train, _ = dlib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                         random_state,
                                                         batch_size)
    return unsupervised_metrics(mus_train)


def compute_truncated_unsupervised_metrics(base_path, representation, overwrite=True):
    """Compute unsupervised metrics score for a given representation with its passive variables removed over all models

    :param base_path: Path of the models
    :param representation: type of representation, can be mean or sampled
    :param overwrite: If true, overwrite previous results, otherwise raises an error if previous results exists
    :return: None
    """
    gin_bindings = [
        "evaluation.evaluation_fn = @truncated_unsupervised_metrics",
        "truncated_unsupervised_metrics.num_train = 10000",
        "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20", "evaluation.name = 'truncated unsupervised metrics'",
        "truncation.truncation_fn = @vp_truncation"
    ]
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in range(10800)]

    for path in model_paths:
        path = pathlib.Path(path)
        result_path = path.parent.parent / "metrics" / representation / "truncated_unsupervised"
        truncation_file = (path.parent.parent / "metrics" / "mean" / "passive_variables" / "results" / "aggregate"
                           / "evaluation.json")
        pv_idx, num_pv = get_pv(truncation_file)
        bindings = gin_bindings + ["truncation.pv_idx = {}".format(pv_idx), "truncation.num_pv = {}".format(num_pv)]
        print("Computing truncated unsupervised metrics of {} using {} representation"
              .format(path.parent.parent, representation))
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=bindings)


def compute_normalized_unsupervised_metrics(base_path, representation, overwrite=True):
    """Compute unsupervised metrics score for a given representation over all models

    :param base_path: Path of the models
    :param representation: type of representation, can be mean or sampled
    :param overwrite: If true, overwrite previous results, otherwise raises an error if previous results exists
    :return: None
    """
    gin_bindings = [
        "evaluation.evaluation_fn = @normalized_unsupervised_metrics",
        "normalized_unsupervised_metrics.num_train = 10000",
        "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20", "evaluation.name = 'normalized unsupervised metrics'"
    ]
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in range(10800)]

    for path in model_paths:
        path = pathlib.Path(path)
        result_path = path.parent.parent / "metrics" / representation / "normalized_unsupervised"
        print("Computing normalized unsupervised metrics of {} using {} representation"
              .format(path.parent.parent, representation))
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=gin_bindings)
