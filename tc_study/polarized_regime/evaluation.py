from tc_study.polarized_regime import utils
import pathlib
import gin
import numpy as np
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils as dislib_utils
from disentanglement_lib.evaluation.metrics import unsupervised_metrics


@gin.configurable(
    "passive_variables_idx",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def passive_variables_indexes(ground_truth_data,
                              representation_function,
                              random_state,
                              artifact_dir=None,
                              num_train=gin.REQUIRED,
                              threshold=0.1,
                              batch_size=16):
    """Get indexes of passive variable by checking which variables have a covariance lower than threshold with
    themselves.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.

    Returns:
      Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    mus_train, _ = dislib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                           random_state,
                                                           batch_size)
    num_codes = mus_train.shape[0]
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    pv_idx = np.where(np.diag(cov_mus) < threshold)[0].tolist()
    num_pv = len(pv_idx)
    scores["passive_variables_idx"] = pv_idx
    scores["num_passive_variables"] = num_pv
    return scores


def compute_passive_variable_indexes(base_path, overwrite=True):
    gin_bindings = [
        "evaluation.evaluation_fn = @passive_variables_idx",
        "passive_variables_idx.num_train = 10000", "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "evaluation.name = 'passive variables'"
    ]
    model_paths = ["{}/{}/postprocessed/mean".format(base_path, i) for i in range(10800)]

    for path in model_paths:
        path = pathlib.Path(path)
        result_path = path.parent.parent / "metrics" / "mean" / "passive_variables"
        print("Computing passive variable indexes of {}".format(path.parent.parent))
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=gin_bindings)


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
    """Computes unsupervised scores based on covariance and mutual information after having removed passive variables

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.

    Returns:
      Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    mus_train, _ = dislib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                           random_state,
                                                           batch_size)
    mus_train, num_vp = utils.make_truncation(mus_train)
    num_codes = mus_train.shape[0]

    scores["num_passive_variables"] = num_vp

    # If all variable are passive or only one factor is kept, return without calculation
    if num_codes <= 1:
        return scores

    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    # Gaussian total correlation.
    scores["gaussian_total_correlation"] = unsupervised_metrics.gaussian_total_correlation(cov_mus)

    # Gaussian Wasserstein correlation.
    scores["gaussian_wasserstein_correlation"] = unsupervised_metrics.gaussian_wasserstein_correlation(cov_mus)
    scores["gaussian_wasserstein_correlation_norm"] = (
            scores["gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_mus)))

    # Compute average mutual information between different factors.
    mus_discrete = dislib_utils.make_discretizer(mus_train)
    mutual_info_matrix = dislib_utils.discrete_mutual_info(mus_discrete, mus_discrete)

    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes ** 2 - num_codes)
    scores["mutual_info_score"] = mutual_info_score
    return scores


def compute_truncated_unsupervised_metrics(base_path, representation, overwrite=True):
    gin_bindings = [
        "evaluation.evaluation_fn = @truncated_unsupervised_metrics",
        "truncated_unsupervised_metrics.num_train = 10000", "evaluation.random_seed = 2051556033",
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
        try:
            assert truncation_file.exists()
        except AssertionError:
            raise FileNotFoundError("Passive variables indexes not found. {} does not exist. "
                                    "Make sure to compute passive variable indexes before truncated scores"
                                    .format(str(truncation_file)))

        bindings = gin_bindings + ["truncation.pv_idx_file = '{}'".format(str(truncation_file))]
        print("Computing truncated unsupervised metrics of {} using {} representation"
              .format(path.parent.parent, representation))
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=bindings)
