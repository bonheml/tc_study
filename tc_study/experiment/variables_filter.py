import pathlib
from functools import partial
from multiprocessing import Pool
import gin
import numpy as np
import tensorflow as tf
from disentanglement_lib.evaluation.metrics import utils as dlib_utils

from tc_study.experiment import logger
from tc_study.experiment.utils import get_model_paths, gin_evaluation
from tc_study.utils.tf_config import set_cpu_option


@gin.configurable(
    "variables_idx",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def variables_idx(ground_truth_data, representation_function, random_state, artifact_dir=None,
                  num_train=gin.REQUIRED, var_threshold=0.1, mean_error_range=0.1, batch_size=16):
    """Get indexes of variables whose variance is lower than a given threshold.

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
     representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param artifact_dir: Optional path to directory where artifacts can be saved.
    :param num_train: Threshold under which variance is considered low enough to correspond to passive variables
    :param var_threshold: Maximum variance of active and passive variables.
    :param mean_error_range: maximum deviation for ideal mean value of active and passive variables.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    mus_train, _ = dlib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                         random_state,
                                                         batch_size)
    num_codes = mus_train.shape[0]
    variances = np.var(mus_train, axis=1)
    means = np.mean(mus_train, axis=1)
    assert num_codes == variances.shape[0] == means.shape[0]

    all_idxs = set(list(range(num_codes)))
    low_var_idxs = set(list(np.where(variances < var_threshold)[0]))
    higher_var_idxs = all_idxs - low_var_idxs
    # We only need to check the upper bound as variance will always be positive
    zero_mean_idxs = set(list(np.where(means <= mean_error_range)[0]))
    one_mean_idxs = set(list(np.where(((1 - mean_error_range) <= means) & (means <= (1 + mean_error_range)))[0]))
    mixed_means_idxs = all_idxs - (zero_mean_idxs | one_mean_idxs)

    scores["passive_variables_idx"] = list(low_var_idxs.intersection(one_mean_idxs))
    scores["active_variables_idx"] = list(low_var_idxs.intersection(zero_mean_idxs))
    scores["mixed_variables_idx"] = list(mixed_means_idxs | higher_var_idxs)
    scores["variances"] = variances.tolist()
    scores["means"] = means.tolist()
    print()

    checksum = len(scores["passive_variables_idx"]) + len(scores["mixed_variables_idx"]) + len(scores["active_variables_idx"])
    try:
        assert checksum == num_codes
    except AssertionError as e:
        logger.info("\nMeans:{}\nVariances:{}\nActive variables idxs:{}\nMixed variables idxs:{}\nPassive variables idxs"
                     ":{}\n".format(scores["means"], scores["variances"], scores["active_variables_idx"],
                                    scores["mixed_variables_idx"], scores["passive_variables_idx"]))
        raise AssertionError("Total number of indexes is {} instead of {}".format(checksum, num_codes))
    return scores


def compute_variable_indexes(path, overwrite=True, multiproc=False):
    """ Get indexes of passive, mixed and active variables

    :param path: Path of the model
    :param overwrite: if true, overwrite existing results
    :param multiproc: True if function is executed in multiprocess Pool, else False
    :return: None
    """
    if multiproc is True:
        tf.keras.backend.clear_session()
        set_cpu_option()

    gin_bindings = [
        "evaluation.evaluation_fn = @variables_idx",
        "variables_idx.num_train = 10000", "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "evaluation.name = 'variables index'"
    ]
    path = pathlib.Path(path)
    result_path = path.parent.parent / "metrics" / "variance" / "filtered_variables"
    logger.info("Computing variable indexes of {}".format(path.parent.parent))
    gin_evaluation(path, result_path, overwrite, gin_bindings)


def compute_all_variable_indexes(base_path, model_ids=None, overwrite=True, nb_proc=None):
    model_paths = get_model_paths(base_path, "variance", model_ids=model_ids)
    if nb_proc is not None:
        f = partial(compute_variable_indexes, overwrite=overwrite, multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, model_paths)
    else:
        for path in model_paths:
            compute_variable_indexes(path, overwrite)
