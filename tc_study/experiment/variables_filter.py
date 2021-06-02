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
                  num_train=gin.REQUIRED, pv_threshold=0.1, mv_threshold=0.9, batch_size=16):
    """Get indexes of variables whose variance is lower than a given threshold.

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
     representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param artifact_dir: Optional path to directory where artifacts can be saved.
    :param num_train: Threshold under which variance is considered low enough to correspond to passive variables
    :param pv_threshold: Maximum variance of passive variables.
    :param mv_threshold: Maximum variance of mixed variables.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    mus_train, _ = dlib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                         random_state,
                                                         batch_size)
    num_codes = mus_train.shape[0]
    variances = np.diag(np.cov(mus_train))
    assert num_codes == variances.shape[0]

    scores["passive_variables_idx"] = list(np.where(variances < pv_threshold)[0])
    scores["mixed_variables_idx"] = list(np.where((variances >= pv_threshold) & (variances < mv_threshold))[0])
    scores["active_variables_idx"] = list(np.where(variances >= mv_threshold)[0])
    scores["variances"] = variances.tolist()
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
    result_path = path.parent.parent / "metrics" / "mean" / "filtered_variables"
    logger.info("Computing variable indexes of {}".format(path.parent.parent))
    gin_evaluation(path, result_path, overwrite, gin_bindings)


def compute_all_variable_indexes(base_path, model_ids=None, overwrite=True, nb_proc=None):
    model_paths = get_model_paths(base_path, "mean", model_ids=model_ids)
    if nb_proc is not None:
        f = partial(compute_variable_indexes, overwrite=overwrite, multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, model_paths)
    else:
        for path in model_paths:
            compute_variable_indexes(path, overwrite)
