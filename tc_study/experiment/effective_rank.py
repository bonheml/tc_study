import pathlib
import tensorflow as tf
from functools import partial
from multiprocessing.pool import Pool

import gin
import numpy as np
from disentanglement_lib.evaluation.metrics import utils

from tc_study.experiment import logger
from tc_study.experiment.utils import get_model_paths, gin_evaluation
from tc_study.utils.tf_config import set_cpu_option


@gin.configurable("effective_rank",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def matrix_effective_rank(ground_truth_data, representation_function, random_state, num_train=gin.REQUIRED, batch_size=16):
    """Computes the effective rank of a representation using implementation from
    Implicit Regularization in Deep Learning May Not Be Explainable by Norms (Razin and Cohen, 2020).
    https://github.com/noamrazin/imp_reg_dl_not_norms

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
    representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param num_train: Number of points used for training.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    scores = {}
    representation, _ = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train, random_state,
        batch_size)
    cov = np.cov(representation)
    singular_values = np.linalg.svd(cov, compute_uv=False)
    non_zero_singular_values = singular_values[singular_values != 0]
    normalized_non_zero_singular_values = non_zero_singular_values / non_zero_singular_values.sum()

    singular_values_entropy = -(normalized_non_zero_singular_values * np.log(normalized_non_zero_singular_values)).sum()
    scores["rank"] = singular_values.shape[0]
    scores["effective_rank"] = np.exp(singular_values_entropy)
    return scores


def compute_effective_ranks(path, representation, overwrite=True, multiproc=False):
    """Compute effective ranks for a given representation

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
        "evaluation.evaluation_fn = @effective_rank",
        "evaluation.random_seed = 0",
        "dataset.name='auto'",
        "evaluation.name = 'effective rank'",
        "effective_rank.num_train = 10000",
    ]
    path = pathlib.Path(path)
    result_path = path.parent.parent / "metrics" / representation / "effective_rank"
    logger.info("Computing effective rank of {} using {} representation".format(path.parent.parent, representation))
    gin_evaluation(path, result_path, overwrite, gin_bindings)


def compute_all_effective_ranks(base_path, representation, model_ids=None, overwrite=True, nb_proc=None):
    model_paths = get_model_paths(base_path, representation, model_ids=model_ids)
    if nb_proc is not None:
        f = partial(compute_effective_ranks, representation=representation, overwrite=overwrite,
                    multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, model_paths)
    else:
        for path in model_paths:
            compute_effective_ranks(path, representation, overwrite)