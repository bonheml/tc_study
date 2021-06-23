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
    "histograms",
    blacklist=["ground_truth_data", "representation_function", "random_state", "artifact_dir"])
def histograms(ground_truth_data, representation_function, random_state, artifact_dir=None,
               num_train=gin.REQUIRED, batch_size=16):
    """Computes histograms

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
    res = []
    for i in range(mus_train.shape[0]):
        f, b = np.histogram(mus_train[i], bins=20)
        res.append([f.tolist(), b.tolist()])
    scores["hist"] = res
    return scores


def compute_histograms(path, representation, overwrite=True, multiproc=False):
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
        "evaluation.evaluation_fn = @histograms",
        "histograms.num_train = 10000",
        "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "evaluation.name = 'histograms'",
    ]
    path = pathlib.Path(path)
    result_path = path.parent.parent / "metrics" / representation / "histograms"
    logger.info("Computing histograms of {} using {} representation"
                .format(path.parent.parent, representation))
    gin_evaluation(path, result_path, overwrite, gin_bindings)


def compute_all_histograms(base_path, representation, model_ids=None, overwrite=True, nb_proc=None):
    model_paths = get_model_paths(base_path, representation, model_ids=model_ids)
    if nb_proc is not None:
        f = partial(compute_histograms, representation=representation, overwrite=overwrite, multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, model_paths)
    else:
        for path in model_paths:
            compute_histograms(path, representation, overwrite)
