import pathlib
from functools import partial
from itertools import combinations_with_replacement
from multiprocessing import Pool

import gin
import numpy as np
from disentanglement_lib.evaluation.metrics import utils as dlib_utils
from disentanglement_lib.evaluation.metrics.downstream_task import _compute_loss

from tc_study.experiment import logger
from tc_study.experiment.utils import get_variables_idx, get_model_paths, gin_evaluation


def compute_truncated_downstream_task(idxs_to_keep, mus_train, ys_train, mus_test, ys_test, train_size, prefix):
    scores = {}
    predictor_model = dlib_utils.make_predictor_fn()

    train = mus_train[idxs_to_keep]
    test = mus_test[idxs_to_keep]

    train_err, test_err = _compute_loss(np.transpose(train), ys_train, np.transpose(test),
                                        ys_test, predictor_model)
    scores["{}.{}:mean_train_accuracy".format(prefix, train_size)] = np.mean(train_err)
    scores["{}.{}:mean_test_accuracy".format(prefix, train_size)] = np.mean(test_err)
    scores["{}.{}:min_train_accuracy".format(prefix, train_size)] = np.min(train_err)
    scores["{}.{}:min_test_accuracy".format(prefix, train_size)] = np.min(test_err)
    for i in range(len(train_err)):
        scores["{}.{}:train_accuracy_factor_{}".format(prefix, train_size, i)] = train_err[i]
        scores["{}.{}:test_accuracy_factor_{}".format(prefix, train_size, i)] = test_err[i]


@gin.configurable("truncated_downstream_task",
                  blacklist=["ground_truth_data", "representation_function", "random_state", "active_variables_idx",
                             "mixed_variables_idx", "passive_variables_idx", "artifact_dir"])
def truncated_downstream_task(ground_truth_data, representation_function, random_state, active_variables_idx,
                              mixed_variables_idx, passive_variables_idx, artifact_dir=None,
                              num_train=gin.REQUIRED, num_test=gin.REQUIRED, batch_size=16):
    """Computes loss of downstream task using representations based on the implementation of downstream_task function
     from disentanglement lib and evaluate on all possible combinations of passive, mixed, and active variables

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
    representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param active_variables_idx: The indexes of active variables
    :param mixed_variables_idx: The indexes of mixed variables
    :param passive_variables_idx: The indexes of passive variables
    :param artifact_dir: Optional path to directory where artifacts can be saved.
    :param num_train: Number of points used for training.
    :param num_test: Number of points used for testing.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    for train_size in num_train:
        mus_train, ys_train = dlib_utils.generate_batch_factor_code(
            ground_truth_data, representation_function, train_size, random_state,
            batch_size)
        num_codes = mus_train.shape[0]
        mus_test, ys_test = dlib_utils.generate_batch_factor_code(
            ground_truth_data, representation_function, num_test, random_state,
            batch_size)

        # Compute full downstream task scores
        compute_truncated_downstream_task(range(num_codes), mus_train, ys_train, mus_test, ys_test, train_size, "full")

        # Compute truncated downstream task scores combining all possible combinations of variable type
        idxs = {"active": active_variables_idx, "passive": mixed_variables_idx, "mixed": passive_variables_idx}
        variables = {"{}_{}".format(v1, v2): idxs[v1] + idxs[v2]
                     for v1, v2 in combinations_with_replacement(["active", "mixed", "passive"], 2)}

        for suffix, indexes in variables.items():
            scores += compute_truncated_downstream_task(indexes, mus_train, ys_train, mus_test, ys_test, train_size,
                                                        suffix)

        return scores


def compute_truncated_downstream_tasks(path, representation, predictor="logistic_regression_cv", overwrite=True):
    """ Compute downstream task scores for a given representation with its passive variables removed

        :param path: Path of the model
        :param representation: type of representation, can be mean or sampled
        :param predictor: predictor function to use, can be logistic_regression_cv or gradient_boosting_classifier
        :param overwrite: If true, overwrite previous results, otherwise raises an error if previous results exists
        :return: None
    """
    # Use the seeds used for sampled representations of pre trained models in Locatello et al. 2019.
    seed = 3830135878 if predictor == "logistic_regression_cv" else 1277901399
    gin_bindings = [
        "evaluation.evaluation_fn = @truncated_downstream_task",
        "truncated_downstream_task.num_train = 10000",
        "evaluation.random_seed = {}".format(seed),
        "dataset.name='auto'", "truncated_downstream_task.num_train = [10, 100, 1000, 10000]",
        "truncated_downstream_task.num_test = 5000",
        "evaluation.name = 'truncated downstream task'",
        "truncation.truncation_fn = @vp_truncation",
        "predictor.predictor_fn = @{}".format(predictor),
    ]
    res_folder = "logistic_regression" if predictor.startswith("logistic_regression") else "boosted_trees"

    path = pathlib.Path(path)
    result_path = path.parent.parent / "metrics" / representation / "truncated_downstream_task_{}".format(res_folder)
    truncation_file = (path.parent.parent / "metrics" / "mean" / "passive_variables" / "results" / "aggregate"
                       / "evaluation.json")
    idxs = ["truncated__downstream_task.{} = {}".format(k, v) for k, v in get_variables_idx(truncation_file).items()]
    bindings = gin_bindings + idxs
    logger.info("Computing truncated downstream tasks of {} using {} representation and {} predictor"
                .format(path.parent.parent, representation, predictor))
    gin_evaluation(path, result_path, overwrite, bindings)


def compute_all_truncated_downstream_tasks(base_path, representation, predictor="logistic_regression_cv",
                                           overwrite=True, nb_proc=None):
    model_paths = get_model_paths(base_path, representation)
    if nb_proc is not None:
        f = partial(compute_truncated_downstream_tasks, representation=representation, predictor=predictor,
                    overwrite=overwrite, multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, model_paths)
    else:
        for path in model_paths:
            compute_truncated_downstream_tasks(path, representation, predictor, overwrite)
