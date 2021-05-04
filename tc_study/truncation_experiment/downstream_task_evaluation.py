import pathlib
import gin
import numpy as np
import pandas as pd
from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils as dlib_utils
from disentanglement_lib.evaluation.metrics.downstream_task import _compute_loss
from tc_study.truncation_experiment import utils, logger
from tc_study.truncation_experiment.utils import get_pv

configs = get_config()


@gin.configurable("truncated_downstream_task",
                  blacklist=["ground_truth_data", "representation_function", "random_state",
                             "artifact_dir"])
def truncated_downstream_task(ground_truth_data, representation_function, random_state, artifact_dir=None,
                              num_train=gin.REQUIRED, num_test=gin.REQUIRED, batch_size=16):
    """Computes loss of downstream task using representations truncated of their passive variables. This follows the
    implementation of downstream_task function from disentanglement lib but truncated passive variables beforehand.

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
    representation for each observation.
    :param random_state: Numpy random state used for randomness.
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
        mus_test, ys_test = dlib_utils.generate_batch_factor_code(
            ground_truth_data, representation_function, num_test, random_state,
            batch_size)
        mus_train, num_vp = utils.make_truncation(mus_train)
        mus_test, _ = utils.make_truncation(mus_test)
        scores["num_passive_variables"] = num_vp

        # If all variable are passive or only one factor is kept, return without calculation
        if mus_train.shape[0] <= 1:
            return scores
        predictor_model = dlib_utils.make_predictor_fn()

        train_err, test_err = _compute_loss(np.transpose(mus_train), ys_train, np.transpose(mus_test),
                                            ys_test, predictor_model)
        scores["{}:mean_train_accuracy".format(train_size)] = np.mean(train_err)
        scores["{}:mean_test_accuracy".format(train_size)] = np.mean(test_err)
        scores["{}:min_train_accuracy".format(train_size)] = np.min(train_err)
        scores["{}:min_test_accuracy".format(train_size)] = np.min(test_err)
        for i in range(len(train_err)):
            scores["{}:train_accuracy_factor_{}".format(train_size, i)] = train_err[i]
            scores["{}:test_accuracy_factor_{}".format(train_size, i)] = test_err[i]

    return scores


def compute_truncated_downstream_task(base_path, representation, predictor="logistic_regression_cv", overwrite=True):
    """ Compute downstream task score for a given representation with its passive variables removed over all models

        :param base_path: Path of the models
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
    df = pd.DataFrame(configs)
    # Remove dip-vae-i models from the models to evaluate
    to_process = set(range(10800)) - set(df.index[df["model.name"] == "dip_vae_i"].to_list())
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in to_process]
    res_folder = "logistic_regression" if predictor.startswith("logistic_regression") else "boosted_trees"

    for path in model_paths:
        path = pathlib.Path(path)
        result_path = path.parent.parent / "metrics" / representation / "truncated_downstream_task_{}".format(res_folder)
        truncation_file = (path.parent.parent / "metrics" / "mean" / "passive_variables" / "results" / "aggregate"
                           / "evaluation.json")
        pv_idx, num_pv = get_pv(truncation_file)
        bindings = gin_bindings + ["truncation.pv_idx = {}".format(pv_idx), "truncation.num_pv = {}".format(num_pv)]
        logger.info("Computing truncated downstream tasks of {} using {} representation and {} predictor"
                    .format(path.parent.parent, representation, predictor))
        try:
            evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=bindings)
        except Exception as e:
            model_id = str(path.parent.parent).split("/")[-1]
            logger.error("{}\t{}".format(model_id, e))
            gin.clear_config()


def compute_downstream_task(base_path, representation, predictor="logistic_regression_cv", overwrite=True):
    """ Compute downstream task score for a given representation over all models

    :param base_path: Path of the models
    :param representation: type of representation, can be mean or sampled
    :param predictor: predictor function to use, can be logistic_regression_cv or gradient_boosting_classifier
    :param overwrite: If true, overwrite previous results, otherwise raises an error if previous results exists
    :return: None
    """
    # Use the seed used for sampled representations of pre trained models in Locatello et al. 2019.
    seed = 3830135878 if predictor == "logistic_regression_cv" else 1277901399
    gin_bindings = [
        "evaluation.evaluation_fn = @downstream_task",
        "downstream_task.num_train = 10000",
        "evaluation.random_seed = {}".format(seed),
        "dataset.name='auto'", "downstream_task.num_train = [10, 100, 1000, 10000]",
        "downstream_task.num_test = 5000",
        "evaluation.name = 'downstream task'",
        "predictor.predictor_fn = @{}".format(predictor),
    ]
    df = pd.DataFrame(configs)
    to_process = set(range(10800)) - set(df.index[df["model.name"] == "dip_vae_i"].to_list())
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in to_process]
    res_folder = "logistic_regression" if predictor.startswith("logistic_regression") else "boosted_trees"

    for i, path in enumerate(model_paths):
        path = pathlib.Path(path)
        result_path = path.parent.parent / "metrics" / representation / "downstream_task_{}".format(res_folder)
        logger.info("Computing downstream tasks of {} using {} representation and {} predictor"
                    .format(path.parent.parent, representation, predictor))
        try:
            evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=gin_bindings)
        except Exception as e:
            model_id = str(path.parent.parent).split("/")[-1]
            logger.error("{}\t{}".format(model_id, e))
            gin.clear_config()
