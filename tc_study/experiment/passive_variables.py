import gin
import numpy as np
import pathlib

from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils as dlib_utils
import pandas as pd
from tc_study.experiment import logger


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
    """Get indexes of variables whose variance is lower than a given threshold.

    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
     representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param artifact_dir: Optional path to directory where artifacts can be saved.
    :param num_train: Threshold under which variance is considered low enough to correspond to passive variables
    :param threshold: Number of points used for training.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    mus_train, _ = dlib_utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train,
                                                         random_state,
                                                         batch_size)
    num_codes = mus_train.shape[0]
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    small_var_idx = np.where(np.diag(cov_mus) < threshold)[0]
    small_max_idx = np.where(mus_train.mean(axis=1)**2 < threshold)
    pv_idx = list(np.intersect1d(small_var_idx, small_max_idx))
    num_pv = len(pv_idx)
    scores["passive_variables_idx"] = pv_idx
    scores["num_passive_variables"] = num_pv
    return scores


def compute_passive_variable_indexes(base_path, overwrite=True):
    """ Get passive variable indexes of all models

    :param base_path: path where the models are stored
    :param overwrite: if true, overwrite existing results
    :return: None
    """
    gin_bindings = [
        "evaluation.evaluation_fn = @passive_variables_idx",
        "passive_variables_idx.num_train = 10000", "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "evaluation.name = 'passive variables'"
    ]
    df = pd.DataFrame(get_config())
    to_process = set(range(10800)) - set(df.index[df["model.name"] == "dip_vae_i"].to_list())
    model_paths = ["{}/{}/postprocessed/mean".format(base_path, i) for i in to_process]

    for path in model_paths:
        path = pathlib.Path(path)
        result_path = path.parent.parent / "metrics" / "mean" / "passive_variables"
        logger.info("Computing passive variable indexes of {}".format(path.parent.parent))
        try:
            evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=gin_bindings)
        except Exception as e:
            model_id = str(path.parent.parent).split("/")[-1]
            logger.error("{}\t{}".format(model_id, e))
            gin.clear_config()