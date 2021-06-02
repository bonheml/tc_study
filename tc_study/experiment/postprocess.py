import pathlib
from functools import partial
from glob import glob
from multiprocessing import Pool

import gin
import tensorflow.compat.v1 as tf
from disentanglement_lib.postprocessing import postprocess

from tc_study.experiment import logger
from tc_study.experiment.utils import get_model_paths
from tc_study.utils.tf_config import set_cpu_option


@gin.configurable("variance_representation", blacklist=["ground_truth_data", "gaussian_encoder", "random_state"])
def variance_representation(ground_truth_data, gaussian_encoder, random_state, save_path):
    """Extracts the variance representation from a Gaussian encoder.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param gaussian_encoder: Function that takes observations as input and outputs a
      dictionary with mean and log variances of the encodings in the keys "mean"
      and "logvar" respectively.
    :param random_state: Numpy random state used for randomness.
    :param save_path: String with path where results can be saved.
    :return: Function that takes as keyword arguments the "mean" and
      "logvar" tensors and returns a tensor with the representation. None as no variables are saved.
    """
    del ground_truth_data, gaussian_encoder, random_state, save_path

    def transform_fn(mean, logvar):
        del mean
        return tf.exp(logvar)

    return transform_fn, None


def postprocess_model(model_path, overwrite=True, multiproc=False):
    if multiproc is True:
        tf.keras.backend.clear_session()
        set_cpu_option()

    post_dir = pathlib.Path(model_path)
    representation = post_dir.name
    seed = {"mean": 2546248239, "sampled": 2357136044, "variance": 0}

    bindings = [
        "postprocess.random_seed = {}".format(seed[representation]),
        "postprocess.name = '{}'".format(representation),
        "postprocess.postprocess_fn = @{}_representation".format(representation),
        "dataset.name='auto'"
    ]

    model_dir = post_dir.parent.parent / "model"

    logger.info("Extracting {} representation of model {}...".format(representation, model_dir))
    try:
        postprocess.postprocess_with_gin(str(model_dir), str(post_dir), overwrite=overwrite, gin_bindings=bindings)
    except Exception as e:
        model_id = str(model_dir.parent).split("/")[-1]
        logger.error("{}\t{}".format(model_id, e))
        gin.clear_config()


def postprocess_models(base_path, representation, model_ids=None, overwrite=True, nb_proc=None):
    models_path = get_model_paths(base_path, representation, model_ids)

    if nb_proc is not None:
        f = partial(postprocess_model, overwrite=overwrite, multiproc=True)
        with Pool(processes=nb_proc) as pool:
            pool.map(f, models_path)
    else:
        for path in models_path:
            postprocess_model(path, overwrite)
