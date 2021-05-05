from functools import partial
from multiprocessing import Pool
import pathlib
import pandas as pd
from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config
from disentanglement_lib.evaluation import evaluate
import tensorflow as tf
from tc_study.experiment import logger
from tc_study.utils.tf_config import set_cpu_option


def compute_passive_variable_index_mp(path, overwrite):
    tf.keras.backend.clear_session()
    set_cpu_option()
    import gin
    from tc_study.experiment.passive_variables import passive_variables_indexes
    gin_bindings = [
        "evaluation.evaluation_fn =  @passive_variables_idx",
        "passive_variables_idx.num_train = 10000", "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "evaluation.name = 'passive variables'"
    ]
    path = pathlib.Path(path)
    model_id = str(path.parent.parent).split("/")[-1]
    result_path = path.parent.parent / "metrics" / "mean" / "passive_variables"
    logger.info("Computing passive variable indexes of {}".format(path.parent.parent))
    try:
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=gin_bindings)
    except Exception as e:
        logger.error("{}\t{}".format(model_id, e))
        gin.clear_config()


def compute_passive_variable_indexes_mp(base_path, overwrite=True, nb_proc=3):
    df = pd.DataFrame(get_config())
    to_process = set(range(10800)) - set(df.index[df["model.name"] == "dip_vae_i"].to_list())
    model_paths = ["{}/{}/postprocessed/mean".format(base_path, i) for i in to_process]
    f = partial(compute_passive_variable_index_mp, overwrite=overwrite)
    with Pool(processes=nb_proc) as pool:
        pool.map(f, model_paths)
