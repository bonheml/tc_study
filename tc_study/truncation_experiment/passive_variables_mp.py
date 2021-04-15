from functools import partial
from multiprocessing import get_context, Pool
import pathlib
from disentanglement_lib.evaluation import evaluate
import tensorflow as tf
from tc_study.truncation_experiment import logger
from tc_study.utils.tf_config import set_cpu_option


def compute_passive_variable_index_mp(path, overwrite):
    tf.keras.backend.clear_session()
    set_cpu_option()
    import gin
    from tc_study.truncation_experiment.passive_variables import passive_variables_indexes
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
    model_paths = ["{}/{}/postprocessed/mean".format(base_path, i) for i in range(10800)]
    f = partial(compute_passive_variable_index_mp, overwrite=overwrite)
    with Pool(processes=nb_proc) as pool:
        pool.map(f, model_paths)
