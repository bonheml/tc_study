import pathlib
from functools import partial
from multiprocessing import Pool
import pandas as pd
from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config
from disentanglement_lib.evaluation import evaluate
import gin
from tc_study.experiment import logger
from tc_study.experiment.utils import get_pv
from tc_study.utils.tf_config import set_cpu_option


def compute_truncated_unsupervised_metric_mp(path, representation, overwrite=True):
    set_cpu_option()
    from tc_study.experiment.unsupervised_evaluation import truncated_unsupervised_metrics
    gin_bindings = [
        "evaluation.evaluation_fn = @truncated_unsupervised_metrics",
        "truncated_unsupervised_metrics.num_train = 10000",
        "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20", "evaluation.name = 'truncated unsupervised metrics'",
        "truncation.truncation_fn = @vp_truncation"
    ]
    path = pathlib.Path(path)
    model_id = str(path.parent.parent).split("/")[-1]
    result_path = path.parent.parent / "metrics" / representation / "truncated_unsupervised"
    truncation_file = (path.parent.parent / "metrics" / "mean" / "passive_variables" / "results" / "aggregate"
                       / "evaluation.json")
    pv_idx, num_pv = get_pv(truncation_file)
    bindings = gin_bindings + ["truncation.pv_idx = {}".format(pv_idx), "truncation.num_pv = {}".format(num_pv)]
    logger.info("Computing truncated unsupervised metrics of {} using {} representation"
                .format(path.parent.parent, representation))
    try:
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=bindings)
    except Exception as e:
        logger.error("{}\t{}".format(model_id, e))
        gin.clear_config()


def compute_truncated_unsupervised_metrics_mp(base_path, representation, overwrite=True, nb_proc=3):
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in range(10800)]
    f = partial(compute_truncated_unsupervised_metric_mp, representation=representation, overwrite=overwrite)
    with Pool(processes=nb_proc) as pool:
        pool.map(f, model_paths)


def compute_normalized_unsupervised_metric_mp(path, representation, overwrite=True):
    set_cpu_option()
    from tc_study.experiment.unsupervised_evaluation import normalized_unsupervised_metrics
    gin_bindings = [
        "evaluation.evaluation_fn = @normalized_unsupervised_metrics",
        "normalized_unsupervised_metrics.num_train = 10000",
        "evaluation.random_seed = 2051556033",
        "dataset.name='auto'", "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20", "evaluation.name = 'normalized unsupervised metrics'"
    ]
    path = pathlib.Path(path)
    model_id = str(path.parent.parent).split("/")[-1]
    result_path = path.parent.parent / "metrics" / representation / "normalized_unsupervised"
    logger.info("Computing normalized unsupervised metrics of {} using {} representation"
                .format(path.parent.parent, representation))
    try:
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=gin_bindings)
    except Exception as e:
        logger.error("{}\t{}".format(model_id, e))
        gin.clear_config()


def compute_normalized_unsupervised_metrics_mp(base_path, representation, overwrite=True, nb_proc=3):
    df = pd.DataFrame(get_config())
    to_process = set(range(10800)) - set(df.index[df["model.name"] == "dip_vae_i"].to_list())
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in to_process]
    f = partial(compute_normalized_unsupervised_metric_mp, representation=representation, overwrite=overwrite)
    with Pool(processes=nb_proc) as pool:
        pool.map(f, model_paths)
