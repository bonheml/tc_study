import json
import numpy as np
import gin
import pandas as pd
from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config
from disentanglement_lib.evaluation import evaluate

from tc_study.experiment import logger


def get_model_paths(base_path, representation, model_ids=None):
    base_path = base_path.rstrip("/")
    configs = get_config()
    df = pd.DataFrame(configs)
    to_process = model_ids if model_ids is not None else range(10800)
    to_process = set(to_process) - set(df.index[df["model.name"] == "dip_vae_i"].to_list()) - set(df.index[df["dataset.name"] == "shapes3d"].to_list())
    model_paths = ["{}/{}/postprocessed/{}".format(base_path, i, representation) for i in to_process]
    return model_paths


def get_variables_idx(var_idx_file):
    check_variables_idx_file(var_idx_file)
    with open(str(var_idx_file)) as f:
        pvs = json.load(f)
    res = {"active_variables_idx": pvs["evaluation_results.active_variables_idx"],
           "mixed_variables_idx": pvs["evaluation_results.mixed_variables_idx"],
           "passive_variables_idx": pvs["evaluation_results.passive_variables_idx"]}
    return res


def check_variables_idx_file(var_idx_file):
    try:
        assert var_idx_file.exists()
    except AssertionError:
        raise FileNotFoundError("Variables index file not found. {} does not exist. "
                                "Make sure to compute passive variable indexes before truncated scores"
                                .format(str(var_idx_file)))


def gin_evaluation(path, result_path, overwrite, bindings):
    try:
        evaluate.evaluate_with_gin(str(path), str(result_path), overwrite=overwrite, gin_bindings=bindings)
    except Exception as e:
        model_id = str(path.parent.parent).split("/")[-1]
        logger.error("{}\t{}".format(model_id, e))
        gin.clear_config()
