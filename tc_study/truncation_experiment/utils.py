import json
import gin
import numpy as np


@gin.configurable("truncation", blacklist=["target"])
def make_truncation(target, truncation_fn=gin.REQUIRED, pv_idx=gin.REQUIRED, num_pv=gin.REQUIRED):
    """Wrapper that creates passive variable filters."""
    return truncation_fn(target, pv_idx, num_pv)


@gin.configurable("vp_truncation", blacklist=["target"])
def _vp_truncation(target, pv_idx=gin.REQUIRED, num_pv=gin.REQUIRED):
    """Return target truncated of its passive variables"""
    if num_pv > 0:
        target = np.delete(target, pv_idx, 0)
    return target, num_pv


def get_pv(pv_file):
    check_pv_file(pv_file)
    with open(str(pv_file)) as f:
        pvs = json.load(f)
    pv_idx = pvs["evaluation_results.passive_variables_idx"]
    num_pv = pvs["evaluation_results.num_passive_variables"]
    return pv_idx, num_pv


def check_pv_file(pv_file):
    try:
        assert pv_file.exists()
    except AssertionError:
        raise FileNotFoundError("Passive variables indexes not found. {} does not exist. "
                                "Make sure to compute passive variable indexes before truncated scores"
                                .format(str(pv_file)))
