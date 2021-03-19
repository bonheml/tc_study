import json
import gin
import numpy as np


@gin.configurable("truncation", blacklist=["target"])
def make_truncation(target, truncation_fn=gin.REQUIRED, pv_idx_file=gin.REQUIRED):
    """Wrapper that creates passive variable filters."""
    return truncation_fn(target, pv_idx_file)


@gin.configurable("vp_truncation", blacklist=["target"])
def _vp_truncation(target, pv_idx_file=gin.REQUIRED):
    """Return target truncated of its passive variables"""
    with open(pv_idx_file) as f:
        pvs = json.load(f)
    pv_idx = pvs["evaluation_results.passive_variables_idx"]
    num_pv = pvs["evaluation_results.num_passive_variables"]
    if num_pv > 0:
        target = np.delete(target, pv_idx, 0)
    return target, num_pv
