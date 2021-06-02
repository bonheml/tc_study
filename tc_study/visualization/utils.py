from itertools import combinations_with_replacement
import pandas as pd
from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config


def get_variables_combinations():
    comb = ["{}_{}".format(v1, v2) for v1, v2 in combinations_with_replacement(["active", "mixed", "passive"], 2)]
    comb += ["full", "active", "mixed", "passive"]
    return comb


def get_model_files(base_path, representation):
    base_path = base_path.rstrip("/")
    configs = get_config()
    df = pd.DataFrame(configs)
    to_process = set(range(10800)) - set(df.index[df["model.name"] == "dip_vae_i"].to_list()) - set(df.index[df["dataset.name"] == "shapes3d"].to_list())
    model_files = ["{}/{}/metrics/{}/histograms/results/aggregate/evaluation.json".format(base_path, i, representation) for i in to_process]

    return model_files, to_process

