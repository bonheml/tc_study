import pathlib
import zipfile
from io import BytesIO

import pandas as pd
import requests
from disentanglement_lib.config.unsupervised_study_v1.sweep import get_config

configs = get_config()


def get_model_info():
    """Retrieve information about the pre-trained models of "Challenging Common Assumptions in the Unsupervised Learning
    of Disentangled Representations" (http://proceedings.mlr.press/v97/locatello19a.html) from disentanglement lib
    (https://github.com/google-research/disentanglement_lib).

    :return: A dataframe containing the datasets and models types used along with model indexes matching each
    (model type, dataset) pair
    """
    df = pd.DataFrame(configs)
    results = {"dataset": [], "model": [], "start_idx": [], "end_idx": []}
    for model in df['model.name'].unique():
        for dataset in df['dataset.name'].unique():
            idxs = df.index[(df['model.name'] == model) & (df['dataset.name'] == dataset)]
            results["dataset"].append(dataset)
            results["model"].append(model)
            results["start_idx"].append(idxs.min())
            results["end_idx"].append(idxs.max())
    df2 = pd.DataFrame.from_dict(results)
    return df2


def download_models(base_path, save_space=False, overwrite=False):
    """ Download all the pretrained models of "Challenging Common Assumptions in the Unsupervised Learning
    of Disentangled Representations" (http://proceedings.mlr.press/v97/locatello19a.html), unzip and save them
    in <base_path>.

    :param base_path: The path where the models will be saved
    :param save_space: If true, only the metrics will be saved to use less space, otherwise, all the content will be saved
    :param overwrite: If true, any existing saved model in <base_path> will be override, otherwise, if any model is present
    in the base_path, download is skipped.
    :return: None
    """
    file = pathlib.Path("{}/0".format(base_path))
    if file.exists() and not overwrite:
        return
    base_url = "https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1"
    for i, url in [(i, "{}/{}.zip".format(base_url, i)) for i in range(10800)]:
        print("Downloading model {} from {}".format(i, url))
        r = requests.get(url)
        raw_content = BytesIO(r.content)
        z = zipfile.ZipFile(raw_content)
        members = [n for n in z.namelist() if n.startswith("{}/metrics/".format(i))] if save_space is True else None
        z.extractall(path=base_path, members=members)


def get_file(path):
    """Returns path relative to file."""
    from pkg_resources import resource_filename
    return resource_filename("tc_study", path)
