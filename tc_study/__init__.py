import multiprocessing
import os
import PIL
import scipy.io as sio
import numpy as np
import pandas as pd
import warnings



warnings.filterwarnings("ignore", category=FutureWarning)

from tensorflow import gfile
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from disentanglement_lib.data.ground_truth.cars3d import CARS3D_PATH
from disentanglement_lib.data.ground_truth.dsprites import DSprites, SCREAM_PATH
from disentanglement_lib.data.ground_truth import dsprites, cars3d
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.utils.aggregate_results import _load


# Monkey patch issues with thumbnail for ScreamDSprites and cars3D
# Use 2D tuple instead of 3D ones for thumbnail to prevent exceptions during data generation


class ChangedInitScreamDSprites(dsprites.ScreamDSprites):
    def __init__(self, latent_factor_indices=None):
        DSprites.__init__(self, latent_factor_indices)
        self.data_shape = [64, 64, 3]
        with gfile.Open(SCREAM_PATH, "rb") as f:
            scream = PIL.Image.open(f)
            scream.thumbnail((350, 274))
            self.scream = np.array(scream) * 1. / 255.


dsprites.ScreamDSprites = ChangedInitScreamDSprites


def _load_mesh(filename):
  """Parses a single source in_fname and rescales contained images."""
  with gfile.Open(os.path.join(CARS3D_PATH, filename), "rb") as f:
    mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
  flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
  rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
  for i in range(flattened_mesh.shape[0]):
    pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
    pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
    rescaled_mesh[i, :, :, :] = np.array(pic)
  return rescaled_mesh * 1. / 255


cars3d._load_mesh = _load_mesh


def _get(pattern):
  files = gfile.Glob(pattern)
  with multiprocessing.Pool() as pool:
    try:
        all_results = pool.map(_load, files)
    except Exception as e:
        pool.close()
        raise(e)
  return pd.DataFrame(all_results)

# Monkey patch _get to make sure that pool are closed the result is returned

aggregate_results._get = _get

