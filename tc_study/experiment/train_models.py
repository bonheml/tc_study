import glob
import os
import time

import gin.tf
import gin.tf
import gin.tf.external_configurables
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.methods.unsupervised.train import _make_input_fn
from disentanglement_lib.utils import results
from tensorflow.contrib import tpu as contrib_tpu


# Note: this is mostly a copy-paste of disentanglement lib train method from
# https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/train.py
# The only modification is that the train function can now keep more than one checkpoint which is needed for
# the experiment


def train_with_gin(model_dir,
                   overwrite=False,
                   gin_config_files=None,
                   gin_bindings=None):
    """Trains a model based on the provided gin configuration.
    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.
    Args:
      model_dir: String with path to directory where model output should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    train(model_dir, overwrite)
    gin.clear_config()


@gin.configurable("custom_model", blacklist=["model_dir", "overwrite"])
def train(model_dir,
          overwrite=False,
          model=gin.REQUIRED,
          training_steps=gin.REQUIRED,
          random_seed=gin.REQUIRED,
          batch_size=gin.REQUIRED,
          eval_steps=1000,
          save_checkpoints_steps=1000,
          keep_checkpoint_max=0,
          name="",
          model_num=None):
    """Adaptation of disentanglement lib train function to keep more than one checkpoint
    Trains the estimator and exports the snapshot and the gin config.
    The use of this function requires the gin binding 'dataset.name' to be
    specified as that determines the data set used for training.
    :param model_dir: String with path to directory where model output should be saved.
    :param overwrite: Boolean indicating whether to overwrite output directory.
    :param model: GaussianEncoderModel that should be trained and exported.
    :param training_steps: Integer with number of training steps.
    :param random_seed: Integer with random seed used for training.
    :param batch_size: Integer with the batch size.
    :param eval_steps: Optional integer with number of steps used for evaluation.
    :param save_checkpoints_steps: Save checkpoints every this many steps.
    :param keep_checkpoint_max: The maximum number of recent checkpoint files to keep.
    As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
    :param name: Optional string with name of the model (can be used to name models).
    :param model_num: Optional integer with model number (can be used to identify models).
    """
    # We do not use the variables 'name' and 'model_num'. Instead, they can be
    # used to name results as they will be part of the saved gin config.
    del name, model_num

    # Delete the output directory if it already exists.
    if tf.gfile.IsDirectory(model_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(model_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    # Create a numpy random state. We will sample the random seeds for training
    # and evaluation from this.
    random_state = np.random.RandomState(random_seed)

    # Obtain the dataset.
    dataset = named_data.get_named_ground_truth_data()

    # We create a TPUEstimator based on the provided model. This is primarily so
    # that we could switch to TPU training in the future. For now, we train
    # locally on GPUs.
    run_config = contrib_tpu.RunConfig(
        tf_random_seed=random_seed,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max,
        tpu_config=contrib_tpu.TPUConfig(iterations_per_loop=500))
    tpu_estimator = contrib_tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model.model_fn,
        model_dir=os.path.join(model_dir, "tf_checkpoint"),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        config=run_config)

    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    # Do the actual training.
    tpu_estimator.train(input_fn=_make_input_fn(dataset, random_state.randint(2 ** 32)), steps=training_steps)

    # Save model as a TFHub module.
    output_shape = named_data.get_named_ground_truth_data().observation_shape
    checkpoint_path = tpu_estimator.latest_checkpoint()
    export_as_multi_tf_hub(model, output_shape, checkpoint_path, model_dir)

    # Save the results. The result dir will contain all the results and config
    # files that we copied along, as we progress in the pipeline. The idea is that
    # these files will be available for analysis at the end.
    results_dict = tpu_estimator.evaluate(input_fn=_make_input_fn(dataset, random_state.randint(2 ** 32),
                                                                  num_batches=eval_steps))
    results_dict["elapsed_time"] = time.time() - experiment_timer
    checkpoints = glob.glob("{}/*.meta".format("/".join(checkpoint_path.split("/")[:-1])))
    results_dirs = ["{}/{}/model/results".format(model_dir, r.rstrip(".meta").split("-")[-1]) for r in checkpoints]
    for results_dir in results_dirs:
        results.update_result_directory(results_dir, "train", results_dict)


@gin.configurable("export_as_multi_tf_hub", whitelist=[])
def export_as_multi_tf_hub(gaussian_encoder_model, observation_shape, checkpoint_path, export_path,
                           drop_collections=None):
    """Exports the provided GaussianEncoderModel as a TFHub module using multiple checkpoints.

    Args:
      gaussian_encoder_model: GaussianEncoderModel to be exported.
      observation_shape: Tuple with the observations shape.
      checkpoint_path: String with path where to load weights from.
      export_path: String with path where to save the TFHub module to.
      drop_collections: List of collections to drop from the graph.
    """

    def module_fn(is_training):
        """Module function used for TFHub export."""
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # Add a signature for the Gaussian encoder.
            image_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[None] + observation_shape)
            mean, logvar = gaussian_encoder_model.gaussian_encoder(
                image_placeholder, is_training)
            hub.add_signature(
                name="gaussian_encoder",
                inputs={"images": image_placeholder},
                outputs={
                    "mean": mean,
                    "logvar": logvar
                })

            # Add a signature for reconstructions.
            latent_vector = gaussian_encoder_model.sample_from_latent_distribution(
                mean, logvar)
            reconstructed_images = gaussian_encoder_model.decode(
                latent_vector, observation_shape, is_training)
            hub.add_signature(
                name="reconstructions",
                inputs={"images": image_placeholder},
                outputs={"images": reconstructed_images})

            # Add a signature for the decoder.
            latent_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[None, mean.get_shape()[1]])
            decoded_images = gaussian_encoder_model.decode(latent_placeholder,
                                                           observation_shape,
                                                           is_training)

            hub.add_signature(
                name="decoder",
                inputs={"latent_vectors": latent_placeholder},
                outputs={"images": decoded_images})

    # Export the module.
    # Two versions of the model are exported:
    #   - one for "test" mode (the default tag)
    #   - one for "training" mode ("is_training" tag)
    # In the case that the encoder/decoder have dropout, or BN layers, these two
    # graphs are different.
    tags_and_args = [
        ({"train"}, {"is_training": True}),
        (set(), {"is_training": False}),
    ]
    spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args,
                                  drop_collections=drop_collections)
    checkpoints = glob.glob("{}/*.meta".format("/".join(checkpoint_path.split("/")[:-1])))
    paths = [("{}/{}/model/tfhub".format(export_path, r.rstrip(".meta").split("-")[-1]), r.rstrip(".meta")) for r in checkpoints]
    for exp_dir, checkpoint_file in paths:
        spec.export(exp_dir, checkpoint_path=checkpoint_file)
