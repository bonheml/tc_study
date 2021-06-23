import gin
import tensorflow as tf
from disentanglement_lib.methods.shared import architectures, optimizers, losses
from disentanglement_lib.methods.unsupervised.vae import make_metric_fn, compute_gaussian_kl, total_correlation
from disentanglement_lib.methods.unsupervised.vae import regularize_diag_off_diag_dip, anneal
from disentanglement_lib.methods.unsupervised.vae import compute_covariance_z_mean, shuffle_codes, BaseVAE


class LoggedBaseVAE(BaseVAE):
    """Abstract base class of a basic Gaussian encoder model."""

    def model_fn(self, features, labels, mode, params):
        """TPUEstimator compatible model function."""
        del labels
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_shape = features.get_shape().as_list()[1:]
        z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        zipped = zip(tf.unstack(z_mean, axis=1), tf.unstack(z_sampled, axis=1), tf.unstack(tf.exp(z_logvar), axis=1))

        for i, (zm, zs, zv) in enumerate(zipped):
            tf.summary.histogram("mean_representation/z{}".format(i), zm)
            tf.summary.histogram("sampled_representation/z{}".format(i), z_sampled)
            tf.summary.histogram("var_representation/z{}".format(i), zv)

        reconstructions = self.decode(z_sampled, data_shape, is_training)
        per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
        reconstruction_loss = tf.reduce_mean(per_sample_loss)
        kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
        loss = tf.add(reconstruction_loss, regularizer, name="loss")
        elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = optimizers.make_vae_optimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
            train_op = tf.group([train_op, update_ops])
            tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            tf.summary.scalar("elbo", -elbo)

            logging_hook = tf.train.LoggingTensorHook({"loss": loss, "reconstruction_loss": reconstruction_loss,
                                                       "elbo": -elbo}, every_n_iter=100)

            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                             "regularizer", "kl_loss"),
                              [reconstruction_loss, -elbo, regularizer, kl_loss]))
        else:
            raise NotImplementedError("Eval mode not supported.")


@gin.configurable("vae_logged")
class BetaVAE(LoggedBaseVAE):
    """BetaVAE model."""

    def __init__(self, beta=gin.REQUIRED):
        """Creates a beta-VAE model.

        Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
        Constrained Variational Framework"
        (https://openreview.net/forum?id=Sy2fzU9gl).

        Args:
          beta: Hyperparameter for the regularizer.

        Returns:
          model_fn: Model function for TPUEstimator.
        """
        self.beta = beta

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss


@gin.configurable("annealed_vae_logged")
class AnnealedVAE(LoggedBaseVAE):
    """AnnealedVAE model."""

    def __init__(self,
                 gamma=gin.REQUIRED,
                 c_max=gin.REQUIRED,
                 iteration_threshold=gin.REQUIRED):
        """Creates an AnnealedVAE model.

        Implementing Eq. 8 of "Understanding disentangling in beta-VAE"
        (https://arxiv.org/abs/1804.03599).

        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        c = anneal(self.c_max, tf.train.get_global_step(), self.iteration_threshold)
        return self.gamma * tf.math.abs(kl_loss - c)


@gin.configurable("factor_vae_logged")
class FactorVAE(LoggedBaseVAE):
    """FactorVAE model."""

    def __init__(self, gamma=gin.REQUIRED):
        """Creates a FactorVAE model.

        Implementing Eq. 2 of "Disentangling by Factorizing"
        (https://arxiv.org/pdf/1802.05983).

        Args:
          gamma: Hyperparameter for the regularizer.
        """
        self.gamma = gamma

    def model_fn(self, features, labels, mode, params):
        """TPUEstimator compatible model function."""
        del labels
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_shape = features.get_shape().as_list()[1:]
        z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        z_shuffle = shuffle_codes(z_sampled)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            logits_z, probs_z = architectures.make_discriminator(
                z_sampled, is_training=is_training)
            _, probs_z_shuffle = architectures.make_discriminator(
                z_shuffle, is_training=is_training)
        reconstructions = self.decode(z_sampled, data_shape, is_training)
        per_sample_loss = losses.make_reconstruction_loss(
            features, reconstructions)
        reconstruction_loss = tf.reduce_mean(per_sample_loss)
        kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        standard_vae_loss = tf.add(reconstruction_loss, kl_loss, name="VAE_loss")
        # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
        tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
        tc_loss = tf.reduce_mean(tc_loss_per_sample, axis=0)
        regularizer = kl_loss + self.gamma * tc_loss
        factor_vae_loss = tf.add(
            standard_vae_loss, self.gamma * tc_loss, name="factor_VAE_loss")
        discr_loss = tf.add(
            0.5 * tf.reduce_mean(tf.log(probs_z[:, 0])),
            0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 1])),
            name="discriminator_loss")
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer_vae = optimizers.make_vae_optimizer()
            optimizer_discriminator = optimizers.make_discriminator_optimizer()
            all_variables = tf.trainable_variables()
            encoder_vars = [var for var in all_variables if "encoder" in var.name]
            decoder_vars = [var for var in all_variables if "decoder" in var.name]
            discriminator_vars = [var for var in all_variables if "discriminator" in var.name]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op_vae = optimizer_vae.minimize(
                loss=factor_vae_loss,
                global_step=tf.train.get_global_step(),
                var_list=encoder_vars + decoder_vars)
            train_op_discr = optimizer_discriminator.minimize(
                loss=-discr_loss,
                global_step=tf.train.get_global_step(),
                var_list=discriminator_vars)
            train_op = tf.group(train_op_vae, train_op_discr, update_ops)
            tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            logging_hook = tf.train.LoggingTensorHook({
                "loss": factor_vae_loss,
                "reconstruction_loss": reconstruction_loss
            },
                every_n_iter=50)
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=factor_vae_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=factor_vae_loss,
                eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                             "kl_loss"),
                              [reconstruction_loss, regularizer, kl_loss]))
        else:
            raise NotImplementedError("Eval mode not supported.")


@gin.configurable("dip_vae_logged")
class DIPVAE(LoggedBaseVAE):
    """DIPVAE model."""

    def __init__(self,
                 lambda_od=gin.REQUIRED,
                 lambda_d_factor=gin.REQUIRED,
                 dip_type="i"):
        """Creates a DIP-VAE model.

        Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
        Concepts from Unlabeled Observations"
        (https://openreview.net/pdf?id=H1kG7GZAW).

        Args:
          lambda_od: Hyperparameter for off diagonal values of covariance matrix.
          lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
            lambda_d = lambda_d_factor*lambda_od.
          dip_type: "i" or "ii".
        """
        self.lambda_od = lambda_od
        self.lambda_d_factor = lambda_d_factor
        self.dip_type = dip_type

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        cov_z_mean = compute_covariance_z_mean(z_mean)
        lambda_d = self.lambda_d_factor * self.lambda_od
        if self.dip_type == "i":  # Eq 6 page 4
            # mu = z_mean is [batch_size, num_latent]
            # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
            cov_dip_regularizer = regularize_diag_off_diag_dip(
                cov_z_mean, self.lambda_od, lambda_d)
        elif self.dip_type == "ii":
            cov_enc = tf.matrix_diag(tf.exp(z_logvar))
            expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
            cov_z = expectation_cov_enc + cov_z_mean
            cov_dip_regularizer = regularize_diag_off_diag_dip(
                cov_z, self.lambda_od, lambda_d)
        else:
            raise NotImplementedError("DIP variant not supported.")
        return kl_loss + cov_dip_regularizer


@gin.configurable("beta_tc_vae_logged")
class BetaTCVAE(LoggedBaseVAE):
    """BetaTCVAE model."""

    def __init__(self, beta=gin.REQUIRED):
        """Creates a beta-TC-VAE model.

        Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
        Disentanglement in Variational Autoencoders"
        (https://arxiv.org/pdf/1802.04942).
        If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.

        Args:
          beta: Hyperparameter total correlation.
        """
        self.beta = beta

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
        return tc + kl_loss
