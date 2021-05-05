import numpy as np
from disentanglement_lib.evaluation.metrics import utils as dlib_utils
from disentanglement_lib.evaluation.metrics import unsupervised_metrics as dlib_unsupervised_metrics


def discretize(samples, num_bins=20):
    """ Discretization based on histograms implemented in disentanglement lib.

    :param samples: sampled data with size (n_factors * n_samples)
    :param num_bins: number of bins to use, default is 20.
    :return: discretized samples.
    """
    discretized = np.zeros_like(samples)

    for i in range(samples.shape[0]):
        discretized[i, :] = np.digitize(samples[i, :], np.histogram(samples[i, :], num_bins)[1][:-1])

    return discretized


def averaged_mi(cov):
    """ Generate averaged continuous mutual information for jointly gaussian distributions

    :param cov: covariance matrix of size (n_factors * n_factors)
    :return: Averaged continuous mutual information
    """
    k = cov.shape[0]
    mi_sum = 0

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            corr2 = (cov[i, j]) ** 2 / (cov[i, i] * cov[j, j])
            mi = - 1 / 2 * np.log(1 - corr2)
            if not np.isnan(mi):
                mi_sum += mi

    mi_avg = 1 / (k ** 2 - k) * mi_sum

    return mi_avg


def discrete_averaged_mi(samples):
    """ Generate averaged discrete mutual information based on Locatello et al. 2019 implementation

    :param samples: sampled data with size (n_factors * n_samples)
    :return: Averaged discrete mutual information
    """
    k = samples.shape[0]

    norm_mutual_info_matrix = dlib_utils.discrete_mutual_info(samples, samples)
    np.fill_diagonal(norm_mutual_info_matrix, 0)
    discrete_mi_avg = np.sum(norm_mutual_info_matrix) / (k ** 2 - k)

    return discrete_mi_avg


def generate_cov_mat(factors, active_variables, corr_min, corr_max):
    """ Generate synthetic covariance matrices for mean and sampled representations, where the only difference between
    the two is that passive variables have a variance of 1 in sampled representations and of 0.02 in mean ones.
    Active variable have correlation scores randomly picked in (corr_min, corr_max) range.
    
    :param factors: dimensionality of the latent representation
    :param active_variables: number of active variables
    :param corr_min: minimum correlation between active variables
    :param corr_max: maximum correlation between active variables
    :return: mean and sampled covariance matrices
    """
    cov_m = np.zeros([factors, factors])
    range_size = (corr_max - corr_min)
    a_corr = np.random.rand((active_variables * (active_variables - 1)) // 2) * range_size + corr_min

    for i in range(factors):
        var = 1 if i < active_variables else 0.02
        cov_m[i, i] = var
        for j in range(i):
            corr = a_corr[(i + j) - 1] if var == 1 else 0.01
            cov_m[i, j] = corr
            cov_m[j, i] = corr

    cov_s = np.array(cov_m, copy=True)
    np.fill_diagonal(cov_s, 1.)

    return cov_m, cov_s


def compare_metrics(factors, active_vars, corr_min=0.2, corr_max=0.2):
    """ Generate synthetic data from gaussian distributions emulating mean and variance representations with varying
    number of factors and active variables.

    :param factors: dimensionality of the latent representation
    :param active_vars: number of active variables
    :param corr_min: minimum correlation between active variables
    :param corr_max: maximum correlation between active variables
    :return: None
    """
    if factors < active_vars or active_vars < 0 or factors <= 0:
        raise ValueError("The number of active variable must be lower or equal to the total number of variables and "
                         "both values must be positive.")

    # Using the same seed and number of sample as in Locatello et al. experiment
    np.random.seed(2051556033)
    n = 10000
    mu = np.repeat(0, factors)

    cov_m, cov_s = generate_cov_mat(factors, active_vars, corr_min, corr_max)
    samples_m, samples_s = np.random.multivariate_normal(mu, cov_m, n).T, np.random.multivariate_normal(mu, cov_s, n).T
    cov_m, cov_s = np.cov(samples_m), np.cov(samples_s)

    print("{:<20} {:<15} {:<15}".format("Representation", "Mean", "Sampled"))
    print('{:-^50}'.format(""))
    print("{:<20} {:<15f} {:<15f}".format("TC", dlib_unsupervised_metrics.gaussian_total_correlation(cov_m),
                                          dlib_unsupervised_metrics.gaussian_total_correlation(cov_s)))
    print("{:<20} {:<15f} {:<15f}".format("Avg MI", averaged_mi(cov_m), averaged_mi(cov_s)))
    print("{:<20} {:<15f} {:<15f}".format("Discrete avg MI", discrete_averaged_mi(discretize(samples_m)),
                                          discrete_averaged_mi(discretize(samples_s))))
    print("{:<20} {:<15f} {:<15f}".format("L2-Wasserstein",
                                          dlib_unsupervised_metrics.gaussian_wasserstein_correlation(cov_m),
                                          dlib_unsupervised_metrics.gaussian_wasserstein_correlation(cov_s)))
