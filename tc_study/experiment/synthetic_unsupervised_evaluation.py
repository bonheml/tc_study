import pandas as pd
from pathlib import Path
import numpy as np
from tc_study.experiment import logger
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
    counter = 0

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            corr2 = (cov[i, j]) ** 2 / (cov[i, i] * cov[j, j])
            mi = - 1 / 2 * np.log(1 - corr2)
            if not np.isnan(mi):
                mi_sum += mi
                # only increment counter if value is added to prevent averaging over discarded values
                counter += 1
            else:
                logger.warning("Nan value encountered for MI(z{},z{})".format(i, j))

    mi_avg = 1 / counter * mi_sum

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


def generate_cov_mat(factors, active_variables, cov_min, cov_max, random_state, pv_var=0.02):
    """ Generate synthetic covariance matrices for mean and sampled representations, where the only difference between
    the two is that passive variables have a variance of 1 in sampled representations and of pv_var in mean ones.
    Active variable have covariance scores randomly picked in (cov_min, cov_max) range.
    
    :param factors: dimensionality of the latent representation
    :param active_variables: number of active variables
    :param cov_min: minimum covariance between active variables
    :param cov_max: maximum covariance between active variables
    :param random_state: np random state used for seeded random generation
    :param pv_var: passive variables variance
    :return: mean and sampled covariance matrices
    """
    cov_m = np.zeros([factors, factors])
    range_size = (cov_max - cov_min)
    a_corr = random_state.rand((active_variables * (active_variables - 1)) // 2) * range_size + cov_min

    for i in range(factors):
        var = 1 if i < active_variables else pv_var
        cov_m[i, i] = var
        for j in range(i):
            corr = a_corr[(i + j) - 1] if var == 1 else 0.01
            cov_m[i, j] = corr
            cov_m[j, i] = corr

    cov_s = np.array(cov_m, copy=True)
    np.fill_diagonal(cov_s, 1.)

    return cov_m, cov_s


def print_results(scores, seed):
    l = len(scores)
    truncated = "Truncated" if scores[0]["truncated"] is True else "Full"
    scores_str = "{:<20} {:<15f} {:<15f}\n"

    msg = "\nSeed {} - {} representations - active variables: {}\n".format(seed, truncated, scores[0]["active_variables"])
    msg += "{:<20} {:<15} {:<15}\n".format("Representation", *[scores[i]["representation"] for i in range(l)])
    msg += "{:-^65}\n".format("")
    msg += scores_str.format("TC", *[scores[i]["gaussian_total_correlation"] for i in range(l)])
    msg += scores_str.format("Avg MI", *[scores[i]["continuous_mutual_info_score"] for i in range(l)])
    msg += scores_str.format("Discrete avg MI", *[scores[i]["discrete_mutual_info_score"] for i in range(l)])
    msg += scores_str.format("L2-Wasserstein", *[scores[i]["gaussian_wasserstein_correlation"] for i in range(l)])
    logger.info(msg)


def generate_gaussian_examples(factors,  active_vars, random_state, cov_min=0.2, cov_max=0.2, noise_strength=0.02,
                               pv_var=0.02, n=10000):
    mu = np.repeat(0, factors)
    samples = []

    cov_m, cov_s = generate_cov_mat(factors, active_vars, cov_min, cov_max, random_state, pv_var)
    samples += [random_state.multivariate_normal(mu, cov_m, n).T]
    sigma = np.diag([noise_strength if i < active_vars else 1 for i in range(factors)])
    samples_n = random_state.multivariate_normal(mu, np.identity(factors), n)
    samples.append(samples[0] + np.dot(samples_n, sigma).T)
    covs = [np.cov(s) for s in samples]

    return covs, samples


def compare_metrics(factors, active_vars, seed, truncated=False, gaussian=True, distrib=(0.2, 0.2),
                    noise_strength=0.02, pv_var=0.02, verbose=False):
    """ Generate synthetic data from gaussian distributions emulating mean and variance representations with varying
    number of factors and active variables.

    :param factors: dimensionality of the latent representation
    :param active_vars: number of active variables
    :param seed: Integer used to fix the random number generation
    :param truncated: False if not truncated, true otherwise
    :param gaussian: False if ring/torus, true otherwise
    :param distrib: if gaussian tuple with (min_cov, max_cov) else (tube_radius, hole_radius)
    :param noise_strength: The noise applied to active variables sampled from mean representations
    :param pv_var: passive variables variance
    :param verbose: If true, print the result, else execute silently
    :return: list of dict of scores
    """
    random_state = np.random.RandomState(seed)
    scores = []

    cov_min, cov_max = distrib
    covs, samples = generate_gaussian_examples(factors, active_vars, random_state, cov_min=cov_min, cov_max=cov_max,
                                               noise_strength=noise_strength, pv_var=pv_var)

    representations = ["mean", "sampled"]
    for i, rep in enumerate(representations):
        cov = covs[i]
        r = {}
        r["representation"] = rep
        r["gaussian_total_correlation"] = dlib_unsupervised_metrics.gaussian_total_correlation(cov)
        r["continuous_mutual_info_score"] = averaged_mi(cov)
        r["discrete_mutual_info_score"] = discrete_averaged_mi(discretize(samples[i]))
        r["gaussian_wasserstein_correlation"] = dlib_unsupervised_metrics.gaussian_wasserstein_correlation(cov)
        r["active_variables"] = active_vars
        r["truncated"] = truncated
        scores.append(r)

    if verbose is True:
        print_results(scores, seed)

    return scores


def gaussian_metrics_comparison(factors, out_path, seed, cov_min=0.2, cov_max=0.2, noise_strength=0.02, pv_var=0.02,
                                verbose=False):
    """ Generate synthetic data from gaussian distributions emulating mean and variance representations with 0 to
    <nb_factors> active variables.

    :param factors: dimensionality of the latent representation
    :param out_path: Folder used to save the results
    :param seed: Integer used to fix the random number generation
    :param cov_min: minimum covariance between active variables
    :param cov_max: maximum covariance between active variables
    :param noise_strength: The noise applied to active variables sampled from mean representations
    :param pv_var: passive variables variance
    :param verbose: If true, print the result, else execute silently
    :return: None
    """
    out_path = Path(out_path).absolute()
    fname = str(out_path / "seed_{}_{}_cov_{}_{}_noise_{}_synthetic.tsv".format(seed, factors, cov_min, cov_max, noise_strength))

    res = []
    for va in range(factors + 1):
        res += compare_metrics(factors, va, seed, truncated=False, distrib=(cov_min, cov_max),
                               noise_strength=noise_strength, pv_var=pv_var, verbose=verbose)
        if va >= 2:
            res += compare_metrics(va, va, seed, truncated=True, distrib=(cov_min, cov_max),
                                   noise_strength=noise_strength, pv_var=pv_var, verbose=verbose)
    df = pd.DataFrame(res)
    df["num_factors"] = factors
    df["cov_min"] = cov_min
    df["cov_max"] = cov_max
    df["noise_strength"] = noise_strength
    df["seed"] = seed
    df.to_csv(fname, sep="\t", index=False)
