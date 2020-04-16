"""Functions for calculating losses

"""
import tensorflow as tf


def normal_kl_divergence(
    mean_a: tf.Tensor, log_std_a: tf.Tensor, mean_b: tf.Tensor, log_std_b: tf.Tensor
):
    r"""Calculate the KL divergence for two univariate normal distributions

    Taking the means and standard deviations of two univariate normal distributions

    Parameters
    ----------
    mean_a : tf.Tensor
        Mean of first distribution
    log_std_a : tf.Tensor
        Standard deviation of first distribution
    mean_b : tf.Tensor
        Mean of second distribution
    log_std_b : tf.Tensor
        Standard deviation of second distribution

    Returns
    -------
    kl_divergence : tf.Tensor
        KL divergence of univariate normals parameterised by their means and standard deviations

    Notes
    -----
    .. math::
        \text{KL} = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}

    Examples
    --------
    >>> mean_1, mean_2 = tf.zeros(2)
    >>> log_std_1, log_std_2 = tf.zeros(2)
    >>> print(normal_kl_divergence(mean_1, mean_2, log_std_1, log_std_2))
    tf.Tensor(0.0, shape=(), dtype=float32)
    """
    std_a = tf.exp(log_std_a)
    std_b = tf.exp(log_std_b)

    term_1 = log_std_b - log_std_a

    numerator = 0.5 * (std_a ** 2 + (mean_a - mean_b) ** 2)

    term_2 = numerator / std_b ** 2

    kl_divergence = term_1 + term_2 - 0.5

    return kl_divergence
