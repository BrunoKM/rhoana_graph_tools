import numpy as np


def gaussian_function(x, peak_height=None, std=1.0, mean=0.0):
    """
    The Gaussian function.
    :param x: float or np.ndarray
    :param peak_height: the constant term in the Gaussian function - determines the maximum value of the Gaussian (peak)
    :param std: standard deviation
    :param mean: mean
    :return: float or np.ndarray with same shape as x
    """
    if peak_height is None:
        const_term = 1 / (std * np.sqrt(2 * np.pi))
    else:
        const_term = peak_height
    exponent = -0.5 * np.square((x - mean) / std)
    res = const_term * np.exp(exponent)
    return res


def inverse_gaussian(x, peak_height=None, std=1.0, mean=0.0):
    """
    The inverse Gaussian function. Gives the positive y such that guassian_function(y) = x.
    :param x: float or np.ndarray
    :param peak_height: the constant term in the Gaussian function - determines the maximum value of the Gaussian (peak)
    :param std: standard deviation
    :param mean: mean
    :return: float or np.ndarray with same shape as x
    """
    if peak_height is None:
        const_term = 1 / (std * np.sqrt(2 * np.pi))
    else:
        const_term = peak_height
    x_norm = x / const_term  # Normalise by the constant
    if type(x) is np.ndarray:
        # Only perform on elements that aren't zero:
        x_positive = x_norm[x_norm > 0]

        res = np.zeros(x.shape)
        # Assign value of nan to results that are invalid (x < 0)
        res[:] = np.nan
        # Fill the valid ones with the calculated values
        res[x_norm > 0] = mean + std * np.sqrt(-2 * np.log(x_positive))
    else:
        if x <= 0:
            res = np.nan
        else:
            res = mean + std * np.sqrt(-2 * np.log(x_positive))
    return res
