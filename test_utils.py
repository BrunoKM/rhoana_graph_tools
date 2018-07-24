import numpy as np
import matplotlib.pyplot as plt
import itertools
from utils import *


def test_inverse_function():
    """
    Test the inverse Gaussian by checking whether it returns the same value as initially passed into the Gaussian
    function.
    """
    x = np.linspace(-3, 3, 1000)
    allowed_err = 0.001
    for peak, std in itertools.product([None, 0.5, 1], [0.5, 1, 10]):
        y = gaussian_function(x, peak_height=peak, std=std)
        y_inv = inverse_gaussian(y, peak_height=peak, std=std)

        # Only consider those 'y's that weren't rounded to zero
        err = np.abs(y_inv[y > 0] - np.abs(x[y > 0]))
        assert np.all(err < allowed_err)
        # Assert that for all the 'y's smaller than or equal to zero the inverse is nan
        assert np.all(np.isnan(y_inv[y <= 0]))
    return


def plot_gaussian_function():
    """
    Test the gaussian_function function from utils by plotting it for visual inspection.
    """
    res = 100  # resolution
    x = np.linspace(-2, 2, res)
    y = []
    y_params = [(None, 1.0), (None, 2.0), (1.0, 1.0), (0.5, 0.2)]
    for peak, std in y_params:
        y.append(gaussian_function(x, peak_height=peak, std=std))
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row')

    axarr[0, 0].plot(x, y[0])
    axarr[0, 0].set_title(f"peak_height: {y_params[0][0]}, std: {y_params[0][1]}")
    axarr[0, 1].plot(x, y[1])
    axarr[0, 1].set_title(f"peak_height: {y_params[1][0]}, std: {y_params[1][1]}")
    axarr[1, 0].plot(x, y[2])
    axarr[1, 0].set_title(f"peak_height: {y_params[2][0]}, std: {y_params[2][1]}")
    axarr[1, 1].plot(x, y[3])
    axarr[1, 1].set_title(f"peak_height: {y_params[3][0]}, std: {y_params[3][1]}")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.show()


if __name__ == '__main__':
    pass
    # plot_gaussian_function()
    # test_inverse_function()
