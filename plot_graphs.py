from generate_graphs import *
from graph_utils import rearrange_adj_matrix
import itertools
import plotly
import matplotlib.pyplot as plt
from utils import gaussian_function


def plot_preferential_by_distance(num_nodes, gauss_std, gauss_height):
    adj_matrix, coord, coord_idxs, distances = random_gaussian_preferential_by_dist(num_nodes, gauss_std, gauss_height)

    # Rearrange the adj_matrix to follow the coord_idxs ordering for ease of use
    adj_matrix = rearrange_adj_matrix(adj_matrix, coord_idxs)
    distances = rearrange_adj_matrix(distances, coord_idxs)
    coord = coord[coord_idxs, :]


    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    num_colours = len(tableau20)
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
    # exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(12, 10))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    ticks = [i / 10.0 for i in range(1, 10)]
    plt.yticks(ticks, fontsize=14, alpha=0.8)
    plt.xticks(ticks, fontsize=14, alpha=0.8)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    ax.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.2)
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # Plot the edges (ignoring the x axis):
    for i, j in itertools.product(coord_idxs, repeat=2):
        if adj_matrix[i, j] == 1:
            distance = distances[i, j]
            node1 = coord[i, 1:]
            node2 = coord[j, 1:]
            x = [node1[0], node2[0]]
            y = [node1[1], node2[1]]
            opacity = min(0.2 + 0.3 * (1.45 - distance), 0.75)
            color_idx = int(((sum(x)*num_colours + sum(y))/14) * num_colours) % num_colours
            # color_idx = int(((sum(x)**2 + sum(y)**2) / 8) * num_colours)
            # color_idx = int((min((0.5 - distance)/0.5, 0.99)) * 10)
            # colors = [(0.1, 0.45 - 0.3*(x/90), x / 90) for x in range(40, 91, 5)]
            color = tableau20[color_idx]
            color = [min(i * (distance)/0.18, 1.0) for i in color]
            plt.plot(x, y, 'o-', color=color, alpha=opacity)
    plt.scatter(coord[:, 1], coord[:, 2], color=(0.8, 0.1, 0.5), alpha=1)
    plt.savefig(f"generated_preferential_attachment_{gauss_std}std.png", bbox_inches="tight", dpi=250)
    plt.show()


    # Plot the distance edge histogram
    dist_array = np.ravel(distances)[np.ravel(adj_matrix != 0)]
    histogram, bins = np.histogram(dist_array, bins=60, range=[0.001, 1.001], density=True)
    histogram_norm = histogram / np.max(histogram)
    print(histogram)
    plt.plot(bins[:-1], histogram_norm, color=tableau20[7], alpha=0.6, lw=1.4, label='Distribution of edge lengths')
    x = np.linspace(0.01, 1, 1000)
    gaussian = gaussian_function(x, peak_height=1, std=gauss_std)
    plt.plot(x, gaussian, color=tableau20[9], alpha=0.8, lw=3.4, label='Distance $\mapsto$ edge probability function')
    # Lastly, plot the distribution of all the distances for comparison

    dist_array = np.ravel(distances)
    histogram, bins = np.histogram(dist_array, bins=60, range=[0.001, 1.001], density=True)
    histogram_norm = histogram / np.max(histogram)
    plt.plot(bins[:-1], histogram_norm, color=tableau20[0], alpha=0.2, lw=2, label='Distribution of all distances')
    plt.legend()
    plt.savefig("histogram.png", bbox_inches="tight", dpi=250)
    return


if __name__ == '__main__':
    # plot_preferential_by_distance(200, 0.125, 1)
    plot_preferential_by_distance(1000, 0.07, 1)