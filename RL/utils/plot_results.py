from matplotlib import pyplot as plt
import torch


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_results_mtsp(pi, input_data, show=True):
    n_cars, batch_size, tour_length = pi.shape
    loc = input_data['loc']
    n_nodes = loc.shape[0]
    cmap = get_cmap(n_cars)
    for b in range(batch_size):
        for i in range(n_cars):
            pi_ = pi[i, b, :]
            d = loc.gather(1, pi_.unsqueeze(-1).expand(tour_length, 2).type(torch.long))
            for j in range(d.shape[1]):
                plt.scatter(d[b, j, 0], d[b, j, 1], color=cmap(i), label ='car id:'+str(i))
                if j+1<d.shape[1]:
                    plt.plot([d[b, j, 0], d[b, j+1, 0]], [d[b, j, 1], d[b, j+1, 1]], color=cmap(i))
        plt.grid()
    if show:
        plt.show()
    return