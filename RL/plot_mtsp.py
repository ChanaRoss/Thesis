import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from generate_data import generate_mtsp_data
from utils import load_model
from problems import MTSP


from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(data, route, ax1, markersize=5):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    depot_marker = '*'
    regular_marker = 's'
    n_cars, tour_length = route.shape
    loc = data['loc']
    n_nodes = loc.shape[0]
    cmap = discrete_cmap(n_cars)
    for i in range(n_cars):
        route_ = route[i, :]
        d = loc.gather(1, route_.unsqueeze(-1).expand(tour_length, 2).type(torch.long))
        for j in range(d.shape[0]):
            if j == 0:
                marker=depot_marker
            else:
                marker=regular_marker
            ax1.scatter(d[j, 0], d[j, 1], color=cmap(i), label ='car id:'+str(i), marker=marker)
            if j+1<d.shape[0]:
                ax1.plot([d[j, 0], d[j+1, 0]], [d[j, 1], d[j+1, 1]], color=cmap(i))



if __name__ == "__main__":
    model, _ = load_model('/Users/chanaross/dev/Thesis/RL/outputs/mtsp_10/mtsp10_rollout_20191203T105431/epoch-49.pt')
    torch.manual_seed(1234)
    dataset = MTSP.make_dataset(size=model.n_nodes, num_samples=10, n_cars=model.n_cars)

    # Need a dataloader to batch instances
    dataloader = DataLoader(dataset, batch_size=1000)

    # Make var works for dicts
    batch = next(iter(dataloader))

    # Run the model
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, pi = model(batch, return_pi=True)
    tours = pi

    # Plot the results
    for i, (data, tour) in enumerate(zip(dataset, tours)):
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)
        # fig.savefig(os.path.join('images', 'cvrp_{}.png'.format(i)))