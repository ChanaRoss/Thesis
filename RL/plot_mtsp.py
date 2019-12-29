import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from generate_data import generate_mtsp_data
from utils import load_model
from problems import MTSP


from matplotlib import pyplot as plt



def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    return plt.cm.get_cmap(base_cmap, N)


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
    for i_c in range(n_cars):
        route_ = route[i_c, :]
        d = loc.gather(0, route_.unsqueeze(-1).expand(tour_length, 2).type(torch.long))
        for j in range(d.shape[0]):
            if j == 0:
                marker=depot_marker
                ax1.scatter([], [], color=cmap(i_c), label='car id:'+str(i_c))
            else:
                marker=regular_marker
            ax1.scatter(d[j, 0], d[j, 1], color=cmap(i_c), marker=marker)
            ax1.text(d[j, 0], d[j, 1], str(j))
            if j+1 < d.shape[0]:
                ax1.plot([d[j, 0], d[j+1, 0]], [d[j, 1], d[j+1, 1]], color=cmap(i_c))



if __name__ == "__main__":
    problem_loc = '/Users/chanaross/dev/Thesis/RL/outputs/mtsp_20/'
    model_loc = []
    # model_loc.append('mtsp10_rollout_20191208T000410/epoch-16.pt')
    # model_loc.append('mtsp10_rollout_20191208T000410/epoch-9.pt')
    # model_loc.append('mtsp10_rollout_20191206T153448/epoch-34.pt')


    # model_loc.append('mtsp10_rollout_20191208T210659/epoch-9.pt')
    # model_loc.append('mtsp10_rollout_20191208T210659/epoch-65.pt')
    # model_loc.append('mtsp10_rollout_20191208T210659/epoch-157.pt')
    # model_loc.append('mtsp10_rollout_20191208T210659/epoch-183.pt')

    # model_loc.append('mtsp10_rollout_20191210T203544/epoch-10.pt')
    # model_loc.append('mtsp10_rollout_20191210T203544/epoch-113.pt')
    # model_loc.append('mtsp10_rollout_20191210T203544/epoch-140.pt')

    model_loc.append('/mtsp20_rollout_20191217T001444/epoch-9.pt')
    # model_loc.append('/mtsp20_rollout_20191217T001444/epoch-26.pt')
    # model_loc.append('/mtsp20_rollout_20191217T001444/epoch-32.pt')
    # model_loc.append('/mtsp20_rollout_20191217T001444/epoch-68.pt')
    model_loc.append('/mtsp20_rollout_20191217T001444/epoch-150.pt')


    # torch.manual_seed(1224)
    torch.manual_seed(40)
    n_samples = 4
    length_out = np.zeros([len(model_loc), n_samples])
    flag_plot_results = True
    fig2, ax2 = plt.subplots(1, 1)
    cmap2 = discrete_cmap(len(model_loc))
    for i_m in range(len(model_loc)):
        model, _ = load_model(problem_loc + model_loc[i_m])
        if i_m == 0:  # create dataset based on first model , then use the same nodes for all models to be checked
            dataset = MTSP.make_dataset(size=model.n_nodes, num_samples=n_samples, n_cars=model.n_cars)
            # Need a dataloader to batch instances
            dataloader = DataLoader(dataset, batch_size=1000)
            # Make var works for dicts
            batch = next(iter(dataloader))
        fig_title = model_loc[i_m]
        # Run the model
        model.eval()
        model.set_decode_type('greedy')
        with torch.no_grad():
            length, log_p, pi = model(batch, return_pi=True)
            length_out[i_m, :] = length.detach().numpy()
            print(model_loc[i_m]+":")
            print(length)
            print("average length is:" + str(length.mean()))
        tours = pi.permute(1, 0, 2)  # new order is [batch_size, n_cars, tour_length]
        if flag_plot_results:
            fig, ax = plt.subplots(n_samples, 1)
            # Plot the results
            for i, (data, tour) in enumerate(zip(dataset, tours)):
                plot_vehicle_routes(data, tour, ax[i])
                ax[i].grid()
            fig.suptitle(fig_title, fontsize=16)

        ax2.plot(range(n_samples), length.detach().numpy(), label='cost -'+model_loc[i_m], marker='*', color=cmap2(i_m))
        plt.legend()
    ax2.grid()
    fig2.legend()
    plt.show()

        # fig.savefig(os.path.join('images', 'cvrp_{}.png'.format(i)))