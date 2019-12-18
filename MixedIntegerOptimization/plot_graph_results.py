import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pickle



def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    return plt.cm.get_cmap(base_cmap, N)


def draw_graph(adj_mat, events_loc):
    h_mat = np.ones_like(adj_mat)
    H = nx.from_numpy_matrix(h_mat)
    G = nx.from_numpy_matrix(adj_mat)
    events_loc_dict = {}
    for i_e in range(events_loc.shape[0]):
        events_loc_dict[i_e] = tuple(events_loc[i_e, :])

    plt.figure()
    nx.draw(G, with_labels=True, pos=events_loc_dict)
    plt.figure()
    nx.draw(H, with_labels=True, pos=events_loc_dict)
    return G


def draw_path(car_paths, events_loc, is_picked_up):
    color_c = discrete_cmap(len(car_paths))
    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    for i_c in range(len(car_paths)):
        cur_car_path = car_paths[i_c]
        s_loc = cur_car_path[0]
        ax.scatter(s_loc[0], s_loc[1], marker='*', color=color_c(i_c), label='car #'+str(i_c))
        ax2.scatter(s_loc[0], s_loc[1], marker='o', s=50, color='m', label='car #' + str(i_c))
        ax.text(s_loc[0], s_loc[1], '0', color=color_c(i_c))
        ax2.text(s_loc[0], s_loc[1], str(i_c), color=color_c(i_c))
        for j in range(0, len(cur_car_path) - 1):
            ax.plot([s_loc[0], cur_car_path[j + 1][0]], [s_loc[1], cur_car_path[j + 1][1]], marker='.',
                    color=color_c(i_c), alpha=0.5)
            s_loc = cur_car_path[j + 1]
    for i_e in range(events_loc.shape[0]):
        if is_picked_up[i_e]:
            color_e = 'k'
        else:
            color_e = 'r'
        ax.scatter(events_loc[i_e, 0], events_loc[i_e, 1], marker='s', s=45, color=color_e, alpha=1)
        ax2.scatter(events_loc[i_e, 0], events_loc[i_e, 1], marker='s', s=45, color='k', alpha=1)
        ax.text(events_loc[i_e, 0], events_loc[i_e, 1]+0.2, str(i_e))
        ax2.text(events_loc[i_e, 0], events_loc[i_e, 1] + 0.2, str(i_e))
    ax.grid()
    ax2.grid()
    # ax2.set_xlim([-1, 4])
    # ax2.set_ylim([-1, 4])
    # fig.legend()


def main():
    data_loc = '/Users/chanaross/dev/Thesis/MixedIntegerOptimization/'
    data_name = 'optimizationResults'

    data = pickle.load(open(data_loc + data_name + '.p', 'rb'))
    adj_mat = data['sol']['cEventsToEvents']
    draw_path(data['cars_paths'], data['event_pos'], data['sol']['isPickedUp'])
    G = draw_graph(adj_mat, data['event_pos'])
    plt.show()
    return



if __name__ == '__main__':
    main()
    print('done!')