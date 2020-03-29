import torch
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.data import DataLoader
from RL_anticipatory.utils import load_model, torch_load_cpu, to_numpy
from RL_anticipatory.problems import problem_anticipatory
from MixedIntegerOptimization.offlineOptimizationProblem_TimeWindow_rl import runMaxFlowOpt, plotResults
from UtilsCode.createGif import create_gif
import time
import os


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    return plt.cm.get_cmap(base_cmap, N)


def plot_vehicle_routes(route, ax1, cmap):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    depot_marker = '*'
    regular_marker = 's'
    for j in range(route.shape[0]):
        if j == 0:
            marker = depot_marker
        else:
            marker = regular_marker
            ax1.plot([route[j-1, 0], route[j, 0]], [route[j-1, 1], route[j, 1]], color=cmap)
        # ax1.text(route[j, 0], route[j, 1], str(j))
        # ax1.scatter(route[j, 0], route[j, 1], color=cmap, marker=marker)


def plot_events(events_loc, ax1):
    for j in range(events_loc.shape[0]):
        ax1.scatter(events_loc[j, 0], events_loc[j, 1], color='r', marker='o')
        ax1.text(events_loc[j, 0], events_loc[j, 1], str(j))


def plot_current_time_anticipatory(cur_time, cars_loc, events_loc, events_time, events_answered_time, nc, ne, gs, fileName):
    fig, ax = plt.subplots()
    ax.set_title('time: {0}'.format(cur_time))
    for c in range(nc):
        ax.scatter(cars_loc[c, 0], cars_loc[c, 1], c='k', alpha=1, marker='s')
    ax.scatter([], [], c='y', label='Future Requests')
    ax.scatter([], [], c='b', label='Opened')
    # ax.scatter([], [], c='b', label='Opened commited')
    ax.scatter([], [], c='r', label='Canceled')
    ax.scatter([], [], c='g', label='Closed')
    for i in range(ne):
        if events_answered_time[i] < cur_time:
            ax.scatter(events_loc[i, 0], events_loc[i, 1], c='g', alpha=0.7, marker='o', s=40)
        elif (events_time[i, 0] < cur_time) and (events_time[i, 1] > cur_time) and (cur_time <= events_answered_time[i]):
            ax.scatter(events_loc[i, 0], events_loc[i, 1], c='b', alpha=0.7, marker='o', s=40)
        elif events_time[i, 0] > cur_time:
            ax.scatter(events_loc[i, 0], events_loc[i, 1], c='y', alpha=0.7, marker='o', s=40)
        elif events_time[i, 1] < cur_time:
            ax.scatter(events_loc[i, 0], events_loc[i, 1], c='r', alpha=0.7, marker='o', s=40)
        else:
            ax.scatter(events_loc[i, 0], events_loc[i, 1], c='b', alpha=0.7, marker='o', s=40)
    ax.set_xlim([-1, gs + 1])
    ax.set_ylim([-1, gs + 1])
    ax.grid(True)
    plt.legend()
    plt.savefig(fileName + '_' + str(cur_time) + '.png')
    plt.close()
    return


def main():
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    network_path = 'outputs/anticipatory_rl_7/'
    fig_save_loc = 'figures/'
    network_names = []
    n_samples = 4
    sim_length = 15
    flag_plot_results = False
    flag_create_gif = True
    run_optimal_results = True
    # network_names.append('anti_with_time_window_20200317T094536/epoch-2741
    network_names.append('mtsp_20200326T235650/epoch-2')
    # network_names.append('mtsp_20200326T235650/epoch-546')
    network_names.append('mtsp_20200327T134224/epoch-2497')
    network_names.append('mtsp_20200329T000930/epoch-2566')

    # network_names.append('anti_with_time_window_20200324T094029/epoch-2911')


    # network_names.append('mtsp_10_20200310T200705/epoch-2715')
    # network_names.append('mtsp_10_20200310T200705/epoch-2550')
    # network_names.append('mtsp_10_20200310T200705/epoch-2388')
    # network_names.append('mtsp_10_20200304T110447/epoch-1250')
    # network_names.append('mtsp_10_20200303T182956/epoch-2')


    fig_c, ax_c = plt.subplots(1, 1)
    cmap = discrete_cmap(len(network_names)+1)
    n_splots = int(np.ceil(n_samples / 2))
    for i_n, network_name in enumerate(network_names):
        if flag_plot_results:
            fig, ax = plt.subplots(n_splots, 2)
        model, args, sim_input_dict, stochastic_input_dict = load_model(network_path + network_name + '.pt', 'anticipatory')
        args['no_cuda'] = False
        args['use_cuda'] = torch.cuda.is_available() and not args['no_cuda']
        args['device'] = torch.device("cuda:0" if args['use_cuda'] else "cpu")
        gs = args['graph_size']
        model.sim_input_dict['sim_length'] = sim_length
        model.stochastic_input_dict['should_calc_anticipatory'] = False
        # model.sim_input_dict['print_debug'] = True
        # cars_route = torch.zeros([n_samples, args['n_cars'], sim_input_dict['sim_length'], 2])
        load_data = torch_load_cpu(network_path+network_name + '.pt')
        if i_n == 0:
            cmap_v = discrete_cmap(args['n_cars']+10)
            dataset = problem_anticipatory.AnticipatoryDataset("", args['n_cars'], args['n_events'], args['events_time_window'],
                                                               sim_input_dict['sim_length'], args['graph_size'],
                                                               args['cancel_cost'], args['close_reward'],
                                                               args['movement_cost'], args['open_cost'],
                                                               args['lam'], args['device'], n_samples)
            dataloader = DataLoader(dataset, batch_size=1000)
            batch_data = next(iter(dataloader))
        t_start = time.time()
        with torch.no_grad():
            model = model.to(args['device'])
            costs_all_options, logits_all_options, actions_chosen, logits_chosen, cost_chosen, state = model(batch_data)
            cars_route = to_numpy(state.cars_route)
            all_events_loc = state.events_loc_dict
            all_events_time = state.events_time_dict
        t_end = time.time()
        total_cost = torch.sum(cost_chosen, dim=1)
        mean_cost = torch.mean(total_cost)
        std_cost = torch.std(total_cost)
        ax_c.plot(range(n_samples), to_numpy(total_cost), marker='*', color=cmap(i_n), label=network_name)
        if flag_plot_results:
            # Plot the results
            fig_title = network_name
            for i, (events, routes) in enumerate(zip(all_events_loc.values(), cars_route)):
                x_p = i // 2
                y_p = i % 2
                for i_c in range(args['n_cars']):
                    route = routes[i_c, ...]
                    plot_vehicle_routes(route, ax[x_p][y_p], cmap_v(i_c))
                plot_events(to_numpy(events), ax[x_p][y_p])
                ax[x_p][y_p].grid()
            fig.suptitle(fig_title, fontsize=16)
        if flag_create_gif:
            if not os.path.exists(fig_save_loc + network_name):
                os.makedirs(fig_save_loc + network_name)
            for i, (events_time, events_loc, routes) in enumerate(zip(all_events_time.values(), all_events_loc.values(), cars_route)):
                fig_name = 'gif'+'_'+str(i)
                n_events = events_loc.shape[0]
                for t in range(sim_input_dict['sim_length']):
                    events_answered_time = to_numpy(state.events_answer_time)[i, ...]
                    plot_current_time_anticipatory(t, routes[:, t, :], events_loc, events_time, events_answered_time,
                                                   args['n_cars'], n_events, gs, fig_save_loc + network_name + '/' + fig_name)
                list_names = [fig_name + '_' + str(t) + '.png' for t in range(sim_input_dict['sim_length'])]
                create_gif(fig_save_loc + network_name + '/', list_names, 1, fig_name)
                [os.remove(fig_save_loc + network_name + '/' + f) for f in list_names]
        print("****************************************************************")
        print("network  - "+network_name)
        print("total run time for " + str(n_samples) + ", is:" + str(t_end - t_start))
        print("mean cost for net:"+str(to_numpy(mean_cost))+", -+ std:"+str(to_numpy(std_cost)))
        n_events_closed = to_numpy(state.events_status['answered'].sum(1).view(-1))
        print("num  events closed:" + str(n_events_closed))
        print("mean num events closed:" + str(np.mean(n_events_closed)))
    if run_optimal_results:
        opt_cost = np.zeros(n_samples)
        n_events_closed_opt = np.zeros(n_samples)
        cars_loc = to_numpy(state.cars_route)[:, :, 0, :]
        all_events_loc = state.events_loc_dict
        all_events_time = state.events_time_dict
        close_reward  = args['close_reward']
        cancel_penalty = args['cancel_cost']
        open_penalty = args['open_cost']
        movement_penalty = args['movement_cost']
        t_start_opt = time.time()
        for i_b in range(n_samples):
            if to_numpy(all_events_time[i_b])[:, 0].size > 0:
                m, obj = runMaxFlowOpt(sim_length, 0, cars_loc[i_b, ...], to_numpy(all_events_loc[i_b]),
                                       to_numpy(all_events_time[i_b])[:, 0], to_numpy(all_events_time[i_b])[:, 1],
                                       close_reward, cancel_penalty, open_penalty, movement_penalty, outputFlag=0)
                opt_cost[i_b] = -obj.getValue()
                dataOut, param, cars_paths = plotResults(m, cars_loc[i_b, ...], to_numpy(all_events_loc[i_b]), to_numpy(all_events_time[i_b])[:, 0],
                                                         to_numpy(all_events_time[i_b])[:, 1], False,
                                                         fig_save_loc, "opt_with_time_window"+str(i_b), args['graph_size'])
                n_events_closed_opt[i_b] = dataOut['closedEvents'][-1]
            else:
                opt_cost[i_b] = 0
                n_events_closed_opt[i_b] = 0
        t_end_opt = time.time()
        print("****************************************************************")
        print("optimization results - ")
        print("total run time for " + str(n_samples) + ", is:" + str(t_end_opt - t_start_opt))
        print("mean cost:"+str(np.mean(opt_cost)) + " -+ std:" +str(np.std(opt_cost)))
        print("optimal cost is:" + str(opt_cost))
        print("num events closed :"+ str(n_events_closed_opt))
        print("mean num events closed:" + str(np.mean(n_events_closed_opt)))
        ax_c.plot(range(n_samples), opt_cost, marker='o', color=cmap(i_n+1), label='optimal results')
        ax_c.set_xlabel('n sample [#]')
        ax_c.set_ylabel('Cost')
    ax_c.grid()
    ax_c.legend()
    plt.show()
    return


if __name__ == '__main__':
    main()
    print("done!")

