import numpy as np
import torch
import time
from torch_geometric.data import DataLoader
from matplotlib import pyplot as plt
# my files
from MixedIntegerOptimization.offlineOptimizationProblem_unlimited_time_multiple_depots import run_mtsp_opt, analysis_and_plot_results
from RL_anticipatory.utils import load_model
from RL_anticipatory.problems import problem_anticipatory


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    return plt.cm.get_cmap(base_cmap, N)


def plot_vehicle_routes(route, ax1, markersize=5):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    depot_marker = '*'
    regular_marker = 's'
    n_cars, tour_length, _ = route.shape
    cmap = discrete_cmap(n_cars)
    for i_c in range(n_cars):
        route_ = route[i_c, :]
        for j in range(route_.shape[0]):
            if j == 0:
                marker = depot_marker
            else:
                marker = regular_marker
                ax1.plot(route[j-1:j, 0], route[j-1:j, 1])
            ax1.text(route_[j, 0], route_[j, 1], str(j))
            ax1.scatter(route_[j, 0], route_[j, 1], color=cmap(i_c), marker=marker)


def plot_events_loc():
    # put events location on grid with opening time as text

    print("hi")


def main():
    figures_loc = '/Users/chanaross/dev/Thesis/RL_anticipatory/figures/'
    problem_loc = '/Users/chanaross/dev/Thesis/RL_anticipatory/outputs/'
    model_loc = []

    # ********************************** grid : 10 ********************************** #
    model_loc.append('anticipatory_rl_15/anticipatory_rl_20200221T145415/epoch-4.pt')
    # model_loc.append('anticipatory_rl_15/anticipatory_rl_20200221T145415/epoch-100.pt')
    model_loc.append('anticipatory_rl_15/anticipatory_rl_20200221T145415/epoch-199.pt')

    seed = 1234
    # torch.manual_seed(1224)
    torch.manual_seed(seed)
    n_samples = 4
    length_out = np.zeros([len(model_loc), n_samples])
    flag_plot_results = True

    fig2, ax2 = plt.subplots(1, 1)
    cmap2 = discrete_cmap(len(model_loc) + 1)
    cost_models = np.zeros([len(model_loc), n_samples])
    state_dict_models = []
    states_out = []
    for i_m in range(len(model_loc)):
        model, args, sim_input_dict, stochastic_input_dict = load_model(problem_loc + model_loc[i_m])
        if i_m == 0:  # create dataset based on first model , then use the same nodes for all models to be checked
            dataset = problem_anticipatory.AnticipatoryDataset("", args['n_cars'], args['events_time_window'],
                                                               sim_input_dict['sim_length'], args['graph_size'],
                                                               args['cancel_cost'], args['close_reward'],
                                                               args['movement_cost'], args['open_cost'],
                                                               args['lam'], n_samples)
            # Need a dataloader to batch instances
            dataloader = DataLoader(dataset, batch_size=1000)
            # Make var works for dicts
            batch_data = next(iter(dataloader))
        fig_title = model_loc[i_m]
        # Run the model
        model.eval()
        model.set_decode_type('sampling')
        state_dict_models.append(model.state_dict())
        with torch.no_grad():
            s_time = time.time()
            tour_length = sim_input_dict['sim_length']+1
            n_repeats = 1
            temp_car_routes = torch.zeros([n_repeats, n_samples, model.n_cars, tour_length, 2])
            temp_costs = torch.zeros([n_repeats, n_samples])
            for i in range(n_repeats):
                costs_all_options, logits_all_options, actions_chosen, logits_chosen, cost_chosen, state = model(batch_data)
                states_out.append(state)
                temp_costs[i, ...] = cost_chosen.sum(1)
                temp_car_routes[i, ...] = state.cars_route
            cost = torch.zeros(n_samples)
            cars_route = torch.zeros([n_samples, model.n_cars, tour_length, 2])
            best_indexs = torch.argmin(temp_costs, 0)
            for i_b in range(n_samples):
                cost[i_b] = temp_costs[best_indexs[i_b], i_b]
                cars_route[i_b, ...] = temp_car_routes[best_indexs[i_b], i_b, ...]
            e_time = time.time()
            print("***************************************************")
            print(model_loc[i_m] + ":")
            print("model run time:"+str(e_time-s_time))
            print(cost)
            print("average length is:" + str(cost.mean()))
            print("max length is: " + str(cost.max()))
            print("min length is:" + str(cost.min()))
            print("std length is: " + str(cost.std()))
        n_splots = int(np.ceil(n_samples / 2))
        if flag_plot_results:
            fig, ax = plt.subplots(n_splots, 2)
            # Plot the results
            for i, (data, route) in enumerate(zip(dataset, cars_route)):
                x_p = i // 2
                y_p = i % 2
                plot_vehicle_routes(route, ax[x_p][y_p])
                ax[x_p][y_p].grid()
            fig.suptitle(fig_title, fontsize=16)

        ax2.plot(range(n_samples), cost.detach().numpy(), label='cost -' + model_loc[i_m], marker='*',
                 color=cmap2(i_m))
        cost_models[i_m, :] = cost.detach().numpy()
        if flag_plot_results:
            ax[0][0].legend()
    print("hi")
    plt.show()

    # # run optimization for comparison
    # opt_file_name = 'opt_results_' + str(n_samples) + '_seed' + str(seed)
    # events_batch_loc = batch_data['loc'].detach().numpy()
    # events_batch_depot = batch_data['depot'].detach().numpy()
    # events_batch_car_loc = batch_data['car_loc'].detach().numpy()
    # opt_cost = np.zeros(n_samples)
    # opt_cost_same_length = np.zeros(n_samples)
    # if flag_plot_results:
    #     fig_opt, ax_opt = plt.subplots(n_splots, 2)
    #     fig_same_length, ax_same_length = plt.subplots(n_splots, 2)
    # opt_time = 0
    # for i in range(events_batch_loc.shape[0]):
    #     car_loc = events_batch_car_loc[i, ...]
    #     depot = events_batch_depot[i, ...]
    #     events_loc = events_batch_loc[i, ...]
    #     s_time = time.time()
    #     print("starting opt num:"+str(i))
    #     # run optimization for same length routes -
    #     same_length_routes = True
    #     m, obj = run_mtsp_opt(car_loc, events_loc, depot, same_length_routes, False)
    #     e_time = time.time()
    #     opt_time += e_time-s_time
    #     # # run analysis and plot results
    #     data_out = analysis_and_plot_results(m, car_loc, events_loc, depot,
    #                                          False, figures_loc, opt_file_name, 10)
    #
    #     opt_cost_same_length[i] = obj.getValue()
    #     if flag_plot_results:
    #         x_p = i // 2
    #         y_p = i % 2
    #         plot_opt_results(data_out[1:, 1:], events_loc, car_loc, ax_same_length[x_p][y_p])
    #         ax_same_length[x_p][y_p].grid()
    #         fig_same_length.suptitle('Optimization results - same length', fontsize=16)
    #
    #     # run optimization for differnt length routes
    #     same_length_routes = False
    #     m, obj = run_mtsp_opt(car_loc, events_loc, depot, same_length_routes, False)
    #     e_time = time.time()
    #     opt_time += e_time - s_time
    #     # # run analysis and plot results
    #     data_out = analysis_and_plot_results(m, car_loc, events_loc, depot,
    #                                          False, figures_loc, opt_file_name, 10)
    #
    #     opt_cost[i] = obj.getValue()
    #     if flag_plot_results:
    #         x_p = i // 2
    #         y_p = i % 2
    #         plot_opt_results(data_out[1:, 1:], events_loc, car_loc, ax_opt[x_p][y_p])
    #         ax_opt[x_p][y_p].grid()
    #         fig_opt.suptitle('Optimization results - different length', fontsize=16)
    #
    #
    # ax2.plot(range(n_samples), opt_cost_same_length, label='cost- optimization same length', marker='s',
    #          color=cmap2(i_m + 1))
    # ax2.plot(range(n_samples), opt_cost, label='cost- optimization different length', marker='s',
    #          color=cmap2(i_m + 1))
    # ax2.grid()
    # ax2.set_ylim([0, 80])
    # print("***************************************************")
    # print("optimization results:")
    # print("opt tot time:"+str(opt_time))
    # print("cost:"+str(opt_cost))
    # print("average cost is:" + str(np.mean(opt_cost)))
    # print("std cost : " + str(np.std(opt_cost)))
    # print("min cost : " + str(np.min(opt_cost)))
    # print("max cost is:" + str(np.max(opt_cost)))
    # fig2.legend()
    #
    # fig_diff, ax_diff = plt.subplots(1, 1)
    # for i_m in range(len(model_loc)):
    #     ax_diff.plot(range(n_samples), cost_models[i_m, :]/opt_cost, label=model_loc[i_m], marker='*')
    # ax_diff.grid()
    # ax_diff.set_xlabel('Run Num #')
    # ax_diff.set_ylabel('|C_model - C_opt|')
    # fig_diff.suptitle('Difference between baseline and models output')
    # fig_diff.legend()
    # fig.savefig(os.path.join('images', 'cvrp_{}.png'.format(i)))


if __name__ == "__main__":
    main()
    print("done!")