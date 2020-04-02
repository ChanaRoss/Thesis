import torch
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.data import DataLoader
from RL_anticipatory.utils import load_model, torch_load_cpu, to_numpy
from RL_anticipatory.problems import problem_anticipatory
import time
import os
import pickle


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    return plt.cm.get_cmap(base_cmap, N)


def main():
    path = 'outputs/anticipatory_rl_7/anti_with_time_window_20200331T160131/'
    n_samples = 10
    model_files = [f for f in os.listdir(path) if f.endswith('.pt')]
    n_networks = len(model_files)
    mean_cost_out = np.zeros([n_networks])
    mean_num_events_closed = np.zeros([n_networks])
    for i, m_file in enumerate(model_files):
        if i % 10 == 0:
            print("calculating model num:" + str(i))
        model, args, sim_input_dict, stochastic_input_dict = load_model(path + m_file, 'anticipatory')
        # args['no_cuda'] = False
        args['use_cuda'] = torch.cuda.is_available() and not args['no_cuda']
        args['device'] = torch.device("cuda:0" if args['use_cuda'] else "cpu")
        gs = args['graph_size']
        #     model.sim_input_dict['sim_length'] = sim_length
        model.stochastic_input_dict['should_calc_anticipatory'] = False
        model.sim_input_dict['is_training'] = False
        # model.sim_input_dict['print_debug'] = True
        load_data = torch_load_cpu(path + m_file)
        if i == 0:
            dataset = problem_anticipatory.AnticipatoryDataset("", args['n_cars'], args['n_events'],
                                                               args['events_time_window'],
                                                               sim_input_dict['sim_length'], args['graph_size'],
                                                               args['cancel_cost'], args['close_reward'],
                                                               args['movement_cost'], args['open_cost'],
                                                               args['lam'], args['device'], n_samples)
            dataloader = DataLoader(dataset, batch_size=1000)
            batch_data = next(iter(dataloader))
            n_events_closed = np.zeros([n_networks, n_samples, sim_input_dict['sim_length']])
        with torch.no_grad():
            model = model.to(args['device'])
            t_start = time.time()
            _, _, actions_chosen, logits_chosen, cost_chosen, state = model(batch_data)
            t_end = time.time()
            mean_cost_out[i] = np.mean(to_numpy(torch.sum(cost_chosen, axis=1)))
            mean_num_events_closed[i] = np.mean(n_events_closed)
    results = {'cost': mean_cost_out,
               'n_events_closed': mean_num_events_closed}
    pickle.dump(results, open(path + 'models_analysis.p', 'wb'))


if __name__ == '__main__':
    main()
    print("done")