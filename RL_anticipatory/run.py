# torch imports -
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
# math imports -
import numpy as np
# system imports -
import time
from tqdm import tqdm
# my code -
from RL_anticipatory.utils import get_inner_model
from RL_anticipatory.problems.problem_anticipatory import AnticipatoryProblem
from RL_anticipatory.nets.RL_Model import AnticipatoryModel


def main():
    torch.manual_seed(1)
    # network parameters -
    batch_size = 10
    n_features = 5
    n_samples = 100
    # problem parameters -
    graph_size = 10
    n_graphs = 500
    end_time = 5
    events_time_window = 5
    n_cars = 2
    # anticipatory parameters -
    stochastic_mat = np.zeros([10, 10, 10, 10])  # should be probability mat [x, y, t, p(n events)]
    n_stochastic_runs = 1
    n_prediction_steps = 5
    # distribution parameters -
    dist_lambda = 2/3
    cancel_cost = 10  # should be positive since all costs are added to total cost
    close_reward = 5  # should be positive since all rewards are subtracted from total cost
    movement_cost = 1  # should be positive since all costs are added to total cost
    open_cost = 1  # should be positive since all costs are added to total cost
    opts = {
        'n_cars': n_cars,
        'events_time_window': events_time_window,
        'end_time': end_time,
        'graph_size': graph_size,
        'cancel_cost': cancel_cost,
        'close_reward': close_reward,
        'movement_cost': movement_cost,
        'open_cost': open_cost,
        'lam': dist_lambda}
    stochastic_input_dict = {'future_mat': stochastic_mat,
                             'n_stochastic_runs': n_stochastic_runs}
    sim_input_dict = {'graph_dim': graph_size,
                      'sim_length': end_time,
                      'events_open_time': events_time_window,
                      'n_prediction_steps': n_prediction_steps,
                      'dist_lambda': dist_lambda,
                      'n_cars': n_cars}
    # Set the device
    opts['device'] = torch.device("cuda:0" if opts['use_cuda'] else "cpu")

    problem = AnticipatoryProblem(opts)
    model = AnticipatoryModel(n_features, graph_size, 128, 0, stochastic_input_dict, sim_input_dict)
    dataset = problem.make_dataset(n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for data in dataloader:
        s_time = time.time()
        all_actions, ll, cost, state = model(data)
        e_time = time.time()
        print("batch run time is:"+str(e_time-s_time))
        print("total cost of batchs is:")
        print(cost)
    return






if __name__ == '__main__':
    main()
    print("done!")

