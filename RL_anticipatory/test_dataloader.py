import numpy as np
import torch
from torch_geometric.data import DataLoader
from RL_anticipatory.problems.problem_anticipatory import AnticipatoryProblem
import os
import time


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # network parameters -
    n_features = 5
    eval_batch_size = 20
    n_epochs = 4000
    # problem parameters -
    graph_size = 7
    epoch_size = 300
    batch_size = 30
    val_size = 20
    end_time = 15
    events_time_window = 4
    n_cars = 2
    n_events = 4
    # anticipatory parameters -
    stochastic_mat = np.zeros([10, 10, 10, 10])  # should be probability mat [x, y, t, p(n events)]
    possible_actions = torch.tensor([  # [0, 0],
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0]])
    n_stochastic_runs = 1
    n_prediction_steps = 5
    # distribution parameters -
    dist_lambda = 0.3  # 2/3
    cancel_cost = 100  # should be positive since all costs are added to total cost
    close_reward = 100  # should be positive since all rewards are subtracted from total cost
    movement_cost = 0  # should be positive since all costs are added to total cost
    open_cost = 50  # should be positive since all costs are added to total cost
    opts = {
        'n_cars': n_cars,
        'n_events': n_events,
        'n_features': n_features,
        'events_time_window': events_time_window,
        'end_time': end_time,
        'graph_size': graph_size,
        'cancel_cost': cancel_cost,
        'close_reward': close_reward,
        'movement_cost': movement_cost,
        'open_cost': open_cost,
        'lam': dist_lambda,
        'batch_size': batch_size,
        'epoch_size': epoch_size,
        'val_size': val_size,
        'n_epochs': n_epochs,
        'lr_model': 1e-4,
        'lr_critic': 1e-4,
        'lr_patience': 5,  # number of epochs before lr is updated
        'lr_decay': 0.95,
        'exp_beta': 0.9,
        'dp': 0,
        'max_grad_norm': 1,
        'encoder_dim': 128,
        'embedding_dim': 128,
        'decode_type': 'sampling',
        'lr_scheduler': 'Reduce',
        'baseline': 'rollout',
        'bl_alpha': 0.05,
        'no_progress_bar': False,
        'no_tensorboard': False,
        'no_cuda': False,
        'eval_only': False,
        'with_baseline': True,
        'bl_warmup_epochs': None,
        'checkpoint_epochs': 1,
        'log_step': 4,
        'eval_batch_size': eval_batch_size,
        'run_name': 'anti_with_time_window',
        'problem': 'anticipatory_rl',
        'output_dir': 'outputs',
        'log_dir': 'logs',
        'load_path': None,
        'resume': None}
    stochastic_input_dict = {'future_mat': stochastic_mat,
                             'should_calc_anticipatory': False,
                             'n_stochastic_runs': n_stochastic_runs}
    sim_input_dict = {'graph_dim': graph_size,
                      'sim_length': end_time,
                      'events_open_time': events_time_window,
                      'n_prediction_steps': n_prediction_steps,
                      'dist_lambda': dist_lambda,
                      'n_cars': n_cars,
                      'should_calc_all_options': False,
                      'print_debug': False}

    opts['run_name'] = "{}_{}".format(opts['run_name'], time.strftime("%Y%m%dT%H%M%S"))
    opts['should_calc_all_options'] = sim_input_dict['should_calc_all_options']
    opts['save_dir'] = os.path.join(
        opts['output_dir'],
        "{}_{}".format(opts['problem'], opts['graph_size']),
        opts['run_name'])

    sim_input_dict['possible_actions'] = possible_actions

    # Set the device
    opts['use_cuda'] = torch.cuda.is_available() and not opts['no_cuda']
    opts['device'] = torch.device("cuda:0" if opts['use_cuda'] else "cpu")
    # create problem
    problem = AnticipatoryProblem(opts)
    # Generate new training data for each epoch
    training_dataset = problem.make_dataset(num_samples=opts['epoch_size'])
    training_dataloader = DataLoader(training_dataset, batch_size=opts['batch_size'], num_workers=0)
    for data_id, data in enumerate(training_dataloader):

        print(data_id)
        print(data)


if __name__ == '__main__':
    main()
    print("done!")

