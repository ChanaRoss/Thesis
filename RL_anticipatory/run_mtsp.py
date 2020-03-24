# torch imports -
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# math imports -
import numpy as np
# system imports -
import json
import os
import time
import pickle
# my code -
from RL_anticipatory.problems.problem_anticipatory import AnticipatoryProblem
from RL_anticipatory.nets.RL_Model_MTSP import MTSPModel
from RL_anticipatory.utils import torch_load_cpu
from RL_anticipatory.train import train_epoch, validate, get_inner_model
from RL_anticipatory.reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline


def run():
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
    events_time_window = 999
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
        'lr_patience': 5,   # number of epochs before lr is updated
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
        'no_cuda': True,
        'eval_only': False,
        'with_baseline': True,
        'bl_warmup_epochs': None,
        'checkpoint_epochs': 1,
        'log_step': 4,
        'eval_batch_size': eval_batch_size,
        'run_name': 'mtsp_7',
        'problem': 'anticipatory_rl',
        'output_dir': 'outputs',
        'log_dir': 'logs',
        'load_path': None,
        'resume': None}
    # '/Users/chanaross/dev/Thesis/RL_anticipatory/outputs/anticipatory_rl_7/mtsp_10_20200310T200705/epoch-2715.pt'}
    stochastic_input_dict = {'future_mat': stochastic_mat,
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
    if opts['bl_warmup_epochs'] is None:
        opts['bl_warmup_epochs'] = 1 if opts['baseline'] == 'rollout' else 0
    assert (opts['bl_warmup_epochs'] == 0) or (opts['baseline'] == 'rollout')
    assert opts['epoch_size'] % opts['batch_size'] == 0, "Epoch size must be integer multiple of batch size!"

    # Optionally configure tensorboard
    tb_logger = None
    if not opts['no_tensorboard']:
        tb_logger = SummaryWriter(os.path.join(opts['log_dir'], "{}".format(opts['problem']), opts['run_name']))
        # log the configuration for this run
        tb_logger.add_text("config/" + os.path.join(opts['log_dir'], "{}".format(opts['problem']), opts['run_name']),
                           json.dumps(opts, indent=True), 0)

    # create save location -
    os.makedirs(opts['save_dir'])

    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts['save_dir'], "args.json"), 'w') as f:
        json.dump(opts, f, indent=True)

    # save stochastic and sim dict so that i can reproduce the runs
    with open(os.path.join(opts['save_dir'], "stochastic_input.pkl"), 'wb') as f:
        pickle.dump(stochastic_input_dict, f)

    with open(os.path.join(opts['save_dir'], "sim_input.json"), 'w') as f:
        json.dump(sim_input_dict, f, indent=True)

    sim_input_dict['possible_actions'] = possible_actions

    # Set the device
    opts['use_cuda'] = torch.cuda.is_available() and not opts['no_cuda']
    opts['device'] = torch.device("cuda:0" if opts['use_cuda'] else "cpu")
    # create problem
    problem = AnticipatoryProblem(opts)

    # Load data from load_path
    load_data = {}
    assert opts['load_path'] is None or opts['resume'] is None, "Only one of load path and resume can be given"
    load_path = opts['load_path'] if opts['load_path'] is not None else opts['resume']
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    torch.autograd.set_detect_anomaly(True)
    model = MTSPModel(n_features, graph_size, opts['embedding_dim'], opts['encoder_dim'], opts['dp'],
                      stochastic_input_dict, sim_input_dict)
    model = model.to(opts['device'])
    if opts['use_cuda'] and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts['baseline'] == 'exponential':
        baseline = ExponentialBaseline(opts['exp_beta'])
    elif opts['baseline'] == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts['baseline'] is None, "Unknown baseline: {}".format(opts['baseline'])
        baseline = NoBaseline()
    if opts['bl_warmup_epochs'] > 0:
        baseline = WarmupBaseline(baseline, opts['bl_warmup_epochs'], warmup_exp_beta=opts['exp_beta'])

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts['lr_model']}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts['lr_critic']}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts['device'])

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    if opts['lr_scheduler'] == 'Reduce':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opts['lr_decay'],
                                                            patience=opts['lr_patience'], verbose=True, min_lr=5e-7)
    else:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts['lr_decay'] ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(num_samples=opts['val_size'])

    if opts['resume']:
        optimizer.param_groups[0]['lr'] = opts['lr_model']
        epoch_resume = int(os.path.splitext(os.path.split(opts['resume'])[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts['use_cuda']:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        _, _ = baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts['epoch_start'] = epoch_resume + 1
    else:
        opts['epoch_start'] = 0

    if opts['eval_only']:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts['epoch_start'], opts['epoch_start'] + opts['n_epochs']):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

    tb_logger.close()
    return


if __name__ == '__main__':
    run()
    print("done!")

