from torch.nn import DataParallel
import torch
import os
import json
import pickle


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def load_problem():
    from RL_anticipatory.problems import problem_anticipatory
    problem = problem_anticipatory
    return problem


def load_stochastic(path):
    data = pickle.load(open(path, 'rb'))
    return data


def load_model(path, epoch=None):
    from RL_anticipatory.nets.RL_Model import AnticipatoryModel

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    stochastic_input_dict = load_stochastic(os.path.join(path, 'stochastic_input.pkl'))
    sim_input_dict = load_args(os.path.join(path, 'sim_input.json'))
    sim_input_dict['possible_actions'] = torch.tensor([#[0, 0],
                                       [0, 1],
                                       [1, 0],
                                       [0, -1],
                                       [-1, 0]])
    if 'encoder_dim' not in args.keys():
        args['encoder_dim'] = 128
        args['embedding_dim'] = 128
    model = AnticipatoryModel(args['n_features'], args['graph_size'], args['embedding_dim'], args['encoder_dim'], 0,
                              stochastic_input_dict, sim_input_dict)
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args, sim_input_dict, stochastic_input_dict