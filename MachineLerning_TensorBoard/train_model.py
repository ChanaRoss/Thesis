import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model_registry import model_registry
from dataset_registry import dataset_registry
import pathlib
import time
import os
import argparse


def seconds_to_string(s):
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = (s % 3600) % 60
    return "{}:{}:{}".format(hours, minutes, secs)


def get_model(config):
    """
    loads a model using the model registry and the information
    given in the config json. config must include the key "model",
    and the corrisponding value must be found in the model registry.
    :param config: json with parameters and name of model to be retrieved
    :return: torch.nn.Module
    """
    if "model" not in config:
        raise Exception("model name wasn't given in config file")
    elif config['model'] not in model_registry:
        raise Exception("given model name is not registered.")
    else:
        model_name = config['model']
        model = model_registry[model_name](**config['model_params'])
        return model


def get_optimizer(config, parameters):
    """
    returns a torch.optim.Optimizer with the given
    parameters and the details provided in the config
    json. json shoud include a key "optimizer" which is mapped
    in the opt_dict.
    :param config: dictionary of config json.
    :param parameters: model parameters to optimize.
    :return: torch.optim.Optimizer
    """
    opt_dict = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "Adagrad": optim.Adagrad
    }
    if ("optimizer" not in config) or ("opt_params" not in config):
        raise Exception("algorithm or parameters not specified in config file.")
    elif config['optimizer'] not in opt_dict:
        raise Exception("{} is an unsupported optimizer. choose from: {}".format(config['optimizer'], list(opt_dict.keys())))
    else:
        algo = config['optimizer']
        algo_params = config['opt_params']
        return opt_dict[algo](parameters, **algo_params)


def get_scheduler(optimizer, config):
    """
    create a learning rate schedular with the given
    optimizer and the parameters chosen in the config json.
    for now creates ReduceLROnPlateau object.
    :return: torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['sched_params'])


def get_loss(config):
    """
    returns loss criteria specified in the config json,
    provided it is mapped in the loss_dict.
    :param config: configuration json.
    :return: torch.nn.Loss
    """
    loss_dict = {
        "MSELoss": nn.MSELoss,
        "NLLLoss": nn.NLLLoss
    }
    if ("loss" not in config) or ("loss_params" not in config):
        raise Exception("loss or loss params not specified in config file.")
    elif config['loss'] not in loss_dict:
        raise Exception("{} not a valid loss function, choose from: {}".format(config['loss'], list(loss_dict.keys())))
    else:
        return loss_dict[config['loss']](**config['loss_params'])


def get_datasets(config):
    if ('dataset' not in config) or ('dataset_params' not in config):
        raise Exception("dataset name and params must be passed in config")
    elif config['dataset'] not in dataset_registry:
        raise Exception("{} is not a supported dataset.".format(config['dataset']))
    else:
        train_set, valid_set = dataset_registry[config['dataset']](**config['dataset_params'])
        return train_set, valid_set


def setup_run(config):
    """
    1. create a log directory if it doesn't exist yet
    2. create a run directory with a unique name.
    3. create summary logging sub directory in the run dir
    4. create checkpoint sub directory in the run dir
    """
    # check if the log directory exists
    log_path = pathlib.Path(config['train_params']['log_path'])
    if not log_path.is_dir():
        log_path.mkdir(parents=True)

    # create run name
    run_name = config['train_params']['run_name'] + time.strftime("%m_%d-%H_%M_%S", time.localtime())

    # create run path
    run_path_name = os.path.join(config['train_params']['log_path'], run_name)
    run_summary_path = os.path.join(run_path_name, "logs")
    run_checkpoint_path = os.path.join(run_path_name, "checkpoints")

    # create directories
    pathlib.Path(run_path_name).mkdir()
    pathlib.Path(run_checkpoint_path).mkdir()
    pathlib.Path(run_summary_path).mkdir()
    return run_path_name


def save_model(model, optimizer, loss, epoch, run_path):
    """
    saves the model and optimizer state dicts as well as epoch
    and loss to a dictionary.
    :param model: torch.nn.Module
    :param optimizer: torch.optim.Optimizer
    :param loss: torch.nn.Loss
    :param epoch: int
    :param run_path: string, directory of current run
    """
    checkpoint_name = time.strftime("%d%m_loss_{}_epoch_{}.p", time.localtime()).format(loss, epoch)
    save_path  = os.path.join(run_path, "checkpoints", checkpoint_name)
    with open(save_path, "wb") as out_checkpoint:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, out_checkpoint)
    return


def train_epoch(model, dataloader, optimizer, epoch, loss, logger):
    """
    carry out training for one epoch:
        - evaluate on the dataloader
        - compute loss
        - take optimizer steps
        - log training loss and layer parameters
    :param model: torch.nn.Module
    :param dataloader: torch.utils.data.DataLoader
    :param optimizer: torch.optim.Optimzer
    :param epoch: int, current epoch
    :param loss: torch.nn.Loss
    :param logger: torch.utils.tensorboard.SummaryWriter
    :return: mean training loss per batch
    """
    # print message
    msg = "training epoch: {}, batch: {}/{}, loss: {}"
    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # activate train mode and move to device
    model.train()
    model.to(device)
    # calculate number of batches in epoch
    epoch_batches = len(dataloader.dataset) // dataloader.batch_size
    total_loss = torch.zeros((1,))
    t0 = time.time()
    for i, (inputs, labels) in enumerate(dataloader):
        # move input tensors to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        else:
            inputs = [inp.to(device) for inp in inputs]

        # move label tensors to device
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        else:
            labels = [inp.to(device) for inp in labels]

        # zero gradients
        optimizer.zero_grad()

        # evaluate model on inputs and labels, should return network output
        output = model.forward(inputs, labels)

        # calculate loss
        current_loss = loss(output.view(-1), labels.view(-1))
        current_loss.backward()

        # sum over all loss
        total_loss += current_loss

        # take optimizer step
        optimizer.step()

        # log batch loss
        logger.add_scalar("loss/train_batch", current_loss, epoch * epoch_batches + i)

        # print message
        print(msg.format(epoch, i, epoch_batches, current_loss.item()))

    t1 = time.time()
    # epoch logging
    mean_epoch_loss = total_loss / epoch_batches
    logger.add_scalar("loss/train_epoch_mean", mean_epoch_loss, epoch)

    for name, param in model.named_parameters():
        logger.add_histogram("parameters/{}".format(name), param.data, epoch)

    print("training epoch: {}, time: {}, mean train loss: {}".format(epoch, seconds_to_string(t1-t0), mean_epoch_loss))
    return mean_epoch_loss


def validation_epoch(model, dataloader, epoch, loss, logger):
    # print message
    msg = "validation epoch: {}, batch: {}/{}, loss: {}"
    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # activate train mode and move to device
    model.eval()
    model.to(device)
    # calculate number of batches in epoch
    epoch_batches = len(dataloader.dataset) // dataloader.batch_size
    t0 = time.time()
    batch_loss = []
    for i, (inputs, labels) in enumerate(dataloader):
        # move input tensors to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        else:
            inputs = [inp.to(device) for inp in inputs]

        # move label tensors to device
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        else:
            labels = [inp.to(device) for inp in labels]

        # evaluate model
        outputs = model(inputs, labels)
        current_loss = loss(outputs.view(-1), labels.view(-1))
        batch_loss.append(current_loss)

        # console and tensorboard logging
        print(msg.format(epoch, i, epoch_batches, current_loss))
        logger.add_scalar("loss/validation_batch", current_loss, epoch*epoch_batches + i)

    t1 = time.time()
    mean_validation_epoch = torch.mean(torch.FloatTensor(batch_loss))
    print("training epoch: {}, time: {}, mean train loss: {}".format(epoch, seconds_to_string(t1 - t0), mean_validation_epoch))

    # write summaries to tensorboard
    logger.add_scalar("loss/validation_epoch_mean", mean_validation_epoch, epoch)
    logger.add_scalar("loss/validation_epoch_max", torch.max(torch.FloatTensor(batch_loss)), epoch)
    logger.add_scalar("loss/validation_epoch_min", torch.min(torch.FloatTensor(batch_loss)), epoch)
    return mean_validation_epoch


def train(config):
    """
    carry out entire training process:
     - load model
     - create optimizer and schedular
     - load train and validation datasets
     - create loss criteria
     - optimize model
     - log to tensorboard
     - save model checkpoints during training
    :param config:
    :return:
    """
    # setup run
    run_path = setup_run(config)

    # get datasets
    train_dataset, validation_dataset = get_datasets(config)

    # create dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_params']['batch_size'],
                              shuffle=config['train_params']['shuffle'])
    valid_loader = DataLoader(validation_dataset,
                              batch_size=config['train_params']['batch_size'],
                              shuffle=config['train_params']['shuffle'])

    # get model
    model = get_model(config)

    # create optimizer
    model_params = [p for p in model.parameters()]
    optimizer = get_optimizer(config, model_params)

    # create scheduler
    scheduler = get_scheduler(optimizer, config)

    # loss function
    loss = get_loss(config)

    # create logs writer
    logger = SummaryWriter(os.path.join(run_path, "logs"))

    # log the model
    #logger.add_graph(model, next(iter(train_loader)))

    # log the configuration for this run
    logger.add_text("config/{}".format(run_path.split("/")[-1]), json.dumps(config), 0)

    # train loop
    checkpoint_metric = torch.zeros((config['train_params']['epochs'],))
    for e in range(config['train_params']['epochs']):
        # train for an epoch
        train_mean_loss = train_epoch(model, train_loader, optimizer, e, loss, logger)

        # carry out validation for epoch
        valid_mean_loss = validation_epoch(model, valid_loader, e, loss, logger)

        # update learning rate based on loss
        scheduler.step(train_mean_loss, epoch=e)

        # epoch logging
        logger.add_scalar("optimizer/lr", optimizer.state_dict()['param_groups'][0]['lr'], e)

        # save checkpoint if necessary
        if train_mean_loss > torch.max(checkpoint_metric):
            save_model(model, optimizer, e, loss, run_path)

        # add current mean loss to checkpoint metric
        checkpoint_metric[e] = float(train_mean_loss)
    print("training complete.")
    logger.close()
    return


def main():
    parser = argparse.ArgumentParser(description="pass configuration to training script.")
    parser.add_argument("--config", type=str, help='json config file path')
    args = parser.parse_args()

    # validate config path
    if not os.path.isfile(args.config):
        raise Exception("pass a valid path to configuration json.")

    # load the configuration file
    config = json.loads(open(args.config, 'r').read())

    # carry out train
    train(config)
    return


if __name__ == '__main__':
    main()
    print('Done.')