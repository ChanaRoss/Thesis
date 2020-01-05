# for technical file handling
import copy
import pickle
import time
import json
import networkx as nx
# for mathematical calculations and statistical distributions
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
import numpy as np
from gurobipy import *
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
import imageio

sns.set()


def create_gif(file_loc, filenames, duration, out_name):
    """
    this function creates a gif from given location and saves the gif and another location
    :param file_loc: location of files to create gif from (image files)
    :param filenames: list of names for all files in gif, list order will be the order of images in gif
    :param duration: location where to save the gif
    :param out_name: gif name (without .gif)
    :return: 
    """
    images = []
    for filename in filenames:
        images.append(imageio.imread(file_loc + filename))
    imageio.mimsave(file_loc + out_name + '.gif', images, duration=duration)


def move_car(car_pos, target_pos):
    """
    this function  moves car one step towards the target on the grid , assuming cars can only move in euclidean space
    :param car_pos: current position of car [1, 2]
    :param target_pos: target position for the car [1, 2]
    :return: 
    """
    updated_car_pos = copy.copy(car_pos)
    delta = target_pos - car_pos
    if delta[0] != 0:
        updated_car_pos[0] += np.sign(delta[0])
    else:
        updated_car_pos[1] += np.sign(delta[1])
    return updated_car_pos


def poisson_random_events(start_time, end_time, lam):
    """
    creates time line assuming poisson distribution
    :param start_time: start time wanted for timeline
    :param end_time: end time wanted for timeline
    :param lam: lambda of poisson distribution (how often will an event appear?)
    :return: array of times for events
    """
    length_time_line = int(end_time - start_time)
    num_events_per_time = np.random.poisson(lam=lam, size=length_time_line)
    event_time = []
    for i in range(length_time_line):
        n_time = num_events_per_time[i]
        if n_time > 0:
            for num in range(n_time):
                event_time.append(i + start_time)
    return np.array(event_time)


def create_events_distribution_poisson(grid_size, start_time, end_time, lam, events_time_window):
    """
    this function creates events time and location distribution for simulation based on input params.
    :param grid_size: size of grid (assuming grid has same size on both axis) [int]
    :param start_time: time to start events from [int]
    :param end_time: time of final event [int]
    :param lam: param. for poisson distribution [float/double]
    :param events_time_window: time each event is opened for [int]
    :return: matrix of events location [n_events, 2] and matrix of events start and finish time [n_events, 2]
    """
    loc_x        = grid_size / 3
    scale_x      = grid_size / 3
    loc_y        = grid_size / 3
    scale_y      = grid_size / 3
    # create random times - poisson distribution
    event_times  = poisson_random_events(start_time, end_time, lam)
    # create random locations - truncated normal distribution
    event_pos_x   = truncnorm.rvs((0 - loc_x) / scale_x, (grid_size - loc_x) / scale_x, loc=loc_x, scale=scale_x,
                                size=len(event_times)).astype(np.int64)
    event_pos_y   = truncnorm.rvs((0 - loc_y) / scale_y, (grid_size - loc_y) / scale_y, loc=loc_y, scale=scale_y,
                                size=len(event_times)).astype(np.int64)

    events_pos           = np.column_stack([event_pos_x, event_pos_y])
    events_time   = np.column_stack([event_times, event_times + events_time_window])
    return events_pos, events_time


def create_events_distribution_uniform(grid_size, n_events):
    events_pos = np.reshape(np.random.rand(n_events*2)*grid_size[0], (n_events, 2))
    return events_pos


def run_mtsp_opt(car_pos, event_pos, output_flag=1):
    """
    this function runs the optimization for determinist problem using gurobi as the optimizer
    :param car_pos: matrix of car positions [n_cars, 2]
    :param event_pos: matrix of event positions [n_events, 2]
    :param output_flag: should output gurobi log
    :return:
    """
    n_events = event_pos.shape[0]
    n_cars = car_pos.shape[0]

    distance_matrix = cdist(event_pos, event_pos, metric='euclidean')

    # Create optimization model
    m = Model('OfflineOpt')

    # Create variables
    x = m.addVars(n_events, n_events, name='c_is_picked_up', vtype=GRB.BINARY)
    u = m.addVars(n_events - 1, name='u_latent_variables', vtype=GRB.INTEGER)
    p = n_events - n_cars
    # all cars leave the depot node
    m.addConstr(sum(x[0, j+1] for j in range(n_events - 1)) == n_cars, "all_cars_depart_from_depot", None, "")
    # all cars return to depot node
    m.addConstr(sum(x[i+1, 0] for i in range(n_events - 1)) == n_cars, "all_cars_return_to_depot", None, "")
    # nodes can't enter themselves
    m.addConstr(sum(x[i, i] for i in range(n_events)) == 0, "all_cars_return_to_depot", None, "")
    # only one tour enters each event
    for j in range(n_events-1):
        m.addConstr(sum(x[i, j+1] for i in range(n_events)) == 1, "one_tour_enters", None, "")
    # only one tour exits each event
    for i in range(n_events-1):
        m.addConstr(sum(x[i+1, j] for j in range(n_events)) == 1, "one_tour_exits", None, "")

    # no sub-tour is included
    for i in range(n_events-1):
        for j in range(n_events - 1):
            if i != j:
                m.addConstr(u[i] - u[j] + p*x[i+1, j+1] <= p-1, "no_subtours", None, "")

    total_cost = 0  # reward for events that are closed after an event

    for i in range(n_events):
        for j in range(n_events):
            total_cost += distance_matrix[i, j] * x[i, j]

    # find the final objective of optimization problem (maximum since we are looking at the rewards)
    obj = total_cost

    # adding constraints and objective to gurobi model
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', output_flag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj

def run_mtsp_opt_three(car_pos, event_pos, output_flag=1):
    """
    this function runs the optimization for determinist problem using gurobi as the optimizer 
    :param car_pos: matrix of car positions [n_cars, 2]
    :param event_pos: matrix of event positions [n_events, 2]
    :param output_flag: should output gurobi log
    :return: 
    """
    n_events = event_pos.shape[0]
    n_cars   = car_pos.shape[0]
    
    distance_matrix = cdist(event_pos, event_pos, metric='euclidean')
    
    # Create optimization model
    m = Model('OfflineOpt')
    
    # Create variables
    x = m.addVars(n_events, n_events, n_cars, name='c_is_picked_up', vtype=GRB.BINARY)
    u = m.addVars(n_events-1, name='u_latent_variables', vtype=GRB.INTEGER)

    # add constraint - for each car
    for k in range(n_cars):
        # leave depot exactly once
        m.addConstr(sum(x[i+1, 0, k] for i in range(n_events-1)) == 1,
                    "leave_depot_once", None, "")
        # return to depot exaclty once
        m.addConstr(sum(x[0, i+1, k] for i in range(n_events-1)) == 1,
                    "return_depot_once", None, "")
        for r in range(n_events-1):
            # for each non depot event, number of times it is visited equals the number of times its left
            m.addConstr(sum(x[i, r+1, k] for i in range(n_events)) == sum(x[r+1, j, k] for j in range(n_events)),
                        "visit_event_once", None, "")

    for i in range(n_events-1):
        # number of times a non depot event is left is exactly one
        m.addConstr(sum(sum(x[i+1, j, k] for k in range(n_cars)) for j in range(n_events)) == 1,
                    "leave_event_once", None, "")
        m.addConstr(sum(x[i+1, i+1, k] for k in range(n_cars)) == 0, "cant_pickup_same_event", None, "")

        for j in range(n_events-1):
            if i != j:
                m.addConstr(u[i]-u[j] + (n_events - n_cars)*sum(x[i+1, j+1, k] for k in range(n_cars)) <=
                            (n_events - n_cars -1), "no_subrouts", None, "")

    for j in range(n_events-1):
        # number of times a non depot event is picked up is exactly one
        m.addConstr(sum(sum(x[i, j+1, k] for k in range(n_cars)) for i in range(n_events)) == 1,
                    "return_event_once", None, "")


    total_cost       = 0   # reward for events that are closed after an event

    for k in range(n_cars):
        for i in range(n_events):
            for j in range(n_events):
                total_cost += distance_matrix[i, j]*x[i, j, k]


    # find the final objective of optimization problem (maximum since we are looking at the rewards)
    obj = total_cost

    # adding constraints and objective to gurobi model 
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', output_flag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def analysis_and_plot_results(m, cars_pos, events_pos, plot_figures, file_loc, file_name, gs, is_three):
    """
    this function creates car paths throughout the simulation and post processes the output of optimization.
    in addition the function plots the results and creats gif if wanted
    :param m: model output
    :param cars_pos: matrix of car positions [n_cars, 2]
    :param events_pos: matrix of event positions [n_events, 2]
    :param plot_figures: flag, should  plot figures then True
    :param file_loc: location to save figures
    :param file_name: figures name
    :param gs: grid size [int]
    :return: 
    """
    plot_gif = False
    if plot_gif:
        if not os.path.isdir(file_loc + file_name):
            os.mkdir(file_loc + file_name)
    n_cars    = cars_pos.shape[0]
    n_events  = events_pos.shape[0]
    
    # get parameters from gurobi output
    param_key = [v.varName.split('[')[0] for v in m.getVars()]
    v_out = m.getVars()
    mat_out = np.zeros([n_events, n_events, n_cars])
    t = 0
    # for i in range(n_events):
    #     for j in range(n_events):
    #         for k in range(n_cars):
    #             mat_out[i, j, k] = v_out[t].x
    #             t += 1

    param = {k: [] for k in param_key}
    for v in m.getVars():
        param[v.varName.split('[')[0]].append(v.x)
    if is_three:
        param['c_is_picked_up'] = np.array(param['c_is_picked_up']).reshape([n_events, n_events, n_cars])
    else:
        param['c_is_picked_up'] = np.array(param['c_is_picked_up']).reshape([n_events, n_events])
    events_loc_dict = {}
    for i_e in range(events_pos.shape[0]):
        events_loc_dict[i_e] = tuple(events_pos[i_e, :])
    if is_three:
        for i in range(n_cars):
            G = nx.from_numpy_matrix(param['c_is_picked_up'][:, :, i])
            plt.figure()
            nx.draw(G, with_labels=True, pos=events_loc_dict)
    else:
        G = nx.from_numpy_matrix(param['c_is_picked_up'])
        plt.figure()
        nx.draw(G, with_labels=True, pos=events_loc_dict)
    plt.show()
    return param['c_is_picked_up']


def main():
    with open('config_unlimited_time.json') as f:
        param_dict = json.load(f)

    # load parameters form json file
    sim_seed       = param_dict['sim_seed']  # seed for event and car data
    is_three       = param_dict['is_three']  # should use three code or two
    grid_size            = [param_dict['grid_size'], param_dict['grid_size']]  # size of environment
    n_cars               = param_dict['n_cars']  # number of cars in simulation
    n_events             = param_dict['n_events']  # number of events in simulation
    length_sim           = param_dict['length_sim']  # full length of simulation
    # set random seed for problem
    np.random.seed(sim_seed)
    plot_figures = param_dict['plot_figures']  # flag if should plot figures [0 - don't plot, 1 - plot]
    print_logs   = param_dict['print_logs']  # flag if should print simulation log [0 - should not print, 1 - should print]
    # name and location for saving output
    file_loc  = param_dict['file_loc']  # location of output file and figures
    file_name = param_dict['file_name']  # name of files


    # create car initial positions
    car_pos = np.reshape(np.random.rand(2 * n_cars)*grid_size[0], (n_cars, 2))

    # create event positions and time
    event_pos = create_events_distribution_uniform(grid_size, n_events)

    # run optimization using gurobi
    s_time = time.time()
    if is_three:
        m, obj = run_mtsp_opt_three(car_pos, event_pos)
    else:
        m, obj = run_mtsp_opt(car_pos, event_pos)
    e_time = time.time()
    print("simulation run time is: " + str(e_time - s_time))
    if print_logs:
        # print results of optimization
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))

        print('Obj: %g' % obj.getValue())


    # # run analysis and plot results
    data_out = analysis_and_plot_results(m, car_pos, event_pos,
                                                            plot_figures, file_loc, file_name, grid_size, is_three)


if __name__ == '__main__':
    main()
    print('Done.')
