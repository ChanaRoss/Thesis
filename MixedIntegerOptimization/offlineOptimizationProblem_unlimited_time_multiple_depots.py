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


def run_mtsp_online_def(car_pos, events_loc, fake_depot, output_flag = 0):
## doesnt' work yet, need to add a constraint to not allow sub routes (or circular routes)
    event_pos = np.row_stack([events_loc, fake_depot])
    n_events = event_pos.shape[0]
    n_cars = car_pos.shape[0]

    # Create optimization model
    m = Model('OfflineOpt')

    # Create variables
    x = m.addVars(n_events, n_events, name='c_events_to_events', vtype=GRB.BINARY)
    y = m.addVars(n_cars, n_events, name='c_cars_to_events', vtype=GRB.BINARY)
    # u = m.addVars(n_events - n_cars, name='u_latent_variables', vtype=GRB.INTEGER)
    # p = np.floor((n_events - n_cars) / n_cars)

    # add constraint for events - maximum one event is picked up after each event
    for i in range(n_events-1):
        m.addConstr(sum(x[i, j] for j in range(n_events)) <= 1, "limited_num_picked_up_after_event", None, "")

    # add constraint for cars - each car picks up maximum one event when starting
    # y_{k,c} is either 0 or 1 for all events
    for i in range(n_cars):
        m.addConstr(sum(y[i, j] for j in range(n_events)) == 1, "limited_num_picked_up_after_first", None, "")


    # add constraint for pickup - if event is picked up then it was either picked up first or after other event
    # event is picked up first by car or after one other event
    for i in range(n_events-1):
        sum1 = sum([y[c, i] for c in range(n_cars)])
        sum2 = sum([x[e, i] for e in range(n_events)])
        m.addConstr(1 == sum1 + sum2, "max_one_pickup_per_event", None, "")

    # event can not be picked up after it is picked up (X_{c,c} = 0 by definition)
    for i in range(n_events):
        m.addConstr(x[i, i] == 0, "no_loops", None, "")  # an event cant pick itself up

    # for h in range(n_events):
    #     sum1_events = sum([x[i, h] for i in range(n_events)])
    #     sum1_cars   = sum([y[k, h] for k in range(n_cars)])
    #     sum2 = sum([x[h, j] for j in range(n_events)])
    #     m.addConstr(sum1_events + sum1_cars - sum2 == 0, "continuation_of_solution", None, "")

    c_events = 0  # reward for events that are closed after an event
    c_cars = 0  # reward for events that are closed after car


    event_distance_matrix = cdist(event_pos, event_pos, metric='cityblock')
    car_distance_matrix = cdist(car_pos, event_pos, metric='cityblock')

    for i in range(n_events):
        for j in range(n_events):
            c_events += event_distance_matrix[i, j] * x[i, j]

        for j in range(n_cars):
            c_cars += car_distance_matrix[j, i] * y[j, i]

    # find the final objective of optimization problem (maximum since we are looking at the rewards)
    obj = c_events + c_cars

    # adding constraints and objective to gurobi model
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', output_flag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def run_mtsp_opt(car_pos, event_loc, fake_depot, same_length_tours = False, output_flag=0):
    """
    this function runs the optimization for determinist problem using gurobi as the optimizer
    :param car_pos: matrix of car positions [n_cars, 2]
    :param event_pos: matrix of event positions [n_events, 2]
    :param output_flag: should output gurobi log
    :return:
    """
    event_pos = np.row_stack([fake_depot, car_pos, event_loc])
    n_events = event_pos.shape[0]
    n_cars = car_pos.shape[0]

    distance_matrix = cdist(event_pos, event_pos, metric='cityblock')

    # Create optimization model
    m = Model('OfflineOpt')

    # Create variables
    x = m.addVars(n_events, n_events, name='c_is_picked_up', vtype=GRB.BINARY)
    u = m.addVars(n_events, name='u_latent_variables', vtype=GRB.INTEGER)
    if same_length_tours:
        p = np.floor((n_events - 1)/n_cars)
    else:
        p = n_events - n_cars
    # all cars leave the depot node
    for i_c in range(n_cars):
        m.addConstr(x[0, i_c+1]  == 1, "all_cars_leave_depot", None, "")
    m.addConstr(sum(x[0, j + 1] for j in range(n_events - 1)) == n_cars, "num_leaving_depot_is_n_cars", None, "")
    # all cars return to depot node
    m.addConstr(sum(x[i + 1, 0] for i in range(n_events - 1)) == n_cars, "num_returning_depot_is_n_cars", None, "")
    # nodes can't enter themselves
    m.addConstr(sum(x[i, i] for i in range(n_events)) == 0, "all_cars_return_to_depot", None, "")
    # only one tour enters each event
    for j in range(n_events - 1):
        m.addConstr(sum(x[i, j + 1] for i in range(n_events)) == 1, "one_tour_enters", None, "")
    # only one tour exits each event
    for i in range(n_events - 1):
        m.addConstr(sum(x[i + 1, j] for j in range(n_events)) == 1, "one_tour_exits", None, "")

    # no sub-tour is included
    for i in range(n_events - 1):
        for j in range(n_events - 1):
            if i != j:
                m.addConstr(u[i] - u[j] + p * x[i + 1, j + 1] <= p - 1, "no_subtours", None, "")

    total_cost = 0  # reward for events that are closed after an event

    for i in range(n_events):
        for j in range(n_events):
            if (i != 0) and (j != 0):
                total_cost += distance_matrix[i, j] * x[i, j]

    # find the final objective of optimization problem (maximum since we are looking at the rewards)
    obj = total_cost

    # adding constraints and objective to gurobi model
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', output_flag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def run_mtsp_vip_opt(car_pos, event_loc, event_cost, fake_depot, output_flag=0):
    """
    this function runs the optimization for determinist problem using gurobi as the optimizer
    :param car_pos: matrix of car positions [n_cars, 2]
    :param event_pos: matrix of event positions [n_events, 2]
    :param output_flag: should output gurobi log
    :return:
    """
    event_pos = np.row_stack([fake_depot, car_pos, event_loc])
    n_events = event_pos.shape[0]
    n_cars = car_pos.shape[0]

    distance_matrix = cdist(event_pos, event_pos, metric='cityblock')

    # Create optimization model
    m = Model('OfflineOpt')

    # Create variables
    x = m.addVars(n_events, n_events, name='c_is_picked_up', vtype=GRB.BINARY)
    u = m.addVars(n_events, name='u_latent_variables', vtype=GRB.INTEGER)
    p = np.floor((n_events - 1)/n_cars)   # n_events - n_cars  #
    # all cars leave the depot node
    for i_c in range(n_cars):
        m.addConstr(x[0, i_c+1]  == 1, "all_cars_leave_depot", None, "")
    m.addConstr(sum(x[0, j + 1] for j in range(n_events - 1)) == n_cars, "num_leaving_depot_is_n_cars", None, "")
    # all cars return to depot node
    m.addConstr(sum(x[i + 1, 0] for i in range(n_events - 1)) == n_cars, "num_returning_depot_is_n_cars", None, "")
    # nodes can't enter themselves
    m.addConstr(sum(x[i, i] for i in range(n_events)) == 0, "all_cars_return_to_depot", None, "")
    # only one tour enters each event
    for j in range(n_events - 1):
        m.addConstr(sum(x[i, j + 1] for i in range(n_events)) == 1, "one_tour_enters", None, "")
    # only one tour exits each event
    for i in range(n_events - 1):
        m.addConstr(sum(x[i + 1, j] for j in range(n_events)) == 1, "one_tour_exits", None, "")

    # no sub-tour is included
    for i in range(n_events - 1):
        for j in range(n_events - 1):
            if i != j:
                m.addConstr(u[i] - u[j] + p * x[i + 1, j + 1] <= p - 1, "no_subtours", None, "")

    total_cost = 0  # reward for events that are closed after an event

    for i in range(n_events):
        for j in range(n_events):
            if (i != 0) and (j != 0):
                # j is an event and need to take into consideration if its a real or fake event
                order_cost = (u[j-1]+1)*event_cost[j-1]
                total_cost += distance_matrix[i, j] * x[i, j] + order_cost

    # find the final objective of optimization problem (maximum since we are looking at the rewards)
    obj = total_cost

    # adding constraints and objective to gurobi model
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', output_flag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def analysis_and_plot_results(m, cars_pos, events_pos, fake_depot, plot_figures, file_loc, file_name, gs):
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
    param = {k: [] for k in param_key}
    for v in m.getVars():
        param[v.varName.split('[')[0]].append(v.x)
    param['c_is_picked_up'] = np.array(param['c_is_picked_up']).reshape(n_events+n_cars+1, n_events+n_cars+1)

    events_loc_dict = {}
    for i_c in range(cars_pos.shape[0]):
        events_loc_dict[i_c] = tuple(cars_pos[i_c, :])
    for i_e in range(events_pos.shape[0]):
        events_loc_dict[i_e+n_cars] = tuple(events_pos[i_e, :])
    G = nx.from_numpy_matrix(param['c_is_picked_up'][1:, 1:])
    if plot_figures:
        plt.figure()
        nx.draw(G, with_labels=True, pos=events_loc_dict)
    if plot_figures:
        plt.show()
    return param['c_is_picked_up']


def main():
    sns.set()
    with open('config_unlimited_time.json') as f:
        param_dict = json.load(f)

    # load parameters form json file
    sim_seed       = param_dict['sim_seed']  # seed for event and car data
    grid_size            = [param_dict['grid_size'], param_dict['grid_size']]  # size of environment
    n_cars               = param_dict['n_cars']  # number of cars in simulation
    n_events             = param_dict['n_events']  # number of events in simulation
    same_length_routes   = param_dict['same_length_routes']  # if true all cars have routes of the same length
    # set random seed for problem
    np.random.seed(sim_seed)
    plot_figures = param_dict['plot_figures']  # flag if should plot figures [0 - don't plot, 1 - plot]
    print_logs   = param_dict['print_logs']  # flag if should print simulation log [0 - should not print, 1 - should print]
    # name and location for saving output
    file_loc  = param_dict['file_loc']  # location of output file and figures
    file_name = param_dict['file_name']  # name of files

    n_real_events = 2
    cost = np.zeros(param_dict['n_runs'])
    for i in range(param_dict['n_runs']):
        # create car initial positions
        car_pos = np.reshape(np.random.rand(2 * n_cars)*grid_size[0], (n_cars, 2))

        # create event positions and time
        event_pos = create_events_distribution_uniform(grid_size, n_events)
        event_cost = np.zeros(n_events+n_cars)
        for i_n in range(n_cars + n_events):
            if i_n < n_cars:
                event_cost[i_n] = 0
            elif i_n < n_cars+n_real_events:
                event_cost[i_n] = 1
            else:
                event_cost[i_n] = 10
        fake_depot = np.zeros((1, 2))
        # run optimization using gurobi
        s_time = time.time()
        m, obj = run_mtsp_opt(car_pos, event_pos, fake_depot, same_length_routes, False)
        e_time = time.time()
        print("simulation run time is: " + str(e_time - s_time))
        if print_logs:
            # print results of optimization
            for v in m.getVars():
                print('%s %g' % (v.varName, v.x))

            print('Obj: %g' % obj.getValue())


        # # run analysis and plot results
        data_out = analysis_and_plot_results(m, car_pos, event_pos, fake_depot,
                                             plot_figures, file_loc, file_name, grid_size)

        cost[i] = obj.getValue()
    print("mean cost is:"+str(np.mean(cost)))
    print("std cost : "+str(np.std(cost)))
    print("max cost is:"+str(np.max(cost)))


if __name__ == '__main__':
    main()
    print('Done.')
