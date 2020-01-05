# for technical file handling
import copy
import pickle
import time
import json
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


def create_events_distribution(grid_size, start_time, end_time, lam, events_time_window):
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
    events_time_window    = np.column_stack([event_times, event_times + events_time_window])
    return events_pos, events_time_window


def calc_reward_and_time(event_pos, car_pos, close_reward):
    """
    this function calculates the reward that will be given assuming event is picked up and
     finds the time at which the event will be picked up 
    :param event_pos: position of events
    :param car_pos: position of cars
    :param close_reward: reward if event is closed
    :return: reward_cars_to_events -   R_{cars,events},
             rewardEventsToEvents - R_{events,events}
    """
    n_cars              = car_pos.shape[0]
    n_events            = event_pos.shape[0]
    dist_events_to_events = cdist(event_pos, event_pos, metric='cityblock')
    dist_cars_to_events   = cdist(car_pos, event_pos, metric='cityblock')

    reward_cars_to_events   = -dist_cars_to_events + np.ones(shape=(n_cars, n_events)) * close_reward
    reward_events_to_events = -dist_events_to_events + np.ones(shape=(n_events, n_events)) * close_reward
    time_events_to_events   =  dist_events_to_events
    time_cars_to_events     =  dist_cars_to_events
    return reward_cars_to_events, reward_events_to_events, time_cars_to_events, time_events_to_events


def run_max_flow_opt(start_time, car_pos, event_pos, events_open_time, events_close_time,
                     close_reward, cancel_penalty, opened_penalty, output_flag=1):
    """
    this function runs the optimization for determinist problem using gurobi as the optimizer 
    :param start_time: time to start the optimization from (compared to full simulation)
    :param car_pos: matrix of car positions [n_cars, 2]
    :param event_pos: matrix of event positions [n_events, 2]
    :param events_open_time: time step at which events open [n_events, 1]
    :param events_close_time: time step at which events close [n_events, 1] 
    :param close_reward: the reward given if event is closed [scalar]
    :param cancel_penalty: penatly given if event is canceled (no one picked it up and the time is past its close time)
    :param opened_penalty: penatly for keeping events waiting. (delta_t open time)*penalty
    :param output_flag: should output gurobi log
    :return: 
    """
    n_events = event_pos.shape[0]
    n_cars   = car_pos.shape[0]
    
    # calculate the reward for all possible actions and the time at which each action will happen
    reward_cars_to_events, reward_events_to_events, time_cars_to_events, time_events_to_events = \
        calc_reward_and_time(event_pos, car_pos, close_reward)
    
    # Create optimization model
    m = Model('OfflineOpt')
    
    # Create variables
    x = m.addVars(n_events, n_events, name='c_events_to_events', vtype=GRB.BINARY)
    y = m.addVars(n_cars, n_events, name='c_cars_to_events', vtype=GRB.BINARY)
    p = m.addVars(n_events, name='is_picked_up', vtype=GRB.BINARY)
    t = m.addVars(n_events, name='pick_up_time')

    # add constraint for events - maximum one event is picked up after each event
    # if p=0 then no events are picked up after,
    # if p=1 then one event is picked up after
    for i in range(n_events):
        m.addConstr(sum(x[i, j] for j in range(n_events)) <= p[i])

    # add constraint for cars - each car picks up maximum one event when starting
    # y_{k,c} is either 0 or 1 for all events
    for i in range(n_cars):
        m.addConstr(sum(y[i, j] for j in range(n_events)) <= 1)

    # add constraint for pickup - if event is picked up then it was either picked up first or after other event
    # p=0 - both sums need to add up to 0, no one picked this event up
    # p=1 - event is picked up first by car or after one other event
    for i in range(n_events):
        sum1 = sum([y[c, i] for c in range(n_cars)])
        sum2 = sum([x[e, i] for e in range(n_events)])
        m.addConstr(p[i] == sum1+sum2)
        
    # add time constraint that the events are picked up at the time they are opened
    for i in range(n_events):
        m.addConstr(t[i] >= events_open_time[i])
        m.addConstr(t[i] <= events_close_time[i])
        
    # add time constraint : time to pick up event is smaller or equal to the time event occurs
    for i in range(n_events):
        for j in range(n_events):
            m.addConstr(t[j] - t[i] >= (events_open_time[j] - events_close_time[i]) +
                        (time_events_to_events[i, j] -
                         (events_open_time[j] - events_close_time[i])) * x[i, j])
        for j in range(n_cars):
            m.addConstr(t[i] >= events_open_time[i] + (start_time + time_cars_to_events[j, i] - events_open_time[i]) * y[j, i])

    # event can not be picked up after it is picked up (X_{c,c} = 0 by definition)
    for i in range(n_events):
        m.addConstr(x[i, i] == 0)  # an event cant pick itself up

    r_events       = 0   # reward for events that are closed after an event
    r_cars         = 0   # reward for events that are closed after car
    p_events       = 0   # penalty for events that are canceled
    p_events_opened = 0  # penatly for time that events waited
    
    for i in range(n_events):
        for j in range(n_events):
            r_events += reward_events_to_events[i, j]*x[i, j]
            p_events_opened -= opened_penalty * (t[j] - events_open_time[j]) * x[i, j]

        for j in range(n_cars):
            r_cars += reward_cars_to_events[j, i]*y[j, i]
            p_events_opened -= opened_penalty * (t[i] - events_open_time[i]) * y[j, i]
        p_events -= cancel_penalty * (1 - p[i])

    # find the final objective of optimization problem (maximum since we are looking at the rewards)
    obj = r_events + r_cars + p_events + p_events_opened

    # adding constraints and objective to gurobi model 
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam('OutputFlag', output_flag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def analysis_and_plot_results(m, cars_pos, events_pos, events_open_time, events_close_time, plot_figures, file_loc, file_name, gs):
    """
    this function creates car paths throughout the simulation and post processes the output of optimization.
    in addition the function plots the results and creats gif if wanted
    :param m: model output
    :param cars_pos: matrix of car positions [n_cars, 2]
    :param events_pos: matrix of event positions [n_events, 2]
    :param events_open_time: vector of events starting time [n_events, 1]
    :param events_close_time: vector of events closing time [n_events, 1]
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
    param['c_events_to_events'] = np.array(param['c_events_to_events']).reshape(n_events, n_events)
    param['c_cars_to_events']   = np.array(param['c_cars_to_events']).reshape(n_cars, n_events)
    param['is_picked_up']      = np.array(param['is_picked_up']).reshape(n_events, 1)
    param['pick_up_time']      = np.array(param['pick_up_time']).reshape(n_events, 1)
    
    e_ids     = np.array(range(n_events))
    # list of locations each car stopped at for pick up
    path_cars  = [[cars_pos[i, :]] for i in range(n_cars)]
    target_pos = []
    # list of number of events per car
    events_per_car = [[] for i in range(n_cars)]
    for i in range(n_cars):
        e_picked_up = e_ids[param['c_cars_to_events'][i, :] == 1]
        if e_picked_up.size > 0:  # car i picked up an event 
            path_cars[i].append(events_pos[e_picked_up, :][0])
            target_pos.append(events_pos[e_picked_up, :][0])
            events_per_car[i].append(e_picked_up[0])
            car_not_finished = True 
            while car_not_finished:  # car picked up more than on event
                e_picked_up = e_ids[param['c_events_to_events'][e_picked_up[0], :] == 1]
                if e_picked_up.size > 0:
                    path_cars[i].append(events_pos[e_picked_up, :][0])
                    events_per_car[i].append(e_picked_up[0])
                else:
                    car_not_finished = False
        else:  # car did not pick up any event
            path_cars[i].append(cars_pos[i, :])
            target_pos.append(cars_pos[i, :])
        path_cars[i] = np.array(path_cars[i])
    # list of car's full path 
    car_full_path = [[cars_pos[i, :]] for i in range(n_cars)]
    current_pos  = copy.deepcopy(cars_pos)
    max_time = np.max(events_close_time).astype(int)
    # for each car find the full route based on the events it picked up and their locations
    for c in range(n_cars):
        k = 0
        for t in range(int(max_time)):
            if cdist(current_pos[c, :].reshape(1, 2), target_pos[c].reshape(1, 2), metric='cityblock') > 0:
                # move car towards target
                current_pos[c, :] = move_car(current_pos[c, :], target_pos[c])
            else:
                if len(events_per_car[c]) > 0:
                    # car has reached current target, check if target can be picked up
                    if t < param['pick_up_time'][events_per_car[c][k]]:
                        # target has not opened yet, needs to wait for target to open
                        target_pos[c] = copy.deepcopy(current_pos[c, :])
                    else:
                        if len(events_per_car[c])-1 > k:
                            # target is opened, advance car to next target
                            k += 1
                            target_pos[c] = copy.deepcopy(events_pos[events_per_car[c][k], :])
                        else:
                            # car reached target and there are no more targets in cars list
                            target_pos[c] = copy.deepcopy(current_pos[c, :])
                        current_pos[c, :] = move_car(current_pos[c, :], target_pos[c])
            car_full_path[c].append(copy.deepcopy(current_pos[c, :]))

    if plot_figures:
        fig, ax = plt.subplots()
        ax.set_title('total results')
        # create legend for plot -
        ax.scatter([], [], c='y', marker='*', label='Opened')
        ax.scatter([], [], c='k', marker='*', label='Created')
        ax.scatter([], [], c='r', marker='*', label='Canceled')
        ax.scatter([], [], c='g', marker='*', label='Closed')

    # create list of accumulated closed/ canceled and opened events at each time step
    num_closed_vec = np.zeros(max_time)
    num_canceled_vec = np.zeros(max_time)
    num_opened_vec = np.zeros(max_time)
    num_total_events_vec = np.zeros(max_time)
    for t in range(int(max_time)):
        num_events_opened   = 0
        num_events_closed   = 0
        num_events_canceled = 0
        for e in range(n_events):
            # event is closed at this time step
            if param['pick_up_time'][e] <= t and param['is_picked_up'][e]:
                num_events_closed += 1
                # event is canceled at this time
            if events_close_time[e]  < t and not param['is_picked_up'][e]:
                num_events_canceled += 1
                # event has not yet been picked up but is opened
            if events_open_time[e] <= t and events_close_time[e] >= t:
                if not param['is_picked_up'][e]:
                    num_events_opened += 1
                elif param['is_picked_up'][e] and param['pick_up_time'][e] > t:
                    num_events_opened += 1

        num_total_events = num_events_closed + num_events_canceled + num_events_opened
        # add resutls to vector vs. time
        num_total_events_vec[t] = num_total_events
        num_closed_vec[t] = num_events_closed
        num_opened_vec[t] = num_events_opened
        num_canceled_vec[t] = num_events_canceled
        if plot_gif:
            current_cars_pos = np.array([c[t] for c in car_full_path])
            plot_for_gif(current_cars_pos, events_pos, param['pick_up_time'], events_open_time, events_close_time,
                         param['is_picked_up'], file_name, t, gs)
    time_vec = np.array(range(int(max_time)))
    data_out = {'closed_events'      : num_closed_vec,
               'canceled_events'     : num_canceled_vec,
               'opened_events'       : num_opened_vec,
               'all_events'          : num_total_events_vec,
               'time'                : time_vec}
    if plot_figures:
        ax.plot(time_vec, num_closed_vec,       c='g', marker='*')
        ax.plot(time_vec, num_canceled_vec,     c='r', marker='*')
        ax.plot(time_vec, num_opened_vec,       c='y', marker='*')
        ax.plot(time_vec, num_total_events_vec, c='k', marker='*')
        ax.grid(True)
        ax.legend()
        plt.show()

    if plot_gif:
        list_names = [file_name + '_' + str(t) + '.png' for t in range(int(max_time))]
        create_gif(file_loc + file_name + '/', list_names, 1, file_name)
    return data_out, param, car_full_path


def plot_for_gif(car_pos, events_pos, events_pick_up_time, events_open_time, events_close_time,
                 is_event_picked_up, file_name, t, gs):
    """
    this function creates figures for gif where each figure represents the current car's location and the locations of
    events opened
    :param car_pos: position of cars at time t [n_cars, 2]
    :param events_pos: position of events [n_events, 2]
    :param events_pick_up_time: time that event was picked up [n_events, 1]
    :param events_open_time: time event started being opened [n_events, 1]
    :param events_close_time: time event closed if not picked up [n_events, 1]
    :param is_event_picked_up: True/False if event is picked up or not [n_events, 1]
    :param gs: grid size [int]
    :param t: current time to plot [int]
    :return: graph to add to list of graphs
    """
    fig, ax = plt.subplots()
    ax.set_title('time: {0}'.format(t))
    for c in range(car_pos.shape[0]):
        ax.scatter(car_pos[c, 0], car_pos[c, 1], c='k', alpha=0.5)
    # create legend for plot
    ax.scatter([], [], c='b', marker='*', label='Opened')
    ax.scatter([], [], c='r', label='Canceled')
    ax.scatter([], [], c='g', label='Closed')
    for i in range(events_pos.shape[0]):
        # plot event as closed
        if t >= events_pick_up_time[i] and is_event_picked_up[i]:
            ax.scatter(events_pos[i, 0], events_pos[i, 1], c='g', alpha=0.2)
        # plot event as opened
        elif (events_open_time[i] <= t) and (events_close_time[i] > t):
            ax.scatter(events_pos[i, 0], events_pos[i, 1], c='b', alpha=0.7)
        # plot event as canceled
        elif t > events_close_time[i]:
            ax.scatter(events_pos[i, 0], events_pos[i, 1], c='r', alpha=0.2)
        # plot event as not yet opened
        else:
            ax.scatter(events_pos[i, 0], events_pos[i, 1], c='y', alpha=0.2)
    ax.set_xlim([-1, gs[0] + 1])
    ax.set_ylim([-1, gs[1] + 1])
    ax.grid(True)
    plt.legend()
    plt.savefig(file_name + '_' + str(t) + '.png')
    plt.close()
    return


def main():
    with open('config.json') as f:
        param_dict = json.load(f)

    # load parameters form json file
    close_reward   = param_dict['close_reward']  # reward for closing event
    cancel_penalty = param_dict['cancel_penalty']  # penalty for canceling an event
    opened_penalty = param_dict['opened_penalty']  # penalty for each time step an event is still opened
    sim_seed       = param_dict['sim_seed']  # seed for event and car data

    grid_size            = [param_dict['grid_size'], param_dict['grid_size']]  # size of enviroment
    n_cars               = param_dict['n_cars']  # number of cars in simulation
    t_start              = param_dict['t_start']  # simulation start time
    delta_open_time      = param_dict['delta_open_time']  # time each event is opened [delta_open_time = close_time - open_time]
    length_sim           = param_dict['length_sim']  # full length of simulation
    lam                  = param_dict['lam']  # lambda, parameter for poisson distribution
    # set random seed for problem
    np.random.seed(sim_seed)
    plot_figures = param_dict['plot_figures']  # flag if should plot figures [0 - don't plot, 1 - plot]
    print_logs   = param_dict['print_logs']  # flag if should print simulation log [0 - should not print, 1 - should print]
    # name and location for saving output
    file_loc  = param_dict['file_loc']  # location of output file and figures
    file_name = param_dict['file_name']  # name of files


    # create car initial positions
    car_pos = np.reshape(np.random.randint(0, grid_size[0], 2 * n_cars), (n_cars, 2))

    # create event positions and time
    event_pos, event_times = create_events_distribution(grid_size[0], 0, length_sim, lam, delta_open_time)
    event_start_time = event_times[:, 0]
    event_end_time = event_times[:, 1]

    # run optimization using gurobi
    s_time = time.time()
    m, obj = run_max_flow_opt(t_start, car_pos, event_pos,
                              event_start_time, event_end_time,
                              close_reward, cancel_penalty, opened_penalty)
    e_time = time.time()
    print("simulation run time is: " + str(e_time - s_time))
    if print_logs:
        # print results of optimization
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))

        print('Obj: %g' % obj.getValue())
    # run analysis and plot results
    data_out, param, cars_paths = analysis_and_plot_results(m, car_pos, event_pos, event_start_time, event_end_time,
                                                            plot_figures, file_loc, file_name, grid_size)
    # save results
    with open(file_loc + file_name + '.p', 'wb') as out:
        pickle.dump({'sol': param,
                     'cars_paths': cars_paths,
                     'sol_post_process': data_out,
                     'n_cars': n_cars,
                     'sim_time': length_sim,
                     'gs': grid_size,
                     'lam': lam,
                     'del_open_time': delta_open_time,
                     'seed': sim_seed,
                     'event_pos': event_pos,
                     'event_start_time': event_start_time,
                     'event_end_time': event_end_time
                     }, out)


if __name__ == '__main__':
    main()
    print('Done.')
