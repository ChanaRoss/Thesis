import numpy as np
import itertools as itr
import random
import sys
import time
import pickle

sys.path.insert(0, '/Users/chanaross/dev/Thesis/MixedIntegerOptimization/')
from offlineOptimizationProblemMaxFlow import runMaxFlowOpt, plotResults
from offlineOptimizationProblem_TimeWindow import runMaxFlowOpt as runMaxFlowOptTimeWindow



def generate_k_unique_car_positions(k, num_cars, row_dim, col_dim):
    time_out = 100*num_cars
    itrs = 0

    car_locations = set()
    possible_coords = list(itr.product(range(row_dim), range(col_dim)))

    while len(car_locations) < k:
        tmp = tuple(sorted(random.sample(possible_coords, num_cars)))  # grab a random set of coordiantes for cars
        if tmp in car_locations:  # if this choice is a duplicate, dont add, increment time out counter
            itrs += 1
            if itrs == time_out:
                print("Error: couldn't generate unique car positions after {} iterations.".format(itrs))
                exit()
        else:
            car_locations.add(tmp)  # this is a unique choice, add to list

    # convert car location sets into ndarray
    car_location_arrs = [np.array(cp) for cp in car_locations]
    return car_location_arrs


def create_events(events_cdf_matrix, eventTimeWindow):
    eventPos = []
    eventTimes = []
    x_size = events_cdf_matrix.shape[0]
    y_size = events_cdf_matrix.shape[1]
    t_size = events_cdf_matrix.shape[2]
    for t in range(t_size):
        numEventsCreated = 0
        for x in range(x_size):
            for y in range(y_size):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = events_cdf_matrix[x, y, t, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t)
                    numEventsCreated += 1
        # print("num events created for time : "+str(t) + ", is:"+str(numEventsCreated))

    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    return eventsPos, eventsTimeWindow


def calc_expected_max_flow(car_positions, cdf_mat, n, tw, optMethod, close_reward=80, cancel_pen=140, open_not_comm=5):
    all_rewards = []
    t0 = time.clock()
    for i in range(n):
        event_pos, event_time = create_events(cdf_mat, tw)
        if event_pos.size > 0:
            if optMethod == 'MaxFlow':
                m, obj = runMaxFlowOpt(0, car_positions, event_pos, event_time[:, 1], close_reward, cancel_pen, open_not_comm)
            else:
                m, obj = runMaxFlowOptTimeWindow(0, car_positions, event_pos, event_time[:, 0], event_time[:, 1],
                                       close_reward, cancel_pen, open_not_comm, 0)
            rew = -obj.getValue()
            all_rewards.append(rew)
    t1 = time.clock()
    print("expected reward calc time (seconds): {}".format(t1 - t0))
    return np.mean(all_rewards)


def main():
    # global parameters
    ROW_MIN, ROW_MAX = 0, 10
    COL_MIN, COL_MAX = 40, 50
    PRED_LEN = 5
    NUM_CARS = 4
    STOCHASTIC_ITERS  = 30
    CAR_PERMUTATION   = 100
    EVENT_TIME_WINDOW = 3
    str_name = "_eval_ncars_" + str(NUM_CARS) + "_nperm_" + str(CAR_PERMUTATION) + ".p"
    DATA_PATH = "/Users/chanaross/dev/Thesis/UberData/4D_ProbabilityMat_allDataLatLonCorrected_20MultiClass_CDF_500gridpickle_30min.p"

    optMethod      = 'MaxFlow'
    OUT_ITEMS_PATH = "/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/network_input_" + optMethod + str_name
    OUT_DEBUG_PATH = "/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/network_input_debug_" + optMethod + str_name
    # load the data
    data = np.load(DATA_PATH)

    # filter out irrelevent area
    assert(0 <= ROW_MIN < ROW_MAX < data.shape[0])
    assert(0 <= COL_MIN < COL_MAX < data.shape[1])
    data = data[ROW_MIN:ROW_MAX, COL_MIN:COL_MAX, :, :]

    # create data dict and debugging data
    item_dict = {}
    deb_dict = {}
    id = 0
    for t in range(0, data.shape[2] - PRED_LEN + 1):
        tmp_mat = data[:, :, t:t+PRED_LEN, :]
        k_car_positions = generate_k_unique_car_positions(CAR_PERMUTATION, NUM_CARS, data.shape[0], data.shape[1])

        for j in range(CAR_PERMUTATION):
            expected_reward = calc_expected_max_flow(k_car_positions[j], tmp_mat, STOCHASTIC_ITERS, EVENT_TIME_WINDOW, optMethod)
            item_dict[id] = (tmp_mat, k_car_positions[j], expected_reward)
            deb_dict[id] = {'time'      : t,
                            'bounds'    : (ROW_MIN, ROW_MAX, COL_MIN, COL_MAX),
                            'ncars'     : NUM_CARS,
                            'predLen'   : PRED_LEN,
                            'stochIter' : STOCHASTIC_ITERS,
                            'timeWindow': EVENT_TIME_WINDOW}
            id += 1

    pickle.dump(item_dict, open(OUT_ITEMS_PATH, "wb"))
    pickle.dump(deb_dict, open(OUT_DEBUG_PATH, "wb"))
    return


if __name__ == '__main__':
    main()
    print('done.')

