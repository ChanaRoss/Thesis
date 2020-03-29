import time
import itertools as iter
from copy import deepcopy
# import my own code:
from RL_anticipatory.utils import move_to, to_numpy
from Simulation.Anticipitory.with_RL.create_distributions import *
from MixedIntegerOptimization.offlineOptimizationProblem_TimeWindow import runMaxFlowOpt as runMaxFlowOptTimeWindow
from MixedIntegerOptimization.offlineOptimizationProblemMaxFlow import runMaxFlowOpt as runMaxFlowOpt


class AnticipatoryState:
    def __init__(self, data_input, stochastic_input_dict, sim_input_dict):
        self.is_training = sim_input_dict['is_training']
        self.print_debug = sim_input_dict['print_debug']
        self.data_input = data_input.clone()
        self.should_calc_all_options = sim_input_dict['should_calc_all_options']
        self.should_calc_anticipatory = stochastic_input_dict['should_calc_anticipatory']
        self.stochastic_mat = stochastic_input_dict['future_mat']  # of shape [x, y, t, p(n events)]
        self.n_stochastic_runs = stochastic_input_dict['n_stochastic_runs']
        self.batch_size = data_input.num_graphs
        self.sim_length = sim_input_dict['sim_length']
        self.dist_lambda = sim_input_dict['dist_lambda']
        self.n_prediction_steps = sim_input_dict['n_prediction_steps']
        self.events_open_time = sim_input_dict['events_open_time']
        data_list = data_input.to_data_list()
        self.n_cars = data_list[0].car_loc.shape[0]
        self.time = 0  # simulation starts at time 0
        self.dim = sim_input_dict['graph_dim']
        self.car_cur_loc = torch.zeros((self.batch_size, self.n_cars, 2), device=data_input['car_loc'].device)
        self.actions_to_move_tensor = move_to(sim_input_dict['possible_actions'], data_input['car_loc'].device)
        self.actions_indexes_all_options, self.actions_to_move_tensor_all_options = self.get_all_actions_tensor()
        self.events_loc_dict = {}
        self.events_time_dict = {}
        self.location_to_events_dict = {}
        n_movement_options = self.actions_to_move_tensor_all_options.shape[0]  # all movements possible

        self.movement_cost = torch.zeros((self.batch_size, self.sim_length), device=data_input['car_loc'].device)
        self.events_cost = torch.zeros_like(self.movement_cost, device=data_input['car_loc'].device)
        self.anticipatory_cost = torch.zeros_like(self.movement_cost, device=data_input['car_loc'].device)

        self.anticipatory_cost_options = torch.zeros((self.batch_size, self.sim_length, n_movement_options),
                                                     device=self.events_cost.device)
        self.movement_cost_options = torch.zeros_like(self.anticipatory_cost_options, device=data_input['car_loc'].device)
        self.events_cost_options = torch.zeros_like(self.anticipatory_cost_options, device=data_input['car_loc'].device)


        n_events = 0
        self.cars_log = {}
        for i_b in range(self.batch_size):
            self.cars_log[i_b] = {}
            for i_c in range(self.n_cars):
                self.cars_log[i_b][i_c] = []
            graph = data_list[i_b]
            self.car_cur_loc[i_b, ...] = graph.car_loc
            self.events_loc_dict[i_b] = graph.events_loc
            self.events_time_dict[i_b] = graph.events_time
            self.location_to_events_dict[i_b] = self.create_location_events_dict(graph.events_loc, graph.events_time)
            if graph.events_loc.shape[0] > n_events:
                n_events = graph.events_loc.shape[0]
        self.n_events = n_events
        self.cars_route = torch.zeros(self.data_input.num_graphs, self.n_cars, self.sim_length+1, 2)
        self.cars_route[:, :, 0, :] = self.car_cur_loc.clone()
        self.actions_chosen = torch.zeros((self.batch_size, self.n_cars, self.sim_length, 2),
                                          device=data_input['car_loc'].device)
        self.events_answer_time = torch.ones((self.batch_size, self.n_events), device=data_input['car_loc'].device)*999
        self.n_events_closed = torch.zeros((self.batch_size, self.sim_length),  device=data_input['car_loc'].device)
        self.events_status = {'answered': torch.zeros((self.batch_size, self.n_events, 1),
                                                      device=data_input['car_loc'].device, dtype=torch.bool),
                              'canceled': torch.zeros((self.batch_size, self.n_events, 1),
                                                      device=data_input['car_loc'].device, dtype=torch.bool)}

    def get_all_actions_tensor(self):
        all_actions = torch.stack([torch.stack(c) for c in
                                   iter.product(self.actions_to_move_tensor, repeat=self.n_cars)])
        all_actions_indexes = torch.stack([torch.stack(c) for c in
                                           iter.product(torch.tensor(range(self.actions_to_move_tensor.shape[0])),
                                                        repeat=self.n_cars)])
        return all_actions_indexes, all_actions

    def create_location_events_dict(self, events_loc, events_time):
        """
        this function creates a dictionary from grid location to events and their time windows.
        it is used to update the time feature in the graph
        :param events_loc: the locations of all events [n_events, 2] (x, y)
        :param events_time: the times of all events [n_events, 2] (open and close time)
        :return: dict where keys are locations and values are (event i_d, [event start time, event end time])
        """
        dict_out = {}
        for i_e in range(events_loc.shape[0]):
            event_loc = to_numpy(events_loc[i_e])
            event_time = events_time[i_e]
            event_loc_tuple = (event_loc[0], event_loc[1])
            if event_loc_tuple in dict_out:
                dict_out[event_loc_tuple].append((i_e, event_time[0], event_time[1]))
            else:
                dict_out[event_loc_tuple] = [(i_e, event_time[0], event_time[1])]
        return dict_out

    def update_state(self, actions):
        """
        this function updates the state after a set of actions is chosen for all cars
        :param actions: torch tensor of size [batch_size, n_cars] where each value represents the action chosen
        :return:
        """
        if self.print_debug:
            print("updating state, t:" + str(self.time))
        s_time = time.time()
        car_cur_loc, delta_movement = self.update_all_cars_state(actions)
        dist_time = 0
        all_events_closed = torch.zeros([self.batch_size], device=self.cars_route.device, dtype=torch.bool)
        for i_b in range(self.batch_size):
            if self.time > 0 and self.n_events_closed[i_b, self.time - 1] == len(self.events_loc_dict):
                all_events_closed[i_b] = True
                # in this case no need to calculate future moves, since all cars should stay put
                # and all stats should stay the same
                self.n_events_closed[i_b, self.time] = self.n_events_closed[i_b, self.time - 1]
                self.cars_route[i_b, :, self.time + 1, :] = self.car_cur_loc[i_b, ...].clone()
                self.actions_chosen[:, :, self.time, :] = torch.zeros_like(delta_movement[i_b, ...],
                                                                           device=delta_movement.device)
            else:
                n_events_currently_closed = 0
                s_time_b = time.time()
                # create opened events information for anticipatory calculation
                opened_events_pos = []
                opened_events_start_time = []
                opened_events_end_time = []
                currently_closed_events_pos = []
                currently_closed_events_start_time = []
                currently_closed_events_end_time = []
                events_loc_ = self.events_loc_dict[i_b]
                n_events_in_batch = events_loc_.shape[0]
                s_d = time.time()
                distance_matrix = torch.cdist(car_cur_loc[i_b, ...].type(torch.float), events_loc_.type(torch.float), p=1)
                e_d = time.time()
                dist_time += e_d-s_d
                if n_events_in_batch > 0 and self.is_training:
                    cars_min_dist, _ = torch.min(distance_matrix, axis=1)
                    min_distance = torch.sum(cars_min_dist)
                    self.movement_cost[i_b, self.time] = self.movement_cost[i_b, self.time] + min_distance
                # find row of relevant event in graph
                loc_row = events_loc_[:, 0] * self.dim + events_loc_[:, 1]
                graph_row = (loc_row + i_b * self.dim * self.dim).type(torch.int)
                # find which events need to be canceled and cancel these events -
                not_answered = torch.logical_not(self.events_status['answered'][i_b, :n_events_in_batch].view(-1))
                expanded_time = torch.tensor([self.time], device=self.car_cur_loc.device).expand(n_events_in_batch)
                time_finished = (self.events_time_dict[i_b][:, 1] == expanded_time)
                should_cancel = time_finished*not_answered
                cancel_indexes = torch.where(should_cancel)[0]
                for i_e_can in cancel_indexes:
                    self.cancel_event(i_b, i_e_can, graph_row[i_e_can])
                    if self.print_debug:
                        print("t:" + str(self.time) + ", canceled event id:" + str(i_e_can) + ", in batch" + str(i_b))
                not_canceled = torch.logical_not(self.events_status['canceled'][i_b,  :n_events_in_batch].view(-1))
                time_opened = (self.events_time_dict[i_b][:, 0] <= self.time) * (self.time <= self.events_time_dict[i_b][:, 1])
                # check if event is opened - (not canceled, not answered and time is within time window)
                is_event_opened = not_answered * not_canceled * time_opened
                opened_indexes = torch.where(is_event_opened)[0]
                for i_e_open in opened_indexes:
                    # in this case the event is opened , so we need to see if a car reached it's location
                    is_answered = torch.where(distance_matrix[:, i_e_open] <= 0.1)[0]
                    if is_answered.size()[0] > 0:
                        # needed for calulating optional choices -
                        currently_closed_events_pos.append(to_numpy(self.events_loc_dict[i_b][i_e_open, ...].detach()))
                        currently_closed_events_start_time.append(to_numpy(self.events_time_dict[i_b][i_e_open, 0].detach()))
                        currently_closed_events_end_time.append(to_numpy(self.events_time_dict[i_b][i_e_open, 1].detach()))
                        # calculate if event is closed -
                        car_index = is_answered[0]  # this is the car chosen to answer this specific event
                        distance_matrix[car_index, :] = 9999  # makes sure you dont use the same car for other events
                        self.cars_log[i_b][car_index.item()].append(i_e_open)
                        self.close_event(i_b, i_e_open, graph_row[i_e_open])
                        n_events_currently_closed += 1
                        if self.print_debug:
                            print("t:" + str(self.time) + ", closed event id:" + str(i_e_open) + ", in batch" + str(i_b))
                    else:
                        open_cost = self.data_input['open_cost'][i_b]
                        self.events_cost[i_b, self.time] = self.events_cost[i_b, self.time] + open_cost
                        opened_events_pos.append(to_numpy(self.events_loc_dict[i_b][i_e_open, ...].detach()))
                        opened_events_start_time.append(to_numpy(self.events_time_dict[i_b][i_e_open, 0].detach()))
                        opened_events_end_time.append(to_numpy(self.events_time_dict[i_b][i_e_open, 1].detach()))

                should_open = (self.events_time_dict[i_b][:, 0] == self.time + 1)
                should_open_indexes = torch.where(should_open)[0]
                for i_e_should_open in should_open_indexes:
                    self.data_input.x[graph_row[i_e_should_open], 3] = self.data_input.x[graph_row[i_e_should_open], 3] + 1
                    event_delta_time = self.events_time_dict[i_b][i_e_should_open, 1] - \
                                       self.events_time_dict[i_b][i_e_should_open, 0]
                    if self.data_input.x[graph_row[i_e_should_open], 4] <= event_delta_time:
                        self.data_input.x[graph_row[i_e_should_open], 4] = event_delta_time
                    if self.print_debug:
                        print("t:" + str(self.time) + ", opened event id:" + str(i_e_should_open) + ", in batch" + str(i_b))

                if self.time > 0:
                    # add number of currently closed events to previous closed events (Accumilitve)
                    self.n_events_closed[i_b, self.time] = self.n_events_closed[i_b, self.time - 1] + n_events_currently_closed
                else:  # time zero
                    self.n_events_closed[i_b, self.time] = n_events_currently_closed

                # calculate anticipatory cost for batch i_b after moving cars (known events and future events)
                if self.should_calc_anticipatory:
                    cars_loc = to_numpy(car_cur_loc[i_b, ...].clone().detach())
                    # create stochastic events (use the same events for all anticipatory calculations)
                    stochastic_event_dict = createStochasticEvents(0, self.n_stochastic_runs, 0, self.sim_length,
                                                                   self.stochastic_mat, self.events_open_time, self.time,
                                                                   'Bm_poisson', np.array([self.dim, self.dim]),
                                                                   self.dist_lambda)
                    new_anticipatory_cost = self.calc_anticipatory_cost(i_b, cars_loc, opened_events_pos,
                                                                        opened_events_start_time, opened_events_end_time,
                                                                        stochastic_event_dict)
                else:
                    new_anticipatory_cost = 0
                self.anticipatory_cost[i_b, self.time] = new_anticipatory_cost
                # calc all other costs optional if needed for loss function -
                if self.should_calc_all_options:
                    actions_chosen = self.actions_to_move_tensor_all_options[actions[i_b]]
                    # add events that were currently closed to list of opened events in order to calculate the action
                    # if we were to choose other events
                    opened_events_pos = opened_events_pos + currently_closed_events_pos
                    opened_events_start_time = opened_events_start_time + currently_closed_events_start_time
                    opened_events_end_time = opened_events_end_time + currently_closed_events_end_time
                    for i_a, optional_actions in enumerate(self.actions_to_move_tensor_all_options):
                        if torch.all(torch.eq(actions_chosen, self.actions_to_move_tensor_all_options[i_a])):
                            self.anticipatory_cost_options[i_b, self.time, i_a] = self.anticipatory_cost[i_b, self.time]
                            self.events_cost_options[i_b, self.time, i_a] = self.events_cost[i_b, self.time]
                            self.movement_cost_options[i_b, self.time, i_a] = self.movement_cost[i_b, self.time]
                        else:
                            # add optional action to car cur location before current action chosen
                            car_loc_options = self.car_cur_loc[i_b, ...].clone() + optional_actions
                            # calc anticipatory cost assuming action is chosen -
                            ant_new = self.calc_anticipatory_cost(i_b, car_loc_options, opened_events_pos,
                                                                  opened_events_start_time, opened_events_end_time,
                                                                  stochastic_event_dict)
                            self.anticipatory_cost_options[i_b, self.time, i_a] = ant_new
                            # calculate the cost of moving the cars in chosen action -
                            self.movement_cost_options[i_b, self.time, i_a] = torch.abs(optional_actions.sum())
                            if len(opened_events_pos) > 0:
                                events_opened_loc = torch.tensor(opened_events_pos)
                                distance_matrix_options = torch.cdist(car_loc_options.type(torch.float),
                                                                      events_opened_loc.type(torch.float), p=1)
                                # calculate event cost for each event opened in current state -
                                for i_e in range(len(opened_events_pos)):
                                    is_answered = torch.where(distance_matrix_options[:, i_e] <= 0.1)[0]
                                    if is_answered.size()[0] > 0:
                                        # add closed event reward to cost
                                        car_index = is_answered[0]  # this is the car chosen to answer this specific event
                                        distance_matrix_options[car_index, :] = 9999
                                        close_event_cost = - self.data_input['close_reward'][i_b]
                                        self.events_cost_options[i_b, self.time, i_a] = self.events_cost_options[i_b, self.time, i_a] + close_event_cost
                                    else:
                                        if opened_events_end_time[i_e] == self.time:
                                            events_cancel_cost = self.data_input['cancel_cost'][i_b]
                                            self.events_cost_options[i_b, self.time, i_a] = self.events_cost_options[i_b, self.time, i_a] + events_cancel_cost
                                        else:
                                            # add opened event to cost
                                            event_opened_cost = self.data_input['open_cost'][i_b]
                                            self.events_cost_options[i_b, self.time, i_a] = self.events_cost_options[i_b, self.time, i_a] + event_opened_cost
                e_time_b = time.time()
                # print("run time for batch:"+str(i_b)+", is:"+str(e_time_b -s_time_b))
                self.car_cur_loc[i_b, ...] = car_cur_loc[i_b, ...].clone()
                self.actions_chosen[i_b, :, self.time, :] = delta_movement[i_b, ...].clone()
                self.cars_route[i_b, :, self.time + 1, :] = car_cur_loc[i_b, ...].clone()
        self.time = self.time + 1  # update simulation time
        e_time = time.time()
        if self.print_debug:
            print("time to update state is:"+str(e_time-s_time))
            print("dist calc took"+str(dist_time))
        return

    def update_all_cars_state(self, actions):
        n_actions = self.actions_to_move_tensor_all_options.shape[0]
        expanded_actions_to_move = self.actions_to_move_tensor_all_options[None, ...].\
            expand(self.batch_size, n_actions, self.n_cars, 2)
        expanded_actions_chosen = actions[:, None, None, None].\
            expand(self.batch_size, 1, self.n_cars, 2)
        delta_movement = expanded_actions_to_move.gather(1, expanded_actions_chosen).squeeze()
        new_loc = self.car_cur_loc.clone() + delta_movement
        dim_lim = torch.tensor([self.dim-1], device=actions.device)[None, :].expand([self.batch_size, self.n_cars])
        zero_lim = torch.zeros_like(dim_lim, device=actions.device)
        can_move_x = ((zero_lim <= new_loc[:, :, 0]) * (new_loc[:, :, 0] <= dim_lim))
        can_move_y = ((zero_lim <= new_loc[:, :, 1]) * (new_loc[:, :, 1] <= dim_lim))
        can_move_cars = can_move_x*can_move_y
        invalid_indexes = torch.where(can_move_cars == False)
        for i_i in range(len(invalid_indexes[0])):
            i_batch_inv = invalid_indexes[0][i_i]
            i_car_inv = invalid_indexes[1][i_i]
            available_actions = self.get_available_moves(self.car_cur_loc[i_batch_inv, i_car_inv], i_car_inv)
            random_delta_movement = available_actions[torch.randint(0, available_actions.shape[0], (1, 1))].view(-1)
            delta_movement[i_batch_inv, i_car_inv, :] = random_delta_movement.clone()
            new_loc[i_batch_inv, i_car_inv] = self.car_cur_loc[i_batch_inv, i_car_inv, :].clone() + random_delta_movement

        batch_ids = torch.tensor(range(self.batch_size), device=self.car_cur_loc.device)
        # subtract 1 from graph every where a car has left
        for i_car in range(self.n_cars):
            loc_row = (self.car_cur_loc[:, i_car, 0]*self.dim + self.car_cur_loc[:, i_car, 1]).int()
            graph_rows = loc_row + self.dim*self.dim*batch_ids
            self.data_input.x[graph_rows, 2] = self.data_input.x[graph_rows, 2] - 1
        new_cost = self.data_input['movement_cost'] * torch.abs(torch.sum(delta_movement, axis=[1, 2]))
        self.movement_cost[:, self.time] = self.movement_cost[:, self.time] + new_cost
        car_cur_loc = new_loc.clone()
        # add 1 to graph every place a new car is
        for i_car in range(self.n_cars):
            loc_row = (new_loc[:, i_car, 0]*self.dim + new_loc[:, i_car, 1]).int()
            graph_rows = loc_row + self.dim*self.dim*batch_ids
            self.data_input.x[graph_rows, 2] = self.data_input.x[graph_rows, 2] + 1
        return car_cur_loc, delta_movement

    def update_car_state(self, i_b, i_c, car_cur_loc, actions):
        cur_loc_temp = car_cur_loc[i_b, i_c, ...].clone()
        delta_movement = self.get_action_from_index(i_c, actions[i_b])
        new_loc_temp = cur_loc_temp.clone() + delta_movement
        can_move_car = (0 <= new_loc_temp[0] <= self.dim - 1) and \
                       (0 <= new_loc_temp[1] <= self.dim - 1)
        if not can_move_car:
            avialable_actions = self.get_available_moves(cur_loc_temp, i_c)
            random_delta_movement = avialable_actions[torch.randint(0, avialable_actions.shape[0], (1, 1))].view(-1)
            delta_movement = random_delta_movement
            new_loc_temp = cur_loc_temp.clone() + random_delta_movement
        self.cars_route[i_b, i_c, self.time+1, ...] = new_loc_temp.clone()
        self.actions_chosen[i_b, i_c, self.time, :] = delta_movement.clone()
        # subtract 1 from graph where the car came from in feature tensor
        loc_row = (cur_loc_temp[0] * self.dim + cur_loc_temp[1]).int()
        graph_row = loc_row + i_b * self.dim * self.dim
        self.data_input.x[graph_row, 2] = self.data_input.x[graph_row, 2] - 1
        assert (self.data_input.x[loc_row + i_b * self.dim * self.dim, 2] >= 0),\
            "tried to subtract 1 from car feature even though there was no car there, t:"+str(self.time)+\
            ", car id:"+str(i_c)+", batch id:"+str(i_b)
        # add this cars movement cost to total movement cost
        new_cost = self.data_input['movement_cost'][i_b] * torch.abs(torch.sum(delta_movement))
        self.movement_cost[i_b, self.time] = self.movement_cost[i_b, self.time] + new_cost
        car_cur_loc[i_b, i_c, ...] = new_loc_temp
        # add 1 to graph where the new car location is in feature tensor
        loc_row = (new_loc_temp[0] * self.dim + new_loc_temp[1]).int()
        graph_row = loc_row + i_b * self.dim * self.dim
        self.data_input.x[graph_row, 2] = self.data_input.x[graph_row, 2] + 1
        return car_cur_loc

    def get_available_moves(self, cur_loc_temp, i_c):
        available_moves = []
        for action in self.actions_to_move_tensor:
            new_loc = cur_loc_temp + action
            if (0 <= new_loc[0] <= self.dim - 1) and (0 <= new_loc[1] <= self.dim - 1):
                available_moves.append(action)
        return torch.stack(available_moves)

    def cancel_event(self, i_b, i_e, graph_row):
        self.events_status['canceled'][i_b, i_e] = True
        cancel_cost = self.data_input['cancel_cost'][i_b]
        self.events_cost[i_b, self.time] = self.events_cost[i_b, self.time] + cancel_cost
        self.events_cost_options[i_b, self.time, :] = self.events_cost_options[i_b, self.time, :] + cancel_cost
        self.update_time_feature(i_b, i_e, graph_row)  # update time feature to be without this event
        self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] - 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to cancel event from row with no events!! event num:"+str(i_e)+", in batch:"+str(i_b)

    def update_time_feature(self, i_b, i_e, graph_row):
        """
        this function updates the time feature for the node of event
        :param i_b: batch id
        :param i_e: event id that needs to be changed
        :param graph_row: graph row of event that should be changed
        :return:
        """
        # get all events at this location in order to update time feature
        event_loc = to_numpy(self.events_loc_dict[i_b][i_e, ...])
        events_in_node_list = self.location_to_events_dict[i_b][(event_loc[0], event_loc[1])]
        max_open_time = 0
        for e in events_in_node_list:
            # event is not the one we are changing and is opened and time is larger than ending time
            if e[0] != i_e and (e[1] >= self.time) and (e[2] > max_open_time):
                max_open_time = e[2]  # update max time to new end time
        self.data_input.x[graph_row, 4] = max_open_time

    def close_event(self, i_b, i_e, graph_row):
        self.events_status['answered'][i_b, i_e] = True
        # subtract reward for closing event from total cost
        self.events_cost[i_b, self.time] = self.events_cost[i_b, self.time] - self.data_input['close_reward'][i_b]
        # close event in graph - subtract one from relevant node
        self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] - 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to close event from row with no events!!"
        self.update_time_feature(i_b, i_e, graph_row)  # update time feature to be without this event
        self.events_answer_time[i_b, i_e] = self.time


    def get_action_from_index(self, i_c, i_a):
        car_movement = self.actions_to_move_tensor_all_options[i_a, i_c, :]
        return car_movement

    def get_cost(self):
        """
        this function combines all costs in problem and returns the total cost of updating state
        it assumes a specific action is chosen
        :return: torch tensor of size [batch_size, time]
        """
        answered_events = self.events_status['answered'].sum(1)
        for i_b in range(self.batch_size):
            self.events_cost[i_b, -1] = self.data_input['cancel_cost'][i_b] * \
                                        (self.events_time_dict[i_b].shape[0] - answered_events[i_b])
        cost = self.movement_cost + self.events_cost + self.anticipatory_cost
        return cost

    def get_optional_costs(self):
        """
        this function returns costs of all possible actions assuming the update state already occurred
        it returns the cost for all times
        :return: tensor of size [batch_size, sim_length, possible_actions_size]
        """
        optional_costs = self.movement_cost_options + self.events_cost_options + self.anticipatory_cost_options
        return optional_costs

    def calc_anticipatory_cost_options(self, i_b, current_events_pos_list, current_event_start_time_list, current_events_end_time_list):
        """
        this function calculates the anticipatory cost assuming cars are in known location and future events are from future matrix
        :return:
        """
        anticipatory_cost_all_actions = torch.zeros(self.actions_to_move_tensor_all_options.shape[0],
                                                    device=self.car_cur_loc.device)
        optimization_method = 'MaxFlow'  # for now assuming anticipatory is max flow only , since timeWindow is not fast
        # create stochastic events
        stochastic_event_dict = createStochasticEvents(0, self.n_stochastic_runs, 0, self.sim_length,
                                                       self.stochastic_mat, self.events_open_time, self.time,
                                                       'Bm_poisson', np.array([self.dim, self.dim]), self.dist_lambda)
        stochastic_cost = torch.zeros((self.n_stochastic_runs, 1), device=self.car_cur_loc.device)
        for i_a, actions in self.actions_to_move_tensor_all_options:
            # update car loc based on action chosen
            cars_pos = to_numpy(self.car_cur_loc[i_b, ...].clone().detach()) + actions
            # calculate cost of deterministic algorithm -
            for j in range(len(stochastic_event_dict)):
                if len(stochastic_event_dict[j]['eventsPos']) + len(current_events_pos_list) > 0:
                    # there are events to be tested in deterministic optimization:
                    temp = [current_events_pos_list.append(e) for e in stochastic_event_dict[j]['eventsPos']]
                    temp = [current_event_start_time_list.append(e[0]) for e in stochastic_event_dict[j]['eventsTimeWindow']]
                    temp = [current_events_end_time_list.append(e[1]) for e in stochastic_event_dict[j]['eventsTimeWindow']]
                    current_events_pos = np.array(current_events_pos_list).reshape(len(current_events_pos_list), 2)
                    current_event_start_time = np.array(current_event_start_time_list)
                    current_events_end_time = np.array(current_events_end_time_list)
                    stime = time.process_time()
                    if optimization_method == 'TimeWindow':
                        m, obj = runMaxFlowOptTimeWindow(0, cars_pos, current_events_pos,
                                                         current_event_start_time, current_events_end_time,
                                                         to_numpy(self.data_input['close_reward'][i_b].detach()),
                                                         to_numpy(self.data_input['cancel_cost'][i_b].detach()),
                                                         to_numpy(self.data_input['open_cost'][i_b].detach()), 0)
                    else:
                        m, obj = runMaxFlowOpt(0, cars_pos, current_events_pos,
                                               current_event_start_time + self.events_open_time,
                                               to_numpy(self.data_input['close_reward'][i_b].detach()),
                                               to_numpy(self.data_input['cancel_cost'][i_b].detach()),
                                               to_numpy(self.data_input['open_cost'][i_b].detach()))
                    etime = time.process_time()
                    # runTime = etime - stime
                    # if shouldPrint:
                    # print("finished MIO for run:"+str(j+1)+"/"+str(len(stochasticEventsDict)))
                    # print("run time of MIO is:"+str(runTime))
                    # print("cost of MIO is:"+str(-obj.getValue()))
                    stochastic_cost[j] = -obj.getValue()
                else:
                    stochastic_cost[j] = 0
            # calculate expected cost of all stochastic runs for this specific batch
            expected_cost = torch.mean(stochastic_cost)
            anticipatory_cost_all_actions[i_a] = expected_cost
        return anticipatory_cost_all_actions

    def calc_anticipatory_cost(self, i_b, cars_pos, current_events_pos, current_event_start_time,
                               current_events_end_time, stochastic_event_dict):
        """
        this function calculates the anticipatory cost assuming cars are in known location and future events are from future matrix
        :return:
        """
        optimization_method = 'MaxFlow'  # for now assuming anticipatory is max flow only , since timeWindow is not fast
        stochastic_cost = torch.zeros((self.n_stochastic_runs, 1), device=self.car_cur_loc.device)
        # calculate cost of deterministic algorithm -
        for j in range(len(stochastic_event_dict)):
            if len(stochastic_event_dict[j]['eventsPos']) + len(current_events_pos) > 0:
                current_events_pos_list = deepcopy(current_events_pos)
                current_event_start_time_list = deepcopy(current_event_start_time)
                current_events_end_time_list = deepcopy(current_events_end_time)
                # there are events to be tested in deterministic optimization:
                temp = [current_events_pos_list.append(e) for e in stochastic_event_dict[j]['eventsPos']]
                temp = [current_event_start_time_list.append(e[0]) for e in stochastic_event_dict[j]['eventsTimeWindow']]
                temp = [current_events_end_time_list.append(e[1]) for e in stochastic_event_dict[j]['eventsTimeWindow']]
                current_events_pos = np.array(current_events_pos_list).reshape(len(current_events_pos_list), 2)
                current_event_start_time = np.array(current_event_start_time_list)
                current_events_end_time = np.array(current_events_end_time_list)
                stime = time.process_time()
                if optimization_method == 'TimeWindow':
                    m, obj = runMaxFlowOptTimeWindow(0, cars_pos, current_events_pos,
                                                     current_event_start_time, current_events_end_time,
                                                     to_numpy(self.data_input['close_reward'][i_b].detach()),
                                                     to_numpy(self.data_input['cancel_cost'][i_b].detach()),
                                                     to_numpy(self.data_input['open_cost'][i_b].detach()), 0)
                else:
                    m, obj = runMaxFlowOpt(0, cars_pos, current_events_pos,
                                           current_event_start_time + self.events_open_time,
                                           to_numpy(self.data_input['close_reward'][i_b].detach()),
                                           to_numpy(self.data_input['cancel_cost'][i_b].detach()),
                                           to_numpy(self.data_input['open_cost'][i_b].detach()))
                    # obj = 0
                etime = time.process_time()
                # runTime = etime - stime
                # if shouldPrint:
                # print("finished MIO for run:"+str(j+1)+"/"+str(len(stochasticEventsDict)))
                # print("run time of MIO is:"+str(runTime))
                # print("cost of MIO is:"+str(-obj.getValue()))
                # stochastic_cost[j] = 0
                stochastic_cost[j] = -obj.getValue()
            else:
                stochastic_cost[j] = 0
        # calculate expected cost of all stochastic runs for this specific batch
        expected_cost = torch.mean(stochastic_cost)
        return expected_cost

    def all_finished(self):
        if self.sim_length > self.time:
            return False
        else:
            return True

