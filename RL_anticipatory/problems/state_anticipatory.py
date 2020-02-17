import time
import itertools as iter
from copy import deepcopy
# import my own code:
from Simulation.Anticipitory.with_RL.create_distributions import *
from MixedIntegerOptimization.offlineOptimizationProblem_TimeWindow import runMaxFlowOpt as runMaxFlowOptTimeWindow
from MixedIntegerOptimization.offlineOptimizationProblemMaxFlow import runMaxFlowOpt as runMaxFlowOpt


class AnticipatoryState:
    def __init__(self, data_input, stochastic_input_dict, sim_input_dict):
        self.data_input = data_input
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
        self.actions_to_move_tensor = self.get_actions_tensor()
        self.actions_indexes_all_options, self.actions_to_move_tensor_all_options = self.get_all_actions_tensor()
        self.events_loc_dict = {}
        self.events_time_dict = {}
        self.movement_cost = torch.zeros((self.batch_size, self.sim_length), device=data_input['car_loc'].device)
        self.events_cost = torch.zeros_like(self.movement_cost, device=data_input['car_loc'].device)
        n_movement_options = self.actions_to_move_tensor_all_options.shape[0]  # all movements possible
        self.anticipatory_cost_options = torch.zeros((self.batch_size, self.sim_length, n_movement_options),
                                                     device=self.events_cost.device)
        self.anticipatory_cost = torch.zeros_like(self.movement_cost, device=data_input['car_loc'].device)
        n_events = 0
        for i_b in range(self.batch_size):
            graph = data_list[i_b]
            self.car_cur_loc[i_b, ...] = graph.car_loc
            self.events_loc_dict[i_b] = graph.events_loc
            self.events_time_dict[i_b] = graph.events_time
            if graph.events_loc.shape[0] > n_events:
                n_events = graph.events_loc.shape[0]
        self.n_events = n_events
        self.cars_route = torch.zeros(self.data_input.num_graphs, self.n_cars, self.sim_length, 2)

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
                                           iter.product(torch.tensor([0, 1, 2, 3, 4]), repeat=self.n_cars)])
        return all_actions_indexes, all_actions

    def get_actions_tensor(self):
        actions_tensor = torch.tensor([[0, 0],
                                       [0, 1],
                                       [1, 0],
                                       [0, -1],
                                       [-1, 0]], device=self.car_cur_loc.device)
        return actions_tensor

    def update_state(self, actions):
        """
        this function updates the state after a set of actions is chosen for all cars
        :param actions: torch tensor of size [batch_size, n_cars] where each value represents the action chosen
        :return:
        """
        car_cur_loc = self.car_cur_loc.clone()
        print("updating state, t:" + str(self.time))
        s_time = time.time()
        for i_b in range(self.batch_size):
            s_time_b = time.time()
            # create opened events information for anticipatory calculation
            opened_events_pos = []
            opened_events_start_time = []
            opened_events_end_time = []
            # move each car to the new location -
            for i_c in range(self.n_cars):
                car_cur_loc = self.update_car_state(i_b, i_c, car_cur_loc, actions)
            events_loc_ = self.events_loc_dict[i_b]
            n_events_in_batch = events_loc_.shape[0]
            distance_matrix = torch.cdist(car_cur_loc[i_b, ...].type(torch.float), events_loc_.type(torch.float), p=1)
            for i_e in range(n_events_in_batch):
                # find row of relevant event in graph
                loc_row = events_loc_[i_e, 0] * self.dim + events_loc_[i_e, 1]
                graph_row = (loc_row + i_b * self.dim * self.dim).type(torch.int)
                # if event close time is equal to current time - event should be canceled
                should_cancel = (self.events_time_dict[i_b][i_e, 1] == self.time)
                if should_cancel:
                    self.cancel_event(i_b, i_e, graph_row)
                # check if event is opened - (not canceled, not answered and time is within time window)
                is_event_opened = (not self.events_status['canceled'][i_b, i_e]) and \
                                  (not self.events_status['answered'][i_b, i_e]) and \
                                  (self.events_time_dict[i_b][i_e, 0] <= self.time) and \
                                  (self.time <= self.events_time_dict[i_b][i_e, 1])
                if is_event_opened:
                    # in this case the event is opened , so we need to see if a car reached it's location
                    is_answered = torch.where(distance_matrix[:, i_e] <= 0.1)[0]
                    if is_answered.size()[0] > 0:
                        car_index = is_answered[0]  # this is the car chosen to answer this specific event
                        distance_matrix[car_index, :] = 9999  # makes sure you dont use the same car for other events
                        self.close_event(i_b, i_e, graph_row)
                    else:
                        open_cost = self.data_input['open_cost'][i_b]
                        self.events_cost[i_b, self.time] = self.events_cost[i_b, self.time] + open_cost
                        opened_events_pos.append(self.events_loc_dict[i_b][i_e, ...].detach().numpy())
                        opened_events_start_time.append(self.events_time_dict[i_b][i_e, 0].detach().numpy())
                        opened_events_end_time.append(self.events_time_dict[i_b][i_e, 1].detach().numpy())
                # if event open time is equal to current time , open event
                # starting time is the following time step
                should_open = (self.events_time_dict[i_b][i_e, 0] == self.time+1)
                # add 1 to graph if event is opened now for the 1st time -
                if should_open:
                    self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] + 1
            # calculate anticipatory cost for batch i_b after moving cars (known events and future events)
            cars_loc = car_cur_loc[i_b, ...].clone().detach().numpy()
            # create stochastic events (use the same events for all anticipatory calculations)
            stochastic_event_dict = createStochasticEvents(0, self.n_stochastic_runs, 0, self.sim_length,
                                                           self.stochastic_mat, self.events_open_time, self.time,
                                                           'Bm_poisson', np.array([self.dim, self.dim]),
                                                           self.dist_lambda)
            new_anticipatory_cost = self.calc_anticipatory_cost(i_b, cars_loc, opened_events_pos,
                                                                opened_events_start_time, opened_events_end_time,
                                                                stochastic_event_dict)
            self.anticipatory_cost[i_b, self.time] = new_anticipatory_cost
            # calc all other costs optional -
            actions_chosen = self.actions_to_move_tensor_all_options[actions[i_b]]
            for i_a, optional_actions in enumerate(self.actions_to_move_tensor_all_options):
                if torch.all(torch.eq(actions_chosen, self.actions_to_move_tensor_all_options[i_a])):
                    self.anticipatory_cost_options[i_b, self.time, i_a] = self.anticipatory_cost[i_b, self.time]
                else:
                    # add optional action to car cur location before current action chosen
                    car_loc_options = self.car_cur_loc[i_b, ...].clone() + optional_actions
                    ant_new = self.calc_anticipatory_cost(i_b, car_loc_options, opened_events_pos,
                                                          opened_events_start_time, opened_events_end_time,
                                                          stochastic_event_dict)
                    self.anticipatory_cost_options[i_b, self.time, i_a] = ant_new
            e_time_b = time.time()
            # print("run time for batch:"+str(i_b)+", is:"+str(e_time_b -s_time_b))
        # update current car location to new locations
        self.car_cur_loc = car_cur_loc
        self.time = self.time + 1  # update simulation time
        e_time = time.time()
        print("time to update state is:"+str(e_time-s_time))
        return

    def update_car_state(self, i_b, i_c, car_cur_loc, actions):
        cur_loc_temp = car_cur_loc[i_b, i_c, ...].clone()
        delta_movement = self.get_action_from_index(i_c, actions[i_b])
        new_loc_temp = cur_loc_temp.clone() + delta_movement
        can_move_car = (0 <= new_loc_temp[0] <= self.dim - 1) and \
                       (0 <= new_loc_temp[1] <= self.dim - 1)
        if can_move_car:  # TODO update to masking instead of choosing same location
            self.cars_route[i_b, i_c, self.time, ...] = new_loc_temp.clone()
            # subtract 1 from graph where the car came from in feature tensor
            loc_row = (cur_loc_temp[0] * self.dim + cur_loc_temp[1]).int()
            graph_row = loc_row + i_b * self.dim * self.dim
            self.data_input.x[graph_row, 2] = self.data_input.x[graph_row, 2] - 1
            assert (self.data_input.x[loc_row + i_b * self.dim * self.dim, 2] >= 0),\
                "tried to subtract 1 from car feature even though there was no car there"
            # add this cars movement cost to total movement cost
            new_cost = self.data_input['movement_cost'][i_b] * torch.sum(delta_movement)
            self.movement_cost[i_b, self.time] = self.movement_cost[i_b, self.time] + new_cost
            car_cur_loc[i_b, i_c, ...] = new_loc_temp
            # add 1 to graph where the new car location is in feature tensor
            loc_row = (new_loc_temp[0] * self.dim + new_loc_temp[1]).int()
            graph_row = loc_row + i_b * self.dim * self.dim
            self.data_input.x[graph_row, 2] = self.data_input.x[graph_row, 2] + 1
        else:
            self.cars_route[i_b, i_c, self.time, ...] = cur_loc_temp.clone()
        return car_cur_loc

    def cancel_event(self, i_b, i_e, graph_row):
        self.events_status['canceled'][i_b, i_e] = True
        self.events_cost[i_b, self.time] = self.events_cost[i_b, self.time] + self.data_input['cancel_cost'][i_b]
        self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] - 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to cancel event from row with no events!! event num:"+str(i_e)+", in batch:"+str(i_b)

    def close_event(self, i_b, i_e, graph_row):
        self.events_status['answered'][i_b, i_e] = True
        # subtract reward for closing event from total cost
        self.events_cost[i_b, self.time] = self.events_cost[i_b, self.time] - self.data_input['close_reward'][i_b]
        # close event in graph - subtract one from relevant node
        self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] - 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to close event from row with no events!!"
        self.events_answer_time[i_b, i_e] = self.time
        if self.time > 0:
            if self.n_events_closed[i_b, self.time] == 0:  # no events closed at this time
                self.n_events_closed[i_b, self.time] = self.n_events_closed[i_b, self.time-1]+1
            else:  # already closed events at this time
                self.n_events_closed[i_b, self.time] = self.n_events_closed[i_b, self.time] + 1
        else:  # time zero
            self.n_events_closed[i_b, self.time] = self.n_events_closed[i_b, self.time] + 1

    def get_action_from_index(self, i_c, i_a):
        car_movement = self.actions_to_move_tensor_all_options[i_a, i_c, :]
        return car_movement

    def get_cost(self):
        """
        this function combines all costs in problem and returns the total cost of updating state
        it assumes a specific action is chosen
        :return: torch tensor of size [batch_size, time]
        """
        cost = self.movement_cost + self.events_cost + self.anticipatory_cost
        return cost

    def get_optional_costs(self):
        """
        this function returns costs of all possible actions assuming the update state already occurred
        it returns the cost for all times
        :return: tensor of size [batch_size, sim_length, possible_actions_size]
        """
        events_cost = self.events_cost.clone()  # batch_size, sim_time_length
        n_actions = self.actions_indexes_all_options.shape[0]
        exp_events_cost = events_cost[:, :, None].expand((self.batch_size, self.sim_length, n_actions))
        movement_cost = self.actions_to_move_tensor_all_options.sum((1, 2)).clone()
        exp_movement_cost = movement_cost[None, None, :].expand((self.batch_size, self.sim_length, n_actions))
        exp_anticipatory_cost = self.anticipatory_cost_options.clone()
        optional_costs = exp_movement_cost + exp_events_cost + exp_anticipatory_cost
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
            cars_pos = self.car_cur_loc[i_b, ...].clone().detach().numpy() + actions
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
                                                         self.data_input['close_reward'][i_b].detach().numpy(),
                                                         self.data_input['cancel_cost'][i_b].detach().numpy(),
                                                         self.data_input['open_cost'][i_b].detach().numpy(), 0)
                    else:
                        m, obj = runMaxFlowOpt(0, cars_pos, current_events_pos,
                                               current_event_start_time + self.events_open_time,
                                               self.data_input['close_reward'][i_b].detach().numpy(),
                                               self.data_input['cancel_cost'][i_b].detach().numpy(),
                                               self.data_input['open_cost'][i_b].detach().numpy())
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
                                                     self.data_input['close_reward'][i_b].detach().numpy(),
                                                     self.data_input['cancel_cost'][i_b].detach().numpy(),
                                                     self.data_input['open_cost'][i_b].detach().numpy(), 0)
                else:
                    m, obj = runMaxFlowOpt(0, cars_pos, current_events_pos,
                                           current_event_start_time + self.events_open_time,
                                           self.data_input['close_reward'][i_b].detach().numpy(),
                                           self.data_input['cancel_cost'][i_b].detach().numpy(),
                                           self.data_input['open_cost'][i_b].detach().numpy())
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
        return expected_cost

    def all_finished(self):
        if self.sim_length > self.time:
            return False
        else:
            return True

