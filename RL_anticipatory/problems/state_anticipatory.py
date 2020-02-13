import time
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
        self.car_cur_loc = torch.zeros(self.batch_size, self.n_cars, 2)
        self.events_loc_dict = {}
        self.events_time_dict = {}
        self.movement_cost = torch.zeros(self.batch_size, device=data_input['car_loc'].device)
        self.events_cost = torch.zeros_like(self.movement_cost, device=data_input['car_loc'].device)
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
        self.actions_to_move_tensor = self.get_actions_tensor()
        self.events_answer_time = torch.ones((self.batch_size, self.n_events), device=data_input['car_loc'].device)*999
        self.events_status = {'answered': torch.zeros((self.batch_size, self.n_events, 1),
                                                      device=data_input['car_loc'].device, dtype=torch.bool),
                              'canceled': torch.zeros((self.batch_size, self.n_events, 1),
                                                      device=data_input['car_loc'].device, dtype=torch.bool)}

    def get_actions_tensor(self):
        actions_tensor = torch.tensor([[0, 0],
                                       [0, 1],
                                       [1, 0],
                                       [0, -1],
                                       [-1, 0]])
        return actions_tensor

    def update_state(self, actions):
        """
        this function updates the state after a set of actions is chosen for all cars
        :param actions: torch tensor of size [batch_size, n_cars] where each value represents the action chosen
        :return:
        """
        car_cur_loc = self.car_cur_loc.clone()
        a_time = 0
        for i_b in range(self.batch_size):
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
                                  (self.events_time_dict[i_b][i_e, 0] >= self.time) and \
                                  (self.events_time_dict[i_b][i_e, 1] <= self.time)
                if is_event_opened:
                    # in this case the event is opened , so we need to see if a car reached it's location
                    is_answered = torch.where(distance_matrix[:, i_e] <= 0.1)[0]
                    if is_answered.size() > 0:
                        car_index = is_answered[0]  # this is the car chosen to answer this specific event
                        distance_matrix[car_index, :] = 9999  # makes sure you dont use the same car for other events
                        self.close_event(i_b, i_e, graph_row)
                    else:
                        self.events_cost[i_b] = self.events_cost[i_b] + self.data_input['open_cost'][i_b]
                        opened_events_pos.append(self.events_loc_dict[i_b][i_e, ...])
                        opened_events_start_time.append(self.events_time_dict[i_b][i_e, 0])
                        opened_events_end_time.append(self.events_time_dict[i_b][i_e, 1])
                # if event open time is equal to current time , open event
                should_open = (self.events_time_dict[i_b][i_e, 0] == self.time)
                # add 1 to graph if event is opened now for the 1st time -
                if should_open:
                    self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] + 1
            # calculate anticipatory cost for batch i_b after moving cars (known events and future events)
            s_time = time.time()
            new_anticipatory_cost = self.calc_anticipatory_cost(i_b, opened_events_pos,
                                                                       opened_events_start_time, opened_events_end_time)
            self.anticipatory_cost[i_b] = self.anticipatory_cost[i_b] + new_anticipatory_cost
            e_time  = time.time()
            a_time += e_time-s_time
        # print("t: "+str(self.time)+", time to run anticipatory :"+str(a_time))
        # update current car location to new locations
        self.car_cur_loc = car_cur_loc
        self.time = self.time + 1  # update simulation time
        return

    def update_car_state(self, i_b, i_c, car_cur_loc, actions):
        cur_loc_temp = car_cur_loc[i_b, i_c, ...].clone()
        delta_movement = self.get_action_from_index(actions[i_b, i_c])
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
            self.movement_cost[i_b] = self.movement_cost[i_b] + \
                                      self.data_input['movement_cost'][i_b] * torch.sum(delta_movement)
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
        self.events_cost[i_b] = self.events_cost[i_b] - self.data_input['cancel_cost'][i_b]
        self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] - 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to cancel event from row with no events!!"

    def close_event(self, i_b, i_e, graph_row):
        self.events_status['answered'][i_b, i_e] = True
        # subtract reward for closing event from total cost
        self.events_cost[i_b] = self.events_cost[i_b] - self.data_input['close_reward'][i_b]
        # close event in graph - subtract one from relevant node
        self.data_input.x[graph_row, 3] = self.data_input.x[graph_row, 3] - 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to close event from row with no events!!"
        self.events_answer_time[i_b, i_e] = self.time.clone()

    def get_action_from_index(self, i):
        car_movement = self.actions_to_move_tensor[i, ...]
        return car_movement

    def get_cost(self):
        """
        this function combines all costs in problem and returns the total cost of updating state
        :return:
        """
        cost = self.movement_cost + self.events_cost + self.anticipatory_cost
        return cost

    def calc_anticipatory_cost(self, i_b, current_events_pos_list, current_event_start_time_list, current_events_end_time_list):
        """
        this function calculates the anticipatory cost assuming cars are in known location and future events are from future matrix
        :return:
        """
        optimization_method = 'MaxFlow'  # for now assuming anticipatory is max flow only , since timeWindow is not fast
        cars_pos = self.car_cur_loc[i_b, ...].clone().detach().numpy()
        # create stochastic events
        stochastic_event_dict = createStochasticEvents(0, self.n_stochastic_runs, 0, self.sim_length,
                                                       self.stochastic_mat, self.events_open_time, self.time,
                                                       'Bm_poisson', np.array([self.dim, self.dim]), self.dist_lambda)
        stochastic_cost = np.zeros(shape=(self.n_stochastic_runs, 1))
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
        expected_cost = np.mean(stochastic_cost)
        return expected_cost

    def all_finished(self):
        if self.sim_length > self.time:
            return False
        else:
            return True

