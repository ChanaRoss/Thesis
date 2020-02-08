import torch
import numpy as np
import time
import sys
from torch_geometric.data import Data, DataLoader, Dataset
# import my own code:
sys.path.insert(0, '/Users/chanaross/dev/Thesis/Simulation/Anticipitory/with_RL/')
from create_distributions import *
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MixedIntegerOptimization/')
from offlineOptimizationProblemMaxFlow import runMaxFlowOpt, plotResults
from offlineOptimizationProblem_TimeWindow import runMaxFlowOpt as runMaxFlowOptTimeWindow
from offlineOptimizationProblem_TimeWindow import plotResults as plotResultsTimeWindow


class AnticipatoryState:
    def __init__(self, data_input, dim, stochastic_mat, n_stochastic_runs, sim_length, dist_lambda,
                 n_prediction_steps, events_open_time):
        self.data_input = data_input
        self.stochastic_mat = stochastic_mat  # of shape [x, y, t, p(n events)]
        self.n_stochastic_runs = n_stochastic_runs
        self.batch_size = data_input.num_graphs
        self.sim_length = sim_length
        self.dist_lambda = dist_lambda
        self.n_prediction_steps = n_prediction_steps
        self.events_open_time = events_open_time
        data_list = data_input.to_data_list()
        self.n_cars = data_list[0].car_loc.shape[0]
        self.time = 0  # simulation starts at time 0
        self.dim = dim
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
            if graph.events_loc.shape[0]>n_events:
                n_events = graph.events_loc.shape[0]
        self.n_events = n_events
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
                graph_row = loc_row + i_b * self.dim * self.dim
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
                        self.events_cost[i_b] += self.data_input['open_cost'][i_b]
                        opened_events_pos.append(self.events_loc_dict[i_b][i_e, ...])
                        opened_events_start_time.append(self.events_time_dict[i_b][i_e, 0])
                        opened_events_end_time.append(self.events_time_dict[i_b][i_e, 1])
                # if event open time is equal to current time , open event
                should_open = (self.events_time_dict[i_b][i_e, 0] == self.time)
                # add 1 to graph if event is opened now for the 1st time -
                if should_open:
                    self.data_input.x[graph_row, 3] += 1
            # calculate anticipatory cost for batch i_b after moving cars (known events and future events)
            s_time = time.time()
            self.anticipatory_cost[i_b] += self.calc_anticipatory_cost(i_b, opened_events_pos,
                                                                       opened_events_start_time, opened_events_end_time)
            e_time  = time.time()
            a_time += e_time-s_time
        print("time to run anticipatory :"+str(a_time))
        # update current car location to new locations
        self.car_cur_loc = car_cur_loc
        return

    def update_car_state(self, i_b, i_c, car_cur_loc, actions):
        cur_loc_temp = car_cur_loc[i_b, i_c, ...]
        # subtract 1 from graph where the car came from in feature tensor
        loc_row = (cur_loc_temp[0] * self.dim + cur_loc_temp[1]).int()
        self.data_input.x[loc_row + i_b * self.dim * self.dim, 2] -= 1
        delta_movement = self.get_action_from_index(actions[i_b, i_c])
        new_loc_temp = cur_loc_temp + delta_movement
        # add this cars movement cost to total movement cost
        self.movement_cost[i_b] += self.data_input['movement_cost'][i_b] * torch.sum(delta_movement)
        car_cur_loc[i_b, i_c, ...] = new_loc_temp
        # add 1 to graph where the new car location is in feature tensor
        loc_row = (cur_loc_temp[0] * self.dim + cur_loc_temp[1]).int()
        self.data_input.x[loc_row + i_b * self.dim * self.dim, 2] += 1
        return car_cur_loc

    def cancel_event(self, i_b, i_e, graph_row):
        self.events_status['canceled'][i_b, i_e] = True
        self.events_cost[i_b] -= self.data_input['cancel_cost'][i_b]
        self.data_input.x[graph_row, 3] -= 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to cancel event from row with no events!!"

    def close_event(self, i_b, i_e, graph_row):
        self.events_status['answered'][i_b, i_e] = True
        # subtract reward for closing event from total cost
        self.events_cost[i_b] -= self.data_input['close_reward'][i_b]
        # close event in graph - subtract one from relevant node
        self.data_input.x[graph_row, 3] -= 1
        assert (self.data_input.x[graph_row, 3] >= 0), \
            "trying to close event from row with no events!!"
        self.events_answer_time[i_b, i_e] = self.time

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
                runTime = etime - stime
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


class AnticipatoryDataset(Dataset):
    def __init__(self, root, n_cars, events_time_window, end_time, graph_size,
                 cancel_cost, close_reward, movement_cost, open_cost, lam,
                 transform=None, pre_transform=None, n_samples=100):
        super(AnticipatoryDataset, self).__init__(root, transform, pre_transform)
        self.n_samples = n_samples
        self.n_cars = n_cars
        self.end_time = end_time
        self.lam = lam
        self.events_time_window = events_time_window
        self.graph_size = int(graph_size)
        self.data = []
        for i in range(self.n_samples):
            events_times = self.get_events_times()
            all_data = {'car_loc': self.get_car_loc(),
                        'events_time': events_times,
                        'events_loc': self.get_events_loc(events_times.shape[0]),
                        'cancel_cost': cancel_cost,  # should be positive (cost is added to total cost)
                        'close_reward': close_reward,  # should be positive (rewards are subtracted from total cost)
                        'movement_cost': movement_cost,  # should be positive (cost is added to total cost)
                        'open_cost': open_cost}
            self.data.append(self.create_init_graph(all_data))

    def create_init_graph(self, all_data):
        vertices = self.create_vertices(all_data)
        edges = self.create_edges(vertices)
        graph = Data(x=vertices, edge_index=edges)
        graph.car_loc = all_data['car_loc']
        graph.events_loc = all_data['events_loc']
        graph.events_time = all_data['events_time']
        graph.cancel_cost = all_data['cancel_cost']
        graph.movement_cost = all_data['movement_cost']
        graph.close_reward = all_data['close_reward']
        graph.open_cost = all_data['open_cost']
        return graph

    def create_vertices(self, all_data):
        """
        this function creates the vertices for the graph assuming 5 features:
        1. x location
        2. y location
        3. num cars at node
        4. num events at node
        5. num time steps until all events at node close
        :return:
        """
        car_loc = all_data['car_loc']
        events_loc = all_data['events_loc']
        events_time = all_data['events_time']
        # feature matrix is [x, y, n_cars, n_events, n_time steps until last event in this node is opened]
        features_out = torch.zeros([self.graph_size*self.graph_size, 5])
        m = torch.ones((self.graph_size, self.graph_size))
        (row, col) = torch.where(m == 1)
        features_out[:, 0:2] = torch.stack((row[:, None], col[:, None]), dim=1).view(self.graph_size*self.graph_size, -1)
        for i in range(car_loc.shape[0]):
            x = car_loc[i, 0].type(torch.long)
            y = car_loc[i, 1].type(torch.long)
            features_out[x * self.graph_size + y, 2] += 1
        for i in range(events_loc.shape[0]):
            if events_time[i, 0] == 0:
                # if event start time is 0 , should add the event to graph and add the event time window to time feature
                x = events_loc[i, 0].type(torch.long)
                y = events_loc[i, 1].type(torch.long)
                delta_t = events_time[i, 1] - events_time[i, 0]
                features_out[x * self.graph_size + y, 3] += 1
                # if event is opened until later than other events, need to update time feature
                if features_out[x * self.graph_size + y, 4] < delta_t:
                    features_out[x * self.graph_size + y, 4] = delta_t
        # features out is [x, y, car_loc, events_loc, max_event_time_window] and is of size [dim*dim, 5]
        return features_out

    def create_edges(self, vertices):
        dim = self.graph_size
        # create all edges of [x, y] and [x-1, y]
        rows = torch.where(vertices[:, 0] > 0)[0]
        edges_start = vertices[rows, 0]*dim + vertices[rows, 1]
        edge_values = torch.stack((vertices[rows, 0]-1, vertices[rows, 1]), dim=1)
        edges_end = edge_values[:, 0]*dim + edge_values[:, 1]
        edges1 = torch.stack((edges_start, edges_end), dim=1)

        # create all edges of [x, y] and [x, y-1]
        rows = torch.where(vertices[:, 1] > 0)[0]
        edges_start = vertices[rows, 0] * dim + vertices[rows, 1]
        edge_values = torch.stack((vertices[rows, 0], vertices[rows, 1]-1), dim=1)
        edges_end = edge_values[:, 0] * dim + edge_values[:, 1]
        edges2 = torch.stack((edges_start, edges_end), dim=1)

        # create all edges of [x, y] and [x+1, y]
        rows = torch.where(vertices[:, 0] < dim - 1)[0]
        edges_start = vertices[rows, 0] * dim + vertices[rows, 1]
        edge_values = torch.stack((vertices[rows, 0] + 1, vertices[rows, 1]), dim=1)
        edges_end = edge_values[:, 0] * dim + edge_values[:, 1]
        edges3 = torch.stack((edges_start, edges_end), dim=1)

        # create all edges of [x, y] and [x, y+1]
        rows = torch.where(vertices[:, 1] < dim - 1)[0]
        edges_start = vertices[rows, 0] * dim + vertices[rows, 1]
        edge_values = torch.stack((vertices[rows, 0], vertices[rows, 1] + 1), dim=1)
        edges_end = edge_values[:, 0] * dim + edge_values[:, 1]
        edges4 = torch.stack((edges_start, edges_end), dim=1)

        return torch.cat((edges1, edges2, edges3, edges4), dim=0).type(torch.int)

    def create_adjacency_matrix(self, edges):
        dim = self.graph_size
        mat_out = np.zeros((dim+1, dim+1))
        for row in edges:
            mat_out[row[0], row[1]] = 1
        return mat_out

    def get_car_loc(self):
        """
        this function creates car locations (random location)
        :return: torch tensor of size [n_cars, 2]
        """
        car_loc = torch.randint(0, self.graph_size, (self.n_cars, 2)).type(torch.FloatTensor)
        return car_loc

    def get_events_loc(self, n_events):
        """
        this function creates events location (random location)
        :return: torch tensor of size [n_events, 2]
        """
        events_loc = create_events_position(np.array([self.graph_size, self.graph_size]), n_events)
        # events_loc = torch.randint(0, self.graph_size, (self.n_events, 2)).type(torch.FloatTensor)
        return torch.tensor(events_loc)

    def get_events_times(self):
        """
        this function returns the events start and end time based on events time window
        :return: torch tensor of size [n_events, 2] (start, end)
        """
        # events_time = torch.randint(0, self.end_time, (self.n_events, 2)).type(torch.FloatTensor)
        # events_time[:, 1] = events_time[:, 0] + self.events_time_window
        events_time = create_events_times(0, self.end_time, self.lam, self.events_time_window)

        return torch.tensor(events_time)

    def raw_file_names(self):
        return "should not use raw file names"

    def processed_file_names(self):
        return "should not use processed file names"

    def __len__(self):
        return len(self.data)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        data = self.data[idx]
        return data


def main():
    # problem parameters -
    graph_size = 10
    n_graphs = 100
    end_time = 24
    events_time_window = 5
    n_cars = 2
    # anticipatory parameters -
    stochastic_mat = np.zeros([10, 10, 10, 10])  # should be probability mat [x, y, t, p(n events)]
    n_stochastic_runs = 2
    n_prediction_steps = 5
    # distribution parameters -
    dist_lambda = 2/3
    cancel_cost = 10  # should be positive since all costs are added to total cost
    close_reward = 5  # should be positive since all rewards are subtracted from total cost
    movement_cost = 1  # should be positive since all costs are added to total cost
    open_cost = 1  # should be positive since all costs are added to total cost
    dataset = AnticipatoryDataset("/data", n_cars, events_time_window, end_time, graph_size,
                                  cancel_cost, close_reward, movement_cost, open_cost, dist_lambda,
                                  None, None, n_graphs)
    dataloader = DataLoader(dataset, batch_size=55)
    for data in dataloader:
        state = AnticipatoryState(data, graph_size, stochastic_mat, n_stochastic_runs,
                                  end_time, dist_lambda, n_prediction_steps, events_time_window)
        actions = torch.randint(0, 5, [55, n_cars])
        s_time = time.time()
        state.update_state(actions)
        e_time = time.time()
        print("total update time :"+str(e_time - s_time))
        print(data)
    return






if __name__ == '__main__':
    main()
    print("done!")

