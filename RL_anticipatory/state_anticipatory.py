import torch
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt
import networkx as nx
# imports from torch -
from torch_geometric.data import Data, DataLoader, Dataset


class AnticipatoryState:
    def __init__(self, data_input, dim):
        self.batch_size, self.n_cars, _ = data_input['car_loc'].shape
        self.time = 0  # simulation starts at time 0
        self.n_events = data_input['events_loc'].shape[1]
        self.dim = dim
        self.events_loc = data_input['events_loc'].clone()
        self.events_time = data_input['events_time'].clone()
        self.car_init_loc = data_input['car_loc'].clone()
        self.car_cur_loc = data_input['car_loc'].clone()
        self.actions_to_move_tensor = self.get_actions_tensor()
        self.graph_list = []  # this is the input to the network
        for i in range(self.batch_size):
            self.graph_list.append(self.create_init_graph())
        self.events_status = {'answered': torch.zeros((self.batch_size, self.n_events, 1), device=data_input['car_loc'].device),
                              'canceled': torch.zeros((self.batch_size, self.n_events, 1), device=data_input['car_loc'].device)}

    def get_actions_tensor(self):
        actions_tensor = torch.tensor([[0, 0],
                                       [0, 1],
                                       [1, 0],
                                       [0, -1],
                                       [-1, 0]])
        return actions_tensor

    def create_init_graph(self):
        vertices = self.create_vertices()
        edges = self.create_edges()
        graph = Data(x=vertices, edge_index=edges)
        return graph

    def update_state(self, actions):
        car_cur_loc = self.car_cur_loc.clone()
        events_status = self.events_status.clone()
        for i_b in range(self.batch_size):
            for i_c in range(self.n_cars):
                cur_loc_temp = car_cur_loc[i_b, i_c, ...]
                new_loc_temp = cur_loc_temp + self.get_action_from_index(actions[i_b, i_c])
                car_cur_loc[i_b, i_c, ...] = new_loc_temp
            events_loc_ = self.events_loc[i_b, ...]
            distance_matrix = torch.cdist(car_cur_loc[i_b, ...].type(torch.float), events_loc_.type(torch.float), p=1)
            for i_e in range(self.n_events):
                is_event_opened = (not self.events_status['canceled'][i_b, i_e]) and \
                                  (not self.events_status['answered'][i_b, i_e]) and \
                                  (self.events_time[i_b, i_e, 0] >= self.time) and \
                                  (self.events_time[i_b, i_e, 1] <= self.time)
                if is_event_opened:
                    # in this case the event is opened , so we need to see if a car reached it's location
                    is_answered = torch.where(distance_matrix[:, i_e] <= 0.1)[0]
                    if is_answered.size() > 0:
                        car_index = is_answered[0]  # this is the car chosen to answer this specific event
                        distance_matrix[car_index, :] = 9999  # makes sure you dont use the same car for other events
                        events_status['answered'][i_b, i_e] = True
                    else:
                        print("hi")
                should_cancel = (self.events_time[i_b, i_e, 1] == self.time)
                if should_cancel:
                    events_status['canceled'][i_b, i_e] = True




        new_state = []
        return new_state

    def create_vertices(self):
        """
        this function creates the vertices for the graph assuming 5 features:
        1. x location
        2. y location
        3. num cars at node
        4. num events at node
        5. num time steps until all events at node close
        :return:
        """
        # feature matrix is [x, y, n_cars, n_events, n_time steps until last event in this node is opened]
        features_out = torch.zeros([self.dim*self.dim, 5])
        m = torch.ones((self.dim, self.dim))
        (row, col) = torch.where(m == 1)
        features_out[:, 0:2] = torch.stack((row[:, None], col[:, None]), dim=1).view(self.dim*self.dim, -1)
        for i in range(self.car_loc.shape[0]):
            x = self.car_loc[i, 0]
            y = self.car_loc[i, 1]
            features_out[x * self.dim + y, 2] += 1
        for i in range(self.events_loc.shape[0]):
            x = self.events_loc[i, 0]
            y = self.events_loc[i, 1]
            delta_t = self.events_time[i, 1] - self.events_time[i, 0]
            features_out[x * self.dim + y, 3] += 1
            if features_out[x * self.dim + y, 4] < delta_t:
                features_out[x * self.dim + y, 4] = delta_t
        # features out is [x, y, car_loc, events_loc] and is of size [dim*dim, 4]
        return features_out

    def create_edges(self):
        vertices = self.vertices
        dim = self.dim
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

    def create_adjacency_matrix(self):
        dim = self.dim
        mat_out = np.zeros((dim+1, dim+1))
        for row in self.edges:
            mat_out[row[0], row[1]] = 1
        return mat_out

    def get_action_from_index(self, i):
        car_movement = self.actions_to_move_tensor[i, ...]
        return car_movement


class AnticipatoryDataset(Dataset):
    def __init__(self, root, n_cars, n_events, events_time_window, end_time, graph_size,
                 transform=None, pre_transform=None, n_samples=100):
        super(AnticipatoryDataset, self).__init__(root, transform, pre_transform)
        self.n_samples = n_samples
        self.n_cars = n_cars
        self.n_events = n_events
        self.end_time = end_time
        self.events_time_window = events_time_window
        self.graph_size = graph_size
        self.data = [
            {'car_loc': self.get_car_loc(),
             'event_loc': self.get_events_loc(),
             'event_time_window': self.get_events_times(),
             }
            for i in range(n_samples)
        ]

    def get_car_loc(self):
        """
        this function creates car locations (random location)
        :return: torch tensor of size [n_cars, 2]
        """
        car_loc = torch.randint(0, self.graph_size, (self.n_cars, 2)).type(torch.FloatTensor)
        return car_loc

    def get_events_loc(self):
        """
        this function creates events location (random location)
        :return: torch tensor of size [n_events, 2]
        """
        events_loc = torch.randint(0, self.graph_size, (self.n_events, 2)).type(torch.FloatTensor)
        return events_loc

    def get_events_times(self):
        """
        this function returns the events start and end time based on events time window
        :return: torch tensor of size [n_events, 2] (start, end)
        """
        events_time = torch.randint(0, self.end_time, (self.n_events, 2)).type(torch.FloatTensor)
        events_time[:, 1] = events_time[:, 0] + self.events_time_window
        return events_time

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
    graph_size = 10
    n_graphs = 100
    end_time = 24
    events_time_window = 5
    n_cars = 3
    n_events = 10
    data_list = []
    # for i in range(n_graphs):
    #     # create random event and car locations
    #     events_loc = np.random.randint(0, graph_size, [n_events, 2])
    #     car_loc = np.random.randint(0, graph_size, [n_cars, 2])
    #     # create graph nodes and edges
    #     net_graph = UberGraph(graph_size, events_loc, car_loc)
    #     vertices = net_graph.vertices
    #     edges = net_graph.edges
    #     adj_mat = net_graph.adj_mat
    #     # save graph to list for dataloder
    #     data_list.append(Data(x=vertices, edge_index=edges))
    #     # create graph as nx graph for plotting reasons
    #     G = nx.from_numpy_matrix(adj_mat)
    # # draw graph to see that it makes sense
    # nx.draw(G, vertices[:, 0:2].numpy(),  with_labels=True)
    # plt.show()
    dataset = AnticipatoryDataset("/data", n_cars, n_events, events_time_window, end_time, graph_size, None, None, n_graphs)
    loader = DataLoader(dataset, batch_size=55)
    print("hi")
    return






if __name__ == '__main__':
    main()
    print("done!")

