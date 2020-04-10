from torch_geometric.data import Dataset, Data
# import from my code -
from Simulation.Anticipitory.with_RL.create_distributions import *


class AnticipatoryProblem:
    def __init__(self, opts):
        self.n_cars = opts['n_cars']
        self.n_events = opts['n_events']
        self.events_time_window = opts['events_time_window']
        self.end_time = opts['end_time']
        self.graph_size = opts['graph_size']
        self.cancel_cost = opts['cancel_cost']
        self.close_reward = opts['close_reward']
        self.movement_cost = opts['movement_cost']
        self.open_cost = opts['open_cost']
        self.lam = opts['lam']
        self.device = opts['device']

    def make_dataset(self, num_samples):
        dataset = AnticipatoryDataset("", self.n_cars, self.n_events, self.events_time_window, self.end_time, self.graph_size,
                                      self.cancel_cost, self.close_reward, self.movement_cost, self.open_cost,
                                      self.lam, self.device, n_samples=num_samples)
        return dataset


class AnticipatoryDataset(Dataset):
    def __init__(self, root, n_cars, n_events, events_time_window, end_time, graph_size,
                 cancel_cost, close_reward, movement_cost, open_cost, lam, device, n_samples=100,
                 transform=None, pre_transform=None):
        super(AnticipatoryDataset, self).__init__(root, transform, pre_transform)
        self.n_samples = n_samples
        self.n_cars = n_cars
        self.n_events = n_events
        self.end_time = end_time
        self.lam = lam
        self.events_time_window = events_time_window
        self.graph_size = int(graph_size)
        self.data = []
        self.device = device
        self.dtype = torch.FloatTensor
        # if self.device.type == 'cpu':
        #     self.dtype = torch.FloatTensor
        # else:
        #     self.dtype = torch.cuda.FloatTensor
        for i in range(self.n_samples):
            events_times = self.get_events_times()
            all_data = {'car_loc': self.get_car_loc(),
                        'events_time': events_times.type(self.dtype),
                        'events_loc': self.get_events_loc(events_times.shape[0]),
                        'cancel_cost': cancel_cost,      # should be positive (cost is added to total cost)
                        'close_reward': close_reward,    # should be positive (rewards are subtracted from total cost)
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
        graph.cancel_cost = torch.tensor([all_data['cancel_cost']], device='cpu')
        graph.movement_cost = torch.tensor([all_data['movement_cost']], device='cpu')
        graph.close_reward = torch.tensor([all_data['close_reward']], device='cpu')
        graph.open_cost = torch.tensor([all_data['open_cost']], device='cpu')
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
        return features_out.type(self.dtype)

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
        all_edges = torch.cat((edges1, edges2, edges3, edges4), dim=0)
        return all_edges.permute(1, 0).type(torch.long)

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
        return car_loc.type(self.dtype)

    def get_events_loc(self, n_events):
        """
        this function creates events location (random location)
        :return: torch tensor of size [n_events, 2]
        """
        # events_loc = torch.randint(0, self.graph_size, (self.n_events, 2)).type(torch.FloatTensor)
        events_loc = create_events_position(np.array([self.graph_size, self.graph_size]), n_events)
        return torch.tensor(events_loc).type(self.dtype)

    def get_events_times(self):
        """
        this function returns the events start and end time based on events time window
        :return: torch tensor of size [n_events, 2] (start, end)
        """
        # events_time = torch.randint(0, self.end_time, (self.n_events, 2)).type(torch.FloatTensor)
        # events_time[:, 1] = events_time[:, 0] + self.events_time_window
        if self.events_time_window > 10:
            events_time_tensor = torch.zeros((self.n_events, 2)).type(self.dtype)
            events_time_tensor[:, 1] = self.end_time + 1
        else:
            events_time = create_events_times(0, self.end_time, self.lam,
                                              self.events_time_window)
            events_time_tensor = torch.tensor(events_time).type(self.dtype)
        return events_time_tensor

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


class AnticipatoryTestDataset(Dataset):
    def __init__(self, root, data_input,
                 transform=None, pre_transform=None):
        super(AnticipatoryTestDataset, self).__init__(root, transform, pre_transform)
        self.n_samples = 1
        self.n_cars = data_input['n_cars']
        self.end_time = data_input['end_time']
        self.lam = data_input['lam']
        self.events_time_window = data_input['events_time_window']
        self.graph_size = int(data_input['graph_size'])
        self.data = []
        self.dtype = torch.FloatTensor  # torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for i in range(self.n_samples):
            all_data = {'car_loc': torch.tensor(data_input['car_loc']),
                        'events_time': torch.tensor(data_input['events_time']).type(self.dtype),
                        'events_loc': torch.tensor(data_input['events_loc']),
                        'cancel_cost': data_input['cancel_cost'],      # should be positive (cost is added to total cost)
                        'close_reward': data_input['close_reward'],    # should be positive (rewards are subtracted from total cost)
                        'movement_cost': data_input['movement_cost'],  # should be positive (cost is added to total cost)
                        'open_cost': data_input['open_cost']}
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
        return features_out.type(self.dtype)

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
        all_edges = torch.cat((edges1, edges2, edges3, edges4), dim=0)
        return all_edges.permute(1, 0).type(torch.long)

    def create_adjacency_matrix(self, edges):
        dim = self.graph_size
        mat_out = np.zeros((dim+1, dim+1))
        for row in edges:
            mat_out[row[0], row[1]] = 1
        return mat_out

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

