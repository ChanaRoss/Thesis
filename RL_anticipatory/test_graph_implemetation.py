import torch
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt
import networkx as nx
# imports from torch -
from torch_geometric.data import Data, DataLoader, Dataset


class UberGraph:
    def __init__(self, dim, events_loc, car_loc):
        self.dim = dim
        self.events_loc = events_loc
        self.car_loc = car_loc
        self.vertices = self.create_vertices()
        self.edges = self.create_edges()
        self.adj_mat = self.create_adjacency_matrix()

    def create_vertices(self):
        features_out = torch.zeros([self.dim*self.dim, 4])
        m = torch.ones((self.dim, self.dim))
        (row, col) = torch.where(m == 1)
        features_out[:, 0:2] = torch.stack((row[:, None], col[:, None]), dim=1).view(self.dim*self.dim, -1)
        for i in range(self.car_loc.shape[0]):
            x = self.events_loc[i, 0]
            y = self.events_loc[i, 1]
            features_out[x * self.dim + y, 2] += 1
        for i in range(self.events_loc.shape[0]):
            x = self.events_loc[i, 0]
            y = self.events_loc[i, 1]
            features_out[x * self.dim + y, 3] += 1
        # features out is [x, y, car_loc, events_loc] and is of size [dim*dim, 4]
        return features_out

    def create_edges(self):
        vertices = self.vertices
        dim = vertices.max().item()+1
        # create all edges of [x, y] and [x-1, y]
        rows = torch.where(vertices[:, 0] > 0)[0]
        edges_start = vertices[rows, 0]*dim + vertices[rows, 1]
        edge_values = torch.stack((vertices[rows, 0]-1, vertices[rows, 1]), dim=1)
        edges_end   = edge_values[:, 0]*dim + edge_values[:, 1]
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
        dim = torch.max(self.edges)
        mat_out = np.zeros((dim+1, dim+1))
        for row in self.edges:
            mat_out[row[0], row[1]] = 1
        return mat_out


class AnticipatoryDataset(Dataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None, n_samples=100):
        super(AnticipatoryDataset, self).__init__(root, transform, pre_transform)
        self.n_samples = n_samples
        self.data = data_list

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
    data_list = []
    for i in range(n_graphs):
        # create random event and car locations
        events_loc = np.random.randint(0, graph_size, [5, 2])
        car_loc = np.random.randint(0, graph_size, [3, 2])
        # create graph nodes and edges
        net_graph = UberGraph(graph_size, events_loc, car_loc)
        vertices = net_graph.vertices
        edges = net_graph.edges
        adj_mat = net_graph.adj_mat
        # save graph to list for dataloder
        data_list.append(Data(x=vertices, edge_index=edges))
        # create graph as nx graph for plotting reasons
        G = nx.from_numpy_matrix(adj_mat)
    # draw graph to see that it makes sense
    nx.draw(G, vertices[:, 0:2].numpy(),  with_labels=True)
    plt.show()
    dataset = AnticipatoryDataset(root="./dataset", data_list=data_list, n_samples=n_graphs)
    loader = DataLoader(dataset, batch_size=55)
    print(vertices.shape)
    print(edges.shape)
    return






if __name__ == '__main__':
    main()
    print("done!")

