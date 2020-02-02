import torch
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt
import networkx as nx
# imports from torch -
from torch_geometric.data import Data, DataLoader


def create_vertices(dim, events_loc, car_loc):
    features_out = torch.zeros([dim*dim, 4])
    m = torch.ones((dim, dim))
    (row, col) = torch.where(m == 1)
    features_out[:, 0:2] = torch.stack((row[:, None], col[:, None]), dim=1).view(dim*dim, -1)
    for i in range(car_loc.shape[0]):
        x = events_loc[i, 0]
        y = events_loc[i, 1]
        features_out[x * dim + y, 2] += 1
    for i in range(events_loc.shape[0]):
        x = events_loc[i, 0]
        y = events_loc[i, 1]
        features_out[x * dim + y, 3] += 1
    # features out is [x, y, car_loc, events_loc] and is of size [dim*dim, 4]
    return features_out


def create_adjacency_matrix(edge_list):
    dim = torch.max(edge_list)
    mat_out = np.zeros((dim+1, dim+1))
    for row in edge_list:
        mat_out[row[0], row[1]] = 1
    return mat_out


def create_edges(vertices):
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


def MyDataSet():



    

def main():
    graph_size = 10
    n_graphs = 100
    data_list = []
    for i in range(n_graphs):
        events_loc = np.random.randint(0, graph_size, [5, 2])
        car_loc = np.random.randint(0, graph_size, [3, 2])
        vertices = create_vertices(graph_size, events_loc, car_loc)
        edges = create_edges(vertices)
        data_list.append(Data(x=vertices, edge_index=edges))
        adj_mat = create_adjacency_matrix(edges)
        G = nx.from_numpy_matrix(adj_mat)
        # nx.draw(G, vertices[:, 0:2].numpy(),  with_labels=True)
        # plt.show()
    loader = DataLoader(data_list, batch_size=55)
    print(vertices.shape)
    print(edges.shape)
    return






if __name__ == '__main__':
    main()
    print("done!")

