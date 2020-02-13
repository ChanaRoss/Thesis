import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn.data_parallel import DataParallel
import torch.nn as nn
import torch.nn.functional as F
from problems.state_anticipatory import AnticipatoryState


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AnticipatoryModel(torch.nn.Module):
    def __init__(self, num_features, num_nodes, embedding_dim, dp, stochastic_input_dict, sim_input_dict):
        super(AnticipatoryModel, self).__init__()
        self.dropout = dp
        self.decode_type = None
        self.sim_input_dict = sim_input_dict
        self.stochastic_input_dict = stochastic_input_dict
        self.n_cars = sim_input_dict['n_cars']
        self.embedding = nn.Sequential(nn.Linear(num_features, embedding_dim),
                                       nn.ReLU(),
                                       nn.Linear(embedding_dim, embedding_dim))
        self.encoder = GATConv(embedding_dim, 16, heads=8, dropout=self.dropout, bias=True)
        self.decoder = GATConv(embedding_dim, self.n_cars, heads=1, dropout=self.dropout, bias=True)
        self.proj_choose_out = nn.Sequential(
            # input to linear layer is num_nodes in each graph in batch * n_cars
            # therefore need to divide num_nodes in batch by num_graphs (batch_size)
            nn.Linear(int(num_nodes*num_nodes*self.n_cars), embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.n_cars*5))

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def forward(self, data_input):
        state = AnticipatoryState(data_input, self.stochastic_input_dict, self.sim_input_dict)
        all_actions = []
        all_logits = []
        while not state.all_finished():
            x_temp = state.data_input.x.clone()
            batch_size = state.data_input.num_graphs
            x = self.embedding(x_temp)
            x = self.encoder(x, state.data_input.edge_index)
            # run model decoder and find next action for each car.
            logit_per_car = self.decoder(x, state.data_input.edge_index)
            logit_ff = self.proj_choose_out(logit_per_car.view(batch_size, -1))
            logit_ff = logit_ff.clone().view(batch_size, self.n_cars, -1)  # change dimensions to [n_batchs, n_cars, 5]
            logit_ff = F.log_softmax(logit_ff, dim=2)  # soft max on action choice dimension
            # get action for each car
            # actions and logits_selected are of size [batch_size, n_cars]
            actions, logits_selected = self.get_actions(logit_ff)
            # update state to new locations -
            state.update_state(actions)
            # add chosen actions and logit to list for output and final calculation
            all_actions.append(actions)
            all_logits.append(logits_selected)
        cost = state.get_cost()
        all_logits = torch.stack(all_logits, 1)
        all_actions = torch.stack(all_actions, 1)
        ll = self.calc_log_likelihood(all_logits)
        # all_actions output is : [n_time_steps, n_batches, n_cars, 5]
        # all_logits output is : [n_time_steps, n_batches, n_cars, 5]
        # cost output is: [n_batches, 1]
        return all_actions, ll, cost, state

    def calc_log_likelihood(self, all_logits):
        batch_size = all_logits.shape[0]
        ll = all_logits.view(batch_size, -1).sum(1)
        return ll

    def get_actions(self, logit_ff):
        probs = logit_ff.exp()
        selected_actions = torch.stack([probs[:, i, :].multinomial(1).squeeze(1) for i in range(self.n_cars)]).permute(1, 0)
        logit_selected = logit_ff.gather(2, selected_actions.unsqueeze(-1).type(torch.long)).squeeze(-1)
        return selected_actions, logit_selected
