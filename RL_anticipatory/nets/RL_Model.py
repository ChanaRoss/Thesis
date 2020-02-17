import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn.data_parallel import DataParallel
import torch.nn as nn
import torch.nn.functional as F
from RL_anticipatory.problems.state_anticipatory import AnticipatoryState


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
            nn.Linear(embedding_dim, 5**self.n_cars))

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def forward(self, data_input):
        state = AnticipatoryState(data_input, self.stochastic_input_dict, self.sim_input_dict)
        all_actions = []
        all_logits = []
        batch_size = state.data_input.num_graphs
        logits_all_options_list = []
        while not state.all_finished():
            x_temp = state.data_input.x.clone()
            x = self.embedding(x_temp)
            x = self.encoder(x, state.data_input.edge_index)
            # run model decoder and find next action for each car.
            logit_per_car = self.decoder(x, state.data_input.edge_index)
            logit_ff = self.proj_choose_out(logit_per_car.view(batch_size, -1))
            logit_ff = logit_ff.clone().view(batch_size, -1)  # change dimensions to [n_batchs, n_options]
            logit_ff = F.log_softmax(logit_ff, dim=1)  # soft max on action choice dimension
            # get action for each car
            # actions and logits_selected are of size [batch_size]
            actions, logits_selected = self.get_actions(logit_ff)
            # update state to new locations -
            state.update_state(actions)
            # add chosen actions and logit to list for output and final calculation
            all_actions.append(actions.clone())
            all_logits.append(logits_selected.clone())
            logits_all_options_list.append(logit_ff.clone())
        costs_all_options = state.get_optional_costs()  # tensor size is: [batch_size, time, options_size]
        logits_all_options = torch.stack(logits_all_options_list, 1) # tensor is [batch_size, time, options_size]
        cost_chosen = state.get_cost()   # tensor size is: [batch_size, time]
        logits_chosen = torch.stack(all_logits, 1)
        actions_chosen = torch.stack(all_actions, 1)
        # actions_chosen output is : [n_time_steps, n_batches]
        # logits_chosen output is : [n_time_steps, n_batches]
        # cost_chosen output is: [n_batches, n_time_steps]
        # costs_all_options output is: [n_batches, n_time_steps, n_options]
        # logits_all_options output is: [n_batches, n_time_steps, n_options]
        return costs_all_options, logits_all_options, actions_chosen, logits_chosen, cost_chosen, state

    def get_actions(self, logit_ff):
        probs = logit_ff.exp()
        # this is the index of selected actions from all 5^n_cars actions possible
        selected_actions = probs.multinomial(1).squeeze(1)   # [batch_size]
        logit_selected = logit_ff.gather(1, selected_actions.unsqueeze(-1).type(torch.long)).squeeze(-1)  # [batch_size]
        return selected_actions, logit_selected

    def calc_reinforce_loss(self, costs_all_options, logits_all_options):
        reinforce_loss = torch.zeros(costs_all_options.shape[0], device=costs_all_options.device)
        for i_b in range(costs_all_options.shape[0]):
            costs_ = costs_all_options[i_b, ...]
            logits_ = logits_all_options[i_b, ...]
            reinforce_loss_ = (costs_*logits_).sum()  # reinforce loss is the [time sum (action sum(logit*cost))]
            reinforce_loss[i_b] = reinforce_loss[i_b] + reinforce_loss_
        return reinforce_loss  # tensor of size [batch_size]

