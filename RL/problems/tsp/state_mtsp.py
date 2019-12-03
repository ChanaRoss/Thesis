import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import itertools as iter


class StateMTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    n_cars: int  # number of cars in problem
    index_to_choices: torch.Tensor  # keeps track of choices based on index chosen by network
    allow_repeated_choices: bool

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):  # TODO make sure this works for multiple cars, now that the dimension in some is different then expected
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StateMTSP, self).__getitem__(key)

    @staticmethod
    def initialize(input, allow_repeated_choices=False, visited_dtype=torch.bool):
        loc = input['loc']
        n_cars = input['n_cars'][0]
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(n_cars, batch_size, 1, dtype=torch.long, device=loc.device)
        index_to_choices = create_index_to_choices(n_cars, n_loc)
        index_size = index_to_choices.shape[0]
        return StateMTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            n_cars=n_cars,
            index_to_choices=index_to_choices,
            allow_repeated_choices=allow_repeated_choices
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.
        tot_cost = self.lengths
        n_cars = self.n_cars
        for i in range(n_cars):
            cur_coord_ = self.cur_coord[i, ...]  # current coordinate of car i
            first_a_ = self.first_a[i, ...]  # initial location of car i
            tot_cost += (self.loc[self.ids, first_a_, :] - cur_coord_).norm(p=2, dim=-1)
        return tot_cost

    def update(self, selected, car_id, step):
        # selected shape is: [batch_size, graph_size]
        # Update the state
        batch_size = selected.shape[0]
        prev_a = self.prev_a
        prev_a[car_id, ...] = selected[:, None].view(batch_size, -1).type(torch.LongTensor)
        # Update should only be called with just 1 parallel step,
        # in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a
        cur_coord = self.cur_coord
        if cur_coord is None:
            cur_coord = torch.zeros(self.n_cars, batch_size, 2, device=prev_a.device)
        visited_ = self.visited_
        lengths = self.lengths
        prev_a_ = prev_a[car_id, ...]
        cur_coord_ = self.loc[self.ids, prev_a_].view(batch_size, -1)
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = lengths + (cur_coord_ - self.cur_coord[car_id, ...]).norm(p=2, dim=-1).view(-1, 1)  # (batch_dim, 1)
        if self.visited_.dtype == torch.bool:
            # Add one dimension since we write a single value
            # add's 1 to wherever we visit now, this creates a vector of 1's wherever we have been already
            visited_ = visited_.scatter(-1, prev_a_[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a_)
        cur_coord[car_id, ...] = cur_coord_
        return self._replace(first_a=first_a, prev_a=prev_a,
                             cur_coord=cur_coord, i=torch.tensor(step),
                             visited_=visited_, lengths=lengths)

    def all_finished(self):
        # all locations have been visited (dont need n steps since now there are multiple cars)
        return self.visited.all()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self, car_id):
        mask = self.visited
        if self.allow_repeated_choices:
            prev_a_ = self.prev_a[car_id, ...]
            mask.scatter(-1, prev_a_[:, :, None], 0)
        return mask

    def get_locations(self, selections):
        selections_per_car = None
        return selections_per_car

        # def get_nn(self, k=None):   # TODO update for multiple cars
    #     # Insert step dimension
    #     # Nodes already visited get inf so they do not make it
    #     if k is None:
    #         k = self.loc.size(-2) - self.i.item()  # Number of remaining
    #     return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    # def get_nn_current(self, k = None):
    #     assert False, "Currently not implemented, look into which neighbours to use in step 0?"
    #     # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
    #     # so it is probably better to use k = None in the first iteration
    #     if k is None:
    #         k = self.loc.size(-2)
    #     k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
    #     return (
    #         self.dist[
    #             self.ids,
    #             self.prev_a
    #         ] +
    #         self.visited.float() * 1e6
    #     ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions


def create_index_to_choices(num_cars, num_nodes):
    choices_tensor = torch.Tensor(list(iter.permutations(range(num_nodes, num_cars))))
    return choices_tensor
