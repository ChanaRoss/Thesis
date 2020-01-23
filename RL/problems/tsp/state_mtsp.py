import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from utils.functions import calc_distance
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
    mask: torch.Tensor  # this is the mask used by network for knowing which indexes are still feasible
    can_repeat: torch.Tensor  # this is needed to see if a car can repeat its previous choice. makes sure there is
    # no infinite loop (once network chooses to repeat node once, that is the choice for future times for this car +
    # last car can't choose to repeat steps)
    finished_route: torch.Tensor  # this is needed in order to ensure that cars that finished their route always choose the previous node

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
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
    def initialize(input, allow_repeated_choices, visited_dtype=torch.bool):
        car_loc = input['car_loc'].clone()   # batch_size, n_cars, 2
        loc = input['loc'].clone()
        n_cars = input['n_cars'][0]
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(n_cars, batch_size, 1, dtype=torch.long, device=loc.device)
        index_to_choices = create_index_to_choices(n_cars, n_loc, loc.device)
        index_size = index_to_choices.shape[0]
        mask = torch.zeros([batch_size, 1, index_size], dtype=torch.bool, device=loc.device)
        can_repeat = torch.ones([n_cars, batch_size], device=loc.device)
        finished_route = torch.zeros_like(can_repeat, device=loc.device)
        visited_ = (  # Visited as mask is easier to understand, as long more memory efficient
            torch.zeros(
                batch_size, 1, n_loc+n_cars,
                dtype=torch.bool, device=loc.device
            )
            if visited_dtype == torch.bool
            else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
        )

        # mark first node as visited since it is now the depot and where all cars start
        for i_c in range(n_cars):
            prev_a_ = torch.ones(batch_size, 1, dtype=torch.long, device=loc.device)*i_c
            if visited_.dtype == torch.bool:
                # Add one dimension since we write a single value
                # add's 1 to wherever we visit now, this creates a vector of 1's wherever we have been already
                visited_ = visited_.scatter(-1, prev_a_[:, :, None], 1)
            else:
                visited_ = mask_long_scatter(visited_, prev_a_)
            prev_a[i_c, ...] = prev_a_
        concat_loc = torch.cat((car_loc, loc), -2)
        return StateMTSP(
            loc=concat_loc,
            dist=calc_distance(concat_loc),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a.clone(),
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=loc.device),   # batch_size
            cur_coord=car_loc.permute(1, 0, 2),     # n_cars, batch_size, 2
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            n_cars=n_cars,
            index_to_choices=index_to_choices,
            allow_repeated_choices=allow_repeated_choices,
            can_repeat=can_repeat,
            finished_route=finished_route,
            mask=mask
        )

    def get_final_cost(self):
        assert self.all_finished()
        # assert self.visited_.
        tot_cost = self.lengths
        n_cars = self.n_cars
        return tot_cost

    def update(self, selected, car_id, step):
        # selected shape is: [batch_size, graph_size]
        # Update the state
        batch_size = selected.shape[0]
        finished_route = self.finished_route
        prev_a = self.prev_a.clone()
        can_repeat = self.can_repeat
        prev_a[car_id, ...] = selected[:, None].view(batch_size, -1).type(torch.LongTensor)
        # Update should only be called with just 1 parallel step,
        # in which case we can check this way if we should update
        first_a = self.first_a
        cur_coord = self.cur_coord
        if cur_coord is None:
            cur_coord = torch.zeros(self.n_cars, batch_size, 2, device=prev_a.device)
        visited_ = self.visited_
        lengths = self.lengths
        prev_a_ = prev_a[car_id, ...]
        cur_coord_ = self.loc[self.ids, prev_a_].view(batch_size, -1)
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = lengths + (cur_coord_ - self.cur_coord[car_id, ...]).norm(p=1, dim=-1).view(-1, 1)  #  (batch_dim, 1)
        if self.visited_.dtype == torch.bool:
            # Add one dimension since we write a single value
            # add's 1 to wherever we visit now, this creates a vector of 1's wherever we have been already
            visited_ = visited_.scatter(-1, prev_a_[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a_)
        if self.allow_repeated_choices:
            new_mask = self._update_mask(selected)
        else:
            new_mask = self.mask
        cur_coord[car_id, ...] = cur_coord_

        for i in range(batch_size):
            # if car repeates the same location we assume it finished its route and from now on should always
            # choose that location
            if self.prev_a[car_id, i] == selected[i]:
                finished_route[car_id, i] = 1
            # the route was finished since all nodes are already taken by other cars
            if visited_[i, 0, :].sum() == self.loc.shape[1]:
                finished_route[:, i] = 1
            # check if all cars finished their route than the last one can't choose the same location and must go to all
            # other locations
            if finished_route[:, i].sum() == self.n_cars - 1:
                can_repeat[:, i] = 0

        return self._replace(first_a=first_a, prev_a=prev_a,
                             cur_coord=cur_coord, i=torch.tensor(step),
                             visited_=visited_, lengths=lengths, mask=new_mask,
                             finished_route=finished_route, can_repeat=can_repeat)

    def all_finished(self):
        # all locations have been visited (dont need n steps since now there are multiple cars)
        return self.visited.all()

    def get_current_node(self):
        return self.prev_a

    def get_options_mask(self):
        # n_choices = self.index_to_choices.shape[0]
        # batch_size = self.mask.shape[0]
        # if self.allow_repeated_choices:
        #     mask = self.mask
        #     for i_c in range(self.n_cars):
        #         prev_a_ = self.prev_a[i_c, ...]
        #         expanded_prev_a = prev_a_.expand([n_choices, 2, batch_size])
        #         mask_addition = (expanded_prev_a == self.index_to_choices[..., None]).sum(1).permute([1, 0])
        #         mask_addition = (mask_addition == 0).byte()
        #         mask = self.mask + mask_addition
        # else:
        mask = self.mask
        return mask

    def get_nodes_mask(self, car_id):
        mask = self.visited.clone()
        if self.allow_repeated_choices:
            batch_size = self.prev_a.shape[1]
            for i in range(batch_size):
                # has repeated location already, therefore only node possible is the one that was last visited
                if self.finished_route[car_id, i]:
                    mask[i, ...] = torch.ones_like(self.visited[i, ...])
                    mask[i, 0, self.prev_a[car_id, i]] = 0
                # has not repeated locations therefore all nodes in graph are possible including the previous node
                elif self.can_repeat[car_id, i]:
                    prev_a_ = self.prev_a[car_id, i, ...]
                    mask[i, ...] = mask[i, ...].scatter(-1, prev_a_[:, None], 0)
                # last car therefore can't repeat previous location, needs to pick up all the other nodes
                else:
                    mask[i, ...] = mask[i, ...]

            # # allow cars that can repeat to stay in same location:
            # prev_a_ = self.prev_a[car_id, self.can_repeat, ...]
            # # update batches where repetition is allowed to mask = 0 (network can re-choose that node)
            # mask[self.can_repeat, ...]    = mask[self.can_repeat, ...].scatter(-1, prev_a_[:, :, None], 0)
            # # force cars that finished their route to only stay at current location
            # prev_a_ = self.prev_a[car_id, self.finished_route, ...]
            # mask[self.finished_route, ...] = mask[self.finished_route, ...].scatter(-1, prev_a_[:, :, None], 0)
        return mask

    def _update_mask(self, selected):
        batch_size = selected.shape[0]
        n_cars    = self.n_cars.item()
        n_choices = self.index_to_choices.shape[0]
        expanded_selected = selected.expand([n_choices, n_cars, batch_size])
        # create addition to mask by checking which tuples include a location that is selected
        # mask_addition final size is : [batch_size, n_options]
        mask_addition = (expanded_selected == self.index_to_choices[..., None]).sum(1).permute([1, 0])
        # add new mask to previous masked nodes (might be larger than 1 therefore need to change back to ones)
        new_mask = torch.zeros_like(self.mask)
        new_mask[:, 0, :] = ((self.mask[:, 0, :] + mask_addition) > 0).byte()
        return new_mask

    def get_selections_from_index(self, selected_index):
        batch_size = selected_index.shape[0]
        n_options = self.index_to_choices.shape[0]
        index_to_choices_expanded = self.index_to_choices.expand([batch_size, n_options, 2])
        selected_index_expanded = selected_index[:, None].expand([batch_size, 2]).view([batch_size, 1, 2]).type(torch.long)
        selected = index_to_choices_expanded.gather(1, selected_index_expanded).squeeze()
        # selected shape is [batch_size, n_cars]
        return selected.view(batch_size, -1).permute([1, 0])  # permuting so that its new size is [n_cars, batch_size]

    def get_locations(self, selections):
        selections_per_car = None
        return selections_per_car

    def construct_solutions(self, actions):
        return actions


def create_index_to_choices(num_cars, num_nodes, device):
    choices_tensor = torch.tensor(list(iter.permutations(range(num_nodes), num_cars.item())), device=device)
    return choices_tensor


