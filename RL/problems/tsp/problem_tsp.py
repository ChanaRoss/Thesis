from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from problems.tsp.state_mtsp import StateMTSP
from utils.beam_search import beam_search
from utils.functions import calc_distance


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):

        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )

        return beam_search(state, beam_size, propose_expansions)


class MTSP(object):
    NAME = 'mtsp'

    @staticmethod
    def get_costs(dataset, pi):
        loc = torch.cat((dataset['car_loc'].clone(), dataset['loc'].clone()), -2)
        pi_assert = pi.permute(1, 0, 2)
        batch_size, tour_length, _ = dataset['loc'].shape
        cost = torch.zeros(batch_size, device=loc.device)
        n_cars = dataset['n_cars'][0].item()
        # Check that tours are valid, i.e. contain 0 to n -1
        # is_full_tour = True
        # for i in range(batch_size):
        #     tour_ = pi_assert[i, ...].unique()
        #     min_val = tour_.min().item()
        #     max_val = tour_.max().item()
        #     if (torch.arange(min_val, max_val+1, device=pi.device).view(-1, 1) != tour_.view(-1, 1)).all():
        #         is_full_tour = False
        #         break
        # assert is_full_tour, "Invalid tour"
        data_size = pi.shape[2]
        for i in range(n_cars):
            pi_ = pi[i, ...]
            # Gather dataset in order of tour
            d = loc.gather(1, pi_.unsqueeze(-1).expand(batch_size, data_size, 2).type(torch.long))
            dist_mat = calc_distance(d)
            # in order to get the total cost need to sum up the cost of each movement, d is already in the correct order
            # therefore if we sum the [i, i+1] value from each row i and column i+1
            cost1 = torch.stack([dist_mat[:, i, i+1] for i in range(dist_mat.shape[1]-1)], dim=1).sum(axis=1)
            # Add all tour costs together for each car
            cost2 = (loc[:, i] - d[:, 0]).norm(p=1, dim=1)  # add the cost of going from car location to the first node
            cost += cost1 + cost2
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MTSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                 coord_limit=1, n_cars=1, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class MTSPDataset(Dataset):
    def __init__(self, filename=None, coord_limit=10, size=50, num_samples=1000000,
                 offset=0, n_cars=1, distribution=None):
        super(MTSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                {
                    'loc': torch.randint(0, coord_limit, (size, 2)).type(torch.FloatTensor),
                    'depot': torch.randint(0, coord_limit, (1, 2)).type(torch.FloatTensor),
                    'car_loc': torch.randint(0, coord_limit, (n_cars, 2)).type(torch.FloatTensor),
                    'n_cars': torch.tensor([n_cars])
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]