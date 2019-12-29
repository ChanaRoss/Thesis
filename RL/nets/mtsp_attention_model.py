import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from nets.graph_decoder import GraphAttentionDecoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class MultiAttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_cars=1,
                 n_nodes=20,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 allow_repeated_choices=False):
        super(MultiAttentionModel, self).__init__()

        self.allow_repeated_choices = allow_repeated_choices

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_mtsp = problem.NAME == 'mtsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.n_cars = n_cars
        self.n_nodes = n_nodes  # number of nodes in problem
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            assert problem.NAME == "tsp" or problem.NAME == "mtsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        self.decoder = nn.ModuleList()
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.ModuleList()
        self.project_fixed_context = nn.ModuleList()

        for i in range(n_cars):
            self.decoder.append(GraphAttentionDecoder(
                embedding_dim=embedding_dim,
                tanh_clipping=tanh_clipping,
                mask_inner=mask_inner,
                mask_logits=mask_logits,
                n_heads=n_heads,
                problem=problem,
                n_cars=n_cars,
                car_id=i
            ))
            self.project_node_embeddings.append(nn.Linear(embedding_dim, 3 * embedding_dim, bias=False))
            self.project_fixed_context.append(nn.Linear(embedding_dim, embedding_dim, bias=False))

        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        final_dim = n_cars*n_nodes  # this is the dimension of the output layer (soft max for each car and all nodes)
        self.project_all_cars_out = torch.nn.Sequential(
          torch.nn.Linear(final_dim, embedding_dim),
          torch.nn.ReLU(),
          torch.nn.Linear(embedding_dim, final_dim),
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)
        pi = pi.permute(0, 2, 1)
        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        _log_p = _log_p.squeeze().permute(0, 2, 1, 3)
        batch_size = _log_p.shape[1]
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(3, a.unsqueeze(-1).type(torch.long)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
        if (log_p < -1000).data.any():
            print("Logprobs should not be -inf, check sampling procedure!")
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        # change order to be: [batch_size, n_cars, tour_length] and reshape to be [batch_size, tour_length*n_cars]
        log_p_out = log_p.permute(1, 0, 2).reshape(batch_size, -1)
        # Calculate log_likelihood
        return log_p_out.sum(1)

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # MTSP
        elif self.is_mtsp:
            return self.init_embed(input['loc'])
        return self.init_embed(input)

    def _inner(self, input_data, embeddings):

        outputs = []
        sequences = []
        state = self.problem.make_state(input_data, self.allow_repeated_choices)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = []
        for i in range(self.n_cars):
            fixed.append(self._precompute(embeddings, i))

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            log_p = torch.zeros(batch_size, self.n_cars, self.n_nodes, device=input_data['loc'].device)
            selected = torch.zeros(self.n_cars, batch_size, device=state.ids.device)
            for i_c in range(self.n_cars):
                if self.shrink_size is not None:
                    unfinished = torch.nonzero(state.get_finished() == 0)
                    if len(unfinished) == 0:
                        break
                    unfinished = unfinished[:, 0]
                    # Check if we can shrink by at least shrink_size and if this leaves at least 16
                    # (otherwise batch norm will not work well and it is inefficient anyway)
                    if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                        # Filter states
                        state = state[unfinished]
                        fixed[i_c] = fixed[i_c][unfinished]
                # this is the log_p per car before masking -
                log_p_per_car = self.decoder[i_c](state, fixed[i_c], embeddings)
                log_p[:, i_c, :] = log_p_per_car
            # log_p_ff output is of size [n_car, batch_size, 1, n_nodes]
            log_p_ff = self.project_all_cars_out(log_p.view(batch_size, -1))\
                .view(batch_size, self.n_cars, 1, -1).permute(1, 0, 2, 3)
            # log_p_ff_temp = log_p_ff.clone()
            # mask = state.get_mask(0)
            # log_p_ff_temp[mask[None, ...].expand_as(log_p_ff_temp)] = -math.inf
            # log_p_ff = log_p_ff_temp.clone()
            selected, state, probs = self._select_nodes_all(selected, log_p_ff, state, i)
            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = probs, selected
                log_p = log_p_.new_zeros((self.n_cars, batch_size, self.n_nodes), *log_p_.size()[1:])
                selected = selected_.new_zeros((self.n_cars, batch_size))
                log_p[:, state.ids[:, 0], :] = log_p_
                selected[:, state.ids[:, 0]] = selected_
            # Collect output of step
            outputs.append(probs)
            sequences.append(selected)
            i += 1
        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _select_nodes_all(self, selected, log_p_ff, state, step):
        probs_out = torch.zeros_like(log_p_ff)
        for i_c in range(self.n_cars):
            mask = state.get_mask(i_c)
            log_p_ff_temp = log_p_ff.clone()
            log_p_ff_temp[mask[None, ...].expand_as(log_p_ff_temp)] = -math.inf
            log_p_ff = log_p_ff_temp.clone()
            probs = F.log_softmax(log_p_ff, dim=3)
            probs_ = probs[i_c, ...]
            selected_car = self._select_node(probs_.exp()[:, 0, :], mask[:, 0, :])
            selected[i_c, ...] = selected_car
            state = state.update(selected_car, i_c, step)  # selected, car index, step
            probs_out[i_c, :, :, :] = probs_
        return selected, state, probs_out

    def sample_many(self, input, batch_rep=1, iter_rep=1):  # might need to updated function to work with mtsp
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _precompute(self, embeddings, n_c, num_steps=1):
        """
        calculate all precomputed vectors for Attention model
        :param embeddings: embeddings of nodes in graph
        :param n_c: car id to calculate for
        :param num_steps: number of step , for parallel calculation
        :return: Attention Fixed vectors
        """
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context[n_c](graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings[n_c](embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _select_node(self, probs, mask):
        # assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            if mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Decode greedy: infeasible action has maximum probability")
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected