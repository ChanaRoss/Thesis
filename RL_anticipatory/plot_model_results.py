import pickle
import json
import os

from RL_anticipatory.nets.RL_Model import AnticipatoryModel
from RL_anticipatory.problems import problem_anticipatory

def main():
    path = '/Users/chanaross/dev/Thesis/RL_anticipatory/outputs/anticipatory_rl_10/anticipatory_rl_20200213T203129/'
    with open(os.join(path, 'args.json'), 'r') as f:
        args = json.load(f)
    with open(os.join(path, 'sim_input.json'), 'r') as f:
        sim_input_dict = json.load(f)
    stochastic_input_dict = pickle.load(open(path, 'rb'))

    model = AnticipatoryModel(5, args['graph_size'], args['embedding_dim'],
                              args['dp'], stochastic_input_dict, sim_input_dict)
    problem = problem_anticipatory.AnticipatoryProblem(args)

    dataset = problem.make_dataset(1)
    all_actions, log_likelihood, cost, state = model(x)



    return


if __name__ == "__main__":
    main()
    print("done!")