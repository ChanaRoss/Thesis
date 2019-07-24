import numpy as np
import torch
import torch.utils.data as data
import pickle
from matplotlib import pyplot as plt



class DataSet_maxFlow:
    def __init__(self, dataIn, shouldFlatten):
        self.dataIn        = dataIn
        self.numCases      = len(dataIn)
        self.shouldFlatten = shouldFlatten

    def __getitem__(self, item):
        events_loc = self.dataIn[item][0]
        car_loc = self.dataIn[item][1]
        expected_val = np.array(self.dataIn[item][2])
        if torch.cuda.is_available():
            carsTensor = torch.Tensor(car_loc).cuda()
            eventTensor  = torch.Tensor(events_loc).cuda()
            expectedValTensor = torch.Tensor(expected_val).cuda()
        else:
            carsTensor = torch.Tensor(car_loc)
            eventTensor  = torch.Tensor(events_loc)
            expectedValTensor = torch.Tensor(expected_val)
        if self.shouldFlatten:
            eventTensor = eventTensor.view(-1)  # should flatten the input of the events dimension
            carsTensor  = carsTensor.view(-1)   # should flatten the input of the events dimension
        # eventTensor is of shape: [grid id, seq, prob] or [x, y, seq, prob] (Depends if flatten is true or false
        # carsTensor: [2, numCars]
        # expectedValTensor: 1 (this is the value of the max flow run)
        return carsTensor, eventTensor, expectedValTensor

    def __len__(self):
        return self.numCases


def main():
    file_loc  = '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/'
    file_name = 'network_input_max_flow_4_30.p'
    dataIn      = pickle.load(open(file_loc + file_name, 'rb'))
    dataset_maxFlow = DataSet_maxFlow(dataIn, True)
    dataloader_maxFlow = data.DataLoader(dataset=dataset_maxFlow, batch_size=50, shuffle=False)
    # a = list(iter(dataset_uber))

    for i_batch, sample_batched in enumerate(dataloader_maxFlow):
        print(i_batch, sample_batched[0].size(),
              sample_batched[1].size(), sample_batched[2].size())
    return


if __name__ == '__main__':
    main()
    print('Done.')
