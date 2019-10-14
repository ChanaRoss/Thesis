import numpy as np
import torch
import torch.utils.data as data
import pickle
import random


class MaxFlowDataset(torch.utils.data.Dataset):
    def __init__(self, dataIn, shouldFlattenCars, shouldFlattenEvents):
        self.dataIn        = dataIn
        self.numCases      = len(dataIn)
        self.shouldFlattenCars = shouldFlattenCars
        self.shouldFlattenEvents = shouldFlattenEvents

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
        if self.shouldFlattenCars:
            carsTensor  = carsTensor.view(-1)   # should flatten the input of the events dimension
        if self.shouldFlattenEvents:
            eventTensor = eventTensor.view(-1)  # should flatten the input of the events dimension
        # eventTensor is of shape: [grid id, seq, prob] or [x, y, seq, prob] (Depends if flatten is true or false
        # carsTensor: [2, numCars]
        # expectedValTensor: 1 (this is the value of the max flow run)
        return (carsTensor, eventTensor), expectedValTensor

    def __len__(self):
        return self.numCases


def partition_dict(d, percentage_d1):
    """
    splits a dictionary into two distinct dicts with
    given proportion of the data to be allocated to
    the first dict, and the remainder to the second.
    :param d: dictionary
    :param percentage_d1: float
    :return: dict(), dict()
    """
    key_list = list(d.keys())
    num_p1 = int(len(d)*percentage_d1)
    p1_keys = set(random.sample(key_list, num_p1))
    p1 = {}
    p2 = {}
    i1 = 0
    i2 = 0
    for (k, v) in d.items():
        if k in p1_keys:
            p1[i1] = v
            i1 += 1
        else:
            p2[i2] = v
            i2 += 1
    return p1, p2


def get_max_flow_dataset(file_path, flatten_cars, flatten_events, percentage_train):
    data_in = pickle.load(open(file_path, 'rb'))
    train_data, validation_data = partition_dict(data_in, percentage_train)

    train_dataset = MaxFlowDataset(train_data, shouldFlattenCars=flatten_cars, shouldFlattenEvents=flatten_events)
    valid_dataset = MaxFlowDataset(validation_data, shouldFlattenCars=flatten_cars, shouldFlattenEvents=flatten_events)
    return train_dataset, valid_dataset
