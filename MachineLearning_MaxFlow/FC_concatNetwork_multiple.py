import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import random
from matplotlib import pyplot as plt
import time, pickle, itertools
from sklearn import metrics
from math import sqrt
import sys
if torch.cuda.is_available():
    sys.path.insert(0, '/home/schanaby@st.technion.ac.il/thesisML/ml_maxFlow/')
else:
    sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/')
from dataLoader_maxflow import DataSet_maxFlow

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
        print("turned x to gpu")
    return Variable(x)

isServerRun = torch.cuda.is_available()
if isServerRun:
    print ('Running using cuda')

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# creating optimization parameters and function
# adam    -(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0)
# SGD     -(params, lr=1e-3,momentum=0, dampening=0, weight_decay=0, nesterov=False)
# Adagrad -(params, lr=0.01, lr_decay=0, weight_decay=0)

def CreateOptimizer(netParams, ot, lr, dmp, mm, eps, wd):
    if ot == 1:
        optim = torch.optim.SGD(netParams, lr, mm, dmp, weight_decay=wd)
    elif ot == 2:
        optim = torch.optim.Adam(netParams, lr, (0.9, 0.999), eps, weight_decay=wd)
    elif ot == 3:
        optim = torch.optim.Adagrad(netParams, lr, weight_decay=wd)
    return optim


class Model(nn.Module):
    def __init__(self, fc_car1, fc_event1, fc_event2, fc_event3, fc_concat1, batch_size, input_event, input_car, num_event_layers):
        super(Model, self).__init__()
        self.input_event      = input_event  # size of input event vector
        self.input_car        = input_car    # size of input car vector
        self.batch_size       = batch_size   # batch size
        self.numEventLayers   = num_event_layers # number of times to use 2nd layer for events
        # network -
        self.fc_car1          = None
        self.fc_event1        = None
        self.fc_event2        = None
        self.fc_event3        = None
        self.fc_concat1       = None
        self.fc_out           = None

        # create normilize layer
        self.batchNorm_event1 = None
        self.batchNorm_event2 = None

        # create non-linaerity
        self.relu = nn.ReLU()

        self.p_dp = None
        self.loss           = None
        self.lossCrit       = None
        self.optimizer      = None
        self.lr             = None
        self.wd             = None
        self.lossWeights    = None
        self.maxEpochs      = None
        self.netSize        = {'fc_car1'    : fc_car1,
                               'fc_event1'  : fc_event1,
                               'fc_event2'  : fc_event2,
                               'fc_event3'  : fc_event3,
                               'fc_concat1' : fc_concat1,
                               'num_event_layers': num_event_layers}

        # output variables (loss, acc ect.)
        self.finalAcc       = 0
        self.finalLoss      = 0
        self.lossVecTrain   = []
        self.lossVecTest    = []
        self.accVecTrain    = []
        self.accVecTest     = []
        self.rmseVecTrain   = []
        self.rmseVecTest    = []

    def forward(self, x_car, x_event):
        # car network -
        outFc_car1     = self.fc_car1(x_car)
        outRel_car1    = self.relu(outFc_car1)
        outDp_car1     = F.dropout(outRel_car1, p=self.p_dp, training=False, inplace=False)
        # event network , layer 1 -
        outFc_event1 = self.fc_event1(x_event)
        outRel_event1 = self.relu(outFc_event1)
        outBatchNorm_event1 = self.batchNorm_event1(outRel_event1)
        outDp_event1    = F.dropout(outBatchNorm_event1, p=self.p_dp, training=False, inplace=False)
        # event network , layer 2 -
        input_layer2 = outDp_event1
        for i in range(self.numEventLayers):
            outFc_event2 = self.fc_event2(input_layer2)
            outRel_event2 = self.relu(outFc_event2)
            outBatchNorm_event2 = self.batchNorm_event2(outRel_event2)
            outDp_event2 = F.dropout(outBatchNorm_event2, p=self.p_dp, training=False, inplace=False)
            input_layer2 = outDp_event2
        # event network , layer 3 -
        outFc_event3 = self.fc_event3(outDp_event2)
        outRel_event3 = self.relu(outFc_event3)
        outDp_event3 = F.dropout(outRel_event3, p=self.p_dp, training=False, inplace=False)
        # combined network -
        input_concat = torch.cat((outDp_car1, outDp_event3), 1)
        outFc_cat  = self.fc_concat1(input_concat)
        outRel_cat = self.relu(outFc_cat)
        outNet     = self.fc_out(outRel_cat)
        return outNet

    def backward(self, outputs, labels):
        self.loss = self.lossCrit(outputs.view(-1), labels.view(-1))
        self.loss.backward()


    def test_spesific(self, testLoader):
        # put model in evaluate mode
        self.eval()
        testCorr = 0.0
        testTot = 0.0
        localLossTest = []
        localAccTest = []
        localRmseTest = []
        for input_cars, input_events, labels in testLoader:
            self.loss = None
            input_cars = input_cars.to(device)
            input_events = input_events.to(device)
            labels = labels.to(device)
            input_carsVar = Variable(input_cars)
            input_eventsVar = Variable(input_events)
            labVar = Variable(labels)
            # compute test result of model
            localBatchSize = labels.shape[0]
            testOut = self.forward(input_carsVar, input_eventsVar)
            self.backward(testOut, labVar)
            localLossTest.append(self.loss.item())
            if isServerRun:
                labTestNp = testOut.type(torch.cuda.LongTensor).cpu().detach().numpy()
                labelsNp = labels.cpu().detach().numpy()
                testCorr = torch.sum(testOut == labels).cpu().detach().numpy() + testCorr
            else:
                labTestNp = testOut.long().detach().numpy()
                labelsNp = labels.detach().numpy()
                testCorr = torch.sum(testOut == labels).detach().numpy() + testCorr
            testTot = labels.size(0) + testTot
            # print("labTest:"+str(labTestNp.size)+", lables:"+str(labelsNp.size))
            rmse = sqrt(metrics.mean_squared_error(labTestNp.reshape(-1), labelsNp.reshape(-1)))
            localAccTest.append(100 * testCorr / testTot)
            localRmseTest.append(rmse)
        accTest = np.average(localAccTest)
        lossTest = np.average(localLossTest)
        rmseTest = np.average(localRmseTest)
        print("test accuarcy is: {0}, rmse is: {1}, loss is: {2}".format(accTest, rmseTest, lossTest))
        return localAccTest, localLossTest, rmse

        # save network

    def saveModel(self, path):
        torch.save(self, path)


def saveFile(file, fileName):
    with open(fileName+'.pkl', 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def partition_dict(d, percentage_d1):
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

def main():
    #####################
    # Generate data
    #####################
    # data loader -
    if isServerRun:
        path = '/home/schanaby@st.technion.ac.il/thesisML/ml_maxFlow/'
    else:
        path = '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/'

    fileName = 'network_input_timeWindow_ncars_4_nperm_100.p'
    # dictionary of n runs, each run is [event mat, car mat, expected result]
    dataInput = pickle.load(open(path + fileName, 'rb'))

    flag_save_network = True

    inputEventSize      = dataInput[0][0].size
    inputCarSize        = dataInput[0][1].size
    # define hyper parameters -
    num_layers_events = 6
    fc_car1Vec      = [128, 256, 12]
    fc_event1Vec    = [64, 8, 256]
    fc_event3Vec    = [128, 16]
    fc_concat1Vec   = [8, 48, 256]
    batch_sizeVec   = [10, 40]
    num_epochs      = 50
    p_dp            = 0  # percentage dropout
    # optimizer parameters -
    lrVec   = [0.001, 0.01, 0.1]
    otVec   = [1, 2]  # [1, 2]
    dmp     = 0
    mm      = 0.9
    eps     = 1e-08
    wdVec   = [2e-3]

    data_train, data_test = partition_dict(dataInput, 0.8)  # 0.8 of inputs will be for train dataset
    dataset_train = DataSet_maxFlow(data_train, shouldFlatten=True)
    dataset_test  = DataSet_maxFlow(data_test, shouldFlatten=True)
    # create case vectors
    networksDict = {}
    itr = itertools.product(fc_car1Vec, fc_event1Vec, fc_event3Vec, fc_concat1Vec, batch_sizeVec, lrVec, otVec, wdVec)
    for i in itr:
        networkStr = 'fcc1_{0}_fce1_{1}_fce3_{2}_fccat_{3}_bs_{4}_lr_{5}_ot_{6}_wd_{7}'.\
            format(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7])
        networksDict[networkStr] = {'fcc1': i[0], 'fce1': i[1], 'fce3': i[2], 'fccat': i[3], 'bs': i[4],
                                    'lr': i[5], 'ot' : i[6], 'wd': i[7]}

    for netConfig in networksDict:
        # output file
        outFile = open('FC_networksOutput.csv', 'w')
        outFile.write('Name;finalAcc;finalLoss;trainTime;numWeights;NumEpochs\n')


        print('Net Parameters: ' + netConfig)

        # create network based on input parameter's -
        fc_car1     = networksDict[netConfig]['fcc1']
        fc_event1   = networksDict[netConfig]['fce1']
        fc_event2   = networksDict[netConfig]['fce1']
        fc_event3   = networksDict[netConfig]['fce3']
        fc_concat   = networksDict[netConfig]['fccat']

        batch_size      = networksDict[netConfig]['bs']
        lr              = networksDict[netConfig]['lr']
        ot              = networksDict[netConfig]['ot']
        wd              = networksDict[netConfig]['wd']

        my_net = Model(fc_car1, fc_event1, fc_event2, fc_event3, fc_concat, batch_size, inputEventSize, inputCarSize, num_layers_events)
        my_net.fc_car1 = nn.Linear(inputCarSize, fc_car1)
        my_net.fc_event1 = nn.Linear(inputEventSize, fc_event1)
        my_net.fc_event2 = nn.Linear(fc_event1, fc_event2)
        my_net.fc_event3 = nn.Linear(fc_event2, fc_event3)
        my_net.fc_concat1 = nn.Linear(fc_event3 + fc_car1, fc_concat)
        my_net.fc_out     = nn.Linear(fc_concat, 1)

        my_net.batchNorm_event1 = nn.BatchNorm1d(fc_event1, eps=1e-5, momentum=0.1, affine=True)
        my_net.batchNorm_event2 = nn.BatchNorm1d(fc_event2, eps=1e-5, momentum=0.1, affine=True)

        my_net.lr = lr
        my_net.p_dp = p_dp
        my_net.wd = wd
        my_net.optimizer = CreateOptimizer(my_net.parameters(), ot, lr, dmp, mm, eps, wd)
        my_net.lossCrit = nn.MSELoss()
        my_net.maxEpochs = num_epochs

        my_net.to(device)
        print("model device is:")
        print(next(my_net.parameters()).device)
        numWeights = sum(param.numel() for param in my_net.parameters())
        print('number of parameters: ', numWeights)

        # creating data loader
        dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_test  = data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
        netOutDict = {}
        labelsOutDict = {}
        for numEpoch in range(num_epochs):
            my_net.loss = None
            # for each epoch, calculate loss for each batch -
            my_net.train()
            localLoss = [4]
            accTrain = [0]
            rmseTrain = [1]
            trainCorr = 0.0
            trainTot = 0.0
            if (1+numEpoch)%40 == 0:
                if my_net.optimizer.param_groups[0]['lr'] > 0.001:
                    my_net.optimizer.param_groups[0]['lr'] = my_net.optimizer.param_groups[0]['lr']/2
                else:
                    my_net.optimizer.param_groups[0]['lr'] = 0.001
            print('lr is: %.6f' % my_net.optimizer.param_groups[0]['lr'])
            netOutList   = []
            labelOutList = []
            for i, (input_cars, input_events, labels) in enumerate(dataloader_train):
                input_carsD = input_cars.to(device)
                input_eventsD = input_events.to(device)
                labelsD = labels.to(device)
                my_net.loss = None
                # create torch variables
                # input is of size [batch_size, flatten input]
                input_carsVar   = Variable(input_carsD).to(device)
                input_eventsVar = Variable(input_eventsD).to(device)
                labVar          = Variable(labelsD).to(device)
                # reset gradient
                my_net.optimizer.zero_grad()
                # forward
                local_batch_size = labels.shape[0]
                # input to LSTM is [seq_size, batch_size, grid_size] , will be transferred as part of the forward
                netOut = my_net.forward(input_carsVar, input_eventsVar)
                # backwards
                my_net.backward(netOut, labVar)
                # optimizer step
                my_net.optimizer.step()
                # local loss function list
                localLoss.append(my_net.loss.item())
                if isServerRun:
                    labTrainNp = netOut.type(torch.cuda.LongTensor).cpu().detach().numpy()
                    # print("number of net labels different from 0 is:" + str(np.sum(labTrainNp > 0)))
                    # print("number of net labels 0 is:"+str(np.sum(labTrainNp == 0)))
                    labelsNp = labels.cpu().detach().numpy()
                    # print("number of real labels different from 0 is:" + str(np.sum(labelsNp > 0)))
                    # print("number of real labels 0 is:" + str(np.sum(labelsNp == 0)))
                    trainCorr = torch.sum(netOut == labels).cpu().detach().numpy() + trainCorr
                else:
                    labTrainNp = netOut.long().detach().numpy()
                    labelsNp = labels.detach().numpy()
                    trainCorr = torch.sum(netOut == labels).detach().numpy() + trainCorr
                netOutList.append(labTrainNp)
                labelOutList.append(labelsNp)
                trainTot = labels.size(0) + trainTot
                rmse = sqrt(metrics.mean_squared_error(labTrainNp.reshape(-1), labelsNp.reshape(-1)))
                accTrain.append(100 * trainCorr / trainTot)
                rmseTrain.append(rmse)
                # output current state
                if (i + 1) % 10 == 0:
                    print('Epoch: [%d/%d1 ], Step: [%d/%d], Loss: %.4f, Acc: %.4f, RMSE: %.4f'
                          % (numEpoch + 1, my_net.maxEpochs, i + 1,
                            dataloader_train.batch_size,
                             my_net.loss.item(), accTrain[-1], rmseTrain[-1]))
            my_net.lossVecTrain.append(np.average(localLoss))
            my_net.accVecTrain.append(np.average(accTrain))
            my_net.rmseVecTrain.append(np.average(rmseTrain))
            # test network for each epoch stage
            accEpochTest, lossEpochTest, rmseEpochTest = my_net.test_spesific(testLoader=dataloader_test)
            my_net.accVecTest.append(accEpochTest)
            my_net.lossVecTest.append(lossEpochTest)
            my_net.rmseVecTest.append(rmseEpochTest)
            netOutDict[numEpoch] = netOutList
            labelsOutDict[numEpoch] = labelOutList
            if (flag_save_network):
                my_net.saveModel(netConfig + "_torch.pkl")
                # outArray = np.stack([np.array(my_net.lossVecTest), np.array(my_net.lossVecTrain),
                #                      np.array(my_net.accVecTest), np.array(my_net.accVecTrain)])
                # np.save("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch)  + "_oArrBatch.npy", outArray)
        my_net.finalAcc  = accEpochTest
        my_net.finalLoss = lossEpochTest
        my_net.finalRmse = rmseEpochTest
        # name, HyperPerams, accur, num total weights
        # err vs epoch, loss vs epoch,
        saveFile(netOutDict, 'netDict')
        saveFile(labelsOutDict, 'labelsDict')
        strWrite = '{0};{1};{2};{3};{4}\n'.format(netConfig, my_net.finalAcc, my_net.finalLoss, numWeights, my_net.maxEpochs)
        outFile.write(strWrite)

    outFile.close()

    return


if __name__ == '__main__':
    main()
    print('Done.')
